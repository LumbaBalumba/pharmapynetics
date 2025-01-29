from typing import Callable, Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.preprocessing import MinMaxScaler as Scaler

from pharmapynetics.preprocessing import TauEstimator
from pharmapynetics.metrics import Metric, WMSE, ClippedWMSE


class BaseModel:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        pass

    def sample(self, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(t)


class PBFTPK(BaseModel):
    D: float
    F: float
    V_d: float
    k_a: float
    k_el: float
    tau_0: float
    tau: float
    t_max: float
    l: float
    clipped: bool
    scaler: Scaler
    base_model: Callable[
        [np.ndarray, float, float, float, float, float, float, float], np.ndarray
    ]
    tau_estimator: TauEstimator
    metric: Metric

    @staticmethod
    def PBFTPK0(
        t: np.ndarray,
        D: float,
        F: float,
        V_d: float,
        k_a: float,
        k_el: float,
        tau_0: float,
        tau: float,
    ) -> np.ndarray:
        X = np.zeros_like(t)

        def absorbtion_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_0
            return F * D / V_d / k_el / (tau - tau_0) * (1 - np.exp(-k_el * (t)))

        idx_a = (tau_0 < t) & (t <= tau)
        X[idx_a] = absorbtion_model(t[idx_a])
        C_max = X[idx_a][-1] if len(X[idx_a]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_0
            return C_max * np.exp(-k_el * (t - tau))

        idx_el = t > tau
        X[idx_el] = elimination_model(t[idx_el])

        return X

    @staticmethod
    def PBFTPK1(
        t: np.ndarray,
        D: float,
        F: float,
        V_d: float,
        k_a: float,
        k_el: float,
        tau_0: float,
        tau: float,
    ) -> np.ndarray:
        X = np.zeros_like(t)

        def absorption_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_0
            return (
                F
                * D
                * k_a
                / V_d
                / (k_a - k_el)
                * (np.exp(-k_el * (t)) - np.exp(-k_a * (t)))
            )

        idx_a = (tau_0 < t) & (t <= tau)
        X[idx_a] = absorption_model(t[idx_a])
        C_max = X[idx_a][-1] if len(X[idx_a]) > 0 else 1

        def elimination_model(t: np.ndarray | float) -> np.ndarray | float:
            t = np.array(t) - tau_0
            return C_max * np.exp(-k_el * (t - tau))

        idx_el = t > tau
        X[idx_el] = elimination_model(t[idx_el])

        return X

    def __init__(
        self,
        l: float = 1.0,
        t_max: float = float("inf"),
        base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"] = "PBFTPK1",
        tau_estimation_method: Literal["minmax"] | Literal["peak"] = "peak",
        clipped: bool = False,
    ) -> None:
        self.initilized = False
        self.l = l
        self.t_max = t_max
        self.base_model = PBFTPK.PBFTPK0 if base_model == "PBFTPK0" else PBFTPK.PBFTPK1
        self.tau_estimator = TauEstimator(tau_estimation_method)
        self.clipped = clipped

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        self.initilized = True

        data = np.column_stack([t, X])
        self.scaler = Scaler()
        data = self.scaler.fit_transform(data)

        t = data[:, 0]
        X = data[:, 1]

        self.tau_0, self.tau = self.tau_estimator.process(t, X, self.t_max)

        X[t < self.tau_0] = 0.0

        self.metric = (
            ClippedWMSE(l=self.l, tau=self.tau, t_max=self.t_max)
            if self.clipped
            else WMSE(l=self.l, tau=self.tau)
        )

        params_initial = [1.3, 0.5, 1, 1, 2]

        cons = LinearConstraint([[0, 0, 0, 1, -1]], ub=-1e-4)

        bounds = Bounds(lb=1e-4, ub=[1e3, 1.0, 1e4, 300, 300])

        def target_function(params):
            D, F, V_d, k_a, k_el = params
            return self.metric.estimate(
                self.base_model(t, D, F, V_d, k_a, k_el, self.tau_0, self.tau), X, t
            )

        res = minimize(
            target_function,
            constraints=cons,
            bounds=bounds,
            x0=params_initial,
            method="SLSQP",
        )

        self.D, self.F, self.V_d, self.k_a, self.k_el = res.x

    def sample(self, t: np.ndarray) -> np.ndarray:
        data = np.column_stack([t, np.zeros_like(t)])

        data = self.scaler.transform(data)

        t = data[:, 0]

        X = self.base_model(
            t, self.D, self.F, self.V_d, self.k_a, self.k_el, self.tau_0, self.tau
        )

        data = np.column_stack([t, X])

        data = self.scaler.inverse_transform(data)

        X = data[:, 1]

        X[X < 0] = 0

        return X


class EnsembledPBFTPK(BaseModel):
    n_models: int
    models: list[PBFTPK]
    base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"]
    tau_estimation_method: Literal["minmax"] | Literal["peak"]
    clipped: bool
    t_max: float

    def __init__(
        self,
        n_models: int = 1,
        l: float | list[float] | np.ndarray = 1.0,
        base_model: Literal["PBFTPK0"] | Literal["PBFTPK1"] = "PBFTPK1",
        tau_estimation_method: Literal["minmax"] | Literal["peak"] = "peak",
        clipped: bool = False,
        t_max: float = float("inf"),
    ):
        self.n_models = n_models
        self.l = np.ones(n_models) * l
        self.models = []
        self.base_model = base_model
        self.tau_estimation_method = tau_estimation_method
        self.clipped = clipped
        self.t_max = t_max

    def fit(self, t: np.ndarray, X: np.ndarray) -> None:
        r = X.copy()
        t_max = self.t_max
        for i in range(self.n_models):
            model = PBFTPK(
                l=self.l[i],
                t_max=t_max,
                base_model=self.base_model,
                tau_estimation_method=self.tau_estimation_method,
                clipped=self.clipped,
            )
            model.fit(t, r)
            self.models.append(model)
            r -= self.models[i].sample(t)
            t_max = model.tau

    def sample(self, t: np.ndarray) -> np.ndarray:
        X = np.zeros_like(t)
        for model in self.models:
            X += model.sample(t)
        return X
