from typing import override
import numpy as np


class Metric:
    def __init__(self) -> None:
        pass

    def array_estimate(
        self, C1: np.ndarray, C2: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        return np.abs(C1 - C2)

    def estimate(self, C1: np.ndarray, C2: np.ndarray, t: np.ndarray) -> float:
        return float(np.mean(self.array_estimate(C1, C2, t)))


class MSE(Metric):
    def __init__(self) -> None:
        super().__init__()

    @override
    def array_estimate(
        self, C1: np.ndarray, C2: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        return (C1 - C2) ** 2


class WMSE(Metric):
    l: float
    tau: float

    def __init__(self, l: float, tau: float) -> None:
        self.l = l
        self.tau = tau
        super().__init__()

    def array_estimate(
        self, C1: np.ndarray, C2: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        est = super().array_estimate(C1, C2, t)
        est[t > self.tau] *= self.l
        return est


class ClippedWMSE(WMSE):
    t_max: float

    def __init__(self, l: float, tau: float, t_max: float) -> None:
        super().__init__(l, tau)
        self.t_max = t_max

    def array_estimate(
        self, C1: np.ndarray, C2: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        est = super().array_estimate(C1, C2, t)
        est[t > self.t_max] *= 0
        return est
