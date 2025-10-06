from typing import Literal

from scipy.signal import find_peaks
import numpy as np


class TauEstimator:
    method: Literal["minmax"] | Literal["peak"]

    def __init__(self, method: Literal["minmax"] | Literal["peak"]) -> None:
        self.method = method

    @staticmethod
    def peak(t: np.ndarray, x: np.ndarray, t_max: float) -> tuple[float, float]:
        peaks, _ = find_peaks(x[t < t_max])
        if len(peaks) > 0:
            tau_2 = float(t[peaks[-1]])
        else:
            tau_2 = t[-1]
        volleys, _ = find_peaks(-x[t < tau_2])
        if len(volleys) > 0:
            tau_1 = float(t[volleys[-1]])
        else:
            tau_1 = t[0]
        return tau_1, tau_2

    @staticmethod
    def minmax(t: np.ndarray, x: np.ndarray, t_max: float) -> tuple[float, float]:
        tau_2 = t[np.argmax(x[t < t_max])]
        tau_1 = t[np.argmin(x[t < tau_2])]
        return tau_1, tau_2

    def process(
        self, t: np.ndarray, x: np.ndarray, t_max: float
    ) -> tuple[float, float]:
        return (
            TauEstimator.minmax(t, x, t_max)
            if self.method == "minmax"
            else TauEstimator.peak(t, x, t_max)
        )
