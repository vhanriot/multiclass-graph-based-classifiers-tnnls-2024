import numpy as np


class ActFun:
    def __init__(self, act_fun: str) -> None:
        self.act_fun = act_fun

    def Chipclass(self, distances) -> np.ndarray:
        max_dist_squared = np.max(distances) ** 2
        ratio = max_dist_squared / (distances + 1e-16)
        w = np.exp(-ratio)
        w = 1 / (w + 1e-16)
        w = w / np.sum(w)
        return w

    def Tanh(self, distances, offset=1) -> np.ndarray:
        return np.tanh(-distances) + offset

    def TanhNorm(self, distances, offset=1) -> np.ndarray:
        w = np.tanh(-distances) + offset
        return w / np.sum(w)

    def Gaussian(self, distances, sigmas: np.ndarray) -> np.ndarray:
        return np.exp((-(distances**2)) / (2 * sigmas**2))

    def forward(self, x, sigmas: np.ndarray | None = None) -> np.ndarray:
        if self.act_fun == "chipclass":
            return self.Chipclass(x)
        elif self.act_fun == "tanh":
            return self.Tanh(x)
        elif self.act_fun == "tanhNorm":
            return self.TanhNorm(x)
        elif self.act_fun == "gaussian":
            return self.Gaussian(x, sigmas)
        else:
            raise ValueError("Activation function not implemented")
