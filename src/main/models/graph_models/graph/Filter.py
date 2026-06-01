import numpy as np


class Filter:
    def __init__(self, flex_reg: bool) -> None:
        self.flex_reg = flex_reg

    def forward(self, quality: np.ndarray, y: np.ndarray, percentages: np.ndarray | None) -> np.ndarray:
        """percentages: percentage of the number of samples to be removed from each class (ex: for 100%=100; 80%=80, 21.07%=21.07)"""
        if self.flex_reg:
            if percentages is None:
                raise ValueError("Percentages shouldn't be null.")
            percentages = percentages / 100
            thresholds = [np.sort(quality[y == yi]) for yi in np.unique(y)]
            thresh = [thresholds[i][int(percentages[i] * len(thresholds[i])) - 1] for i in range(len(thresholds))]
        else:
            thresh = [np.mean(quality[y == i]) for i in np.unique(y)]

        valid_samples = [quality[i] >= thresh[int(yi)] for i, yi in enumerate(y)]

        return 1 * np.array(valid_samples)
