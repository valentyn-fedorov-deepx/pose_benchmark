from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np


class PoseModel(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Return (width, height) expected by the model."""
        ...

    @abstractmethod
    def warmup(self):
        """Optional warmup on dummy data."""
        ...

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run inference on a single BGR frame.

        Returns keypoints as (K, 3) array [x, y, score] in image coords.
        """
        ...

    def close(self):
        """Optional cleanup."""
        pass
