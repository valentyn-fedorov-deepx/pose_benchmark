import numpy as np

from .base import PoseModel


class MediaPipeBlazePoseOnnxModel(PoseModel):
    """Placeholder for a BlazePose/MediaPipe ONNX model.

    NOTE: Implementing full BlazePose post-processing (anchors, decoding, ROI tracking)
    is non-trivial. This stub is provided so the benchmark can be extended later
    without changing the rest of the pipeline.

    For now this model is not used by default in run_benchmark.py.
    """

    def __init__(self, device: str = 'cuda'):
        self._device = device
        raise NotImplementedError(
            'MediaPipeBlazePoseOnnxModel is a stub. Integrate a specific BlazePose ONNX '
            'and its post-processing here if/when you are ready to benchmark it.'
        )

    def name(self) -> str:
        return f'mediapipe-blazepose-onnx-{self._device}'

    def input_size(self):
        return (0, 0)

    def warmup(self):
        pass

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float32)

    def close(self):
        pass
