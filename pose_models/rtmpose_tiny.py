import numpy as np

from rtmlib import Body

from .base import PoseModel


class RTMPoseTinyModel(PoseModel):
    def __init__(self, device: str = 'cuda'):
        backend = 'onnxruntime'
        if device != 'cuda':
            device = 'cpu'
        self._device = device
        # lightweight mode should pick a small RTMPose model under the hood
        self._body = Body(mode='lightweight', backend=backend, device=self._device)

    def name(self) -> str:
        return f'rtmpose-tiny-{self._device}'

    def input_size(self):
        # Determined internally by rtmlib; not strictly needed for benchmark
        return (0, 0)

    def warmup(self):
        # rtmlib handles its own warmup; no-op here
        pass

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Body returns keypoints and scores for detected persons.
        keypoints, scores = self._body(frame_bgr)
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # take first detected person
        kpts = keypoints[0]  # (K, 2)
        scrs = scores[0]     # (K,)
        out = np.concatenate([kpts, scrs[..., None]], axis=-1).astype(np.float32)
        return out

    def close(self):
        pass
