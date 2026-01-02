import numpy as np
import torch
from ultralytics import YOLO

from .base import PoseModel


class YoloV8PoseModel(PoseModel):
    def __init__(self, device: str = 'cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self._device = device
        self.model = YOLO('yolov8n-pose.pt')
        # Ultralytics handles device internally via .to()
        self.model.to(self._device)

    def name(self) -> str:
        return f'yolov8n-pose-{self._device}'

    def input_size(self):
        # YOLOv8n-pose default
        return (640, 640)

    def warmup(self):
        # a couple of dummy inferences
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            _ = self.model(dummy, verbose=False)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        # YOLO expects BGR np.ndarray; handles resize/normalize
        results = self.model(frame_bgr, verbose=False)[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # take the person with highest confidence (largest box area as fallback)
        boxes = results.boxes
        scores = boxes.conf.cpu().numpy() if boxes is not None else None

        idx = 0
        if scores is not None and len(scores) > 1:
            idx = int(np.argmax(scores))

        kpts_xy = results.keypoints.xy[idx].cpu().numpy()  # (K, 2) in image coords
        kpts_conf = results.keypoints.conf[idx].cpu().numpy()  # (K,)

        kpts = np.concatenate([kpts_xy, kpts_conf[..., None]], axis=-1).astype(np.float32)
        return kpts

    def close(self):
        # let torch/ultralytics handle cleanup
        pass
