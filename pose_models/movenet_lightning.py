import os
import urllib.request
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .base import PoseModel


MOVENET_ONNX_URL = 'https://github.com/shoz-f/onnx_interp/releases/download/models/movenet_singlepose.onnx'
MOVENET_ONNX_LOCAL = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'movenet_singlepose_lightning.onnx')


class MoveNetLightningOnnxModel(PoseModel):
    def __init__(self, device: str = 'cuda'):
        self._device = device
        self.session = self._load_session()
        self.input_name = self.session.get_inputs()[0].name
        inp_shape = self.session.get_inputs()[0].shape
        # expected NHWC: [1, 192, 192, 3]
        self._h = int(inp_shape[1])
        self._w = int(inp_shape[2])

    def _download_model(self):
        os.makedirs(os.path.dirname(MOVENET_ONNX_LOCAL), exist_ok=True)
        print(f'Downloading MoveNet Lightning ONNX to {MOVENET_ONNX_LOCAL} ...')
        urllib.request.urlretrieve(MOVENET_ONNX_URL, MOVENET_ONNX_LOCAL)

    def _load_session(self):
        if not os.path.exists(MOVENET_ONNX_LOCAL):
            self._download_model()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self._device != 'cuda':
            providers = ['CPUExecutionProvider']
        return ort.InferenceSession(MOVENET_ONNX_LOCAL, providers=providers)

    def name(self) -> str:
        return f'movenet-lightning-onnx-{self._device}'

    def input_size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    def warmup(self):
        dummy = np.zeros((1, self._h, self._w, 3), dtype=np.int32)
        for _ in range(2):
            _ = self.session.run(None, {self.input_name: dummy})

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        # MoveNet ONNX expects int32 [0,255]
        input_tensor = resized.astype(np.int32)[None, ...]  # (1, H, W, 3)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        # shape: [1, 1, 17, 3]; last dim: [y, x, score] normalized [0,1]
        kpts = outputs[0].reshape(17, 3)
        y = kpts[:, 0] * h
        x = kpts[:, 1] * w
        s = kpts[:, 2]
        out = np.stack([x, y, s], axis=-1).astype(np.float32)
        return out

    def close(self):
        pass
