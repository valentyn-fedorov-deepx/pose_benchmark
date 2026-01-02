"""Export YOLOv8n-Pose to ONNX for later DLC (SNPE) conversion.

Usage (from E:/pose_benchmark):

    python -m export_to_onnx.export_yolov8_pose_onnx --output models/yolov8n-pose.onnx

Then on Qualcomm SNPE side you can run e.g.:

    snpe-onnx-to-dlc --input_network models/yolov8n-pose.onnx \
                     --output_path models/yolov8n-pose.dlc
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=str, default='models/yolov8n-pose.onnx')
    ap.add_argument('--opset', type=int, default=12)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO('yolov8n-pose.pt')
    model.export(format='onnx', opset=args.opset, dynamic=False, imgsz=640, simplify=True)
    # Ultralytics saves the ONNX next to the .pt; move/rename if needed.
    print('Export complete. Check the generated .onnx file in the working directory.')


if __name__ == '__main__':
    main()
