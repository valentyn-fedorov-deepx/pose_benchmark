# Pose Estimation Benchmark: YOLOv8n-Pose vs MoveNet vs RTMPose

This repository implements a reproducible benchmark comparing three single-person pose estimation models on an **NVIDIA RTX 3050 Ti**.

**The Goal:** Evaluate runtime performance (FPS), latency, and keypoint stability under three different pipeline configurations.

**The Models:**
1. **YOLOv8n-Pose** (PyTorch/Ultralytics) - *Integrated detection & pose.*
2. **MoveNet SinglePose Lightning** (ONNX) - *Ultra-fast, mobile-optimized.*
3. **RTMPose-Tiny** (ONNX via `rtmlib`) - *High precision, lightweight.*

---

## 1. Visual Comparison (Outdoor Scenario)

The following grids demonstrate how each model performs on `outdoor_01.mp4` under different input strategies.

### Scenario A: Full-Frame Baseline
All models process the entire frame. Note how smaller models (MoveNet/RTMPose) might struggle with detail when the person is far away.

<table>
  <tr>
    <th width="33%">YOLOv8n-Pose</th>
    <th width="33%">MoveNet Lightning</th>
    <th width="33%">RTMPose (Tiny)</th>
  </tr>
  <tr>
    <td><video src="output/visualizations/fullframe_yolov8n-pose-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/fullframe_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/fullframe_rtmpose-tiny-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
  </tr>
</table>

### Scenario B: Center ROI (Region of Interest)
We simulate a scenario where the person is roughly centered. MoveNet and RTMPose receive a cropped input, significantly improving keypoint stability compared to Full-Frame.

<table>
  <tr>
    <th width="33%">YOLOv8n-Pose (Ref)</th>
    <th width="33%">MoveNet (Center ROI)</th>
    <th width="33%">RTMPose (Center ROI)</th>
  </tr>
  <tr>
    <td><video src="output/visualizations/centerroi_yolov8n-pose-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/centerroi_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/centerroi_rtmpose-tiny-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
  </tr>
</table>

### Scenario C: YOLO ROI (Detector + Pose Pipeline)
A realistic 2-stage pipeline: **YOLOv8** detects the person, and **MoveNet/RTMPose** runs on the cropped bounding box. This offers the highest accuracy but adds the latency of the detector.

<table>
  <tr>
    <th width="33%">YOLOv8n-Pose (Ref)</th>
    <th width="33%">MoveNet (YOLO ROI)</th>
    <th width="33%">RTMPose (YOLO ROI)</th>
  </tr>
  <tr>
    <td><video src="output/visualizations/yoloroidetect_yolov8n-pose-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/yoloroi_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
    <td><video src="output/visualizations/yoloroi_rtmpose-tiny-cuda/outdoor_01.gif" width="100%" controls autoplay loop muted></video></td>
  </tr>
</table>

---

## 2. Quick Start

### Installation
Requires Python 3.10+ and NVIDIA Drivers (CUDA 12.x).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Benchmark

Run the following commands to generate the data and visualizations for all profiles:

```powershell
# 1. Full-frame benchmark (Standard)
python run_pipelines.py --profile fullframe --device cuda

# 2. Center ROI benchmark (Fast crop)
python run_pipelines.py --profile center_roi --device cuda

# 3. YOLO ROI benchmark (High accuracy)
python run_pipelines.py --profile yolo_roi --device cuda
```

---

## 3. Analyzing Results

To generate performance plots (FPS and Latency comparisons) based on the runs above:

```powershell
python analyze_results.py
```

Results are saved to:

* **Plots:** `output/plots/` (e.g., `fps_yolo_roi.png`, `latency_fullframe.png`)
* **Raw Metrics:** `output/benchmark_*.json`

## 4. Export for Mobile

If you identify a winning model and wish to deploy it to mobile (e.g., via SNPE/DLC for Qualcomm), you can export the YOLOv8 model to ONNX using:

```powershell
python -m export_to_onnx.export_yolov8_pose_onnx --output models/yolov8n-pose.onnx
```