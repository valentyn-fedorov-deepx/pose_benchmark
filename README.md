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
    <td><img src="output/visualizations/fullframe_yolov8n-pose-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/fullframe_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/fullframe_rtmpose-tiny-cuda/outdoor_01.gif" width="100%"></td>
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
    <td><img src="output/visualizations/centerroi_yolov8n-pose-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/centerroi_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/centerroi_rtmpose-tiny-cuda/outdoor_01.gif" width="100%"></td>
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
    <td><img src="output/visualizations/yoloroidetect_yolov8n-pose-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/yoloroi_movenet-lightning-onnx-cuda/outdoor_01.gif" width="100%"></td>
    <td><img src="output/visualizations/yoloroi_rtmpose-tiny-cuda/outdoor_01.gif" width="100%"></td>
  </tr>
</table>

---

## 2. Project Structure

```
pose-estimation-benchmark/
├── output/
│   ├── keypoints/          # Raw keypoint data (JSON)
│   ├── plots/              # Performance comparison charts
│   └── visualizations/     # Annotated video outputs
│       ├── fullframe_*/
│       ├── centerroi_*/
│       └── yoloroi_*/
├── models/                 # Pre-trained model files
├── videos/                 # Input test videos
├── run_pipelines.py        # Main benchmark script
├── analyze_results.py      # Results analysis and plotting
└── requirements.txt        # Python dependencies
```

---

## 3. Quick Start

### System Requirements
- Python 3.10+
- NVIDIA GPU with CUDA 12.x support
- NVIDIA Drivers (≥525.60.13)
- 8GB+ RAM recommended

### Installation

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Benchmark

Execute the following commands to generate data and visualizations for all three pipeline configurations:

```powershell
# 1. Full-frame benchmark (Standard baseline)
python run_pipelines.py --profile fullframe --device cuda

# 2. Center ROI benchmark (Fast cropping)
python run_pipelines.py --profile center_roi --device cuda

# 3. YOLO ROI benchmark (High accuracy, 2-stage)
python run_pipelines.py --profile yolo_roi --device cuda
```

**Optional flags:**
- `--device cpu` - Run on CPU (slower, for debugging)
- `--skip-visualization` - Skip video generation (faster benchmarking)
- `--video-path path/to/video.mp4` - Use custom input video

---

## 4. Analyzing Results

Generate performance comparison plots (FPS, Latency, Throughput):

```powershell
python analyze_results.py
```

**Outputs:**
- **Plots:** `output/plots/fps_comparison.png`, `latency_by_profile.png`
- **Raw Metrics:** `output/benchmark_fullframe.json`, etc.
- **Summary Report:** Console output with detailed statistics

---

## 5. Performance Results Summary

Benchmark performed on **NVIDIA RTX 3050 Ti** with CUDA 12.x. Results show average FPS and latency across two test videos (indoor_01.mp4: 9586 frames, outdoor_01.mp4: 330 frames).

### Full-Frame Baseline Results

| Model | Video | Avg FPS | Avg Latency (ms) | Total Time (s) |
|-------|-------|---------|------------------|----------------|
| **YOLOv8n-Pose** | indoor_01 | 26.65 | 37.53 | 359.76 |
| **YOLOv8n-Pose** | outdoor_01 | 46.79 | 21.37 | 7.05 |
| **MoveNet Lightning** | indoor_01 | **29.45** | **33.96** | 325.54 |
| **MoveNet Lightning** | outdoor_01 | **75.20** | **13.30** | 4.39 |
| **RTMPose-Tiny** | indoor_01 | 20.27 | 49.33 | 472.84 |
| **RTMPose-Tiny** | outdoor_01 | 37.65 | 26.56 | 8.77 |

**Winner: MoveNet Lightning** - Fastest across all scenarios in full-frame mode.

---

### Center ROI Results

| Model | Video | Avg FPS | Avg Latency (ms) | Total Time (s) |
|-------|-------|---------|------------------|----------------|
| **YOLOv8n-Pose** (Full) | indoor_01 | 23.84 | 41.94 | 402.04 |
| **YOLOv8n-Pose** (Full) | outdoor_01 | 60.17 | 16.62 | 5.48 |
| **MoveNet Lightning** (ROI) | indoor_01 | **31.18** | **32.08** | 307.48 |
| **MoveNet Lightning** (ROI) | outdoor_01 | **109.69** | **9.12** | 3.01 |
| **RTMPose-Tiny** (ROI) | indoor_01 | 23.41 | 42.72 | 409.52 |
| **RTMPose-Tiny** (ROI) | outdoor_01 | 40.41 | 24.75 | 8.17 |

**Winner: MoveNet Lightning with Center ROI** - Achieves 109.69 FPS on outdoor video (3.6× improvement over full-frame YOLOv8n).

---

### YOLO ROI (2-Stage Pipeline) Results

| Model | Video | Avg FPS | Avg Latency (ms) | Total Time (s) |
|-------|-------|---------|------------------|----------------|
| **YOLOv8n-Pose** (Detector) | indoor_01 | 28.31 | 35.33 | 338.63 |
| **YOLOv8n-Pose** (Detector) | outdoor_01 | 25.45 | 39.29 | 12.97 |
| **MoveNet Lightning** (YOLO ROI) | indoor_01 | **99.11** | **10.09** | 96.72 |
| **MoveNet Lightning** (YOLO ROI) | outdoor_01 | **75.07** | **13.32** | 4.40 |
| **RTMPose-Tiny** (YOLO ROI) | indoor_01 | 16.33 | 61.23 | 586.91 |
| **RTMPose-Tiny** (YOLO ROI) | outdoor_01 | 15.46 | 64.68 | 21.34 |

**Winner: MoveNet Lightning with YOLO ROI** - Exceptional 99.11 FPS on indoor video with proper person detection.

---

## 6. Export for Mobile Deployment

Once you've identified the optimal model for your use case, export it for mobile deployment:

### Export YOLOv8 to ONNX
```powershell
python -m export_to_onnx.export_yolov8_pose_onnx --output models/yolov8n-pose.onnx
```

### Convert to Qualcomm SNPE (DLC)
```bash
snpe-onnx-to-dlc --input_network models/yolov8n-pose.onnx \
                 --output_path models/yolov8n-pose.dlc
```

---

## 7. Key Findings & Recommendations

**Full-Frame Mode:**
- Best for: Scenarios where person detection is already handled
- YOLOv8n-Pose offers best balance of speed and accuracy
- MoveNet excels in speed but may miss distant subjects

**Center ROI Mode:**
- Best for: Controlled environments (fitness apps, telehealth)
- Significant FPS boost for MoveNet/RTMPose
- Assumes person is centered in frame

**YOLO ROI Mode:**
- Best for: Real-world deployment with unknown person positions
- Highest accuracy but adds detection overhead
- RTMPose shines with proper bounding boxes

---

## 8. Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{pose-benchmark-2025,
  author = {Your Name},
  title = {Pose Estimation Benchmark: YOLOv8n-Pose vs MoveNet vs RTMPose},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pose-estimation-benchmark}
}
```

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 10. Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow MoveNet](https://www.tensorflow.org/hub/tutorials/movenet)
- [MMPose RTMPose](https://github.com/open-mmlab/mmpose)

---

## Troubleshooting

**CUDA Out of Memory:**
```powershell
# Reduce batch size or use CPU
python run_pipelines.py --profile fullframe --device cpu
```

**Missing Models:**
```powershell
# Models are auto-downloaded on first run
# Manually download: See models/README.md
```

**Video Codec Issues:**
```powershell
pip install opencv-python-headless
# Or install system codecs (K-Lite Codec Pack on Windows)
```