import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils.video_io import iter_video_frames, get_video_fps, create_video_writer
from utils.timing import Timer
from utils.skeletons import draw_keypoints
from pose_models.yolov8_pose import YoloV8PoseModel
from pose_models.movenet_lightning import MoveNetLightningOnnxModel
from pose_models.rtmpose_tiny import RTMPoseTinyModel
# from pose_models.mediapipe_blazepose import MediaPipeBlazePoseOnnxModel  # optional stub


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / 'videos'
OUTPUT_DIR = ROOT / 'output'
VIS_DIR = OUTPUT_DIR / 'visualizations'
KP_DIR = OUTPUT_DIR / 'keypoints'


def ensure_dirs():
    for d in [VIDEOS_DIR, OUTPUT_DIR, VIS_DIR, KP_DIR]:
        os.makedirs(d, exist_ok=True)


def get_models(device: str = 'cuda'):
    models = []
    models.append(YoloV8PoseModel(device=device))
    models.append(MoveNetLightningOnnxModel(device=device))
    models.append(RTMPoseTinyModel(device=device))
    # BlazePose stub is not enabled by default because it is not implemented yet
    # models.append(MediaPipeBlazePoseOnnxModel(device=device))
    return models


def benchmark_model_on_video(model, video_path: Path, save_vis: bool = True, save_kp: bool = True):
    print(f"\n=== {model.name()} on {video_path.name} ===")
    fps = get_video_fps(str(video_path))
    frames = list(iter_video_frames(str(video_path)))
    if len(frames) == 0:
        print("No frames in video, skipping.")
        return None

    h, w = frames[0].shape[:2]

    # optional visualization writer
    writer = None
    if save_vis:
        out_model_dir = VIS_DIR / model.name()
        out_model_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = out_model_dir / video_path.name
        writer = create_video_writer(str(out_video_path), fps or 30.0, (w, h))

    # keypoints storage
    all_kpts = []

    # Warmup
    try:
        model.warmup()
    except Exception as e:
        print(f"Warmup failed for {model.name()}: {e}")

    timer = Timer()

    # Inference loop
    for frame in tqdm(frames, desc=f"{model.name()} {video_path.name}"):
        timer.start()
        kpts = model.infer(frame)
        timer.stop()

        if save_kp:
            all_kpts.append(kpts.astype(np.float32))

        if writer is not None:
            if kpts is None or kpts.shape[0] == 0:
                vis = frame
            else:
                vis = draw_keypoints(frame, kpts)
            writer.write(vis)

    if writer is not None:
        writer.release()

    total_time = timer.get()
    num_frames = len(frames)
    avg_fps = num_frames / total_time if total_time > 0 else 0.0
    avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0

    print(f"Frames: {num_frames}, total_time: {total_time:.3f}s, FPS: {avg_fps:.2f}, ms/frame: {avg_ms:.2f}")

    if save_kp:
        out_model_dir = KP_DIR / model.name()
        out_model_dir.mkdir(parents=True, exist_ok=True)
        out_kp_path = out_model_dir / (video_path.stem + '.npz')
        # store as array of variable-length frames (object mode) if shapes differ
        np.savez_compressed(out_kp_path, keypoints=np.array(all_kpts, dtype=object))

    return {
        'model': model.name(),
        'video': video_path.name,
        'num_frames': num_frames,
        'total_time_sec': total_time,
        'avg_fps': avg_fps,
        'avg_ms_per_frame': avg_ms,
    }


def main():
    ensure_dirs()

    video_files = [p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}]
    if not video_files:
        print(f"No videos found in {VIDEOS_DIR}. Drop some video files there and run again.")
        return

    device = 'cuda'
    models = get_models(device=device)

    results = []
    for model in models:
        for vf in video_files:
            try:
                res = benchmark_model_on_video(model, vf, save_vis=True, save_kp=True)
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"Error while running {model.name()} on {vf.name}: {e}")
        try:
            model.close()
        except Exception:
            pass

    # Save aggregate results
    out_json = OUTPUT_DIR / 'benchmark_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == '__main__':
    main()
