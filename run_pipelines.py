import json
import os
from pathlib import Path
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from utils.video_io import iter_video_frames, get_video_fps, create_video_writer
from utils.timing import Timer
from utils.skeletons import draw_keypoints
from pose_models.yolov8_pose import YoloV8PoseModel
from pose_models.movenet_lightning import MoveNetLightningOnnxModel
from pose_models.rtmpose_tiny import RTMPoseTinyModel


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / 'videos'
OUTPUT_DIR = ROOT / 'output'
VIS_DIR = OUTPUT_DIR / 'visualizations'
KP_DIR = OUTPUT_DIR / 'keypoints'
PLOTS_DIR = OUTPUT_DIR / 'plots'


def ensure_dirs():
    for d in [VIDEOS_DIR, OUTPUT_DIR, VIS_DIR, KP_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description='Pose pipelines benchmark')
    ap.add_argument('--profile', type=str, default='fullframe',
                    choices=['fullframe', 'center_roi', 'yolo_roi'],
                    help='Benchmark pipeline profile')
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--no-vis', action='store_true', help='Do not save visualization videos')
    ap.add_argument('--no-save-kp', action='store_true', help='Do not save keypoints')
    return ap.parse_args()


def get_models(device: str = 'cuda'):
    models = {
        'yolo': YoloV8PoseModel(device=device),
        'movenet': MoveNetLightningOnnxModel(device=device),
        'rtmpose': RTMPoseTinyModel(device=device),
    }
    return models


def compute_center_roi(frame, scale: float = 0.6):
    """Central ROI, scale in (0,1], relative to frame size."""
    h, w = frame.shape[:2]
    roi_w = int(w * scale)
    roi_h = int(h * scale)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    return x1, y1, x2, y2


def enlarge_bbox(box, frame_or_shape, scale: float = 1.3):
    """Enlarge [x1, y1, x2, y2] box by scale around its center and clamp to frame.

    frame_or_shape can be either a NumPy frame (H,W,3) or a tuple/list (H,W).
    """
    # Determine height/width depending on input type
    if hasattr(frame_or_shape, 'shape'):
        h, w = frame_or_shape.shape[:2]
    else:
        h, w = frame_or_shape[:2]

    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2))
    ny2 = min(h, int(cy + bh / 2))
    return nx1, ny1, nx2, ny2


def yolo_person_bbox(yolo_model, frame_bgr):
    """Get best person bbox [x1, y1, x2, y2] from YOLOv8 pose model on full frame."""
    from ultralytics.engine.results import Results

    res: Results = yolo_model.model(frame_bgr, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None
    conf = boxes.conf.cpu().numpy()
    idx = int(np.argmax(conf))
    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()
    return [int(x1), int(y1), int(x2), int(y2)]


def run_profile_fullframe(models, video_path: Path, args):
    """Baseline: кожна модель отримує повний кадр."""
    fps = get_video_fps(str(video_path))
    frames = list(iter_video_frames(str(video_path)))
    if not frames:
        return []
    h, w = frames[0].shape[:2]

    results = []

    for key, model in models.items():
        print(f"\n=== [fullframe] {model.name()} on {video_path.name} ===")
        writer = None
        if not args.no_vis:
            out_dir = VIS_DIR / f'fullframe_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_video = out_dir / video_path.name
            writer = create_video_writer(str(out_video), fps or 30.0, (w, h))

        all_kpts = []
        try:
            model.warmup()
        except Exception as e:
            print(f"Warmup failed for {model.name()}: {e}")

        timer = Timer()
        for frame in tqdm(frames, desc=f"fullframe {model.name()} {video_path.name}"):
            timer.start()
            kpts = model.infer(frame)
            timer.stop()

            if not args.no_save_kp:
                all_kpts.append(kpts.astype(np.float32))

            if writer is not None:
                vis = frame if kpts is None or kpts.shape[0] == 0 else draw_keypoints(frame, kpts)
                writer.write(vis)

        if writer is not None:
            writer.release()

        total_time = timer.get()
        num_frames = len(frames)
        avg_fps = num_frames / total_time if total_time > 0 else 0.0
        avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0
        print(f"Frames: {num_frames}, total: {total_time:.3f}s, FPS: {avg_fps:.2f}, ms/frame: {avg_ms:.2f}")

        if not args.no_save_kp:
            out_dir = KP_DIR / f'fullframe_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_kp = out_dir / (video_path.stem + '.npz')
            np.savez_compressed(out_kp, keypoints=np.array(all_kpts, dtype=object))

        results.append({
            'profile': 'fullframe',
            'model': model.name(),
            'video': video_path.name,
            'num_frames': num_frames,
            'total_time_sec': total_time,
            'avg_fps': avg_fps,
            'avg_ms_per_frame': avg_ms,
        })

    return results


def run_profile_center_roi(models, video_path: Path, args):
    """Center ROI: для MoveNet/RTMPose обрізаємо центральну область."""
    fps = get_video_fps(str(video_path))
    frames = list(iter_video_frames(str(video_path)))
    if not frames:
        return []
    h, w = frames[0].shape[:2]

    results = []

    # YOLO: як baseline, повний кадр (щоб був референс)
    yolo = models['yolo']
    print(f"\n=== [center_roi] {yolo.name()} (full frame) on {video_path.name} ===")
    writer_y = None
    if not args.no_vis:
        out_dir = VIS_DIR / f'centerroi_{yolo.name()}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / video_path.name
        writer_y = create_video_writer(str(out_video), fps or 30.0, (w, h))

    all_kpts_y = []
    try:
        yolo.warmup()
    except Exception as e:
        print(f"Warmup failed for {yolo.name()}: {e}")

    timer_y = Timer()
    for frame in tqdm(frames, desc=f"center_roi {yolo.name()} {video_path.name}"):
        timer_y.start()
        kpts = yolo.infer(frame)
        timer_y.stop()
        if not args.no_save_kp:
            all_kpts_y.append(kpts.astype(np.float32))
        if writer_y is not None:
            vis = frame if kpts is None or kpts.shape[0] == 0 else draw_keypoints(frame, kpts)
            writer_y.write(vis)

    if writer_y is not None:
        writer_y.release()

    total_time = timer_y.get()
    num_frames = len(frames)
    avg_fps = num_frames / total_time if total_time > 0 else 0.0
    avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0
    results.append({
        'profile': 'center_roi',
        'model': yolo.name() + '_full',
        'video': video_path.name,
        'num_frames': num_frames,
        'total_time_sec': total_time,
        'avg_fps': avg_fps,
        'avg_ms_per_frame': avg_ms,
    })
    if not args.no_save_kp:
        out_dir = KP_DIR / f'centerroi_{yolo.name()}_full'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_kp = out_dir / (video_path.stem + '.npz')
        np.savez_compressed(out_kp, keypoints=np.array(all_kpts_y, dtype=object))

    # MoveNet + RTMPose: центральний ROI
    for key in ['movenet', 'rtmpose']:
        model = models[key]
        print(f"\n=== [center_roi] {model.name()} (ROI) on {video_path.name} ===")
        writer = None
        if not args.no_vis:
            out_dir = VIS_DIR / f'centerroi_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_video = out_dir / video_path.name
            writer = create_video_writer(str(out_video), fps or 30.0, (w, h))

        all_kpts = []
        try:
            model.warmup()
        except Exception as e:
            print(f"Warmup failed for {model.name()}: {e}")

        timer = Timer()
        for frame in tqdm(frames, desc=f"center_roi {model.name()} {video_path.name}"):
            x1, y1, x2, y2 = compute_center_roi(frame, scale=0.6)
            roi = frame[y1:y2, x1:x2]

            timer.start()
            kpts_roi = model.infer(roi)
            timer.stop()

            # Переводимо координати з ROI в глобальні
            if kpts_roi is None or kpts_roi.shape[0] == 0:
                kpts_global = np.zeros((0, 3), dtype=np.float32)
            else:
                kpts_global = kpts_roi.copy()
                kpts_global[:, 0] += x1
                kpts_global[:, 1] += y1

            if not args.no_save_kp:
                all_kpts.append(kpts_global.astype(np.float32))

            if writer is not None:
                vis = frame if kpts_global.shape[0] == 0 else draw_keypoints(frame, kpts_global)
                writer.write(vis)

        if writer is not None:
            writer.release()

        total_time = timer.get()
        num_frames = len(frames)
        avg_fps = num_frames / total_time if total_time > 0 else 0.0
        avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0
        results.append({
            'profile': 'center_roi',
            'model': model.name() + '_centerroi',
            'video': video_path.name,
            'num_frames': num_frames,
            'total_time_sec': total_time,
            'avg_fps': avg_fps,
            'avg_ms_per_frame': avg_ms,
        })
        if not args.no_save_kp:
            out_dir = KP_DIR / f'centerroi_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_kp = out_dir / (video_path.stem + '.npz')
            np.savez_compressed(out_kp, keypoints=np.array(all_kpts, dtype=object))

    return results


def run_profile_yolo_roi(models, video_path: Path, args):
    """YOLO-based ROI для MoveNet/RTMPose."""
    fps = get_video_fps(str(video_path))
    frames = list(iter_video_frames(str(video_path)))
    if not frames:
        return []
    h, w = frames[0].shape[:2]

    results = []

    yolo = models['yolo']
    print(f"\n=== [yolo_roi] {yolo.name()} (detector+pose) on {video_path.name} ===")
    writer_y = None
    if not args.no_vis:
        out_dir = VIS_DIR / f'yoloroidetect_{yolo.name()}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / video_path.name
        writer_y = create_video_writer(str(out_video), fps or 30.0, (w, h))

    all_kpts_y = []
    try:
        yolo.warmup()
    except Exception as e:
        print(f"Warmup failed for {yolo.name()}: {e}")

    timer_y = Timer()
    for frame in tqdm(frames, desc=f"yolo_roi {yolo.name()} {video_path.name}"):
        timer_y.start()
        kpts = yolo.infer(frame)
        timer_y.stop()
        if not args.no_save_kp:
            all_kpts_y.append(kpts.astype(np.float32))
        if writer_y is not None:
            vis = frame if kpts is None or kpts.shape[0] == 0 else draw_keypoints(frame, kpts)
            writer_y.write(vis)

    if writer_y is not None:
        writer_y.release()

    total_time = timer_y.get()
    num_frames = len(frames)
    avg_fps = num_frames / total_time if total_time > 0 else 0.0
    avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0
    results.append({
        'profile': 'yolo_roi',
        'model': yolo.name() + '_detectorpose',
        'video': video_path.name,
        'num_frames': num_frames,
        'total_time_sec': total_time,
        'avg_fps': avg_fps,
        'avg_ms_per_frame': avg_ms,
    })
    if not args.no_save_kp:
        out_dir = KP_DIR / f'yoloroidetect_{yolo.name()}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_kp = out_dir / (video_path.stem + '.npz')
        np.savez_compressed(out_kp, keypoints=np.array(all_kpts_y, dtype=object))

    # MoveNet + RTMPose: ROI з YOLO bbox
    for key in ['movenet', 'rtmpose']:
        model = models[key]
        print(f"\n=== [yolo_roi] {model.name()} (YOLO ROI) on {video_path.name} ===")
        writer = None
        if not args.no_vis:
            out_dir = VIS_DIR / f'yoloroI_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_video = out_dir / video_path.name
            writer = create_video_writer(str(out_video), fps or 30.0, (w, h))

        all_kpts = []
        try:
            model.warmup()
        except Exception as e:
            print(f"Warmup failed for {model.name()}: {e}")

        timer = Timer()
        for frame in tqdm(frames, desc=f"yolo_roi {model.name()} {video_path.name}"):
            box = yolo_person_bbox(yolo, frame)
            if box is None:
                # fallback: central ROI
                x1, y1, x2, y2 = compute_center_roi(frame, scale=0.6)
            else:
                x1, y1, x2, y2 = enlarge_bbox(box, frame, scale=1.3)
            roi = frame[y1:y2, x1:x2]

            timer.start()
            kpts_roi = model.infer(roi)
            timer.stop()

            if kpts_roi is None or kpts_roi.shape[0] == 0:
                kpts_global = np.zeros((0, 3), dtype=np.float32)
            else:
                kpts_global = kpts_roi.copy()
                kpts_global[:, 0] += x1
                kpts_global[:, 1] += y1

            if not args.no_save_kp:
                all_kpts.append(kpts_global.astype(np.float32))

            if writer is not None:
                vis = frame if kpts_global.shape[0] == 0 else draw_keypoints(frame, kpts_global)
                writer.write(vis)

        if writer is not None:
            writer.release()

        total_time = timer.get()
        num_frames = len(frames)
        avg_fps = num_frames / total_time if total_time > 0 else 0.0
        avg_ms = (total_time / num_frames * 1000.0) if num_frames > 0 else 0.0
        results.append({
            'profile': 'yolo_roi',
            'model': model.name() + '_yoloroi',
            'video': video_path.name,
            'num_frames': num_frames,
            'total_time_sec': total_time,
            'avg_fps': avg_fps,
            'avg_ms_per_frame': avg_ms,
        })
        if not args.no_save_kp:
            out_dir = KP_DIR / f'yoloroI_{model.name()}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_kp = out_dir / (video_path.stem + '.npz')
            np.savez_compressed(out_kp, keypoints=np.array(all_kpts, dtype=object))

    return results


def main():
    args = parse_args()
    ensure_dirs()

    video_files = [p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}]
    if not video_files:
        print(f"No videos found in {VIDEOS_DIR}. Drop some video files there and run again.")
        return

    models = get_models(device=args.device)

    all_results = []
    for vf in video_files:
        if args.profile == 'fullframe':
            res = run_profile_fullframe(models, vf, args)
        elif args.profile == 'center_roi':
            res = run_profile_center_roi(models, vf, args)
        elif args.profile == 'yolo_roi':
            res = run_profile_yolo_roi(models, vf, args)
        else:
            raise ValueError(f"Unknown profile {args.profile}")
        all_results.extend(res)

    # cleanup
    for m in models.values():
        try:
            m.close()
        except Exception:
            pass

    # save results
    out_json = OUTPUT_DIR / f'benchmark_{args.profile}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved profile results to {out_json}")


if __name__ == '__main__':
    main()
