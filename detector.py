import cv2
import numpy as np
import json
import argparse
from pathlib import Path

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def detect_features(gray, max_points, quality):
    """Detect Shi-Tomasi corners for sparse tracking."""
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=quality,
                                   minDistance=7, blockSize=7)
    if pts is None:
        return None
    return pts.reshape(-1, 2)


def compute_flow_grid(prev_gray, curr_gray, grid_step=16):
    """Compute dense optical flow, then downsample to a grid."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    h, w = flow.shape[:2]
    grid_flow = []
    for y in range(0, h, grid_step):
        row = []
        for x in range(0, w, grid_step):
            dx, dy = flow[y, x].tolist()
            row.append([round(dx, 4), round(dy, 4)])
        grid_flow.append(row)
    return grid_flow, flow


def compute_sparse_flow(prev_gray, curr_gray, prev_pts):
    """Track sparse feature points with Lucas-Kanade. Returns (curr_pts, prev_pts, curr_pts)."""
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)
    if curr_pts is None:
        return None, None
    status = status.reshape(-1).astype(bool)
    return prev_pts[status], curr_pts[status]


def flow_to_arrows(frame, flow, grid_step=16, arrow_scale=1.0):
    """Draw flow arrows on the frame at grid points."""
    vis = frame.copy()
    h, w = flow.shape[:2]
    for y in range(grid_step // 2, h, grid_step):
        for x in range(grid_step // 2, w, grid_step):
            dx, dy = flow[y, x]
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 0.5:
                continue
            tip = (int(x + dx * arrow_scale), int(y + dy * arrow_scale))
            cv2.arrowedLine(vis, (x, y), tip, (0, 255, 0), 1, tipLength=0.3)
    return vis


def sparse_to_viz(frame, prev_pts, curr_pts):
    """Draw tracked sparse points and their motion vectors."""
    vis = frame.copy()
    for i in range(len(prev_pts)):
        x1, y1 = prev_pts[i].ravel()
        x2, y2 = curr_pts[i].ravel()
        dx, dy = (x2 - x1) * 8, (y2 - y1) * 8
        tip = (int(x1 + dx), int(y1 + dy))
        cv2.circle(vis, (int(x2), int(y2)), 3, (0, 255, 255), -1)
        cv2.arrowedLine(vis, (int(x1), int(y1)), tip, (0, 255, 0), 1, tipLength=0.3)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Extract optical flow from video to JSON")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize scale (default: 1.0)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--sparse", action="store_true", help="Sparse Lucas-Kanade tracking (default: dense grid)")
    parser.add_argument("-s", "--step", type=int, default=16, help="Dense grid step (default: 16)")
    parser.add_argument("-n", "--max-points", type=int, default=500, help="Sparse max feature points (default: 500)")
    parser.add_argument("-q", "--quality", type=float, default=0.01, help="Sparse corner quality (default: 0.01)")
    parser.add_argument("--viz", action="store_true", help="Save flow visualization frames")
    parser.add_argument("--viz-step", type=int, default=10, help="Save viz every N frames (default: 10)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = int(orig_w * args.scale)
    out_h = int(orig_h * args.scale)

    start = max(0, args.start)
    end = min(total_frames, args.end) if args.end else total_frames

    mode = "sparse" if args.sparse else "dense"

    # Build run folder name
    video_stem = Path(args.video).stem
    parts = [video_stem, mode]
    if args.sparse:
        parts.append(f"n{args.max_points}")
    else:
        parts.append(f"s{args.step}")
    if args.scale != 1.0:
        parts.append(f"scale{args.scale}")
    run_name = "_".join(parts)

    script_dir = Path(__file__).resolve().parent
    run_dir = script_dir / "output" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output or str(run_dir / "flow.json")
    viz_dir = run_dir / "viz"
    if args.viz:
        viz_dir.mkdir(exist_ok=True)

    print(f"Video: {Path(args.video).name}")
    print(f"Resolution: {orig_w}x{orig_h} -> {out_w}x{out_h} (scale={args.scale})")
    print(f"FPS: {fps:.2f}")
    print(f"Mode: {mode}")
    if args.sparse:
        print(f"Max points: {args.max_points}  Quality: {args.quality}")
    else:
        print(f"Grid step: {args.step}px  Grid size: {out_h // args.step} x {out_w // args.step}")
    print(f"Frames: {start} ~ {end} ({end - start} frames)")
    print(f"Output: {run_dir}/")
    print()

    if args.scale != 1.0:
        new_size = (out_w, out_h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, prev_frame = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame")
    if args.scale != 1.0:
        prev_frame = cv2.resize(prev_frame, new_size)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if args.sparse:
        result = run_sparse(cap, prev_gray, prev_frame, start, end, args, viz_dir if args.viz else None)
    else:
        result = run_dense(cap, prev_gray, prev_frame, start, end, args, viz_dir if args.viz else None)

    with open(json_path, "w") as f:
        json.dump(result, f)

    size_mb = Path(json_path).stat().st_size / (1024 * 1024)
    print(f"\nDone. {len(result['frames'])} frames -> {run_dir}/")
    print(f"  flow.json ({size_mb:.1f} MB)")
    if args.viz:
        viz_files = sorted(viz_dir.glob("*.png"))
        print(f"  viz/ ({len(viz_files)} images)")
        if viz_files:
            import shutil
            preview = run_dir / "preview.png"
            shutil.copy2(str(viz_files[0]), str(preview))
            print(f"  preview.png (copied from {viz_files[0].name})")


def run_dense(cap, prev_gray, prev_frame, start, end, args, viz_dir):
    result = {
        "mode": "dense",
        "video": str(Path(args.video).resolve()),
        "width": prev_gray.shape[1],
        "height": prev_gray.shape[0],
        "grid_step": args.step,
        "grid_cols": prev_gray.shape[1] // args.step,
        "grid_rows": prev_gray.shape[0] // args.step,
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "total_frames": end - start,
        "frames": [],
    }

    new_size = (result["width"], result["height"]) if args.scale != 1.0 else None

    for i in range(start + 1, end):
        ret, curr_frame = cap.read()
        if not ret:
            break
        if new_size:
            curr_frame = cv2.resize(curr_frame, new_size)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        grid, flow = compute_flow_grid(prev_gray, curr_gray, args.step)
        result["frames"].append({"frame": i, "flow": grid})

        if viz_dir and (i - start) % args.viz_step == 0:
            cv2.imwrite(str(viz_dir / f"frame_{i:06d}.png"),
                        flow_to_arrows(curr_frame, flow, args.step))

        pct = (i - start) / (end - start) * 100
        print(f"  [{pct:5.1f}%] Frame {i}/{end - 1}")
        prev_gray = curr_gray

    return result


def run_sparse(cap, prev_gray, prev_frame, start, end, args, viz_dir):
    pts = detect_features(prev_gray, args.max_points, args.quality)
    if pts is None:
        raise SystemExit("No features detected on first frame")
    n_pts = len(pts)
    pt_ids = np.arange(n_pts, dtype=np.int32)
    max_id = n_pts

    result = {
        "mode": "sparse",
        "video": str(Path(args.video).resolve()),
        "width": prev_gray.shape[1],
        "height": prev_gray.shape[0],
        "max_points": args.max_points,
        "quality": args.quality,
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "total_frames": end - start,
        "frames": [],
    }

    new_size = (result["width"], result["height"]) if args.scale != 1.0 else None

    for i in range(start + 1, end):
        ret, curr_frame = cap.read()
        if not ret:
            break
        if new_size:
            curr_frame = cv2.resize(curr_frame, new_size)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        good_prev, good_curr = compute_sparse_flow(prev_gray, curr_gray, pts)
        if good_prev is None or len(good_prev) < 10:
            # Re-detect features
            pts = detect_features(curr_gray, args.max_points, args.quality)
            if pts is None:
                pts = np.empty((0, 2))
                pt_ids = np.array([], dtype=np.int32)
            else:
                pt_ids = np.arange(max_id, max_id + len(pts), dtype=np.int32)
                max_id += len(pts)
            result["frames"].append({"frame": i, "points": []})
        else:
            good_prev = good_prev.reshape(-1, 2)
            good_curr = good_curr.reshape(-1, 2)
            dx = good_curr[:, 0] - good_prev[:, 0]
            dy = good_curr[:, 1] - good_prev[:, 1]
            frame_data = []
            for k in range(len(good_curr)):
                frame_data.append([
                    int(pt_ids[k]),
                    round(float(good_curr[k, 0]), 2), round(float(good_curr[k, 1]), 2),
                    round(float(dx[k]), 4), round(float(dy[k]), 4),
                ])
            result["frames"].append({"frame": i, "points": frame_data})

            if viz_dir and (i - start) % args.viz_step == 0:
                cv2.imwrite(str(viz_dir / f"frame_{i:06d}.png"),
                            sparse_to_viz(curr_frame, good_prev, good_curr))

            pts = good_curr.reshape(-1, 1, 2)

            # Re-detect when too few points remain
            if len(pts) < args.max_points * 0.3:
                new_pts = detect_features(curr_gray, args.max_points - len(pts), args.quality)
                if new_pts is not None:
                    pts = np.vstack([pts, new_pts.reshape(-1, 1, 2)])
                    new_ids = np.arange(max_id, max_id + len(new_pts), dtype=np.int32)
                    pt_ids = np.concatenate([pt_ids, new_ids])
                    max_id += len(new_pts)

        pct = (i - start) / (end - start) * 100
        print(f"  [{pct:5.1f}%] Frame {i}/{end - 1}  points={len(pts)}")
        prev_gray = curr_gray

    return result


if __name__ == "__main__":
    main()
