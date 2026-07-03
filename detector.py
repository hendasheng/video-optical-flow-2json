import cv2
import numpy as np
import json
import argparse
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description="Extract dense optical flow grid from video")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("-o", "--output", help="Output JSON path (default: video_name_flow.json)")
    parser.add_argument("-s", "--step", type=int, default=16, help="Grid sampling step (default: 16)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize scale, e.g. 0.5 = half resolution")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
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

    print(f"Video: {Path(args.video).name}")
    print(f"Resolution: {orig_w}x{orig_h} -> {out_w}x{out_h} (scale={args.scale})")
    print(f"FPS: {fps:.2f}")
    print(f"Grid step: {args.step}px")
    print(f"Frames: {start} ~ {end} ({end - start} frames)")
    print(f"Estimated grid size: {out_h // args.step} x {out_w // args.step}")
    print()

    if args.scale != 1.0:
        new_size = (out_w, out_h)

    ret, prev_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, prev_frame = cap.read()
    if not ret:
        raise SystemExit("Cannot read first frame")

    if args.scale != 1.0:
        prev_frame = cv2.resize(prev_frame, new_size)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    result = {
        "video": str(Path(args.video).resolve()),
        "width": out_w,
        "height": out_h,
        "grid_step": args.step,
        "grid_cols": out_w // args.step,
        "grid_rows": out_h // args.step,
        "fps": round(fps, 2),
        "total_frames": end - start,
        "frames": [],
    }

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = args.output or str(output_dir / f"{Path(args.video).stem}_flow.json")

    if args.viz:
        viz_dir = output_dir / f"{Path(args.video).stem}_viz"
        viz_dir.mkdir(exist_ok=True)

    frame_idx = start

    for i in range(start + 1, end):
        ret, curr_frame = cap.read()
        if not ret:
            break

        if args.scale != 1.0:
            curr_frame = cv2.resize(curr_frame, new_size)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        grid, flow = compute_flow_grid(prev_gray, curr_gray, args.step)
        result["frames"].append({"frame": i, "flow": grid})

        if args.viz and (i - start) % args.viz_step == 0:
            viz_img = flow_to_arrows(curr_frame, flow, args.step)
            viz_path = str(viz_dir / f"frame_{i:06d}.png")
            cv2.imwrite(viz_path, viz_img)

        frame_idx = i
        pct = (i - start) / (end - start) * 100
        print(f"  [{pct:5.1f}%] Frame {i}/{end - 1}")

        prev_gray = curr_gray

    cap.release()

    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nDone. {len(result['frames'])} frames -> {output_path} ({size_mb:.1f} MB)")
    if args.viz:
        print(f"Viz frames -> {viz_dir}/ ({len(list(viz_dir.glob('*.png')))} images)")


if __name__ == "__main__":
    main()
