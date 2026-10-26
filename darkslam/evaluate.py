"""Run with: python -m darkslam.evaluate --dataset <CLEARIR sequence>."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import (pair_images, read_trajectory, seconds, validate_images,
                   write_associations, write_settings)
from .metrics import evaluate_ate

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLEAR-IR → SuperPoint → DarkSLAM IR-D evaluation")
    p.add_argument("--dataset", type=Path, required=True, help="Sequence directory containing infra1/ and depth/")
    p.add_argument("--output", type=Path, default=ROOT / "results/ird", help="New output directory (must be empty)")
    p.add_argument("--mode", choices=["clearir", "raw"], default="clearir")
    p.add_argument("--model", type=Path, default=ROOT / "models/clearir.h5")
    p.add_argument("--vocabulary", type=Path, default=ROOT / "Vocabulary/superpoint_voc.dbow3")
    p.add_argument("--superpoint", type=Path, default=ROOT / "Vocabulary/SuperPointNet.pt")
    p.add_argument("--binary", type=Path, default=ROOT / "build/bin/rgbd_tum_headless")
    p.add_argument("--settings", type=Path, help="Override automatic sequence CameraInfo calibration")
    p.add_argument("--ir-dir", default="infra1", help="Left IR, registered to depth (default: infra1)")
    p.add_argument("--depth-dir", default="depth")
    p.add_argument("--timestamp-unit", choices=["auto", "ns", "s"], default="auto", help="Units for image filenames")
    p.add_argument("--max-difference", type=float, default=0.02, help="Max IR/depth pairing difference, seconds")
    p.add_argument("--depth-factor", type=float, default=1000.0, help="Depth units per metre; ignored with --settings")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--device", choices=["cpu", "auto"], default="auto",
                   help="Device for CLEAR-IR and SuperPoint (default: available GPU, otherwise CPU)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--prepare-only", action="store_true", help="Write enhanced images, settings, and associations without SLAM")
    p.add_argument("--video", action="store_true")
    p.add_argument("--plot", action="store_true",
                   help="Save trajectory.png with SE(3)-aligned estimate vs ground truth (XY, XZ, YZ)")
    p.add_argument("--groundtruth", type=Path, help="TUM trajectory or CLEARIR CSV; defaults to sequence vicon.csv")
    p.add_argument("--no-groundtruth", action="store_true", help="Run tracking without computing ATE")
    p.add_argument("--gt-timestamp-unit", choices=["auto", "ns", "s"], default="auto")
    p.add_argument("--gt-max-difference", type=float, default=0.02)
    p.add_argument("--gt-offset", type=float, default=0.0, help="Seconds added to estimated timestamps for GT matching")
    return p


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args) -> None:
    for field in ["dataset", "output", "model", "vocabulary", "superpoint", "binary", "settings", "groundtruth"]:
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if args.stride < 1 or args.threads < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("Stride, threads, and max-frames must be positive")
    if args.max_difference < 0 or args.gt_max_difference < 0 or args.depth_factor <= 0:
        raise ValueError("Invalid timestamp tolerance or depth factor")
    if args.groundtruth and args.no_groundtruth:
        raise ValueError("Choose either --groundtruth or --no-groundtruth")
    if args.plot:
        if args.no_groundtruth:
            raise ValueError("--plot requires ground truth; remove --no-groundtruth")
        if args.prepare_only:
            raise ValueError("--plot requires a SLAM trajectory; remove --prepare-only")
        from importlib.util import find_spec
        if find_spec("matplotlib") is None:
            raise ValueError("--plot requires matplotlib; install requirements.txt")
    required = [] if args.prepare_only else [args.binary, args.vocabulary, args.superpoint]
    if args.mode == "clearir":
        required.append(args.model)
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing required asset: {path}. See README.md setup instructions")
    if not args.prepare_only and not os.access(args.binary, os.X_OK):
        raise ValueError(f"Binary is not executable: {args.binary}")
    gt_path = None if args.no_groundtruth else (args.groundtruth or args.dataset / "vicon.csv")
    gt = None
    if gt_path:
        if not gt_path.is_file():
            raise ValueError(f"Missing ground truth: {gt_path}; supply --groundtruth or --no-groundtruth")
        gt = read_trajectory(gt_path, args.gt_timestamp_unit)
        if len(gt) < 3:
            raise ValueError("Ground truth requires at least three valid poses")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}; choose a new --output")
    pairs, statistics = pair_images(args.dataset, args.ir_dir, args.depth_dir,
                                    args.timestamp_unit, args.max_difference,
                                    args.stride, args.max_frames)
    validate_images(pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = write_settings(args.dataset, ROOT / "configs/clearir.yaml",
                              args.output / "settings.yaml", args.settings, args.depth_factor)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    started = time.monotonic()
    if args.mode == "clearir":
        manifest = args.output / "input_pairs.json"
        manifest.write_text(json.dumps([(a, str(ir), b, str(depth)) for a, ir, b, depth in pairs]))
        worker = [sys.executable, "-m", "darkslam.clearir", "--manifest", str(manifest),
                  "--model", str(args.model), "--output", str(args.output / "enhanced"),
                  "--device", args.device, "--threads", str(args.threads)]
        with (args.output / "clearir.log").open("w") as log:
            result = subprocess.run(worker, cwd=ROOT, stderr=log)
        if result.returncode:
            raise RuntimeError(f"CLEAR-IR failed; see {args.output / 'clearir.log'}")
        pairs = [(a, args.output / "enhanced" / f"{i:08d}.png", b, depth)
                 for i, (a, _, b, depth) in enumerate(pairs)]
    associations = args.output / "associations.txt"
    write_associations(pairs, associations)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset": statistics, "groundtruth": str(gt_path) if gt_path else None,
                "preprocessing_seconds": time.monotonic() - started,
                "model_sha256": file_hash(args.model) if args.mode == "clearir" else None}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared {len(pairs)} pairs in {args.output}")
        return
    # Run SLAM in a fresh process after TensorFlow exits, releasing its CPU/GPU
    # allocations. This also avoids TensorFlow/LibTorch runtime interactions.
    command = [str(args.binary), str(args.vocabulary), str(settings), str(associations), str(args.output)]
    if args.video:
        command.append("--video")
    env = os.environ.copy()
    env["DARKSLAM_SUPERPOINT_MODEL"] = str(args.superpoint)
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    print(f"Running DarkSLAM on {len(pairs)} IR-D pairs; log: {args.output / 'slam.log'}", flush=True)
    with (args.output / "slam.log").open("w") as log:
        with subprocess.Popen(command, cwd=args.output, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("IR-D progress:", "Loading SuperPoint model from:")):
                    print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f"SLAM exited with {returncode}; see {args.output / 'slam.log'}")
    trajectory = read_trajectory(args.output / "trajectory.txt", "s")
    if len(trajectory) < 3:
        raise RuntimeError(f"SLAM produced fewer than three poses; see {args.output / 'slam.log'}")
    metrics = {"input_pairs": len(pairs), "estimated_poses": len(trajectory),
               "tracked_fraction": len(trajectory) / len(pairs),
               "pipeline_seconds": time.monotonic() - started}
    if gt is not None:
        ate, aligned = evaluate_ate(gt, trajectory, args.gt_max_difference, args.gt_offset)
        metrics.update(ate)
        with (args.output / "aligned_positions.txt").open("w") as stream:
            stream.write("# timestamp x y z (SE3 aligned, metres)\n")
            for stamp, x, y, z in aligned:
                stream.write(f"{seconds(stamp)} {x:.9f} {y:.9f} {z:.9f}\n")
        if args.plot:
            from .plotting import plot_trajectory
            path = plot_trajectory(gt, trajectory, args.output / "trajectory.png",
                                   args.gt_max_difference, args.gt_offset)
            print(f"Trajectory plot saved to {path}", flush=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"IR-D evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
