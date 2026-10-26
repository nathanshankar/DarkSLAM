"""Headless trajectory comparison using the evaluation's fixed-scale SE(3) fit."""
from pathlib import Path

import numpy as np

from .data import read_trajectory
from .metrics import align_trajectories


def plot_trajectory(groundtruth: dict, estimated: dict, destination: Path,
                    tolerance: float = 0.02, offset: float = 0.0) -> Path:
    # Import only when plotting is requested. Agg works without a display server.
    import warnings

    with warnings.catch_warnings():
        # Some ROS/system installations mix versions of the optional 3D toolkit.
        # Planar projections do not need it; suppress only that irrelevant warning.
        warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

    stamps, target, aligned = align_trajectories(groundtruth, estimated, tolerance, offset)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    figure = Figure(figsize=(14, 5), layout="constrained")
    FigureCanvasAgg(figure)
    try:
        for panel, (first, second, name) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")], 1):
            axis = figure.add_subplot(1, 3, panel)
            for positions, label, color, style in [
                (target, "Ground truth (matched)", "#333333", "-"),
                (aligned, "Estimate (SE(3) aligned)", "#0072B2", "--"),
            ]:
                axis.plot(positions[:, first], positions[:, second], color=color, linestyle=style,
                          label=label, linewidth=1.8)
            axis.set(title=f"{name} projection", xlabel=f"{name[0]} [m]", ylabel=f"{name[1]} [m]")
            # Equal distance scales on both axes avoid exaggerating drift.
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(True, alpha=0.25)
            if panel == 1:
                axis.legend(loc="best", fontsize=9)
        figure.suptitle("Estimated trajectory vs ground truth — SE(3), scale = 1\n"
                        f"ATE RMSE: {rmse:.4f} m · {len(stamps)} matched poses", fontsize=12)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200)
    finally:
        figure.clear()
    return destination


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Plot an existing TUM trajectory against CSV/TUM ground truth")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Plot filename (default: trajectory.png beside input trajectory)")
    parser.add_argument("--gt-timestamp-unit", choices=["auto", "ns", "s"], default="auto")
    parser.add_argument("--gt-max-difference", type=float, default=0.02)
    parser.add_argument("--gt-offset", type=float, default=0.0, help="Seconds added to estimated timestamps for matching")
    args = parser.parse_args()
    try:
        path = plot_trajectory(read_trajectory(args.groundtruth, args.gt_timestamp_unit),
                               read_trajectory(args.trajectory, "s"),
                               args.output or args.trajectory.parent / "trajectory.png",
                               args.gt_max_difference, args.gt_offset)
    except (ImportError, ValueError, OSError, RuntimeError) as error:
        print(f"Trajectory plotting failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"Trajectory plot saved to {path}")


if __name__ == "__main__":
    main()
