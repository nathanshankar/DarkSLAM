"""Metric-scale, rigidly aligned translational ATE; no fitted scale."""
import numpy as np

from .data import match_timestamps


def align_trajectories(groundtruth: dict, estimated: dict, tolerance: float = 0.02,
                       offset: float = 0.0) -> tuple:
    """Return matched estimate timestamps, GT positions, and SE(3)-aligned positions.

    Both metrics and plotting use this exact association and fixed-scale fit.
    Returned timestamps retain the estimate's original timebase.
    """
    shift = int(round(offset * 1e9))
    shifted = {stamp + shift: pose for stamp, pose in estimated.items()}
    matches = match_timestamps(list(shifted), list(groundtruth), int(tolerance * 1e9))
    if len(matches) < 3:
        raise ValueError("Fewer than three GT/estimated timestamp matches; check units and --gt-offset")
    source = np.array([shifted[a][:3] for a, _ in matches])
    target = np.array([groundtruth[b][:3] for _, b in matches])
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(source_zero, tol=1e-8) < 1:
        raise ValueError("Estimated trajectory has no translation; cannot align it")
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_mean - rotation @ source_mean
    aligned = (rotation @ source.T).T + translation
    stamps = [a - shift for a, _ in matches]
    return stamps, target, aligned


def evaluate_ate(groundtruth: dict, estimated: dict, tolerance: float = 0.02,
                 offset: float = 0.0) -> tuple:
    stamps, target, aligned = align_trajectories(groundtruth, estimated, tolerance, offset)
    errors = np.linalg.norm(aligned - target, axis=1)
    result = {
        "alignment": "SE3 (fixed metric scale)", "scale": 1.0,
        "matched_poses": len(stamps), "estimated_poses": len(estimated),
        "groundtruth_poses": len(groundtruth),
        "estimated_pose_match_fraction": len(stamps) / len(estimated),
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_mean_m": float(errors.mean()), "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(errors.max()),
        "evaluated_duration_s": (stamps[-1] - stamps[0]) / 1e9,
        "gt_offset_seconds_added_to_estimate": offset,
    }
    return result, [(stamp, *xyz) for stamp, xyz in zip(stamps, aligned)]
