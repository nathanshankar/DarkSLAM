"""Timestamp association, camera settings, and CLEARIR trajectory loading."""
import csv
from bisect import bisect_left, bisect_right
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import yaml


def timestamp_ns(value: str, unit: str = "auto") -> int:
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError(f"Invalid timestamp: {value}")
    if unit == "auto":
        unit = "ns" if abs(number) >= Decimal("1e14") else "s"
    return int(number if unit == "ns" else number * 1_000_000_000)


def seconds(stamp: int) -> str:
    return format(Decimal(int(stamp)) / 1_000_000_000, ".9f")


def match_timestamps(first: list, second: list, tolerance_ns: int) -> list:
    """Closest-first, one-to-one timestamp matching, ordered by first timestamp."""
    if tolerance_ns < 0:
        raise ValueError("Timestamp tolerance must be nonnegative")
    second = sorted(second)
    candidates = []
    for a in first:
        low = bisect_left(second, a - tolerance_ns)
        high = bisect_right(second, a + tolerance_ns)
        candidates.extend((abs(a - b), a, b) for b in second[low:high])
    used_first, used_second, matches = set(), set(), []
    for _, a, b in sorted(candidates):
        if a not in used_first and b not in used_second:
            matches.append((a, b))
            used_first.add(a)
            used_second.add(b)
    return sorted(matches)


def image_index(directory: Path, unit: str) -> dict:
    if not directory.is_dir():
        raise ValueError(f"Image directory not found: {directory}")
    index = {}
    for path in sorted(directory.glob("*.png")):
        stamp = timestamp_ns(path.stem, unit)
        if stamp in index:
            raise ValueError(f"Duplicate image timestamp: {path}")
        index[stamp] = path.resolve()
    if not index:
        raise ValueError(f"No timestamp-named PNG images in {directory}")
    return index


def pair_images(dataset: Path, ir_dir: str, depth_dir: str, unit: str,
                tolerance: float, stride: int, limit: int | None) -> tuple:
    ir = image_index(dataset / ir_dir, unit)
    depth = image_index(dataset / depth_dir, unit)
    matches = match_timestamps(list(ir), list(depth), int(tolerance * 1e9))
    selected = matches[::stride]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No IR/depth timestamp pairs; check timestamp units and tolerance")
    return [(a, ir[a], b, depth[b]) for a, b in selected], {
        "ir_images": len(ir), "depth_images": len(depth),
        "associated_pairs": len(matches), "selected_pairs": len(selected),
        "max_pair_delta_seconds": max(abs(a-b) for a, b in selected) / 1e9,
    }


def validate_images(pairs: list) -> None:
    for _, ir_path, _, depth_path in pairs:
        ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if ir is None or depth is None:
            raise ValueError(f"Cannot decode {ir_path} or {depth_path}; run git lfs pull")
        if ir.dtype != np.uint8 or ir.shape != (480, 640):
            raise ValueError(f"Expected uint8 640x480 grayscale IR: {ir_path}")
        if depth.dtype != np.uint16 or depth.shape != ir.shape:
            raise ValueError(f"Expected registered uint16 640x480 depth: {depth_path}")


def write_associations(pairs: list, destination: Path) -> None:
    # Absolute paths are accepted by the C++ runner; never copy or quantize depth.
    with destination.open("w") as stream:
        stream.write("# ir_timestamp ir_image depth_timestamp depth_image (seconds)\n")
        for a, ir, b, depth in pairs:
            if any(c.isspace() for c in str(ir) + str(depth)):
                raise ValueError("TUM association image paths cannot contain whitespace")
            stream.write(f"{seconds(a)} {ir} {seconds(b)} {depth}\n")


def write_settings(dataset: Path, template: Path, destination: Path,
                   explicit: Path | None, depth_factor: float) -> Path:
    if explicit:
        # An explicit settings file is the authority for camera/depth calibration.
        text = explicit.read_text()
    else:
        text = template.read_text()
        info_path = dataset / "camera_info/depth.txt"
        if not info_path.is_file():
            raise ValueError(
                f"Missing {info_path}. For the CLEARIR camera, add --settings {template}; "
                "for another camera, supply --settings with its IR/depth calibration."
            )
        info = next(yaml.safe_load_all(info_path.read_text()))
        if (info["height"], info["width"]) != (480, 640):
            raise ValueError("CameraInfo resolution must be 640x480")
        if info.get("distortion_model") != "plumb_bob":
            raise ValueError("Unsupported distortion model; supply --settings")
        k, d = info["k"], list(info["d"])
        if len(k) != 9 or len(d) not in (4, 5):
            raise ValueError("Invalid pinhole CameraInfo")
        d += [0.0] * (5 - len(d))
        values = {"Camera1.fx": k[0], "Camera1.fy": k[4],
                  "Camera1.cx": k[2], "Camera1.cy": k[5],
                  **dict(zip(["Camera1.k1", "Camera1.k2", "Camera1.p1", "Camera1.p2", "Camera1.k3"], d)),
                  "RGBD.DepthMapFactor": depth_factor}
        text = "\n".join(f"{line.split(':')[0]}: {float(values[line.split(':')[0]])}"
                         if line.split(":")[0] in values else line
                         for line in text.splitlines()) + "\n"
    destination.write_text(text)
    settings = cv2.FileStorage(str(destination), cv2.FILE_STORAGE_READ)
    try:
        if not settings.isOpened():
            raise ValueError(f"Could not open settings: {destination}")
        if (settings.getNode("Camera.height").real(), settings.getNode("Camera.width").real()) != (480, 640):
            raise ValueError("Settings must describe the native 640x480 image geometry")
        for name in ["Camera1.fx", "Camera1.fy", "RGBD.DepthMapFactor"]:
            if settings.getNode(name).real() <= 0:
                raise ValueError(f"Settings require a positive {name}")
    finally:
        settings.release()
    return destination


def read_trajectory(path: Path, unit: str = "auto") -> dict:
    poses = {}
    with path.open() as stream:
        if path.suffix.lower() == ".csv":
            rows = csv.DictReader(stream)
            records = ([row[k] for k in ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]]
                       for row in rows)
        else:
            records = (line.split() for line in stream if line.strip() and not line.lstrip().startswith("#"))
        for row in records:
            if len(row) != 8:
                raise ValueError(f"Expected timestamp tx ty tz qx qy qz qw in {path}")
            pose = np.asarray(row[1:], dtype=float)
            if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError(f"Invalid pose in {path}")
            stamp = timestamp_ns(row[0], unit)
            if stamp in poses:
                raise ValueError(f"Duplicate pose timestamp in {path}: {row[0]}")
            poses[stamp] = pose
    return poses
