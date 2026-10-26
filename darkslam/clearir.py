"""Inference adapter for the supplied, serialized CLEAR-IR model.

The H5 contains the authoritative architecture and weights. Importing the
original research script would start training, download VGG, and select a GPU.
"""
from pathlib import Path

import cv2
import numpy as np


class CLEARIR:
    def __init__(self, model_path: Path, device: str = "auto", threads: int = 2):
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        else:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        expected = (None, 480, 640, 1)
        if self.model.input_shape != expected or self.model.output_shape != expected:
            raise ValueError(f"Expected CLEAR-IR input/output {expected}, got "
                             f"{self.model.input_shape}/{self.model.output_shape}")
        print(f"CLEAR-IR model device: {self.model.weights[0].value.device}", flush=True)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Preserve the camera geometry: no resizing or per-frame min/max scaling.
        if image.shape != (480, 640) or image.dtype != np.uint8:
            raise ValueError("CLEAR-IR requires 640x480, single-channel uint8 IR images")
        batch = image.astype(np.float32)[None, ..., None] / 255.0
        prediction = np.asarray(self.model(batch, training=False))[0, ..., 0]
        if not np.isfinite(prediction).all():
            raise ValueError("CLEAR-IR returned non-finite pixel values")
        return np.clip(prediction * 255.0, 0, 255).astype(np.uint8)


def enhance_sequence(model_path: Path, pairs: list, output: Path,
                     device: str, threads: int) -> list:
    enhancer = CLEARIR(model_path, device, threads)
    output.mkdir(parents=True, exist_ok=True)
    enhanced = []
    for index, (ir_stamp, ir_path, depth_stamp, depth_path) in enumerate(pairs):
        image = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode {ir_path}; run git lfs pull in the dataset repo")
        prediction = enhancer(image)
        target = output / f"{index:08d}.png"
        if not cv2.imwrite(str(target), prediction):
            raise OSError(f"Could not write {target}")
        enhanced.append((ir_stamp, target, depth_stamp, depth_path))
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(pairs):
            print(f"CLEAR-IR: {index + 1}/{len(pairs)} frames", flush=True)
    return enhanced


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CLEAR-IR inference worker")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "auto"], required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    pairs = [(a, Path(ir), b, Path(depth)) for a, ir, b, depth in json.loads(args.manifest.read_text())]
    enhance_sequence(args.model, pairs, args.output, args.device, args.threads)
