import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from darkslam.data import (match_timestamps, pair_images, read_trajectory,
                           seconds, timestamp_ns, validate_images,
                           write_associations, write_settings)
from darkslam.metrics import align_trajectories, evaluate_ate

ROOT = Path(__file__).resolve().parents[1]


class TimestampTests(unittest.TestCase):
    def test_nanoseconds_preserve_submicrosecond_precision(self):
        stamp = 1769105011871997254
        self.assertEqual(timestamp_ns(str(stamp)), stamp)
        self.assertEqual(timestamp_ns(seconds(stamp), "s"), stamp)

    def test_pairing_is_one_to_one_and_closest_first(self):
        self.assertEqual(match_timestamps([100, 110, 200], [106, 201], 10),
                         [(110, 106), (200, 201)])
        self.assertEqual(match_timestamps([100], [111], 10), [])

    def test_csv_and_tum_normalize_to_same_timebase(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "vicon.csv"
            tum = Path(directory) / "trajectory.txt"
            csv.write_text("timestamp,x,y,z,qx,qy,qz,qw\n1769105011871997254,1,2,3,0,0,0,1\n")
            tum.write_text("1769105011.871997254 1 2 3 0 0 0 1\n")
            a, b = read_trajectory(csv), read_trajectory(tum)
            self.assertEqual(list(a), list(b))
            np.testing.assert_array_equal(list(a.values()), list(b.values()))


class GeometryTests(unittest.TestCase):
    def test_raw_pairing_does_not_change_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "infra1").mkdir()
            (root / "depth").mkdir()
            cv2.imwrite(str(root / "infra1/1769105011871997254.png"), np.zeros((480, 640), np.uint8))
            depth = np.full((480, 640), 2345, np.uint16)
            path = root / "depth/1769105011872997254.png"
            cv2.imwrite(str(path), depth)
            original = path.read_bytes()
            pairs, stats = pair_images(root, "infra1", "depth", "auto", .02, 1, None)
            validate_images(pairs)
            write_associations(pairs, root / "associations.txt")
            self.assertEqual(stats["selected_pairs"], 1)
            self.assertIn("1769105011.871997254", (root / "associations.txt").read_text())
            self.assertEqual(path.read_bytes(), original)

    def test_settings_uses_sequence_calibration_and_depth_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "camera_info").mkdir()
            (root / "camera_info/depth.txt").write_text(
                "height: 480\nwidth: 640\ndistortion_model: plumb_bob\n"
                "k: [400, 0, 320, 0, 401, 240, 0, 0, 1]\nd: [0, 0, 0, 0, 0]\n")
            settings = write_settings(root, ROOT / "configs/clearir.yaml", root / "settings.yaml", None, 5000)
            fs = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
            self.assertEqual(fs.getNode("Camera1.fx").real(), 400)
            self.assertEqual(fs.getNode("Camera1.fy").real(), 401)
            self.assertEqual(fs.getNode("RGBD.DepthMapFactor").real(), 5000)
            fs.release()


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, .1]])

    def poses(self, xyz, offset=0):
        return {1_000_000_000 + i * 100_000_000 + offset: np.r_[p, 0, 0, 0, 1]
                for i, p in enumerate(xyz)}

    def test_rigid_frame_change_has_zero_ate(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        gt = self.poses(self.positions @ rotation.T + [3, 4, 5])
        metrics, _ = evaluate_ate(gt, self.poses(self.positions))
        self.assertLess(metrics["ate_rmse_m"], 1e-10)

    def test_metric_scale_error_is_not_fitted_away(self):
        metrics, _ = evaluate_ate(self.poses(self.positions), self.poses(self.positions * 2))
        self.assertEqual(metrics["scale"], 1)
        self.assertGreater(metrics["ate_rmse_m"], .5)

    def test_offset_is_added_to_estimate(self):
        metrics, _ = evaluate_ate(self.poses(self.positions, 1_000_000_000),
                                  self.poses(self.positions), offset=1.)
        self.assertEqual(metrics["matched_poses"], 4)

    def test_missing_matches_fail(self):
        with self.assertRaisesRegex(ValueError, "three"):
            evaluate_ate(self.poses(self.positions, 10_000_000_000), self.poses(self.positions))

    def test_plot_alignment_matches_metric_export_with_offset(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        reference = self.positions @ rotation.T + [3, 4, 5]
        gt = self.poses(reference, 1_000_000_000)
        # A pose outside the matched interval must not affect fitting or plotting.
        gt[100_000_000_000] = np.array([100, 200, 300, 0, 0, 0, 1.])
        estimated = self.poses(self.positions)
        metrics, exported = evaluate_ate(gt, estimated, offset=1.)
        stamps, target, aligned = align_trajectories(gt, estimated, offset=1.)
        self.assertEqual(stamps, sorted(estimated))
        np.testing.assert_allclose(target, reference)
        np.testing.assert_allclose(aligned, reference)
        np.testing.assert_allclose(np.array(exported)[:, 1:], aligned)
        self.assertEqual(metrics["matched_poses"], len(stamps))


if __name__ == "__main__":
    unittest.main()
