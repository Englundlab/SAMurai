import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import tifffile

from app.core import (filter_boxes, illumination_correct, boxes_to_shapes,
                               shapes_to_boxes, export_result)


class CoreTests(unittest.TestCase):
    def test_duplicate_boxes_keep_one_and_nested_box_removed(self):
        boxes = [[0, 0, 20, 20], [0, 0, 20, 20], [2, 2, 8, 8], [30, 30, 40, 40]]
        b, s, counts = filter_boxes(boxes, [.7, .9, .95, .2])
        self.assertEqual(len(b), 1)
        self.assertAlmostEqual(float(s[0]), .9, places=5)

    def test_no_detections_and_strict_threshold(self):
        self.assertEqual(len(filter_boxes([], [])[0]), 0)
        self.assertEqual(len(filter_boxes([[0, 0, 5, 5]], [.25])[0]), 0)

    def test_coordinates_and_clipping(self):
        b = np.array([[2, 3, 9, 11], [-2, -1, 30, 40]])
        got = shapes_to_boxes(boxes_to_shapes(b), (20, 25))
        np.testing.assert_array_equal(got, [[2, 3, 9, 11], [0, 0, 25, 20]])
        with self.assertRaises(ValueError):
            shapes_to_boxes([np.array([[0, 2], [2, 4], [4, 2], [2, 0]])], (20, 20))

    def test_constant_illumination_unchanged(self):
        a = np.full((16, 20), 128, dtype=np.uint8)
        np.testing.assert_allclose(illumination_correct(a, 4), a / 255., atol=1e-6)

    def test_box_validation_tolerates_float32_edit_drift(self):
        shape = boxes_to_shapes([[2200.123, 1300.456, 2290.789, 1390.321]])[0].astype(np.float32)
        shape[1, 0] = np.nextafter(shape[1, 0], np.float32(np.inf))
        shape[2, 1] = np.nextafter(shape[2, 1], np.float32(np.inf))
        # This reproduces a false rotation rejection in the old validator.
        self.assertGreater(len(np.unique(shape[:, 0].round(4))), 2)
        got = shapes_to_boxes([shape], (2048, 2880))
        np.testing.assert_allclose(got[0], [shape[:, 1].min(), shape[:, 0].min(),
                                          shape[:, 1].max(), shape[:, 0].max()])
        skewed = shape.copy()
        skewed[1, 0] += 2
        with self.assertRaises(ValueError):
            shapes_to_boxes([skewed], (2048, 2880))

    def test_exports_follow_edited_labels_and_calibration(self):
        a = np.zeros((12, 15), dtype=np.uint32)
        a[2:5, 3:7] = 8  # nonconsecutive IDs must survive export
        a[2, 3] = 0  # manual erase
        with tempfile.TemporaryDirectory() as tmp:
            p = export_result(a, tmp, "example.png", .5, "um")
            np.testing.assert_array_equal(tifffile.imread(p / "labels.tif"), a)
            np.testing.assert_array_equal(tifffile.imread(p / "binary.tif"), (a > 0) * 255)
            with (p / "measurements.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label_id"], "8")
            self.assertEqual(float(rows[0]["area_px2"]), 11)
            self.assertEqual(float(rows[0]["area"]), 2.75)
            self.assertEqual(p.name, "example")
            with self.assertRaises(FileExistsError):
                export_result(a, tmp, "example.png")
            (p / "user_notes.txt").write_text("keep")
            self.assertEqual(export_result(a, tmp, "example.png", overwrite=True), p)
            self.assertTrue((p / "user_notes.txt").exists())

    def test_selectable_exports_and_excel_values(self):
        from openpyxl import load_workbook
        a = np.zeros((12, 15), dtype=np.uint32)
        a[2:5, 3:7] = 8
        with tempfile.TemporaryDirectory() as tmp:
            p = export_result(a, tmp, "sample.png", .5, "um", formats=["csv", "xlsx", "colored"])
            self.assertEqual({f.name for f in p.iterdir()},
                             {"measurements.csv", "measurements.xlsx", "colored_instances.tif", "session.json"})
            rgb = tifffile.imread(p / "colored_instances.tif")
            self.assertEqual(rgb.shape, (12, 15, 3))
            self.assertTrue((rgb[a == 0] == 0).all())
            self.assertTrue((rgb[2, 3] > 0).all())
            workbook = load_workbook(p / "measurements.xlsx", data_only=True)
            sheet = workbook.active
            self.assertEqual(sheet["A2"].value, 8)
            self.assertEqual(sheet["B2"].value, 12)
            self.assertEqual(sheet["F2"].value, 3.)
            self.assertEqual(sheet["J2"].value, "um")
            self.assertEqual(sheet["F2"].data_type, "n")
            workbook.close()
            for option, filename in (("binary", "binary.tif"), ("labels", "labels.tif"),
                                     ("csv", "measurements.csv"), ("xlsx", "measurements.xlsx"),
                                     ("colored", "colored_instances.tif")):
                p = export_result(a, tmp, "sample.png", formats=[option], folder_name=option)
                self.assertEqual({f.name for f in p.iterdir()}, {filename, "session.json"})
            with self.assertRaises(ValueError):
                export_result(a, tmp, "sample.png", formats=[])

    def test_failed_export_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("app.core.write_measurement_xlsx", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    export_result(np.zeros((4, 4), dtype=np.uint32), tmp, "test.png", formats=["labels", "xlsx"])
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
