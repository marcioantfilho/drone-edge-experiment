from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from benchmark import _exact_operating_metrics
from plot_results import add_latency_columns, aggregate_seeds, knee_table
from visdrone2yolo import update_dataset_yaml


class DatasetSplitTests(unittest.TestCase):
    def test_converting_test_does_not_replace_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "VisDrone.yaml"
            yaml_path.write_text(yaml.safe_dump({
                "path": str(root), "train": "images/train", "val": "images/val"
            }))
            cfg = update_dataset_yaml(yaml_path, root, "test")
            self.assertEqual(cfg["train"], "images/train")
            self.assertEqual(cfg["val"], "images/val")
            self.assertEqual(cfg["test"], "images/test")


class MetricTests(unittest.TestCase):
    def test_exact_macro_and_micro_metrics(self):
        # Rows=predicted, columns=true, last index=background.
        matrix = np.array([[8, 1, 2], [2, 9, 1], [0, 1, 0]], dtype=float)
        metrics = _exact_operating_metrics(matrix)
        self.assertAlmostEqual(metrics["recall_micro"], 17 / 21)
        self.assertAlmostEqual(metrics["precision_micro"], 17 / 23)
        self.assertEqual(metrics["tp"], 17)
        self.assertEqual(metrics["fp"], 6)
        self.assertEqual(metrics["fn"], 4)

    def test_latency_uses_decimal_kb_and_operating_postprocess(self):
        df = pd.DataFrame([{
            "mean_pixels": 1_000_000, "mean_kb": 100.0,
            "pre_op_ms": 2.0, "post_op_ms": 3.0,
            "fwd_median_ms": 5.0, "fwd_p95_ms": 7.0,
        }])
        out = add_latency_columns(df, {
            "encode_ms_per_mpixel": 8.0, "overhead_ms": 15.0,
            "bandwidths_mbps": [10],
        })
        self.assertAlmostEqual(out.loc[0, "tx_ms@10"], 80.0)
        self.assertAlmostEqual(out.loc[0, "e2e_ms@10"], 113.0)

    def test_selection_aggregates_seeds_before_optimizing(self):
        rows = []
        for seed, recall in [(0, 0.7), (1, 0.9), (2, 0.8), (67, 0.8)]:
            rows.append({
                "model": "m", "split": "val", "capture_tag": "360p",
                "capture_h": 360, "imgsz": 640, "precision_mode": "fp32",
                "seed": seed, "recall_op": recall, "precision_op": 0.8,
                "map50": 0.7, "map50_95": 0.5, "mean_kb": 25.0,
                "e2e_ms@10": 100.0,
            })
        agg = aggregate_seeds(pd.DataFrame(rows), [
            "recall_op", "precision_op", "map50", "map50_95", "e2e_ms@10"
        ])
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg.loc[0, "n_seeds"], 4)
        table = knee_table(agg, {"bandwidths_mbps": [10]}, "recall_op", 0.5, "mean")
        self.assertEqual(table.loc[0, "capture"], "360p")


if __name__ == "__main__":
    unittest.main()
