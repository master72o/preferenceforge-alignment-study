"""
Unit Tests for PreferenceForge CLI Commands.
"""

import os
import json
import subprocess
import sys
import pytest


def test_cli_benchmark_command(tmp_path):
    # Create mini jsonl
    mini_data = [
        {
            "pair_id": "t1",
            "prompt": "Test prompt",
            "response_chosen": "Good response",
            "response_rejected": "Bad response",
            "domain": "coding",
            "rewards_v1": {"chosen": 0.4, "rejected": 0.1},
            "rewards_v2": {"chosen": 1.2, "rejected": -0.5},
            "rewards_v3": {"chosen": 2.5, "rejected": -1.5},
            "rewards_v4_length_debiased": {"chosen": 2.3, "rejected": -1.6},
        }
    ]
    ds_path = tmp_path / "mini.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for d in mini_data:
            f.write(json.dumps(d) + "\n")

    out_dir = tmp_path / "results"
    rep_path = tmp_path / "report.md"

    cmd = [
        sys.executable,
        "-m",
        "preferenceforge.cli",
        "benchmark",
        "--dataset",
        str(ds_path),
        "--output-dir",
        str(out_dir),
        "--report-path",
        str(rep_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Benchmark completed successfully" in res.stdout
    assert os.path.exists(out_dir / "preferenceforge_summary.json")
    assert os.path.exists(out_dir / "preferenceforge_predictions.csv")
    assert os.path.exists(rep_path)


def test_cli_calibrate_command(tmp_path):
    mini_data = [
        {
            "pair_id": "t1",
            "prompt": "Test prompt",
            "response_chosen": "Good response",
            "response_rejected": "Bad response",
            "domain": "coding",
            "rewards_v3": {"chosen": 2.5, "rejected": -1.5},
        }
    ]
    ds_path = tmp_path / "mini.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for d in mini_data:
            f.write(json.dumps(d) + "\n")

    cmd = [
        sys.executable,
        "-m",
        "preferenceforge.cli",
        "calibrate",
        "--dataset",
        str(ds_path),
        "--model",
        "RM_v3_Margin_Scaled",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Expected Calibration Error" in res.stdout or "ECE" in res.stdout


def test_cli_bias_audit_command(tmp_path):
    mini_data = [
        {
            "pair_id": "t1",
            "prompt": "Test prompt",
            "response_chosen": "Short",
            "response_rejected": "Long long long long response",
            "domain": "coding",
            "rewards_v1": {"chosen": 0.4, "rejected": 0.1},
        }
    ]
    ds_path = tmp_path / "mini.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for d in mini_data:
            f.write(json.dumps(d) + "\n")

    cmd = [
        sys.executable,
        "-m",
        "preferenceforge.cli",
        "bias-audit",
        "--dataset",
        str(ds_path),
        "--model",
        "RM_v1_Baseline",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Pearson Length Corr" in res.stdout


def test_cli_train_dry_run(tmp_path):
    mini_data = [
        {
            "pair_id": "t1",
            "prompt": "Test prompt",
            "response_chosen": "Good",
            "response_rejected": "Bad",
        }
    ]
    ds_path = tmp_path / "mini.jsonl"
    with open(ds_path, "w", encoding="utf-8") as f:
        for d in mini_data:
            f.write(json.dumps(d) + "\n")

    out_dir = tmp_path / "adapter_checkpoints"

    cmd = [
        sys.executable,
        "-m",
        "preferenceforge.cli",
        "train",
        "--dataset-path",
        str(ds_path),
        "--output-dir",
        str(out_dir),
        "--dry-run",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Dry-run mode enabled" in res.stdout
    assert os.path.exists(out_dir / "adapter_config.json")
