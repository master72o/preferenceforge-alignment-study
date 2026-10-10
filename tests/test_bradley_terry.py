"""
Unit Tests for Bradley-Terry Modeling, Temperature Scaling & Expected Calibration Error (ECE).
"""

import pytest
import numpy as np
from preferenceforge.core.bradley_terry import BradleyTerryEngine
from preferenceforge.core.schema import PreferenceRecord


def test_sigmoid_numerical_stability():
    assert BradleyTerryEngine.sigmoid(0.0) == 0.5
    assert BradleyTerryEngine.sigmoid(100.0) > 0.9999
    assert BradleyTerryEngine.sigmoid(-100.0) < 0.0001


def test_preference_probability_dynamics():
    # Equal rewards
    assert BradleyTerryEngine.preference_probability(1.0, 1.0) == pytest.approx(0.5, abs=1e-3)
    # Chosen strictly higher
    p_high = BradleyTerryEngine.preference_probability(3.0, -1.0)
    assert p_high > 0.95
    # Chosen strictly lower
    p_low = BradleyTerryEngine.preference_probability(-1.0, 3.0)
    assert p_low < 0.05


def test_temperature_scaling_effect():
    r_w, r_l = 2.0, 0.0
    p_sharp = BradleyTerryEngine.preference_probability(r_w, r_l, temperature=0.5)
    p_base = BradleyTerryEngine.preference_probability(r_w, r_l, temperature=1.0)
    p_soft = BradleyTerryEngine.preference_probability(r_w, r_l, temperature=4.0)

    assert p_sharp > p_base > p_soft > 0.5


def test_ece_perfect_calibration():
    # Confidence matches empirical accuracy perfectly
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    ece, mce, bins = BradleyTerryEngine.compute_ece_and_bins(probs, labels, n_bins=10)
    assert ece < 0.01
    assert len(bins) == 10


def test_ece_miscalibration():
    # Model is 99% confident but accuracy is 0%
    probs = [0.99, 0.99, 0.99, 0.99]
    labels = [0, 0, 0, 0]
    ece, mce, bins = BradleyTerryEngine.compute_ece_and_bins(probs, labels, n_bins=10)
    assert ece > 0.90
    assert mce > 0.90


def test_temperature_optimization():
    margins = [2.0, 1.5, 0.5, -0.5, 3.0]
    labels = [1, 1, 1, 0, 1]
    opt_temp = BradleyTerryEngine.optimize_temperature(margins, labels, initial_temp=1.0)
    assert 0.05 <= opt_temp <= 10.0


def test_evaluate_model():
    rec = PreferenceRecord(
        pair_id="p1",
        prompt="Explain recursion",
        response_chosen="A function that calls itself.",
        response_rejected="Loops forever until crash.",
        rewards_v3={"chosen": 2.5, "rejected": -1.5},
    )
    metrics = BradleyTerryEngine.evaluate_model([rec], model_key="RM_v3_Margin_Scaled")
    assert metrics.total_samples == 1
    assert metrics.accuracy == 1.0
    assert metrics.mean_margin == 4.0
    assert metrics.brier_score < 0.05
