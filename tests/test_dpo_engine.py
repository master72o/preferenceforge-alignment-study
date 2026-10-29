"""
Unit Tests for Direct Preference Optimization (DPO) Implicit Rewards & Objectives.
"""

import pytest
from preferenceforge.core.dpo_engine import DPOEngine
from preferenceforge.core.schema import PreferenceRecord


def test_implicit_reward_calculation():
    # Policy logp = -10.0, Ref logp = -12.0 => log ratio = +2.0 => beta * ratio = 0.2
    r_hat = DPOEngine.compute_implicit_reward(policy_logp=-10.0, ref_logp=-12.0, beta=0.1)
    assert r_hat == pytest.approx(0.2, abs=1e-4)


def test_implicit_margin_and_loss():
    # Chosen improved relative to ref (+2.0 log ratio), Rejected degraded (-2.0 log ratio)
    margin = DPOEngine.compute_implicit_margin(
        pol_w=-10.0, pol_l=-20.0, ref_w=-12.0, ref_l=-18.0, beta=0.1
    )
    # ( -10 - -12 ) - ( -20 - -18 ) = 2.0 - (-2.0) = 4.0 => beta * 4.0 = 0.4
    assert margin == pytest.approx(0.4, abs=1e-4)

    loss = DPOEngine.compute_dpo_loss(margin)
    assert loss < 0.6  # -log(sigma(0.4)) approx 0.513


def test_evaluate_dpo_records():
    rec = PreferenceRecord(
        pair_id="p1",
        prompt="Write a test",
        response_chosen="def test_foo(): pass",
        response_rejected="bad code",
        policy_logps={"chosen": -8.0, "rejected": -16.0},
        ref_logps={"chosen": -10.0, "rejected": -12.0},
    )
    res = DPOEngine.evaluate_dpo_records([rec], beta=0.1)
    assert res["total_samples"] == 1
    assert res["accuracy"] == 1.0
    assert res["mean_margin"] > 0.0
    assert res["mean_dpo_loss"] > 0.0


def test_beta_sweep():
    rec = PreferenceRecord(
        pair_id="p1",
        prompt="Write a test",
        response_chosen="chosen",
        response_rejected="rejected",
        policy_logps={"chosen": -10.0, "rejected": -20.0},
        ref_logps={"chosen": -12.0, "rejected": -15.0},
    )
    sweep = DPOEngine.sweep_beta([rec], beta_values=[0.05, 0.1, 0.2])
    assert len(sweep) == 3
    # Margin should scale linearly with beta
    margins = [s["mean_margin"] for s in sweep]
    assert margins[1] == pytest.approx(margins[0] * 2, rel=1e-3)
    assert margins[2] == pytest.approx(margins[1] * 2, rel=1e-3)


def test_ipo_loss_and_simpo_margin():
    # For beta = 0.1, target log-ratio diff is 1 / (2 * 0.1) = 5.0
    assert DPOEngine.compute_ipo_loss(5.0, beta=0.1) == pytest.approx(0.0, abs=1e-6)
    assert DPOEngine.compute_ipo_loss(3.0, beta=0.1) == pytest.approx(4.0, abs=1e-6)

    # SimPO length-normalized margin:
    # pol_w = -10 over 10 tokens (-1.0/tok), pol_l = -30 over 15 tokens (-2.0/tok)
    # beta * (-1.0 - (-2.0)) - gamma = 2.0 * 1.0 - 0.5 = 1.5
    simpo_margin = DPOEngine.compute_simpo_margin(
        pol_w=-10.0, pol_l=-30.0, len_w=10, len_l=15, beta=2.0, gamma=0.5
    )
    assert simpo_margin == pytest.approx(1.5, abs=1e-4)


def test_audit_kl_drift_and_over_optimization():
    normal_rec = PreferenceRecord(
        pair_id="p_norm",
        prompt="Normal prompt",
        response_chosen="good",
        response_rejected="bad",
        policy_logps={"chosen": -10.0, "rejected": -18.0},
        ref_logps={"chosen": -12.0, "rejected": -15.0},
    )
    drifted_rec = PreferenceRecord(
        pair_id="p_drift",
        prompt="Drifted prompt",
        response_chosen="good",
        response_rejected="bad",
        policy_logps={"chosen": -5.0, "rejected": -45.0},
        ref_logps={"chosen": -12.0, "rejected": -15.0},  # rejected drift = -30.0 (> 15.0 threshold)
    )
    audit = DPOEngine.audit_kl_drift([normal_rec, drifted_rec], beta=0.1, kl_threshold=15.0)
    assert audit["total_samples"] == 2
    assert audit["over_optimized_count"] == 1
    assert audit["over_optimization_rate"] == 0.5
    assert audit["outlier_ids"] == ["p_drift"]
    assert audit["max_abs_kl_drift"] == 30.0
