"""
Unit Tests for Length Bias & Verbosity Exploitation Detector.
"""

import pytest
from preferenceforge.core.length_bias import LengthBiasDetector
from preferenceforge.core.schema import PreferenceRecord


def test_length_delta_calculation():
    rec = PreferenceRecord(
        pair_id="p1",
        prompt="Define API",
        response_chosen="Application Programming Interface.",
        response_rejected="An API is an Application Programming Interface that lets two applications talk to each other over networks with protocols.",
    )
    assert rec.chosen_token_estimate < rec.rejected_token_estimate
    assert rec.length_delta_tokens < 0


def test_discordant_pair_detection():
    # 2 pairs where chosen is shorter (discordant), 2 where chosen is longer (concordant)
    records = [
        PreferenceRecord(
            pair_id="p1",
            prompt="Prompt 1",
            response_chosen="Short accurate answer.",
            response_rejected="Extremely long verbose inaccurate rambling answer that wastes lots of tokens.",
            rewards_v1={"chosen": 0.1, "rejected": 0.5},  # Fails discordant pair!
        ),
        PreferenceRecord(
            pair_id="p2",
            prompt="Prompt 2",
            response_chosen="Another short correct fact.",
            response_rejected="Very long padded rambling paragraph that should not be chosen.",
            rewards_v1={"chosen": 0.2, "rejected": 0.4},  # Fails discordant pair!
        ),
        PreferenceRecord(
            pair_id="p3",
            prompt="Prompt 3",
            response_chosen="Long detailed comprehensive step-by-step solution to this problem.",
            response_rejected="Short wrong.",
            rewards_v1={"chosen": 0.6, "rejected": 0.1},  # Passes concordant pair
        ),
    ]

    metrics = LengthBiasDetector.analyze_length_bias(records, model_key="RM_v1_Baseline")
    assert metrics.discordant_count == 2
    assert metrics.concordant_count == 1
    # Discordant accuracy is 0.0 because rewards were higher on rejected
    assert metrics.discordant_accuracy == 0.0
    assert metrics.concordant_accuracy == 1.0
    assert metrics.verbosity_bias_flag is True
    assert metrics.bias_severity in ["moderate", "severe"]


def test_length_regularization():
    raw_margin = 2.0
    # Chosen has 50 tokens, Rejected has 10 tokens => delta = +40 tokens
    reg_margin = LengthBiasDetector.apply_length_regularization(
        raw_margin=raw_margin, chosen_tokens=50, rejected_tokens=10, penalty_coef=0.02
    )
    # penalty = 0.02 * 40 = 0.8 => 2.0 - 0.8 = 1.2
    assert reg_margin == pytest.approx(1.2, abs=1e-4)
