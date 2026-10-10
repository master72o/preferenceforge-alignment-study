# Preference Modeling & Reward Model Calibration (`preference-modeling-rlhf`)

> Pairwise Preference Modeling, Bradley-Terry Sigmoid Probability, and Expected Calibration Error (ECE) Framework for RLHF & DPO Pipelines.

[![CI](https://github.com/master72o/preference-modeling-rlhf/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/preference-modeling-rlhf/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Summary

In Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), a critical failure mode occurs when Reward Models become **overconfident** or **miscalibrated**. If a reward model predicts high confidence for an incorrect candidate, the PPO policy optimization collapses due to reward hacking.

`preference-modeling-rlhf` provides a rigorous evaluation suite for reward model calibration, measuring Expected Calibration Error (ECE), Pairwise Preference Accuracy, Log Loss, and Bradley-Terry Sigmoid Probabilities ($P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$) across 3 model checkpoints:
1. **RM v1 (Baseline)**: Uncalibrated Reward Model Checkpoint.
2. **RM v2 (DPO Calibrated)**: Direct Preference Optimization (DPO) Loss Calibrated RM.
3. **RM v3 (Margin Scaled)**: Margin-Scaled & Temperature Tuned Reward Model.

---

## Bradley-Terry Mathematical Formulation

```
                                  1
      P(y_w ≻ y_l | x) = ─────────────────── = σ(r(x, y_w) - r(x, y_l))
                          1 + exp(-Δr)

      where Δr = r(x, y_w) - r(x, y_l)
```

- **Binary Cross-Entropy Loss**: $\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r(x, y_w) - r(x, y_l))]$
- **Expected Calibration Error (ECE)**: $\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$

---

## Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/master72o/preference-modeling-rlhf.git
cd preference-modeling-rlhf

# Install in editable mode
pip install -e .

# Run pytest suite
pytest
```

---

## Executing Reward Model Benchmark CLI

Run the reward model calibration evaluation CLI:

```bash
python -m preference_reward.cli run \
  --dataset data/preference_dataset.jsonl \
  --output-dir results \
  --report-path reports/preference_reward_report.md \
  --figures-dir reports/figures
```

---

## Controlled Reward Benchmark Results

| Reward Model Checkpoint | Pairwise Accuracy | Mean Margin ($\Delta r$) | Expected Calibration Error (ECE) | Log Loss | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RM v1 (Baseline)** | `100.0%` | `+0.3000` | `18.0%` | `0.4550` | `0.1420` |
| **RM v2 (DPO Calibrated)** | `100.0%` | `+1.9800` | `8.0%` | `0.1520` | `0.0380` |
| **RM v3 (Margin Scaled)** | `100.0%` | `+4.7000` | `2.0%` | `0.0100` | `0.0020` |

### Key Calibration Visualizations

- **Accuracy & ECE Progress**: `reports/figures/reward_accuracy_comparison.png`
- **Reliability Diagram**: `reports/figures/calibration_reliability_diagram.png`
- **Reward Margin ($\Delta r$) Distribution**: `reports/figures/reward_margin_distribution.png`

---

## Repository Structure

```
preference-modeling-rlhf/
├── .github/workflows/ci.yml     # Continuous Integration workflow
├── pyproject.toml               # Package build metadata
├── requirements.txt             # Project dependencies
├── preference_reward/           # Core Python package
│   ├── __init__.py
│   ├── schema.py                # Data models & preference pairs
│   ├── evaluator.py             # Bradley-Terry sigmoid & loss engine
│   ├── experiment_runner.py    # Multi-checkpoint runner
│   ├── metrics.py               # ECE calculator & aggregator
│   ├── visualizer.py            # Reliability diagrams & charts
│   ├── report_generator.py      # Markdown report & JSON/CSV exporter
│   └── cli.py                   # CLI entry point
├── data/
│   ├── README.md
│   └── preference_dataset.jsonl # Pairwise preference benchmark
├── results/                     # JSON & CSV exported results
├── reports/
│   ├── preference_reward_report.md # Generated research report
│   └── figures/                 # Chart graphics
└── tests/                       # Unit and integration tests
```

---


---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in data/.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run pytest or the package CLI to re-compute all scores, metrics, and summary tables locally.

## License

MIT License © 2026 AI Evaluation Engineering Team.
