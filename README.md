# PreferenceForge: Open Alignment & Preference Optimization Studio (`preferenceforge-alignment-study`)

> End-to-End Alignment & Preference Optimization Framework: Bradley-Terry Reward Modeling, DPO Training Pipeline, Length Bias Diagnostics, and Calibration Observatory.

[![CI](https://github.com/master72o/preferenceforge-alignment-study/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/preferenceforge-alignment-study/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/master72o/preferenceforge-alignment-study/blob/main/notebooks/preferenceforge_gpu_run.ipynb)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Studio_Live-FF4B4B.svg?logo=streamlit)](https://github.com/master72o/preferenceforge-alignment-study)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Executive Overview

In modern post-training alignment—spanning Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO)—two pervasive failure modes silently undermine model quality:

1. **Reward Miscalibration**: Uncalibrated reward models output extreme probabilities for ambiguous or incorrect candidates, triggering policy collapse or catastrophic forgetting during PPO/RLHF.
2. **Verbosity Exploitation (Length Bias)**: Preference models frequently reward superficial padding and rambling responses over concise, high-density answers. When tested on **discordant pairs** (where the superior answer is intentionally shorter), standard models often pick the inferior verbose response.

**PreferenceForge** provides an end-to-end alignment engineering studio that bridges mathematical formulation, empirical diagnostics, and free GPU training:
- **Bradley-Terry Calibration Engine**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Brier score, and Platt temperature scaling ($T^*$).
- **Direct Preference Optimization (DPO) Pipeline**: Modular training orchestration using Hugging Face `transformers`, `trl` (`DPOTrainer`), and `peft` (LoRA/QLoRA).
- **Length Bias & Verbosity Observatory**: Quantifies token-reward correlation ($r$), isolates discordant pairs, and evaluates length-regularized reward objectives.
- **Exportable Free GPU Notebook**: A self-contained Google Colab & Kaggle Jupyter notebook running DPO on a free NVIDIA T4 GPU.
- **Interactive Streamlit Web Studio**: Live side-by-side preference arena, calibration curves, and annotator agreement inspector.

---

## 📐 Mathematical Foundations

### 1. Bradley-Terry Preference Probability
Given a prompt $x$ and candidate responses $y_w$ (chosen) and $y_l$ (rejected), the probability that $y_w$ is preferred over $y_l$ is parameterized as:

$$P(y_w \succ y_l \mid x) = \sigma\left(\frac{r_\theta(x, y_w) - r_\theta(x, y_l)}{T}\right) = \frac{1}{1 + \exp\left(-\frac{\Delta r}{T}\right)}$$

where $\Delta r = r(x, y_w) - r(x, y_l)$ is the reward margin and $T$ is the temperature scaling factor.

### 2. Direct Preference Optimization (DPO)
DPO reparameterizes the RLHF objective directly through the policy $\pi_\theta$ and frozen reference model $\pi_{ref}$, bypassing explicit reward modeling:

$$\mathcal{L}_{DPO}(\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)} \right) \right]$$

The implicit reward induced by policy $\pi_\theta$ is defined as:

$$\hat{r}_\theta(x, y) = \beta \cdot \left(\log \pi_\theta(y \mid x) - \log \pi_{ref}(y \mid x)\right)$$

### 3. Expected Calibration Error (ECE)
Grouping predicted preference probabilities into $B$ equal-width bins:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$

Platt temperature scaling optimizes $T^*$ by minimizing Negative Log-Likelihood (NLL) over the validation calibration set:

$$T^* = \arg\min_T -\sum_{i=1}^{N} \left[ y_i \log \sigma\left(\frac{\Delta r_i}{T}\right) + (1 - y_i) \log \left(1 - \sigma\left(\frac{\Delta r_i}{T}\right)\right) \right]$$

---

## 📊 Controlled Benchmark Results

The benchmark evaluates **35 multi-domain preference pairs** across 4 progressive alignment checkpoints:

| Checkpoint | Pairwise Accuracy | Mean Margin ($\Delta r$) | ECE | MCE | Brier Score | Length Corr ($r$) | Discordant Acc | Verbosity Bias |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RM v1 (Baseline)** | `88.6%` | `+0.393` | `38.7%` | `58.2%` | `0.1840` | `+0.524` | `75.0%` | `SEVERE` 🚨 |
| **RM v2 (DPO Calibrated)** | `100.0%` | `+2.111` | `12.4%` | `24.1%` | `0.0410` | `+0.704` | `100.0%` | `MODERATE` |
| **RM v3 (Margin Scaled)** | `100.0%` | `+4.491` | `2.2%` | `3.8%` | `0.0032` | `+0.766` | `100.0%` | `MODERATE` |
| **RM v4 (Length-Debiased)** | `100.0%` | `+4.603` | `1.1%` | `1.8%` | `0.0018` | `+0.313` | `100.0%` | `MINIMAL` ✅ |

### Key Experimental Insights
1. **Calibration Breakthrough**: Scaling reward margins and optimizing temperature reduces Expected Calibration Error from **38.7%** (v1) down to **1.1%** (v4), ensuring that high predicted probability reliably reflects genuine quality differences.
2. **Exposing Length Bias**: Baseline uncalibrated models exhibit a steep correlation ($r = +0.524$) between token difference and reward margin. In adversarial scenarios where the superior response is concise, uncalibrated models suffer failure rates up to 25%.
3. **Regularization Recovery**: Applying length-normalized reward adjustments suppresses verbosity correlation down to $r = +0.313$ while sustaining 100% pairwise ranking accuracy across all domains.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data ["Pairwise Preference Data"]
        D1[data/preference_dataset.jsonl] --> D2[Multi-Domain Prompts]
        D2 --> D3[Annotator Ratings & Error Tags]
        D2 --> D4[Adversarial Length-Bias Pairs]
    end

    subgraph Core ["PreferenceForge Core Engine"]
        BT[BradleyTerryEngine] --> CAL[ECE & Platt Temperature Optimizer]
        DPO[DPOEngine] --> IMP[Implicit Reward & Beta Sweeps]
        LB[LengthBiasDetector] --> SUB[Discordant vs Concordant Subgroups]
    end

    subgraph Execution ["Execution & Training"]
        TRAIN[preferenceforge.training.dpo_trainer] --> LORA[LoRA / QLoRA Adapter]
        NOTEBOOK[notebooks/preferenceforge_gpu_run.ipynb] --> COLAB[Free Google Colab T4 GPU]
    end

    subgraph Interface ["Evaluation & UI"]
        CLI[preferenceforge CLI] --> EXP[results/ & reports/]
        STREAMLIT[app.py] --> DASH[Streamlit Web Studio]
    end

    Data --> Core
    Core --> Execution
    Core --> Interface
```

---

## ⚡ Quickstart

### 1. Installation
```bash
# Clone repository
git clone https://github.com/master72o/preferenceforge-alignment-study.git
cd preferenceforge-alignment-study

# Install in editable mode with development dependencies
pip install -e .
pip install pytest pytest-cov
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Launch Interactive Streamlit Studio
```bash
streamlit run app.py
```

---

## 🛠️ Command-Line Interface (CLI)

PreferenceForge provides an integrated CLI for automated benchmarking, calibration audits, and training:

```bash
# 1. Run full alignment benchmark across all checkpoints
preferenceforge benchmark \
  --dataset data/preference_dataset.jsonl \
  --output-dir results \
  --report-path reports/preference_reward_report.md

# 2. Inspect Bradley-Terry ECE and optimize temperature scaling
preferenceforge calibrate \
  --dataset data/preference_dataset.jsonl \
  --model RM_v3_Margin_Scaled

# 3. Audit verbosity and length bias
preferenceforge bias-audit \
  --dataset data/preference_dataset.jsonl \
  --model RM_v1_Baseline

# 4. Dry-run DPO LoRA training pipeline
preferenceforge train \
  --model-name HuggingFaceTB/SmolLM-135M-Instruct \
  --dataset-path data/preference_dataset.jsonl \
  --beta 0.1 \
  --dry-run
```

---

## 🚀 Free GPU Training on Google Colab

Train an actual open-source model using Direct Preference Optimization (DPO) and LoRA on a free Google Colab or Kaggle T4 GPU:

1. Click the **[Open in Colab](https://colab.research.google.com/github/master72o/preferenceforge-alignment-study/blob/main/notebooks/preferenceforge_gpu_run.ipynb)** badge at the top of this repository.
2. Ensure GPU runtime is selected (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
3. Run all cells: the notebook downloads `HuggingFaceTB/SmolLM-135M-Instruct`, loads the PreferenceForge dataset, runs 50 steps of DPO fine-tuning with TRL, and evaluates the resulting adapter's calibration and length bias!

---

## 📂 Repository Structure

```
preferenceforge-alignment-study/
├── .github/workflows/ci.yml         # GitHub Actions CI workflow (Python 3.10)
├── configs/                         # YAML training configs
│   ├── dpo_smollm_lora.yaml         # SmolLM-135M DPO config
│   └── dpo_qwen_lora.yaml           # Qwen2.5-0.5B DPO config
├── data/                            # Standardized pairwise preference benchmark
│   ├── README.md
│   └── preference_dataset.jsonl     # 35 multi-domain preference pairs
├── notebooks/
│   └── preferenceforge_gpu_run.ipynb # Exportable Google Colab / Kaggle notebook
├── preferenceforge/                 # Core flagship package
│   ├── __init__.py
│   ├── cli.py                       # CLI entry points
│   ├── core/
│   │   ├── bradley_terry.py         # BT sigmoid, ECE, Platt scaling
│   │   ├── dpo_engine.py            # DPO implicit rewards & beta sweep
│   │   ├── evaluator.py             # Benchmark orchestrator
│   │   ├── length_bias.py           # Verbosity detector & regularizer
│   │   └── schema.py                # Data dataclasses
│   └── training/
│       ├── __init__.py
│       └── dpo_trainer.py           # HF Transformers + TRL DPOTrainer script
├── preference_reward/               # Backward-compatibility package shim
│   ├── __init__.py
│   ├── cli.py                       # Legacy CLI runner
│   ├── evaluator.py
│   ├── experiment_runner.py
│   ├── metrics.py
│   ├── report_generator.py
│   ├── schema.py
│   └── visualizer.py
├── reports/                         # Research reports and generated figures
│   ├── preference_reward_report.md  # Benchmark report
│   └── figures/                     # Generated PNG charts
├── results/                         # Exported JSON and CSV records
├── tests/                           # 24 unit and integration tests
├── app.py                           # Interactive Streamlit Web Studio
├── pyproject.toml                   # Package build metadata & scripts
└── README.md                        # Documentation
```

---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in `data/preference_dataset.jsonl`.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run `pytest` or `preferenceforge benchmark` to re-compute all scores, metrics, calibration curves, and summary tables locally.

---

## 📜 License

MIT License © 2026 Master72o / AI Alignment Engineering Team.
