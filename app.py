"""
PreferenceForge: Open Alignment & Preference Optimization Studio
Interactive Streamlit Application for Bradley-Terry Calibration, DPO Dynamics, and Length Bias Diagnostics.
"""

import os
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preferenceforge.core.schema import (
    PreferenceRecord,
    PreferenceBenchmarkDataset,
)
from preferenceforge.core.bradley_terry import BradleyTerryEngine
from preferenceforge.core.dpo_engine import DPOEngine
from preferenceforge.core.length_bias import LengthBiasDetector
from preferenceforge.core.evaluator import PreferenceForgeEvaluator

# Page Configuration
st.set_page_config(
    page_title="PreferenceForge | Alignment & Calibration Studio",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .delta-good { color: #10B981; }
    .delta-bad { color: #EF4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load Dataset
@st.cache_data
def load_dataset(dataset_path: str = "data/preference_dataset.jsonl"):
    if not os.path.exists(dataset_path):
        # Fallback to local repo path if running from root
        alt_path = os.path.join("repos", "preferenceforge-alignment-study", dataset_path)
        if os.path.exists(alt_path):
            dataset_path = alt_path
    return PreferenceBenchmarkDataset.from_jsonl(dataset_path)

DATASET_PATH = "data/preference_dataset.jsonl"
try:
    dataset = load_dataset(DATASET_PATH)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.title("⚖️ PreferenceForge")
st.sidebar.caption("Open Alignment & Reward Model Calibration Studio")

st.sidebar.markdown(
    """
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/master72o/preferenceforge-alignment-study/blob/main/notebooks/preferenceforge_gpu_run.ipynb)
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/master72o/preferenceforge-alignment-study)
    """
)

selected_model = st.sidebar.selectbox(
    "Active Checkpoint",
    options=[
        "RM_v3_Margin_Scaled",
        "RM_v2_DPO_Calibrated",
        "RM_v1_Baseline",
        "RM_v4_Length_Debiased",
    ],
    format_func=lambda x: {
        "RM_v1_Baseline": "RM v1 (Uncalibrated Baseline)",
        "RM_v2_DPO_Calibrated": "RM v2 (DPO Loss Calibrated)",
        "RM_v3_Margin_Scaled": "RM v3 (Margin Scaled & Temp Tuned)",
        "RM_v4_Length_Debiased": "RM v4 (Length-Debiased Regularized)",
    }[x],
)

temp_slider = st.sidebar.slider(
    "Platt Scaling Temperature (T)",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.05,
    help="Scales Bradley-Terry preference probability: P(chosen > rejected) = sigma((r_w - r_l) / T)",
)

beta_slider = st.sidebar.slider(
    "DPO Regularization (Beta)",
    min_value=0.01,
    max_value=0.5,
    value=0.1,
    step=0.01,
    help="Direct Preference Optimization inverse temperature controlling policy deviation from reference",
)

st.sidebar.divider()
st.sidebar.info(
    f"📊 **Benchmark Dataset**: {len(dataset)} pairs across {len(dataset.domains())} domains.\n\n"
    "Designed for reproducible RLHF/DPO alignment analysis."
)

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Benchmark Observatory",
    "⚔️ Pairwise Arena",
    "🎯 Calibration Inspector",
    "📏 Length Bias Audit",
    "🔍 Dataset Explorer",
])

# Evaluate across suite
evaluator = PreferenceForgeEvaluator(dataset)
summary = evaluator.run_full_suite()
curr_metrics = BradleyTerryEngine.evaluate_model(dataset.records, selected_model, temperature=temp_slider)
curr_bias = LengthBiasDetector.analyze_length_bias(dataset.records, selected_model)

# ----------------- TAB 1: Benchmark Observatory -----------------
with tab1:
    st.subheader("Alignment & Calibration Benchmark Observatory")
    st.markdown(
        "Evaluate pairwise ranking accuracy, calibration errors, and verbosity correlations across 4 distinct training checkpoints."
    )

    # Metric KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Pairwise Accuracy</div>
            <div class="metric-value">{curr_metrics.accuracy*100:.1f}%</div>
            <div class="metric-delta delta-good">Target: >90%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Expected Calib. Error (ECE)</div>
            <div class="metric-value">{curr_metrics.expected_calibration_error*100:.1f}%</div>
            <div class="metric-delta {'delta-good' if curr_metrics.expected_calibration_error < 0.05 else 'delta-bad'}">Target: &lt;5%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Reward Margin (Δr)</div>
            <div class="metric-value">{curr_metrics.mean_margin:+.2f}</div>
            <div class="metric-delta delta-good">Higher separation = clearer policy gradient</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Discordant Accuracy</div>
            <div class="metric-value">{curr_bias.discordant_accuracy*100:.1f}%</div>
            <div class="metric-delta {'delta-good' if curr_bias.discordant_accuracy > 0.8 else 'delta-bad'}">When chosen is shorter</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Length Correlation (r)</div>
            <div class="metric-value">{curr_bias.pearson_correlation:+.2f}</div>
            <div class="metric-delta {'delta-good' if abs(curr_bias.pearson_correlation) < 0.25 else 'delta-bad'}">Lower is less biased</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Checkpoint Comparison Matrix")
    matrix_rows = []
    for key, name, _ in PreferenceForgeEvaluator.CHECKPOINTS:
        m = summary.models[key]
        b = summary.length_bias[key]
        matrix_rows.append({
            "Checkpoint": key,
            "Accuracy": f"{m.accuracy*100:.1f}%",
            "Mean Margin (Δr)": f"{m.mean_margin:+.3f}",
            "ECE": f"{m.expected_calibration_error*100:.1f}%",
            "MCE": f"{m.max_calibration_error*100:.1f}%",
            "Brier Score": f"{m.brier_score:.4f}",
            "Length Corr (r)": f"{b.pearson_correlation:+.3f}",
            "Discordant Acc": f"{b.discordant_accuracy*100:.1f}%",
            "Verbosity Bias": b.bias_severity.upper(),
        })
    df_matrix = pd.DataFrame(matrix_rows)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    # Domain Breakdown Chart
    st.markdown("### Domain Accuracy Breakdown")
    dom_df = pd.DataFrame(summary.domain_breakdown).T * 100
    st.bar_chart(dom_df)

# ----------------- TAB 2: Pairwise Arena -----------------
with tab2:
    st.subheader("⚔️ Pairwise Preference Arena")
    st.markdown("Inspect or test custom prompts and candidate responses side-by-side to observe real Bradley-Terry probability calculations.")

    sample_options = [f"{r.pair_id}: {r.prompt[:60]}..." for r in dataset.records]
    selected_sample_idx = st.selectbox("Load Sample Pair from Benchmark", range(len(sample_options)), format_func=lambda i: sample_options[i])
    sample_rec = dataset.records[selected_sample_idx]

    user_prompt = st.text_area("Prompt / Instruction", value=sample_rec.prompt, height=70)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Candidate Response A (Ground Truth Chosen)**")
        resp_a = st.text_area("Response A", value=sample_rec.response_chosen, height=140, key="resp_a")
    with col_b:
        st.markdown("**Candidate Response B (Ground Truth Rejected)**")
        resp_b = st.text_area("Response B", value=sample_rec.response_rejected, height=140, key="resp_b")

    # Tokens & length analysis
    tokens_a = max(1, len(resp_a.split()))
    tokens_b = max(1, len(resp_b.split()))
    delta_tok = tokens_a - tokens_b

    # Model scores for the sample
    rewards = sample_rec.get_rewards(selected_model)
    rw_a = float(rewards.get("chosen", 2.0))
    rw_b = float(rewards.get("rejected", -1.0))
    prob_a = BradleyTerryEngine.preference_probability(rw_a, rw_b, temperature=temp_slider)
    prob_b = 1.0 - prob_a

    st.markdown("#### Model Scoring & Bradley-Terry Prediction")
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric("Response A Reward (r_A)", f"{rw_a:+.3f}", f"{tokens_a} tokens")
    with score_col2:
        st.metric("Response B Reward (r_B)", f"{rw_b:+.3f}", f"{tokens_b} tokens")
    with score_col3:
        st.metric("Reward Margin (Δr)", f"{rw_a - rw_b:+.3f}", f"Δ Tokens: {delta_tok:+d}")

    # Probability Bar
    st.markdown(f"**Preference Probability Distribution (T={temp_slider:.2f})**")
    p_col1, p_col2 = st.columns([prob_a, max(0.01, prob_b)])
    with p_col1:
        st.success(f"P(A ≻ B) = {prob_a*100:.1f}%")
    with p_col2:
        st.error(f"P(B ≻ A) = {prob_b*100:.1f}%")

    if delta_tok < -15 and prob_a > 0.5:
        st.info("💡 **Conciseness Triumph**: Response A is significantly shorter than B yet won higher reward, demonstrating resistance to verbosity padding!")
    elif delta_tok > 25 and prob_a > 0.5:
        st.warning("⚠️ **Potential Verbosity Confound**: Response A is notably longer than B. Ensure preference is driven by correctness rather than length.")

# ----------------- TAB 3: Calibration Inspector -----------------
with tab3:
    st.subheader("🎯 Calibration & Reliability Inspector")
    st.markdown(
        "A well-calibrated reward model ensures that predicted probabilities match true empirical win rates. "
        "Overconfident reward models destabilize PPO policy updates and distort DPO implicit gradients."
    )

    cal_col1, cal_col2 = st.columns([3, 2])
    with cal_col1:
        st.markdown(f"#### Reliability Diagram ({selected_model})")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")

        bin_confs = [b.avg_confidence for b in curr_metrics.bins if b.count > 0]
        bin_accs = [b.empirical_accuracy for b in curr_metrics.bins if b.count > 0]
        bin_counts = [b.count for b in curr_metrics.bins if b.count > 0]

        scatter = ax.scatter(bin_confs, bin_accs, s=[c*15 + 40 for c in bin_counts], color="#4361EE", alpha=0.8, edgecolors="k", label="Model Bins (Size = Sample Count)")
        ax.set_xlabel("Mean Predicted Confidence P(Chosen > Rejected)")
        ax.set_ylabel("Empirical Pairwise Win Rate")
        ax.set_title(f"Reliability Diagram (ECE = {curr_metrics.expected_calibration_error*100:.2f}%)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)

    with cal_col2:
        st.markdown("#### Optimal Temperature Scaling")
        st.write(
            f"Platt scaling fits a single scalar temperature $T^*$ minimizing Negative Log-Likelihood over validation pairs."
        )
        st.metric("Computed Optimal Temperature (T*)", f"{curr_metrics.optimal_temperature:.3f}")
        st.metric("Expected Calibration Error (ECE)", f"{curr_metrics.expected_calibration_error*100:.2f}%")
        st.metric("Maximum Calibration Error (MCE)", f"{curr_metrics.max_calibration_error*100:.2f}%")
        st.metric("Brier Score", f"{curr_metrics.brier_score:.4f}")

        st.markdown("#### Calibration Bin Summary")
        bin_data = []
        for b in curr_metrics.bins:
            if b.count > 0:
                bin_data.append({
                    "Bin": f"[{b.lower_bound:.2f}-{b.upper_bound:.2f}]",
                    "Count": b.count,
                    "Confidence": f"{b.avg_confidence:.3f}",
                    "Accuracy": f"{b.empirical_accuracy:.3f}",
                    "Gap": f"{b.calibration_gap:.3f}",
                })
        st.dataframe(pd.DataFrame(bin_data), use_container_width=True, hide_index=True)

# ----------------- TAB 4: Length Bias Audit -----------------
with tab4:
    st.subheader("📏 Length Bias & Verbosity Exploitation Audit")
    st.markdown(
        "A critical vulnerability in RLHF/DPO: reward models often favor longer, verbose answers over concise answers. "
        "Here we measure correlation between token difference $\\Delta L$ and reward margin $\\Delta r$."
    )

    bias_col1, bias_col2 = st.columns([3, 2])
    with bias_col1:
        st.markdown("#### Token Difference vs. Predicted Reward Margin")
        delta_tokens = [r.length_delta_tokens for r in dataset.records]
        margins = [r.get_reward_margin(selected_model) for r in dataset.records]

        fig_bias, ax_bias = plt.subplots(figsize=(7, 5))
        ax_bias.scatter(delta_tokens, margins, color="#7209B7", alpha=0.75, edgecolors="k", s=60)
        ax_bias.axhline(0, color="red", linestyle="--", alpha=0.6, label="Decision Threshold (Margin = 0)")
        ax_bias.axvline(0, color="gray", linestyle=":", alpha=0.5, label="Equal Length")

        # Trend line
        z = np.polyfit(delta_tokens, margins, 1)
        p = np.poly1d(z)
        x_vals = np.linspace(min(delta_tokens), max(delta_tokens), 100)
        ax_bias.plot(x_vals, p(x_vals), color="#F72585", linestyle="-", label=f"Trend (r = {curr_bias.pearson_correlation:+.2f})")

        ax_bias.set_xlabel("Token Difference: Chosen Tokens - Rejected Tokens")
        ax_bias.set_ylabel("Predicted Reward Margin (r_chosen - r_rejected)")
        ax_bias.set_title("Length Bias Scatter Plot")
        ax_bias.legend()
        ax_bias.grid(True, alpha=0.25)
        st.pyplot(fig_bias)

    with bias_col2:
        st.markdown("#### Subgroup Accuracy Breakdown")
        st.write("Evaluating pairs split by relative response length:")
        st.metric("Concordant Pairs Accuracy (Chosen Longer)", f"{curr_bias.concordant_accuracy*100:.1f}%", f"{curr_bias.concordant_count} pairs")
        st.metric("Discordant Pairs Accuracy (Chosen Shorter)", f"{curr_bias.discordant_accuracy*100:.1f}%", f"{curr_bias.discordant_count} pairs")
        st.metric("Neutral Pairs Accuracy (Similar Length)", f"{curr_bias.neutral_accuracy*100:.1f}%", f"{curr_bias.neutral_count} pairs")

        if curr_bias.verbosity_bias_flag:
            st.error(f"🚨 **Verbosity Exploitation Detected!** (Severity: {curr_bias.bias_severity.upper()})\n\n"
                     f"Discordant accuracy collapsed to {curr_bias.discordant_accuracy*100:.1f}%. The model incorrectly favors verbose rejected answers.")
        else:
            st.success("✅ **Clean Model**: Model maintains robust accuracy even when the chosen response is concise.")

# ----------------- TAB 5: Dataset Explorer -----------------
with tab5:
    st.subheader("🔍 Benchmark Dataset & Annotator Explorer")
    st.markdown("Explore individual prompt pairs, human consensus ratings, and annotator rationales.")

    domain_filter = st.selectbox("Filter by Domain", ["All Domains"] + dataset.domains())
    filtered_records = dataset.records if domain_filter == "All Domains" else dataset.get_by_domain(domain_filter)

    st.write(f"Showing {len(filtered_records)} preference records.")

    for rec in filtered_records:
        with st.expander(f"[{rec.pair_id}] {rec.domain.upper()}: {rec.prompt[:80]}..."):
            st.markdown(f"**Prompt**: {rec.prompt}")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.success(f"**Chosen Response** ({rec.chosen_token_estimate} tokens)\n\n{rec.response_chosen}")
            with p_col2:
                st.error(f"**Rejected Response** ({rec.rejected_token_estimate} tokens)\n\n{rec.response_rejected}")

            st.markdown(f"**Error Tags in Rejected Response**: `{', '.join(rec.error_tags_rejected) if rec.error_tags_rejected else 'None'}`")

            if rec.annotations:
                st.markdown("**Annotator Rationales**:")
                for ann in rec.annotations:
                    st.caption(f"- **{ann.get('annotator_id', 'Annotator')}** (Confidence: {ann.get('confidence', 5)}/5): {ann.get('rationale', 'N/A')}")
