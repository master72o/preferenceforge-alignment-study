"""
Direct Preference Optimization (DPO) Implicit Reward & Objective Analysis Engine.
Implements analytical formulations from Rafailov et al. (NeurIPS 2023).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from preferenceforge.core.schema import PreferenceRecord


class DPOEngine:
    """Direct Preference Optimization (DPO) mathematical engine."""

    @staticmethod
    def sigmoid(z: float) -> float:
        if z >= 0:
            ez = np.exp(-z)
            return float(1.0 / (1.0 + ez))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))

    @classmethod
    def compute_implicit_reward(
        cls, policy_logp: float, ref_logp: float, beta: float = 0.1
    ) -> float:
        """r_hat(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))."""
        return float(beta * (policy_logp - ref_logp))

    @classmethod
    def compute_implicit_margin(
        cls,
        pol_w: float,
        pol_l: float,
        ref_w: float,
        ref_l: float,
        beta: float = 0.1,
    ) -> float:
        """Delta r_hat = beta * ((log pi(yw) - log ref(yw)) - (log pi(yl) - log ref(yl)))."""
        r_w = cls.compute_implicit_reward(pol_w, ref_w, beta=beta)
        r_l = cls.compute_implicit_reward(pol_l, ref_l, beta=beta)
        return float(r_w - r_l)

    @classmethod
    def compute_dpo_loss(cls, margin: float) -> float:
        """L_DPO = -log sigma(Delta r_hat)."""
        prob = cls.sigmoid(margin)
        prob = np.clip(prob, 1e-7, 1.0 - 1e-7)
        return float(-np.log(prob))

    @classmethod
    def compute_ipo_loss(cls, log_ratio_diff: float, beta: float = 0.1) -> float:
        """
        Identity Preference Optimization (IPO) quadratic loss (Azar et al., AISTATS 2024):
        L_IPO = (Delta log(pi / pi_ref) - 1 / (2 * beta))^2.
        Prevents unbounded margin overfitting when deterministic preferences dominate.
        """
        b = max(1e-5, float(beta))
        target = 1.0 / (2.0 * b)
        return float((log_ratio_diff - target) ** 2)

    @classmethod
    def compute_simpo_margin(
        cls,
        pol_w: float,
        pol_l: float,
        len_w: int,
        len_l: int,
        beta: float = 2.0,
        gamma: float = 0.5,
    ) -> float:
        """
        Simple Preference Optimization (SimPO) reference-free length-normalized margin (Meng et al., NeurIPS 2024):
        Delta r_SimPO = (beta / |y_w|) * log pi(y_w|x) - (beta / |y_l|) * log pi(y_l|x) - gamma.
        """
        lw = max(1, int(len_w))
        ll = max(1, int(len_l))
        avg_logp_w = float(pol_w) / lw
        avg_logp_l = float(pol_l) / ll
        return float(beta * (avg_logp_w - avg_logp_l) - gamma)

    @classmethod
    def audit_kl_drift(
        cls,
        records: List[PreferenceRecord],
        beta: float = 0.1,
        kl_threshold: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Audits policy-to-reference log-ratio drift and detects reward over-optimization outliers.
        """
        valid_records = [
            r for r in records if r.policy_logps and r.ref_logps and "chosen" in r.policy_logps
        ]
        if not valid_records:
            return {
                "total_samples": 0,
                "mean_chosen_kl_drift": 0.0,
                "mean_rejected_kl_drift": 0.0,
                "max_abs_kl_drift": 0.0,
                "mean_ipo_loss": 0.0,
                "over_optimized_count": 0,
                "over_optimization_rate": 0.0,
                "outlier_ids": [],
            }

        chosen_drifts = []
        rejected_drifts = []
        max_abs_drifts = []
        ipo_losses = []
        outlier_ids = []

        for rec in valid_records:
            pol_w = float(rec.policy_logps.get("chosen", -15.0))
            pol_l = float(rec.policy_logps.get("rejected", -20.0))
            ref_w = float(rec.ref_logps.get("chosen", -16.0))
            ref_l = float(rec.ref_logps.get("rejected", -18.0))

            drift_w = pol_w - ref_w
            drift_l = pol_l - ref_l
            log_ratio_diff = drift_w - drift_l
            max_abs = max(abs(drift_w), abs(drift_l))

            chosen_drifts.append(drift_w)
            rejected_drifts.append(drift_l)
            max_abs_drifts.append(max_abs)
            ipo_losses.append(cls.compute_ipo_loss(log_ratio_diff, beta=beta))

            if max_abs > kl_threshold:
                outlier_ids.append(rec.pair_id)

        n = len(valid_records)
        return {
            "total_samples": n,
            "mean_chosen_kl_drift": round(float(np.mean(chosen_drifts)), 4),
            "mean_rejected_kl_drift": round(float(np.mean(rejected_drifts)), 4),
            "max_abs_kl_drift": round(float(np.max(max_abs_drifts)), 4),
            "mean_ipo_loss": round(float(np.mean(ipo_losses)), 4),
            "over_optimized_count": len(outlier_ids),
            "over_optimization_rate": round(len(outlier_ids) / n, 4),
            "outlier_ids": outlier_ids,
        }

    @classmethod
    def evaluate_dpo_records(
        cls,
        records: List[PreferenceRecord],
        beta: float = 0.1,
    ) -> Dict[str, Any]:
        """Evaluates DPO implicit rewards, margins, and loss across preference records."""
        valid_records = [
            r for r in records if r.policy_logps and r.ref_logps and "chosen" in r.policy_logps
        ]
        if not valid_records:
            return {
                "total_samples": 0,
                "accuracy": 0.0,
                "mean_margin": 0.0,
                "mean_loss": 0.0,
                "beta": beta,
            }

        margins = []
        losses = []
        accuracies = []
        implicit_rewards_chosen = []
        implicit_rewards_rejected = []

        for rec in valid_records:
            pol_w = rec.policy_logps.get("chosen", -15.0)
            pol_l = rec.policy_logps.get("rejected", -20.0)
            ref_w = rec.ref_logps.get("chosen", -16.0)
            ref_l = rec.ref_logps.get("rejected", -18.0)

            r_w = cls.compute_implicit_reward(pol_w, ref_w, beta=beta)
            r_l = cls.compute_implicit_reward(pol_l, ref_l, beta=beta)
            margin = r_w - r_l
            loss = cls.compute_dpo_loss(margin)

            implicit_rewards_chosen.append(r_w)
            implicit_rewards_rejected.append(r_l)
            margins.append(margin)
            losses.append(loss)
            accuracies.append(1.0 if margin > 0.0 else 0.0)

        return {
            "total_samples": len(valid_records),
            "beta": beta,
            "accuracy": round(float(np.mean(accuracies)), 4),
            "mean_margin": round(float(np.mean(margins)), 4),
            "mean_dpo_loss": round(float(np.mean(losses)), 4),
            "mean_implicit_reward_chosen": round(float(np.mean(implicit_rewards_chosen)), 4),
            "mean_implicit_reward_rejected": round(float(np.mean(implicit_rewards_rejected)), 4),
            "margin_std": round(float(np.std(margins)), 4),
        }

    @classmethod
    def sweep_beta(
        cls,
        records: List[PreferenceRecord],
        beta_values: List[float] = None,
    ) -> List[Dict[str, Any]]:
        """Sweeps beta parameter to analyze regularization vs implicit margin dynamics."""
        if beta_values is None:
            beta_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        results = []
        for b in beta_values:
            eval_res = cls.evaluate_dpo_records(records, beta=b)
            results.append({
                "beta": b,
                "mean_margin": eval_res.get("mean_margin", 0.0),
                "mean_dpo_loss": eval_res.get("mean_dpo_loss", 0.0),
                "accuracy": eval_res.get("accuracy", 0.0),
            })
        return results
