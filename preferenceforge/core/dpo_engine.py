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
