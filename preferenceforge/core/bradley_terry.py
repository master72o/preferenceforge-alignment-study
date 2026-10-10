"""
Bradley-Terry Preference Probability Modeling, Temperature Scaling & Calibration Engine.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple
from preferenceforge.core.schema import (
    PreferenceRecord,
    CalibrationBin,
    CalibrationMetrics,
)


class BradleyTerryEngine:
    """Mathematical formulation of Bradley-Terry pairwise preference modeling & calibration."""

    @staticmethod
    def sigmoid(z: float) -> float:
        """Numerically stable sigmoid function."""
        if z >= 0:
            ez = np.exp(-z)
            return float(1.0 / (1.0 + ez))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))

    @classmethod
    def preference_probability(
        cls, reward_w: float, reward_l: float, temperature: float = 1.0
    ) -> float:
        """P(w > l | x) = sigma((r_w - r_l) / T)."""
        temp = max(1e-4, float(temperature))
        margin = (reward_w - reward_l) / temp
        prob = cls.sigmoid(margin)
        return float(np.clip(prob, 1e-7, 1.0 - 1e-7))

    @staticmethod
    def log_loss(prob: float, y_true: float = 1.0) -> float:
        p = float(np.clip(prob, 1e-7, 1.0 - 1e-7))
        return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    @staticmethod
    def brier_score(prob: float, y_true: float = 1.0) -> float:
        return float((prob - y_true) ** 2)

    @classmethod
    def compute_ece_and_bins(
        cls,
        probabilities: List[float],
        labels: List[int],
        n_bins: int = 10,
    ) -> Tuple[float, float, List[CalibrationBin]]:
        """Computes Expected Calibration Error (ECE), Max Calibration Error (MCE), and calibration bins."""
        if not probabilities or not labels or len(probabilities) != len(labels):
            return 0.0, 0.0, []

        total_samples = len(probabilities)
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bins: List[CalibrationBin] = []

        probs_arr = np.array(probabilities)
        labels_arr = np.array(labels)

        for i in range(n_bins):
            bin_lower = float(bin_boundaries[i])
            bin_upper = float(bin_boundaries[i + 1])

            if i == 0:
                mask = (probs_arr >= bin_lower) & (probs_arr <= bin_upper)
            else:
                mask = (probs_arr > bin_lower) & (probs_arr <= bin_upper)

            count = int(np.sum(mask))
            if count > 0:
                bin_acc = float(np.mean(labels_arr[mask]))
                bin_conf = float(np.mean(probs_arr[mask]))
                gap = abs(bin_acc - bin_conf)
                ece += (count / total_samples) * gap
                mce = max(mce, gap)
            else:
                bin_acc = 0.0
                bin_conf = (bin_lower + bin_upper) / 2.0
                gap = 0.0

            bins.append(
                CalibrationBin(
                    bin_index=i,
                    lower_bound=round(bin_lower, 3),
                    upper_bound=round(bin_upper, 3),
                    count=count,
                    avg_confidence=round(bin_conf, 4),
                    empirical_accuracy=round(bin_acc, 4),
                    calibration_gap=round(gap, 4),
                )
            )

        return float(ece), float(mce), bins

    @classmethod
    def optimize_temperature(
        cls,
        margins: List[float],
        labels: List[int],
        initial_temp: float = 1.0,
    ) -> float:
        """Finds optimal temperature T* that minimizes Negative Log-Likelihood (Platt/Temperature Scaling)."""
        if not margins or not labels:
            return 1.0

        margins_arr = np.array(margins, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.float64)

        def nll_objective(t: np.ndarray) -> float:
            temp = max(1e-3, float(t[0]))
            scaled_margins = margins_arr / temp
            # sigmoid prob
            probs = 1.0 / (1.0 + np.exp(-scaled_margins))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(labels_arr * np.log(probs) + (1.0 - labels_arr) * np.log(1.0 - probs))
            return float(loss)

        res = minimize(
            nll_objective,
            x0=np.array([initial_temp]),
            bounds=[(0.05, 10.0)],
            method="L-BFGS-B",
        )
        if res.success:
            return float(res.x[0])
        return float(initial_temp)

    @classmethod
    def evaluate_model(
        cls,
        records: List[PreferenceRecord],
        model_key: str,
        temperature: float = 1.0,
        optimize_temp: bool = True,
    ) -> CalibrationMetrics:
        """Evaluates a set of preference records for a given reward model checkpoint."""
        if not records:
            return CalibrationMetrics(
                total_samples=0,
                accuracy=0.0,
                mean_margin=0.0,
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                brier_score=0.0,
                log_loss=0.0,
                optimal_temperature=1.0,
            )

        margins = []
        labels = []  # 1 if chosen was preferred (ground truth is 1 for chosen)
        probs = []
        log_losses = []
        briers = []

        for rec in records:
            rewards = rec.get_rewards(model_key)
            rw = float(rewards.get("chosen", 0.0))
            rl = float(rewards.get("rejected", 0.0))
            margin = rw - rl
            margins.append(margin)

            # Ground truth is chosen > rejected (label 1)
            is_correct = 1 if margin > 0.0 else 0
            labels.append(is_correct)

            prob = cls.preference_probability(rw, rl, temperature=temperature)
            probs.append(prob)
            log_losses.append(cls.log_loss(prob, y_true=1.0))
            briers.append(cls.brier_score(prob, y_true=1.0))

        accuracy = float(np.mean(labels))
        mean_margin = float(np.mean(margins))
        ece, mce, bins = cls.compute_ece_and_bins(probs, labels)
        mean_log_loss = float(np.mean(log_losses))
        mean_brier = float(np.mean(briers))

        opt_temp = 1.0
        if optimize_temp and len(records) >= 5:
            opt_temp = cls.optimize_temperature(margins, labels, initial_temp=temperature)

        return CalibrationMetrics(
            total_samples=len(records),
            accuracy=round(accuracy, 4),
            mean_margin=round(mean_margin, 4),
            expected_calibration_error=round(ece, 4),
            max_calibration_error=round(mce, 4),
            brier_score=round(mean_brier, 4),
            log_loss=round(mean_log_loss, 4),
            optimal_temperature=round(opt_temp, 4),
            bins=bins,
        )
