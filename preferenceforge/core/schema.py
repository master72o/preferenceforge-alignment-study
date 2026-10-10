"""
Data Schemas for PreferenceForge Alignment & Calibration Studio.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional
import json


class ModelArchitecture(str, Enum):
    REWARD_MODEL = "reward_model"
    DPO_DIRECT = "dpo_direct"
    PPO_ACTOR = "ppo_actor"


@dataclass
class PreferenceRecord:
    """A single pairwise preference instance with multi-model rewards and annotator data."""
    pair_id: str
    prompt: str
    response_chosen: str
    response_rejected: str
    domain: str = "general"
    human_margin: float = 1.0
    error_tags_rejected: List[str] = field(default_factory=list)
    rewards_v1: Dict[str, float] = field(default_factory=dict)
    rewards_v2: Dict[str, float] = field(default_factory=dict)
    rewards_v3: Dict[str, float] = field(default_factory=dict)
    rewards_v4_length_debiased: Dict[str, float] = field(default_factory=dict)
    policy_logps: Dict[str, float] = field(default_factory=dict)
    ref_logps: Dict[str, float] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def chosen_char_length(self) -> int:
        return len(self.response_chosen)

    @property
    def rejected_char_length(self) -> int:
        return len(self.response_rejected)

    @property
    def chosen_token_estimate(self) -> int:
        return max(1, len(self.response_chosen.split()))

    @property
    def rejected_token_estimate(self) -> int:
        return max(1, len(self.response_rejected.split()))

    @property
    def length_delta_chars(self) -> int:
        """Positive if chosen is longer, negative if rejected is longer."""
        return self.chosen_char_length - self.rejected_char_length

    @property
    def length_delta_tokens(self) -> int:
        return self.chosen_token_estimate - self.rejected_token_estimate

    def get_rewards(self, model_key: str) -> Dict[str, float]:
        """Returns chosen and rejected reward for a given model key."""
        key_map = {
            "RM_v1_Baseline": self.rewards_v1,
            "RM_v2_DPO_Calibrated": self.rewards_v2,
            "RM_v3_Margin_Scaled": self.rewards_v3,
            "RM_v4_Length_Debiased": self.rewards_v4_length_debiased,
            "v1": self.rewards_v1,
            "v2": self.rewards_v2,
            "v3": self.rewards_v3,
            "v4": self.rewards_v4_length_debiased,
        }
        return key_map.get(model_key, self.rewards_v3)

    def get_reward_margin(self, model_key: str) -> float:
        r = self.get_rewards(model_key)
        return float(r.get("chosen", 0.0) - r.get("rejected", 0.0))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceRecord":
        return cls(
            pair_id=str(data.get("pair_id", data.get("id", "pair_000"))),
            prompt=str(data.get("prompt", "")),
            response_chosen=str(data.get("response_chosen", data.get("response_a", ""))),
            response_rejected=str(data.get("response_rejected", data.get("response_b", ""))),
            domain=str(data.get("domain", "general")),
            human_margin=float(data.get("human_margin", 1.0)),
            error_tags_rejected=list(data.get("error_tags_rejected", [])),
            rewards_v1=dict(data.get("rewards_v1", {"chosen": 0.5, "rejected": 0.0})),
            rewards_v2=dict(data.get("rewards_v2", {"chosen": 1.2, "rejected": -0.6})),
            rewards_v3=dict(data.get("rewards_v3", {"chosen": 2.5, "rejected": -1.5})),
            rewards_v4_length_debiased=dict(data.get("rewards_v4_length_debiased", {"chosen": 2.3, "rejected": -1.6})),
            policy_logps=dict(data.get("policy_logps", {})),
            ref_logps=dict(data.get("ref_logps", {})),
            annotations=list(data.get("annotations", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceBenchmarkDataset:
    records: List[PreferenceRecord]

    @classmethod
    def from_jsonl(cls, file_path: str) -> "PreferenceBenchmarkDataset":
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(PreferenceRecord.from_dict(json.loads(line)))
        return cls(records=records)

    def __len__(self) -> int:
        return len(self.records)

    def get_by_domain(self, domain: str) -> List[PreferenceRecord]:
        return [r for r in self.records if r.domain == domain]

    def domains(self) -> List[str]:
        return sorted(list(set(r.domain for r in self.records)))


@dataclass
class CalibrationBin:
    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    avg_confidence: float
    empirical_accuracy: float
    calibration_gap: float


@dataclass
class CalibrationMetrics:
    total_samples: int
    accuracy: float
    mean_margin: float
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    log_loss: float
    optimal_temperature: float = 1.0
    bins: List[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["accuracy_pct"] = round(self.accuracy * 100, 2)
        res["ece_pct"] = round(self.expected_calibration_error * 100, 2)
        res["mce_pct"] = round(self.max_calibration_error * 100, 2)
        return res


@dataclass
class LengthBiasMetrics:
    pearson_correlation: float
    spearman_correlation: float
    p_value: float
    concordant_accuracy: float  # Chosen is longer
    discordant_accuracy: float  # Chosen is shorter
    neutral_accuracy: float     # Similar length (+- 15%)
    concordant_count: int
    discordant_count: int
    neutral_count: int
    verbosity_bias_flag: bool
    bias_severity: str          # "minimal", "moderate", "severe"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCheckpointConfig:
    key: str
    display_name: str
    description: str
    architecture: ModelArchitecture


@dataclass
class EvaluationSummary:
    dataset_size: int
    domains: List[str]
    models: Dict[str, CalibrationMetrics]
    length_bias: Dict[str, LengthBiasMetrics]
    domain_breakdown: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_size": self.dataset_size,
            "domains": self.domains,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "length_bias": {k: v.to_dict() for k, v in self.length_bias.items()},
            "domain_breakdown": self.domain_breakdown,
        }
