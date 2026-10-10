"""
PreferenceForge: End-to-End Alignment & Preference Optimization Studio.
Bradley-Terry Reward Modeling, DPO Training Pipeline, Length Bias Diagnostics, and Calibration Observatory.
"""

from preferenceforge.core.schema import (
    PreferenceRecord,
    PreferenceBenchmarkDataset,
    ModelCheckpointConfig,
    CalibrationMetrics,
    LengthBiasMetrics,
    EvaluationSummary,
)
from preferenceforge.core.bradley_terry import BradleyTerryEngine
from preferenceforge.core.dpo_engine import DPOEngine
from preferenceforge.core.length_bias import LengthBiasDetector
from preferenceforge.core.evaluator import PreferenceForgeEvaluator

__version__ = "2.0.0"

__all__ = [
    "PreferenceRecord",
    "PreferenceBenchmarkDataset",
    "ModelCheckpointConfig",
    "CalibrationMetrics",
    "LengthBiasMetrics",
    "EvaluationSummary",
    "BradleyTerryEngine",
    "DPOEngine",
    "LengthBiasDetector",
    "PreferenceForgeEvaluator",
    "__version__",
]
