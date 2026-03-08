"""pcmabinf — Post Contextual Multi-Armed Bandit Inference.

Public API re-exports.
"""

from pcmabinf.config import ExperimentConfig
from pcmabinf.data import BanditData, OPEResult
from pcmabinf.estimators import OPEEstimator
from pcmabinf.logging_policy import LoggingConfig, run_logging_policy
from pcmabinf.metrics import ESTIMATOR_FIELDS, compute_coverage_metrics
from pcmabinf.policy import (
    ConstantPolicy,
    ContextualPolicy,
    FrequencyPolicy,
    MostFrequentPolicy,
    TargetPolicyProtocol,
    UniformPolicy,
    make_target_policy,
)
from pcmabinf.simulate import run_bandit_simulations, run_ope_simulations
from pcmabinf.world import OpenMLCC18World

__all__ = [
    "BanditData",
    "OPEResult",
    "ExperimentConfig",
    "LoggingConfig",
    "OpenMLCC18World",
    "TargetPolicyProtocol",
    "ConstantPolicy",
    "UniformPolicy",
    "MostFrequentPolicy",
    "FrequencyPolicy",
    "ContextualPolicy",
    "make_target_policy",
    "run_logging_policy",
    "OPEEstimator",
    "run_bandit_simulations",
    "run_ope_simulations",
    "compute_coverage_metrics",
    "ESTIMATOR_FIELDS",
]
