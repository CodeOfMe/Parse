"""
PARSE: Principled Architecture Retention through Scenario-Embedded Pruning

Core package providing capability-preserving model compression via tri-axial
(Language x Discipline x Scenario) decomposition, CIT computation, FFN
transplantation, and dynamic capability routing.

Usage:
    from parse import (
        ComputeCIT, TransplantFFN, CapabilityRouter,
        DualFlywheelTrainer, evaluate_capabilities,
        PreservationProfile, PARSEConfig,
        export_to_gguf,
    )
"""

from .config import PARSEConfig, PreservationProfile, get_profile
from .core.cit import ComputeCIT
from .core.transplant import TransplantFFN
from .core.model import build_parse_model
from .trainer.flywheel import DualFlywheelTrainer
from .eval.metrics import evaluate_capabilities
from .export import export_to_gguf

__all__ = [
    "ComputeCIT",
    "TransplantFFN",
    "CapabilityRouter",
    "DualFlywheelTrainer",
    "evaluate_capabilities",
    "PARSEConfig",
    "PreservationProfile",
    "get_profile",
    "build_parse_model",
    "export_to_gguf",
]
