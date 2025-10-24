"""
Bayesian Uncertainty Quantification for Meta-Learned Physics-Informed Neural Networks

This package provides the core implementation of BayesianMetaPINN and related methods
for principled uncertainty quantification in physics-informed neural networks.
"""

from .bayesian_meta_pinn import BayesianMetaPINN
from .ensemble_meta_pinn import EnsembleMetaPINN
from .mc_dropout_meta_pinn import MCDropoutMetaPINN
from .calibration_metrics import CalibrationMetrics
from .decomposition_validator import UncertaintyDecompositionValidator
from .ood_detection import OODDetectionEvaluator

__version__ = "1.0.0"
__author__ = "Brandon Yee, Wilson Collins, Ben Pellegrini, Caden Wang"

__all__ = [
    "BayesianMetaPINN",
    "EnsembleMetaPINN", 
    "MCDropoutMetaPINN",
    "CalibrationMetrics",
    "UncertaintyDecompositionValidator",
    "OODDetectionEvaluator"
]