"""Configuration management system for uncertainty quantification experiments.

This module provides comprehensive configuration management for Bayesian
uncertainty quantification experiments, including model parameters,
training settings, and evaluation configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import yaml
import json
from pathlib import Path
import torch


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    
    # Network architecture
    input_dim: int = 2
    output_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = 'tanh'
    
    # Initialization
    weight_init: str = 'xavier_normal'
    bias_init: str = 'zeros'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")


@dataclass
class BayesianConfig:
    """Configuration for Bayesian meta-learning."""
    
    # Variational inference
    variational_family: str = 'diagonal_gaussian'  # 'diagonal_gaussian', 'full_gaussian'
    prior_type: str = 'physics_informed'  # 'standard', 'physics_informed', 'laplace'
    
    # Prior parameters
    prior_mean: float = 0.0
    prior_std: float = 1.0
    physics_prior_weight: float = 1.0
    
    # ELBO optimization
    kl_weight_schedule: str = 'linear_warmup'  # 'constant', 'linear_warmup', 'cosine'
    kl_warmup_steps: int = 1000
    kl_final_weight: float = 1.0
    
    # Posterior sampling
    num_posterior_samples: int = 100
    reparameterization: bool = True
    
    # Natural gradients
    use_natural_gradients: bool = False
    natural_gradient_lr: float = 0.01
    
    def __post_init__(self):
        """Validate Bayesian configuration."""
        valid_families = ['diagonal_gaussian', 'full_gaussian']
        if self.variational_family not in valid_families:
            raise ValueError(f"variational_family must be one of {valid_families}")
        
        valid_priors = ['standard', 'physics_informed', 'laplace']
        if self.prior_type not in valid_priors:
            raise ValueError(f"prior_type must be one of {valid_priors}")
        
        if self.prior_std <= 0:
            raise ValueError("prior_std must be positive")
        
        if self.num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be positive")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    num_models: int = 10
    parallel_training: bool = True
    different_initializations: bool = True
    different_architectures: bool = False
    bootstrap_data: bool = False
    
    def __post_init__(self):
        """Validate ensemble configuration."""
        if self.num_models <= 0:
            raise ValueError("num_models must be positive")


@dataclass
class MCDropoutConfig:
    """Configuration for Monte Carlo Dropout."""
    
    dropout_rate: float = 0.1
    num_mc_samples: int = 100
    dropout_schedule: str = 'constant'  # 'constant', 'adaptive'
    min_dropout_rate: float = 0.05
    
    def __post_init__(self):
        """Validate MC Dropout configuration."""
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive")
        if not 0 <= self.min_dropout_rate <= self.dropout_rate:
            raise ValueError("min_dropout_rate must be between 0 and dropout_rate")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Meta-training
    num_meta_iterations: int = 10000
    meta_batch_size: int = 16
    meta_lr: float = 1e-3
    meta_optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    
    # Few-shot adaptation
    adaptation_steps: int = 10
    adaptation_lr: float = 1e-2
    adaptation_optimizer: str = 'sgd'
    
    # Loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    initial_loss_weight: float = 1.0
    
    # Regularization
    weight_decay: float = 0.0
    gradient_clip_norm: Optional[float] = None
    
    # Learning rate scheduling
    lr_schedule: str = 'constant'  # 'constant', 'cosine', 'step'
    lr_decay_steps: int = 1000
    lr_decay_rate: float = 0.9
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.num_meta_iterations <= 0:
            raise ValueError("num_meta_iterations must be positive")
        if self.meta_batch_size <= 0:
            raise ValueError("meta_batch_size must be positive")
        if self.meta_lr <= 0:
            raise ValueError("meta_lr must be positive")
        if self.adaptation_steps <= 0:
            raise ValueError("adaptation_steps must be positive")
        if self.adaptation_lr <= 0:
            raise ValueError("adaptation_lr must be positive")


@dataclass
class CalibrationConfig:
    """Configuration for calibration evaluation."""
    
    # ECE computation
    num_bins: int = 10
    bin_strategy: str = 'equal_width'  # 'equal_width', 'equal_frequency'
    
    # Coverage analysis
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    # Reliability diagrams
    plot_reliability: bool = True
    save_plots: bool = True
    
    def __post_init__(self):
        """Validate calibration configuration."""
        if self.num_bins <= 0:
            raise ValueError("num_bins must be positive")
        
        for level in self.confidence_levels:
            if not 0 < level < 1:
                raise ValueError(f"Confidence level {level} must be between 0 and 1")


@dataclass
class OODConfig:
    """Configuration for out-of-distribution detection."""
    
    # OOD scenarios
    scenarios: List[str] = field(default_factory=lambda: [
        'spatial_extrapolation', 'interpolation_gap', 'parameter_shift', 'high_noise'
    ])
    
    # Detection metrics
    detection_metrics: List[str] = field(default_factory=lambda: [
        'auroc', 'aupr', 'fpr_at_95_tpr'
    ])
    
    # Data generation
    num_in_distribution: int = 500
    num_ood: int = 500
    
    def __post_init__(self):
        """Validate OOD configuration."""
        valid_scenarios = [
            'spatial_extrapolation', 'interpolation_gap', 
            'parameter_shift', 'high_noise'
        ]
        for scenario in self.scenarios:
            if scenario not in valid_scenarios:
                raise ValueError(f"Unknown OOD scenario: {scenario}")


@dataclass
class ExperimentConfig:
    """Configuration for experimental validation."""
    
    # Problem settings
    pde_types: List[str] = field(default_factory=lambda: [
        'heat', 'burgers', 'poisson', 'navier_stokes'
    ])
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    k_shot_values: List[int] = field(default_factory=lambda: [1, 5, 10, 25])
    
    # Evaluation
    num_test_tasks: int = 50
    num_runs: int = 5
    
    # Statistical analysis
    significance_level: float = 0.05
    multiple_comparison_correction: str = 'bonferroni'  # 'bonferroni', 'fdr'
    effect_size_threshold: float = 0.5  # Cohen's d
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if self.num_test_tasks <= 0:
            raise ValueError("num_test_tasks must be positive")
        if self.num_runs <= 0:
            raise ValueError("num_runs must be positive")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")


@dataclass
class UncertaintyConfig:
    """Main configuration class for uncertainty quantification experiments."""
    
    # Model configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    mc_dropout: MCDropoutConfig = field(default_factory=MCDropoutConfig)
    
    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Evaluation configurations
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Method selection
    uncertainty_method: str = 'bayesian'  # 'bayesian', 'ensemble', 'mc_dropout'
    
    # Device and precision
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    dtype: str = 'float32'  # 'float32', 'float64'
    
    # Logging and output
    log_level: str = 'INFO'
    output_dir: str = 'results'
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1000
    
    def __post_init__(self):
        """Validate main configuration and set derived parameters."""
        valid_methods = ['bayesian', 'ensemble', 'mc_dropout']
        if self.uncertainty_method not in valid_methods:
            raise ValueError(f"uncertainty_method must be one of {valid_methods}")
        
        valid_dtypes = ['float32', 'float64']
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        
        # Set device automatically if needed
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'UncertaintyConfig':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            UncertaintyConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'UncertaintyConfig':
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            UncertaintyConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UncertaintyConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            UncertaintyConfig instance
        """
        # Extract nested configurations
        network_config = NetworkConfig(**config_dict.get('network', {}))
        bayesian_config = BayesianConfig(**config_dict.get('bayesian', {}))
        ensemble_config = EnsembleConfig(**config_dict.get('ensemble', {}))
        mc_dropout_config = MCDropoutConfig(**config_dict.get('mc_dropout', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        calibration_config = CalibrationConfig(**config_dict.get('calibration', {}))
        ood_config = OODConfig(**config_dict.get('ood', {}))
        experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))
        
        # Extract main configuration parameters
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['network', 'bayesian', 'ensemble', 'mc_dropout',
                                  'training', 'calibration', 'ood', 'experiment']}
        
        return cls(
            network=network_config,
            bayesian=bayesian_config,
            ensemble=ensemble_config,
            mc_dropout=mc_dropout_config,
            training=training_config,
            calibration=calibration_config,
            ood=ood_config,
            experiment=experiment_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'network': self.network.__dict__,
            'bayesian': self.bayesian.__dict__,
            'ensemble': self.ensemble.__dict__,
            'mc_dropout': self.mc_dropout.__dict__,
            'training': self.training.__dict__,
            'calibration': self.calibration.__dict__,
            'ood': self.ood.__dict__,
            'experiment': self.experiment.__dict__,
            'uncertainty_method': self.uncertainty_method,
            'device': self.device,
            'dtype': self.dtype,
            'log_level': self.log_level,
            'output_dir': self.output_dir,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_frequency': self.checkpoint_frequency
        }
    
    def save_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
        """
        config_dict = self.to_dict()
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype from string specification.
        
        Returns:
            PyTorch dtype
        """
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64
        }
        return dtype_map[self.dtype]
    
    def get_device(self) -> torch.device:
        """Get PyTorch device.
        
        Returns:
            PyTorch device
        """
        return torch.device(self.device)
    
    def setup_reproducibility(self) -> None:
        """Set up reproducibility settings."""
        import random
        import numpy as np
        
        # Set random seeds
        random.seed(self.experiment.random_seed)
        np.random.seed(self.experiment.random_seed)
        torch.manual_seed(self.experiment.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.random_seed)
            torch.cuda.manual_seed_all(self.experiment.random_seed)
        
        # Set deterministic behavior
        if self.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_output_directory(self) -> Path:
        """Create output directory for results.
        
        Returns:
            Path to created output directory
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def create_default_config() -> UncertaintyConfig:
    """Create default configuration for uncertainty quantification.
    
    Returns:
        Default UncertaintyConfig instance
    """
    return UncertaintyConfig()


def create_bayesian_config() -> UncertaintyConfig:
    """Create configuration optimized for Bayesian meta-learning.
    
    Returns:
        UncertaintyConfig optimized for BayesianMetaPINN
    """
    config = UncertaintyConfig()
    config.uncertainty_method = 'bayesian'
    config.bayesian.variational_family = 'diagonal_gaussian'
    config.bayesian.prior_type = 'physics_informed'
    config.bayesian.num_posterior_samples = 100
    config.training.num_meta_iterations = 15000
    config.training.meta_lr = 5e-4
    return config


def create_ensemble_config() -> UncertaintyConfig:
    """Create configuration optimized for ensemble methods.
    
    Returns:
        UncertaintyConfig optimized for EnsembleMetaPINN
    """
    config = UncertaintyConfig()
    config.uncertainty_method = 'ensemble'
    config.ensemble.num_models = 10
    config.ensemble.parallel_training = True
    config.training.num_meta_iterations = 10000
    return config


def create_mc_dropout_config() -> UncertaintyConfig:
    """Create configuration optimized for MC Dropout.
    
    Returns:
        UncertaintyConfig optimized for MCDropoutMetaPINN
    """
    config = UncertaintyConfig()
    config.uncertainty_method = 'mc_dropout'
    config.mc_dropout.dropout_rate = 0.1
    config.mc_dropout.num_mc_samples = 100
    config.training.num_meta_iterations = 10000
    return config