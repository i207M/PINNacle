"""
Noise injection framework for uncertainty quantification in PINNs.

This module provides various noise injection methods for systematic evaluation
of calibration quality across different noise levels and types.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class NoiseType(Enum):
    """Enumeration of supported noise types."""
    GAUSSIAN = "gaussian"
    HETEROSCEDASTIC = "heteroscedastic"
    OUTLIER_CONTAMINATION = "outlier_contamination"


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: NoiseType
    noise_level: float
    random_seed: Optional[int] = None
    outlier_fraction: float = 0.05  # For outlier contamination
    outlier_scale: float = 10.0     # Scale factor for outliers
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.noise_level <= 0:
            raise ValueError("noise_level must be positive")
        if not 0 <= self.outlier_fraction <= 1:
            raise ValueError("outlier_fraction must be between 0 and 1")
        if self.outlier_scale <= 0:
            raise ValueError("outlier_scale must be positive")


class BaseNoiseInjector(ABC):
    """Base class for noise injection methods."""
    
    def __init__(self, config: NoiseConfig):
        self.config = config
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def inject_noise(self, clean_data: torch.Tensor, 
                    true_solution: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inject noise into clean data."""
        pass
    
    @abstractmethod
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get statistics about the injected noise."""
        pass


class GaussianNoiseInjector(BaseNoiseInjector):
    """Inject Gaussian noise with configurable standard deviation."""
    
    def __init__(self, config: NoiseConfig):
        super().__init__(config)
        if config.noise_type != NoiseType.GAUSSIAN:
            raise ValueError("Config must specify GAUSSIAN noise type")
    
    def inject_noise(self, clean_data: torch.Tensor, 
                    true_solution: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inject Gaussian noise: y_noisy = y_clean + N(0, σ²)
        
        Args:
            clean_data: Clean observations [N, output_dim]
            true_solution: Not used for Gaussian noise
            
        Returns:
            Noisy observations with same shape as clean_data
        """
        noise = torch.randn_like(clean_data) * self.config.noise_level
        return clean_data + noise
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get Gaussian noise statistics."""
        return {
            'noise_type': 'gaussian',
            'noise_level': self.config.noise_level,
            'theoretical_variance': self.config.noise_level ** 2,
            'theoretical_std': self.config.noise_level
        }


class HeteroscedasticNoiseInjector(BaseNoiseInjector):
    """Inject signal-dependent heteroscedastic noise."""
    
    def __init__(self, config: NoiseConfig):
        super().__init__(config)
        if config.noise_type != NoiseType.HETEROSCEDASTIC:
            raise ValueError("Config must specify HETEROSCEDASTIC noise type")
    
    def inject_noise(self, clean_data: torch.Tensor, 
                    true_solution: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inject heteroscedastic noise: y_noisy = y_clean + N(0, σ²(1 + |y_clean|)²)
        
        Args:
            clean_data: Clean observations [N, output_dim]
            true_solution: Not used, clean_data is used for signal dependency
            
        Returns:
            Noisy observations with signal-dependent variance
        """
        # Signal-dependent standard deviation: σ(1 + |u_true|)
        signal_dependent_std = self.config.noise_level * (1 + torch.abs(clean_data))
        noise = torch.randn_like(clean_data) * signal_dependent_std
        return clean_data + noise
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get heteroscedastic noise statistics."""
        return {
            'noise_type': 'heteroscedastic',
            'base_noise_level': self.config.noise_level,
            'signal_dependent': True
        }


class OutlierContaminationInjector(BaseNoiseInjector):
    """Inject mixture of Gaussian noise with heavy-tailed outliers."""
    
    def __init__(self, config: NoiseConfig):
        super().__init__(config)
        if config.noise_type != NoiseType.OUTLIER_CONTAMINATION:
            raise ValueError("Config must specify OUTLIER_CONTAMINATION noise type")
    
    def inject_noise(self, clean_data: torch.Tensor, 
                    true_solution: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inject outlier contamination: 95% Gaussian + 5% heavy-tailed outliers
        
        Args:
            clean_data: Clean observations [N, output_dim]
            true_solution: Not used for outlier contamination
            
        Returns:
            Noisy observations with outlier contamination
        """
        # Generate outlier mask (5% of points become outliers)
        outlier_mask = torch.rand(clean_data.shape[0]) < self.config.outlier_fraction
        
        # Base Gaussian noise for all points
        gaussian_noise = torch.randn_like(clean_data) * self.config.noise_level
        
        # Heavy-tailed outlier noise (scaled Gaussian)
        outlier_noise = torch.randn_like(clean_data) * (
            self.config.noise_level * self.config.outlier_scale
        )
        
        # Mix noise types
        noise = gaussian_noise.clone()
        if outlier_mask.any():
            noise[outlier_mask] = outlier_noise[outlier_mask]
        
        return clean_data + noise
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get outlier contamination statistics."""
        return {
            'noise_type': 'outlier_contamination',
            'base_noise_level': self.config.noise_level,
            'outlier_fraction': self.config.outlier_fraction,
            'outlier_scale': self.config.outlier_scale,
            'effective_outlier_std': self.config.noise_level * self.config.outlier_scale
        }


class NoiseInjectionFramework:
    """Main framework for noise injection with multiple noise types."""
    
    # Supported noise levels as specified in requirements
    SUPPORTED_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]
    
    def __init__(self):
        self._injectors = {}
    
    def create_injector(self, noise_type: NoiseType, noise_level: float,
                       random_seed: Optional[int] = None,
                       **kwargs) -> BaseNoiseInjector:
        """
        Create a noise injector for specified type and level.
        
        Args:
            noise_type: Type of noise to inject
            noise_level: Standard deviation for noise (σ)
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific noise types
            
        Returns:
            Configured noise injector
        """
        if noise_level not in self.SUPPORTED_NOISE_LEVELS:
            raise ValueError(f"Noise level {noise_level} not in supported levels: "
                           f"{self.SUPPORTED_NOISE_LEVELS}")
        
        config = NoiseConfig(
            noise_type=noise_type,
            noise_level=noise_level,
            random_seed=random_seed,
            **kwargs
        )
        
        if noise_type == NoiseType.GAUSSIAN:
            return GaussianNoiseInjector(config)
        elif noise_type == NoiseType.HETEROSCEDASTIC:
            return HeteroscedasticNoiseInjector(config)
        elif noise_type == NoiseType.OUTLIER_CONTAMINATION:
            return OutlierContaminationInjector(config)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
    
    def inject_systematic_noise(self, clean_data: torch.Tensor,
                               noise_configs: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Inject noise systematically across all supported levels and types.
        
        Args:
            clean_data: Clean observations to add noise to
            noise_configs: Optional configurations for specific noise types
            
        Returns:
            Dictionary mapping noise description to noisy data
        """
        if noise_configs is None:
            noise_configs = {}
        
        results = {}
        
        for noise_level in self.SUPPORTED_NOISE_LEVELS:
            for noise_type in NoiseType:
                # Get specific config for this noise type
                type_config = noise_configs.get(noise_type.value, {})
                
                # Create injector
                injector = self.create_injector(
                    noise_type=noise_type,
                    noise_level=noise_level,
                    **type_config
                )
                
                # Inject noise
                noisy_data = injector.inject_noise(clean_data)
                
                # Store result with descriptive key
                key = f"{noise_type.value}_sigma_{noise_level}"
                results[key] = noisy_data
        
        return results
    
    def validate_noise_properties(self, clean_data: torch.Tensor,
                                 noisy_data: torch.Tensor,
                                 injector: BaseNoiseInjector) -> Dict[str, float]:
        """
        Validate that injected noise has expected statistical properties.
        
        Args:
            clean_data: Original clean data
            noisy_data: Data with injected noise
            injector: The noise injector used
            
        Returns:
            Validation metrics
        """
        noise = noisy_data - clean_data
        
        empirical_stats = {
            'empirical_mean': noise.mean().item(),
            'empirical_std': noise.std().item(),
            'empirical_variance': noise.var().item(),
            'data_points': noise.numel()
        }
        
        theoretical_stats = injector.get_noise_statistics()
        
        # Combine empirical and theoretical statistics
        validation_results = {**empirical_stats, **theoretical_stats}
        
        # Add validation checks
        if injector.config.noise_type == NoiseType.GAUSSIAN:
            # For Gaussian noise, check if empirical std matches theoretical
            std_error = abs(empirical_stats['empirical_std'] - 
                          theoretical_stats['theoretical_std'])
            validation_results['std_error'] = std_error
            validation_results['std_valid'] = std_error < 0.1 * theoretical_stats['theoretical_std']
        
        return validation_results


# Convenience functions for common use cases
def inject_gaussian_noise(data: torch.Tensor, sigma: float, 
                         seed: Optional[int] = None) -> torch.Tensor:
    """Convenience function for Gaussian noise injection."""
    framework = NoiseInjectionFramework()
    injector = framework.create_injector(NoiseType.GAUSSIAN, sigma, seed)
    return injector.inject_noise(data)


def inject_heteroscedastic_noise(data: torch.Tensor, base_sigma: float,
                                seed: Optional[int] = None) -> torch.Tensor:
    """Convenience function for heteroscedastic noise injection."""
    framework = NoiseInjectionFramework()
    injector = framework.create_injector(NoiseType.HETEROSCEDASTIC, base_sigma, seed)
    return injector.inject_noise(data)


def inject_outlier_noise(data: torch.Tensor, sigma: float,
                        outlier_fraction: float = 0.05,
                        outlier_scale: float = 10.0,
                        seed: Optional[int] = None) -> torch.Tensor:
    """Convenience function for outlier contamination."""
    framework = NoiseInjectionFramework()
    injector = framework.create_injector(
        NoiseType.OUTLIER_CONTAMINATION, sigma, seed,
        outlier_fraction=outlier_fraction,
        outlier_scale=outlier_scale
    )
    return injector.inject_noise(data)