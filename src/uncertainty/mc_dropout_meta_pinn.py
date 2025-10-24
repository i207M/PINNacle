"""MCDropoutMetaPINN implementation for uncertainty quantification.

This module implements Monte Carlo Dropout for approximate Bayesian inference
in meta-learned Physics-Informed Neural Networks, providing uncertainty
quantification through test-time dropout sampling.
"""

import copy
import math
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np

from .base import (
    UncertaintyMetaLearner, 
    UncertaintyPrediction, 
    TaskDistribution, 
    Task,
    ConvergenceError,
    DecompositionError
)
from ..meta_learning.meta_pinn import MetaPINN
from ..meta_learning.config import MetaPINNConfig


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer that remains active during inference."""
    
    def __init__(self, p: float = 0.1):
        """Initialize MC Dropout layer.
        
        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode."""
        return torch.nn.functional.dropout(x, p=self.p, training=True)


class MCDropoutNetwork(nn.Module):
    """Neural network with MC Dropout layers inserted after activations."""
    
    def __init__(self, base_network: nn.Module, dropout_rate: float = 0.1):
        """Initialize MC Dropout network.
        
        Args:
            base_network: Base neural network
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = self._add_dropout_layers(base_network)
        
        # Copy other attributes from base network
        if hasattr(base_network, '_input_transform'):
            self._input_transform = base_network._input_transform
        if hasattr(base_network, '_output_transform'):
            self._output_transform = base_network._output_transform
    
    def _add_dropout_layers(self, base_network: nn.Module) -> nn.ModuleList:
        """Add MC Dropout layers after each activation function."""
        layers = nn.ModuleList()
        
        # Handle different network architectures
        if hasattr(base_network, 'linears'):
            # PINNacle FNN structure
            for i, linear_layer in enumerate(base_network.linears):
                layers.append(linear_layer)
                
                # Add activation and dropout for all layers except the last
                if i < len(base_network.linears) - 1:
                    # Add activation (assuming tanh, but could be configurable)
                    layers.append(nn.Tanh())
                    # Add MC Dropout
                    layers.append(MCDropout(self.dropout_rate))
        
        elif isinstance(base_network, nn.Sequential):
            # Sequential network structure
            for layer in base_network:
                layers.append(layer)
                # Add dropout after activation functions
                if isinstance(layer, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.GELU)):
                    layers.append(MCDropout(self.dropout_rate))
        
        else:
            # Fallback: treat as single module
            layers.append(base_network)
            layers.append(MCDropout(self.dropout_rate))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network with MC Dropout."""
        current_x = x
        
        # Apply input transform if exists
        if hasattr(self, '_input_transform') and self._input_transform is not None:
            current_x = self._input_transform(current_x)
        
        # Forward through layers
        for layer in self.layers:
            current_x = layer(current_x)
        
        # Apply output transform if exists
        if hasattr(self, '_output_transform') and self._output_transform is not None:
            current_x = self._output_transform(x, current_x)
        
        return current_x


class MCDropoutMetaPINN(UncertaintyMetaLearner):
    """Monte Carlo Dropout Meta-Learning Physics-Informed Neural Network.
    
    This class implements uncertainty quantification using Monte Carlo Dropout,
    where dropout layers remain active during test time to provide approximate
    Bayesian inference through sampling different network configurations.
    
    Args:
        base_model_config: Configuration for base MetaPINN model
        dropout_rate: Dropout probability (default: 0.1)
        num_mc_samples: Number of MC samples for uncertainty estimation
        device: Device for computation
        heuristic_split_ratio: Ratio for splitting total uncertainty into epistemic/aleatoric
    """
    
    def __init__(self,
                 base_model_config: MetaPINNConfig,
                 dropout_rate: float = 0.1,
                 num_mc_samples: int = 100,
                 device: Union[str, torch.device] = 'cpu',
                 heuristic_split_ratio: float = 0.8):
        
        self.base_model_config = base_model_config
        self.dropout_rate = dropout_rate
        self.num_mc_samples = num_mc_samples
        self.device = torch.device(device)
        self.heuristic_split_ratio = heuristic_split_ratio  # Fraction treated as epistemic
        
        # Initialize base MetaPINN model
        self.base_model = MetaPINN(base_model_config)
        
        # Replace network with MC Dropout version
        self.mc_dropout_network = MCDropoutNetwork(
            self.base_model.network, 
            dropout_rate=dropout_rate
        )
        self.base_model.network = self.mc_dropout_network
        
        # Adaptation state
        self._is_adapted = False
        self.adapted_model = None
        self.adaptation_history = []
        
        # Move to device
        self.to(self.device)
    
    def to(self, device: Union[str, torch.device]) -> 'MCDropoutMetaPINN':
        """Move model to device."""
        self.device = torch.device(device)
        self.base_model.network = self.base_model.network.to(device)
        self.base_model.device = self.device
        
        if self.adapted_model is not None:
            self.adapted_model.network = self.adapted_model.network.to(device)
            self.adapted_model.device = self.device
        
        return self
    
    def meta_train(self, task_distribution: TaskDistribution, 
                   num_iterations: int) -> Dict[str, float]:
        """Meta-train the MC Dropout model on task distribution.
        
        Args:
            task_distribution: Distribution of training tasks
            num_iterations: Number of meta-training iterations
            
        Returns:
            Dictionary containing training metrics
        """
        print(f"Meta-training MC Dropout MetaPINN for {num_iterations} iterations...")
        
        # Ensure dropout is enabled during training
        self.base_model.network.train()
        
        # Use base MetaPINN training with MC Dropout network
        result = self.base_model.meta_train(task_distribution, num_iterations)
        
        # Add MC Dropout specific metrics
        result['dropout_rate'] = self.dropout_rate
        result['num_mc_samples'] = self.num_mc_samples
        result['model_type'] = 'mc_dropout'
        
        return result
    
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor,
              num_steps: int = 10) -> 'MCDropoutMetaPINN':
        """Adapt to new task using support data with MC Dropout.
        
        Args:
            support_data: Support set inputs [k_shot, input_dim]
            support_targets: Support set targets [k_shot, output_dim]
            num_steps: Number of adaptation steps
            
        Returns:
            Self (for method chaining)
        """
        # Validate inputs
        self.validate_inputs(support_data)
        self.validate_targets(support_targets)
        
        # Move to device
        support_data = support_data.to(self.device)
        support_targets = support_targets.to(self.device)
        
        # Create adapted model
        self.adapted_model = copy.deepcopy(self.base_model)
        
        # Create task data structure expected by MetaPINN
        from ..meta_learning.task import TaskData
        task_data = TaskData(
            inputs=support_data,
            outputs=support_targets,
            collocation_points=support_data,  # Use same points for physics loss
            boundary_data=None,
            initial_data=None
        )
        
        # Create a dummy task for adaptation
        from ..meta_learning.task import Task
        dummy_task = Task(
            problem_type='adaptation',
            parameters={},
            support_data=task_data,
            query_data=task_data,  # Same as support for adaptation
            metadata={'dropout_rate': self.dropout_rate}
        )
        
        # Adaptation history
        self.adaptation_history = []
        
        # Adapt the model with dropout enabled
        start_time = time.time()
        try:
            # Ensure dropout is enabled during adaptation
            self.adapted_model.network.train()
            
            # Use MetaPINN's adapt method
            self.adapted_model = self.adapted_model.adapt(
                task_data, dummy_task, 
                k_shots=len(support_data), 
                adaptation_steps=num_steps
            )
            
            adaptation_time = time.time() - start_time
            
            self.adaptation_history.append({
                'adaptation_time': adaptation_time,
                'adaptation_steps': num_steps,
                'dropout_rate': self.dropout_rate,
                'success': True
            })
            
            print(f"MC Dropout adaptation completed in {adaptation_time:.2f}s")
            
        except Exception as e:
            adaptation_time = time.time() - start_time
            self.adaptation_history.append({
                'adaptation_time': adaptation_time,
                'adaptation_steps': num_steps,
                'dropout_rate': self.dropout_rate,
                'success': False,
                'error': str(e)
            })
            
            print(f"MC Dropout adaptation failed: {e}")
            raise ConvergenceError(f"MC Dropout adaptation failed: {e}")
        
        self._is_adapted = True
        return self
    
    def predict_with_uncertainty(self, query_points: torch.Tensor,
                                num_samples: int = None) -> UncertaintyPrediction:
        """Predict with uncertainty quantification using MC Dropout sampling.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            num_samples: Number of MC samples (uses default if None)
            
        Returns:
            UncertaintyPrediction with MC Dropout-based uncertainty
        """
        if num_samples is None:
            num_samples = self.num_mc_samples
        
        # Validate inputs
        self.validate_inputs(query_points)
        query_points = query_points.to(self.device)
        
        # Use adapted model if available, otherwise use base model
        model_to_use = self.adapted_model if self._is_adapted else self.base_model
        
        # Enable dropout for inference (MC Dropout)
        model_to_use.network.train()  # This enables dropout
        
        # Collect MC samples
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Each forward pass uses different dropout mask
                sample_prediction = model_to_use.forward(query_points)
                samples.append(sample_prediction)
        
        # Stack samples: [num_samples, batch_size, output_dim]
        samples = torch.stack(samples)
        
        # Compute statistics
        mean_prediction = samples.mean(dim=0)
        
        # Total uncertainty: variance across MC samples
        if num_samples > 1:
            total_variance = samples.var(dim=0)
        else:
            # Single sample case
            total_variance = torch.zeros_like(mean_prediction)
        
        # Heuristic split of uncertainty into epistemic and aleatoric
        # This is a limitation of MC Dropout - it cannot cleanly separate the two
        epistemic_uncertainty = self.heuristic_split_ratio * total_variance
        aleatoric_uncertainty = (1.0 - self.heuristic_split_ratio) * total_variance
        
        # Validate uncertainty decomposition
        self._validate_mc_dropout_uncertainty(epistemic_uncertainty, aleatoric_uncertainty)
        
        return UncertaintyPrediction(
            mean=mean_prediction,
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            samples=samples
        )
    
    def get_epistemic_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract epistemic (model) uncertainty from MC Dropout variance.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Epistemic uncertainty [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points, num_samples=20)
        return prediction.epistemic
    
    def get_aleatoric_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract aleatoric (data) uncertainty from MC Dropout variance.
        
        Note: MC Dropout cannot reliably separate epistemic and aleatoric uncertainty.
        This uses a heuristic split of the total variance.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Aleatoric uncertainty [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points, num_samples=20)
        return prediction.aleatoric
    
    def _validate_mc_dropout_uncertainty(self, epistemic: torch.Tensor, 
                                       aleatoric: torch.Tensor) -> None:
        """Validate MC Dropout uncertainty properties.
        
        Args:
            epistemic: Epistemic uncertainty tensor
            aleatoric: Aleatoric uncertainty tensor
            
        Raises:
            DecompositionError: If uncertainty is invalid
        """
        # Check for non-negative uncertainties
        if torch.any(epistemic < 0) or torch.any(aleatoric < 0):
            raise DecompositionError("Uncertainties must be non-negative")
        
        # Check for NaN or Inf values
        if torch.any(torch.isnan(epistemic)) or torch.any(torch.isnan(aleatoric)):
            raise DecompositionError("Uncertainties contain NaN values")
        
        if torch.any(torch.isinf(epistemic)) or torch.any(torch.isinf(aleatoric)):
            raise DecompositionError("Uncertainties contain Inf values")
        
        # Check reasonable magnitude (heuristic)
        if torch.any(epistemic > 1000) or torch.any(aleatoric > 1000):
            raise DecompositionError("Uncertainties are unreasonably large")
    
    def get_dropout_statistics(self, query_points: torch.Tensor, 
                              num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Compute detailed statistics from MC Dropout sampling.
        
        Args:
            query_points: Query inputs for statistics computation
            num_samples: Number of MC samples
            
        Returns:
            Dictionary with detailed MC Dropout statistics
        """
        if num_samples is None:
            num_samples = self.num_mc_samples
        
        # Validate inputs
        self.validate_inputs(query_points)
        query_points = query_points.to(self.device)
        
        # Use adapted model if available
        model_to_use = self.adapted_model if self._is_adapted else self.base_model
        
        # Enable dropout for inference
        model_to_use.network.train()
        
        # Collect samples and intermediate statistics
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                sample_prediction = model_to_use.forward(query_points)
                samples.append(sample_prediction)
        
        samples = torch.stack(samples)  # [num_samples, batch_size, output_dim]
        
        # Compute comprehensive statistics
        mean_pred = samples.mean(dim=0)
        var_pred = samples.var(dim=0)
        std_pred = samples.std(dim=0)
        
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = {}
        for p in percentiles:
            percentile_values[f'percentile_{p}'] = torch.quantile(samples, p/100.0, dim=0)
        
        # Prediction intervals
        prediction_intervals = {
            '90_percent': (percentile_values['percentile_5'], percentile_values['percentile_95']),
            '50_percent': (percentile_values['percentile_25'], percentile_values['percentile_75'])
        }
        
        # Coefficient of variation
        cv = std_pred / (torch.abs(mean_pred) + 1e-8)
        
        return {
            'mean': mean_pred,
            'variance': var_pred,
            'std': std_pred,
            'coefficient_of_variation': cv,
            'samples': samples,
            **percentile_values,
            'prediction_intervals': prediction_intervals,
            'num_samples': num_samples,
            'dropout_rate': self.dropout_rate
        }
    
    def calibrate_dropout_rate(self, validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
                              dropout_rates: List[float] = None) -> float:
        """Calibrate dropout rate using validation data.
        
        Args:
            validation_data: List of (inputs, targets) tuples for validation
            dropout_rates: List of dropout rates to test
            
        Returns:
            Optimal dropout rate based on validation performance
        """
        if dropout_rates is None:
            dropout_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        print(f"Calibrating dropout rate using {len(validation_data)} validation tasks...")
        
        best_dropout_rate = self.dropout_rate
        best_score = float('inf')
        
        original_dropout_rate = self.dropout_rate
        
        for dropout_rate in dropout_rates:
            print(f"Testing dropout rate: {dropout_rate}")
            
            # Update dropout rate
            self.dropout_rate = dropout_rate
            self._update_network_dropout_rate(dropout_rate)
            
            # Evaluate on validation data
            total_score = 0.0
            
            for inputs, targets in validation_data:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions with uncertainty
                predictions = self.predict_with_uncertainty(inputs, num_samples=50)
                
                # Compute calibration score (negative log-likelihood)
                mse = torch.mean((predictions.mean - targets) ** 2)
                uncertainty_penalty = torch.mean(predictions.total)
                
                # Combined score: accuracy + uncertainty calibration
                score = mse + 0.1 * uncertainty_penalty
                total_score += score.item()
            
            avg_score = total_score / len(validation_data)
            print(f"Dropout rate {dropout_rate}: avg_score = {avg_score:.6f}")
            
            if avg_score < best_score:
                best_score = avg_score
                best_dropout_rate = dropout_rate
        
        # Restore best dropout rate
        self.dropout_rate = best_dropout_rate
        self._update_network_dropout_rate(best_dropout_rate)
        
        print(f"Best dropout rate: {best_dropout_rate} (score: {best_score:.6f})")
        
        return best_dropout_rate
    
    def _update_network_dropout_rate(self, new_dropout_rate: float) -> None:
        """Update dropout rate in the network."""
        for layer in self.base_model.network.layers:
            if isinstance(layer, MCDropout):
                layer.p = new_dropout_rate
        
        if self.adapted_model is not None:
            for layer in self.adapted_model.network.layers:
                if isinstance(layer, MCDropout):
                    layer.p = new_dropout_rate
    
    @property
    def is_adapted(self) -> bool:
        """Check if model has been adapted to a task."""
        return self._is_adapted
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state for new task."""
        self._is_adapted = False
        self.adapted_model = None
        self.adaptation_history = []
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of MC Dropout adaptation process.
        
        Returns:
            Dictionary with adaptation statistics
        """
        if not self.adaptation_history:
            return {'status': 'No adaptation performed'}
        
        latest_adaptation = self.adaptation_history[-1]
        
        summary = {
            'adapted': self._is_adapted,
            'dropout_rate': self.dropout_rate,
            'num_mc_samples': self.num_mc_samples,
            'heuristic_split_ratio': self.heuristic_split_ratio,
            'latest_adaptation': latest_adaptation,
            'adaptation_history': self.adaptation_history
        }
        
        if latest_adaptation['success']:
            summary.update({
                'adaptation_time': latest_adaptation['adaptation_time'],
                'adaptation_steps': latest_adaptation['adaptation_steps']
            })
        else:
            summary['error'] = latest_adaptation.get('error', 'Unknown error')
        
        return summary
    
    def state_dict(self) -> Dict:
        """Get MC Dropout model state dictionary."""
        return {
            'base_model_config': self.base_model_config,
            'dropout_rate': self.dropout_rate,
            'num_mc_samples': self.num_mc_samples,
            'heuristic_split_ratio': self.heuristic_split_ratio,
            'base_model_state': self.base_model.network.state_dict(),
            'adaptation_history': self.adaptation_history,
            'is_adapted': self._is_adapted
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load MC Dropout model state dictionary."""
        self.base_model_config = state_dict['base_model_config']
        self.dropout_rate = state_dict['dropout_rate']
        self.num_mc_samples = state_dict['num_mc_samples']
        self.heuristic_split_ratio = state_dict['heuristic_split_ratio']
        
        # Load base model state
        self.base_model.network.load_state_dict(state_dict['base_model_state'])
        
        self.adaptation_history = state_dict.get('adaptation_history', [])
        self._is_adapted = state_dict.get('is_adapted', False)


def create_mc_dropout_meta_pinn(config: Dict) -> MCDropoutMetaPINN:
    """Factory function to create MCDropoutMetaPINN with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MCDropoutMetaPINN instance
    """
    # Extract MC Dropout-specific config
    mc_dropout_config = {
        'dropout_rate': config.get('dropout_rate', 0.1),
        'num_mc_samples': config.get('num_mc_samples', 100),
        'device': config.get('device', 'cpu'),
        'heuristic_split_ratio': config.get('heuristic_split_ratio', 0.8)
    }
    
    # Create base model config
    base_model_config = MetaPINNConfig(
        layers=config.get('layers', [2, 64, 64, 1]),
        activation=config.get('activation', 'tanh'),
        meta_lr=config.get('meta_lr', 0.001),
        adapt_lr=config.get('adapt_lr', 0.01),
        adaptation_steps=config.get('adaptation_steps', 10),
        meta_batch_size=config.get('meta_batch_size', 4),
        device=config.get('device', 'cpu')
    )
    
    return MCDropoutMetaPINN(
        base_model_config=base_model_config,
        **mc_dropout_config
    )