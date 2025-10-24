"""BayesianMetaPINN implementation with uncertainty decomposition.

This module implements the core BayesianMetaPINN class that combines
variational Bayesian inference with meta-learning for Physics-Informed
Neural Networks, providing epistemic and aleatoric uncertainty decomposition.
"""

import math
from typing import Dict, List, Optional, Tuple, Callable, Union
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
from .variational_layers import (
    VariationalLinear, 
    create_variational_network,
    compute_network_kl_divergence
)
from .amortized_inference import (
    AmortizedBayesianNetwork,
    EfficientPosteriorSampler,
    create_amortized_network,
    optimize_memory_usage
)
from .physics_priors import PhysicsInformedPrior, create_physics_informed_prior
from .elbo_optimization import ELBOOptimizer, create_elbo_optimizer


class BayesianMetaPINN(UncertaintyMetaLearner):
    """Bayesian Meta-Learning Physics-Informed Neural Network.
    
    This class implements variational Bayesian inference for meta-learned PINNs,
    providing calibrated uncertainty quantification through epistemic and 
    aleatoric uncertainty decomposition.
    
    Args:
        network_architecture: Dictionary specifying network architecture
        pde_type: Type of PDE being solved
        input_dim: Input dimension (spatial + temporal)
        output_dim: Output dimension (solution components)
        physics_informed_prior: Whether to use physics-informed priors
        variational_family: Type of variational family ('diagonal_gaussian')
        meta_lr: Meta-learning rate
        adaptation_lr: Adaptation learning rate
        adaptation_steps: Number of adaptation steps
        num_posterior_samples: Number of posterior samples for uncertainty
        device: Device for computation
    """
    
    def __init__(self,
                 network_architecture: Dict,
                 pde_type: str = 'heat',
                 input_dim: int = 2,
                 output_dim: int = 1,
                 physics_informed_prior: bool = True,
                 variational_family: str = 'diagonal_gaussian',
                 meta_lr: float = 0.001,
                 adaptation_lr: float = 0.01,
                 adaptation_steps: int = 10,
                 num_posterior_samples: int = 100,
                 device: Union[str, torch.device] = 'cpu'):
        
        self.network_architecture = network_architecture
        self.pde_type = pde_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_informed_prior = physics_informed_prior
        self.variational_family = variational_family
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.num_posterior_samples = num_posterior_samples
        self.device = torch.device(device)
        
        # Build network and prior
        self.network = self._build_network()
        self.amortized_network = self._build_amortized_network()
        self.prior = self._build_prior()
        
        # Efficient posterior sampler
        self.posterior_sampler = EfficientPosteriorSampler(
            self.amortized_network, 
            cache_size=1000, 
            batch_size=min(100, num_posterior_samples)
        )
        
        # Initialize ELBO optimizer
        self.elbo_optimizer = self._build_elbo_optimizer()
        
        # Aleatoric uncertainty parameters (learned noise)
        self.aleatoric_log_var = nn.Parameter(torch.log(torch.tensor(0.01)))
        
        # Adaptation state
        self._is_adapted = False
        self.adapted_network = None
        self.adaptation_history = []
        
        # Move to device
        self.to(self.device)
    
    def _build_network(self) -> nn.Module:
        """Build variational neural network."""
        layer_dims = self.network_architecture.get('dims', [self.input_dim, 64, 64, self.output_dim])
        activation = self.network_architecture.get('activation', 'tanh')
        
        # Create variational network
        network = create_variational_network(
            layer_dims=layer_dims,
            activation=activation,
            prior_mean=0.0,
            prior_std=1.0
        )
        
        return network
    
    def _build_amortized_network(self) -> AmortizedBayesianNetwork:
        """Build amortized Bayesian network for efficient inference."""
        layer_dims = self.network_architecture.get('dims', [self.input_dim, 64, 64, self.output_dim])
        activation = self.network_architecture.get('activation', 'tanh')
        
        # Create amortized network
        amortized_network = create_amortized_network(
            layer_dims=layer_dims,
            activation=activation,
            prior_mean=0.0,
            prior_std=1.0,
            num_samples=self.num_posterior_samples
        )
        
        return amortized_network
    
    def _build_prior(self) -> PhysicsInformedPrior:
        """Build physics-informed prior."""
        if self.physics_informed_prior:
            prior = create_physics_informed_prior(
                pde_type=self.pde_type,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                base_mean=0.0,
                base_std=1.0,
                physics_weight=1.0
            )
        else:
            # Standard Gaussian prior
            prior = PhysicsInformedPrior(
                pde_type=self.pde_type,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                base_mean=0.0,
                base_std=1.0,
                physics_weight=0.0  # No physics weighting
            )
        
        return prior
    
    def _build_elbo_optimizer(self) -> ELBOOptimizer:
        """Build ELBO optimizer."""
        config = {
            'optimizer_type': 'adam',
            'lr': self.meta_lr,
            'kl_schedule': {
                'schedule_type': 'linear',
                'start_weight': 0.0,
                'end_weight': 1.0,
                'num_steps': 1000,
                'warmup_steps': 100
            },
            'physics_weight': 1.0,
            'convergence_patience': 100,
            'min_improvement': 1e-6
        }
        
        return create_elbo_optimizer(self.network, self.prior, config)
    
    @property
    def aleatoric_std(self) -> torch.Tensor:
        """Get aleatoric uncertainty standard deviation."""
        return torch.exp(0.5 * self.aleatoric_log_var)
    
    def to(self, device: Union[str, torch.device]) -> 'BayesianMetaPINN':
        """Move model to device."""
        self.device = torch.device(device)
        self.network = self.network.to(device)
        self.aleatoric_log_var = self.aleatoric_log_var.to(device)
        
        if self.adapted_network is not None:
            self.adapted_network = self.adapted_network.to(device)
        
        return self
    
    def meta_train(self, task_distribution: TaskDistribution, 
                   num_iterations: int) -> Dict[str, float]:
        """Meta-train the model on task distribution.
        
        Args:
            task_distribution: Distribution of training tasks
            num_iterations: Number of meta-training iterations
            
        Returns:
            Dictionary containing training metrics
        """
        self.network.train()
        training_metrics = {
            'elbo_history': [],
            'data_likelihood_history': [],
            'physics_likelihood_history': [],
            'kl_divergence_history': [],
            'meta_loss_history': []
        }
        
        for iteration in range(num_iterations):
            # Sample batch of tasks
            task_batch = task_distribution.sample_batch(batch_size=4)
            
            meta_loss = 0.0
            iteration_metrics = {
                'elbo': 0.0,
                'data_likelihood': 0.0,
                'physics_likelihood': 0.0,
                'kl_divergence': 0.0
            }
            
            for task in task_batch:
                # Sample support and query sets
                support_data, support_targets = task.sample_support(k_shot=5)
                query_data, query_targets = task.sample_query(num_query=20)
                
                # Move to device
                support_data = support_data.to(self.device)
                support_targets = support_targets.to(self.device)
                query_data = query_data.to(self.device)
                query_targets = query_targets.to(self.device)
                
                # Fast adaptation on support set
                adapted_network = self._fast_adapt(task, support_data, support_targets)
                
                # Evaluate on query set
                task_loss, task_metrics = self._evaluate_task(
                    adapted_network, task, query_data, query_targets
                )
                
                meta_loss += task_loss
                for key in iteration_metrics:
                    iteration_metrics[key] += task_metrics[key]
            
            # Average over task batch
            meta_loss /= len(task_batch)
            for key in iteration_metrics:
                iteration_metrics[key] /= len(task_batch)
            
            # Meta-update
            meta_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update meta-parameters
            self.elbo_optimizer.optimizer.step()
            self.elbo_optimizer.optimizer.zero_grad()
            
            # Record metrics
            training_metrics['meta_loss_history'].append(meta_loss.item())
            for key in iteration_metrics:
                training_metrics[f'{key}_history'].append(iteration_metrics[key])
            
            # Print progress
            if (iteration + 1) % 100 == 0:
                print(f"Meta-iteration {iteration + 1}/{num_iterations}, "
                      f"Meta-loss: {meta_loss.item():.6f}, "
                      f"ELBO: {iteration_metrics['elbo']:.6f}")
        
        # Compute final metrics
        final_metrics = {
            'final_meta_loss': training_metrics['meta_loss_history'][-1],
            'final_elbo': training_metrics['elbo_history'][-1],
            'avg_meta_loss': np.mean(training_metrics['meta_loss_history'][-100:]),
            'avg_elbo': np.mean(training_metrics['elbo_history'][-100:]),
            'training_history': training_metrics
        }
        
        return final_metrics
    
    def _fast_adapt(self, task: Task, support_data: torch.Tensor, 
                   support_targets: torch.Tensor) -> nn.Module:
        """Perform fast adaptation on support set.
        
        Args:
            task: Task instance
            support_data: Support set inputs
            support_targets: Support set targets
            
        Returns:
            Adapted network
        """
        # Create a copy of the network for adaptation
        adapted_network = self._copy_network()
        
        # Create optimizer for adaptation
        adaptation_optimizer = torch.optim.Adam(adapted_network.parameters(), lr=self.adaptation_lr)
        
        # Adaptation loop
        for step in range(self.adaptation_steps):
            adaptation_optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_network(support_data)
            
            # Compute PDE residuals
            pde_residuals = task.get_pde_residual(support_data, predictions)
            
            # Compute ELBO loss
            kl_divergence = compute_network_kl_divergence(adapted_network, self.prior)
            
            loss_dict = self.elbo_optimizer.elbo_loss(
                predictions=predictions,
                targets=support_targets,
                pde_residuals=pde_residuals,
                kl_divergence=kl_divergence,
                num_data_points=len(support_data)
            )
            
            # Backward pass
            loss_dict['loss'].backward()
            adaptation_optimizer.step()
        
        return adapted_network
    
    def _copy_network(self) -> nn.Module:
        """Create a copy of the network for adaptation."""
        # Create new network with same architecture
        copied_network = create_variational_network(
            layer_dims=self.network_architecture.get('dims', [self.input_dim, 64, 64, self.output_dim]),
            activation=self.network_architecture.get('activation', 'tanh'),
            prior_mean=0.0,
            prior_std=1.0
        ).to(self.device)
        
        # Copy parameters
        copied_network.load_state_dict(self.network.state_dict())
        
        return copied_network
    
    def _evaluate_task(self, network: nn.Module, task: Task, 
                      query_data: torch.Tensor, query_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Evaluate adapted network on query set.
        
        Args:
            network: Adapted network
            task: Task instance
            query_data: Query set inputs
            query_targets: Query set targets
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Forward pass
        predictions = network(query_data)
        
        # Compute PDE residuals
        pde_residuals = task.get_pde_residual(query_data, predictions)
        
        # Compute KL divergence
        kl_divergence = compute_network_kl_divergence(network, self.prior)
        
        # Compute ELBO loss
        loss_dict = self.elbo_optimizer.elbo_loss(
            predictions=predictions,
            targets=query_targets,
            pde_residuals=pde_residuals,
            kl_divergence=kl_divergence,
            num_data_points=len(query_data)
        )
        
        metrics = {
            'elbo': loss_dict['elbo'].item(),
            'data_likelihood': loss_dict['data_likelihood'].item(),
            'physics_likelihood': loss_dict['physics_likelihood'].item(),
            'kl_divergence': loss_dict['kl_divergence'].item()
        }
        
        return loss_dict['loss'], metrics
    
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor,
              num_steps: int = None) -> 'BayesianMetaPINN':
        """Adapt to new task using support data.
        
        Args:
            support_data: Support set inputs [k_shot, input_dim]
            support_targets: Support set targets [k_shot, output_dim]
            num_steps: Number of adaptation steps (uses default if None)
            
        Returns:
            Self (for method chaining)
        """
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Validate inputs
        self.validate_inputs(support_data)
        self.validate_targets(support_targets)
        
        # Move to device
        support_data = support_data.to(self.device)
        support_targets = support_targets.to(self.device)
        
        # Create adapted network
        self.adapted_network = self._copy_network()
        
        # Create optimizer for adaptation
        adaptation_optimizer = torch.optim.Adam(
            list(self.adapted_network.parameters()) + [self.aleatoric_log_var], 
            lr=self.adaptation_lr
        )
        
        # Adaptation history
        self.adaptation_history = []
        
        # Adaptation loop
        for step in range(num_steps):
            adaptation_optimizer.zero_grad()
            
            # Forward pass
            predictions = self.adapted_network(support_data)
            
            # Data likelihood with learned aleatoric uncertainty
            data_loss = torch.mean((predictions - support_targets) ** 2) / (2 * torch.exp(self.aleatoric_log_var))
            data_loss += 0.5 * self.aleatoric_log_var  # Regularization term
            
            # KL divergence
            kl_loss = compute_network_kl_divergence(self.adapted_network, self.prior)
            kl_loss /= len(support_data)  # Scale by number of data points
            
            # Total loss
            total_loss = data_loss + kl_loss
            
            # Backward pass
            total_loss.backward()
            adaptation_optimizer.step()
            
            # Record adaptation history
            self.adaptation_history.append({
                'step': step,
                'total_loss': total_loss.item(),
                'data_loss': data_loss.item(),
                'kl_loss': kl_loss.item(),
                'aleatoric_std': self.aleatoric_std.item()
            })
        
        self._is_adapted = True
        return self
    
    def predict_with_uncertainty(self, query_points: torch.Tensor,
                                num_samples: int = None, use_amortized: bool = True) -> UncertaintyPrediction:
        """Predict with uncertainty quantification.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            num_samples: Number of posterior samples (uses default if None)
            use_amortized: Whether to use amortized inference for efficiency
            
        Returns:
            UncertaintyPrediction with epistemic and aleatoric uncertainty
        """
        if num_samples is None:
            num_samples = self.num_posterior_samples
        
        # Validate inputs
        self.validate_inputs(query_points)
        query_points = query_points.to(self.device)
        
        if use_amortized and hasattr(self, 'amortized_network'):
            return self._predict_with_amortized_inference(query_points, num_samples)
        else:
            return self._predict_with_standard_inference(query_points, num_samples)
    
    def _predict_with_amortized_inference(self, query_points: torch.Tensor,
                                        num_samples: int) -> UncertaintyPrediction:
        """Predict using amortized inference for efficiency."""
        # Use amortized network for efficient sampling
        network = self.amortized_network
        network.eval()
        
        with torch.no_grad():
            # Single forward pass for distributional parameters
            mean_prediction, epistemic_variance = network.forward_distributional(query_points)
            
            # Generate samples efficiently if needed
            if num_samples > 1:
                samples = self.posterior_sampler.sample_predictions(
                    query_points, num_samples, use_cache=True
                )
                # Recompute epistemic uncertainty from samples for accuracy
                epistemic_uncertainty = samples.var(dim=0)
            else:
                samples = mean_prediction.unsqueeze(0)
                epistemic_uncertainty = epistemic_variance
        
        # Aleatoric uncertainty: learned noise parameter
        aleatoric_uncertainty = torch.full_like(mean_prediction, self.aleatoric_std.item() ** 2)
        
        # Validate uncertainty decomposition
        self._validate_uncertainty_decomposition(epistemic_uncertainty, aleatoric_uncertainty)
        
        return UncertaintyPrediction(
            mean=mean_prediction,
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            samples=samples
        )
    
    def _predict_with_standard_inference(self, query_points: torch.Tensor,
                                       num_samples: int) -> UncertaintyPrediction:
        """Predict using standard inference (for comparison/fallback)."""
        # Use adapted network if available, otherwise use base network
        network = self.adapted_network if self._is_adapted else self.network
        network.eval()
        
        # Sample from posterior
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                sample_prediction = network(query_points)
                samples.append(sample_prediction)
        
        # Stack samples: [num_samples, batch_size, output_dim]
        samples = torch.stack(samples)
        
        # Compute statistics
        mean_prediction = samples.mean(dim=0)
        
        # Epistemic uncertainty: variance across posterior samples
        if num_samples > 1:
            epistemic_uncertainty = samples.var(dim=0)
        else:
            # For single sample, epistemic uncertainty is zero
            epistemic_uncertainty = torch.zeros_like(mean_prediction)
        
        # Aleatoric uncertainty: learned noise parameter
        aleatoric_uncertainty = torch.full_like(mean_prediction, self.aleatoric_std.item() ** 2)
        
        # Validate uncertainty decomposition
        self._validate_uncertainty_decomposition(epistemic_uncertainty, aleatoric_uncertainty)
        
        return UncertaintyPrediction(
            mean=mean_prediction,
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            samples=samples
        )
    
    def get_epistemic_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract epistemic (model) uncertainty.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Epistemic uncertainty [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points, num_samples=20)  # Use same as test
        return prediction.epistemic
    
    def get_aleatoric_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract aleatoric (data) uncertainty.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Aleatoric uncertainty [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points, num_samples=20)  # Use same as test
        return prediction.aleatoric
    
    def _validate_uncertainty_decomposition(self, epistemic: torch.Tensor, 
                                          aleatoric: torch.Tensor) -> None:
        """Validate uncertainty decomposition properties.
        
        Args:
            epistemic: Epistemic uncertainty tensor
            aleatoric: Aleatoric uncertainty tensor
            
        Raises:
            DecompositionError: If decomposition is invalid
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
    
    @property
    def is_adapted(self) -> bool:
        """Check if model has been adapted to a task."""
        return self._is_adapted
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state for new task."""
        self._is_adapted = False
        self.adapted_network = None
        self.adaptation_history = []
        
        # Reset aleatoric uncertainty to default
        with torch.no_grad():
            self.aleatoric_log_var.fill_(math.log(0.01))
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of adaptation process.
        
        Returns:
            Dictionary with adaptation statistics
        """
        if not self.adaptation_history:
            return {'status': 'No adaptation performed'}
        
        final_metrics = self.adaptation_history[-1]
        
        return {
            'adapted': self._is_adapted,
            'adaptation_steps': len(self.adaptation_history),
            'final_loss': final_metrics['total_loss'],
            'final_data_loss': final_metrics['data_loss'],
            'final_kl_loss': final_metrics['kl_loss'],
            'learned_aleatoric_std': final_metrics['aleatoric_std'],
            'loss_reduction': (
                self.adaptation_history[0]['total_loss'] - final_metrics['total_loss']
                if len(self.adaptation_history) > 1 else 0.0
            ),
            'adaptation_history': self.adaptation_history
        }
    
    def optimize_memory_usage(self, max_memory_gb: float = 4.0) -> Dict[str, Union[int, float]]:
        """Optimize memory usage for large-scale uncertainty quantification.
        
        Args:
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            Dictionary with optimization recommendations
        """
        if hasattr(self, 'amortized_network'):
            return optimize_memory_usage(self.amortized_network, max_memory_gb)
        else:
            # Fallback for standard network
            total_params = sum(p.numel() for p in self.network.parameters())
            param_memory_mb = total_params * 4 / (1024 ** 2)
            
            return {
                'total_parameters': total_params,
                'parameter_memory_mb': param_memory_mb,
                'max_memory_gb': max_memory_gb,
                'recommendation': 'Use amortized inference for better memory optimization'
            }
    
    def clear_inference_cache(self) -> None:
        """Clear inference cache to free memory."""
        if hasattr(self, 'posterior_sampler'):
            self.posterior_sampler.clear_cache()
    
    def get_inference_stats(self) -> Dict[str, Union[int, float]]:
        """Get inference performance statistics.
        
        Returns:
            Dictionary with inference statistics
        """
        stats = {
            'num_posterior_samples': self.num_posterior_samples,
            'is_adapted': self._is_adapted,
            'has_amortized_network': hasattr(self, 'amortized_network'),
            'has_posterior_sampler': hasattr(self, 'posterior_sampler')
        }
        
        if hasattr(self, 'posterior_sampler'):
            cache_stats = self.posterior_sampler.get_cache_stats()
            stats.update(cache_stats)
        
        return stats
    
    def parameters(self):
        """Get all model parameters."""
        params = list(self.network.parameters()) + [self.aleatoric_log_var]
        if hasattr(self.elbo_optimizer.elbo_loss, 'parameters'):
            params.extend(self.elbo_optimizer.elbo_loss.parameters())
        return params
    
    def state_dict(self) -> Dict:
        """Get model state dictionary."""
        return {
            'network': self.network.state_dict(),
            'aleatoric_log_var': self.aleatoric_log_var,
            'prior_state': {
                'pde_type': self.prior.pde_type.value if hasattr(self.prior.pde_type, 'value') else self.prior.pde_type,
                'input_dim': self.prior.input_dim,
                'output_dim': self.prior.output_dim,
                'physics_weight': self.prior.physics_weight
            },
            'config': {
                'network_architecture': self.network_architecture,
                'pde_type': self.pde_type,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'physics_informed_prior': self.physics_informed_prior,
                'variational_family': self.variational_family,
                'meta_lr': self.meta_lr,
                'adaptation_lr': self.adaptation_lr,
                'adaptation_steps': self.adaptation_steps,
                'num_posterior_samples': self.num_posterior_samples
            }
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state dictionary."""
        self.network.load_state_dict(state_dict['network'])
        self.aleatoric_log_var.data = state_dict['aleatoric_log_var'].data
        
        # Update configuration if provided
        if 'config' in state_dict:
            config = state_dict['config']
            self.network_architecture = config['network_architecture']
            self.pde_type = config['pde_type']
            self.input_dim = config['input_dim']
            self.output_dim = config['output_dim']
            self.physics_informed_prior = config['physics_informed_prior']
            self.variational_family = config['variational_family']
            self.meta_lr = config['meta_lr']
            self.adaptation_lr = config['adaptation_lr']
            self.adaptation_steps = config['adaptation_steps']
            self.num_posterior_samples = config['num_posterior_samples']


def create_bayesian_meta_pinn(config: Dict) -> BayesianMetaPINN:
    """Factory function to create BayesianMetaPINN with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured BayesianMetaPINN instance
    """
    default_config = {
        'network_architecture': {'dims': [2, 64, 64, 1], 'activation': 'tanh'},
        'pde_type': 'heat',
        'input_dim': 2,
        'output_dim': 1,
        'physics_informed_prior': True,
        'variational_family': 'diagonal_gaussian',
        'meta_lr': 0.001,
        'adaptation_lr': 0.01,
        'adaptation_steps': 10,
        'num_posterior_samples': 100,
        'device': 'cpu'
    }
    
    # Update with provided config
    default_config.update(config)
    
    return BayesianMetaPINN(**default_config)