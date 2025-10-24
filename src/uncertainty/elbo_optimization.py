"""ELBO optimization framework for Bayesian meta-learning.

This module implements the Evidence Lower Bound (ELBO) optimization framework
for variational Bayesian inference in Physics-Informed Neural Networks.
"""

import math
from typing import Dict, List, Optional, Tuple, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import numpy as np

from .base import UncertaintyQuantificationError, ConvergenceError
from .variational_layers import VariationalLinear, compute_network_kl_divergence
from .physics_priors import PhysicsInformedPrior


class ELBOLoss(nn.Module):
    """Evidence Lower Bound loss for variational Bayesian inference.
    
    The ELBO combines three components:
    1. Data likelihood: log p(y|x, θ)
    2. Physics likelihood: log p(PDE residual = 0|x, θ)  
    3. KL divergence: KL(q(θ|φ) || p(θ))
    
    ELBO = E_q[log p(y|x, θ)] + E_q[log p(PDE|x, θ)] - KL(q(θ|φ) || p(θ))
    
    Args:
        physics_weight: Weight for physics likelihood term
        kl_weight: Weight for KL divergence term (for annealing)
        likelihood_type: Type of likelihood ('gaussian', 'laplace')
        noise_std: Standard deviation for likelihood (if fixed)
        learn_noise: Whether to learn noise parameter
    """
    
    def __init__(self, 
                 physics_weight: float = 1.0,
                 kl_weight: float = 1.0,
                 likelihood_type: str = 'gaussian',
                 noise_std: Optional[float] = None,
                 learn_noise: bool = True):
        super().__init__()
        
        self.physics_weight = physics_weight
        self.kl_weight = kl_weight
        self.likelihood_type = likelihood_type
        self.learn_noise = learn_noise
        
        # Initialize noise parameter
        if noise_std is not None:
            if learn_noise:
                self.log_noise_std = nn.Parameter(torch.log(torch.tensor(noise_std)))
            else:
                self.register_buffer('log_noise_std', torch.log(torch.tensor(noise_std)))
        else:
            # Initialize with reasonable default
            if learn_noise:
                self.log_noise_std = nn.Parameter(torch.log(torch.tensor(0.1)))
            else:
                self.register_buffer('log_noise_std', torch.log(torch.tensor(0.1)))
    
    @property
    def noise_std(self) -> torch.Tensor:
        """Get current noise standard deviation."""
        return torch.exp(self.log_noise_std)
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                pde_residuals: torch.Tensor,
                kl_divergence: torch.Tensor,
                num_data_points: int) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss and its components.
        
        Args:
            predictions: Model predictions [batch_size, output_dim]
            targets: Target values [batch_size, output_dim]
            pde_residuals: PDE residual values [batch_size, 1]
            kl_divergence: KL divergence between posterior and prior
            num_data_points: Total number of data points (for proper scaling)
            
        Returns:
            Dictionary containing loss components and total ELBO
        """
        # Compute data likelihood
        data_likelihood = self._compute_data_likelihood(predictions, targets)
        
        # Compute physics likelihood
        physics_likelihood = self._compute_physics_likelihood(pde_residuals)
        
        # Scale KL divergence by number of data points (standard in variational inference)
        scaled_kl = kl_divergence / num_data_points
        
        # Compute ELBO (note: we minimize negative ELBO)
        elbo = data_likelihood + self.physics_weight * physics_likelihood - self.kl_weight * scaled_kl
        loss = -elbo  # Negative because we minimize
        
        return {
            'loss': loss,
            'elbo': elbo,
            'data_likelihood': data_likelihood,
            'physics_likelihood': physics_likelihood,
            'kl_divergence': kl_divergence,
            'scaled_kl': scaled_kl,
            'noise_std': self.noise_std
        }
    
    def _compute_data_likelihood(self, predictions: torch.Tensor, 
                               targets: torch.Tensor) -> torch.Tensor:
        """Compute data likelihood term.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Data likelihood (log probability)
        """
        if self.likelihood_type == 'gaussian':
            # Gaussian likelihood: log N(y|μ, σ²)
            mse = torch.mean((predictions - targets) ** 2)
            log_likelihood = -0.5 * (
                mse / (self.noise_std ** 2) +
                torch.log(2 * math.pi * self.noise_std ** 2)
            )
        elif self.likelihood_type == 'laplace':
            # Laplace likelihood: log Laplace(y|μ, b)
            mae = torch.mean(torch.abs(predictions - targets))
            b = self.noise_std / math.sqrt(2)  # Scale parameter for Laplace
            log_likelihood = -mae / b - torch.log(2 * b)
        else:
            raise ValueError(f"Unsupported likelihood type: {self.likelihood_type}")
        
        return log_likelihood
    
    def _compute_physics_likelihood(self, pde_residuals: torch.Tensor) -> torch.Tensor:
        """Compute physics likelihood term.
        
        The physics likelihood encourages the PDE residual to be close to zero.
        We model this as a Gaussian likelihood with small variance.
        
        Args:
            pde_residuals: PDE residual values
            
        Returns:
            Physics likelihood (log probability)
        """
        # Use small fixed variance for physics constraint
        physics_noise_std = 0.01
        
        # Gaussian likelihood for PDE residuals being zero
        mse_residual = torch.mean(pde_residuals ** 2)
        log_likelihood = -0.5 * (
            mse_residual / (physics_noise_std ** 2) +
            torch.log(2 * math.pi * physics_noise_std ** 2)
        )
        
        return log_likelihood
    
    def update_kl_weight(self, new_weight: float) -> None:
        """Update KL weight for annealing schedule."""
        self.kl_weight = new_weight


class KLAnnealingScheduler:
    """KL divergence weight annealing scheduler.
    
    Implements various annealing schedules for the KL weight to improve
    training stability and convergence.
    
    Args:
        schedule_type: Type of annealing ('linear', 'cosine', 'exponential', 'cyclical')
        start_weight: Initial KL weight
        end_weight: Final KL weight
        num_steps: Number of steps for annealing
        warmup_steps: Number of warmup steps (KL weight = 0)
    """
    
    def __init__(self, 
                 schedule_type: str = 'linear',
                 start_weight: float = 0.0,
                 end_weight: float = 1.0,
                 num_steps: int = 1000,
                 warmup_steps: int = 100):
        
        self.schedule_type = schedule_type
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_weight(self, step: Optional[int] = None) -> float:
        """Get KL weight for current or specified step.
        
        Args:
            step: Step number (uses internal counter if None)
            
        Returns:
            KL weight for the step
        """
        if step is None:
            step = self.current_step
        
        # Warmup phase: KL weight = 0
        if step < self.warmup_steps:
            return 0.0
        
        # Annealing phase
        progress = (step - self.warmup_steps) / (self.num_steps - self.warmup_steps)
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        if self.schedule_type == 'linear':
            weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        elif self.schedule_type == 'cosine':
            weight = self.start_weight + 0.5 * (self.end_weight - self.start_weight) * (
                1 - math.cos(math.pi * progress)
            )
        
        elif self.schedule_type == 'exponential':
            # Exponential growth from start to end
            if self.start_weight == 0:
                weight = self.end_weight * (1 - math.exp(-5 * progress))
            else:
                ratio = self.end_weight / self.start_weight
                weight = self.start_weight * (ratio ** progress)
        
        elif self.schedule_type == 'cyclical':
            # Cyclical annealing (useful for avoiding local minima)
            cycle_length = self.num_steps // 4  # 4 cycles
            cycle_progress = (step % cycle_length) / cycle_length
            weight = self.start_weight + cycle_progress * (self.end_weight - self.start_weight)
        
        else:
            raise ValueError(f"Unsupported schedule type: {self.schedule_type}")
        
        return weight
    
    def step(self) -> float:
        """Advance scheduler by one step and return current weight."""
        weight = self.get_weight()
        self.current_step += 1
        return weight
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_step = 0


class NaturalGradientOptimizer:
    """Natural gradient optimizer for variational parameters.
    
    Implements natural gradient descent for variational Bayesian inference,
    which can provide better convergence properties than standard gradient descent.
    
    Args:
        parameters: Variational parameters to optimize
        lr: Learning rate
        damping: Damping factor for numerical stability
        update_freq: Frequency of Fisher information matrix updates
    """
    
    def __init__(self, 
                 parameters,
                 lr: float = 0.01,
                 damping: float = 1e-4,
                 update_freq: int = 10):
        
        self.param_groups = [{'params': list(parameters)}]
        self.lr = lr
        self.damping = damping
        self.update_freq = update_freq
        self.step_count = 0
        
        # Initialize Fisher information matrix approximation
        self.fisher_info = {}
        self._initialize_fisher_info()
    
    def _initialize_fisher_info(self) -> None:
        """Initialize Fisher information matrix approximation."""
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    # Initialize as identity matrix (scaled)
                    self.fisher_info[param] = torch.eye(
                        param.numel(), 
                        device=param.device,
                        dtype=param.dtype
                    ) * self.damping
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform one optimization step using natural gradients.
        
        Args:
            closure: Optional closure to re-evaluate the model
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update Fisher information matrix periodically
        if self.step_count % self.update_freq == 0:
            self._update_fisher_info()
        
        # Apply natural gradient updates
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Flatten gradient
                    grad_flat = param.grad.view(-1)
                    
                    # Compute natural gradient: F^(-1) * grad
                    fisher_inv = torch.inverse(
                        self.fisher_info[param] + 
                        self.damping * torch.eye(param.numel(), device=param.device)
                    )
                    natural_grad = torch.mv(fisher_inv, grad_flat)
                    
                    # Apply update
                    param.data.add_(natural_grad.view(param.shape), alpha=-self.lr)
        
        self.step_count += 1
        return loss
    
    def _update_fisher_info(self) -> None:
        """Update Fisher information matrix approximation.
        
        Uses the current gradients to update the Fisher information matrix
        using a moving average approach.
        """
        momentum = 0.9
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad_flat = param.grad.view(-1)
                    
                    # Outer product approximation: F ≈ E[∇log p * ∇log p^T]
                    outer_product = torch.outer(grad_flat, grad_flat)
                    
                    # Moving average update
                    self.fisher_info[param] = (
                        momentum * self.fisher_info[param] + 
                        (1 - momentum) * outer_product
                    )
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()


class ELBOOptimizer:
    """High-level ELBO optimizer that coordinates all components.
    
    This class manages the ELBO optimization process, including:
    - Loss computation and backpropagation
    - KL weight annealing
    - Natural gradient optimization
    - Convergence monitoring
    
    Args:
        network: Variational neural network
        prior: Physics-informed prior
        optimizer_type: Type of optimizer ('adam', 'natural_gradient', 'sgd')
        lr: Learning rate
        kl_schedule: KL annealing schedule configuration
        physics_weight: Weight for physics likelihood
        convergence_patience: Patience for convergence detection
        min_improvement: Minimum improvement threshold for convergence
    """
    
    def __init__(self,
                 network: nn.Module,
                 prior: PhysicsInformedPrior,
                 optimizer_type: str = 'adam',
                 lr: float = 0.001,
                 kl_schedule: Optional[Dict] = None,
                 physics_weight: float = 1.0,
                 convergence_patience: int = 100,
                 min_improvement: float = 1e-6):
        
        self.network = network
        self.prior = prior
        self.physics_weight = physics_weight
        self.convergence_patience = convergence_patience
        self.min_improvement = min_improvement
        
        # Initialize ELBO loss
        self.elbo_loss = ELBOLoss(
            physics_weight=physics_weight,
            kl_weight=1.0,  # Will be updated by scheduler
            likelihood_type='gaussian',
            learn_noise=True
        )
        
        # Initialize KL annealing scheduler
        if kl_schedule is None:
            kl_schedule = {
                'schedule_type': 'linear',
                'start_weight': 0.0,
                'end_weight': 1.0,
                'num_steps': 1000,
                'warmup_steps': 100
            }
        
        self.kl_scheduler = KLAnnealingScheduler(**kl_schedule)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type, lr)
        
        # Training state
        self.training_history = {
            'elbo': [],
            'data_likelihood': [],
            'physics_likelihood': [],
            'kl_divergence': [],
            'kl_weight': [],
            'noise_std': []
        }
        
        self.step_count = 0
        self.best_elbo = float('-inf')
        self.patience_counter = 0
    
    def _create_optimizer(self, optimizer_type: str, lr: float):
        """Create optimizer based on type."""
        if optimizer_type == 'adam':
            return optim.Adam(
                list(self.network.parameters()) + list(self.elbo_loss.parameters()),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'natural_gradient':
            return NaturalGradientOptimizer(
                list(self.network.parameters()) + list(self.elbo_loss.parameters()),
                lr=lr
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                list(self.network.parameters()) + list(self.elbo_loss.parameters()),
                lr=lr,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def optimize_step(self, 
                     data_batch: Tuple[torch.Tensor, torch.Tensor],
                     physics_batch: torch.Tensor,
                     pde_residual_fn: Callable) -> Dict[str, float]:
        """Perform one optimization step.
        
        Args:
            data_batch: Tuple of (inputs, targets) for data likelihood
            physics_batch: Input points for physics likelihood
            pde_residual_fn: Function to compute PDE residuals
            
        Returns:
            Dictionary with loss components and metrics
        """
        self.optimizer.zero_grad()
        
        # Update KL weight
        kl_weight = self.kl_scheduler.step()
        self.elbo_loss.update_kl_weight(kl_weight)
        
        # Forward pass for data likelihood
        data_inputs, data_targets = data_batch
        data_predictions = self.network(data_inputs)
        
        # Forward pass for physics likelihood
        physics_predictions = self.network(physics_batch)
        pde_residuals = pde_residual_fn(physics_batch, physics_predictions)
        
        # Compute KL divergence
        kl_divergence = compute_network_kl_divergence(self.network, self.prior)
        
        # Compute ELBO loss
        num_data_points = len(data_inputs)
        loss_dict = self.elbo_loss(
            predictions=data_predictions,
            targets=data_targets,
            pde_residuals=pde_residuals,
            kl_divergence=kl_divergence,
            num_data_points=num_data_points
        )
        
        # Backward pass
        loss_dict['loss'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update training history
        self._update_history(loss_dict, kl_weight)
        
        # Check convergence
        converged = self._check_convergence(loss_dict['elbo'].item())
        
        # Prepare return dictionary
        result = {
            'loss': loss_dict['loss'].item(),
            'elbo': loss_dict['elbo'].item(),
            'data_likelihood': loss_dict['data_likelihood'].item(),
            'physics_likelihood': loss_dict['physics_likelihood'].item(),
            'kl_divergence': loss_dict['kl_divergence'].item(),
            'kl_weight': kl_weight,
            'noise_std': loss_dict['noise_std'].item(),
            'converged': converged,
            'step': self.step_count
        }
        
        self.step_count += 1
        return result
    
    def _update_history(self, loss_dict: Dict[str, torch.Tensor], kl_weight: float) -> None:
        """Update training history."""
        self.training_history['elbo'].append(loss_dict['elbo'].item())
        self.training_history['data_likelihood'].append(loss_dict['data_likelihood'].item())
        self.training_history['physics_likelihood'].append(loss_dict['physics_likelihood'].item())
        self.training_history['kl_divergence'].append(loss_dict['kl_divergence'].item())
        self.training_history['kl_weight'].append(kl_weight)
        self.training_history['noise_std'].append(loss_dict['noise_std'].item())
    
    def _check_convergence(self, current_elbo: float) -> bool:
        """Check if training has converged.
        
        Args:
            current_elbo: Current ELBO value
            
        Returns:
            True if converged, False otherwise
        """
        # Check if ELBO improved
        if current_elbo > self.best_elbo + self.min_improvement:
            self.best_elbo = current_elbo
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.convergence_patience
    
    def get_training_summary(self) -> Dict[str, Union[float, List[float]]]:
        """Get summary of training progress.
        
        Returns:
            Dictionary with training statistics
        """
        if not self.training_history['elbo']:
            return {'status': 'No training performed'}
        
        recent_window = min(100, len(self.training_history['elbo']))
        recent_elbos = self.training_history['elbo'][-recent_window:]
        
        return {
            'total_steps': self.step_count,
            'best_elbo': self.best_elbo,
            'current_elbo': self.training_history['elbo'][-1],
            'recent_elbo_mean': np.mean(recent_elbos),
            'recent_elbo_std': np.std(recent_elbos),
            'current_kl_weight': self.training_history['kl_weight'][-1],
            'current_noise_std': self.training_history['noise_std'][-1],
            'convergence_patience_remaining': max(0, self.convergence_patience - self.patience_counter),
            'training_history': self.training_history
        }
    
    def reset(self) -> None:
        """Reset optimizer state for new task."""
        self.step_count = 0
        self.best_elbo = float('-inf')
        self.patience_counter = 0
        self.kl_scheduler.reset()
        
        # Reset training history
        for key in self.training_history:
            self.training_history[key] = []
        
        # Reset optimizer state
        if hasattr(self.optimizer, 'state'):
            self.optimizer.state = {}


def create_elbo_optimizer(network: nn.Module,
                         prior: PhysicsInformedPrior,
                         config: Optional[Dict] = None) -> ELBOOptimizer:
    """Factory function to create ELBO optimizer with default configuration.
    
    Args:
        network: Variational neural network
        prior: Physics-informed prior
        config: Optional configuration dictionary
        
    Returns:
        Configured ELBOOptimizer instance
    """
    default_config = {
        'optimizer_type': 'adam',
        'lr': 0.001,
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
    
    if config is not None:
        default_config.update(config)
    
    return ELBOOptimizer(network, prior, **default_config)