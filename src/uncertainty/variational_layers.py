"""Variational layers for Bayesian neural networks.

This module implements variational linear layers with Gaussian posteriors
for Bayesian meta-learning in Physics-Informed Neural Networks.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class VariationalLinear(nn.Module):
    """Variational linear layer with Gaussian posterior distribution.
    
    This layer maintains a Gaussian posterior distribution over weights and biases,
    enabling Bayesian inference through the reparameterization trick.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias parameters
        prior_mean: Prior mean for weights and biases
        prior_std: Prior standard deviation for weights and biases
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior_mean: float = 0.0, prior_std: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
        # Weight posterior parameters: q(W) = N(μ_W, σ²_W)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        # Bias posterior parameters: q(b) = N(μ_b, σ²_b)
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_log_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize variational parameters using Xavier initialization."""
        # Initialize weight means with Xavier normal
        nn.init.xavier_normal_(self.weight_mu)
        
        # Initialize weight log-sigmas to small negative values (small initial variance)
        nn.init.constant_(self.weight_log_sigma, -3.0)
        
        if self.use_bias:
            # Initialize bias means to zero
            nn.init.zeros_(self.bias_mu)
            # Initialize bias log-sigmas to small negative values
            nn.init.constant_(self.bias_log_sigma, -3.0)
    
    @property
    def weight_sigma(self) -> torch.Tensor:
        """Get weight standard deviations from log-sigma parameters."""
        return torch.exp(self.weight_log_sigma)
    
    @property
    def bias_sigma(self) -> Optional[torch.Tensor]:
        """Get bias standard deviations from log-sigma parameters."""
        if self.use_bias:
            return torch.exp(self.bias_log_sigma)
        return None
    
    def sample_weights(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample weights and biases using reparameterization trick.
        
        Uses the reparameterization trick: θ = μ + σ ⊙ ε, where ε ~ N(0, I)
        
        Returns:
            Tuple of (sampled_weights, sampled_biases)
        """
        # Sample weights: W = μ_W + σ_W ⊙ ε_W
        epsilon_w = torch.randn_like(self.weight_mu)
        sampled_weights = self.weight_mu + self.weight_sigma * epsilon_w
        
        # Sample biases if used: b = μ_b + σ_b ⊙ ε_b
        sampled_biases = None
        if self.use_bias:
            epsilon_b = torch.randn_like(self.bias_mu)
            sampled_biases = self.bias_mu + self.bias_sigma * epsilon_b
        
        return sampled_weights, sampled_biases
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass using sampled weights.
        
        Args:
            input: Input tensor [batch_size, in_features]
            
        Returns:
            Output tensor [batch_size, out_features]
        """
        # Sample weights and biases
        weights, biases = self.sample_weights()
        
        # Compute linear transformation
        output = F.linear(input, weights, biases)
        
        return output
    
    def kl_divergence(self, prior: Optional['GaussianPrior'] = None) -> torch.Tensor:
        """Compute KL divergence between posterior and prior.
        
        For Gaussian distributions:
        KL(q(θ)||p(θ)) = 0.5 * [log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1]
        
        Args:
            prior: Prior distribution. If None, uses standard Gaussian prior.
            
        Returns:
            KL divergence scalar
        """
        if prior is None:
            # Use default prior parameters
            prior_mean = self.prior_mean
            prior_std = self.prior_std
        else:
            prior_mean = prior.get_mean(self.weight_mu.shape)
            prior_std = prior.get_std(self.weight_mu.shape)
            
            # Convert tensors to scalars if they are constant
            if isinstance(prior_mean, torch.Tensor) and prior_mean.numel() == 1:
                prior_mean = prior_mean.item()
            if isinstance(prior_std, torch.Tensor) and prior_std.numel() == 1:
                prior_std = prior_std.item()
        
        # KL divergence for weights
        weight_kl = self._gaussian_kl_divergence(
            self.weight_mu, self.weight_sigma, prior_mean, prior_std
        )
        
        total_kl = weight_kl
        
        # KL divergence for biases
        if self.use_bias:
            if prior is None:
                bias_prior_mean = self.prior_mean
                bias_prior_std = self.prior_std
            else:
                bias_prior_mean = prior.get_mean(self.bias_mu.shape)
                bias_prior_std = prior.get_std(self.bias_mu.shape)
                
                # Convert tensors to scalars if they are constant
                if isinstance(bias_prior_mean, torch.Tensor) and bias_prior_mean.numel() == 1:
                    bias_prior_mean = bias_prior_mean.item()
                if isinstance(bias_prior_std, torch.Tensor) and bias_prior_std.numel() == 1:
                    bias_prior_std = bias_prior_std.item()
            
            bias_kl = self._gaussian_kl_divergence(
                self.bias_mu, self.bias_sigma, bias_prior_mean, bias_prior_std
            )
            total_kl = total_kl + bias_kl
        
        return total_kl
    
    def _gaussian_kl_divergence(self, mu_q: torch.Tensor, sigma_q: torch.Tensor,
                               mu_p: Union[float, torch.Tensor], sigma_p: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute KL divergence between two Gaussian distributions.
        
        Args:
            mu_q: Posterior mean
            sigma_q: Posterior standard deviation
            mu_p: Prior mean (scalar or tensor)
            sigma_p: Prior standard deviation (scalar or tensor)
            
        Returns:
            KL divergence
        """
        # Convert prior parameters to tensors with proper broadcasting
        if isinstance(mu_p, torch.Tensor):
            if mu_p.shape != mu_q.shape:
                mu_p = mu_p.expand_as(mu_q)
        else:
            mu_p = torch.full_like(mu_q, mu_p)
            
        if isinstance(sigma_p, torch.Tensor):
            if sigma_p.shape != sigma_q.shape:
                sigma_p = sigma_p.expand_as(sigma_q)
        else:
            sigma_p = torch.full_like(sigma_q, sigma_p)
        
        # KL(q||p) = 0.5 * [log(σ²_p/σ²_q) + (σ²_q + (μ_q - μ_p)²)/σ²_p - 1]
        var_q = sigma_q ** 2
        var_p = sigma_p ** 2
        
        kl = 0.5 * (
            torch.log(var_p / var_q) +
            (var_q + (mu_q - mu_p) ** 2) / var_p - 1
        )
        
        return kl.sum()
    
    def get_weight_statistics(self) -> dict:
        """Get statistics of weight posterior distribution.
        
        Returns:
            Dictionary containing weight statistics
        """
        stats = {
            'weight_mean_norm': torch.norm(self.weight_mu).item(),
            'weight_std_mean': self.weight_sigma.mean().item(),
            'weight_std_max': self.weight_sigma.max().item(),
            'weight_std_min': self.weight_sigma.min().item(),
        }
        
        if self.use_bias:
            stats.update({
                'bias_mean_norm': torch.norm(self.bias_mu).item(),
                'bias_std_mean': self.bias_sigma.mean().item(),
                'bias_std_max': self.bias_sigma.max().item(),
                'bias_std_min': self.bias_sigma.min().item(),
            })
        
        return stats
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.use_bias}, prior_mean={self.prior_mean}, ' \
               f'prior_std={self.prior_std}'


class GaussianPrior:
    """Gaussian prior distribution for variational layers.
    
    This class defines a Gaussian prior that can be either standard
    or physics-informed based on PDE structure.
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def get_mean(self, shape: torch.Size) -> float:
        """Get prior mean for given parameter shape.
        
        Args:
            shape: Parameter tensor shape
            
        Returns:
            Prior mean value
        """
        return self.mean
    
    def get_std(self, shape: torch.Size) -> float:
        """Get prior standard deviation for given parameter shape.
        
        Args:
            shape: Parameter tensor shape
            
        Returns:
            Prior standard deviation value
        """
        return self.std
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of value under prior.
        
        Args:
            value: Parameter values
            
        Returns:
            Log probability
        """
        dist = Normal(self.mean, self.std)
        return dist.log_prob(value).sum()


def create_variational_network(layer_dims: list, activation: str = 'tanh',
                              prior_mean: float = 0.0, prior_std: float = 1.0) -> nn.Module:
    """Create a neural network with variational linear layers.
    
    Args:
        layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function name ('tanh', 'relu', 'sigmoid')
        prior_mean: Prior mean for all layers
        prior_std: Prior standard deviation for all layers
        
    Returns:
        Sequential model with variational layers
    """
    if len(layer_dims) < 2:
        raise ValueError("layer_dims must have at least 2 elements")
    
    # Activation function mapping
    activation_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU
    }
    
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}")
    
    layers = []
    
    # Create variational linear layers with activations
    for i in range(len(layer_dims) - 1):
        in_dim = layer_dims[i]
        out_dim = layer_dims[i + 1]
        
        # Add variational linear layer
        layers.append(VariationalLinear(
            in_dim, out_dim, 
            bias=True, 
            prior_mean=prior_mean, 
            prior_std=prior_std
        ))
        
        # Add activation (except for output layer)
        if i < len(layer_dims) - 2:
            layers.append(activation_map[activation]())
    
    return nn.Sequential(*layers)


def compute_network_kl_divergence(network: nn.Module, prior: Optional[GaussianPrior] = None) -> torch.Tensor:
    """Compute total KL divergence for all variational layers in network.
    
    Args:
        network: Neural network containing VariationalLinear layers
        prior: Prior distribution for all layers
        
    Returns:
        Total KL divergence
    """
    total_kl = 0.0
    
    for module in network.modules():
        if isinstance(module, VariationalLinear):
            total_kl = total_kl + module.kl_divergence(prior)
    
    return total_kl


def get_network_weight_statistics(network: nn.Module) -> dict:
    """Get weight statistics for all variational layers in network.
    
    Args:
        network: Neural network containing VariationalLinear layers
        
    Returns:
        Dictionary with aggregated weight statistics
    """
    all_stats = {}
    layer_count = 0
    
    for name, module in network.named_modules():
        if isinstance(module, VariationalLinear):
            layer_stats = module.get_weight_statistics()
            for key, value in layer_stats.items():
                stat_name = f"layer_{layer_count}_{key}"
                all_stats[stat_name] = value
            layer_count += 1
    
    # Compute aggregate statistics
    if layer_count > 0:
        weight_means = [all_stats[f"layer_{i}_weight_mean_norm"] for i in range(layer_count)]
        weight_stds = [all_stats[f"layer_{i}_weight_std_mean"] for i in range(layer_count)]
        
        all_stats['total_layers'] = layer_count
        all_stats['avg_weight_mean_norm'] = sum(weight_means) / len(weight_means)
        all_stats['avg_weight_std'] = sum(weight_stds) / len(weight_stds)
    
    return all_stats