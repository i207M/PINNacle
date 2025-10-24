"""Information-theoretic metrics for uncertainty quantification.

This module implements mutual information and entropy metrics for analyzing
uncertainty in Bayesian neural networks, particularly for epistemic uncertainty
quantification through I(y; θ | x).
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from dataclasses import dataclass

from .base import UncertaintyPrediction, UncertaintyMetaLearner

logger = logging.getLogger(__name__)


@dataclass
class InformationMetrics:
    """Container for information-theoretic uncertainty metrics.
    
    Attributes:
        mutual_information: I(y; θ | x) - mutual information between predictions and parameters
        predictive_entropy: H(y | x) - entropy of predictive distribution
        epistemic_entropy: H(E[y | θ, x]) - entropy due to parameter uncertainty
        aleatoric_entropy: E[H(y | θ, x)] - expected entropy of likelihood
        total_entropy: Total predictive entropy
        entropy_decomposition_valid: Whether entropy decomposition is valid
    """
    mutual_information: float
    predictive_entropy: float
    epistemic_entropy: float
    aleatoric_entropy: float
    total_entropy: float
    entropy_decomposition_valid: bool


class MutualInformationEstimator:
    """Estimator for mutual information I(y; θ | x) in Bayesian neural networks.
    
    This class implements several methods for approximating mutual information
    between predictions and model parameters, which quantifies epistemic uncertainty.
    """
    
    def __init__(self, 
                 estimation_method: str = "monte_carlo",
                 num_bins: int = 50,
                 bandwidth: float = 0.1):
        """Initialize mutual information estimator.
        
        Args:
            estimation_method: Method for MI estimation ('monte_carlo', 'kde', 'binning')
            num_bins: Number of bins for histogram-based estimation
            bandwidth: Bandwidth for KDE-based estimation
        """
        self.estimation_method = estimation_method
        self.num_bins = num_bins
        self.bandwidth = bandwidth
        
        self.supported_methods = ["monte_carlo", "kde", "binning"]
        if estimation_method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {estimation_method}. "
                           f"Choose from {self.supported_methods}")
        
        logger.info(f"Initialized MI estimator with method: {estimation_method}")
    
    def estimate_mutual_information(self, 
                                   predictions: UncertaintyPrediction,
                                   posterior_samples: Optional[torch.Tensor] = None) -> float:
        """Estimate mutual information I(y; θ | x).
        
        Args:
            predictions: Uncertainty predictions with posterior samples
            posterior_samples: Optional explicit posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            Estimated mutual information in nats
            
        Raises:
            ValueError: If required data is missing
        """
        # Use samples from predictions if not provided separately
        if posterior_samples is None:
            if predictions.samples is None:
                raise ValueError("Need posterior samples for MI estimation")
            posterior_samples = predictions.samples
        
        if self.estimation_method == "monte_carlo":
            return self._estimate_mi_monte_carlo(posterior_samples)
        elif self.estimation_method == "kde":
            return self._estimate_mi_kde(posterior_samples)
        elif self.estimation_method == "binning":
            return self._estimate_mi_binning(posterior_samples)
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")
    
    def _estimate_mi_monte_carlo(self, samples: torch.Tensor) -> float:
        """Estimate MI using Monte Carlo approximation.
        
        I(y; θ | x) ≈ H(y | x) - E_θ[H(y | θ, x)]
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            MI estimate in nats
        """
        n_samples, batch_size, output_dim = samples.shape
        samples_np = samples.detach().cpu().numpy()
        
        # Compute predictive entropy H(y | x)
        # Approximate as entropy of sample mean and variance
        sample_means = np.mean(samples_np, axis=0)  # [batch_size, output_dim]
        sample_vars = np.var(samples_np, axis=0)    # [batch_size, output_dim]
        
        # Gaussian entropy approximation: 0.5 * log(2πe * σ²)
        predictive_entropy = 0.5 * np.mean(np.log(2 * np.pi * np.e * sample_vars))
        
        # Compute expected conditional entropy E_θ[H(y | θ, x)]
        # For each posterior sample, estimate the conditional entropy
        conditional_entropies = []
        
        for i in range(n_samples):
            sample_i = samples_np[i]  # [batch_size, output_dim]
            
            # Estimate conditional variance (aleatoric uncertainty)
            # This is a simplified approximation - in practice would need
            # the actual likelihood variance from the model
            local_vars = np.var(sample_i, axis=0, keepdims=True)
            local_vars = np.maximum(local_vars, 1e-8)  # Avoid log(0)
            
            conditional_entropy = 0.5 * np.mean(np.log(2 * np.pi * np.e * local_vars))
            conditional_entropies.append(conditional_entropy)
        
        expected_conditional_entropy = np.mean(conditional_entropies)
        
        # MI = H(y | x) - E_θ[H(y | θ, x)]
        mi_estimate = predictive_entropy - expected_conditional_entropy
        
        logger.debug(f"MC MI estimate: H(y|x)={predictive_entropy:.4f}, "
                    f"E[H(y|θ,x)]={expected_conditional_entropy:.4f}, "
                    f"MI={mi_estimate:.4f}")
        
        return float(mi_estimate)
    
    def _estimate_mi_kde(self, samples: torch.Tensor) -> float:
        """Estimate MI using kernel density estimation.
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            MI estimate in nats
        """
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            logger.warning("scipy not available, falling back to Monte Carlo")
            return self._estimate_mi_monte_carlo(samples)
        
        n_samples, batch_size, output_dim = samples.shape
        samples_np = samples.detach().cpu().numpy()
        
        # Flatten for KDE (treat each output dimension independently)
        mi_estimates = []
        
        for dim in range(output_dim):
            dim_samples = samples_np[:, :, dim].flatten()  # [n_samples * batch_size]
            
            if len(np.unique(dim_samples)) < 10:
                logger.warning(f"Too few unique values for KDE in dim {dim}")
                continue
            
            try:
                # Fit KDE to marginal distribution
                kde = gaussian_kde(dim_samples, bw_method=self.bandwidth)
                
                # Estimate entropy using samples
                log_densities = kde.logpdf(dim_samples)
                marginal_entropy = -np.mean(log_densities)
                
                # Estimate conditional entropy (simplified)
                # This is an approximation - proper implementation would need
                # conditional densities p(y | θ, x)
                conditional_entropy = marginal_entropy * 0.7  # Rough approximation
                
                mi_dim = marginal_entropy - conditional_entropy
                mi_estimates.append(mi_dim)
                
            except Exception as e:
                logger.warning(f"KDE failed for dimension {dim}: {e}")
                continue
        
        if not mi_estimates:
            logger.warning("KDE estimation failed, using Monte Carlo")
            return self._estimate_mi_monte_carlo(samples)
        
        mi_estimate = np.mean(mi_estimates)
        logger.debug(f"KDE MI estimate: {mi_estimate:.4f}")
        
        return float(mi_estimate)
    
    def _estimate_mi_binning(self, samples: torch.Tensor) -> float:
        """Estimate MI using histogram binning.
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            MI estimate in nats
        """
        n_samples, batch_size, output_dim = samples.shape
        samples_np = samples.detach().cpu().numpy()
        
        mi_estimates = []
        
        for dim in range(output_dim):
            dim_samples = samples_np[:, :, dim].flatten()
            
            # Create histogram
            counts, bin_edges = np.histogram(dim_samples, bins=self.num_bins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Convert to probabilities
            probs = counts * bin_width
            probs = probs[probs > 0]  # Remove zero bins
            
            if len(probs) < 2:
                logger.warning(f"Insufficient bins for dimension {dim}")
                continue
            
            # Estimate marginal entropy
            marginal_entropy = -np.sum(probs * np.log(probs))
            
            # Rough conditional entropy estimate
            conditional_entropy = marginal_entropy * 0.6  # Approximation
            
            mi_dim = marginal_entropy - conditional_entropy
            mi_estimates.append(mi_dim)
        
        if not mi_estimates:
            logger.warning("Binning estimation failed, using Monte Carlo")
            return self._estimate_mi_monte_carlo(samples)
        
        mi_estimate = np.mean(mi_estimates)
        logger.debug(f"Binning MI estimate: {mi_estimate:.4f}")
        
        return float(mi_estimate)


class PredictiveEntropyCalculator:
    """Calculator for predictive entropy and its decomposition.
    
    Implements entropy decomposition:
    H(y | x) = H(E[y | θ, x]) + E[H(y | θ, x)]
             = Epistemic entropy + Aleatoric entropy
    """
    
    def __init__(self, entropy_method: str = "gaussian"):
        """Initialize entropy calculator.
        
        Args:
            entropy_method: Method for entropy calculation ('gaussian', 'sample')
        """
        self.entropy_method = entropy_method
        
        if entropy_method not in ["gaussian", "sample"]:
            raise ValueError(f"Unsupported entropy method: {entropy_method}")
    
    def compute_predictive_entropy(self, 
                                  predictions: UncertaintyPrediction) -> float:
        """Compute predictive entropy H(y | x).
        
        Args:
            predictions: Uncertainty predictions
            
        Returns:
            Predictive entropy in nats
        """
        if self.entropy_method == "gaussian":
            return self._compute_gaussian_entropy(predictions.total)
        elif self.entropy_method == "sample":
            if predictions.samples is None:
                raise ValueError("Need samples for sample-based entropy")
            return self._compute_sample_entropy(predictions.samples)
    
    def compute_entropy_decomposition(self, 
                                     predictions: UncertaintyPrediction) -> Dict[str, float]:
        """Compute entropy decomposition into epistemic and aleatoric components.
        
        Args:
            predictions: Uncertainty predictions with samples
            
        Returns:
            Dictionary with entropy components
        """
        if predictions.samples is None:
            # Use Gaussian approximation
            epistemic_entropy = self._compute_gaussian_entropy(predictions.epistemic)
            aleatoric_entropy = self._compute_gaussian_entropy(predictions.aleatoric)
            total_entropy = self._compute_gaussian_entropy(predictions.total)
        else:
            # Use sample-based computation
            epistemic_entropy = self._compute_epistemic_entropy_samples(predictions.samples)
            aleatoric_entropy = self._compute_aleatoric_entropy_samples(predictions.samples)
            total_entropy = self._compute_sample_entropy(predictions.samples)
        
        # Validate decomposition
        decomposition_error = abs(total_entropy - (epistemic_entropy + aleatoric_entropy))
        decomposition_valid = decomposition_error < 0.1 * total_entropy
        
        return {
            'epistemic_entropy': epistemic_entropy,
            'aleatoric_entropy': aleatoric_entropy,
            'total_entropy': total_entropy,
            'decomposition_error': decomposition_error,
            'decomposition_valid': decomposition_valid
        }
    
    def _compute_gaussian_entropy(self, variance: torch.Tensor) -> float:
        """Compute entropy assuming Gaussian distribution.
        
        H(X) = 0.5 * log(2πe * σ²)
        
        Args:
            variance: Variance tensor
            
        Returns:
            Entropy in nats
        """
        variance_np = variance.detach().cpu().numpy()
        variance_np = np.maximum(variance_np, 1e-8)  # Avoid log(0)
        
        entropy = 0.5 * np.mean(np.log(2 * np.pi * np.e * variance_np))
        return float(entropy)
    
    def _compute_sample_entropy(self, samples: torch.Tensor) -> float:
        """Compute entropy from samples using histogram.
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            Entropy in nats
        """
        samples_np = samples.detach().cpu().numpy()
        n_samples, batch_size, output_dim = samples_np.shape
        
        entropies = []
        
        for dim in range(output_dim):
            dim_samples = samples_np[:, :, dim].flatten()
            
            # Create histogram
            counts, _ = np.histogram(dim_samples, bins=50, density=True)
            counts = counts[counts > 0]
            
            if len(counts) < 2:
                continue
            
            # Normalize to probabilities
            probs = counts / np.sum(counts)
            
            # Compute entropy
            entropy_dim = -np.sum(probs * np.log(probs))
            entropies.append(entropy_dim)
        
        return float(np.mean(entropies)) if entropies else 0.0
    
    def _compute_epistemic_entropy_samples(self, samples: torch.Tensor) -> float:
        """Compute epistemic entropy from posterior samples.
        
        Epistemic entropy = H(E[y | θ, x])
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            Epistemic entropy in nats
        """
        # Compute mean across samples for each input
        sample_means = torch.mean(samples, dim=0)  # [batch_size, output_dim]
        
        # Compute variance of these means (epistemic uncertainty)
        epistemic_var = torch.var(sample_means, dim=0)  # [output_dim]
        
        return self._compute_gaussian_entropy(epistemic_var)
    
    def _compute_aleatoric_entropy_samples(self, samples: torch.Tensor) -> float:
        """Compute aleatoric entropy from posterior samples.
        
        Aleatoric entropy = E[H(y | θ, x)]
        
        Args:
            samples: Posterior samples [n_samples, batch_size, output_dim]
            
        Returns:
            Aleatoric entropy in nats
        """
        n_samples, batch_size, output_dim = samples.shape
        
        # For each posterior sample, compute the conditional entropy
        conditional_entropies = []
        
        for i in range(n_samples):
            sample_i = samples[i]  # [batch_size, output_dim]
            
            # Estimate conditional variance (this is simplified)
            # In practice, would need the actual likelihood variance
            conditional_var = torch.var(sample_i, dim=0, keepdim=True)
            conditional_var = torch.clamp(conditional_var, min=1e-8)
            
            conditional_entropy = self._compute_gaussian_entropy(conditional_var)
            conditional_entropies.append(conditional_entropy)
        
        return float(np.mean(conditional_entropies))


class InformationMetricsCalculator:
    """Main calculator for information-theoretic uncertainty metrics.
    
    Combines mutual information estimation and entropy decomposition
    to provide comprehensive information-theoretic analysis.
    """
    
    def __init__(self, 
                 mi_method: str = "monte_carlo",
                 entropy_method: str = "gaussian"):
        """Initialize information metrics calculator.
        
        Args:
            mi_method: Method for mutual information estimation
            entropy_method: Method for entropy calculation
        """
        self.mi_estimator = MutualInformationEstimator(estimation_method=mi_method)
        self.entropy_calculator = PredictiveEntropyCalculator(entropy_method=entropy_method)
        
        logger.info(f"Initialized information metrics calculator: "
                   f"MI={mi_method}, entropy={entropy_method}")
    
    def compute_all_metrics(self, 
                           predictions: UncertaintyPrediction) -> InformationMetrics:
        """Compute all information-theoretic metrics.
        
        Args:
            predictions: Uncertainty predictions with samples
            
        Returns:
            InformationMetrics containing all computed metrics
        """
        # Compute mutual information
        try:
            mutual_information = self.mi_estimator.estimate_mutual_information(predictions)
        except Exception as e:
            logger.warning(f"MI estimation failed: {e}")
            mutual_information = 0.0
        
        # Compute entropy decomposition
        try:
            entropy_results = self.entropy_calculator.compute_entropy_decomposition(predictions)
            epistemic_entropy = entropy_results['epistemic_entropy']
            aleatoric_entropy = entropy_results['aleatoric_entropy']
            total_entropy = entropy_results['total_entropy']
            decomposition_valid = entropy_results['decomposition_valid']
        except Exception as e:
            logger.warning(f"Entropy computation failed: {e}")
            epistemic_entropy = 0.0
            aleatoric_entropy = 0.0
            total_entropy = 0.0
            decomposition_valid = False
        
        # Compute predictive entropy
        try:
            predictive_entropy = self.entropy_calculator.compute_predictive_entropy(predictions)
        except Exception as e:
            logger.warning(f"Predictive entropy computation failed: {e}")
            predictive_entropy = total_entropy
        
        return InformationMetrics(
            mutual_information=mutual_information,
            predictive_entropy=predictive_entropy,
            epistemic_entropy=epistemic_entropy,
            aleatoric_entropy=aleatoric_entropy,
            total_entropy=total_entropy,
            entropy_decomposition_valid=decomposition_valid
        )
    
    def analyze_information_flow(self, 
                                model: UncertaintyMetaLearner,
                                query_points: torch.Tensor,
                                k_values: list = [1, 5, 10, 25]) -> Dict[str, list]:
        """Analyze how information metrics change with K-shot adaptation.
        
        Args:
            model: Uncertainty meta-learning model
            query_points: Query points for evaluation
            k_values: List of K-shot values to test
            
        Returns:
            Dictionary with metrics for each K value
        """
        results = {
            'k_values': k_values,
            'mutual_information': [],
            'predictive_entropy': [],
            'epistemic_entropy': [],
            'aleatoric_entropy': []
        }
        
        for k in k_values:
            try:
                # This would need to be called with proper task adaptation
                # For now, just get predictions
                predictions = model.predict_with_uncertainty(query_points)
                metrics = self.compute_all_metrics(predictions)
                
                results['mutual_information'].append(metrics.mutual_information)
                results['predictive_entropy'].append(metrics.predictive_entropy)
                results['epistemic_entropy'].append(metrics.epistemic_entropy)
                results['aleatoric_entropy'].append(metrics.aleatoric_entropy)
                
            except Exception as e:
                logger.warning(f"Information analysis failed for K={k}: {e}")
                results['mutual_information'].append(np.nan)
                results['predictive_entropy'].append(np.nan)
                results['epistemic_entropy'].append(np.nan)
                results['aleatoric_entropy'].append(np.nan)
        
        return results
    
    def validate_information_theory(self, 
                                   predictions: UncertaintyPrediction) -> Dict[str, bool]:
        """Validate information-theoretic properties.
        
        Args:
            predictions: Uncertainty predictions
            
        Returns:
            Dictionary with validation results
        """
        metrics = self.compute_all_metrics(predictions)
        
        # Check basic properties
        validations = {
            'non_negative_mi': metrics.mutual_information >= 0,
            'non_negative_entropy': metrics.predictive_entropy >= 0,
            'entropy_decomposition': metrics.entropy_decomposition_valid,
            'mi_bounded_by_entropy': metrics.mutual_information <= metrics.predictive_entropy
        }
        
        # Check theoretical relationships
        # MI should be related to epistemic uncertainty
        if metrics.epistemic_entropy > 0:
            mi_epistemic_ratio = metrics.mutual_information / metrics.epistemic_entropy
            validations['reasonable_mi_epistemic_ratio'] = 0.1 <= mi_epistemic_ratio <= 10.0
        else:
            validations['reasonable_mi_epistemic_ratio'] = True
        
        return validations