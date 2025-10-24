"""Calibration metrics and evaluation framework for uncertainty quantification.

This module implements comprehensive calibration evaluation tools including
Expected Calibration Error (ECE), Maximum Calibration Error (MCE), reliability
diagrams, coverage analysis, and Continuous Ranked Probability Score (CRPS).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats

from .base import UncertaintyPrediction, CalibrationError


@dataclass
class CalibrationResults:
    """Results from calibration evaluation.
    
    Attributes:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error  
        coverage: Prediction interval coverage
        sharpness: Average prediction interval width
        crps: Continuous Ranked Probability Score
        reliability_data: Data for reliability diagram plotting
    """
    ece: float
    mce: float
    coverage: float
    sharpness: float
    crps: float
    reliability_data: Dict[str, torch.Tensor]


class CalibrationMetrics:
    """Comprehensive calibration evaluation metrics.
    
    This class implements various calibration metrics to assess whether
    uncertainty estimates match true error rates.
    """
    
    @staticmethod
    def expected_calibration_error(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        num_bins: int = 10,
        norm: str = 'l1'
    ) -> float:
        """Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted confidence and observed
        accuracy across confidence bins.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values [batch_size, output_dim]
            num_bins: Number of bins for calibration computation
            norm: Norm to use ('l1' or 'l2')
            
        Returns:
            Expected Calibration Error value
            
        Raises:
            CalibrationError: If computation fails
        """
        try:
            # Convert uncertainty to confidence
            confidences = CalibrationMetrics._uncertainty_to_confidence(predictions)
            
            # Compute accuracies (1 - normalized absolute error)
            accuracies = CalibrationMetrics._compute_accuracies(predictions, targets)
            
            # Create bins
            bin_boundaries = torch.linspace(0, 1, num_bins + 1)
            ece = 0.0
            
            for i in range(num_bins):
                # Find samples in this bin
                bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                
                if bin_mask.sum() > 0:
                    # Compute bin statistics
                    bin_confidence = confidences[bin_mask].mean()
                    bin_accuracy = accuracies[bin_mask].mean()
                    bin_weight = bin_mask.sum().float() / len(confidences)
                    
                    # Add to ECE
                    if norm == 'l1':
                        ece += bin_weight * torch.abs(bin_confidence - bin_accuracy)
                    elif norm == 'l2':
                        ece += bin_weight * (bin_confidence - bin_accuracy) ** 2
                    else:
                        raise ValueError(f"Unknown norm: {norm}")
            
            return ece.item()
            
        except Exception as e:
            raise CalibrationError(f"ECE computation failed: {e}")
    
    @staticmethod
    def maximum_calibration_error(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between confidence and accuracy
        across all bins.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values [batch_size, output_dim]
            num_bins: Number of bins for calibration computation
            
        Returns:
            Maximum Calibration Error value
            
        Raises:
            CalibrationError: If computation fails
        """
        try:
            # Convert uncertainty to confidence
            confidences = CalibrationMetrics._uncertainty_to_confidence(predictions)
            
            # Compute accuracies
            accuracies = CalibrationMetrics._compute_accuracies(predictions, targets)
            
            # Create bins
            bin_boundaries = torch.linspace(0, 1, num_bins + 1)
            max_error = 0.0
            
            for i in range(num_bins):
                # Find samples in this bin
                bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                
                if bin_mask.sum() > 0:
                    # Compute bin statistics
                    bin_confidence = confidences[bin_mask].mean()
                    bin_accuracy = accuracies[bin_mask].mean()
                    
                    # Update maximum error
                    bin_error = torch.abs(bin_confidence - bin_accuracy)
                    max_error = max(max_error, bin_error.item())
            
            return max_error
            
        except Exception as e:
            raise CalibrationError(f"MCE computation failed: {e}")
    
    @staticmethod
    def _uncertainty_to_confidence(predictions: UncertaintyPrediction) -> torch.Tensor:
        """Convert uncertainty to confidence level.
        
        For regression problems, confidence is computed as:
        confidence = 1 - normalized_uncertainty
        
        Args:
            predictions: Model predictions with uncertainty
            
        Returns:
            Confidence values [batch_size, output_dim]
        """
        # Flatten for easier computation
        total_uncertainty = predictions.total.flatten()
        
        # Normalize uncertainty to [0, 1] range
        min_uncertainty = total_uncertainty.min()
        max_uncertainty = total_uncertainty.max()
        
        if max_uncertainty > min_uncertainty:
            normalized_uncertainty = (total_uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        else:
            # All uncertainties are the same
            normalized_uncertainty = torch.zeros_like(total_uncertainty)
        
        # Convert to confidence
        confidence = 1.0 - normalized_uncertainty
        
        # Reshape back to original shape
        return confidence.reshape(predictions.total.shape)
    
    @staticmethod
    def _compute_accuracies(predictions: UncertaintyPrediction, targets: torch.Tensor) -> torch.Tensor:
        """Compute accuracies for regression problems.
        
        Accuracy is defined as 1 - normalized_absolute_error
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            
        Returns:
            Accuracy values [batch_size, output_dim]
        """
        # Compute absolute errors
        absolute_errors = torch.abs(predictions.mean - targets)
        
        # Flatten for normalization
        flat_errors = absolute_errors.flatten()
        flat_targets = targets.flatten()
        
        # Normalize errors by target range
        target_range = flat_targets.max() - flat_targets.min()
        if target_range > 0:
            normalized_errors = flat_errors / target_range
        else:
            # All targets are the same
            normalized_errors = torch.zeros_like(flat_errors)
        
        # Clip to [0, 1] and convert to accuracy
        normalized_errors = torch.clamp(normalized_errors, 0, 1)
        accuracies = 1.0 - normalized_errors
        
        # Reshape back to original shape
        return accuracies.reshape(absolute_errors.shape)


class ReliabilityDiagram:
    """Generate and visualize reliability diagrams for calibration assessment.
    
    Reliability diagrams plot observed frequency vs expected confidence,
    with perfect calibration represented by the diagonal y=x line.
    """
    
    def __init__(self, num_bins: int = 10):
        """Initialize reliability diagram generator.
        
        Args:
            num_bins: Number of bins for reliability computation
        """
        self.num_bins = num_bins
    
    def compute_reliability_data(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute data for reliability diagram.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values [batch_size, output_dim]
            
        Returns:
            Dictionary containing bin data for plotting
        """
        # Convert uncertainty to confidence
        confidences = CalibrationMetrics._uncertainty_to_confidence(predictions)
        
        # Compute accuracies
        accuracies = CalibrationMetrics._compute_accuracies(predictions, targets)
        
        # Flatten for binning
        confidences = confidences.flatten()
        accuracies = accuracies.flatten()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(self.num_bins):
            # Find samples in this bin
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.sum() > 0:
                bin_confidences.append(confidences[bin_mask].mean())
                bin_accuracies.append(accuracies[bin_mask].mean())
                bin_counts.append(bin_mask.sum())
            else:
                bin_confidences.append(torch.tensor(0.0))
                bin_accuracies.append(torch.tensor(0.0))
                bin_counts.append(torch.tensor(0))
        
        return {
            'bin_confidences': torch.stack(bin_confidences),
            'bin_accuracies': torch.stack(bin_accuracies),
            'bin_counts': torch.stack(bin_counts),
            'bin_boundaries': bin_boundaries
        }
    
    def plot_reliability_diagram(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot reliability diagram with perfect calibration line.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Compute reliability data
        reliability_data = self.compute_reliability_data(predictions, targets)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot reliability curve
        bin_confidences = reliability_data['bin_confidences'].numpy()
        bin_accuracies = reliability_data['bin_accuracies'].numpy()
        bin_counts = reliability_data['bin_counts'].numpy()
        
        # Only plot bins with data
        valid_bins = bin_counts > 0
        if valid_bins.sum() > 0:
            ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                   'o-', label='Model', linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        
        # Formatting
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add bin count information
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            if count > 0:
                ax.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CoverageAnalysis:
    """Coverage and sharpness analysis for prediction intervals.
    
    This class implements prediction interval coverage computation and
    sharpness measurement for uncertainty quantification evaluation.
    """
    
    @staticmethod
    def coverage_analysis(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Analyze prediction interval coverage and sharpness.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values [batch_size, output_dim]
            confidence_level: Confidence level for prediction intervals (e.g., 0.95)
            
        Returns:
            Dictionary containing coverage and sharpness metrics
        """
        # Compute z-score for confidence level
        alpha = 1 - confidence_level
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2))
        
        # Compute prediction intervals
        std_dev = torch.sqrt(predictions.total)
        lower_bound = predictions.mean - z_score * std_dev
        upper_bound = predictions.mean + z_score * std_dev
        
        # Compute coverage (fraction of targets within intervals)
        within_interval = (targets >= lower_bound) & (targets <= upper_bound)
        coverage = within_interval.float().mean()
        
        # Compute sharpness (average interval width)
        interval_width = upper_bound - lower_bound
        sharpness = interval_width.mean()
        
        return {
            'coverage': coverage.item(),
            'sharpness': sharpness.item(),
            'target_coverage': confidence_level,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    @staticmethod
    def coverage_vs_sharpness_analysis(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        confidence_levels: List[float] = None
    ) -> Dict[str, List[float]]:
        """Analyze coverage vs sharpness trade-off across confidence levels.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            confidence_levels: List of confidence levels to analyze
            
        Returns:
            Dictionary containing coverage and sharpness for each confidence level
        """
        if confidence_levels is None:
            confidence_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
        
        coverages = []
        sharpnesses = []
        
        for conf_level in confidence_levels:
            analysis = CoverageAnalysis.coverage_analysis(
                predictions, targets, conf_level
            )
            coverages.append(analysis['coverage'])
            sharpnesses.append(analysis['sharpness'])
        
        return {
            'confidence_levels': confidence_levels,
            'coverages': coverages,
            'sharpnesses': sharpnesses
        }
    
    @staticmethod
    def plot_coverage_vs_sharpness(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        confidence_levels: List[float] = None,
        title: str = "Coverage vs Sharpness Trade-off",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot coverage vs sharpness trade-off.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            confidence_levels: List of confidence levels to analyze
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Compute coverage vs sharpness data
        analysis = CoverageAnalysis.coverage_vs_sharpness_analysis(
            predictions, targets, confidence_levels
        )
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot coverage vs sharpness
        ax.plot(analysis['sharpnesses'], analysis['coverages'], 'o-', 
               linewidth=2, markersize=8, label='Model')
        
        # Plot ideal coverage line
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, 
                  label='Target Coverage (95%)')
        
        # Formatting
        ax.set_xlabel('Sharpness (Average Interval Width)')
        ax.set_ylabel('Coverage')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate points with confidence levels
        for i, (sharp, cov, conf) in enumerate(zip(
            analysis['sharpnesses'], analysis['coverages'], analysis['confidence_levels']
        )):
            ax.annotate(f'{conf:.0%}', (sharp, cov), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ContinuousRankedProbabilityScore:
    """Continuous Ranked Probability Score (CRPS) for proper scoring.
    
    CRPS is a proper scoring rule that evaluates the quality of probabilistic
    predictions by comparing the predicted distribution to the observed value.
    """
    
    @staticmethod
    def crps_gaussian(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor
    ) -> float:
        """Compute CRPS for Gaussian predictive distributions.
        
        For a Gaussian distribution N(μ, σ²), the CRPS has a closed-form solution:
        CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
        where z = (y - μ)/σ, Φ is the CDF, and φ is the PDF of standard normal.
        
        Args:
            predictions: Model predictions with uncertainty (assumed Gaussian)
            targets: True target values [batch_size, output_dim]
            
        Returns:
            Mean CRPS value across all predictions
        """
        # Extract mean and standard deviation
        mu = predictions.mean
        sigma = torch.sqrt(predictions.total)
        
        # Avoid division by zero
        sigma = torch.clamp(sigma, min=1e-8)
        
        # Standardize targets
        z = (targets - mu) / sigma
        
        # Standard normal CDF and PDF
        normal_dist = torch.distributions.Normal(0, 1)
        phi_z = normal_dist.cdf(z)  # CDF
        pdf_z = torch.exp(normal_dist.log_prob(z))  # PDF
        
        # CRPS formula for Gaussian
        crps = sigma * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        
        return crps.mean().item()
    
    @staticmethod
    def crps_empirical(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor
    ) -> float:
        """Compute CRPS using empirical samples from posterior.
        
        This method uses the empirical formula:
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        where X, X' are independent samples from the predictive distribution.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values [batch_size, output_dim]
            
        Returns:
            Mean CRPS value across all predictions
            
        Raises:
            ValueError: If no posterior samples are available
        """
        if predictions.samples is None:
            raise ValueError("Empirical CRPS requires posterior samples")
        
        samples = predictions.samples  # [n_samples, batch_size, output_dim]
        n_samples = samples.shape[0]
        
        # Expand targets to match samples shape
        targets_expanded = targets.unsqueeze(0).expand_as(samples)
        
        # Compute E[|X - y|]
        term1 = torch.abs(samples - targets_expanded).mean(dim=0)
        
        # Compute E[|X - X'|] using all pairs
        samples_i = samples.unsqueeze(1)  # [n_samples, 1, batch_size, output_dim]
        samples_j = samples.unsqueeze(0)  # [1, n_samples, batch_size, output_dim]
        
        pairwise_diff = torch.abs(samples_i - samples_j)
        term2 = pairwise_diff.mean(dim=(0, 1))
        
        # CRPS formula
        crps = term1 - 0.5 * term2
        
        return crps.mean().item()


class ProperScoringRules:
    """Framework for proper scoring rule evaluation.
    
    This class provides a unified interface for evaluating different
    proper scoring rules for uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize proper scoring rules framework."""
        self.scoring_rules = {
            'crps_gaussian': ContinuousRankedProbabilityScore.crps_gaussian,
            'crps_empirical': ContinuousRankedProbabilityScore.crps_empirical,
            'negative_log_likelihood': self._negative_log_likelihood,
            'brier_score': self._brier_score
        }
    
    def evaluate_all_scores(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate all available proper scoring rules.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            
        Returns:
            Dictionary containing all scoring rule results
        """
        scores = {}
        
        for rule_name, rule_func in self.scoring_rules.items():
            try:
                scores[rule_name] = rule_func(predictions, targets)
            except Exception as e:
                print(f"Warning: {rule_name} computation failed: {e}")
                scores[rule_name] = float('nan')
        
        return scores
    
    @staticmethod
    def _negative_log_likelihood(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor
    ) -> float:
        """Compute negative log-likelihood for Gaussian predictions.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            
        Returns:
            Mean negative log-likelihood
        """
        # Create Gaussian distribution
        mu = predictions.mean
        sigma = torch.sqrt(predictions.total)
        sigma = torch.clamp(sigma, min=1e-8)  # Avoid numerical issues
        
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(targets)
        
        return -log_prob.mean().item()
    
    @staticmethod
    def _brier_score(
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        threshold: float = 0.0
    ) -> float:
        """Compute Brier score for binary events (e.g., |error| > threshold).
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            threshold: Threshold for defining binary events
            
        Returns:
            Brier score
        """
        # Define binary event: |error| > threshold
        errors = torch.abs(predictions.mean - targets)
        binary_outcomes = (errors > threshold).float()
        
        # Predict probability of event using uncertainty
        # Higher uncertainty -> higher probability of large error
        sigma = torch.sqrt(predictions.total)
        prob_large_error = 1 - torch.exp(-sigma / (threshold + 1e-8))
        
        # Brier score: (probability - outcome)²
        brier = (prob_large_error - binary_outcomes) ** 2
        
        return brier.mean().item()


# Comprehensive calibration evaluation function
def evaluate_calibration(
    predictions: UncertaintyPrediction,
    targets: torch.Tensor,
    num_bins: int = 10,
    confidence_levels: List[float] = None,
    compute_crps: bool = True
) -> CalibrationResults:
    """Comprehensive calibration evaluation.
    
    This function computes all calibration metrics in a single call.
    
    Args:
        predictions: Model predictions with uncertainty
        targets: True target values
        num_bins: Number of bins for ECE/MCE computation
        confidence_levels: Confidence levels for coverage analysis
        compute_crps: Whether to compute CRPS
        
    Returns:
        CalibrationResults containing all metrics
    """
    # Compute ECE and MCE
    ece = CalibrationMetrics.expected_calibration_error(predictions, targets, num_bins)
    mce = CalibrationMetrics.maximum_calibration_error(predictions, targets, num_bins)
    
    # Compute coverage and sharpness
    coverage_results = CoverageAnalysis.coverage_analysis(predictions, targets)
    
    # Compute reliability data
    reliability_diagram = ReliabilityDiagram(num_bins)
    reliability_data = reliability_diagram.compute_reliability_data(predictions, targets)
    
    # Compute CRPS
    crps = 0.0
    if compute_crps:
        try:
            crps = ContinuousRankedProbabilityScore.crps_gaussian(predictions, targets)
        except Exception as e:
            print(f"Warning: CRPS computation failed: {e}")
            crps = float('nan')
    
    return CalibrationResults(
        ece=ece,
        mce=mce,
        coverage=coverage_results['coverage'],
        sharpness=coverage_results['sharpness'],
        crps=crps,
        reliability_data=reliability_data
    )