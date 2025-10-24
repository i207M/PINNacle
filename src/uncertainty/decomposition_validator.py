"""Uncertainty decomposition validation framework.

This module implements validation tools for epistemic/aleatoric uncertainty
decomposition, ensuring that epistemic uncertainty decreases with more data
while aleatoric uncertainty remains constant.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from scipy import stats
from dataclasses import dataclass

from .base import UncertaintyMetaLearner, UncertaintyPrediction, Task, DecompositionError

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResults:
    """Results from uncertainty decomposition validation.
    
    Attributes:
        epistemic_decreasing: Whether epistemic uncertainty decreases with K
        aleatoric_constant: Whether aleatoric uncertainty remains constant
        epistemic_slope: Slope of log(epistemic) vs log(K) regression
        epistemic_r_squared: R² of epistemic decrease regression
        epistemic_p_value: P-value for epistemic decrease significance
        aleatoric_cv: Coefficient of variation for aleatoric uncertainty
        aleatoric_mean: Mean aleatoric uncertainty across K values
        aleatoric_std: Standard deviation of aleatoric uncertainty
        decomposition_valid: Overall validation result
        k_values: K-shot values tested
        epistemic_by_k: Mean epistemic uncertainty for each K
        aleatoric_by_k: Mean aleatoric uncertainty for each K
        statistical_power: Statistical power of the tests
    """
    epistemic_decreasing: bool
    aleatoric_constant: bool
    epistemic_slope: float
    epistemic_r_squared: float
    epistemic_p_value: float
    aleatoric_cv: float
    aleatoric_mean: float
    aleatoric_std: float
    decomposition_valid: bool
    k_values: List[int]
    epistemic_by_k: Dict[int, float]
    aleatoric_by_k: Dict[int, float]
    statistical_power: float


class UncertaintyDecompositionValidator:
    """Validator for epistemic/aleatoric uncertainty decomposition.
    
    This class implements statistical tests to validate that:
    1. Epistemic uncertainty decreases with more support data (K-shot)
    2. Aleatoric uncertainty remains constant across different K values
    3. The decomposition follows theoretical expectations
    """
    
    def __init__(self, 
                 epistemic_slope_threshold: float = -0.3,
                 aleatoric_cv_threshold: float = 0.2,
                 significance_level: float = 0.05,
                 min_power: float = 0.8):
        """Initialize the decomposition validator.
        
        Args:
            epistemic_slope_threshold: Minimum slope for epistemic decrease
            aleatoric_cv_threshold: Maximum CV for aleatoric constancy
            significance_level: Statistical significance threshold
            min_power: Minimum statistical power required
        """
        self.epistemic_slope_threshold = epistemic_slope_threshold
        self.aleatoric_cv_threshold = aleatoric_cv_threshold
        self.significance_level = significance_level
        self.min_power = min_power
        
        logger.info(f"Initialized decomposition validator with thresholds: "
                   f"epistemic_slope < {epistemic_slope_threshold}, "
                   f"aleatoric_cv < {aleatoric_cv_threshold}")
    
    def validate_decomposition(self, 
                              model: UncertaintyMetaLearner,
                              test_tasks: List[Task],
                              k_values: List[int] = [1, 5, 10, 25, 50],
                              num_query_points: int = 100,
                              num_samples: int = 100) -> DecompositionResults:
        """Validate uncertainty decomposition across K-shot scenarios.
        
        Args:
            model: Uncertainty meta-learning model to validate
            test_tasks: List of test tasks for validation
            k_values: List of K-shot values to test
            num_query_points: Number of query points per task
            num_samples: Number of posterior samples for uncertainty estimation
            
        Returns:
            DecompositionResults containing validation results
            
        Raises:
            DecompositionError: If validation setup is invalid
        """
        if not test_tasks:
            raise DecompositionError("No test tasks provided")
        
        if len(k_values) < 3:
            raise DecompositionError("Need at least 3 K values for regression analysis")
        
        if max(k_values) > 100:
            logger.warning(f"Large K values may be computationally expensive: {max(k_values)}")
        
        logger.info(f"Validating decomposition on {len(test_tasks)} tasks "
                   f"with K values {k_values}")
        
        # Collect uncertainty data across tasks and K values
        epistemic_data = []
        aleatoric_data = []
        k_shot_data = []
        
        for task_idx, task in enumerate(test_tasks):
            logger.debug(f"Processing task {task_idx + 1}/{len(test_tasks)}")
            
            for k in k_values:
                try:
                    # Sample support and query data
                    support_data, support_targets = task.sample_support(k)
                    query_data, query_targets = task.sample_query(num_query_points)
                    
                    # Adapt model to task
                    model.reset_adaptation()
                    adapted_model = model.adapt(support_data, support_targets)
                    
                    # Get uncertainty predictions
                    predictions = adapted_model.predict_with_uncertainty(
                        query_data, num_samples=num_samples
                    )
                    
                    # Collect uncertainty values
                    epistemic_values = predictions.epistemic.detach().cpu().numpy().flatten()
                    aleatoric_values = predictions.aleatoric.detach().cpu().numpy().flatten()
                    
                    # Filter out invalid values
                    valid_mask = (
                        np.isfinite(epistemic_values) & 
                        np.isfinite(aleatoric_values) &
                        (epistemic_values >= 0) & 
                        (aleatoric_values >= 0)
                    )
                    
                    if not np.any(valid_mask):
                        logger.warning(f"No valid uncertainty values for task {task_idx}, K={k}")
                        continue
                    
                    epistemic_data.extend(epistemic_values[valid_mask])
                    aleatoric_data.extend(aleatoric_values[valid_mask])
                    k_shot_data.extend([k] * np.sum(valid_mask))
                    
                except Exception as e:
                    logger.warning(f"Failed to process task {task_idx}, K={k}: {e}")
                    continue
        
        if not epistemic_data:
            raise DecompositionError("No valid uncertainty data collected")
        
        # Convert to numpy arrays
        epistemic_data = np.array(epistemic_data)
        aleatoric_data = np.array(aleatoric_data)
        k_shot_data = np.array(k_shot_data)
        
        logger.info(f"Collected {len(epistemic_data)} uncertainty measurements")
        
        # Analyze decomposition properties
        return self._analyze_decomposition(epistemic_data, aleatoric_data, k_shot_data, k_values)
    
    def _analyze_decomposition(self, 
                              epistemic: np.ndarray,
                              aleatoric: np.ndarray,
                              k_values_data: np.ndarray,
                              k_values_list: List[int]) -> DecompositionResults:
        """Analyze uncertainty decomposition properties.
        
        Args:
            epistemic: Array of epistemic uncertainty values
            aleatoric: Array of aleatoric uncertainty values
            k_values_data: Array of corresponding K values
            k_values_list: List of unique K values tested
            
        Returns:
            DecompositionResults with analysis results
        """
        # Test 1: Epistemic uncertainty decreases with K
        epistemic_results = self._test_epistemic_decrease(epistemic, k_values_data)
        
        # Test 2: Aleatoric uncertainty remains constant
        aleatoric_results = self._test_aleatoric_constancy(aleatoric, k_values_data, k_values_list)
        
        # Compute statistical power
        statistical_power = self._compute_statistical_power(epistemic, aleatoric, k_values_data)
        
        # Overall validation
        decomposition_valid = (
            epistemic_results['decreasing'] and 
            aleatoric_results['constant'] and
            statistical_power >= self.min_power
        )
        
        return DecompositionResults(
            epistemic_decreasing=epistemic_results['decreasing'],
            aleatoric_constant=aleatoric_results['constant'],
            epistemic_slope=epistemic_results['slope'],
            epistemic_r_squared=epistemic_results['r_squared'],
            epistemic_p_value=epistemic_results['p_value'],
            aleatoric_cv=aleatoric_results['cv'],
            aleatoric_mean=aleatoric_results['mean'],
            aleatoric_std=aleatoric_results['std'],
            decomposition_valid=decomposition_valid,
            k_values=k_values_list,
            epistemic_by_k=epistemic_results['by_k'],
            aleatoric_by_k=aleatoric_results['by_k'],
            statistical_power=statistical_power
        )
    
    def _test_epistemic_decrease(self, 
                                epistemic: np.ndarray,
                                k_values: np.ndarray) -> Dict[str, Any]:
        """Test if epistemic uncertainty decreases with K.
        
        Fits log(epistemic) = a + b * log(K) and tests if b < threshold.
        
        Args:
            epistemic: Epistemic uncertainty values
            k_values: Corresponding K values
            
        Returns:
            Dictionary with regression results
        """
        # Add small epsilon to avoid log(0)
        log_epistemic = np.log(epistemic + 1e-8)
        log_k = np.log(k_values)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_epistemic)
        
        # Test if slope is significantly negative
        decreasing = slope < self.epistemic_slope_threshold and p_value < self.significance_level
        
        # Compute mean epistemic by K
        epistemic_by_k = {}
        for k in np.unique(k_values):
            mask = k_values == k
            epistemic_by_k[int(k)] = float(np.mean(epistemic[mask]))
        
        logger.info(f"Epistemic regression: slope={slope:.4f}, R²={r_value**2:.4f}, "
                   f"p={p_value:.4f}, decreasing={decreasing}")
        
        return {
            'decreasing': decreasing,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'by_k': epistemic_by_k
        }
    
    def _test_aleatoric_constancy(self, 
                                 aleatoric: np.ndarray,
                                 k_values: np.ndarray,
                                 k_values_list: List[int]) -> Dict[str, Any]:
        """Test if aleatoric uncertainty remains constant across K.
        
        Args:
            aleatoric: Aleatoric uncertainty values
            k_values: Corresponding K values
            k_values_list: List of unique K values
            
        Returns:
            Dictionary with constancy test results
        """
        # Compute mean aleatoric for each K
        aleatoric_by_k = {}
        aleatoric_means = []
        
        for k in k_values_list:
            mask = k_values == k
            if np.any(mask):
                mean_aleatoric = np.mean(aleatoric[mask])
                aleatoric_by_k[k] = float(mean_aleatoric)
                aleatoric_means.append(mean_aleatoric)
            else:
                logger.warning(f"No data for K={k}")
        
        if len(aleatoric_means) < 2:
            logger.warning("Insufficient data for aleatoric constancy test")
            return {
                'constant': False,
                'cv': float('inf'),
                'mean': 0.0,
                'std': 0.0,
                'by_k': aleatoric_by_k
            }
        
        # Compute coefficient of variation
        aleatoric_means = np.array(aleatoric_means)
        mean_aleatoric = np.mean(aleatoric_means)
        std_aleatoric = np.std(aleatoric_means)
        cv = std_aleatoric / (mean_aleatoric + 1e-8)
        
        # Test constancy
        constant = cv < self.aleatoric_cv_threshold
        
        logger.info(f"Aleatoric constancy: CV={cv:.4f}, mean={mean_aleatoric:.4f}, "
                   f"constant={constant}")
        
        return {
            'constant': constant,
            'cv': cv,
            'mean': mean_aleatoric,
            'std': std_aleatoric,
            'by_k': aleatoric_by_k
        }
    
    def _compute_statistical_power(self, 
                                  epistemic: np.ndarray,
                                  aleatoric: np.ndarray,
                                  k_values: np.ndarray) -> float:
        """Compute statistical power of the decomposition tests.
        
        Args:
            epistemic: Epistemic uncertainty values
            aleatoric: Aleatoric uncertainty values
            k_values: Corresponding K values
            
        Returns:
            Estimated statistical power [0, 1]
        """
        # Simple power estimation based on sample size and effect size
        n_samples = len(epistemic)
        
        # Effect size for epistemic decrease (Cohen's d)
        k_unique = np.unique(k_values)
        if len(k_unique) < 2:
            return 0.0
        
        # Compare first and last K values
        mask_low = k_values == k_unique[0]
        mask_high = k_values == k_unique[-1]
        
        if not (np.any(mask_low) and np.any(mask_high)):
            return 0.0
        
        epistemic_low = epistemic[mask_low]
        epistemic_high = epistemic[mask_high]
        
        # Cohen's d for epistemic decrease
        pooled_std = np.sqrt(
            ((len(epistemic_low) - 1) * np.var(epistemic_low) + 
             (len(epistemic_high) - 1) * np.var(epistemic_high)) /
            (len(epistemic_low) + len(epistemic_high) - 2)
        )
        
        if pooled_std == 0:
            effect_size = 0.0
        else:
            effect_size = abs(np.mean(epistemic_low) - np.mean(epistemic_high)) / pooled_std
        
        # Rough power estimation (simplified)
        # In practice, would use proper power analysis
        if effect_size > 0.8 and n_samples > 100:
            power = 0.9
        elif effect_size > 0.5 and n_samples > 50:
            power = 0.8
        elif effect_size > 0.2 and n_samples > 30:
            power = 0.6
        else:
            power = 0.4
        
        return min(power, 1.0)
    
    def validate_single_task(self, 
                            model: UncertaintyMetaLearner,
                            task: Task,
                            k_values: List[int] = [1, 5, 10, 25],
                            num_query_points: int = 50) -> Dict[str, Any]:
        """Validate decomposition on a single task for debugging.
        
        Args:
            model: Uncertainty meta-learning model
            task: Single task to validate
            k_values: K-shot values to test
            num_query_points: Number of query points
            
        Returns:
            Dictionary with single-task validation results
        """
        results = {
            'k_values': k_values,
            'epistemic_means': [],
            'aleatoric_means': [],
            'epistemic_stds': [],
            'aleatoric_stds': []
        }
        
        for k in k_values:
            try:
                # Sample data
                support_data, support_targets = task.sample_support(k)
                query_data, query_targets = task.sample_query(num_query_points)
                
                # Adapt and predict
                model.reset_adaptation()
                adapted_model = model.adapt(support_data, support_targets)
                predictions = adapted_model.predict_with_uncertainty(query_data)
                
                # Collect statistics
                epistemic = predictions.epistemic.detach().cpu().numpy().flatten()
                aleatoric = predictions.aleatoric.detach().cpu().numpy().flatten()
                
                results['epistemic_means'].append(np.mean(epistemic))
                results['aleatoric_means'].append(np.mean(aleatoric))
                results['epistemic_stds'].append(np.std(epistemic))
                results['aleatoric_stds'].append(np.std(aleatoric))
                
            except Exception as e:
                logger.warning(f"Failed single task validation for K={k}: {e}")
                results['epistemic_means'].append(np.nan)
                results['aleatoric_means'].append(np.nan)
                results['epistemic_stds'].append(np.nan)
                results['aleatoric_stds'].append(np.nan)
        
        return results
    
    def create_validation_report(self, results: DecompositionResults) -> str:
        """Create a human-readable validation report.
        
        Args:
            results: Decomposition validation results
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("=" * 60)
        report.append("UNCERTAINTY DECOMPOSITION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall result
        status = "✓ PASSED" if results.decomposition_valid else "✗ FAILED"
        report.append(f"Overall Validation: {status}")
        report.append("")
        
        # Epistemic uncertainty analysis
        report.append("Epistemic Uncertainty Analysis:")
        report.append(f"  Decreasing with K: {'✓' if results.epistemic_decreasing else '✗'}")
        report.append(f"  Slope: {results.epistemic_slope:.4f} (threshold: {self.epistemic_slope_threshold})")
        report.append(f"  R²: {results.epistemic_r_squared:.4f}")
        report.append(f"  P-value: {results.epistemic_p_value:.4f}")
        report.append("")
        
        # Aleatoric uncertainty analysis
        report.append("Aleatoric Uncertainty Analysis:")
        report.append(f"  Constant across K: {'✓' if results.aleatoric_constant else '✗'}")
        report.append(f"  Coefficient of Variation: {results.aleatoric_cv:.4f} (threshold: {self.aleatoric_cv_threshold})")
        report.append(f"  Mean: {results.aleatoric_mean:.4f}")
        report.append(f"  Std: {results.aleatoric_std:.4f}")
        report.append("")
        
        # Statistical power
        report.append(f"Statistical Power: {results.statistical_power:.3f} (minimum: {self.min_power})")
        report.append("")
        
        # K-shot breakdown
        report.append("Uncertainty by K-shot:")
        report.append("  K  | Epistemic | Aleatoric")
        report.append("-----|-----------|----------")
        for k in results.k_values:
            epistemic = results.epistemic_by_k.get(k, 0.0)
            aleatoric = results.aleatoric_by_k.get(k, 0.0)
            report.append(f"  {k:2d} | {epistemic:8.4f} | {aleatoric:8.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)