"""
Uncertainty Evaluator Orchestration for Pinnacle v2.0

This module provides comprehensive evaluation protocol for uncertainty quantification methods,
extending the existing Pinnacle framework with calibration, decomposition, and OOD evaluation.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass, asdict
import json

from src.uncertainty.base import UncertaintyMetaLearner, UncertaintyPrediction
from src.uncertainty.calibration_metrics import CalibrationMetrics
from src.uncertainty.decomposition_validator import UncertaintyDecompositionValidator
from src.uncertainty.ood_detection import OODDetectionEvaluator
from src.meta_learning.task import Task
from src.meta_learning.evaluation_framework import MetaLearningEvaluationFramework

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEvaluationResults:
    """Container for comprehensive uncertainty evaluation results."""
    model_name: str
    k_shot: int
    calibration_results: Dict[str, float]
    decomposition_results: Dict[str, Any]
    ood_results: Dict[str, float]
    timing_results: Dict[str, float]
    statistical_summary: Dict[str, float]


@dataclass
class ComprehensiveUncertaintyResults:
    """Container for results across all K values and models."""
    evaluation_config: Dict[str, Any]
    results_by_model: Dict[str, Dict[int, UncertaintyEvaluationResults]]
    cross_model_comparison: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    summary_report: Dict[str, Any]


class UncertaintyEvaluatorOrchestrator:
    """
    Comprehensive uncertainty evaluation orchestrator for Pinnacle v2.0.
    
    Provides systematic evaluation protocol across K ∈ {1, 5, 10, 25} with
    calibration, decomposition, and OOD evaluation capabilities.
    """
    
    def __init__(
        self,
        k_shot_values: List[int] = None,
        output_dir: str = "uncertainty_evaluation",
        confidence_level: float = 0.95,
        num_bootstrap_samples: int = 1000,
        device: str = "cuda",
        verbose: bool = True
    ):
        """
        Initialize uncertainty evaluator orchestrator.
        
        Args:
            k_shot_values: List of K values for K-shot evaluation
            output_dir: Directory for saving evaluation results
            confidence_level: Confidence level for statistical analysis
            num_bootstrap_samples: Number of bootstrap samples for confidence intervals
            device: Computing device
            verbose: Whether to print progress information
        """
        self.k_shot_values = k_shot_values or [1, 5, 10, 25]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.device = device
        self.verbose = verbose
        
        # Initialize component evaluators
        self.calibration_metrics = CalibrationMetrics()
        self.decomposition_validator = UncertaintyDecompositionValidator()
        self.ood_evaluator = OODDetectionEvaluator()
        
        # Initialize base evaluation framework for backward compatibility
        self.base_evaluator = MetaLearningEvaluationFramework(
            output_dir=str(self.output_dir / "base_evaluation"),
            evaluation_shots=self.k_shot_values,
            confidence_level=confidence_level,
            device=device,
            verbose=verbose
        )
        
        logger.info(f"UncertaintyEvaluatorOrchestrator initialized with K={self.k_shot_values}")
    
    def evaluate_comprehensive_uncertainty(
        self,
        models: Dict[str, UncertaintyMetaLearner],
        test_tasks: List[Task],
        problem_type: str = "heat",
        noise_level: float = 0.05
    ) -> ComprehensiveUncertaintyResults:
        """
        Perform comprehensive uncertainty evaluation across all models and K values.
        
        Args:
            models: Dictionary mapping model names to uncertainty models
            test_tasks: List of test tasks for evaluation
            problem_type: Type of PDE problem being evaluated
            noise_level: Noise level in the test data
            
        Returns:
            ComprehensiveUncertaintyResults containing all evaluation results
        """
        logger.info(f"Starting comprehensive uncertainty evaluation on {len(models)} models")
        
        evaluation_config = {
            "k_shot_values": self.k_shot_values,
            "num_test_tasks": len(test_tasks),
            "problem_type": problem_type,
            "noise_level": noise_level,
            "confidence_level": self.confidence_level,
            "device": self.device
        }
        
        results_by_model = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            if self.verbose:
                print(f"\nEvaluating model: {model_name}")
            
            model_results = {}
            
            # Evaluate across all K values
            for k_shot in self.k_shot_values:
                if self.verbose:
                    print(f"  K-shot evaluation: K={k_shot}")
                
                k_results = self._evaluate_single_k_shot(
                    model, test_tasks, k_shot, model_name
                )
                model_results[k_shot] = k_results
            
            results_by_model[model_name] = model_results
        
        # Perform cross-model comparison
        cross_model_comparison = self._perform_cross_model_comparison(results_by_model)
        
        # Statistical significance testing
        statistical_significance = self._compute_statistical_significance(results_by_model)
        
        # Generate summary report
        summary_report = self._generate_summary_report(
            results_by_model, cross_model_comparison, statistical_significance
        )
        
        # Create comprehensive results
        comprehensive_results = ComprehensiveUncertaintyResults(
            evaluation_config=evaluation_config,
            results_by_model=results_by_model,
            cross_model_comparison=cross_model_comparison,
            statistical_significance=statistical_significance,
            summary_report=summary_report
        )
        
        # Save results
        self._save_comprehensive_results(comprehensive_results)
        
        if self.verbose:
            print(f"\nComprehensive evaluation completed. Results saved to: {self.output_dir}")
        
        return comprehensive_results
    
    def _evaluate_single_k_shot(
        self,
        model: UncertaintyMetaLearner,
        test_tasks: List[Task],
        k_shot: int,
        model_name: str
    ) -> UncertaintyEvaluationResults:
        """Evaluate single model for specific K-shot scenario."""
        
        # Collect predictions and targets for all test tasks
        all_predictions = []
        all_targets = []
        adaptation_times = []
        inference_times = []
        
        for task_idx, task in enumerate(test_tasks):
            try:
                # Sample support and query data
                task_data = task.get_task_data()
                
                # Sample K support points
                support_indices = np.random.choice(
                    len(task_data.x_physics), size=min(k_shot, len(task_data.x_physics)), replace=False
                )
                support_data = task_data.x_physics[support_indices]
                support_targets = task_data.u_physics[support_indices]
                
                # Sample query points (remaining data)
                query_indices = np.setdiff1d(np.arange(len(task_data.x_physics)), support_indices)
                if len(query_indices) == 0:
                    continue
                    
                query_data = task_data.x_physics[query_indices]
                query_targets = task_data.u_physics[query_indices]
                
                # Measure adaptation time
                start_time = time.time()
                adapted_model = model.adapt(support_data, support_targets, num_steps=10)
                adaptation_time = time.time() - start_time
                adaptation_times.append(adaptation_time)
                
                # Measure inference time
                start_time = time.time()
                predictions = adapted_model.predict_with_uncertainty(query_data, num_samples=100)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions)
                all_targets.append(query_targets)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate task {task_idx} for {model_name}: {e}")
                continue
        
        if not all_predictions:
            logger.error(f"No successful evaluations for {model_name} with K={k_shot}")
            return self._create_empty_results(model_name, k_shot)
        
        # Combine all predictions and targets
        combined_predictions = self._combine_predictions(all_predictions)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Evaluate calibration
        calibration_results = self._evaluate_calibration(combined_predictions, combined_targets)
        
        # Evaluate uncertainty decomposition
        decomposition_results = self._evaluate_decomposition(
            model, test_tasks, k_shot
        )
        
        # Evaluate OOD detection
        ood_results = self._evaluate_ood_detection(model, test_tasks)
        
        # Compute timing statistics
        timing_results = {
            "mean_adaptation_time": np.mean(adaptation_times),
            "std_adaptation_time": np.std(adaptation_times),
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "total_evaluation_time": sum(adaptation_times) + sum(inference_times)
        }
        
        # Compute statistical summary
        statistical_summary = self._compute_statistical_summary(
            combined_predictions, combined_targets, calibration_results
        )
        
        return UncertaintyEvaluationResults(
            model_name=model_name,
            k_shot=k_shot,
            calibration_results=calibration_results,
            decomposition_results=decomposition_results,
            ood_results=ood_results,
            timing_results=timing_results,
            statistical_summary=statistical_summary
        )
    
    def _combine_predictions(self, predictions_list: List[UncertaintyPrediction]) -> UncertaintyPrediction:
        """Combine multiple UncertaintyPrediction objects."""
        means = torch.cat([pred.mean for pred in predictions_list], dim=0)
        epistemics = torch.cat([pred.epistemic for pred in predictions_list], dim=0)
        aleatorics = torch.cat([pred.aleatoric for pred in predictions_list], dim=0)
        
        # Combine samples if available
        samples = None
        if all(pred.samples is not None for pred in predictions_list):
            samples = torch.cat([pred.samples for pred in predictions_list], dim=1)
        
        return UncertaintyPrediction(
            mean=means,
            epistemic=epistemics,
            aleatoric=aleatorics,
            samples=samples
        )
    
    def _evaluate_calibration(
        self, 
        predictions: UncertaintyPrediction, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate calibration metrics."""
        try:
            ece = self.calibration_metrics.expected_calibration_error(predictions, targets)
            mce = self.calibration_metrics.maximum_calibration_error(predictions, targets)
            coverage_results = self.calibration_metrics.coverage_analysis(predictions, targets)
            crps = self.calibration_metrics.continuous_ranked_probability_score(predictions, targets)
            
            return {
                "ece": ece,
                "mce": mce,
                "coverage": coverage_results["coverage"],
                "sharpness": coverage_results["sharpness"],
                "crps": crps
            }
        except Exception as e:
            logger.warning(f"Calibration evaluation failed: {e}")
            return {
                "ece": float('nan'),
                "mce": float('nan'),
                "coverage": float('nan'),
                "sharpness": float('nan'),
                "crps": float('nan')
            }
    
    def _evaluate_decomposition(
        self,
        model: UncertaintyMetaLearner,
        test_tasks: List[Task],
        current_k: int
    ) -> Dict[str, Any]:
        """Evaluate uncertainty decomposition properties."""
        try:
            # Use subset of tasks for decomposition validation
            sample_tasks = test_tasks[:min(10, len(test_tasks))]
            
            decomposition_results = self.decomposition_validator.validate_decomposition(
                model, sample_tasks, k_values=[1, 5, 10, 25]
            )
            
            return decomposition_results
        except Exception as e:
            logger.warning(f"Decomposition evaluation failed: {e}")
            return {
                "epistemic_decreasing": False,
                "aleatoric_constant": False,
                "decomposition_valid": False,
                "error": str(e)
            }
    
    def _evaluate_ood_detection(
        self,
        model: UncertaintyMetaLearner,
        test_tasks: List[Task]
    ) -> Dict[str, float]:
        """Evaluate out-of-distribution detection performance."""
        try:
            # Test spatial extrapolation scenario
            ood_results = self.ood_evaluator.evaluate_ood_detection(
                model, "spatial_extrapolation", num_in_distribution=100, num_ood=100
            )
            
            return ood_results
        except Exception as e:
            logger.warning(f"OOD evaluation failed: {e}")
            return {
                "auroc": float('nan'),
                "aupr": float('nan'),
                "fpr_at_95_tpr": float('nan')
            }
    
    def _compute_statistical_summary(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        calibration_results: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute statistical summary of predictions."""
        try:
            # Compute prediction errors
            errors = torch.abs(predictions.mean - targets)
            
            # Compute uncertainty statistics
            total_uncertainty = predictions.epistemic + predictions.aleatoric
            
            return {
                "mean_absolute_error": errors.mean().item(),
                "std_absolute_error": errors.std().item(),
                "mean_epistemic_uncertainty": predictions.epistemic.mean().item(),
                "std_epistemic_uncertainty": predictions.epistemic.std().item(),
                "mean_aleatoric_uncertainty": predictions.aleatoric.mean().item(),
                "std_aleatoric_uncertainty": predictions.aleatoric.std().item(),
                "mean_total_uncertainty": total_uncertainty.mean().item(),
                "uncertainty_error_correlation": torch.corrcoef(
                    torch.stack([errors.flatten(), total_uncertainty.flatten()])
                )[0, 1].item()
            }
        except Exception as e:
            logger.warning(f"Statistical summary computation failed: {e}")
            return {}
    
    def _perform_cross_model_comparison(
        self, 
        results_by_model: Dict[str, Dict[int, UncertaintyEvaluationResults]]
    ) -> Dict[str, Any]:
        """Perform cross-model comparison analysis."""
        comparison_results = {
            "best_model_by_metric": {},
            "performance_rankings": {},
            "improvement_analysis": {}
        }
        
        # Define metrics to compare
        metrics = ["ece", "coverage", "crps", "auroc"]
        
        for metric in metrics:
            model_scores = {}
            
            for model_name, k_results in results_by_model.items():
                scores = []
                for k_shot, results in k_results.items():
                    if metric in ["ece", "crps"]:  # Lower is better
                        score = results.calibration_results.get(metric, float('inf'))
                    elif metric == "coverage":  # Closer to 0.95 is better
                        score = abs(results.calibration_results.get(metric, 0.5) - 0.95)
                    elif metric == "auroc":  # Higher is better
                        score = 1.0 - results.ood_results.get(metric, 0.5)
                    else:
                        continue
                    
                    if not np.isnan(score):
                        scores.append(score)
                
                if scores:
                    model_scores[model_name] = np.mean(scores)
            
            if model_scores:
                # Find best model for this metric
                best_model = min(model_scores, key=model_scores.get)
                comparison_results["best_model_by_metric"][metric] = {
                    "model": best_model,
                    "score": model_scores[best_model]
                }
                
                # Create ranking
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
                comparison_results["performance_rankings"][metric] = sorted_models
        
        return comparison_results
    
    def _compute_statistical_significance(
        self,
        results_by_model: Dict[str, Dict[int, UncertaintyEvaluationResults]]
    ) -> Dict[str, Any]:
        """Compute statistical significance of differences between models."""
        from scipy import stats
        
        significance_results = {
            "pairwise_comparisons": {},
            "bonferroni_correction": {},
            "effect_sizes": {}
        }
        
        model_names = list(results_by_model.keys())
        
        # Perform pairwise comparisons for ECE
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                
                # Collect ECE values across K shots
                ece_values_1 = []
                ece_values_2 = []
                
                for k_shot in self.k_shot_values:
                    if k_shot in results_by_model[model1] and k_shot in results_by_model[model2]:
                        ece1 = results_by_model[model1][k_shot].calibration_results.get("ece")
                        ece2 = results_by_model[model2][k_shot].calibration_results.get("ece")
                        
                        if not (np.isnan(ece1) or np.isnan(ece2)):
                            ece_values_1.append(ece1)
                            ece_values_2.append(ece2)
                
                if len(ece_values_1) >= 2:  # Need at least 2 points for t-test
                    try:
                        # Paired t-test
                        t_stat, p_value = stats.ttest_rel(ece_values_1, ece_values_2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (np.var(ece_values_1) + np.var(ece_values_2)) / 2
                        )
                        cohens_d = (np.mean(ece_values_1) - np.mean(ece_values_2)) / pooled_std
                        
                        significance_results["pairwise_comparisons"][comparison_key] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "cohens_d": cohens_d
                        }
                        
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {comparison_key}: {e}")
        
        # Apply Bonferroni correction
        num_comparisons = len(significance_results["pairwise_comparisons"])
        if num_comparisons > 0:
            corrected_alpha = 0.05 / num_comparisons
            
            for comparison_key, results in significance_results["pairwise_comparisons"].items():
                results["bonferroni_significant"] = results["p_value"] < corrected_alpha
            
            significance_results["bonferroni_correction"] = {
                "num_comparisons": num_comparisons,
                "corrected_alpha": corrected_alpha
            }
        
        return significance_results
    
    def _generate_summary_report(
        self,
        results_by_model: Dict[str, Dict[int, UncertaintyEvaluationResults]],
        cross_model_comparison: Dict[str, Any],
        statistical_significance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        summary = {
            "evaluation_overview": {
                "num_models_evaluated": len(results_by_model),
                "k_shot_values": self.k_shot_values,
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_summary": {},
            "calibration_analysis": {},
            "uncertainty_decomposition_summary": {},
            "ood_detection_summary": {},
            "computational_efficiency": {},
            "recommendations": []
        }
        
        # Performance summary
        for model_name, k_results in results_by_model.items():
            model_summary = {
                "average_ece": np.mean([
                    r.calibration_results.get("ece", float('nan')) 
                    for r in k_results.values()
                ]),
                "average_coverage": np.mean([
                    r.calibration_results.get("coverage", float('nan')) 
                    for r in k_results.values()
                ]),
                "average_auroc": np.mean([
                    r.ood_results.get("auroc", float('nan')) 
                    for r in k_results.values()
                ]),
                "decomposition_valid": any([
                    r.decomposition_results.get("decomposition_valid", False)
                    for r in k_results.values()
                ])
            }
            summary["performance_summary"][model_name] = model_summary
        
        # Generate recommendations
        best_ece_model = cross_model_comparison["best_model_by_metric"].get("ece", {}).get("model")
        if best_ece_model:
            summary["recommendations"].append(
                f"For best calibration (ECE < 0.05): Use {best_ece_model}"
            )
        
        best_auroc_model = cross_model_comparison["best_model_by_metric"].get("auroc", {}).get("model")
        if best_auroc_model:
            summary["recommendations"].append(
                f"For best OOD detection (AUROC > 0.90): Use {best_auroc_model}"
            )
        
        return summary
    
    def _create_empty_results(self, model_name: str, k_shot: int) -> UncertaintyEvaluationResults:
        """Create empty results for failed evaluations."""
        return UncertaintyEvaluationResults(
            model_name=model_name,
            k_shot=k_shot,
            calibration_results={
                "ece": float('nan'),
                "mce": float('nan'),
                "coverage": float('nan'),
                "sharpness": float('nan'),
                "crps": float('nan')
            },
            decomposition_results={
                "epistemic_decreasing": False,
                "aleatoric_constant": False,
                "decomposition_valid": False
            },
            ood_results={
                "auroc": float('nan'),
                "aupr": float('nan'),
                "fpr_at_95_tpr": float('nan')
            },
            timing_results={
                "mean_adaptation_time": float('nan'),
                "mean_inference_time": float('nan')
            },
            statistical_summary={}
        )
    
    def _save_comprehensive_results(self, results: ComprehensiveUncertaintyResults):
        """Save comprehensive results to disk."""
        
        # Convert to serializable format
        results_dict = {
            "evaluation_config": results.evaluation_config,
            "results_by_model": {},
            "cross_model_comparison": results.cross_model_comparison,
            "statistical_significance": results.statistical_significance,
            "summary_report": results.summary_report
        }
        
        # Convert UncertaintyEvaluationResults to dict
        for model_name, k_results in results.results_by_model.items():
            results_dict["results_by_model"][model_name] = {}
            for k_shot, eval_results in k_results.items():
                results_dict["results_by_model"][model_name][k_shot] = asdict(eval_results)
        
        # Save to JSON
        output_file = self.output_dir / "comprehensive_uncertainty_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Comprehensive results saved to: {output_file}")
        
        # Save summary report separately
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results.summary_report, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to: {summary_file}")


def create_uncertainty_evaluator(
    k_shot_values: List[int] = None,
    output_dir: str = "uncertainty_evaluation",
    **kwargs
) -> UncertaintyEvaluatorOrchestrator:
    """
    Factory function to create uncertainty evaluator with default settings.
    
    Args:
        k_shot_values: List of K values for evaluation
        output_dir: Output directory for results
        **kwargs: Additional arguments for UncertaintyEvaluatorOrchestrator
        
    Returns:
        Configured UncertaintyEvaluatorOrchestrator instance
    """
    return UncertaintyEvaluatorOrchestrator(
        k_shot_values=k_shot_values,
        output_dir=output_dir,
        **kwargs
    )