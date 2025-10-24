"""
Comprehensive experimental validation framework for Bayesian uncertainty quantification.

This module implements the main calibration comparison experiment across multiple PDE types,
noise levels, and uncertainty quantification methods with statistical significance testing.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging
from scipy import stats
from itertools import product
import warnings

from .base import UncertaintyMetaLearner, UncertaintyPrediction
from .bayesian_meta_pinn import BayesianMetaPINN
from .ensemble_meta_pinn import EnsembleMetaPINN
from .mc_dropout_meta_pinn import MCDropoutMetaPINN
from .calibration_metrics import CalibrationMetrics
from .noisy_problems import NoisyProblemGenerator
from .config import UncertaintyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experimental run."""
    method: str
    pde_type: str
    noise_level: float
    k_shot: int
    task_id: int
    ece: float
    mce: float
    coverage: float
    sharpness: float
    crps: float
    inference_time: float
    memory_usage: float


@dataclass
class StatisticalTestResult:
    """Results from statistical significance testing."""
    method_a: str
    method_b: str
    metric: str
    t_statistic: float
    p_value: float
    p_value_corrected: float
    cohens_d: float
    significant: bool
    effect_size_interpretation: str


class CalibrationComparisonExperiment:
    """Main calibration comparison experiment framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment with configuration."""
        self.config = config
        self.results: List[ExperimentResult] = []
        self.statistical_results: List[StatisticalTestResult] = []
        
        # Experiment parameters
        self.pde_types = config.get('pde_types', ['heat', 'burgers', 'poisson', 'navier_stokes', 'reaction_diffusion'])
        self.noise_levels = config.get('noise_levels', [0.01, 0.05, 0.1])
        self.k_shot_values = config.get('k_shot_values', [1, 5, 10, 25])
        self.num_test_tasks = config.get('num_test_tasks', 50)
        self.num_posterior_samples = config.get('num_posterior_samples', 100)
        self.random_seed = config.get('random_seed', 42)
        
        # Methods to compare
        self.methods = {
            'bayesian': self._create_bayesian_model,
            'ensemble': self._create_ensemble_model,
            'mc_dropout': self._create_mc_dropout_model
        }
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info(f"Initialized calibration comparison experiment with {len(self.pde_types)} PDE types, "
                   f"{len(self.noise_levels)} noise levels, {len(self.k_shot_values)} K-shot values")
    
    def run_experiment(self) -> pd.DataFrame:
        """Run the complete calibration comparison experiment."""
        logger.info("Starting comprehensive calibration comparison experiment...")
        
        # Generate all experimental conditions
        conditions = list(product(
            self.methods.keys(),
            self.pde_types,
            self.noise_levels,
            self.k_shot_values
        ))
        
        total_experiments = len(conditions) * self.num_test_tasks
        logger.info(f"Running {total_experiments} individual experiments...")
        
        experiment_count = 0
        
        for method_name, pde_type, noise_level, k_shot in conditions:
            logger.info(f"Running {method_name} on {pde_type} PDE with noise={noise_level}, K={k_shot}")
            
            # Create model for this condition
            model = self.methods[method_name](pde_type)
            
            # Generate noisy problems for this PDE type and noise level
            problem_generator = NoisyProblemGenerator(pde_type=pde_type)
            
            # Run experiments on multiple test tasks
            for task_id in range(self.num_test_tasks):
                try:
                    result = self._run_single_experiment(
                        model, method_name, pde_type, noise_level, k_shot, 
                        task_id, problem_generator
                    )
                    self.results.append(result)
                    
                    experiment_count += 1
                    if experiment_count % 50 == 0:
                        logger.info(f"Completed {experiment_count}/{total_experiments} experiments")
                        
                except Exception as e:
                    logger.warning(f"Experiment failed for {method_name}-{pde_type}-{noise_level}-{k_shot}-{task_id}: {e}")
                    continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'method': r.method,
                'pde_type': r.pde_type,
                'noise_level': r.noise_level,
                'k_shot': r.k_shot,
                'task_id': r.task_id,
                'ece': r.ece,
                'mce': r.mce,
                'coverage': r.coverage,
                'sharpness': r.sharpness,
                'crps': r.crps,
                'inference_time': r.inference_time,
                'memory_usage': r.memory_usage
            }
            for r in self.results
        ])
        
        logger.info(f"Experiment completed. Collected {len(results_df)} valid results.")
        return results_df
    
    def _run_single_experiment(self, model: UncertaintyMetaLearner, method_name: str,
                              pde_type: str, noise_level: float, k_shot: int,
                              task_id: int, problem_generator: NoisyProblemGenerator) -> ExperimentResult:
        """Run a single experimental condition."""
        # Generate noisy problem
        problem = problem_generator.generate_noisy_problem(
            noise_type='gaussian',
            noise_level=noise_level,
            seed=self.random_seed + task_id
        )
        
        # Sample support and query data
        support_data, support_targets = problem.sample_support(k_shot)
        query_data, query_targets = problem.sample_query(100)  # Fixed query size
        
        # Meta-train model (simplified for experiment)
        if not hasattr(model, '_is_meta_trained'):
            # Quick meta-training for experiment
            model.meta_train(problem_generator.get_task_distribution(), num_iterations=100)
            model._is_meta_trained = True
        
        # Adapt to support data
        adapted_model = model.adapt(support_data, support_targets, num_steps=10)
        
        # Measure inference time and memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            cpu_start_time = time.time()
        
        # Get predictions with uncertainty
        predictions = adapted_model.predict_with_uncertainty(
            query_data, num_samples=self.num_posterior_samples
        )
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # milliseconds
            memory_usage = torch.cuda.memory_allocated() - start_memory
        else:
            inference_time = (time.time() - cpu_start_time) * 1000  # convert to ms
            memory_usage = 0  # Cannot measure CPU memory easily
        
        # Compute calibration metrics
        calibration_metrics = CalibrationMetrics()
        
        ece = calibration_metrics.expected_calibration_error(predictions, query_targets)
        mce = calibration_metrics.maximum_calibration_error(predictions, query_targets)
        coverage_results = calibration_metrics.coverage_analysis(predictions, query_targets)
        crps = calibration_metrics.continuous_ranked_probability_score(predictions, query_targets)
        
        return ExperimentResult(
            method=method_name,
            pde_type=pde_type,
            noise_level=noise_level,
            k_shot=k_shot,
            task_id=task_id,
            ece=ece,
            mce=mce,
            coverage=coverage_results['coverage'],
            sharpness=coverage_results['sharpness'],
            crps=crps,
            inference_time=inference_time,
            memory_usage=memory_usage
        )
    
    def perform_statistical_analysis(self, results_df: pd.DataFrame) -> List[StatisticalTestResult]:
        """Perform statistical significance testing with Bonferroni correction."""
        logger.info("Performing statistical significance testing...")
        
        metrics = ['ece', 'mce', 'coverage', 'sharpness', 'crps']
        methods = results_df['method'].unique()
        method_pairs = [(m1, m2) for i, m1 in enumerate(methods) for m2 in methods[i+1:]]
        
        statistical_results = []
        
        for metric in metrics:
            logger.info(f"Testing significance for {metric}")
            
            for method_a, method_b in method_pairs:
                # Get data for both methods
                data_a = results_df[results_df['method'] == method_a][metric].values
                data_b = results_df[results_df['method'] == method_b][metric].values
                
                # Ensure we have paired data (same experimental conditions)
                if len(data_a) != len(data_b):
                    logger.warning(f"Unequal sample sizes for {method_a} vs {method_b} on {metric}")
                    continue
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(data_a, data_b)
                
                # Compute Cohen's d for effect size
                cohens_d = self._compute_cohens_d(data_a, data_b)
                effect_size_interpretation = self._interpret_effect_size(cohens_d)
                
                statistical_results.append(StatisticalTestResult(
                    method_a=method_a,
                    method_b=method_b,
                    metric=metric,
                    t_statistic=t_stat,
                    p_value=p_value,
                    p_value_corrected=0.0,  # Will be filled after Bonferroni correction
                    cohens_d=cohens_d,
                    significant=False,  # Will be updated after correction
                    effect_size_interpretation=effect_size_interpretation
                ))
        
        # Apply Bonferroni correction
        p_values = [r.p_value for r in statistical_results]
        corrected_p_values = self._bonferroni_correction(p_values)
        
        for i, result in enumerate(statistical_results):
            result.p_value_corrected = corrected_p_values[i]
            result.significant = corrected_p_values[i] < 0.05
        
        self.statistical_results = statistical_results
        
        # Log significant results
        significant_results = [r for r in statistical_results if r.significant]
        logger.info(f"Found {len(significant_results)} statistically significant differences")
        
        for result in significant_results:
            logger.info(f"{result.method_a} vs {result.method_b} on {result.metric}: "
                       f"p={result.p_value_corrected:.4f}, d={result.cohens_d:.3f} ({result.effect_size_interpretation})")
        
        return statistical_results
    
    def _compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons."""
        n_comparisons = len(p_values)
        return [min(p * n_comparisons, 1.0) for p in p_values]
    
    def _create_bayesian_model(self, pde_type: str) -> BayesianMetaPINN:
        """Create BayesianMetaPINN model for given PDE type."""
        config = UncertaintyConfig.get_default_config()
        config['pde_type'] = pde_type
        return BayesianMetaPINN(config)
    
    def _create_ensemble_model(self, pde_type: str) -> EnsembleMetaPINN:
        """Create EnsembleMetaPINN model for given PDE type."""
        config = UncertaintyConfig.get_default_config()
        config['pde_type'] = pde_type
        config['num_models'] = 5  # Smaller ensemble for faster experiments
        return EnsembleMetaPINN(config)
    
    def _create_mc_dropout_model(self, pde_type: str) -> MCDropoutMetaPINN:
        """Create MCDropoutMetaPINN model for given PDE type."""
        config = UncertaintyConfig.get_default_config()
        config['pde_type'] = pde_type
        config['dropout_rate'] = 0.1
        return MCDropoutMetaPINN(config)
    
    def save_results(self, results_df: pd.DataFrame, output_dir: Path):
        """Save experimental results and statistical analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_df.to_csv(output_dir / 'calibration_experiment_results.csv', index=False)
        
        # Save statistical analysis
        if self.statistical_results:
            stats_df = pd.DataFrame([
                {
                    'method_a': r.method_a,
                    'method_b': r.method_b,
                    'metric': r.metric,
                    't_statistic': r.t_statistic,
                    'p_value': r.p_value,
                    'p_value_corrected': r.p_value_corrected,
                    'cohens_d': r.cohens_d,
                    'significant': r.significant,
                    'effect_size': r.effect_size_interpretation
                }
                for r in self.statistical_results
            ])
            stats_df.to_csv(output_dir / 'statistical_analysis_results.csv', index=False)
        
        # Save experiment configuration
        with open(output_dir / 'experiment_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary report of experimental results."""
        summary = {}
        
        # Overall statistics
        summary['total_experiments'] = len(results_df)
        summary['methods_tested'] = results_df['method'].unique().tolist()
        summary['pde_types_tested'] = results_df['pde_type'].unique().tolist()
        summary['noise_levels_tested'] = sorted(results_df['noise_level'].unique().tolist())
        
        # Performance by method
        method_performance = {}
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            method_performance[method] = {
                'mean_ece': float(method_data['ece'].mean()),
                'std_ece': float(method_data['ece'].std()),
                'mean_coverage': float(method_data['coverage'].mean()),
                'std_coverage': float(method_data['coverage'].std()),
                'mean_inference_time': float(method_data['inference_time'].mean()),
                'std_inference_time': float(method_data['inference_time'].std()),
                'target_ece_achieved': float((method_data['ece'] < 0.05).mean()),
                'target_coverage_achieved': float(((method_data['coverage'] >= 0.93) & 
                                                 (method_data['coverage'] <= 0.97)).mean())
            }
        
        summary['method_performance'] = method_performance
        
        # Best performing method by metric
        summary['best_methods'] = {
            'lowest_ece': results_df.groupby('method')['ece'].mean().idxmin(),
            'best_coverage': results_df.groupby('method').apply(
                lambda x: abs(x['coverage'].mean() - 0.95)
            ).idxmin(),
            'fastest_inference': results_df.groupby('method')['inference_time'].mean().idxmin()
        }
        
        # Significant improvements
        if self.statistical_results:
            significant_improvements = []
            for result in self.statistical_results:
                if result.significant and result.metric == 'ece' and result.cohens_d < -0.5:  # Large improvement in ECE
                    significant_improvements.append({
                        'superior_method': result.method_a if result.cohens_d < 0 else result.method_b,
                        'inferior_method': result.method_b if result.cohens_d < 0 else result.method_a,
                        'effect_size': abs(result.cohens_d),
                        'p_value': result.p_value_corrected
                    })
            summary['significant_improvements'] = significant_improvements
        
        return summary


def run_calibration_experiment(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the main calibration comparison experiment."""
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'pde_types': ['heat', 'burgers', 'poisson'],  # Reduced for faster testing
            'noise_levels': [0.01, 0.05, 0.1],
            'k_shot_values': [1, 5, 10],
            'num_test_tasks': 20,  # Reduced for faster testing
            'num_posterior_samples': 50,  # Reduced for faster testing
            'random_seed': 42
        }
    
    # Run experiment
    experiment = CalibrationComparisonExperiment(config)
    results_df = experiment.run_experiment()
    
    # Perform statistical analysis
    experiment.perform_statistical_analysis(results_df)
    
    # Generate summary
    summary = experiment.generate_summary_report(results_df)
    
    # Save results
    output_dir = Path('results/calibration_experiment')
    experiment.save_results(results_df, output_dir)
    
    return results_df, summary


if __name__ == "__main__":
    # Run the experiment
    results, summary = run_calibration_experiment()
    
    print("Calibration Comparison Experiment Summary:")
    print("=" * 50)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Methods tested: {summary['methods_tested']}")
    print(f"Best ECE method: {summary['best_methods']['lowest_ece']}")
    print(f"Best coverage method: {summary['best_methods']['best_coverage']}")
    print(f"Fastest method: {summary['best_methods']['fastest_inference']}")
    
    if 'significant_improvements' in summary:
        print(f"\nSignificant improvements found: {len(summary['significant_improvements'])}")
        for improvement in summary['significant_improvements']:
            print(f"  {improvement['superior_method']} > {improvement['inferior_method']} "
                  f"(d={improvement['effect_size']:.3f}, p={improvement['p_value']:.4f})")