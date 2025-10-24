"""
Ablation study framework for Bayesian uncertainty quantification.

This module implements comprehensive ablation studies to understand the impact of different
design choices in BayesianMetaPINN, including prior types, variational families, and
optimization hyperparameters.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import yaml
import logging
from itertools import product
import copy

from .base import UncertaintyMetaLearner, UncertaintyPrediction
from .bayesian_meta_pinn import BayesianMetaPINN
from .physics_priors import PhysicsInformedPrior, StandardPrior, LaplacePrior
from .variational_layers import VariationalLinear, FullCovarianceVariationalLinear
from .calibration_metrics import CalibrationMetrics
from .noisy_problems import NoisyProblemGenerator
from .config import UncertaintyConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from a single ablation study run."""
    study_type: str
    configuration: Dict[str, Any]
    ece: float
    mce: float
    coverage: float
    sharpness: float
    crps: float
    elbo_final: float
    convergence_iterations: int
    training_time: float


class PriorTypeAblation:
    """Ablation study comparing different prior types."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize prior type ablation study."""
        self.base_config = base_config
        self.prior_types = {
            'standard': self._create_standard_prior,
            'physics_informed': self._create_physics_informed_prior,
            'laplace': self._create_laplace_prior
        }
        self.results: List[AblationResult] = []
    
    def run_ablation(self, pde_type: str = 'heat', noise_level: float = 0.05, 
                    num_test_tasks: int = 20) -> List[AblationResult]:
        """Run prior type ablation study."""
        logger.info(f"Running prior type ablation for {pde_type} PDE with noise={noise_level}")
        
        problem_generator = NoisyProblemGenerator(pde_type=pde_type)
        
        for prior_name, prior_factory in self.prior_types.items():
            logger.info(f"Testing {prior_name} prior")
            
            # Create model with specific prior
            config = copy.deepcopy(self.base_config)
            config['prior_type'] = prior_name
            
            model = self._create_model_with_prior(config, prior_factory)
            
            # Run evaluation
            result = self._evaluate_model(
                model, prior_name, problem_generator, 
                noise_level, num_test_tasks
            )
            
            self.results.append(result)
        
        return self.results
    
    def _create_standard_prior(self, input_dim: int, output_dim: int) -> StandardPrior:
        """Create standard Gaussian prior."""
        return StandardPrior(
            mean=0.0,
            std=1.0,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def _create_physics_informed_prior(self, input_dim: int, output_dim: int) -> PhysicsInformedPrior:
        """Create physics-informed prior."""
        return PhysicsInformedPrior(
            pde_type=self.base_config.get('pde_type', 'heat'),
            input_dim=input_dim,
            output_dim=output_dim,
            boundary_weight=1.0,
            symmetry_weight=0.5
        )
    
    def _create_laplace_prior(self, input_dim: int, output_dim: int) -> LaplacePrior:
        """Create Laplace (L1) prior for sparsity."""
        return LaplacePrior(
            location=0.0,
            scale=1.0,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def _create_model_with_prior(self, config: Dict[str, Any], 
                                prior_factory: Callable) -> BayesianMetaPINN:
        """Create BayesianMetaPINN with specific prior."""
        model = BayesianMetaPINN(config)
        
        # Replace priors in all variational layers
        for layer in model.network:
            if isinstance(layer, VariationalLinear):
                layer.prior = prior_factory(layer.in_features, layer.out_features)
        
        return model
    
    def _evaluate_model(self, model: BayesianMetaPINN, prior_name: str,
                       problem_generator: NoisyProblemGenerator,
                       noise_level: float, num_test_tasks: int) -> AblationResult:
        """Evaluate model performance for ablation study."""
        # Meta-train model
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            cpu_start_time = time.time()
        
        task_distribution = problem_generator.get_task_distribution()
        training_results = model.meta_train(task_distribution, num_iterations=500)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            training_time = time.time() - cpu_start_time
        
        # Evaluate on test tasks
        ece_scores = []
        mce_scores = []
        coverage_scores = []
        sharpness_scores = []
        crps_scores = []
        
        calibration_metrics = CalibrationMetrics()
        
        for task_id in range(num_test_tasks):
            # Generate test problem
            problem = problem_generator.generate_noisy_problem(
                noise_type='gaussian',
                noise_level=noise_level,
                seed=42 + task_id
            )
            
            # Sample data
            support_data, support_targets = problem.sample_support(k=5)
            query_data, query_targets = problem.sample_query(50)
            
            # Adapt and predict
            adapted_model = model.adapt(support_data, support_targets)
            predictions = adapted_model.predict_with_uncertainty(query_data)
            
            # Compute metrics
            ece = calibration_metrics.expected_calibration_error(predictions, query_targets)
            mce = calibration_metrics.maximum_calibration_error(predictions, query_targets)
            coverage_results = calibration_metrics.coverage_analysis(predictions, query_targets)
            crps = calibration_metrics.continuous_ranked_probability_score(predictions, query_targets)
            
            ece_scores.append(ece)
            mce_scores.append(mce)
            coverage_scores.append(coverage_results['coverage'])
            sharpness_scores.append(coverage_results['sharpness'])
            crps_scores.append(crps)
        
        return AblationResult(
            study_type='prior_type',
            configuration={'prior_type': prior_name},
            ece=np.mean(ece_scores),
            mce=np.mean(mce_scores),
            coverage=np.mean(coverage_scores),
            sharpness=np.mean(sharpness_scores),
            crps=np.mean(crps_scores),
            elbo_final=training_results.get('final_elbo', 0.0),
            convergence_iterations=training_results.get('convergence_iterations', 500),
            training_time=training_time
        )


class VariationalFamilyAblation:
    """Ablation study comparing different variational families."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize variational family ablation study."""
        self.base_config = base_config
        self.variational_families = {
            'diagonal_gaussian': self._create_diagonal_model,
            'full_covariance': self._create_full_covariance_model,
            'structured_covariance': self._create_structured_covariance_model
        }
        self.results: List[AblationResult] = []
    
    def run_ablation(self, pde_type: str = 'heat', noise_level: float = 0.05,
                    num_test_tasks: int = 20) -> List[AblationResult]:
        """Run variational family ablation study."""
        logger.info(f"Running variational family ablation for {pde_type} PDE")
        
        problem_generator = NoisyProblemGenerator(pde_type=pde_type)
        
        for family_name, model_factory in self.variational_families.items():
            logger.info(f"Testing {family_name} variational family")
            
            try:
                # Create model with specific variational family
                config = copy.deepcopy(self.base_config)
                config['variational_family'] = family_name
                
                model = model_factory(config)
                
                # Run evaluation
                result = self._evaluate_model(
                    model, family_name, problem_generator,
                    noise_level, num_test_tasks
                )
                
                self.results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {family_name}: {e}")
                continue
        
        return self.results
    
    def _create_diagonal_model(self, config: Dict[str, Any]) -> BayesianMetaPINN:
        """Create model with diagonal Gaussian variational family."""
        config['variational_family'] = 'diagonal_gaussian'
        return BayesianMetaPINN(config)
    
    def _create_full_covariance_model(self, config: Dict[str, Any]) -> BayesianMetaPINN:
        """Create model with full covariance Gaussian variational family."""
        # Create base model
        model = BayesianMetaPINN(config)
        
        # Replace variational layers with full covariance versions
        new_layers = []
        for layer in model.network:
            if isinstance(layer, VariationalLinear):
                new_layer = FullCovarianceVariationalLinear(
                    layer.in_features, 
                    layer.out_features,
                    prior=layer.prior
                )
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)
        
        model.network = torch.nn.Sequential(*new_layers)
        return model
    
    def _create_structured_covariance_model(self, config: Dict[str, Any]) -> BayesianMetaPINN:
        """Create model with structured covariance (block diagonal)."""
        # For now, use diagonal as placeholder for structured covariance
        # In practice, this would implement block-diagonal or other structured forms
        logger.warning("Structured covariance not fully implemented, using diagonal")
        return self._create_diagonal_model(config)
    
    def _evaluate_model(self, model: BayesianMetaPINN, family_name: str,
                       problem_generator: NoisyProblemGenerator,
                       noise_level: float, num_test_tasks: int) -> AblationResult:
        """Evaluate model performance for variational family ablation."""
        # Similar evaluation as in PriorTypeAblation
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        else:
            import time
            cpu_start_time = time.time()
        
        task_distribution = problem_generator.get_task_distribution()
        training_results = model.meta_train(task_distribution, num_iterations=300)  # Reduced for full covariance
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            training_time = time.time() - cpu_start_time
        
        # Evaluate on test tasks
        ece_scores = []
        mce_scores = []
        coverage_scores = []
        sharpness_scores = []
        crps_scores = []
        
        calibration_metrics = CalibrationMetrics()
        
        for task_id in range(num_test_tasks):
            problem = problem_generator.generate_noisy_problem(
                noise_type='gaussian',
                noise_level=noise_level,
                seed=42 + task_id
            )
            
            support_data, support_targets = problem.sample_support(k=5)
            query_data, query_targets = problem.sample_query(50)
            
            adapted_model = model.adapt(support_data, support_targets)
            predictions = adapted_model.predict_with_uncertainty(query_data)
            
            ece = calibration_metrics.expected_calibration_error(predictions, query_targets)
            mce = calibration_metrics.maximum_calibration_error(predictions, query_targets)
            coverage_results = calibration_metrics.coverage_analysis(predictions, query_targets)
            crps = calibration_metrics.continuous_ranked_probability_score(predictions, query_targets)
            
            ece_scores.append(ece)
            mce_scores.append(mce)
            coverage_scores.append(coverage_results['coverage'])
            sharpness_scores.append(coverage_results['sharpness'])
            crps_scores.append(crps)
        
        return AblationResult(
            study_type='variational_family',
            configuration={'variational_family': family_name},
            ece=np.mean(ece_scores),
            mce=np.mean(mce_scores),
            coverage=np.mean(coverage_scores),
            sharpness=np.mean(sharpness_scores),
            crps=np.mean(crps_scores),
            elbo_final=training_results.get('final_elbo', 0.0),
            convergence_iterations=training_results.get('convergence_iterations', 300),
            training_time=training_time
        )


class HyperparameterAblation:
    """Ablation study for KL weight and temperature scaling."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize hyperparameter ablation study."""
        self.base_config = base_config
        self.results: List[AblationResult] = []
    
    def run_kl_weight_ablation(self, kl_weights: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                              pde_type: str = 'heat', noise_level: float = 0.05,
                              num_test_tasks: int = 15) -> List[AblationResult]:
        """Run KL weight ablation study."""
        logger.info(f"Running KL weight ablation with weights: {kl_weights}")
        
        problem_generator = NoisyProblemGenerator(pde_type=pde_type)
        
        for kl_weight in kl_weights:
            logger.info(f"Testing KL weight: {kl_weight}")
            
            config = copy.deepcopy(self.base_config)
            config['kl_weight'] = kl_weight
            
            model = BayesianMetaPINN(config)
            
            result = self._evaluate_hyperparameter(
                model, 'kl_weight', kl_weight, problem_generator,
                noise_level, num_test_tasks
            )
            
            self.results.append(result)
        
        return self.results
    
    def run_temperature_scaling_ablation(self, temperatures: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
                                       pde_type: str = 'heat', noise_level: float = 0.05,
                                       num_test_tasks: int = 15) -> List[AblationResult]:
        """Run temperature scaling ablation study."""
        logger.info(f"Running temperature scaling ablation with temperatures: {temperatures}")
        
        problem_generator = NoisyProblemGenerator(pde_type=pde_type)
        
        for temperature in temperatures:
            logger.info(f"Testing temperature: {temperature}")
            
            config = copy.deepcopy(self.base_config)
            config['temperature'] = temperature
            
            model = BayesianMetaPINN(config)
            
            result = self._evaluate_hyperparameter(
                model, 'temperature', temperature, problem_generator,
                noise_level, num_test_tasks
            )
            
            self.results.append(result)
        
        return self.results
    
    def _evaluate_hyperparameter(self, model: BayesianMetaPINN, param_name: str, param_value: float,
                                problem_generator: NoisyProblemGenerator,
                                noise_level: float, num_test_tasks: int) -> AblationResult:
        """Evaluate model with specific hyperparameter setting."""
        # Meta-train model
        task_distribution = problem_generator.get_task_distribution()
        training_results = model.meta_train(task_distribution, num_iterations=400)
        
        # Evaluate on test tasks
        ece_scores = []
        mce_scores = []
        coverage_scores = []
        sharpness_scores = []
        crps_scores = []
        
        calibration_metrics = CalibrationMetrics()
        
        for task_id in range(num_test_tasks):
            problem = problem_generator.generate_noisy_problem(
                noise_type='gaussian',
                noise_level=noise_level,
                seed=42 + task_id
            )
            
            support_data, support_targets = problem.sample_support(k=5)
            query_data, query_targets = problem.sample_query(50)
            
            adapted_model = model.adapt(support_data, support_targets)
            predictions = adapted_model.predict_with_uncertainty(query_data)
            
            ece = calibration_metrics.expected_calibration_error(predictions, query_targets)
            mce = calibration_metrics.maximum_calibration_error(predictions, query_targets)
            coverage_results = calibration_metrics.coverage_analysis(predictions, query_targets)
            crps = calibration_metrics.continuous_ranked_probability_score(predictions, query_targets)
            
            ece_scores.append(ece)
            mce_scores.append(mce)
            coverage_scores.append(coverage_results['coverage'])
            sharpness_scores.append(coverage_results['sharpness'])
            crps_scores.append(crps)
        
        return AblationResult(
            study_type=param_name,
            configuration={param_name: param_value},
            ece=np.mean(ece_scores),
            mce=np.mean(mce_scores),
            coverage=np.mean(coverage_scores),
            sharpness=np.mean(sharpness_scores),
            crps=np.mean(crps_scores),
            elbo_final=training_results.get('final_elbo', 0.0),
            convergence_iterations=training_results.get('convergence_iterations', 400),
            training_time=0.0  # Not measured for hyperparameter ablation
        )


class ComprehensiveAblationStudy:
    """Comprehensive ablation study framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comprehensive ablation study."""
        self.config = config
        self.all_results: List[AblationResult] = []
        
        # Initialize individual ablation studies
        self.prior_ablation = PriorTypeAblation(config)
        self.variational_ablation = VariationalFamilyAblation(config)
        self.hyperparameter_ablation = HyperparameterAblation(config)
    
    def run_all_ablations(self, pde_type: str = 'heat', noise_level: float = 0.05) -> Dict[str, List[AblationResult]]:
        """Run all ablation studies."""
        logger.info("Starting comprehensive ablation study...")
        
        results = {}
        
        # Prior type ablation
        logger.info("Running prior type ablation...")
        prior_results = self.prior_ablation.run_ablation(pde_type, noise_level, num_test_tasks=15)
        results['prior_type'] = prior_results
        self.all_results.extend(prior_results)
        
        # Variational family ablation
        logger.info("Running variational family ablation...")
        variational_results = self.variational_ablation.run_ablation(pde_type, noise_level, num_test_tasks=15)
        results['variational_family'] = variational_results
        self.all_results.extend(variational_results)
        
        # KL weight ablation
        logger.info("Running KL weight ablation...")
        kl_results = self.hyperparameter_ablation.run_kl_weight_ablation(
            kl_weights=[0.1, 0.5, 1.0, 2.0], pde_type=pde_type, noise_level=noise_level, num_test_tasks=10
        )
        results['kl_weight'] = kl_results
        self.all_results.extend(kl_results)
        
        # Temperature scaling ablation
        logger.info("Running temperature scaling ablation...")
        temp_results = self.hyperparameter_ablation.run_temperature_scaling_ablation(
            temperatures=[0.5, 1.0, 1.5, 2.0], pde_type=pde_type, noise_level=noise_level, num_test_tasks=10
        )
        results['temperature'] = temp_results
        self.all_results.extend(temp_results)
        
        logger.info(f"Completed comprehensive ablation study with {len(self.all_results)} total experiments")
        return results
    
    def save_results(self, results: Dict[str, List[AblationResult]], output_dir: Path):
        """Save ablation study results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual ablation results
        for study_type, study_results in results.items():
            df = pd.DataFrame([
                {
                    'study_type': r.study_type,
                    'configuration': str(r.configuration),
                    'ece': r.ece,
                    'mce': r.mce,
                    'coverage': r.coverage,
                    'sharpness': r.sharpness,
                    'crps': r.crps,
                    'elbo_final': r.elbo_final,
                    'convergence_iterations': r.convergence_iterations,
                    'training_time': r.training_time
                }
                for r in study_results
            ])
            df.to_csv(output_dir / f'{study_type}_ablation_results.csv', index=False)
        
        # Save combined results
        combined_df = pd.DataFrame([
            {
                'study_type': r.study_type,
                'configuration': str(r.configuration),
                'ece': r.ece,
                'mce': r.mce,
                'coverage': r.coverage,
                'sharpness': r.sharpness,
                'crps': r.crps,
                'elbo_final': r.elbo_final,
                'convergence_iterations': r.convergence_iterations,
                'training_time': r.training_time
            }
            for r in self.all_results
        ])
        combined_df.to_csv(output_dir / 'comprehensive_ablation_results.csv', index=False)
        
        # Save configuration
        with open(output_dir / 'ablation_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Ablation study results saved to {output_dir}")
    
    def generate_summary(self, results: Dict[str, List[AblationResult]]) -> Dict[str, Any]:
        """Generate summary of ablation study results."""
        summary = {}
        
        for study_type, study_results in results.items():
            if not study_results:
                continue
            
            # Find best configuration for each metric
            best_ece = min(study_results, key=lambda x: x.ece)
            best_coverage = min(study_results, key=lambda x: abs(x.coverage - 0.95))
            
            summary[study_type] = {
                'num_configurations': len(study_results),
                'best_ece': {
                    'configuration': best_ece.configuration,
                    'ece': best_ece.ece,
                    'coverage': best_ece.coverage
                },
                'best_coverage': {
                    'configuration': best_coverage.configuration,
                    'ece': best_coverage.ece,
                    'coverage': best_coverage.coverage
                },
                'mean_ece': np.mean([r.ece for r in study_results]),
                'std_ece': np.std([r.ece for r in study_results]),
                'mean_coverage': np.mean([r.coverage for r in study_results]),
                'std_coverage': np.std([r.coverage for r in study_results])
            }
        
        return summary


def run_ablation_studies(config_path: Optional[str] = None) -> Tuple[Dict[str, List[AblationResult]], Dict[str, Any]]:
    """Run comprehensive ablation studies."""
    # Load configuration
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = UncertaintyConfig.get_default_config()
        config.update({
            'pde_type': 'heat',
            'noise_level': 0.05,
            'random_seed': 42
        })
    
    # Run ablation studies
    ablation_study = ComprehensiveAblationStudy(config)
    results = ablation_study.run_all_ablations()
    
    # Generate summary
    summary = ablation_study.generate_summary(results)
    
    # Save results
    output_dir = Path('results/ablation_studies')
    ablation_study.save_results(results, output_dir)
    
    return results, summary


if __name__ == "__main__":
    # Run ablation studies
    results, summary = run_ablation_studies()
    
    print("Ablation Study Summary:")
    print("=" * 40)
    
    for study_type, study_summary in summary.items():
        print(f"\n{study_type.upper()} ABLATION:")
        print(f"  Configurations tested: {study_summary['num_configurations']}")
        print(f"  Best ECE: {study_summary['best_ece']['ece']:.4f} with {study_summary['best_ece']['configuration']}")
        print(f"  Best coverage: {study_summary['best_coverage']['coverage']:.4f} with {study_summary['best_coverage']['configuration']}")
        print(f"  Mean ECE: {study_summary['mean_ece']:.4f} ± {study_summary['std_ece']:.4f}")