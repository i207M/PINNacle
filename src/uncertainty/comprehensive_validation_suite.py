"""
Comprehensive validation suite for Bayesian uncertainty quantification.

This module orchestrates all experimental validation components to run the complete
experimental validation suite, generate comprehensive results with statistical analysis,
and create publication-ready figures and tables.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import yaml
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .experimental_validation import CalibrationComparisonExperiment, run_calibration_experiment
from .ablation_studies import AblationStudyFramework
from .performance_benchmarking import PerformanceBenchmarkingSuite
from .ood_detection import OODDetectionEvaluator
from .decomposition_validator import UncertaintyDecompositionValidator
from .visualization_tools import create_calibration_plots, create_performance_plots, create_ood_plots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveValidationSuite:
    """Complete experimental validation suite for Bayesian uncertainty quantification."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the comprehensive validation suite."""
        self.config = self._load_config(config_path)
        self.results = {}
        self.figures = {}
        self.tables = {}
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'results/comprehensive_validation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to file
        log_file = self.output_dir / 'validation_suite.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("Initialized Comprehensive Validation Suite")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Configuration: {self.config}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Comprehensive default configuration
            config = {
                'output_dir': 'results/comprehensive_validation',
                'random_seed': 42,
                
                # Main calibration experiment
                'calibration_experiment': {
                    'pde_types': ['heat', 'burgers', 'poisson', 'navier_stokes', 'reaction_diffusion'],
                    'noise_levels': [0.01, 0.05, 0.1, 0.2],
                    'k_shot_values': [1, 5, 10, 25],
                    'num_test_tasks': 50,
                    'num_posterior_samples': 100
                },
                
                # Ablation studies
                'ablation_studies': {
                    'prior_types': ['standard', 'physics_informed', 'laplace'],
                    'variational_families': ['diagonal', 'full_covariance'],
                    'kl_weights': [0.1, 0.5, 1.0, 2.0],
                    'temperature_scales': [0.5, 1.0, 1.5, 2.0]
                },
                
                # Performance benchmarking
                'performance_benchmarking': {
                    'batch_sizes': [1, 10, 50, 100],
                    'num_posterior_samples': [10, 50, 100, 200],
                    'memory_profiling': True,
                    'timing_iterations': 10
                },
                
                # OOD detection evaluation
                'ood_evaluation': {
                    'scenarios': ['spatial_extrapolation', 'interpolation_gap', 'parameter_shift', 'high_noise'],
                    'num_in_dist': 1000,
                    'num_ood': 1000
                },
                
                # Uncertainty decomposition validation
                'decomposition_validation': {
                    'k_values': [1, 2, 5, 10, 25, 50, 100],
                    'num_tasks': 20,
                    'num_query_points': 100
                }
            }
        
        return config
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run the complete experimental validation suite."""
        logger.info("Starting comprehensive experimental validation suite...")
        start_time = datetime.now()
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        try:
            # 1. Main calibration comparison experiment
            logger.info("Running main calibration comparison experiment...")
            self.results['calibration'] = self._run_calibration_experiment()
            
            # 2. Ablation studies
            logger.info("Running ablation studies...")
            self.results['ablation'] = self._run_ablation_studies()
            
            # 3. Performance benchmarking
            logger.info("Running performance benchmarking...")
            self.results['performance'] = self._run_performance_benchmarking()
            
            # 4. OOD detection evaluation
            logger.info("Running OOD detection evaluation...")
            self.results['ood'] = self._run_ood_evaluation()
            
            # 5. Uncertainty decomposition validation
            logger.info("Running uncertainty decomposition validation...")
            self.results['decomposition'] = self._run_decomposition_validation()
            
            # 6. Generate comprehensive analysis
            logger.info("Generating comprehensive analysis...")
            analysis = self._generate_comprehensive_analysis()
            
            # 7. Create publication-ready figures and tables
            logger.info("Creating publication-ready figures and tables...")
            self._create_publication_materials()
            
            # 8. Save all results
            logger.info("Saving all results...")
            self._save_all_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Comprehensive validation suite completed successfully!")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return {
                'results': self.results,
                'analysis': analysis,
                'duration': str(duration),
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            raise
    
    def _run_calibration_experiment(self) -> Dict[str, Any]:
        """Run the main calibration comparison experiment."""
        config = self.config['calibration_experiment']
        
        # Create temporary config file
        temp_config_path = self.output_dir / 'temp_calibration_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run experiment
        results_df, summary = run_calibration_experiment(str(temp_config_path))
        
        # Clean up temp file
        temp_config_path.unlink()
        
        return {
            'results_df': results_df,
            'summary': summary,
            'config': config
        }
    
    def _run_ablation_studies(self) -> Dict[str, Any]:
        """Run ablation studies."""
        config = self.config['ablation_studies']
        
        ablation_framework = AblationStudyFramework(config)
        
        # Run all ablation studies
        results = {}
        
        # Prior type ablation
        logger.info("Running prior type ablation study...")
        results['prior_types'] = ablation_framework.run_prior_ablation()
        
        # Variational family ablation
        logger.info("Running variational family ablation study...")
        results['variational_families'] = ablation_framework.run_variational_family_ablation()
        
        # KL weight ablation
        logger.info("Running KL weight ablation study...")
        results['kl_weights'] = ablation_framework.run_kl_weight_ablation()
        
        # Temperature scaling ablation
        logger.info("Running temperature scaling ablation study...")
        results['temperature_scales'] = ablation_framework.run_temperature_ablation()
        
        return results
    
    def _run_performance_benchmarking(self) -> Dict[str, Any]:
        """Run performance benchmarking."""
        config = self.config['performance_benchmarking']
        
        benchmark_suite = PerformanceBenchmarkingSuite(config)
        
        # Run comprehensive benchmarks
        results = benchmark_suite.run_comprehensive_benchmarks()
        
        return results
    
    def _run_ood_evaluation(self) -> Dict[str, Any]:
        """Run OOD detection evaluation."""
        config = self.config['ood_evaluation']
        
        ood_evaluator = OODDetectionEvaluator()
        
        results = {}
        for scenario in config['scenarios']:
            logger.info(f"Evaluating OOD detection for {scenario}...")
            results[scenario] = ood_evaluator.evaluate_all_methods(
                scenario=scenario,
                num_in_dist=config['num_in_dist'],
                num_ood=config['num_ood']
            )
        
        return results
    
    def _run_decomposition_validation(self) -> Dict[str, Any]:
        """Run uncertainty decomposition validation."""
        config = self.config['decomposition_validation']
        
        validator = UncertaintyDecompositionValidator()
        
        # Validate decomposition for all methods
        results = validator.validate_all_methods(
            k_values=config['k_values'],
            num_tasks=config['num_tasks'],
            num_query_points=config['num_query_points']
        )
        
        return results
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of all results."""
        analysis = {}
        
        # Calibration analysis
        if 'calibration' in self.results:
            analysis['calibration'] = self._analyze_calibration_results()
        
        # Performance analysis
        if 'performance' in self.results:
            analysis['performance'] = self._analyze_performance_results()
        
        # OOD analysis
        if 'ood' in self.results:
            analysis['ood'] = self._analyze_ood_results()
        
        # Decomposition analysis
        if 'decomposition' in self.results:
            analysis['decomposition'] = self._analyze_decomposition_results()
        
        # Overall conclusions
        analysis['conclusions'] = self._generate_conclusions()
        
        return analysis
    
    def _analyze_calibration_results(self) -> Dict[str, Any]:
        """Analyze calibration experiment results."""
        results_df = self.results['calibration']['results_df']
        
        analysis = {}
        
        # Target achievement rates
        analysis['target_achievement'] = {}
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method]
            analysis['target_achievement'][method] = {
                'ece_below_0_05': float((method_data['ece'] < 0.05).mean()),
                'coverage_in_range': float(((method_data['coverage'] >= 0.93) & 
                                          (method_data['coverage'] <= 0.97)).mean()),
                'mean_ece': float(method_data['ece'].mean()),
                'std_ece': float(method_data['ece'].std()),
                'mean_coverage': float(method_data['coverage'].mean()),
                'std_coverage': float(method_data['coverage'].std())
            }
        
        # Best performing method
        mean_ece_by_method = results_df.groupby('method')['ece'].mean()
        analysis['best_method'] = {
            'method': mean_ece_by_method.idxmin(),
            'ece': float(mean_ece_by_method.min())
        }
        
        # Performance by PDE type
        analysis['performance_by_pde'] = {}
        for pde_type in results_df['pde_type'].unique():
            pde_data = results_df[results_df['pde_type'] == pde_type]
            analysis['performance_by_pde'][pde_type] = {
                'best_method': pde_data.groupby('method')['ece'].mean().idxmin(),
                'mean_ece_by_method': pde_data.groupby('method')['ece'].mean().to_dict()
            }
        
        return analysis
    
    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance benchmarking results."""
        performance_results = self.results['performance']
        
        analysis = {}
        
        # Efficiency comparison
        if 'timing_results' in performance_results:
            timing_results = performance_results['timing_results']
            
            # Find fastest method
            mean_times = {method: np.mean(times) for method, times in timing_results.items()}
            analysis['fastest_method'] = min(mean_times, key=mean_times.get)
            analysis['timing_comparison'] = mean_times
            
            # Check if BayesianMetaPINN is 4-5x faster than ensemble
            if 'bayesian' in mean_times and 'ensemble' in mean_times:
                speedup = mean_times['ensemble'] / mean_times['bayesian']
                analysis['bayesian_speedup'] = {
                    'speedup_factor': speedup,
                    'target_achieved': 4.0 <= speedup <= 6.0  # Allow some margin
                }
        
        # Memory usage analysis
        if 'memory_results' in performance_results:
            memory_results = performance_results['memory_results']
            analysis['memory_comparison'] = memory_results
            
            # Check memory efficiency targets
            if 'bayesian' in memory_results:
                bayesian_memory = memory_results['bayesian']
                analysis['memory_efficiency'] = {
                    'bayesian_memory_gb': bayesian_memory / (1024**3),  # Convert to GB
                    'target_achieved': bayesian_memory < 5 * (1024**3)  # < 5 GB
                }
        
        return analysis
    
    def _analyze_ood_results(self) -> Dict[str, Any]:
        """Analyze OOD detection results."""
        ood_results = self.results['ood']
        
        analysis = {}
        
        # Performance by scenario
        for scenario, scenario_results in ood_results.items():
            analysis[scenario] = {}
            
            for method, metrics in scenario_results.items():
                analysis[scenario][method] = {
                    'auroc': metrics['auroc'],
                    'target_achieved': metrics['auroc'] > 0.90,
                    'aupr': metrics.get('aupr', 0.0),
                    'fpr_at_95_tpr': metrics.get('fpr_at_95_tpr', 1.0)
                }
        
        # Overall best method for OOD detection
        overall_auroc = {}
        for scenario, scenario_results in ood_results.items():
            for method, metrics in scenario_results.items():
                if method not in overall_auroc:
                    overall_auroc[method] = []
                overall_auroc[method].append(metrics['auroc'])
        
        mean_auroc = {method: np.mean(aurocs) for method, aurocs in overall_auroc.items()}
        analysis['best_ood_method'] = max(mean_auroc, key=mean_auroc.get)
        analysis['mean_auroc_by_method'] = mean_auroc
        
        return analysis
    
    def _analyze_decomposition_results(self) -> Dict[str, Any]:
        """Analyze uncertainty decomposition results."""
        decomposition_results = self.results['decomposition']
        
        analysis = {}
        
        for method, method_results in decomposition_results.items():
            analysis[method] = {
                'decomposition_valid': method_results.get('decomposition_valid', False),
                'epistemic_decreasing': method_results.get('epistemic_decreasing', False),
                'aleatoric_constant': method_results.get('aleatoric_constant', False),
                'epistemic_slope': method_results.get('epistemic_slope', 0.0),
                'epistemic_r_squared': method_results.get('epistemic_r_squared', 0.0),
                'aleatoric_cv': method_results.get('aleatoric_cv', 1.0)
            }
        
        # Find method with best decomposition
        valid_methods = [method for method, results in analysis.items() 
                        if results['decomposition_valid']]
        
        if valid_methods:
            analysis['best_decomposition_method'] = valid_methods[0]  # Could rank by additional criteria
        
        return analysis
    
    def _generate_conclusions(self) -> Dict[str, Any]:
        """Generate overall conclusions from all experiments."""
        conclusions = {}
        
        # Main findings
        conclusions['main_findings'] = []
        
        # Calibration conclusions
        if 'calibration' in self.results:
            calibration_analysis = self.results.get('analysis', {}).get('calibration', {})
            best_method = calibration_analysis.get('best_method', {})
            
            if best_method:
                conclusions['main_findings'].append(
                    f"Best calibration method: {best_method['method']} with ECE = {best_method['ece']:.4f}"
                )
                
                # Check if target is achieved
                target_achieved = best_method['ece'] < 0.05
                conclusions['main_findings'].append(
                    f"ECE < 0.05 target {'achieved' if target_achieved else 'not achieved'}"
                )
        
        # Performance conclusions
        if 'performance' in self.results:
            performance_analysis = self.results.get('analysis', {}).get('performance', {})
            
            if 'bayesian_speedup' in performance_analysis:
                speedup_info = performance_analysis['bayesian_speedup']
                conclusions['main_findings'].append(
                    f"BayesianMetaPINN speedup: {speedup_info['speedup_factor']:.1f}x faster than ensemble"
                )
                conclusions['main_findings'].append(
                    f"4-5x speedup target {'achieved' if speedup_info['target_achieved'] else 'not achieved'}"
                )
        
        # OOD conclusions
        if 'ood' in self.results:
            ood_analysis = self.results.get('analysis', {}).get('ood', {})
            
            if 'best_ood_method' in ood_analysis:
                best_ood_method = ood_analysis['best_ood_method']
                mean_auroc = ood_analysis['mean_auroc_by_method'][best_ood_method]
                conclusions['main_findings'].append(
                    f"Best OOD detection method: {best_ood_method} with mean AUROC = {mean_auroc:.3f}"
                )
                conclusions['main_findings'].append(
                    f"AUROC > 0.90 target {'achieved' if mean_auroc > 0.90 else 'not achieved'}"
                )
        
        # Overall recommendation
        conclusions['recommendation'] = self._generate_method_recommendation()
        
        return conclusions
    
    def _generate_method_recommendation(self) -> str:
        """Generate overall method recommendation based on all results."""
        # This is a simplified recommendation logic
        # In practice, you might want more sophisticated multi-criteria decision making
        
        scores = {'bayesian': 0, 'ensemble': 0, 'mc_dropout': 0}
        
        # Score based on calibration
        if 'calibration' in self.results:
            calibration_analysis = self.results.get('analysis', {}).get('calibration', {})
            best_method = calibration_analysis.get('best_method', {}).get('method')
            if best_method in scores:
                scores[best_method] += 3  # High weight for calibration
        
        # Score based on performance
        if 'performance' in self.results:
            performance_analysis = self.results.get('analysis', {}).get('performance', {})
            fastest_method = performance_analysis.get('fastest_method')
            if fastest_method in scores:
                scores[fastest_method] += 2  # Medium weight for performance
        
        # Score based on OOD detection
        if 'ood' in self.results:
            ood_analysis = self.results.get('analysis', {}).get('ood', {})
            best_ood_method = ood_analysis.get('best_ood_method')
            if best_ood_method in scores:
                scores[best_ood_method] += 2  # Medium weight for OOD
        
        # Score based on decomposition validity
        if 'decomposition' in self.results:
            decomposition_analysis = self.results.get('analysis', {}).get('decomposition', {})
            for method, results in decomposition_analysis.items():
                if method in scores and results.get('decomposition_valid', False):
                    scores[method] += 1  # Low weight for decomposition
        
        # Find best method
        best_method = max(scores, key=scores.get)
        
        return f"Recommended method: {best_method} (score: {scores[best_method]})"
    
    def _create_publication_materials(self):
        """Create publication-ready figures and tables."""
        # Create figures directory
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Create tables directory
        tables_dir = self.output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        # Generate calibration plots
        if 'calibration' in self.results:
            self._create_calibration_figures(figures_dir)
        
        # Generate performance plots
        if 'performance' in self.results:
            self._create_performance_figures(figures_dir)
        
        # Generate OOD plots
        if 'ood' in self.results:
            self._create_ood_figures(figures_dir)
        
        # Generate summary tables
        self._create_summary_tables(tables_dir)
    
    def _create_calibration_figures(self, figures_dir: Path):
        """Create calibration-related figures."""
        results_df = self.results['calibration']['results_df']
        
        # ECE comparison plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=results_df, x='method', y='ece')
        plt.axhline(y=0.05, color='red', linestyle='--', label='Target ECE < 0.05')
        plt.title('Expected Calibration Error by Method')
        plt.ylabel('Expected Calibration Error')
        plt.xlabel('Method')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / 'ece_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Coverage analysis plot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=results_df, x='method', y='coverage')
        plt.axhline(y=0.95, color='red', linestyle='--', label='Target Coverage = 0.95')
        plt.axhspan(0.93, 0.97, alpha=0.2, color='green', label='Acceptable Range')
        plt.title('Coverage Analysis by Method')
        plt.ylabel('Coverage')
        plt.xlabel('Method')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / 'coverage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance by PDE type heatmap
        pivot_data = results_df.groupby(['method', 'pde_type'])['ece'].mean().unstack()
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r')
        plt.title('Mean ECE by Method and PDE Type')
        plt.tight_layout()
        plt.savefig(figures_dir / 'ece_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_figures(self, figures_dir: Path):
        """Create performance-related figures."""
        performance_results = self.results['performance']
        
        # Timing comparison
        if 'timing_results' in performance_results:
            timing_data = performance_results['timing_results']
            
            methods = list(timing_data.keys())
            mean_times = [np.mean(timing_data[method]) for method in methods]
            std_times = [np.std(timing_data[method]) for method in methods]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(methods, mean_times, yerr=std_times, capsize=5)
            plt.title('Inference Time Comparison')
            plt.ylabel('Inference Time (ms)')
            plt.xlabel('Method')
            
            # Add speedup annotations
            if 'bayesian' in timing_data and 'ensemble' in timing_data:
                bayesian_time = np.mean(timing_data['bayesian'])
                ensemble_time = np.mean(timing_data['ensemble'])
                speedup = ensemble_time / bayesian_time
                plt.text(0.5, max(mean_times) * 0.8, f'Speedup: {speedup:.1f}x', 
                        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'timing_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_ood_figures(self, figures_dir: Path):
        """Create OOD detection figures."""
        ood_results = self.results['ood']
        
        # AUROC comparison across scenarios
        auroc_data = []
        for scenario, scenario_results in ood_results.items():
            for method, metrics in scenario_results.items():
                auroc_data.append({
                    'scenario': scenario,
                    'method': method,
                    'auroc': metrics['auroc']
                })
        
        auroc_df = pd.DataFrame(auroc_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=auroc_df, x='scenario', y='auroc', hue='method')
        plt.axhline(y=0.90, color='red', linestyle='--', label='Target AUROC > 0.90')
        plt.title('OOD Detection Performance by Scenario')
        plt.ylabel('AUROC')
        plt.xlabel('OOD Scenario')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'ood_auroc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_tables(self, tables_dir: Path):
        """Create summary tables."""
        # Main results summary table
        if 'calibration' in self.results:
            results_df = self.results['calibration']['results_df']
            
            summary_stats = results_df.groupby('method').agg({
                'ece': ['mean', 'std'],
                'coverage': ['mean', 'std'],
                'crps': ['mean', 'std'],
                'inference_time': ['mean', 'std']
            }).round(4)
            
            summary_stats.to_csv(tables_dir / 'calibration_summary.csv')
        
        # Target achievement table
        target_achievement = {}
        if 'calibration' in self.results:
            results_df = self.results['calibration']['results_df']
            
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method]
                target_achievement[method] = {
                    'ECE < 0.05': f"{(method_data['ece'] < 0.05).mean():.2%}",
                    'Coverage ∈ [0.93, 0.97]': f"{((method_data['coverage'] >= 0.93) & (method_data['coverage'] <= 0.97)).mean():.2%}",
                    'Mean ECE': f"{method_data['ece'].mean():.4f}",
                    'Mean Coverage': f"{method_data['coverage'].mean():.3f}"
                }
        
        target_df = pd.DataFrame(target_achievement).T
        target_df.to_csv(tables_dir / 'target_achievement.csv')
    
    def _save_all_results(self):
        """Save all results to files."""
        # Save main results as JSON
        results_file = self.output_dir / 'all_results.json'
        
        # Convert DataFrames to dictionaries for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, pd.DataFrame):
                        json_results[key][subkey] = subvalue.to_dict('records')
                    else:
                        json_results[key][subkey] = subvalue
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save configuration
        config_file = self.output_dir / 'validation_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"All results saved to {self.output_dir}")


def run_comprehensive_validation(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the comprehensive validation suite."""
    suite = ComprehensiveValidationSuite(config_path)
    return suite.run_complete_validation()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive validation suite')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results/comprehensive_validation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.config:
        config_path = args.config
    else:
        config_path = None
    
    # Run validation
    results = run_comprehensive_validation(config_path)
    
    print("Comprehensive Validation Suite Completed!")
    print("=" * 50)
    print(f"Duration: {results['duration']}")
    print(f"Output directory: {results['output_dir']}")
    
    if 'analysis' in results and 'conclusions' in results['analysis']:
        conclusions = results['analysis']['conclusions']
        print("\nMain Findings:")
        for finding in conclusions.get('main_findings', []):
            print(f"  • {finding}")
        
        print(f"\nRecommendation: {conclusions.get('recommendation', 'No recommendation available')}")