"""
Target Performance Metrics Validator

This module validates that BayesianMetaPINN achieves all target performance metrics:
- ECE < 0.05 target
- Coverage ∈ [0.93, 0.97] 
- AUROC > 0.90 for OOD detection
- Computational efficiency targets (4-5× faster than ensembles)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from src.uncertainty.comprehensive_validation_suite import ComprehensiveValidationSuite
from src.uncertainty.experimental_validation import run_calibration_experiment
from src.uncertainty.performance_benchmarking import PerformanceBenchmarkingSuite
from src.uncertainty.ood_detection import OODDetectionEvaluator

logger = logging.getLogger(__name__)


class TargetMetricsValidator:
    """Validator for target performance metrics."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """Initialize the target metrics validator."""
        self.results_dir = Path(results_dir) if results_dir else Path('results/target_validation')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Target thresholds
        self.targets = {
            'ece_threshold': 0.05,
            'coverage_lower': 0.93,
            'coverage_upper': 0.97,
            'auroc_threshold': 0.90,
            'speedup_lower': 4.0,
            'speedup_upper': 6.0,  # Allow some margin
            'memory_threshold_gb': 5.0,
            'inference_time_threshold_ms': 10.0
        }
        
        logger.info(f"Initialized Target Metrics Validator with targets: {self.targets}")
    
    def validate_all_targets(self, validation_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Validate all target performance metrics."""
        logger.info("Starting target performance metrics validation...")
        
        if validation_results is None:
            # Run comprehensive validation if results not provided
            logger.info("No validation results provided, running comprehensive validation...")
            suite = ComprehensiveValidationSuite()
            validation_results = suite.run_complete_validation()
        
        # Extract results
        results = validation_results.get('results', {})
        
        # Validate each target
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'targets': self.targets,
            'validations': {},
            'overall_success': True,
            'summary': {}
        }
        
        # 1. Validate calibration targets (ECE < 0.05, Coverage ∈ [0.93, 0.97])
        logger.info("Validating calibration targets...")
        calibration_validation = self._validate_calibration_targets(results.get('calibration', {}))
        validation_report['validations']['calibration'] = calibration_validation
        
        # 2. Validate OOD detection targets (AUROC > 0.90)
        logger.info("Validating OOD detection targets...")
        ood_validation = self._validate_ood_targets(results.get('ood', {}))
        validation_report['validations']['ood'] = ood_validation
        
        # 3. Validate computational efficiency targets
        logger.info("Validating computational efficiency targets...")
        efficiency_validation = self._validate_efficiency_targets(results.get('performance', {}))
        validation_report['validations']['efficiency'] = efficiency_validation
        
        # 4. Generate overall assessment
        validation_report['overall_success'] = self._assess_overall_success(validation_report['validations'])
        validation_report['summary'] = self._generate_summary(validation_report['validations'])
        
        # Save validation report
        self._save_validation_report(validation_report)
        
        logger.info(f"Target validation completed. Overall success: {validation_report['overall_success']}")
        
        return validation_report
    
    def _validate_calibration_targets(self, calibration_results: Dict) -> Dict[str, Any]:
        """Validate calibration-related targets."""
        validation = {
            'ece_target': {'threshold': self.targets['ece_threshold'], 'achieved': False, 'details': {}},
            'coverage_target': {
                'range': [self.targets['coverage_lower'], self.targets['coverage_upper']], 
                'achieved': False, 
                'details': {}
            }
        }
        
        if not calibration_results or 'results_df' not in calibration_results:
            logger.warning("No calibration results found for validation")
            return validation
        
        results_df = calibration_results['results_df']
        
        # Validate ECE < 0.05 for BayesianMetaPINN
        bayesian_results = results_df[results_df['method'] == 'bayesian']
        
        if len(bayesian_results) > 0:
            mean_ece = bayesian_results['ece'].mean()
            std_ece = bayesian_results['ece'].std()
            ece_success_rate = (bayesian_results['ece'] < self.targets['ece_threshold']).mean()
            
            validation['ece_target']['achieved'] = mean_ece < self.targets['ece_threshold']
            validation['ece_target']['details'] = {
                'mean_ece': float(mean_ece),
                'std_ece': float(std_ece),
                'success_rate': float(ece_success_rate),
                'num_experiments': len(bayesian_results),
                'target_achieved': validation['ece_target']['achieved']
            }
            
            # Validate coverage ∈ [0.93, 0.97]
            mean_coverage = bayesian_results['coverage'].mean()
            std_coverage = bayesian_results['coverage'].std()
            coverage_in_range = ((bayesian_results['coverage'] >= self.targets['coverage_lower']) & 
                               (bayesian_results['coverage'] <= self.targets['coverage_upper'])).mean()
            
            validation['coverage_target']['achieved'] = (
                self.targets['coverage_lower'] <= mean_coverage <= self.targets['coverage_upper']
            )
            validation['coverage_target']['details'] = {
                'mean_coverage': float(mean_coverage),
                'std_coverage': float(std_coverage),
                'success_rate': float(coverage_in_range),
                'num_experiments': len(bayesian_results),
                'target_achieved': validation['coverage_target']['achieved']
            }
        else:
            logger.warning("No BayesianMetaPINN results found in calibration data")
        
        return validation
    
    def _validate_ood_targets(self, ood_results: Dict) -> Dict[str, Any]:
        """Validate OOD detection targets."""
        validation = {
            'auroc_target': {'threshold': self.targets['auroc_threshold'], 'achieved': False, 'details': {}}
        }
        
        if not ood_results:
            logger.warning("No OOD results found for validation")
            return validation
        
        # Collect AUROC scores for BayesianMetaPINN across all scenarios
        bayesian_aurocs = []
        scenario_details = {}
        
        for scenario, scenario_results in ood_results.items():
            if 'bayesian' in scenario_results:
                auroc = scenario_results['bayesian']['auroc']
                bayesian_aurocs.append(auroc)
                scenario_details[scenario] = {
                    'auroc': float(auroc),
                    'target_achieved': auroc > self.targets['auroc_threshold']
                }
        
        if bayesian_aurocs:
            mean_auroc = np.mean(bayesian_aurocs)
            std_auroc = np.std(bayesian_aurocs)
            success_rate = np.mean([auroc > self.targets['auroc_threshold'] for auroc in bayesian_aurocs])
            
            validation['auroc_target']['achieved'] = mean_auroc > self.targets['auroc_threshold']
            validation['auroc_target']['details'] = {
                'mean_auroc': float(mean_auroc),
                'std_auroc': float(std_auroc),
                'success_rate': float(success_rate),
                'num_scenarios': len(bayesian_aurocs),
                'scenario_details': scenario_details,
                'target_achieved': validation['auroc_target']['achieved']
            }
        else:
            logger.warning("No BayesianMetaPINN AUROC results found")
        
        return validation
    
    def _validate_efficiency_targets(self, performance_results: Dict) -> Dict[str, Any]:
        """Validate computational efficiency targets."""
        validation = {
            'speedup_target': {
                'range': [self.targets['speedup_lower'], self.targets['speedup_upper']], 
                'achieved': False, 
                'details': {}
            },
            'memory_target': {'threshold_gb': self.targets['memory_threshold_gb'], 'achieved': False, 'details': {}},
            'inference_time_target': {'threshold_ms': self.targets['inference_time_threshold_ms'], 'achieved': False, 'details': {}}
        }
        
        if not performance_results:
            logger.warning("No performance results found for validation")
            return validation
        
        # Validate speedup target (4-5× faster than ensembles)
        if 'timing_results' in performance_results:
            timing_results = performance_results['timing_results']
            
            if 'bayesian' in timing_results and 'ensemble' in timing_results:
                bayesian_time = np.mean(timing_results['bayesian'])
                ensemble_time = np.mean(timing_results['ensemble'])
                speedup = ensemble_time / bayesian_time
                
                validation['speedup_target']['achieved'] = (
                    self.targets['speedup_lower'] <= speedup <= self.targets['speedup_upper']
                )
                validation['speedup_target']['details'] = {
                    'bayesian_time_ms': float(bayesian_time),
                    'ensemble_time_ms': float(ensemble_time),
                    'speedup_factor': float(speedup),
                    'target_range': [self.targets['speedup_lower'], self.targets['speedup_upper']],
                    'target_achieved': validation['speedup_target']['achieved']
                }
        
        # Validate memory target (< 5 GB for BayesianMetaPINN)
        if 'memory_results' in performance_results:
            memory_results = performance_results['memory_results']
            
            if 'bayesian' in memory_results:
                bayesian_memory_bytes = memory_results['bayesian']
                bayesian_memory_gb = bayesian_memory_bytes / (1024**3)
                
                validation['memory_target']['achieved'] = bayesian_memory_gb < self.targets['memory_threshold_gb']
                validation['memory_target']['details'] = {
                    'bayesian_memory_gb': float(bayesian_memory_gb),
                    'threshold_gb': self.targets['memory_threshold_gb'],
                    'target_achieved': validation['memory_target']['achieved']
                }
        
        # Validate inference time target (< 10 ms/query for BayesianMetaPINN)
        if 'timing_results' in performance_results:
            timing_results = performance_results['timing_results']
            
            if 'bayesian' in timing_results:
                bayesian_time = np.mean(timing_results['bayesian'])
                
                validation['inference_time_target']['achieved'] = bayesian_time < self.targets['inference_time_threshold_ms']
                validation['inference_time_target']['details'] = {
                    'bayesian_time_ms': float(bayesian_time),
                    'threshold_ms': self.targets['inference_time_threshold_ms'],
                    'target_achieved': validation['inference_time_target']['achieved']
                }
        
        return validation
    
    def _assess_overall_success(self, validations: Dict) -> bool:
        """Assess overall success based on all validations."""
        all_targets_achieved = True
        
        # Check calibration targets
        calibration = validations.get('calibration', {})
        if not (calibration.get('ece_target', {}).get('achieved', False) and 
                calibration.get('coverage_target', {}).get('achieved', False)):
            all_targets_achieved = False
        
        # Check OOD targets
        ood = validations.get('ood', {})
        if not ood.get('auroc_target', {}).get('achieved', False):
            all_targets_achieved = False
        
        # Check efficiency targets (at least speedup and inference time)
        efficiency = validations.get('efficiency', {})
        if not (efficiency.get('speedup_target', {}).get('achieved', False) and 
                efficiency.get('inference_time_target', {}).get('achieved', False)):
            all_targets_achieved = False
        
        return all_targets_achieved
    
    def _generate_summary(self, validations: Dict) -> Dict[str, Any]:
        """Generate summary of validation results."""
        summary = {
            'targets_achieved': 0,
            'total_targets': 0,
            'success_rate': 0.0,
            'failed_targets': [],
            'achieved_targets': [],
            'recommendations': []
        }
        
        # Count targets
        for category, category_validations in validations.items():
            for target_name, target_info in category_validations.items():
                summary['total_targets'] += 1
                if target_info.get('achieved', False):
                    summary['targets_achieved'] += 1
                    summary['achieved_targets'].append(f"{category}.{target_name}")
                else:
                    summary['failed_targets'].append(f"{category}.{target_name}")
        
        # Calculate success rate
        if summary['total_targets'] > 0:
            summary['success_rate'] = summary['targets_achieved'] / summary['total_targets']
        
        # Generate recommendations for failed targets
        if summary['failed_targets']:
            summary['recommendations'] = self._generate_recommendations(validations)
        
        return summary
    
    def _generate_recommendations(self, validations: Dict) -> List[str]:
        """Generate recommendations for improving failed targets."""
        recommendations = []
        
        # Calibration recommendations
        calibration = validations.get('calibration', {})
        if not calibration.get('ece_target', {}).get('achieved', False):
            recommendations.append(
                "ECE target not achieved. Consider: (1) Increasing KL weight for stronger regularization, "
                "(2) Using temperature scaling post-hoc calibration, (3) Improving physics-informed priors"
            )
        
        if not calibration.get('coverage_target', {}).get('achieved', False):
            recommendations.append(
                "Coverage target not achieved. Consider: (1) Adjusting aleatoric uncertainty modeling, "
                "(2) Improving posterior approximation quality, (3) Using conformal prediction methods"
            )
        
        # OOD recommendations
        ood = validations.get('ood', {})
        if not ood.get('auroc_target', {}).get('achieved', False):
            recommendations.append(
                "OOD detection target not achieved. Consider: (1) Improving epistemic uncertainty estimation, "
                "(2) Using ensemble methods for OOD detection, (3) Adding OOD-specific training objectives"
            )
        
        # Efficiency recommendations
        efficiency = validations.get('efficiency', {})
        if not efficiency.get('speedup_target', {}).get('achieved', False):
            recommendations.append(
                "Speedup target not achieved. Consider: (1) Optimizing variational inference implementation, "
                "(2) Using amortized inference techniques, (3) Reducing number of posterior samples"
            )
        
        if not efficiency.get('memory_target', {}).get('achieved', False):
            recommendations.append(
                "Memory target not achieved. Consider: (1) Using lower precision (float16), "
                "(2) Implementing gradient checkpointing, (3) Reducing model size"
            )
        
        if not efficiency.get('inference_time_target', {}).get('achieved', False):
            recommendations.append(
                "Inference time target not achieved. Consider: (1) Optimizing forward pass, "
                "(2) Using compiled models (TorchScript), (3) Reducing posterior sampling overhead"
            )
        
        return recommendations
    
    def _save_validation_report(self, validation_report: Dict):
        """Save validation report to file."""
        report_file = self.results_dir / 'target_validation_report.json'
        
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
        
        # Also save a human-readable summary
        summary_file = self.results_dir / 'target_validation_summary.txt'
        self._save_human_readable_summary(validation_report, summary_file)
    
    def _save_human_readable_summary(self, validation_report: Dict, summary_file: Path):
        """Save human-readable summary of validation results."""
        with open(summary_file, 'w') as f:
            f.write("TARGET PERFORMANCE METRICS VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Validation Date: {validation_report['timestamp']}\n")
            f.write(f"Overall Success: {'✓ PASSED' if validation_report['overall_success'] else '✗ FAILED'}\n\n")
            
            # Summary statistics
            summary = validation_report['summary']
            f.write(f"Targets Achieved: {summary['targets_achieved']}/{summary['total_targets']} "
                   f"({summary['success_rate']:.1%})\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n\n")
            
            validations = validation_report['validations']
            
            # Calibration targets
            if 'calibration' in validations:
                f.write("1. CALIBRATION TARGETS:\n")
                cal = validations['calibration']
                
                ece = cal.get('ece_target', {})
                f.write(f"   ECE < 0.05: {'✓' if ece.get('achieved') else '✗'}")
                if 'details' in ece and 'mean_ece' in ece['details']:
                    f.write(f" (achieved: {ece['details']['mean_ece']:.4f})")
                f.write("\n")
                
                cov = cal.get('coverage_target', {})
                f.write(f"   Coverage ∈ [0.93, 0.97]: {'✓' if cov.get('achieved') else '✗'}")
                if 'details' in cov and 'mean_coverage' in cov['details']:
                    f.write(f" (achieved: {cov['details']['mean_coverage']:.3f})")
                f.write("\n\n")
            
            # OOD targets
            if 'ood' in validations:
                f.write("2. OOD DETECTION TARGETS:\n")
                ood = validations['ood']
                
                auroc = ood.get('auroc_target', {})
                f.write(f"   AUROC > 0.90: {'✓' if auroc.get('achieved') else '✗'}")
                if 'details' in auroc and 'mean_auroc' in auroc['details']:
                    f.write(f" (achieved: {auroc['details']['mean_auroc']:.3f})")
                f.write("\n\n")
            
            # Efficiency targets
            if 'efficiency' in validations:
                f.write("3. COMPUTATIONAL EFFICIENCY TARGETS:\n")
                eff = validations['efficiency']
                
                speedup = eff.get('speedup_target', {})
                f.write(f"   4-5× Speedup: {'✓' if speedup.get('achieved') else '✗'}")
                if 'details' in speedup and 'speedup_factor' in speedup['details']:
                    f.write(f" (achieved: {speedup['details']['speedup_factor']:.1f}×)")
                f.write("\n")
                
                memory = eff.get('memory_target', {})
                f.write(f"   Memory < 5 GB: {'✓' if memory.get('achieved') else '✗'}")
                if 'details' in memory and 'bayesian_memory_gb' in memory['details']:
                    f.write(f" (achieved: {memory['details']['bayesian_memory_gb']:.1f} GB)")
                f.write("\n")
                
                time = eff.get('inference_time_target', {})
                f.write(f"   Inference < 10 ms: {'✓' if time.get('achieved') else '✗'}")
                if 'details' in time and 'bayesian_time_ms' in time['details']:
                    f.write(f" (achieved: {time['details']['bayesian_time_ms']:.1f} ms)")
                f.write("\n\n")
            
            # Recommendations
            if summary.get('recommendations'):
                f.write("RECOMMENDATIONS FOR IMPROVEMENT:\n")
                f.write("-" * 35 + "\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"{i}. {rec}\n\n")
        
        logger.info(f"Human-readable summary saved to: {summary_file}")


def validate_target_metrics(results_dir: Optional[str] = None, 
                          validation_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Convenience function to validate target performance metrics.
    
    Args:
        results_dir: Directory to save validation results
        validation_results: Pre-computed validation results (optional)
    
    Returns:
        Dictionary containing validation report
    """
    validator = TargetMetricsValidator(results_dir)
    return validator.validate_all_targets(validation_results)


if __name__ == "__main__":
    # Run target validation
    print("Running target performance metrics validation...")
    
    validation_report = validate_target_metrics()
    
    print(f"\nValidation completed!")
    print(f"Overall success: {validation_report['overall_success']}")
    print(f"Targets achieved: {validation_report['summary']['targets_achieved']}/{validation_report['summary']['total_targets']}")
    
    if not validation_report['overall_success']:
        print("\nFailed targets:")
        for target in validation_report['summary']['failed_targets']:
            print(f"  - {target}")