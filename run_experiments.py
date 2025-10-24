#!/usr/bin/env python3
"""
Run complete experimental validation suite for Bayesian uncertainty quantification.

This script executes the comprehensive validation suite including:
- Main calibration comparison experiments
- Ablation studies
- Performance benchmarking
- OOD detection evaluation
- Uncertainty decomposition validation
- Publication-ready figure and table generation
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from uncertainty.comprehensive_validation_suite import run_comprehensive_validation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation_suite.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the comprehensive validation suite."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive validation suite for Bayesian uncertainty quantification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_validation.py
  python run_complete_validation.py --config configs/validation_config.yaml
  python run_complete_validation.py --output-dir results/final_validation
  python run_complete_validation.py --quick-test
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results/comprehensive_validation',
        help='Output directory for results (default: results/comprehensive_validation)'
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run a quick test with reduced parameters'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Comprehensive Validation Suite")
    logger.info("=" * 60)
    logger.info(f"Configuration file: {args.config or 'Default configuration'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quick test mode: {args.quick_test}")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Create configuration if quick test is requested
        if args.quick_test:
            config_path = create_quick_test_config(args.output_dir)
            logger.info(f"Created quick test configuration: {config_path}")
        else:
            config_path = args.config
        
        # Run the comprehensive validation suite
        results = run_comprehensive_validation(config_path)
        
        # Print summary
        print_validation_summary(results)
        
        logger.info("Comprehensive validation suite completed successfully!")
        logger.info(f"Results saved to: {results['output_dir']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Validation suite interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        logger.exception("Full traceback:")
        return 1


def create_quick_test_config(output_dir: str) -> str:
    """Create a quick test configuration for faster validation."""
    import yaml
    
    quick_config = {
        'output_dir': output_dir,
        'random_seed': 42,
        
        # Reduced parameters for quick testing
        'calibration_experiment': {
            'pde_types': ['heat', 'burgers'],  # Only 2 PDE types
            'noise_levels': [0.05, 0.1],      # Only 2 noise levels
            'k_shot_values': [1, 5],          # Only 2 K-shot values
            'num_test_tasks': 10,             # Reduced test tasks
            'num_posterior_samples': 20       # Reduced samples
        },
        
        'ablation_studies': {
            'prior_types': ['standard', 'physics_informed'],
            'variational_families': ['diagonal'],
            'kl_weights': [0.5, 1.0],
            'temperature_scales': [1.0, 1.5]
        },
        
        'performance_benchmarking': {
            'batch_sizes': [1, 10],
            'num_posterior_samples': [10, 50],
            'memory_profiling': True,
            'timing_iterations': 3
        },
        
        'ood_evaluation': {
            'scenarios': ['spatial_extrapolation', 'parameter_shift'],
            'num_in_dist': 100,
            'num_ood': 100
        },
        
        'decomposition_validation': {
            'k_values': [1, 5, 10],
            'num_tasks': 5,
            'num_query_points': 50
        }
    }
    
    # Save quick config
    config_path = Path(output_dir) / 'quick_test_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    return str(config_path)


def print_validation_summary(results: dict):
    """Print a summary of validation results."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION SUITE SUMMARY")
    print("=" * 60)
    
    print(f"Duration: {results.get('duration', 'Unknown')}")
    print(f"Output Directory: {results.get('output_dir', 'Unknown')}")
    
    if 'analysis' in results and 'conclusions' in results['analysis']:
        conclusions = results['analysis']['conclusions']
        
        print("\nMAIN FINDINGS:")
        print("-" * 30)
        for finding in conclusions.get('main_findings', []):
            print(f"  • {finding}")
        
        print(f"\nRECOMMENDATION:")
        print("-" * 30)
        print(f"  {conclusions.get('recommendation', 'No recommendation available')}")
    
    # Print component summaries
    if 'results' in results:
        component_results = results['results']
        
        print(f"\nCOMPONENT RESULTS:")
        print("-" * 30)
        
        if 'calibration' in component_results:
            print("  ✓ Calibration comparison experiment completed")
            
        if 'ablation' in component_results:
            print("  ✓ Ablation studies completed")
            
        if 'performance' in component_results:
            print("  ✓ Performance benchmarking completed")
            
        if 'ood' in component_results:
            print("  ✓ OOD detection evaluation completed")
            
        if 'decomposition' in component_results:
            print("  ✓ Uncertainty decomposition validation completed")
    
    print("\nFILES GENERATED:")
    print("-" * 30)
    output_dir = Path(results.get('output_dir', '.'))
    
    if output_dir.exists():
        # List key output files
        key_files = [
            'all_results.json',
            'validation_config.yaml',
            'figures/ece_comparison.png',
            'figures/coverage_comparison.png',
            'figures/timing_comparison.png',
            'tables/calibration_summary.csv',
            'tables/target_achievement.csv'
        ]
        
        for file_path in key_files:
            full_path = output_dir / file_path
            if full_path.exists():
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)