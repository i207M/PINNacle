"""Performance benchmarking suite for uncertainty quantification methods.

This module implements comprehensive timing measurements, memory usage profiling,
and efficiency comparison framework for all uncertainty quantification methods.
"""

import time
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

from .base import UncertaintyMetaLearner, TaskDistribution, Task
from .bayesian_meta_pinn import BayesianMetaPINN
from .ensemble_meta_pinn import EnsembleMetaPINN
from .mc_dropout_meta_pinn import MCDropoutMetaPINN
from .calibration_metrics import CalibrationMetrics


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    method_name: str
    meta_training_time: float
    adaptation_time: float
    inference_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    gpu_memory_mb: float
    calibration_ece: float
    calibration_coverage: float
    num_parameters: int
    throughput_samples_per_second: float


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""
    num_meta_iterations: int = 1000
    num_adaptation_steps: int = 10
    num_inference_samples: int = 100
    batch_size: int = 32
    k_shot: int = 5
    num_query: int = 50
    num_test_tasks: int = 20
    device: str = 'cpu'
    measure_gpu_memory: bool = False
    warmup_iterations: int = 5


class MemoryProfiler:
    """Memory usage profiler for uncertainty quantification methods."""
    
    def __init__(self, measure_gpu: bool = False):
        self.measure_gpu = measure_gpu and torch.cuda.is_available()
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.memory_history = []
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024 ** 2)
        
        memory_info = {'cpu_memory_mb': cpu_memory_mb}
        
        # GPU memory
        if self.measure_gpu:
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            memory_info.update({
                'gpu_memory_mb': gpu_memory_mb,
                'gpu_memory_reserved_mb': gpu_memory_reserved_mb
            })
        
        return memory_info
    
    def record_memory(self) -> None:
        """Record current memory usage."""
        current_memory = self._get_current_memory()
        self.memory_history.append(current_memory)
        
        # Update peak memory
        if current_memory['cpu_memory_mb'] > self.peak_memory['cpu_memory_mb']:
            self.peak_memory = current_memory
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage delta from baseline."""
        current_memory = self._get_current_memory()
        
        delta = {}
        for key in current_memory:
            delta[key] = current_memory[key] - self.baseline_memory.get(key, 0)
        
        return delta
    
    def get_peak_memory_delta(self) -> Dict[str, float]:
        """Get peak memory usage delta from baseline."""
        delta = {}
        for key in self.peak_memory:
            delta[key] = self.peak_memory[key] - self.baseline_memory.get(key, 0)
        
        return delta
    
    def reset(self) -> None:
        """Reset memory profiler."""
        gc.collect()
        if self.measure_gpu:
            torch.cuda.empty_cache()
        
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.memory_history = []


@contextmanager
def timer():
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    return end_time - start_time


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = []
        self.memory_profiler = MemoryProfiler(measure_gpu=self.config.measure_gpu_memory)
    
    def benchmark_method(self, method: UncertaintyMetaLearner, 
                        method_name: str,
                        task_distribution: TaskDistribution,
                        test_tasks: List[Task]) -> PerformanceMetrics:
        """Benchmark a single uncertainty quantification method.
        
        Args:
            method: Uncertainty quantification method to benchmark
            method_name: Name of the method
            task_distribution: Task distribution for meta-training
            test_tasks: Test tasks for evaluation
            
        Returns:
            PerformanceMetrics containing all performance measurements
        """
        print(f"Benchmarking {method_name}...")
        
        # Reset memory profiler
        self.memory_profiler.reset()
        
        # Move method to device
        if hasattr(method, 'to'):
            method = method.to(self.config.device)
        
        # Warmup
        self._warmup(method, task_distribution)
        
        # Benchmark meta-training
        meta_training_time = self._benchmark_meta_training(method, task_distribution)
        
        # Benchmark adaptation and inference
        adaptation_times = []
        inference_times = []
        calibration_metrics = []
        
        for task in test_tasks[:self.config.num_test_tasks]:
            # Benchmark adaptation
            adaptation_time = self._benchmark_adaptation(method, task)
            adaptation_times.append(adaptation_time)
            
            # Benchmark inference
            inference_time, predictions, targets = self._benchmark_inference(method, task)
            inference_times.append(inference_time)
            
            # Compute calibration metrics
            if predictions is not None and targets is not None:
                try:
                    ece = CalibrationMetrics.expected_calibration_error(predictions, targets)
                    coverage_results = CalibrationMetrics.coverage_analysis(predictions, targets)
                    calibration_metrics.append({
                        'ece': ece,
                        'coverage': coverage_results['coverage']
                    })
                except Exception as e:
                    print(f"Warning: Calibration computation failed for {method_name}: {e}")
        
        # Get memory usage
        memory_delta = self.memory_profiler.get_memory_delta()
        peak_memory_delta = self.memory_profiler.get_peak_memory_delta()
        
        # Count parameters
        num_parameters = self._count_parameters(method)
        
        # Compute throughput
        avg_inference_time = np.mean(inference_times) if inference_times else float('inf')
        throughput = self.config.num_inference_samples / avg_inference_time if avg_inference_time > 0 else 0.0
        
        # Aggregate calibration metrics
        avg_ece = np.mean([m['ece'] for m in calibration_metrics]) if calibration_metrics else float('nan')
        avg_coverage = np.mean([m['coverage'] for m in calibration_metrics]) if calibration_metrics else float('nan')
        
        return PerformanceMetrics(
            method_name=method_name,
            meta_training_time=meta_training_time,
            adaptation_time=np.mean(adaptation_times) if adaptation_times else 0.0,
            inference_time=avg_inference_time,
            memory_usage_mb=memory_delta.get('cpu_memory_mb', 0.0),
            peak_memory_mb=peak_memory_delta.get('cpu_memory_mb', 0.0),
            gpu_memory_mb=memory_delta.get('gpu_memory_mb', 0.0),
            calibration_ece=avg_ece,
            calibration_coverage=avg_coverage,
            num_parameters=num_parameters,
            throughput_samples_per_second=throughput
        )
    
    def _warmup(self, method: UncertaintyMetaLearner, task_distribution: TaskDistribution) -> None:
        """Warmup method with a few iterations."""
        print("  Warming up...")
        
        for _ in range(self.config.warmup_iterations):
            task = task_distribution.sample_task()
            support_data, support_targets = task.sample_support(self.config.k_shot)
            query_data, query_targets = task.sample_query(10)  # Small query set for warmup
            
            if hasattr(method, 'to'):
                support_data = support_data.to(self.config.device)
                support_targets = support_targets.to(self.config.device)
                query_data = query_data.to(self.config.device)
            
            # Quick adaptation and inference
            try:
                adapted_method = method.adapt(support_data, support_targets, num_steps=2)
                _ = adapted_method.predict_with_uncertainty(query_data, num_samples=10)
            except Exception as e:
                print(f"    Warning: Warmup failed: {e}")
                break
    
    def _benchmark_meta_training(self, method: UncertaintyMetaLearner, 
                                task_distribution: TaskDistribution) -> float:
        """Benchmark meta-training phase."""
        print("  Benchmarking meta-training...")
        
        self.memory_profiler.record_memory()
        
        start_time = time.perf_counter()
        
        try:
            results = method.meta_train(task_distribution, self.config.num_meta_iterations)
            training_time = time.perf_counter() - start_time
            
            print(f"    Meta-training completed in {training_time:.2f}s")
            return training_time
            
        except Exception as e:
            print(f"    Meta-training failed: {e}")
            return float('inf')
    
    def _benchmark_adaptation(self, method: UncertaintyMetaLearner, task: Task) -> float:
        """Benchmark adaptation phase."""
        support_data, support_targets = task.sample_support(self.config.k_shot)
        
        if hasattr(method, 'to'):
            support_data = support_data.to(self.config.device)
            support_targets = support_targets.to(self.config.device)
        
        self.memory_profiler.record_memory()
        
        start_time = time.perf_counter()
        
        try:
            adapted_method = method.adapt(support_data, support_targets, 
                                        num_steps=self.config.num_adaptation_steps)
            adaptation_time = time.perf_counter() - start_time
            return adaptation_time
            
        except Exception as e:
            print(f"    Adaptation failed: {e}")
            return float('inf')
    
    def _benchmark_inference(self, method: UncertaintyMetaLearner, task: Task) -> Tuple[float, Optional[Any], Optional[torch.Tensor]]:
        """Benchmark inference phase."""
        # Get adapted method
        support_data, support_targets = task.sample_support(self.config.k_shot)
        query_data, query_targets = task.sample_query(self.config.num_query)
        
        if hasattr(method, 'to'):
            support_data = support_data.to(self.config.device)
            support_targets = support_targets.to(self.config.device)
            query_data = query_data.to(self.config.device)
            query_targets = query_targets.to(self.config.device)
        
        try:
            adapted_method = method.adapt(support_data, support_targets, 
                                        num_steps=self.config.num_adaptation_steps)
            
            self.memory_profiler.record_memory()
            
            start_time = time.perf_counter()
            
            predictions = adapted_method.predict_with_uncertainty(
                query_data, num_samples=self.config.num_inference_samples
            )
            
            inference_time = time.perf_counter() - start_time
            
            return inference_time, predictions, query_targets
            
        except Exception as e:
            print(f"    Inference failed: {e}")
            return float('inf'), None, None
    
    def _count_parameters(self, method: UncertaintyMetaLearner) -> int:
        """Count total number of parameters in method."""
        try:
            if hasattr(method, 'parameters'):
                return sum(p.numel() for p in method.parameters() if p.requires_grad)
            else:
                return 0
        except Exception:
            return 0
    
    def benchmark_all_methods(self, methods: Dict[str, UncertaintyMetaLearner],
                             task_distribution: TaskDistribution,
                             test_tasks: List[Task]) -> List[PerformanceMetrics]:
        """Benchmark all uncertainty quantification methods.
        
        Args:
            methods: Dictionary of method_name -> method_instance
            task_distribution: Task distribution for meta-training
            test_tasks: Test tasks for evaluation
            
        Returns:
            List of PerformanceMetrics for all methods
        """
        results = []
        
        for method_name, method in methods.items():
            try:
                metrics = self.benchmark_method(method, method_name, task_distribution, test_tasks)
                results.append(metrics)
                self.results.append(metrics)
                
                print(f"  {method_name} Results:")
                print(f"    Meta-training: {metrics.meta_training_time:.2f}s")
                print(f"    Adaptation: {metrics.adaptation_time:.4f}s")
                print(f"    Inference: {metrics.inference_time:.4f}s")
                print(f"    Memory: {metrics.memory_usage_mb:.1f}MB")
                print(f"    Parameters: {metrics.num_parameters:,}")
                print(f"    Throughput: {metrics.throughput_samples_per_second:.1f} samples/s")
                print(f"    ECE: {metrics.calibration_ece:.4f}")
                print()
                
            except Exception as e:
                print(f"  Benchmarking {method_name} failed: {e}")
                continue
        
        return results
    
    def create_efficiency_comparison(self, results: List[PerformanceMetrics] = None) -> Dict[str, Any]:
        """Create efficiency comparison framework.
        
        Args:
            results: List of performance metrics (uses stored results if None)
            
        Returns:
            Dictionary with efficiency comparison data
        """
        if results is None:
            results = self.results
        
        if not results:
            return {'error': 'No benchmark results available'}
        
        # Create comparison framework
        comparison = {
            'methods': [r.method_name for r in results],
            'meta_training_times': [r.meta_training_time for r in results],
            'adaptation_times': [r.adaptation_time for r in results],
            'inference_times': [r.inference_time for r in results],
            'memory_usage': [r.memory_usage_mb for r in results],
            'calibration_ece': [r.calibration_ece for r in results],
            'calibration_coverage': [r.calibration_coverage for r in results],
            'throughput': [r.throughput_samples_per_second for r in results],
            'num_parameters': [r.num_parameters for r in results]
        }
        
        # Compute efficiency ratios (relative to baseline)
        if len(results) > 1:
            # Use first method as baseline
            baseline_idx = 0
            baseline = results[baseline_idx]
            
            comparison['efficiency_ratios'] = {
                'inference_speedup': [
                    baseline.inference_time / r.inference_time if r.inference_time > 0 else 0
                    for r in results
                ],
                'memory_efficiency': [
                    baseline.memory_usage_mb / r.memory_usage_mb if r.memory_usage_mb > 0 else 0
                    for r in results
                ],
                'calibration_improvement': [
                    baseline.calibration_ece / r.calibration_ece if r.calibration_ece > 0 else 0
                    for r in results
                ]
            }
        
        # Compute trade-offs
        comparison['trade_offs'] = self._compute_trade_offs(results)
        
        return comparison
    
    def _compute_trade_offs(self, results: List[PerformanceMetrics]) -> Dict[str, List[float]]:
        """Compute time vs calibration quality trade-offs."""
        trade_offs = {
            'time_vs_calibration': [],
            'memory_vs_calibration': [],
            'efficiency_score': []
        }
        
        for result in results:
            # Time vs calibration (lower is better for both)
            if result.inference_time > 0 and not np.isnan(result.calibration_ece):
                time_cal_ratio = result.inference_time * result.calibration_ece
                trade_offs['time_vs_calibration'].append(time_cal_ratio)
            else:
                trade_offs['time_vs_calibration'].append(float('inf'))
            
            # Memory vs calibration
            if result.memory_usage_mb > 0 and not np.isnan(result.calibration_ece):
                memory_cal_ratio = result.memory_usage_mb * result.calibration_ece
                trade_offs['memory_vs_calibration'].append(memory_cal_ratio)
            else:
                trade_offs['memory_vs_calibration'].append(float('inf'))
            
            # Overall efficiency score (higher is better)
            if (result.inference_time > 0 and result.memory_usage_mb > 0 and 
                not np.isnan(result.calibration_ece) and result.calibration_ece > 0):
                efficiency = result.throughput_samples_per_second / (result.memory_usage_mb * result.calibration_ece)
                trade_offs['efficiency_score'].append(efficiency)
            else:
                trade_offs['efficiency_score'].append(0.0)
        
        return trade_offs
    
    def generate_performance_report(self, results: List[PerformanceMetrics] = None) -> str:
        """Generate comprehensive performance report.
        
        Args:
            results: List of performance metrics (uses stored results if None)
            
        Returns:
            Formatted performance report string
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No benchmark results available."
        
        report = []
        report.append("=" * 80)
        report.append("UNCERTAINTY QUANTIFICATION PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append()
        
        # Configuration
        report.append("Benchmark Configuration:")
        report.append(f"  Meta-training iterations: {self.config.num_meta_iterations}")
        report.append(f"  Adaptation steps: {self.config.num_adaptation_steps}")
        report.append(f"  Inference samples: {self.config.num_inference_samples}")
        report.append(f"  Test tasks: {self.config.num_test_tasks}")
        report.append(f"  Device: {self.config.device}")
        report.append()
        
        # Individual method results
        report.append("Individual Method Performance:")
        report.append("-" * 40)
        
        for result in results:
            report.append(f"Method: {result.method_name}")
            report.append(f"  Meta-training time: {result.meta_training_time:.2f}s")
            report.append(f"  Adaptation time: {result.adaptation_time:.4f}s")
            report.append(f"  Inference time: {result.inference_time:.4f}s")
            report.append(f"  Memory usage: {result.memory_usage_mb:.1f}MB")
            report.append(f"  Peak memory: {result.peak_memory_mb:.1f}MB")
            if result.gpu_memory_mb > 0:
                report.append(f"  GPU memory: {result.gpu_memory_mb:.1f}MB")
            report.append(f"  Parameters: {result.num_parameters:,}")
            report.append(f"  Throughput: {result.throughput_samples_per_second:.1f} samples/s")
            report.append(f"  Calibration ECE: {result.calibration_ece:.4f}")
            report.append(f"  Coverage: {result.calibration_coverage:.3f}")
            report.append()
        
        # Comparison
        if len(results) > 1:
            comparison = self.create_efficiency_comparison(results)
            
            report.append("Method Comparison:")
            report.append("-" * 40)
            
            # Find best performing methods
            best_inference = min(results, key=lambda x: x.inference_time)
            best_memory = min(results, key=lambda x: x.memory_usage_mb)
            best_calibration = min(results, key=lambda x: x.calibration_ece if not np.isnan(x.calibration_ece) else float('inf'))
            best_throughput = max(results, key=lambda x: x.throughput_samples_per_second)
            
            report.append(f"Fastest inference: {best_inference.method_name} ({best_inference.inference_time:.4f}s)")
            report.append(f"Lowest memory: {best_memory.method_name} ({best_memory.memory_usage_mb:.1f}MB)")
            report.append(f"Best calibration: {best_calibration.method_name} (ECE: {best_calibration.calibration_ece:.4f})")
            report.append(f"Highest throughput: {best_throughput.method_name} ({best_throughput.throughput_samples_per_second:.1f} samples/s)")
            report.append()
            
            # Efficiency ratios
            if 'efficiency_ratios' in comparison:
                report.append("Efficiency Ratios (relative to baseline):")
                baseline_name = results[0].method_name
                
                for i, method_name in enumerate(comparison['methods']):
                    if i == 0:
                        continue  # Skip baseline
                    
                    speedup = comparison['efficiency_ratios']['inference_speedup'][i]
                    memory_eff = comparison['efficiency_ratios']['memory_efficiency'][i]
                    cal_imp = comparison['efficiency_ratios']['calibration_improvement'][i]
                    
                    report.append(f"  {method_name} vs {baseline_name}:")
                    report.append(f"    Inference speedup: {speedup:.2f}x")
                    report.append(f"    Memory efficiency: {memory_eff:.2f}x")
                    report.append(f"    Calibration improvement: {cal_imp:.2f}x")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str, results: List[PerformanceMetrics] = None) -> None:
        """Save benchmark results to file.
        
        Args:
            filename: Output filename
            results: List of performance metrics (uses stored results if None)
        """
        if results is None:
            results = self.results
        
        import json
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                'method_name': result.method_name,
                'meta_training_time': result.meta_training_time,
                'adaptation_time': result.adaptation_time,
                'inference_time': result.inference_time,
                'memory_usage_mb': result.memory_usage_mb,
                'peak_memory_mb': result.peak_memory_mb,
                'gpu_memory_mb': result.gpu_memory_mb,
                'calibration_ece': result.calibration_ece if not np.isnan(result.calibration_ece) else None,
                'calibration_coverage': result.calibration_coverage if not np.isnan(result.calibration_coverage) else None,
                'num_parameters': result.num_parameters,
                'throughput_samples_per_second': result.throughput_samples_per_second
            }
            serializable_results.append(result_dict)
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump({
                'config': {
                    'num_meta_iterations': self.config.num_meta_iterations,
                    'num_adaptation_steps': self.config.num_adaptation_steps,
                    'num_inference_samples': self.config.num_inference_samples,
                    'batch_size': self.config.batch_size,
                    'k_shot': self.config.k_shot,
                    'num_query': self.config.num_query,
                    'num_test_tasks': self.config.num_test_tasks,
                    'device': self.config.device
                },
                'results': serializable_results
            }, f, indent=2)


def create_benchmark_suite(config: BenchmarkConfig = None) -> PerformanceBenchmark:
    """Factory function to create performance benchmark suite.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Configured PerformanceBenchmark instance
    """
    return PerformanceBenchmark(config)


def run_comprehensive_benchmark(methods: Dict[str, UncertaintyMetaLearner],
                               task_distribution: TaskDistribution,
                               test_tasks: List[Task],
                               config: BenchmarkConfig = None,
                               save_results: bool = True,
                               results_filename: str = "benchmark_results.json") -> List[PerformanceMetrics]:
    """Run comprehensive benchmark on all methods.
    
    Args:
        methods: Dictionary of method_name -> method_instance
        task_distribution: Task distribution for meta-training
        test_tasks: Test tasks for evaluation
        config: Benchmark configuration
        save_results: Whether to save results to file
        results_filename: Filename for saved results
        
    Returns:
        List of PerformanceMetrics for all methods
    """
    benchmark = create_benchmark_suite(config)
    
    print("Starting comprehensive performance benchmark...")
    print(f"Benchmarking {len(methods)} methods on {len(test_tasks)} test tasks")
    print()
    
    results = benchmark.benchmark_all_methods(methods, task_distribution, test_tasks)
    
    # Generate and print report
    report = benchmark.generate_performance_report(results)
    print(report)
    
    # Save results if requested
    if save_results:
        benchmark.save_results(results_filename, results)
        print(f"Results saved to {results_filename}")
    
    return results