"""EnsembleMetaPINN implementation for uncertainty quantification.

This module implements ensemble-based uncertainty quantification by training
multiple independent MetaPINN models with different random initializations
and computing uncertainty as prediction variance across ensemble members.
"""

import copy
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import (
    UncertaintyMetaLearner, 
    UncertaintyPrediction, 
    TaskDistribution, 
    Task,
    ConvergenceError,
    DecompositionError
)
from ..meta_learning.meta_pinn import MetaPINN
from ..meta_learning.config import MetaPINNConfig


class EnsembleMetaPINN(UncertaintyMetaLearner):
    """Ensemble of independent MetaPINN models for uncertainty quantification.
    
    This class implements uncertainty quantification by training multiple
    independent MetaPINN models with different random initializations and
    computing uncertainty as prediction variance across ensemble members.
    
    Args:
        base_model_config: Configuration for base MetaPINN models
        num_models: Number of ensemble members (default: 10)
        parallel_training: Whether to train models in parallel
        device: Device for computation
        random_seed: Base random seed for reproducible ensemble initialization
    """
    
    def __init__(self,
                 base_model_config: MetaPINNConfig,
                 num_models: int = 10,
                 parallel_training: bool = False,
                 device: Union[str, torch.device] = 'cpu',
                 random_seed: int = 42):
        
        self.base_model_config = base_model_config
        self.num_models = num_models
        self.parallel_training = parallel_training
        self.device = torch.device(device)
        self.random_seed = random_seed
        
        # Initialize ensemble members with different random seeds
        self.models = []
        self._initialize_ensemble()
        
        # Adaptation state
        self._is_adapted = False
        self.adapted_models = None
        self.adaptation_history = []
        
        # Move to device
        self.to(self.device)
    
    def _initialize_ensemble(self) -> None:
        """Initialize ensemble members with different random seeds."""
        self.models = []
        
        for i in range(self.num_models):
            # Set different random seed for each model
            model_seed = self.random_seed + i
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            
            # Create model with same config but different initialization
            model = MetaPINN(self.base_model_config)
            self.models.append(model)
        
        print(f"Initialized ensemble with {self.num_models} MetaPINN models")
    
    def to(self, device: Union[str, torch.device]) -> 'EnsembleMetaPINN':
        """Move all ensemble models to device."""
        self.device = torch.device(device)
        
        for model in self.models:
            model.network = model.network.to(device)
            model.device = self.device
        
        if self.adapted_models is not None:
            for model in self.adapted_models:
                model.network = model.network.to(device)
                model.device = self.device
        
        return self
    
    def meta_train(self, task_distribution: TaskDistribution, 
                   num_iterations: int) -> Dict[str, float]:
        """Meta-train all ensemble members on task distribution.
        
        Args:
            task_distribution: Distribution of training tasks
            num_iterations: Number of meta-training iterations
            
        Returns:
            Dictionary containing aggregated training metrics
        """
        print(f"Meta-training ensemble of {self.num_models} models for {num_iterations} iterations...")
        
        if self.parallel_training:
            return self._meta_train_parallel(task_distribution, num_iterations)
        else:
            return self._meta_train_sequential(task_distribution, num_iterations)
    
    def _meta_train_sequential(self, task_distribution: TaskDistribution, 
                              num_iterations: int) -> Dict[str, float]:
        """Sequential meta-training of ensemble members."""
        training_results = []
        
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i+1}/{self.num_models}...")
            
            # Set different random seed for training
            torch.manual_seed(self.random_seed + i + self.num_models)
            np.random.seed(self.random_seed + i + self.num_models)
            
            # Sample different task batches for each model
            model_task_distribution = self._create_model_specific_task_distribution(
                task_distribution, model_id=i
            )
            
            # Train individual model
            start_time = time.time()
            result = model.meta_train(model_task_distribution, num_iterations)
            training_time = time.time() - start_time
            
            result['training_time'] = training_time
            result['model_id'] = i
            training_results.append(result)
            
            print(f"Model {i+1} completed in {training_time:.2f}s, "
                  f"final loss: {result.get('final_meta_loss', 'N/A'):.6f}")
        
        return self._aggregate_training_results(training_results)
    
    def _meta_train_parallel(self, task_distribution: TaskDistribution, 
                            num_iterations: int) -> Dict[str, float]:
        """Parallel meta-training of ensemble members."""
        training_results = []
        
        def train_single_model(model_info):
            model_id, model = model_info
            
            # Set different random seed for training
            torch.manual_seed(self.random_seed + model_id + self.num_models)
            np.random.seed(self.random_seed + model_id + self.num_models)
            
            # Sample different task batches for each model
            model_task_distribution = self._create_model_specific_task_distribution(
                task_distribution, model_id=model_id
            )
            
            # Train individual model
            start_time = time.time()
            result = model.meta_train(model_task_distribution, num_iterations)
            training_time = time.time() - start_time
            
            result['training_time'] = training_time
            result['model_id'] = model_id
            
            return result
        
        # Use ThreadPoolExecutor for parallel training
        max_workers = min(self.num_models, 4)  # Limit to avoid resource exhaustion
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit training jobs
            future_to_model = {
                executor.submit(train_single_model, (i, model)): i 
                for i, model in enumerate(self.models)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    training_results.append(result)
                    print(f"Model {model_id+1} completed, "
                          f"final loss: {result.get('final_meta_loss', 'N/A'):.6f}")
                except Exception as e:
                    print(f"Model {model_id+1} training failed: {e}")
                    raise ConvergenceError(f"Ensemble member {model_id} training failed: {e}")
        
        return self._aggregate_training_results(training_results)
    
    def _create_model_specific_task_distribution(self, task_distribution: TaskDistribution, 
                                               model_id: int) -> TaskDistribution:
        """Create model-specific task distribution for diversity."""
        # For now, return the same distribution
        # In practice, could implement bootstrap sampling or other diversity techniques
        return task_distribution
    
    def _aggregate_training_results(self, training_results: List[Dict]) -> Dict[str, float]:
        """Aggregate training results across ensemble members."""
        if not training_results:
            raise ConvergenceError("No successful training results to aggregate")
        
        # Extract metrics that exist in all results
        common_metrics = set(training_results[0].keys())
        for result in training_results[1:]:
            common_metrics &= set(result.keys())
        
        # Remove non-numeric keys
        numeric_metrics = []
        for metric in common_metrics:
            if isinstance(training_results[0][metric], (int, float)):
                numeric_metrics.append(metric)
        
        # Compute aggregated statistics
        aggregated = {}
        for metric in numeric_metrics:
            values = [result[metric] for result in training_results]
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
            aggregated[f'min_{metric}'] = np.min(values)
            aggregated[f'max_{metric}'] = np.max(values)
        
        # Add ensemble-specific metrics
        aggregated['num_models'] = len(training_results)
        aggregated['successful_models'] = len(training_results)
        aggregated['total_training_time'] = sum(r.get('training_time', 0) for r in training_results)
        
        # Store individual results
        aggregated['individual_results'] = training_results
        
        return aggregated
    
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor,
              num_steps: int = 10) -> 'EnsembleMetaPINN':
        """Adapt all ensemble members to new task using support data.
        
        Args:
            support_data: Support set inputs [k_shot, input_dim]
            support_targets: Support set targets [k_shot, output_dim]
            num_steps: Number of adaptation steps
            
        Returns:
            Self (for method chaining)
        """
        # Validate inputs
        self.validate_inputs(support_data)
        self.validate_targets(support_targets)
        
        # Move to device
        support_data = support_data.to(self.device)
        support_targets = support_targets.to(self.device)
        
        # Create adapted models
        self.adapted_models = []
        self.adaptation_history = []
        
        print(f"Adapting ensemble of {self.num_models} models...")
        
        for i, model in enumerate(self.models):
            # Create a copy of the model for adaptation
            adapted_model = copy.deepcopy(model)
            
            # Create task data structure expected by MetaPINN
            from ..meta_learning.task import TaskData
            task_data = TaskData(
                inputs=support_data,
                outputs=support_targets,
                collocation_points=support_data,  # Use same points for physics loss
                boundary_data=None,
                initial_data=None
            )
            
            # Create a dummy task for adaptation
            from ..meta_learning.task import Task
            dummy_task = Task(
                problem_type='adaptation',
                parameters={},
                support_data=task_data,
                query_data=task_data,  # Same as support for adaptation
                metadata={'model_id': i}
            )
            
            # Adapt the model
            start_time = time.time()
            try:
                # Use MetaPINN's adapt method
                adapted_model = adapted_model.adapt(task_data, dummy_task, 
                                                  k_shots=len(support_data), 
                                                  adaptation_steps=num_steps)
                
                adaptation_time = time.time() - start_time
                
                self.adapted_models.append(adapted_model)
                self.adaptation_history.append({
                    'model_id': i,
                    'adaptation_time': adaptation_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"Warning: Model {i} adaptation failed: {e}")
                # Use original model if adaptation fails
                self.adapted_models.append(model)
                self.adaptation_history.append({
                    'model_id': i,
                    'adaptation_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                })
        
        self._is_adapted = True
        
        successful_adaptations = sum(1 for h in self.adaptation_history if h['success'])
        print(f"Successfully adapted {successful_adaptations}/{self.num_models} models")
        
        return self
    
    def predict_with_uncertainty(self, query_points: torch.Tensor,
                                num_samples: int = None) -> UncertaintyPrediction:
        """Predict with uncertainty quantification using ensemble variance.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            num_samples: Ignored for ensemble (uses all models)
            
        Returns:
            UncertaintyPrediction with ensemble-based uncertainty
        """
        # Validate inputs
        self.validate_inputs(query_points)
        query_points = query_points.to(self.device)
        
        # Use adapted models if available, otherwise use base models
        models_to_use = self.adapted_models if self._is_adapted else self.models
        
        if not models_to_use:
            raise RuntimeError("No models available for prediction")
        
        # Collect predictions from all ensemble members
        predictions = []
        
        for i, model in enumerate(models_to_use):
            try:
                model.network.eval()
                with torch.no_grad():
                    pred = model.forward(query_points)
                    predictions.append(pred)
            except Exception as e:
                print(f"Warning: Model {i} prediction failed: {e}")
                # Skip failed predictions
                continue
        
        if not predictions:
            raise RuntimeError("All ensemble predictions failed")
        
        # Stack predictions: [num_models, batch_size, output_dim]
        predictions = torch.stack(predictions)
        
        # Compute ensemble statistics
        mean_prediction = predictions.mean(dim=0)
        
        # Ensemble uncertainty: variance across models
        if len(predictions) > 1:
            ensemble_variance = predictions.var(dim=0)
        else:
            # Single model case
            ensemble_variance = torch.zeros_like(mean_prediction)
        
        # For ensemble methods, we treat all uncertainty as epistemic
        # since we cannot reliably separate epistemic and aleatoric
        epistemic_uncertainty = ensemble_variance
        aleatoric_uncertainty = torch.zeros_like(mean_prediction)
        
        # Validate uncertainty
        self._validate_ensemble_uncertainty(epistemic_uncertainty, aleatoric_uncertainty)
        
        return UncertaintyPrediction(
            mean=mean_prediction,
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            samples=predictions
        )
    
    def get_epistemic_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract epistemic (model) uncertainty from ensemble variance.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Epistemic uncertainty [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points)
        return prediction.epistemic
    
    def get_aleatoric_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract aleatoric (data) uncertainty.
        
        For ensemble methods, aleatoric uncertainty is not directly available
        and is returned as zeros.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Aleatoric uncertainty (zeros) [batch_size, output_dim]
        """
        prediction = self.predict_with_uncertainty(query_points)
        return prediction.aleatoric
    
    def _validate_ensemble_uncertainty(self, epistemic: torch.Tensor, 
                                     aleatoric: torch.Tensor) -> None:
        """Validate ensemble uncertainty properties.
        
        Args:
            epistemic: Epistemic uncertainty tensor
            aleatoric: Aleatoric uncertainty tensor
            
        Raises:
            DecompositionError: If uncertainty is invalid
        """
        # Check for non-negative uncertainties
        if torch.any(epistemic < 0) or torch.any(aleatoric < 0):
            raise DecompositionError("Uncertainties must be non-negative")
        
        # Check for NaN or Inf values
        if torch.any(torch.isnan(epistemic)) or torch.any(torch.isnan(aleatoric)):
            raise DecompositionError("Uncertainties contain NaN values")
        
        if torch.any(torch.isinf(epistemic)) or torch.any(torch.isinf(aleatoric)):
            raise DecompositionError("Uncertainties contain Inf values")
    
    @property
    def is_adapted(self) -> bool:
        """Check if ensemble has been adapted to a task."""
        return self._is_adapted
    
    def reset_adaptation(self) -> None:
        """Reset adaptation state for new task."""
        self._is_adapted = False
        self.adapted_models = None
        self.adaptation_history = []
    
    def get_ensemble_diversity(self, query_points: torch.Tensor) -> Dict[str, float]:
        """Compute ensemble diversity metrics.
        
        Args:
            query_points: Query inputs for diversity computation
            
        Returns:
            Dictionary with diversity metrics
        """
        # Validate inputs
        self.validate_inputs(query_points)
        query_points = query_points.to(self.device)
        
        # Use adapted models if available
        models_to_use = self.adapted_models if self._is_adapted else self.models
        
        # Collect predictions
        predictions = []
        for model in models_to_use:
            model.network.eval()
            with torch.no_grad():
                pred = model.forward(query_points)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_models, batch_size, output_dim]
        
        # Compute diversity metrics
        mean_pred = predictions.mean(dim=0)
        
        # Average pairwise disagreement
        pairwise_disagreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                disagreement = torch.mean((predictions[i] - predictions[j]) ** 2)
                pairwise_disagreements.append(disagreement.item())
        
        # Disagreement with ensemble mean
        disagreements_with_mean = []
        for i in range(len(predictions)):
            disagreement = torch.mean((predictions[i] - mean_pred) ** 2)
            disagreements_with_mean.append(disagreement.item())
        
        return {
            'mean_pairwise_disagreement': np.mean(pairwise_disagreements) if pairwise_disagreements else 0.0,
            'std_pairwise_disagreement': np.std(pairwise_disagreements) if pairwise_disagreements else 0.0,
            'mean_disagreement_with_ensemble': np.mean(disagreements_with_mean),
            'std_disagreement_with_ensemble': np.std(disagreements_with_mean),
            'prediction_variance': torch.mean(predictions.var(dim=0)).item(),
            'num_models': len(predictions)
        }
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of ensemble adaptation process.
        
        Returns:
            Dictionary with adaptation statistics
        """
        if not self.adaptation_history:
            return {'status': 'No adaptation performed'}
        
        successful_adaptations = [h for h in self.adaptation_history if h['success']]
        failed_adaptations = [h for h in self.adaptation_history if not h['success']]
        
        summary = {
            'adapted': self._is_adapted,
            'total_models': len(self.adaptation_history),
            'successful_adaptations': len(successful_adaptations),
            'failed_adaptations': len(failed_adaptations),
            'success_rate': len(successful_adaptations) / len(self.adaptation_history),
        }
        
        if successful_adaptations:
            adaptation_times = [h['adaptation_time'] for h in successful_adaptations]
            summary.update({
                'mean_adaptation_time': np.mean(adaptation_times),
                'std_adaptation_time': np.std(adaptation_times),
                'total_adaptation_time': np.sum(adaptation_times)
            })
        
        if failed_adaptations:
            summary['failure_reasons'] = [h.get('error', 'Unknown') for h in failed_adaptations]
        
        summary['adaptation_history'] = self.adaptation_history
        
        return summary
    
    def state_dict(self) -> Dict:
        """Get ensemble state dictionary."""
        return {
            'base_model_config': self.base_model_config,
            'num_models': self.num_models,
            'random_seed': self.random_seed,
            'model_states': [model.network.state_dict() for model in self.models],
            'adaptation_history': self.adaptation_history,
            'is_adapted': self._is_adapted
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load ensemble state dictionary."""
        self.base_model_config = state_dict['base_model_config']
        self.num_models = state_dict['num_models']
        self.random_seed = state_dict['random_seed']
        
        # Load individual model states
        for i, model_state in enumerate(state_dict['model_states']):
            if i < len(self.models):
                self.models[i].network.load_state_dict(model_state)
        
        self.adaptation_history = state_dict.get('adaptation_history', [])
        self._is_adapted = state_dict.get('is_adapted', False)


def create_ensemble_meta_pinn(config: Dict) -> EnsembleMetaPINN:
    """Factory function to create EnsembleMetaPINN with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured EnsembleMetaPINN instance
    """
    # Extract ensemble-specific config
    ensemble_config = {
        'num_models': config.get('num_models', 10),
        'parallel_training': config.get('parallel_training', False),
        'device': config.get('device', 'cpu'),
        'random_seed': config.get('random_seed', 42)
    }
    
    # Create base model config
    base_model_config = MetaPINNConfig(
        layers=config.get('layers', [2, 64, 64, 1]),
        activation=config.get('activation', 'tanh'),
        meta_lr=config.get('meta_lr', 0.001),
        adapt_lr=config.get('adapt_lr', 0.01),
        adaptation_steps=config.get('adaptation_steps', 10),
        meta_batch_size=config.get('meta_batch_size', 4),
        device=config.get('device', 'cpu')
    )
    
    return EnsembleMetaPINN(
        base_model_config=base_model_config,
        **ensemble_config
    )