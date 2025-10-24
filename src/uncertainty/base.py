"""Base interfaces and data structures for uncertainty quantification.

This module defines the core abstractions for uncertainty-aware meta-learning
models and structured uncertainty outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import torch


class UncertaintyQuantificationError(Exception):
    """Base exception for uncertainty quantification-related errors."""
    pass


class CalibrationError(UncertaintyQuantificationError):
    """Raised when calibration computation fails."""
    pass


class DecompositionError(UncertaintyQuantificationError):
    """Raised when uncertainty decomposition is invalid."""
    pass


class ConvergenceError(UncertaintyQuantificationError):
    """Raised when variational inference fails to converge."""
    pass


@dataclass
class UncertaintyPrediction:
    """Container for predictions with uncertainty quantification.
    
    Attributes:
        mean: Mean predictions [batch_size, output_dim]
        epistemic: Epistemic (model) uncertainty [batch_size, output_dim]
        aleatoric: Aleatoric (data) uncertainty [batch_size, output_dim]
        total: Total uncertainty (epistemic + aleatoric) [batch_size, output_dim]
        samples: Optional posterior samples [n_samples, batch_size, output_dim]
    """
    mean: torch.Tensor
    epistemic: torch.Tensor
    aleatoric: torch.Tensor
    samples: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Compute total uncertainty and validate shapes."""
        # Compute total uncertainty
        self.total = self.epistemic + self.aleatoric
        
        # Validate shapes
        if not (self.mean.shape == self.epistemic.shape == self.aleatoric.shape):
            raise ValueError(
                f"Shape mismatch: mean {self.mean.shape}, "
                f"epistemic {self.epistemic.shape}, aleatoric {self.aleatoric.shape}"
            )
        
        # Validate non-negative uncertainties
        if torch.any(self.epistemic < 0) or torch.any(self.aleatoric < 0):
            raise ValueError("Uncertainties must be non-negative")
        
        # Validate samples shape if provided
        if self.samples is not None:
            expected_shape = (self.samples.shape[0], *self.mean.shape)
            if self.samples.shape != expected_shape:
                raise ValueError(
                    f"Samples shape {self.samples.shape} doesn't match "
                    f"expected {expected_shape}"
                )
    
    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.mean.device
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.mean.shape[0]
    
    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self.mean.shape[-1] if len(self.mean.shape) > 1 else 1
    
    def to(self, device: Union[str, torch.device]) -> 'UncertaintyPrediction':
        """Move all tensors to specified device."""
        return UncertaintyPrediction(
            mean=self.mean.to(device),
            epistemic=self.epistemic.to(device),
            aleatoric=self.aleatoric.to(device),
            samples=self.samples.to(device) if self.samples is not None else None
        )
    
    def detach(self) -> 'UncertaintyPrediction':
        """Detach all tensors from computation graph."""
        return UncertaintyPrediction(
            mean=self.mean.detach(),
            epistemic=self.epistemic.detach(),
            aleatoric=self.aleatoric.detach(),
            samples=self.samples.detach() if self.samples is not None else None
        )


class UncertaintyMetaLearner(ABC):
    """Base interface for uncertainty-aware meta-learning models.
    
    This abstract class defines the core interface that all uncertainty
    quantification methods must implement for meta-learning with PINNs.
    """
    
    @abstractmethod
    def meta_train(self, task_distribution, num_iterations: int) -> Dict[str, float]:
        """Meta-train the model on a task distribution.
        
        Args:
            task_distribution: Distribution of training tasks
            num_iterations: Number of meta-training iterations
            
        Returns:
            Dictionary containing training metrics and losses
            
        Raises:
            ConvergenceError: If training fails to converge
        """
        pass
    
    @abstractmethod
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor,
              num_steps: int = 10) -> 'UncertaintyMetaLearner':
        """Adapt to new task using support data.
        
        Args:
            support_data: Support set inputs [k_shot, input_dim]
            support_targets: Support set targets [k_shot, output_dim]
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model instance
            
        Raises:
            ConvergenceError: If adaptation fails to converge
        """
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, query_points: torch.Tensor,
                                num_samples: int = 100) -> UncertaintyPrediction:
        """Predict with uncertainty quantification.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            num_samples: Number of posterior samples for uncertainty estimation
            
        Returns:
            UncertaintyPrediction containing mean, epistemic, and aleatoric uncertainty
            
        Raises:
            DecompositionError: If uncertainty decomposition fails
        """
        pass
    
    @abstractmethod
    def get_epistemic_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract epistemic (model) uncertainty.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Epistemic uncertainty [batch_size, output_dim]
        """
        pass
    
    @abstractmethod
    def get_aleatoric_uncertainty(self, query_points: torch.Tensor) -> torch.Tensor:
        """Extract aleatoric (data) uncertainty.
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Aleatoric uncertainty [batch_size, output_dim]
        """
        pass
    
    def predict(self, query_points: torch.Tensor) -> torch.Tensor:
        """Standard prediction without uncertainty (for compatibility).
        
        Args:
            query_points: Query inputs [batch_size, input_dim]
            
        Returns:
            Mean predictions [batch_size, output_dim]
        """
        uncertainty_pred = self.predict_with_uncertainty(query_points, num_samples=1)
        return uncertainty_pred.mean
    
    @property
    @abstractmethod
    def is_adapted(self) -> bool:
        """Check if model has been adapted to a task."""
        pass
    
    @abstractmethod
    def reset_adaptation(self) -> None:
        """Reset adaptation state for new task."""
        pass
    
    def validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate input tensor format and shape.
        
        Args:
            inputs: Input tensor to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(inputs)}")
        
        if len(inputs.shape) < 2:
            raise ValueError(f"Expected at least 2D tensor, got shape {inputs.shape}")
        
        if torch.any(torch.isnan(inputs)) or torch.any(torch.isinf(inputs)):
            raise ValueError("Input contains NaN or Inf values")
    
    def validate_targets(self, targets: torch.Tensor) -> None:
        """Validate target tensor format and shape.
        
        Args:
            targets: Target tensor to validate
            
        Raises:
            ValueError: If targets are invalid
        """
        if not isinstance(targets, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(targets)}")
        
        if len(targets.shape) < 2:
            raise ValueError(f"Expected at least 2D tensor, got shape {targets.shape}")
        
        if torch.any(torch.isnan(targets)) or torch.any(torch.isinf(targets)):
            raise ValueError("Targets contain NaN or Inf values")


class TaskDistribution(ABC):
    """Abstract base class for task distributions used in meta-learning.
    
    This class defines the interface for sampling tasks during meta-training
    and evaluation.
    """
    
    @abstractmethod
    def sample_task(self) -> 'Task':
        """Sample a single task from the distribution.
        
        Returns:
            Task instance
        """
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List['Task']:
        """Sample a batch of tasks from the distribution.
        
        Args:
            batch_size: Number of tasks to sample
            
        Returns:
            List of Task instances
        """
        pass


class Task(ABC):
    """Abstract base class for individual tasks in meta-learning.
    
    Each task represents a specific PDE problem instance with support
    and query data.
    """
    
    @abstractmethod
    def sample_support(self, k_shot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample support set for few-shot adaptation.
        
        Args:
            k_shot: Number of support examples
            
        Returns:
            Tuple of (support_inputs, support_targets)
        """
        pass
    
    @abstractmethod
    def sample_query(self, num_query: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample query set for evaluation.
        
        Args:
            num_query: Number of query examples
            
        Returns:
            Tuple of (query_inputs, query_targets)
        """
        pass
    
    @abstractmethod
    def get_pde_residual(self, inputs: torch.Tensor, 
                        predictions: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual for physics-informed loss.
        
        Args:
            inputs: Input coordinates
            predictions: Model predictions
            
        Returns:
            PDE residual values
        """
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Get input dimension of the task."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Get output dimension of the task."""
        pass