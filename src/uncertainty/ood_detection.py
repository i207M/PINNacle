"""Out-of-distribution detection framework for uncertainty quantification.

This module implements OOD detection using epistemic uncertainty as the detection
score, along with comprehensive evaluation metrics and scenario generators.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base import UncertaintyMetaLearner, UncertaintyPrediction, Task


@dataclass
class OODResults:
    """Results from OOD detection evaluation.
    
    Attributes:
        auroc: Area Under ROC Curve
        aupr: Area Under Precision-Recall Curve  
        fpr_at_95_tpr: False Positive Rate at 95% True Positive Rate
        scenario: Name of OOD scenario tested
        num_in_dist: Number of in-distribution samples
        num_ood: Number of out-of-distribution samples
        threshold_95_tpr: Threshold value at 95% TPR
        roc_curve_data: Dictionary with FPR, TPR, and thresholds for ROC curve
    """
    auroc: float
    aupr: float
    fpr_at_95_tpr: float
    scenario: str
    num_in_dist: int
    num_ood: int
    threshold_95_tpr: float
    roc_curve_data: Dict[str, np.ndarray]


class OODDetectionEvaluator:
    """Evaluator for out-of-distribution detection using uncertainty.
    
    This class implements comprehensive OOD detection evaluation using epistemic
    uncertainty as the detection score. It computes AUROC, AUPR, FPR@95%TPR,
    and provides ROC curve analysis.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize OOD detection evaluator.
        
        Args:
            device: Device for tensor computations
        """
        self.device = device or torch.device('cpu')
        
    def evaluate_ood_detection(self, 
                              model: UncertaintyMetaLearner,
                              in_dist_data: torch.Tensor,
                              ood_data: torch.Tensor,
                              scenario: str = "unknown",
                              num_samples: int = 100) -> OODResults:
        """Evaluate OOD detection performance.
        
        Args:
            model: Uncertainty-aware model
            in_dist_data: In-distribution test data [n_in, input_dim]
            ood_data: Out-of-distribution test data [n_ood, input_dim]
            scenario: Name of OOD scenario
            num_samples: Number of posterior samples for uncertainty estimation
            
        Returns:
            OODResults containing all evaluation metrics
            
        Raises:
            ValueError: If data shapes are incompatible
        """
        if not model.is_adapted:
            raise ValueError("Model must be adapted before OOD evaluation")
            
        # Validate inputs
        self._validate_ood_data(in_dist_data, ood_data)
        
        # Move data to device
        in_dist_data = in_dist_data.to(self.device)
        ood_data = ood_data.to(self.device)
        
        # Get uncertainty predictions
        in_dist_predictions = model.predict_with_uncertainty(in_dist_data, num_samples)
        ood_predictions = model.predict_with_uncertainty(ood_data, num_samples)
        
        # Use epistemic uncertainty as OOD score
        in_dist_scores = self._extract_ood_scores(in_dist_predictions)
        ood_scores = self._extract_ood_scores(ood_predictions)
        
        # Compute detection metrics
        return self._compute_detection_metrics(
            in_dist_scores, ood_scores, scenario,
            len(in_dist_data), len(ood_data)
        )
    
    def _extract_ood_scores(self, predictions: UncertaintyPrediction) -> torch.Tensor:
        """Extract OOD detection scores from uncertainty predictions.
        
        Uses epistemic uncertainty as the primary OOD detection score,
        as it should be higher for out-of-distribution inputs.
        
        Args:
            predictions: Uncertainty predictions
            
        Returns:
            OOD detection scores [batch_size]
        """
        # Use mean epistemic uncertainty across output dimensions
        if len(predictions.epistemic.shape) > 1:
            scores = predictions.epistemic.mean(dim=-1)
        else:
            scores = predictions.epistemic.squeeze()
            
        return scores
    
    def _compute_detection_metrics(self, 
                                  in_dist_scores: torch.Tensor,
                                  ood_scores: torch.Tensor,
                                  scenario: str,
                                  num_in_dist: int,
                                  num_ood: int) -> OODResults:
        """Compute comprehensive OOD detection metrics.
        
        Args:
            in_dist_scores: Uncertainty scores for in-distribution data
            ood_scores: Uncertainty scores for OOD data
            scenario: Name of OOD scenario
            num_in_dist: Number of in-distribution samples
            num_ood: Number of OOD samples
            
        Returns:
            OODResults with all computed metrics
        """
        # Convert to numpy for sklearn compatibility
        in_dist_np = in_dist_scores.detach().cpu().numpy()
        ood_np = ood_scores.detach().cpu().numpy()
        
        # Combine scores and create labels (0 = in-dist, 1 = OOD)
        all_scores = np.concatenate([in_dist_np, ood_np])
        all_labels = np.concatenate([
            np.zeros(len(in_dist_np)),
            np.ones(len(ood_np))
        ])
        
        # Compute ROC metrics
        roc_data = self._compute_roc_metrics(all_labels, all_scores)
        
        # Compute precision-recall metrics
        aupr = self._compute_aupr(all_labels, all_scores)
        
        # Find threshold at 95% TPR and compute FPR
        threshold_95_tpr, fpr_at_95_tpr = self._compute_fpr_at_tpr(
            all_labels, all_scores, target_tpr=0.95
        )
        
        return OODResults(
            auroc=roc_data['auroc'],
            aupr=aupr,
            fpr_at_95_tpr=fpr_at_95_tpr,
            scenario=scenario,
            num_in_dist=num_in_dist,
            num_ood=num_ood,
            threshold_95_tpr=threshold_95_tpr,
            roc_curve_data=roc_data
        )
    
    def _compute_roc_metrics(self, labels: np.ndarray, 
                            scores: np.ndarray) -> Dict[str, Any]:
        """Compute ROC curve and AUROC.
        
        Args:
            labels: Binary labels (0 = in-dist, 1 = OOD)
            scores: Detection scores (higher = more likely OOD)
            
        Returns:
            Dictionary with AUROC and ROC curve data
        """
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
        except ImportError:
            raise ImportError("sklearn is required for ROC computation")
        
        # Compute AUROC
        auroc = roc_auc_score(labels, scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        return {
            'auroc': auroc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def _compute_aupr(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve.
        
        Args:
            labels: Binary labels (0 = in-dist, 1 = OOD)
            scores: Detection scores
            
        Returns:
            AUPR value
        """
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            raise ImportError("sklearn is required for AUPR computation")
        
        return average_precision_score(labels, scores)
    
    def _compute_fpr_at_tpr(self, labels: np.ndarray, scores: np.ndarray,
                           target_tpr: float = 0.95) -> Tuple[float, float]:
        """Compute FPR at specified TPR threshold.
        
        Args:
            labels: Binary labels (0 = in-dist, 1 = OOD)
            scores: Detection scores
            target_tpr: Target true positive rate
            
        Returns:
            Tuple of (threshold_value, fpr_at_target_tpr)
        """
        try:
            from sklearn.metrics import roc_curve
        except ImportError:
            raise ImportError("sklearn is required for FPR@TPR computation")
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find index where TPR >= target_tpr
        idx = np.argmax(tpr >= target_tpr)
        
        if idx == 0 and tpr[0] < target_tpr:
            # If we can't achieve target TPR, return worst case
            return thresholds[0], 1.0
        
        return thresholds[idx], fpr[idx]
    
    def _validate_ood_data(self, in_dist_data: torch.Tensor, 
                          ood_data: torch.Tensor) -> None:
        """Validate OOD evaluation data.
        
        Args:
            in_dist_data: In-distribution data
            ood_data: Out-of-distribution data
            
        Raises:
            ValueError: If data is invalid
        """
        if len(in_dist_data.shape) != len(ood_data.shape):
            raise ValueError(
                f"Data dimension mismatch: in-dist {in_dist_data.shape}, "
                f"OOD {ood_data.shape}"
            )
        
        if in_dist_data.shape[-1] != ood_data.shape[-1]:
            raise ValueError(
                f"Input dimension mismatch: in-dist {in_dist_data.shape[-1]}, "
                f"OOD {ood_data.shape[-1]}"
            )
        
        if len(in_dist_data) == 0 or len(ood_data) == 0:
            raise ValueError("Empty datasets provided")
        
        # Check for NaN/Inf values
        if torch.any(torch.isnan(in_dist_data)) or torch.any(torch.isinf(in_dist_data)):
            raise ValueError("In-distribution data contains NaN or Inf values")
        
        if torch.any(torch.isnan(ood_data)) or torch.any(torch.isinf(ood_data)):
            raise ValueError("OOD data contains NaN or Inf values")
    
    def generate_roc_curve_plot_data(self, results: OODResults) -> Dict[str, np.ndarray]:
        """Generate data for ROC curve plotting.
        
        Args:
            results: OOD detection results
            
        Returns:
            Dictionary with plotting data
        """
        return {
            'fpr': results.roc_curve_data['fpr'],
            'tpr': results.roc_curve_data['tpr'],
            'auroc': results.auroc,
            'scenario': results.scenario,
            'diagonal_line': np.linspace(0, 1, 100)  # Perfect random classifier
        }
    
    def compute_detection_threshold(self, results: OODResults, 
                                   target_fpr: float = 0.05) -> float:
        """Compute detection threshold for specified FPR.
        
        Args:
            results: OOD detection results
            target_fpr: Target false positive rate
            
        Returns:
            Threshold value achieving target FPR
        """
        fpr = results.roc_curve_data['fpr']
        thresholds = results.roc_curve_data['thresholds']
        
        # Find threshold that achieves target FPR
        idx = np.argmax(fpr <= target_fpr)
        
        if idx == 0 and fpr[0] > target_fpr:
            # If we can't achieve target FPR, return highest threshold
            return thresholds[0]
        
        return thresholds[idx]


class OODScenarioGenerator(ABC):
    """Abstract base class for OOD scenario generators.
    
    This class defines the interface for generating different types of
    out-of-distribution scenarios for systematic evaluation.
    """
    
    @abstractmethod
    def generate_in_distribution_data(self, num_samples: int) -> torch.Tensor:
        """Generate in-distribution data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            In-distribution data [num_samples, input_dim]
        """
        pass
    
    @abstractmethod
    def generate_ood_data(self, num_samples: int) -> torch.Tensor:
        """Generate out-of-distribution data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD data [num_samples, input_dim]
        """
        pass
    
    @property
    @abstractmethod
    def scenario_name(self) -> str:
        """Get name of OOD scenario."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get description of OOD scenario."""
        pass


class SpatialExtrapolationGenerator(OODScenarioGenerator):
    """Generator for spatial extrapolation OOD scenarios.
    
    Creates OOD data by extending the spatial domain beyond the training region.
    For example, if training domain is [0, 1], OOD domain might be [-0.5, 0] ∪ [1, 1.5].
    """
    
    def __init__(self, 
                 in_dist_bounds: List[Tuple[float, float]],
                 extrapolation_factor: float = 0.5,
                 device: Optional[torch.device] = None):
        """Initialize spatial extrapolation generator.
        
        Args:
            in_dist_bounds: List of (min, max) bounds for each input dimension
            extrapolation_factor: Factor by which to extend domain
            device: Device for tensor operations
        """
        self.in_dist_bounds = in_dist_bounds
        self.extrapolation_factor = extrapolation_factor
        self.device = device or torch.device('cpu')
        self.input_dim = len(in_dist_bounds)
        
        # Compute OOD bounds
        self.ood_bounds = self._compute_ood_bounds()
    
    def _compute_ood_bounds(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Compute OOD bounds for extrapolation.
        
        Returns:
            List of ((left_min, left_max), (right_min, right_max)) for each dimension
        """
        ood_bounds = []
        for min_val, max_val in self.in_dist_bounds:
            domain_size = max_val - min_val
            extension = domain_size * self.extrapolation_factor
            
            # Left extrapolation region
            left_bounds = (min_val - extension, min_val)
            # Right extrapolation region  
            right_bounds = (max_val, max_val + extension)
            
            ood_bounds.append((left_bounds, right_bounds))
        
        return ood_bounds
    
    def generate_in_distribution_data(self, num_samples: int) -> torch.Tensor:
        """Generate in-distribution data within training bounds.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            In-distribution data [num_samples, input_dim]
        """
        data = torch.zeros(num_samples, self.input_dim, device=self.device)
        
        for i, (min_val, max_val) in enumerate(self.in_dist_bounds):
            data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        return data
    
    def generate_ood_data(self, num_samples: int) -> torch.Tensor:
        """Generate OOD data in extrapolation regions.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD data [num_samples, input_dim]
        """
        data = torch.zeros(num_samples, self.input_dim, device=self.device)
        
        for i, ((left_min, left_max), (right_min, right_max)) in enumerate(self.ood_bounds):
            # Randomly choose left or right extrapolation region
            left_samples = torch.rand(num_samples, device=self.device) < 0.5
            right_samples = ~left_samples
            
            # Generate samples in left region
            if left_samples.any():
                n_left = left_samples.sum()
                data[left_samples, i] = (
                    torch.rand(n_left, device=self.device) * (left_max - left_min) + left_min
                )
            
            # Generate samples in right region
            if right_samples.any():
                n_right = right_samples.sum()
                data[right_samples, i] = (
                    torch.rand(n_right, device=self.device) * (right_max - right_min) + right_min
                )
        
        return data
    
    @property
    def scenario_name(self) -> str:
        """Get scenario name."""
        return "spatial_extrapolation"
    
    @property
    def description(self) -> str:
        """Get scenario description."""
        return f"Spatial extrapolation beyond training domain by factor {self.extrapolation_factor}"


class InterpolationGapGenerator(OODScenarioGenerator):
    """Generator for interpolation gap OOD scenarios.
    
    Creates OOD data in regions that were missing from the training distribution.
    For example, if training data covers [0, 0.4] ∪ [0.6, 1], OOD data is in [0.4, 0.6].
    """
    
    def __init__(self,
                 full_bounds: List[Tuple[float, float]],
                 gap_regions: List[List[Tuple[float, float]]],
                 device: Optional[torch.device] = None):
        """Initialize interpolation gap generator.
        
        Args:
            full_bounds: Full domain bounds for each dimension
            gap_regions: List of gap regions for each dimension
            device: Device for tensor operations
        """
        self.full_bounds = full_bounds
        self.gap_regions = gap_regions
        self.device = device or torch.device('cpu')
        self.input_dim = len(full_bounds)
        
        # Compute in-distribution regions (complement of gaps)
        self.in_dist_regions = self._compute_in_dist_regions()
    
    def _compute_in_dist_regions(self) -> List[List[Tuple[float, float]]]:
        """Compute in-distribution regions as complement of gaps.
        
        Returns:
            List of in-distribution regions for each dimension
        """
        in_dist_regions = []
        
        for dim_idx, ((full_min, full_max), gaps) in enumerate(
            zip(self.full_bounds, self.gap_regions)
        ):
            # Sort gaps by start position
            sorted_gaps = sorted(gaps, key=lambda x: x[0])
            
            regions = []
            current_pos = full_min
            
            for gap_start, gap_end in sorted_gaps:
                if current_pos < gap_start:
                    regions.append((current_pos, gap_start))
                current_pos = max(current_pos, gap_end)
            
            # Add final region if needed
            if current_pos < full_max:
                regions.append((current_pos, full_max))
            
            in_dist_regions.append(regions)
        
        return in_dist_regions
    
    def generate_in_distribution_data(self, num_samples: int) -> torch.Tensor:
        """Generate in-distribution data outside gap regions.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            In-distribution data [num_samples, input_dim]
        """
        data = torch.zeros(num_samples, self.input_dim, device=self.device)
        
        for dim_idx, regions in enumerate(self.in_dist_regions):
            if not regions:
                raise ValueError(f"No in-distribution regions for dimension {dim_idx}")
            
            # Compute total length of all regions
            total_length = sum(end - start for start, end in regions)
            region_weights = [(end - start) / total_length for start, end in regions]
            
            # Sample region indices based on weights
            region_probs = torch.tensor(region_weights, device=self.device)
            region_indices = torch.multinomial(region_probs, num_samples, replacement=True)
            
            # Generate samples in selected regions
            for i in range(num_samples):
                region_idx = region_indices[i]
                start, end = regions[region_idx]
                data[i, dim_idx] = torch.rand(1, device=self.device) * (end - start) + start
        
        return data
    
    def generate_ood_data(self, num_samples: int) -> torch.Tensor:
        """Generate OOD data within gap regions.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD data [num_samples, input_dim]
        """
        data = torch.zeros(num_samples, self.input_dim, device=self.device)
        
        for dim_idx, gaps in enumerate(self.gap_regions):
            if not gaps:
                raise ValueError(f"No gap regions for dimension {dim_idx}")
            
            # Compute total length of all gaps
            total_length = sum(end - start for start, end in gaps)
            gap_weights = [(end - start) / total_length for start, end in gaps]
            
            # Sample gap indices based on weights
            gap_probs = torch.tensor(gap_weights, device=self.device)
            gap_indices = torch.multinomial(gap_probs, num_samples, replacement=True)
            
            # Generate samples in selected gaps
            for i in range(num_samples):
                gap_idx = gap_indices[i]
                start, end = gaps[gap_idx]
                data[i, dim_idx] = torch.rand(1, device=self.device) * (end - start) + start
        
        return data
    
    @property
    def scenario_name(self) -> str:
        """Get scenario name."""
        return "interpolation_gap"
    
    @property
    def description(self) -> str:
        """Get scenario description."""
        return "Interpolation in regions missing from training distribution"


class ParameterShiftGenerator(OODScenarioGenerator):
    """Generator for parameter shift OOD scenarios.
    
    Creates OOD data by shifting PDE parameters outside the training distribution.
    This tests the model's ability to detect novel physical parameter regimes.
    """
    
    def __init__(self,
                 spatial_bounds: List[Tuple[float, float]],
                 in_dist_param_ranges: Dict[str, Tuple[float, float]],
                 ood_param_ranges: Dict[str, Tuple[float, float]],
                 device: Optional[torch.device] = None):
        """Initialize parameter shift generator.
        
        Args:
            spatial_bounds: Spatial domain bounds for each dimension
            in_dist_param_ranges: In-distribution parameter ranges
            ood_param_ranges: Out-of-distribution parameter ranges
            device: Device for tensor operations
        """
        self.spatial_bounds = spatial_bounds
        self.in_dist_param_ranges = in_dist_param_ranges
        self.ood_param_ranges = ood_param_ranges
        self.device = device or torch.device('cpu')
        self.input_dim = len(spatial_bounds)
        
        # Validate parameter ranges
        if set(in_dist_param_ranges.keys()) != set(ood_param_ranges.keys()):
            raise ValueError("In-dist and OOD parameter ranges must have same keys")
    
    def generate_in_distribution_data(self, num_samples: int) -> torch.Tensor:
        """Generate in-distribution data with normal parameter ranges.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            In-distribution data [num_samples, input_dim + n_params]
        """
        # Generate spatial coordinates
        spatial_data = torch.zeros(num_samples, self.input_dim, device=self.device)
        for i, (min_val, max_val) in enumerate(self.spatial_bounds):
            spatial_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        # Generate in-distribution parameters
        param_data = torch.zeros(num_samples, len(self.in_dist_param_ranges), device=self.device)
        for i, (param_name, (min_val, max_val)) in enumerate(self.in_dist_param_ranges.items()):
            param_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        return torch.cat([spatial_data, param_data], dim=1)
    
    def generate_ood_data(self, num_samples: int) -> torch.Tensor:
        """Generate OOD data with shifted parameter ranges.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD data [num_samples, input_dim + n_params]
        """
        # Generate spatial coordinates (same as in-distribution)
        spatial_data = torch.zeros(num_samples, self.input_dim, device=self.device)
        for i, (min_val, max_val) in enumerate(self.spatial_bounds):
            spatial_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        # Generate OOD parameters
        param_data = torch.zeros(num_samples, len(self.ood_param_ranges), device=self.device)
        for i, (param_name, (min_val, max_val)) in enumerate(self.ood_param_ranges.items()):
            param_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        return torch.cat([spatial_data, param_data], dim=1)
    
    @property
    def scenario_name(self) -> str:
        """Get scenario name."""
        return "parameter_shift"
    
    @property
    def description(self) -> str:
        """Get scenario description."""
        return f"Parameter shift from {self.in_dist_param_ranges} to {self.ood_param_ranges}"


class BoundaryConditionShiftGenerator(OODScenarioGenerator):
    """Generator for boundary condition shift OOD scenarios.
    
    Creates OOD data by changing boundary condition types or values,
    testing the model's sensitivity to boundary condition changes.
    """
    
    def __init__(self,
                 spatial_bounds: List[Tuple[float, float]],
                 in_dist_bc_type: str = "dirichlet",
                 ood_bc_type: str = "neumann",
                 in_dist_bc_values: Tuple[float, float] = (0.0, 1.0),
                 ood_bc_values: Tuple[float, float] = (2.0, 3.0),
                 device: Optional[torch.device] = None):
        """Initialize boundary condition shift generator.
        
        Args:
            spatial_bounds: Spatial domain bounds
            in_dist_bc_type: In-distribution boundary condition type
            ood_bc_type: Out-of-distribution boundary condition type
            in_dist_bc_values: In-distribution BC values
            ood_bc_values: Out-of-distribution BC values
            device: Device for tensor operations
        """
        self.spatial_bounds = spatial_bounds
        self.in_dist_bc_type = in_dist_bc_type
        self.ood_bc_type = ood_bc_type
        self.in_dist_bc_values = in_dist_bc_values
        self.ood_bc_values = ood_bc_values
        self.device = device or torch.device('cpu')
        self.input_dim = len(spatial_bounds)
    
    def generate_in_distribution_data(self, num_samples: int) -> torch.Tensor:
        """Generate in-distribution data with normal boundary conditions.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            In-distribution data [num_samples, input_dim + 2]  # +2 for BC values
        """
        # Generate spatial coordinates
        spatial_data = torch.zeros(num_samples, self.input_dim, device=self.device)
        for i, (min_val, max_val) in enumerate(self.spatial_bounds):
            spatial_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        # Add boundary condition values
        bc_data = torch.zeros(num_samples, 2, device=self.device)
        bc_data[:, 0] = self.in_dist_bc_values[0]  # Left BC
        bc_data[:, 1] = self.in_dist_bc_values[1]  # Right BC
        
        return torch.cat([spatial_data, bc_data], dim=1)
    
    def generate_ood_data(self, num_samples: int) -> torch.Tensor:
        """Generate OOD data with shifted boundary conditions.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD data [num_samples, input_dim + 2]
        """
        # Generate spatial coordinates (same as in-distribution)
        spatial_data = torch.zeros(num_samples, self.input_dim, device=self.device)
        for i, (min_val, max_val) in enumerate(self.spatial_bounds):
            spatial_data[:, i] = torch.rand(num_samples, device=self.device) * (max_val - min_val) + min_val
        
        # Add OOD boundary condition values
        bc_data = torch.zeros(num_samples, 2, device=self.device)
        bc_data[:, 0] = self.ood_bc_values[0]  # Left BC
        bc_data[:, 1] = self.ood_bc_values[1]  # Right BC
        
        return torch.cat([spatial_data, bc_data], dim=1)
    
    @property
    def scenario_name(self) -> str:
        """Get scenario name."""
        return "boundary_condition_shift"
    
    @property
    def description(self) -> str:
        """Get scenario description."""
        return f"BC shift from {self.in_dist_bc_type}({self.in_dist_bc_values}) to {self.ood_bc_type}({self.ood_bc_values})"


class OODScenarioFactory:
    """Factory for creating different OOD scenario generators.
    
    This factory provides a convenient interface for creating and managing
    different types of OOD scenarios for systematic evaluation.
    """
    
    @staticmethod
    def create_spatial_extrapolation(domain_bounds: List[Tuple[float, float]],
                                   extrapolation_factor: float = 0.5,
                                   device: Optional[torch.device] = None) -> SpatialExtrapolationGenerator:
        """Create spatial extrapolation scenario.
        
        Args:
            domain_bounds: Training domain bounds
            extrapolation_factor: Factor by which to extend domain
            device: Device for computations
            
        Returns:
            SpatialExtrapolationGenerator instance
        """
        return SpatialExtrapolationGenerator(
            in_dist_bounds=domain_bounds,
            extrapolation_factor=extrapolation_factor,
            device=device
        )
    
    @staticmethod
    def create_interpolation_gap(full_bounds: List[Tuple[float, float]],
                               gap_fraction: float = 0.2,
                               device: Optional[torch.device] = None) -> InterpolationGapGenerator:
        """Create interpolation gap scenario.
        
        Args:
            full_bounds: Full domain bounds
            gap_fraction: Fraction of domain to use as gap
            device: Device for computations
            
        Returns:
            InterpolationGapGenerator instance
        """
        # Create centered gap for each dimension
        gap_regions = []
        for min_val, max_val in full_bounds:
            domain_size = max_val - min_val
            gap_size = domain_size * gap_fraction
            gap_start = min_val + (domain_size - gap_size) / 2
            gap_end = gap_start + gap_size
            gap_regions.append([(gap_start, gap_end)])
        
        return InterpolationGapGenerator(
            full_bounds=full_bounds,
            gap_regions=gap_regions,
            device=device
        )
    
    @staticmethod
    def create_parameter_shift(spatial_bounds: List[Tuple[float, float]],
                             param_name: str,
                             in_dist_range: Tuple[float, float],
                             shift_factor: float = 2.0,
                             device: Optional[torch.device] = None) -> ParameterShiftGenerator:
        """Create parameter shift scenario.
        
        Args:
            spatial_bounds: Spatial domain bounds
            param_name: Name of parameter to shift
            in_dist_range: In-distribution parameter range
            shift_factor: Factor by which to shift parameter
            device: Device for computations
            
        Returns:
            ParameterShiftGenerator instance
        """
        # Compute OOD range by shifting
        in_min, in_max = in_dist_range
        param_center = (in_min + in_max) / 2
        param_width = in_max - in_min
        
        # Shift range by shift_factor * width
        shift_amount = shift_factor * param_width
        ood_range = (in_min + shift_amount, in_max + shift_amount)
        
        return ParameterShiftGenerator(
            spatial_bounds=spatial_bounds,
            in_dist_param_ranges={param_name: in_dist_range},
            ood_param_ranges={param_name: ood_range},
            device=device
        )
    
    @staticmethod
    def create_boundary_condition_shift(spatial_bounds: List[Tuple[float, float]],
                                      in_dist_values: Tuple[float, float] = (0.0, 1.0),
                                      shift_factor: float = 3.0,
                                      device: Optional[torch.device] = None) -> BoundaryConditionShiftGenerator:
        """Create boundary condition shift scenario.
        
        Args:
            spatial_bounds: Spatial domain bounds
            in_dist_values: In-distribution BC values
            shift_factor: Factor by which to shift BC values
            device: Device for computations
            
        Returns:
            BoundaryConditionShiftGenerator instance
        """
        # Shift BC values
        ood_values = (
            in_dist_values[0] + shift_factor,
            in_dist_values[1] + shift_factor
        )
        
        return BoundaryConditionShiftGenerator(
            spatial_bounds=spatial_bounds,
            in_dist_bc_values=in_dist_values,
            ood_bc_values=ood_values,
            device=device
        )
    
    @staticmethod
    def get_all_scenarios(spatial_bounds: List[Tuple[float, float]],
                         device: Optional[torch.device] = None) -> Dict[str, OODScenarioGenerator]:
        """Get all available OOD scenarios.
        
        Args:
            spatial_bounds: Spatial domain bounds
            device: Device for computations
            
        Returns:
            Dictionary mapping scenario names to generators
        """
        return {
            'spatial_extrapolation': OODScenarioFactory.create_spatial_extrapolation(
                spatial_bounds, device=device
            ),
            'interpolation_gap': OODScenarioFactory.create_interpolation_gap(
                spatial_bounds, device=device
            ),
            'parameter_shift': OODScenarioFactory.create_parameter_shift(
                spatial_bounds, 'diffusion_coeff', (0.1, 1.0), device=device
            ),
            'boundary_condition_shift': OODScenarioFactory.create_boundary_condition_shift(
                spatial_bounds, device=device
            )
        }


def evaluate_all_ood_scenarios(model: UncertaintyMetaLearner,
                              spatial_bounds: List[Tuple[float, float]],
                              num_samples: int = 500,
                              device: Optional[torch.device] = None) -> Dict[str, OODResults]:
    """Evaluate model on all OOD scenarios.
    
    Args:
        model: Uncertainty-aware model
        spatial_bounds: Spatial domain bounds
        num_samples: Number of samples per scenario
        device: Device for computations
        
    Returns:
        Dictionary mapping scenario names to results
    """
    evaluator = OODDetectionEvaluator(device=device)
    scenarios = OODScenarioFactory.get_all_scenarios(spatial_bounds, device=device)
    
    results = {}
    for scenario_name, generator in scenarios.items():
        # Generate data
        in_dist_data = generator.generate_in_distribution_data(num_samples)
        ood_data = generator.generate_ood_data(num_samples)
        
        # Evaluate
        result = evaluator.evaluate_ood_detection(
            model, in_dist_data, ood_data, scenario_name
        )
        results[scenario_name] = result
    
    return results