"""Physics-informed priors for Bayesian neural networks.

This module implements physics-informed prior distributions that encode
PDE structure, boundary conditions, and symmetries into the Bayesian
meta-learning framework.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from enum import Enum

from .variational_layers import GaussianPrior


class PDEType(Enum):
    """Enumeration of supported PDE types."""
    HEAT = "heat"
    BURGERS = "burgers"
    POISSON = "poisson"
    NAVIER_STOKES = "navier_stokes"
    WAVE = "wave"
    REACTION_DIFFUSION = "reaction_diffusion"


class BoundaryConditionType(Enum):
    """Enumeration of boundary condition types."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    PERIODIC = "periodic"


class PhysicsInformedPrior(GaussianPrior):
    """Physics-informed prior that encodes PDE structure and constraints.
    
    This prior adapts the mean and variance based on:
    1. PDE type and expected solution characteristics
    2. Boundary conditions and their enforcement
    3. Physical symmetries and conservation laws
    4. Problem-specific scaling and dimensionality
    
    Args:
        pde_type: Type of PDE being solved
        input_dim: Spatial dimension of the problem
        output_dim: Number of solution components
        domain_bounds: Physical domain boundaries
        boundary_conditions: List of boundary condition specifications
        base_mean: Base prior mean (default: 0.0)
        base_std: Base prior standard deviation (default: 1.0)
        physics_weight: Weight for physics-informed adjustments (default: 1.0)
    """
    
    def __init__(self, 
                 pde_type: Union[PDEType, str],
                 input_dim: int = 2,
                 output_dim: int = 1,
                 domain_bounds: Optional[List[Tuple[float, float]]] = None,
                 boundary_conditions: Optional[List[Dict]] = None,
                 base_mean: float = 0.0,
                 base_std: float = 1.0,
                 physics_weight: float = 1.0):
        
        super().__init__(base_mean, base_std)
        
        # Convert string to enum if needed
        if isinstance(pde_type, str):
            pde_type = PDEType(pde_type.lower())
        
        self.pde_type = pde_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Set default domain bounds if not provided
        if domain_bounds is None:
            domain_bounds = [(0.0, 1.0)] * input_dim
        self.domain_bounds = domain_bounds
        
        # Parse boundary conditions
        self.boundary_conditions = boundary_conditions or []
        
        # Compute physics-informed adjustments
        self._compute_physics_adjustments()
    
    def _compute_physics_adjustments(self) -> None:
        """Compute physics-informed adjustments to prior parameters."""
        # Get PDE-specific scaling factors
        self.pde_scaling = self._get_pde_scaling()
        
        # Get boundary condition adjustments
        self.boundary_scaling = self._get_boundary_scaling()
        
        # Get symmetry-based adjustments
        self.symmetry_scaling = self._get_symmetry_scaling()
        
        # Compute domain characteristic length scale
        self.length_scale = self._compute_length_scale()
    
    def _get_pde_scaling(self) -> Dict[str, float]:
        """Get scaling factors based on PDE type.
        
        Returns:
            Dictionary with mean and std scaling factors
        """
        # PDE-specific scaling based on typical solution magnitudes
        pde_scalings = {
            PDEType.HEAT: {'mean_scale': 0.0, 'std_scale': 0.5},
            PDEType.BURGERS: {'mean_scale': 0.0, 'std_scale': 1.0},
            PDEType.POISSON: {'mean_scale': 0.0, 'std_scale': 0.3},
            PDEType.NAVIER_STOKES: {'mean_scale': 0.0, 'std_scale': 1.5},
            PDEType.WAVE: {'mean_scale': 0.0, 'std_scale': 0.8},
            PDEType.REACTION_DIFFUSION: {'mean_scale': 0.5, 'std_scale': 0.7}
        }
        
        return pde_scalings.get(self.pde_type, {'mean_scale': 0.0, 'std_scale': 1.0})
    
    def _get_boundary_scaling(self) -> Dict[str, float]:
        """Get scaling adjustments based on boundary conditions.
        
        Returns:
            Dictionary with boundary-based scaling factors
        """
        if not self.boundary_conditions:
            return {'mean_scale': 1.0, 'std_scale': 1.0}
        
        # Analyze boundary condition types
        bc_types = [bc.get('type', 'dirichlet') for bc in self.boundary_conditions]
        bc_values = [bc.get('value', 0.0) for bc in self.boundary_conditions]
        
        # Adjust based on boundary condition characteristics
        mean_adjustment = 0.0
        std_adjustment = 1.0
        
        # Dirichlet BCs provide strong constraints
        if 'dirichlet' in bc_types:
            dirichlet_values = [val for bc, val in zip(bc_types, bc_values) if bc == 'dirichlet']
            if dirichlet_values:
                mean_adjustment = sum(dirichlet_values) / len(dirichlet_values) * 0.1
                std_adjustment *= 0.8  # Reduce uncertainty near boundaries
        
        # Neumann BCs affect gradient constraints
        if 'neumann' in bc_types:
            std_adjustment *= 1.2  # Slightly increase uncertainty for gradient BCs
        
        # Periodic BCs enforce symmetry
        if 'periodic' in bc_types:
            std_adjustment *= 0.9  # Reduce uncertainty due to periodicity constraint
        
        return {'mean_scale': mean_adjustment, 'std_scale': std_adjustment}
    
    def _get_symmetry_scaling(self) -> Dict[str, float]:
        """Get scaling based on physical symmetries.
        
        Returns:
            Dictionary with symmetry-based scaling factors
        """
        symmetry_factors = {'mean_scale': 1.0, 'std_scale': 1.0}
        
        # Adjust based on PDE symmetries
        if self.pde_type in [PDEType.HEAT, PDEType.POISSON]:
            # These PDEs often have reflection symmetry
            symmetry_factors['std_scale'] *= 0.95
        
        elif self.pde_type == PDEType.WAVE:
            # Wave equations have time-reversal symmetry
            symmetry_factors['std_scale'] *= 0.9
        
        elif self.pde_type == PDEType.NAVIER_STOKES:
            # Incompressible NS has divergence-free constraint
            symmetry_factors['std_scale'] *= 1.1
        
        return symmetry_factors
    
    def _compute_length_scale(self) -> float:
        """Compute characteristic length scale of the domain.
        
        Returns:
            Characteristic length scale
        """
        domain_sizes = [high - low for low, high in self.domain_bounds]
        return math.sqrt(sum(size**2 for size in domain_sizes))
    
    def get_mean(self, shape: torch.Size) -> torch.Tensor:
        """Get physics-informed prior mean for given parameter shape.
        
        Args:
            shape: Parameter tensor shape
            
        Returns:
            Prior mean tensor
        """
        # Base mean
        base_mean = torch.full(shape, self.mean)
        
        # Apply physics-informed adjustments
        pde_adjustment = self.pde_scaling['mean_scale']
        boundary_adjustment = self.boundary_scaling['mean_scale']
        
        # Combine adjustments
        physics_mean = base_mean + self.physics_weight * (
            pde_adjustment + boundary_adjustment
        )
        
        return physics_mean
    
    def get_std(self, shape: torch.Size) -> torch.Tensor:
        """Get physics-informed prior standard deviation for given parameter shape.
        
        Args:
            shape: Parameter tensor shape
            
        Returns:
            Prior standard deviation tensor
        """
        # Base standard deviation
        base_std = torch.full(shape, self.std)
        
        # Apply physics-informed scaling
        pde_scale = self.pde_scaling['std_scale']
        boundary_scale = self.boundary_scaling['std_scale']
        symmetry_scale = self.symmetry_scaling['std_scale']
        
        # Length scale adjustment (smaller domains need tighter priors)
        length_scale_factor = 1.0 / (1.0 + self.length_scale)
        
        # Combine all scaling factors
        total_scale = pde_scale * boundary_scale * symmetry_scale * length_scale_factor
        
        # Apply physics weight
        physics_std = base_std * (1.0 + self.physics_weight * (total_scale - 1.0))
        
        # Ensure positive standard deviation
        physics_std = torch.clamp(physics_std, min=1e-6)
        
        return physics_std
    
    def get_layer_specific_prior(self, layer_index: int, total_layers: int, 
                                shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get layer-specific prior parameters.
        
        Different layers in the network may need different prior characteristics
        based on their role in representing the solution.
        
        Args:
            layer_index: Index of the layer (0-based)
            total_layers: Total number of layers in the network
            shape: Parameter tensor shape
            
        Returns:
            Tuple of (prior_mean, prior_std)
        """
        # Get base physics-informed parameters
        base_mean = self.get_mean(shape)
        base_std = self.get_std(shape)
        
        # Layer-specific adjustments
        if layer_index == 0:
            # Input layer: encode boundary and initial conditions more strongly
            layer_std = base_std * 0.8
        elif layer_index == total_layers - 1:
            # Output layer: should be more constrained to physical values
            layer_std = base_std * 0.6
        else:
            # Hidden layers: allow more flexibility for feature learning
            layer_std = base_std * 1.2
        
        return base_mean, layer_std
    
    def compute_physics_regularization(self, network: nn.Module, 
                                     sample_points: torch.Tensor) -> torch.Tensor:
        """Compute physics-based regularization term.
        
        This adds a regularization term that encourages the network to
        satisfy physical constraints even in the prior.
        
        Args:
            network: Neural network to regularize
            sample_points: Points to evaluate physics constraints
            
        Returns:
            Physics regularization loss
        """
        if sample_points.numel() == 0:
            return torch.tensor(0.0, device=sample_points.device)
        
        # Enable gradient computation for physics constraints
        sample_points.requires_grad_(True)
        
        # Forward pass
        predictions = network(sample_points)
        
        # Compute physics-based regularization based on PDE type
        if self.pde_type == PDEType.HEAT:
            reg_loss = self._heat_equation_regularization(sample_points, predictions)
        elif self.pde_type == PDEType.POISSON:
            reg_loss = self._poisson_equation_regularization(sample_points, predictions)
        elif self.pde_type == PDEType.BURGERS:
            reg_loss = self._burgers_equation_regularization(sample_points, predictions)
        else:
            # Generic smoothness regularization
            reg_loss = self._smoothness_regularization(sample_points, predictions)
        
        return reg_loss * self.physics_weight
    
    def _heat_equation_regularization(self, points: torch.Tensor, 
                                    predictions: torch.Tensor) -> torch.Tensor:
        """Regularization for heat equation: ∂u/∂t = α∇²u."""
        if points.shape[-1] < 2:  # Need at least space and time
            return torch.tensor(0.0, device=points.device)
        
        # Compute gradients
        u_t = torch.autograd.grad(predictions.sum(), points, create_graph=True)[0][:, -1]  # Time derivative
        
        # Compute Laplacian (spatial derivatives)
        laplacian = torch.zeros_like(predictions.squeeze())
        for i in range(points.shape[-1] - 1):  # Exclude time dimension
            u_x = torch.autograd.grad(predictions.sum(), points, create_graph=True)[0][:, i]
            u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, i]
            laplacian += u_xx
        
        # Heat equation residual (assuming α = 1)
        residual = u_t - laplacian
        return torch.mean(residual**2)
    
    def _poisson_equation_regularization(self, points: torch.Tensor, 
                                       predictions: torch.Tensor) -> torch.Tensor:
        """Regularization for Poisson equation: ∇²u = f."""
        # Compute Laplacian
        laplacian = torch.zeros_like(predictions.squeeze())
        for i in range(points.shape[-1]):
            u_x = torch.autograd.grad(predictions.sum(), points, create_graph=True)[0][:, i]
            u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, i]
            laplacian += u_xx
        
        # Encourage smoothness (Laplacian close to zero for homogeneous case)
        return torch.mean(laplacian**2)
    
    def _burgers_equation_regularization(self, points: torch.Tensor, 
                                       predictions: torch.Tensor) -> torch.Tensor:
        """Regularization for Burgers equation: ∂u/∂t + u∂u/∂x = ν∇²u."""
        if points.shape[-1] < 2:
            return torch.tensor(0.0, device=points.device)
        
        # Compute derivatives
        grads = torch.autograd.grad(predictions.sum(), points, create_graph=True)[0]
        u_t = grads[:, -1]  # Time derivative
        u_x = grads[:, 0]   # Spatial derivative
        
        # Second spatial derivative
        u_xx = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0][:, 0]
        
        # Burgers equation residual (assuming ν = 0.01)
        nu = 0.01
        residual = u_t + predictions.squeeze() * u_x - nu * u_xx
        return torch.mean(residual**2)
    
    def _smoothness_regularization(self, points: torch.Tensor, 
                                 predictions: torch.Tensor) -> torch.Tensor:
        """Generic smoothness regularization based on gradient magnitude."""
        grads = torch.autograd.grad(predictions.sum(), points, create_graph=True)[0]
        grad_magnitude = torch.norm(grads, dim=-1)
        return torch.mean(grad_magnitude**2)
    
    def update_from_task(self, task_info: Dict) -> None:
        """Update prior parameters based on specific task information.
        
        Args:
            task_info: Dictionary containing task-specific information
        """
        # Update boundary conditions if provided
        if 'boundary_conditions' in task_info:
            self.boundary_conditions = task_info['boundary_conditions']
            self.boundary_scaling = self._get_boundary_scaling()
        
        # Update domain bounds if provided
        if 'domain_bounds' in task_info:
            self.domain_bounds = task_info['domain_bounds']
            self.length_scale = self._compute_length_scale()
        
        # Update PDE parameters if provided
        if 'pde_parameters' in task_info:
            # Could adjust scaling based on PDE parameters (e.g., diffusion coefficient)
            pde_params = task_info['pde_parameters']
            if 'diffusion_coeff' in pde_params:
                # Adjust std based on diffusion coefficient
                diff_coeff = pde_params['diffusion_coeff']
                self.pde_scaling['std_scale'] *= math.sqrt(diff_coeff)


class LaplacePrior(PhysicsInformedPrior):
    """Laplace approximation-based prior for transfer learning.
    
    This prior uses a Laplace approximation around the MAP estimate
    from previous tasks to inform the prior for new tasks.
    """
    
    def __init__(self, map_estimate: torch.Tensor, hessian: torch.Tensor, 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.map_estimate = map_estimate
        self.hessian = hessian
        
        # Compute precision matrix (inverse of covariance)
        self.precision = hessian
        
        # Compute covariance (with regularization for numerical stability)
        regularization = 1e-6 * torch.eye(hessian.shape[0], device=hessian.device)
        self.covariance = torch.inverse(hessian + regularization)
    
    def get_mean(self, shape: torch.Size) -> torch.Tensor:
        """Get Laplace prior mean."""
        if shape.numel() == self.map_estimate.numel():
            return self.map_estimate.view(shape)
        else:
            # Fallback to physics-informed mean for different shapes
            return super().get_mean(shape)
    
    def get_std(self, shape: torch.Size) -> torch.Tensor:
        """Get Laplace prior standard deviation."""
        if shape.numel() == self.covariance.shape[0]:
            # Extract diagonal of covariance matrix
            std = torch.sqrt(torch.diag(self.covariance))
            return std.view(shape)
        else:
            # Fallback to physics-informed std for different shapes
            return super().get_std(shape)


def create_physics_informed_prior(pde_type: str, 
                                input_dim: int = 2,
                                output_dim: int = 1,
                                **kwargs) -> PhysicsInformedPrior:
    """Factory function to create physics-informed priors.
    
    Args:
        pde_type: Type of PDE ('heat', 'burgers', 'poisson', etc.)
        input_dim: Spatial dimension
        output_dim: Number of solution components
        **kwargs: Additional arguments for PhysicsInformedPrior
        
    Returns:
        PhysicsInformedPrior instance
    """
    return PhysicsInformedPrior(
        pde_type=pde_type,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )


def create_boundary_conditions(bc_specs: List[Dict]) -> List[Dict]:
    """Create standardized boundary condition specifications.
    
    Args:
        bc_specs: List of boundary condition specifications
        
    Returns:
        Standardized boundary condition list
    """
    standardized_bcs = []
    
    for bc_spec in bc_specs:
        bc = {
            'type': bc_spec.get('type', 'dirichlet').lower(),
            'location': bc_spec.get('location', 'boundary'),
            'value': bc_spec.get('value', 0.0),
            'function': bc_spec.get('function', None)
        }
        
        # Validate boundary condition type
        if bc['type'] not in ['dirichlet', 'neumann', 'robin', 'periodic']:
            raise ValueError(f"Unsupported boundary condition type: {bc['type']}")
        
        standardized_bcs.append(bc)
    
    return standardized_bcs