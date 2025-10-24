"""
Comprehensive noisy problem suite for uncertainty quantification evaluation.

This module extends existing PDE problems with controllable noise injection
and adds new problems like Reaction-Diffusion equations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from abc import ABC, abstractmethod
import deepxde as dde

from .noise_injection import (
    NoiseInjectionFramework, NoiseType, BaseNoiseInjector
)


class NoisyPDEProblem:
    """
    Wrapper for PDE problems with noise injection capabilities.
    
    This class wraps existing PDE problems and adds noise to observations
    while maintaining the original PDE structure and boundary conditions.
    """
    
    def __init__(self, base_problem, noise_injector: BaseNoiseInjector,
                 problem_name: str = "unknown"):
        """
        Initialize noisy PDE problem.
        
        Args:
            base_problem: Original PDE problem (from src.pde)
            noise_injector: Noise injection method
            problem_name: Descriptive name for the problem
        """
        self.base_problem = base_problem
        self.noise_injector = noise_injector
        self.problem_name = problem_name
        
        # Copy essential attributes from base problem
        self.pde = base_problem.pde
        self.geom = base_problem.geom
        self.bbox = base_problem.bbox
        self.output_dim = base_problem.output_dim
        self.input_dim = base_problem.input_dim
        
        # Store original reference solution if available
        self.clean_ref_sol = getattr(base_problem, 'ref_sol', None)
        self.clean_ref_data = getattr(base_problem, 'ref_data', None)
        
        # Create noisy versions
        self._create_noisy_reference_data()
    
    def _create_noisy_reference_data(self):
        """Create noisy versions of reference data."""
        if self.clean_ref_data is not None:
            # Extract clean solution values
            clean_solution = torch.tensor(self.clean_ref_data[:, self.input_dim:], 
                                        dtype=torch.float32)
            
            # Inject noise
            noisy_solution = self.noise_injector.inject_noise(clean_solution)
            
            # Create noisy reference data
            self.noisy_ref_data = self.clean_ref_data.copy()
            self.noisy_ref_data[:, self.input_dim:] = noisy_solution.numpy()
        else:
            self.noisy_ref_data = None
    
    def get_noisy_observations(self, x_points: torch.Tensor) -> torch.Tensor:
        """
        Get noisy observations at specified points.
        
        Args:
            x_points: Input points [N, input_dim]
            
        Returns:
            Noisy observations [N, output_dim]
        """
        if self.clean_ref_sol is not None:
            # Use analytical solution if available
            clean_values = torch.tensor(self.clean_ref_sol(x_points.numpy()), 
                                      dtype=torch.float32)
            if clean_values.dim() == 1:
                clean_values = clean_values.unsqueeze(1)
        else:
            # Use interpolation from reference data
            if self.clean_ref_data is None:
                raise ValueError("No reference solution or data available")
            
            # Simple nearest neighbor interpolation (could be improved)
            from scipy.spatial.distance import cdist
            distances = cdist(x_points.numpy(), 
                            self.clean_ref_data[:, :self.input_dim])
            nearest_indices = np.argmin(distances, axis=1)
            clean_values = torch.tensor(
                self.clean_ref_data[nearest_indices, self.input_dim:],
                dtype=torch.float32
            )
        
        # Inject noise
        return self.noise_injector.inject_noise(clean_values)
    
    def get_noise_statistics(self) -> Dict[str, Any]:
        """Get statistics about the injected noise."""
        stats = self.noise_injector.get_noise_statistics()
        stats['problem_name'] = self.problem_name
        return stats


class ReactionDiffusionPDE:
    """
    Reaction-Diffusion equation implementation.
    
    Implements: ∂u/∂t = D∇²u + f(u) where f(u) is the reaction term.
    """
    
    def __init__(self, diffusion_coeff: float = 0.1, 
                 reaction_type: str = "fisher_kpp",
                 domain_bounds: List[float] = [0, 1, 0, 1, 0, 1],
                 reaction_params: Optional[Dict[str, float]] = None):
        """
        Initialize Reaction-Diffusion PDE.
        
        Args:
            diffusion_coeff: Diffusion coefficient D
            reaction_type: Type of reaction term ("fisher_kpp", "bistable", "cubic")
            domain_bounds: [x_min, x_max, y_min, y_max, t_min, t_max]
            reaction_params: Parameters for reaction term
        """
        self.diffusion_coeff = diffusion_coeff
        self.reaction_type = reaction_type
        self.domain_bounds = domain_bounds
        self.reaction_params = reaction_params or {}
        
        # Set up geometry
        self.bbox = domain_bounds
        spatial_domain = dde.geometry.Rectangle(
            xmin=[domain_bounds[0], domain_bounds[2]], 
            xmax=[domain_bounds[1], domain_bounds[3]]
        )
        time_domain = dde.geometry.TimeDomain(domain_bounds[4], domain_bounds[5])
        self.geom = dde.geometry.GeometryXTime(spatial_domain, time_domain)
        
        # Output dimension
        self.output_dim = 1
        self.input_dim = 3  # x, y, t
        
        # Set up PDE
        self.pde = self._create_pde()
        
        # Create reference solution (analytical for simple cases)
        self.ref_sol = self._create_reference_solution()
    
    def _create_pde(self):
        """Create the PDE function."""
        def reaction_diffusion_pde(x, u):
            # Spatial derivatives
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            laplacian = u_xx + u_yy
            
            # Time derivative
            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            
            # Reaction term
            reaction = self._reaction_term(u)
            
            # PDE: u_t - D*∇²u - f(u) = 0
            return u_t - self.diffusion_coeff * laplacian - reaction
        
        return reaction_diffusion_pde
    
    def _reaction_term(self, u):
        """Compute reaction term f(u)."""
        if self.reaction_type == "fisher_kpp":
            # Fisher-KPP: f(u) = r*u*(1-u)
            r = self.reaction_params.get('growth_rate', 1.0)
            return r * u * (1 - u)
        
        elif self.reaction_type == "bistable":
            # Bistable: f(u) = r*u*(1-u)*(u-a)
            r = self.reaction_params.get('growth_rate', 1.0)
            a = self.reaction_params.get('threshold', 0.3)
            return r * u * (1 - u) * (u - a)
        
        elif self.reaction_type == "cubic":
            # Cubic: f(u) = r*u - u³
            r = self.reaction_params.get('growth_rate', 1.0)
            return r * u - u**3
        
        else:
            raise ValueError(f"Unknown reaction type: {self.reaction_type}")
    
    def _create_reference_solution(self):
        """Create reference solution (analytical when possible)."""
        if self.reaction_type == "fisher_kpp":
            # Traveling wave solution for Fisher-KPP
            def ref_sol(x):
                # Simple traveling wave: u(x,y,t) = 1/(1 + exp(k*(x - c*t)))
                r = self.reaction_params.get('growth_rate', 1.0)
                c = 2 * np.sqrt(r * self.diffusion_coeff)  # Wave speed
                k = np.sqrt(r / self.diffusion_coeff)       # Wave steepness
                
                wave_arg = k * (x[:, 0] - c * x[:, 2])
                return 1.0 / (1.0 + np.exp(wave_arg))
        else:
            # For other types, use a simple Gaussian initial condition evolution
            def ref_sol(x):
                # Gaussian blob that diffuses over time
                x_center, y_center = 0.5, 0.5
                sigma_0 = 0.1
                t = x[:, 2]
                
                # Diffusion spreads the Gaussian
                sigma_t = np.sqrt(sigma_0**2 + 2 * self.diffusion_coeff * t)
                
                gaussian = np.exp(-((x[:, 0] - x_center)**2 + (x[:, 1] - y_center)**2) / 
                                (2 * sigma_t**2))
                
                # Add reaction effects (simplified)
                if self.reaction_type == "cubic":
                    r = self.reaction_params.get('growth_rate', 1.0)
                    gaussian *= np.exp(r * t / 2)  # Exponential growth
                
                return np.clip(gaussian, 0, 1)
        
        return ref_sol


class NoisyProblemFactory:
    """
    Factory for creating noisy PDE problems systematically.
    
    This factory creates noisy versions of standard PDE problems
    for systematic evaluation of uncertainty quantification methods.
    """
    
    def __init__(self):
        self.noise_framework = NoiseInjectionFramework()
        self._problem_registry = {}
        self._register_standard_problems()
    
    def _register_standard_problems(self):
        """Register standard PDE problems."""
        # Import PDE classes (avoiding circular imports)
        try:
            from ..pde.heat import Heat2D_VaryingCoef
            from ..pde.burgers import Burgers1D
            from ..pde.poisson import Poisson1D, Poisson2D_Classic
            from ..pde.ns import NS2D_Classic
            
            self._problem_registry = {
                'heat_2d': Heat2D_VaryingCoef,
                'burgers_1d': Burgers1D,
                'poisson_1d': Poisson1D,
                'poisson_2d': Poisson2D_Classic,
                'navier_stokes_2d': NS2D_Classic,
                'reaction_diffusion': ReactionDiffusionPDE
            }
        except ImportError as e:
            print(f"Warning: Could not import some PDE classes: {e}")
            # Register only what's available
            self._problem_registry = {
                'reaction_diffusion': ReactionDiffusionPDE
            }
    
    def create_noisy_problem(self, problem_type: str, noise_type: NoiseType,
                           noise_level: float, problem_params: Optional[Dict] = None,
                           noise_params: Optional[Dict] = None,
                           random_seed: Optional[int] = None) -> NoisyPDEProblem:
        """
        Create a noisy PDE problem.
        
        Args:
            problem_type: Type of PDE problem ('heat_2d', 'burgers_1d', etc.)
            noise_type: Type of noise to inject
            noise_level: Noise level (σ)
            problem_params: Parameters for the base PDE problem
            noise_params: Additional parameters for noise injection
            random_seed: Random seed for reproducibility
            
        Returns:
            NoisyPDEProblem instance
        """
        if problem_type not in self._problem_registry:
            raise ValueError(f"Unknown problem type: {problem_type}. "
                           f"Available: {list(self._problem_registry.keys())}")
        
        # Create base problem
        problem_class = self._problem_registry[problem_type]
        problem_params = problem_params or {}
        
        try:
            base_problem = problem_class(**problem_params)
        except Exception as e:
            print(f"Warning: Could not create {problem_type} with params {problem_params}: {e}")
            # Try with default parameters
            base_problem = problem_class()
        
        # Create noise injector
        noise_params = noise_params or {}
        noise_injector = self.noise_framework.create_injector(
            noise_type=noise_type,
            noise_level=noise_level,
            random_seed=random_seed,
            **noise_params
        )
        
        # Create noisy problem
        return NoisyPDEProblem(
            base_problem=base_problem,
            noise_injector=noise_injector,
            problem_name=f"{problem_type}_{noise_type.value}_sigma_{noise_level}"
        )
    
    def create_comprehensive_test_suite(self, 
                                      problem_types: Optional[List[str]] = None,
                                      noise_levels: Optional[List[float]] = None,
                                      random_seed: Optional[int] = None) -> Dict[str, NoisyPDEProblem]:
        """
        Create comprehensive test suite with all combinations.
        
        Args:
            problem_types: List of problem types to include (default: all available)
            noise_levels: List of noise levels to test (default: all supported)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping problem names to NoisyPDEProblem instances
        """
        if problem_types is None:
            problem_types = list(self._problem_registry.keys())
        
        if noise_levels is None:
            noise_levels = self.noise_framework.SUPPORTED_NOISE_LEVELS
        
        test_suite = {}
        
        for problem_type in problem_types:
            for noise_level in noise_levels:
                for noise_type in NoiseType:
                    try:
                        # Create problem with specific noise configuration
                        noise_params = {}
                        if noise_type == NoiseType.OUTLIER_CONTAMINATION:
                            noise_params = {'outlier_fraction': 0.05, 'outlier_scale': 10.0}
                        
                        noisy_problem = self.create_noisy_problem(
                            problem_type=problem_type,
                            noise_type=noise_type,
                            noise_level=noise_level,
                            noise_params=noise_params,
                            random_seed=random_seed
                        )
                        
                        # Store with descriptive key
                        key = f"{problem_type}_{noise_type.value}_sigma_{noise_level}"
                        test_suite[key] = noisy_problem
                        
                    except Exception as e:
                        print(f"Warning: Could not create {problem_type} with "
                              f"{noise_type.value} noise σ={noise_level}: {e}")
                        continue
        
        return test_suite
    
    def get_available_problems(self) -> List[str]:
        """Get list of available problem types."""
        return list(self._problem_registry.keys())
    
    def validate_problem_suite(self, test_suite: Dict[str, NoisyPDEProblem],
                             num_test_points: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Validate that all problems in test suite work correctly.
        
        Args:
            test_suite: Dictionary of noisy problems
            num_test_points: Number of points to test
            
        Returns:
            Validation results for each problem
        """
        validation_results = {}
        
        for problem_name, noisy_problem in test_suite.items():
            try:
                # Generate random test points in domain
                bbox = noisy_problem.bbox
                if len(bbox) == 2:  # 1D problem
                    test_points = torch.rand(num_test_points, 1)
                    test_points = test_points * (bbox[1] - bbox[0]) + bbox[0]
                elif len(bbox) == 4:  # 2D problem
                    test_points = torch.rand(num_test_points, 2)
                    test_points[:, 0] = test_points[:, 0] * (bbox[1] - bbox[0]) + bbox[0]
                    test_points[:, 1] = test_points[:, 1] * (bbox[3] - bbox[2]) + bbox[2]
                elif len(bbox) == 6:  # 2D + time problem
                    test_points = torch.rand(num_test_points, 3)
                    test_points[:, 0] = test_points[:, 0] * (bbox[1] - bbox[0]) + bbox[0]
                    test_points[:, 1] = test_points[:, 1] * (bbox[3] - bbox[2]) + bbox[2]
                    test_points[:, 2] = test_points[:, 2] * (bbox[5] - bbox[4]) + bbox[4]
                else:
                    raise ValueError(f"Unsupported bbox format: {bbox}")
                
                # Get noisy observations
                noisy_obs = noisy_problem.get_noisy_observations(test_points)
                
                # Compute validation metrics
                validation_results[problem_name] = {
                    'success': True,
                    'num_points': num_test_points,
                    'output_shape': list(noisy_obs.shape),
                    'mean_value': noisy_obs.mean().item(),
                    'std_value': noisy_obs.std().item(),
                    'min_value': noisy_obs.min().item(),
                    'max_value': noisy_obs.max().item(),
                    **noisy_problem.get_noise_statistics()
                }
                
            except Exception as e:
                validation_results[problem_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return validation_results


# Convenience functions for common use cases
def create_noisy_heat_problem(noise_level: float = 0.1, 
                             noise_type: NoiseType = NoiseType.GAUSSIAN,
                             random_seed: Optional[int] = None) -> NoisyPDEProblem:
    """Create noisy 2D heat equation problem."""
    factory = NoisyProblemFactory()
    return factory.create_noisy_problem('heat_2d', noise_type, noise_level, 
                                      random_seed=random_seed)


def create_noisy_burgers_problem(noise_level: float = 0.1,
                                noise_type: NoiseType = NoiseType.GAUSSIAN,
                                random_seed: Optional[int] = None) -> NoisyPDEProblem:
    """Create noisy 1D Burgers equation problem."""
    factory = NoisyProblemFactory()
    return factory.create_noisy_problem('burgers_1d', noise_type, noise_level,
                                      random_seed=random_seed)


def create_noisy_reaction_diffusion_problem(noise_level: float = 0.1,
                                          noise_type: NoiseType = NoiseType.GAUSSIAN,
                                          reaction_type: str = "fisher_kpp",
                                          random_seed: Optional[int] = None) -> NoisyPDEProblem:
    """Create noisy reaction-diffusion problem."""
    factory = NoisyProblemFactory()
    problem_params = {'reaction_type': reaction_type}
    return factory.create_noisy_problem('reaction_diffusion', noise_type, noise_level,
                                      problem_params=problem_params,
                                      random_seed=random_seed)