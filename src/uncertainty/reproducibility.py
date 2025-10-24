"""
Reproducibility infrastructure for Bayesian uncertainty quantification experiments.

This module provides comprehensive tools for ensuring reproducible experiments,
including configuration management, random seed control, checkpoint saving/loading,
and experiment reproduction scripts.
"""

import torch
import numpy as np
import random
import yaml
import json
import pickle
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import shutil
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for reproducibility."""
    # Model configuration
    model_type: str
    network_architecture: Dict[str, Any]
    prior_type: str
    variational_family: str
    
    # Training configuration
    num_meta_iterations: int
    num_adaptation_steps: int
    learning_rate: float
    kl_weight: float
    temperature: float
    
    # Experiment configuration
    pde_types: List[str]
    noise_levels: List[float]
    k_shot_values: List[int]
    num_test_tasks: int
    num_posterior_samples: int
    
    # Reproducibility configuration
    random_seed: int
    torch_seed: int
    numpy_seed: int
    python_seed: int
    
    # System configuration
    device: str
    torch_version: str
    numpy_version: str
    python_version: str
    
    # Experiment metadata
    experiment_name: str
    description: str
    timestamp: str
    git_commit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class ReproducibilityManager:
    """Manager for ensuring experiment reproducibility."""
    
    def __init__(self, base_output_dir: Union[str, Path] = "experiments"):
        """Initialize reproducibility manager."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # System information
        self.system_info = self._collect_system_info()
        
    def create_experiment_config(self, 
                               experiment_name: str,
                               description: str,
                               model_config: Dict[str, Any],
                               experiment_params: Dict[str, Any],
                               random_seed: int = 42) -> ExperimentConfig:
        """Create complete experiment configuration."""
        
        # Set all random seeds
        self.set_random_seeds(random_seed)
        
        # Get git commit if available
        git_commit = self._get_git_commit()
        
        config = ExperimentConfig(
            # Model configuration
            model_type=model_config.get('model_type', 'bayesian'),
            network_architecture=model_config.get('network_architecture', {'dims': [2, 64, 64, 1]}),
            prior_type=model_config.get('prior_type', 'physics_informed'),
            variational_family=model_config.get('variational_family', 'diagonal_gaussian'),
            
            # Training configuration
            num_meta_iterations=model_config.get('num_meta_iterations', 1000),
            num_adaptation_steps=model_config.get('num_adaptation_steps', 10),
            learning_rate=model_config.get('learning_rate', 1e-3),
            kl_weight=model_config.get('kl_weight', 1.0),
            temperature=model_config.get('temperature', 1.0),
            
            # Experiment configuration
            pde_types=experiment_params.get('pde_types', ['heat', 'burgers', 'poisson']),
            noise_levels=experiment_params.get('noise_levels', [0.01, 0.05, 0.1]),
            k_shot_values=experiment_params.get('k_shot_values', [1, 5, 10]),
            num_test_tasks=experiment_params.get('num_test_tasks', 50),
            num_posterior_samples=experiment_params.get('num_posterior_samples', 100),
            
            # Reproducibility configuration
            random_seed=random_seed,
            torch_seed=random_seed,
            numpy_seed=random_seed,
            python_seed=random_seed,
            
            # System configuration
            device=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            python_version=sys.version,
            
            # Experiment metadata
            experiment_name=experiment_name,
            description=description,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit
        )
        
        return config
    
    def setup_experiment_directory(self, config: ExperimentConfig) -> Path:
        """Set up experiment directory with proper structure."""
        # Create experiment directory
        exp_dir = self.base_output_dir / f"{config.experiment_name}_{config.get_hash()}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (exp_dir / "config").mkdir(exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        
        # Save configuration
        self.save_config(config, exp_dir / "config" / "experiment_config.yaml")
        
        # Save system information
        with open(exp_dir / "config" / "system_info.json", 'w') as f:
            json.dump(self.system_info, f, indent=2)
        
        # Save requirements
        self._save_requirements(exp_dir / "config" / "requirements.txt")
        
        # Create reproduction script
        self._create_reproduction_script(config, exp_dir)
        
        logger.info(f"Experiment directory created: {exp_dir}")
        return exp_dir
    
    def set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        # Python random
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # PyTorch random
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Additional PyTorch settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"All random seeds set to {seed}")
    
    def save_config(self, config: ExperimentConfig, filepath: Path):
        """Save experiment configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: Path) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ExperimentConfig.from_dict(config_dict)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, exp_dir: Path, 
                       checkpoint_name: Optional[str] = None):
        """Save model checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = exp_dir / "checkpoints" / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as latest checkpoint
        latest_path = exp_dir / "checkpoints" / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       checkpoint_path: Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def save_results(self, results: Dict[str, Any], exp_dir: Path, 
                    filename: str = "results.json"):
        """Save experiment results."""
        results_path = exp_dir / "results" / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved: {results_path}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for reproducibility."""
        info = {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            info['gpu_info'] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_available': torch.cuda.memory_available(i)
                }
                info['gpu_info'].append(gpu_info)
        
        return info
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Could not get git commit hash")
            return None
    
    def _save_requirements(self, filepath: Path):
        """Save Python requirements."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                check=True
            )
            with open(filepath, 'w') as f:
                f.write(result.stdout)
            logger.info(f"Requirements saved: {filepath}")
        except subprocess.CalledProcessError:
            logger.warning("Could not save requirements")
    
    def _create_reproduction_script(self, config: ExperimentConfig, exp_dir: Path):
        """Create script to reproduce the experiment."""
        script_content = f'''#!/usr/bin/env python3
"""
Reproduction script for experiment: {config.experiment_name}
Generated on: {config.timestamp}
Git commit: {config.git_commit or 'Unknown'}
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from uncertainty.reproducibility import ReproducibilityManager
from uncertainty.experimental_validation import run_calibration_experiment
from uncertainty.ablation_studies import run_ablation_studies

def main():
    """Reproduce the experiment."""
    # Load configuration
    config_path = Path(__file__).parent / "config" / "experiment_config.yaml"
    
    # Set up reproducibility
    repro_manager = ReproducibilityManager()
    config = repro_manager.load_config(config_path)
    
    # Set random seeds
    repro_manager.set_random_seeds(config.random_seed)
    
    print(f"Reproducing experiment: {{config.experiment_name}}")
    print(f"Configuration hash: {{config.get_hash()}}")
    
    # Run experiment based on type
    if "calibration" in config.experiment_name.lower():
        print("Running calibration comparison experiment...")
        results, summary = run_calibration_experiment(str(config_path))
        print("Calibration experiment completed!")
        
    elif "ablation" in config.experiment_name.lower():
        print("Running ablation studies...")
        results, summary = run_ablation_studies(str(config_path))
        print("Ablation studies completed!")
        
    else:
        print("Unknown experiment type. Please check the configuration.")
        return
    
    print("Experiment reproduction completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        script_path = exp_dir / "reproduce_experiment.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Reproduction script created: {script_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj


class DockerManager:
    """Manager for Docker-based reproducibility."""
    
    def __init__(self, base_image: str = "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime"):
        """Initialize Docker manager."""
        self.base_image = base_image
    
    def create_dockerfile(self, exp_dir: Path, requirements_file: str = "requirements.txt"):
        """Create Dockerfile for experiment reproduction."""
        dockerfile_content = f'''FROM {self.base_image}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY config/{requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Copy experiment code
COPY . .

# Set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

# Make reproduction script executable
RUN chmod +x reproduce_experiment.py

# Default command
CMD ["python", "reproduce_experiment.py"]
'''
        
        dockerfile_path = exp_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create .dockerignore
        dockerignore_content = '''__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.git/
.gitignore
*.log
.DS_Store
'''
        
        with open(exp_dir / ".dockerignore", 'w') as f:
            f.write(dockerignore_content)
        
        logger.info(f"Dockerfile created: {dockerfile_path}")
    
    def create_docker_compose(self, exp_dir: Path, experiment_name: str):
        """Create docker-compose.yml for easy experiment running."""
        compose_content = f'''version: '3.8'

services:
  {experiment_name.replace('_', '-')}:
    build: .
    volumes:
      - ./results:/workspace/results
      - ./logs:/workspace/logs
    environment:
      - PYTHONHASHSEED=42
      - CUBLAS_WORKSPACE_CONFIG=:16:8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''
        
        compose_path = exp_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info(f"Docker Compose file created: {compose_path}")
    
    def create_run_script(self, exp_dir: Path, experiment_name: str):
        """Create shell script to run experiment in Docker."""
        script_content = f'''#!/bin/bash

# Build and run experiment in Docker
echo "Building Docker image for {experiment_name}..."
docker-compose build

echo "Running experiment in Docker container..."
docker-compose up

echo "Experiment completed. Results are in ./results/"
'''
        
        script_path = exp_dir / "run_docker.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Docker run script created: {script_path}")


def setup_reproducible_experiment(experiment_name: str,
                                description: str,
                                model_config: Dict[str, Any],
                                experiment_params: Dict[str, Any],
                                random_seed: int = 42,
                                use_docker: bool = True) -> Path:
    """Set up a fully reproducible experiment."""
    
    # Initialize managers
    repro_manager = ReproducibilityManager()
    docker_manager = DockerManager() if use_docker else None
    
    # Create experiment configuration
    config = repro_manager.create_experiment_config(
        experiment_name=experiment_name,
        description=description,
        model_config=model_config,
        experiment_params=experiment_params,
        random_seed=random_seed
    )
    
    # Set up experiment directory
    exp_dir = repro_manager.setup_experiment_directory(config)
    
    # Create Docker files if requested
    if use_docker and docker_manager:
        docker_manager.create_dockerfile(exp_dir)
        docker_manager.create_docker_compose(exp_dir, experiment_name)
        docker_manager.create_run_script(exp_dir, experiment_name)
    
    # Create README
    readme_content = f'''# {experiment_name}

{description}

## Experiment Details

- **Timestamp**: {config.timestamp}
- **Git Commit**: {config.git_commit or 'Unknown'}
- **Configuration Hash**: {config.get_hash()}
- **Random Seed**: {config.random_seed}

## Reproduction

### Local Reproduction

```bash
python reproduce_experiment.py
```

### Docker Reproduction

```bash
./run_docker.sh
```

Or manually:

```bash
docker-compose build
docker-compose up
```

## Configuration

See `config/experiment_config.yaml` for complete configuration details.

## Results

Results will be saved in the `results/` directory.
'''
    
    with open(exp_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Reproducible experiment setup completed: {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    # Example usage
    model_config = {
        'model_type': 'bayesian',
        'network_architecture': {'dims': [2, 64, 64, 1]},
        'prior_type': 'physics_informed',
        'variational_family': 'diagonal_gaussian',
        'num_meta_iterations': 1000,
        'learning_rate': 1e-3,
        'kl_weight': 1.0
    }
    
    experiment_params = {
        'pde_types': ['heat', 'burgers'],
        'noise_levels': [0.01, 0.05],
        'k_shot_values': [1, 5],
        'num_test_tasks': 10,
        'num_posterior_samples': 50
    }
    
    exp_dir = setup_reproducible_experiment(
        experiment_name="test_calibration_experiment",
        description="Test calibration comparison experiment for reproducibility",
        model_config=model_config,
        experiment_params=experiment_params,
        random_seed=42,
        use_docker=True
    )
    
    print(f"Test experiment setup completed: {exp_dir}")