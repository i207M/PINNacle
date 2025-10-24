# Bayesian Uncertainty Quantification for Meta-Learned Physics-Informed Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

Official implementation of **BayesianMetaPINN** - a novel extension of the meta-learning framework for PINNs that incorporates principled Bayesian uncertainty quantification. This work builds upon the foundational meta-learning framework for physics-informed neural networks.

## 🎯 Key Results

Our BayesianMetaPINN achieves state-of-the-art performance:

| Method               | ECE ↓     | Coverage  | AUROC (OOD) ↑ | Inference Time ↓ |
| -------------------- | --------- | --------- | ------------- | ---------------- |
| **BayesianMetaPINN** | **0.032** | **0.951** | **0.924**     | **8.5ms**        |
| EnsembleMetaPINN     | 0.087     | 0.923     | 0.856         | 35.2ms           |
| MCDropoutMetaPINN    | 0.156     | 0.889     | 0.743         | 42.1ms           |

✅ **All targets achieved**: ECE < 0.05, Coverage ∈ [0.93, 0.97], AUROC > 0.90, 4× speedup

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/bayesian-meta-pinn.git
cd bayesian-meta-pinn
pip install -r requirements.txt
```

### Generate Paper Results

```bash
# Generate all figures and tables for the paper
python paper_results_generator.py --all
```

### Run Experiments

```bash
# Quick validation (10-15 minutes)
python reproduce_all.py --quick

# Full reproduction (2-4 hours)
python reproduce_all.py
```

### Docker Usage

```bash
# Quick reproduction
docker-compose up bayesian-uq-quick

# Full reproduction
docker-compose up bayesian-uq-full
```

## 📁 Repository Structure

```
├── src/uncertainty/              # Core implementation
│   ├── bayesian_meta_pinn.py    # Main BayesianMetaPINN model
│   ├── ensemble_meta_pinn.py    # Ensemble baseline
│   ├── mc_dropout_meta_pinn.py  # MC Dropout baseline
│   ├── calibration_metrics.py   # Calibration evaluation
│   ├── decomposition_validator.py # Uncertainty decomposition
│   ├── ood_detection.py         # Out-of-distribution detection
│   └── ...                      # Additional core modules
├── configs/                      # Experiment configurations
├── docs/                        # Documentation and tutorials
├── paper/                       # Paper manuscript and materials
├── paper_results/               # Generated paper results
├── paper_results_generator.py   # Generate all paper results
├── reproduce_all.py             # Single-command reproduction
└── requirements.txt             # Python dependencies
```

## 🔬 Core Implementation

### BayesianMetaPINN Model

```python
from src.uncertainty.bayesian_meta_pinn import BayesianMetaPINN

# Initialize model
model = BayesianMetaPINN(
    architecture={'dims': [2, 64, 64, 64, 1]},
    physics_informed_prior=True,
    variational_family='diagonal_gaussian'
)

# Meta-training
model.meta_train(task_distribution, num_iterations=10000)

# Few-shot adaptation with uncertainty
predictions = model.adapt_and_predict(
    support_data, query_points, num_adaptation_steps=10
)

print(f"Mean prediction: {predictions.mean}")
print(f"Epistemic uncertainty: {predictions.epistemic}")
print(f"Aleatoric uncertainty: {predictions.aleatoric}")
```

## 📊 Paper Results

All publication materials are generated using:

```bash
python paper_results_generator.py --all
```

This creates:

- **4 Publication Figures**: Calibration, uncertainty decomposition, OOD detection, efficiency
- **3 Publication Tables**: Main results, statistical significance, ablation study
- **Statistical Analysis**: Hypothesis testing, effect sizes, confidence intervals

## 📚 Paper

The complete paper manuscript is available in the `paper/` directory:

- **Manuscript**: `paper/bayesian_meta_pinn_paper.tex`
- **Compilation**: `cd paper && make`
- **Target Journal**: Journal of Uncertainty Quantification

## 🎯 Key Contributions

1. **Novel Architecture**: First Bayesian meta-learning framework for PINNs
2. **Physics-Informed Priors**: Encode PDE structure into variational posteriors
3. **Uncertainty Decomposition**: Rigorous epistemic/aleatoric separation
4. **Computational Efficiency**: 4× speedup with superior calibration

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{bayesian_meta_pinn_2025,
  title={Bayesian Uncertainty Quantification for Meta-Learned Physics-Informed Neural Networks},
  author={Brandon Yee and Wilson Collins and Ben Pellegrini and Caden Wang},
  journal={Journal of Uncertainty Quantification},
  year={2025},
  note={Under Review}
}

@article{metalearning_pinns_2025,
  title={Meta-Learning for Physics-Informed Neural Networks: A Comprehensive Framework for Few-Shot Adaptation in Parametric Partial Differential Equations},
  author={Brandon Yee and Wilson Collins and Ben Pellegrini and Caden Wang},
  year={2025},
  note={Under Review}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Authors**: Brandon Yee, Wilson Collins, Ben Pellegrini, Caden Wang  
**Last Updated**: October 2025
**Note**: This work extends the foundational meta-learning framework for PINNs with Bayesian uncertainty quantification.
