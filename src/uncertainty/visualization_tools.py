"""
Visualization and Reporting Tools for Pinnacle v2.0

This module provides comprehensive visualization capabilities for uncertainty quantification
evaluation, including calibration plots, uncertainty decomposition visualization, and
OOD detection performance plots.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
from dataclasses import dataclass

from src.uncertainty.base import UncertaintyPrediction
from src.uncertainty.uncertainty_evaluator import (
    UncertaintyEvaluationResults, 
    ComprehensiveUncertaintyResults
)

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters."""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 300
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    save_format: str = 'png'
    transparent: bool = False


class UncertaintyVisualizationTools:
    """
    Comprehensive visualization tools for uncertainty quantification evaluation.
    
    Provides methods for creating calibration plots, uncertainty decomposition
    visualizations, and OOD detection performance plots.
    """
    
    def __init__(
        self,
        output_dir: str = "uncertainty_plots",
        plot_config: PlotConfig = None
    ):
        """
        Initialize visualization tools.
        
        Args:
            output_dir: Directory for saving plots
            plot_config: Configuration for plot styling
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config or PlotConfig()
        
        # Configure matplotlib
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'legend.fontsize': self.config.legend_size,
            'figure.dpi': self.config.dpi
        })
        
        logger.info(f"UncertaintyVisualizationTools initialized: {output_dir}")
    
    def create_reliability_diagram(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        model_name: str = "Model",
        num_bins: int = 10,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create reliability diagram for calibration assessment.
        
        Args:
            predictions: Model predictions with uncertainty
            targets: True target values
            model_name: Name of the model for plot title
            num_bins: Number of bins for reliability diagram
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert predictions to confidence levels
        confidences = self._prediction_to_confidence(predictions)
        accuracies = self._compute_accuracies(predictions, targets)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if i == num_bins - 1:  # Include right boundary for last bin
                bin_mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if bin_mask.sum() > 0:
                bin_confidences.append(confidences[bin_mask].mean().item())
                bin_accuracies.append(accuracies[bin_mask].mean().item())
                bin_counts.append(bin_mask.sum().item())
            else:
                bin_confidences.append(bin_centers[i])
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        # Plot 1: Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
        ax1.bar(bin_centers, bin_accuracies, width=1/num_bins, alpha=0.7, 
                edgecolor='black', label='Observed Accuracy')
        ax1.plot(bin_centers, bin_confidences, 'ro-', markersize=6, 
                linewidth=2, label='Mean Confidence')
        
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Confidence histogram
        ax2.hist(confidences.numpy(), bins=num_bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Confidence Distribution - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"reliability_diagram_{model_name.lower().replace(' ', '_')}.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, transparent=self.config.transparent, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Reliability diagram saved: {save_path}")
        return str(save_path)
    
    def create_ece_comparison_plot(
        self,
        results: ComprehensiveUncertaintyResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create ECE comparison plot across models and K values.
        
        Args:
            results: Comprehensive uncertainty evaluation results
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract ECE values
        model_names = list(results.results_by_model.keys())
        k_values = results.evaluation_config["k_shot_values"]
        
        # Plot 1: ECE vs K-shot
        for model_name in model_names:
            ece_values = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    ece = results.results_by_model[model_name][k].calibration_results.get("ece", np.nan)
                    ece_values.append(ece)
                else:
                    ece_values.append(np.nan)
            
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(ece_values)
            if np.any(valid_indices):
                ax1.plot(np.array(k_values)[valid_indices], np.array(ece_values)[valid_indices], 
                        'o-', linewidth=2, markersize=6, label=model_name)
        
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target ECE < 0.05')
        ax1.set_xlabel('K-shot')
        ax1.set_ylabel('Expected Calibration Error (ECE)')
        ax1.set_title('ECE vs K-shot Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Average ECE comparison
        avg_ece_values = []
        for model_name in model_names:
            ece_list = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    ece = results.results_by_model[model_name][k].calibration_results.get("ece", np.nan)
                    if not np.isnan(ece):
                        ece_list.append(ece)
            
            avg_ece = np.mean(ece_list) if ece_list else np.nan
            avg_ece_values.append(avg_ece)
        
        # Filter out NaN values
        valid_models = [(name, ece) for name, ece in zip(model_names, avg_ece_values) if not np.isnan(ece)]
        if valid_models:
            names, values = zip(*valid_models)
            bars = ax2.bar(names, values, alpha=0.7, edgecolor='black')
            
            # Color bars based on performance (green if < 0.05, red otherwise)
            for bar, value in zip(bars, values):
                if value < 0.05:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
        
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target ECE < 0.05')
        ax2.set_ylabel('Average ECE')
        ax2.set_title('Average ECE Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"ece_comparison.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, transparent=self.config.transparent, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ECE comparison plot saved: {save_path}")
        return str(save_path)
    
    def create_uncertainty_decomposition_plot(
        self,
        predictions: UncertaintyPrediction,
        targets: torch.Tensor,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create uncertainty decomposition visualization.
        
        Args:
            predictions: Model predictions with uncertainty decomposition
            targets: True target values
            model_name: Name of the model for plot title
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Compute prediction errors
        errors = torch.abs(predictions.mean - targets).numpy()
        epistemic = predictions.epistemic.numpy().flatten()
        aleatoric = predictions.aleatoric.numpy().flatten()
        total_uncertainty = (predictions.epistemic + predictions.aleatoric).numpy().flatten()
        
        # Plot 1: Epistemic vs Aleatoric scatter
        ax1.scatter(epistemic, aleatoric, alpha=0.6, s=20)
        ax1.set_xlabel('Epistemic Uncertainty')
        ax1.set_ylabel('Aleatoric Uncertainty')
        ax1.set_title(f'Epistemic vs Aleatoric Uncertainty - {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        max_val = max(epistemic.max(), aleatoric.max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Uncertainty')
        ax1.legend()
        
        # Plot 2: Uncertainty vs Error correlation
        ax2.scatter(total_uncertainty, errors, alpha=0.6, s=20)
        ax2.set_xlabel('Total Uncertainty')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title(f'Uncertainty vs Error - {model_name}')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(total_uncertainty, errors)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Uncertainty distribution
        ax3.hist(epistemic, bins=30, alpha=0.7, label='Epistemic', density=True)
        ax3.hist(aleatoric, bins=30, alpha=0.7, label='Aleatoric', density=True)
        ax3.set_xlabel('Uncertainty')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Uncertainty Distribution - {model_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Uncertainty decomposition ratio
        uncertainty_ratio = epistemic / (epistemic + aleatoric + 1e-8)
        ax4.hist(uncertainty_ratio, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Epistemic / Total Uncertainty')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Uncertainty Decomposition Ratio - {model_name}')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Equal Split')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"uncertainty_decomposition_{model_name.lower().replace(' ', '_')}.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, transparent=self.config.transparent, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Uncertainty decomposition plot saved: {save_path}")
        return str(save_path)
    
    def create_ood_detection_plot(
        self,
        in_dist_scores: torch.Tensor,
        ood_scores: torch.Tensor,
        model_name: str = "Model",
        scenario: str = "OOD",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create OOD detection performance visualization.
        
        Args:
            in_dist_scores: Uncertainty scores for in-distribution data
            ood_scores: Uncertainty scores for out-of-distribution data
            model_name: Name of the model for plot title
            scenario: OOD scenario name
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data for ROC and PR curves
        scores = torch.cat([in_dist_scores, ood_scores]).numpy()
        labels = torch.cat([torch.zeros(len(in_dist_scores)), torch.ones(len(ood_scores))]).numpy()
        
        # Plot 1: Score distributions
        ax1.hist(in_dist_scores.numpy(), bins=30, alpha=0.7, label='In-Distribution', density=True)
        ax1.hist(ood_scores.numpy(), bins=30, alpha=0.7, label='Out-of-Distribution', density=True)
        ax1.set_xlabel('Uncertainty Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Score Distributions - {model_name} ({scenario})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'ROC Curve - {model_name} ({scenario})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        ax3.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax3.axhline(y=labels.mean(), color='k', linestyle='--', alpha=0.7, 
                   label=f'Random Classifier ({labels.mean():.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title(f'Precision-Recall Curve - {model_name} ({scenario})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Detection performance metrics
        # Find FPR at 95% TPR
        idx_95_tpr = np.argmax(tpr >= 0.95)
        fpr_at_95_tpr = fpr[idx_95_tpr] if idx_95_tpr < len(fpr) else 1.0
        
        metrics = ['AUROC', 'AUPR', 'FPR@95%TPR']
        values = [roc_auc, pr_auc, fpr_at_95_tpr]
        colors = ['green' if v > 0.9 else 'orange' if v > 0.7 else 'red' for v in values[:2]]
        colors.append('green' if fpr_at_95_tpr < 0.1 else 'orange' if fpr_at_95_tpr < 0.2 else 'red')
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Score')
        ax4.set_title(f'Detection Performance - {model_name} ({scenario})')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"ood_detection_{model_name.lower().replace(' ', '_')}_{scenario.lower()}.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, transparent=self.config.transparent, bbox_inches='tight')
        plt.close()
        
        logger.info(f"OOD detection plot saved: {save_path}")
        return str(save_path)
    
    def create_comprehensive_report(
        self,
        results: ComprehensiveUncertaintyResults,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive visual report with all key plots.
        
        Args:
            results: Comprehensive uncertainty evaluation results
            save_path: Optional custom save path
            
        Returns:
            Path to saved report
        """
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # Extract data for plotting
        model_names = list(results.results_by_model.keys())
        k_values = results.evaluation_config["k_shot_values"]
        
        # Plot 1: ECE Performance Matrix
        ax1 = plt.subplot(4, 2, 1)
        ece_matrix = []
        for model_name in model_names:
            ece_row = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    ece = results.results_by_model[model_name][k].calibration_results.get("ece", np.nan)
                    ece_row.append(ece)
                else:
                    ece_row.append(np.nan)
            ece_matrix.append(ece_row)
        
        ece_matrix = np.array(ece_matrix)
        im1 = ax1.imshow(ece_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        ax1.set_xticks(range(len(k_values)))
        ax1.set_xticklabels(k_values)
        ax1.set_yticks(range(len(model_names)))
        ax1.set_yticklabels(model_names)
        ax1.set_xlabel('K-shot')
        ax1.set_ylabel('Model')
        ax1.set_title('Expected Calibration Error (ECE) Heatmap')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Coverage Performance Matrix
        ax2 = plt.subplot(4, 2, 2)
        coverage_matrix = []
        for model_name in model_names:
            coverage_row = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    coverage = results.results_by_model[model_name][k].calibration_results.get("coverage", np.nan)
                    coverage_row.append(abs(coverage - 0.95) if not np.isnan(coverage) else np.nan)
                else:
                    coverage_row.append(np.nan)
            coverage_matrix.append(coverage_row)
        
        coverage_matrix = np.array(coverage_matrix)
        im2 = ax2.imshow(coverage_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        ax2.set_xticks(range(len(k_values)))
        ax2.set_xticklabels(k_values)
        ax2.set_yticks(range(len(model_names)))
        ax2.set_yticklabels(model_names)
        ax2.set_xlabel('K-shot')
        ax2.set_ylabel('Model')
        ax2.set_title('Coverage Deviation from 95% Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: AUROC Performance
        ax3 = plt.subplot(4, 2, 3)
        for model_name in model_names:
            auroc_values = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    auroc = results.results_by_model[model_name][k].ood_results.get("auroc", np.nan)
                    auroc_values.append(auroc)
                else:
                    auroc_values.append(np.nan)
            
            valid_indices = ~np.isnan(auroc_values)
            if np.any(valid_indices):
                ax3.plot(np.array(k_values)[valid_indices], np.array(auroc_values)[valid_indices], 
                        'o-', linewidth=2, markersize=6, label=model_name)
        
        ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target AUROC > 0.90')
        ax3.set_xlabel('K-shot')
        ax3.set_ylabel('AUROC')
        ax3.set_title('OOD Detection Performance (AUROC)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Computational Efficiency
        ax4 = plt.subplot(4, 2, 4)
        adaptation_times = []
        inference_times = []
        for model_name in model_names:
            # Average across K values
            adapt_times = []
            infer_times = []
            for k in k_values:
                if k in results.results_by_model[model_name]:
                    adapt_time = results.results_by_model[model_name][k].timing_results.get("mean_adaptation_time", np.nan)
                    infer_time = results.results_by_model[model_name][k].timing_results.get("mean_inference_time", np.nan)
                    if not np.isnan(adapt_time):
                        adapt_times.append(adapt_time)
                    if not np.isnan(infer_time):
                        infer_times.append(infer_time)
            
            adaptation_times.append(np.mean(adapt_times) if adapt_times else np.nan)
            inference_times.append(np.mean(infer_times) if infer_times else np.nan)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, adaptation_times, width, label='Adaptation Time', alpha=0.7)
        bars2 = ax4.bar(x + width/2, inference_times, width, label='Inference Time', alpha=0.7)
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Computational Efficiency')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Performance Summary Radar Chart
        ax5 = plt.subplot(4, 2, (5, 6))
        
        # Prepare data for radar chart
        metrics = ['ECE (inv)', 'Coverage', 'AUROC', 'Speed (inv)']
        
        # Calculate normalized scores for each model
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name in model_names:
            # Calculate average metrics
            avg_ece = np.nanmean([
                results.results_by_model[model_name][k].calibration_results.get("ece", np.nan)
                for k in k_values if k in results.results_by_model[model_name]
            ])
            avg_coverage = np.nanmean([
                abs(results.results_by_model[model_name][k].calibration_results.get("coverage", 0.95) - 0.95)
                for k in k_values if k in results.results_by_model[model_name]
            ])
            avg_auroc = np.nanmean([
                results.results_by_model[model_name][k].ood_results.get("auroc", np.nan)
                for k in k_values if k in results.results_by_model[model_name]
            ])
            avg_speed = np.nanmean([
                results.results_by_model[model_name][k].timing_results.get("mean_inference_time", np.nan)
                for k in k_values if k in results.results_by_model[model_name]
            ])
            
            # Normalize scores (higher is better)
            scores = [
                1 - min(avg_ece, 0.2) / 0.2 if not np.isnan(avg_ece) else 0,  # ECE (inverted)
                1 - min(avg_coverage, 0.1) / 0.1 if not np.isnan(avg_coverage) else 0,  # Coverage deviation (inverted)
                avg_auroc if not np.isnan(avg_auroc) else 0,  # AUROC
                1 - min(avg_speed, 10) / 10 if not np.isnan(avg_speed) else 0  # Speed (inverted)
            ]
            scores += scores[:1]  # Complete the circle
            
            ax5.plot(angles, scores, 'o-', linewidth=2, label=model_name)
            ax5.fill(angles, scores, alpha=0.25)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics)
        ax5.set_ylim(0, 1)
        ax5.set_title('Overall Performance Comparison')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True)
        
        # Plot 7: Statistical Significance Matrix
        ax7 = plt.subplot(4, 2, 7)
        
        if results.statistical_significance.get("pairwise_comparisons"):
            comparisons = results.statistical_significance["pairwise_comparisons"]
            
            # Create significance matrix
            n_models = len(model_names)
            sig_matrix = np.zeros((n_models, n_models))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        comp_key = f"{model1}_vs_{model2}"
                        alt_key = f"{model2}_vs_{model1}"
                        
                        if comp_key in comparisons:
                            sig_matrix[i, j] = 1 if comparisons[comp_key]["significant"] else 0
                        elif alt_key in comparisons:
                            sig_matrix[i, j] = 1 if comparisons[alt_key]["significant"] else 0
            
            im7 = ax7.imshow(sig_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax7.set_xticks(range(n_models))
            ax7.set_xticklabels(model_names, rotation=45)
            ax7.set_yticks(range(n_models))
            ax7.set_yticklabels(model_names)
            ax7.set_title('Statistical Significance Matrix')
            plt.colorbar(im7, ax=ax7)
        
        # Plot 8: Summary Text
        ax8 = plt.subplot(4, 2, 8)
        ax8.axis('off')
        
        # Create summary text
        summary_text = "EVALUATION SUMMARY\n\n"
        
        if results.summary_report.get("performance_summary"):
            best_ece_model = min(
                results.summary_report["performance_summary"].items(),
                key=lambda x: x[1].get("average_ece", float('inf'))
            )[0]
            summary_text += f"Best Calibration: {best_ece_model}\n"
            
            best_auroc_model = max(
                results.summary_report["performance_summary"].items(),
                key=lambda x: x[1].get("average_auroc", 0)
            )[0]
            summary_text += f"Best OOD Detection: {best_auroc_model}\n\n"
        
        if results.summary_report.get("recommendations"):
            summary_text += "RECOMMENDATIONS:\n"
            for rec in results.summary_report["recommendations"]:
                summary_text += f"• {rec}\n"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive report
        if save_path is None:
            save_path = self.output_dir / f"comprehensive_uncertainty_report.{self.config.save_format}"
        
        plt.savefig(save_path, dpi=self.config.dpi, transparent=self.config.transparent, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive report saved: {save_path}")
        return str(save_path)
    
    def _prediction_to_confidence(self, predictions: UncertaintyPrediction) -> torch.Tensor:
        """Convert uncertainty predictions to confidence levels."""
        total_uncertainty = predictions.epistemic + predictions.aleatoric
        # Normalize uncertainty to [0, 1] and convert to confidence
        max_uncertainty = total_uncertainty.max() + 1e-8
        normalized_uncertainty = total_uncertainty / max_uncertainty
        confidence = 1.0 - normalized_uncertainty
        return confidence
    
    def _compute_accuracies(self, predictions: UncertaintyPrediction, targets: torch.Tensor) -> torch.Tensor:
        """Compute prediction accuracies for calibration assessment."""
        # For regression, use relative accuracy based on prediction intervals
        total_uncertainty = predictions.epistemic + predictions.aleatoric
        
        # Create prediction intervals (95% confidence)
        z_score = 1.96  # 95% confidence interval
        lower_bound = predictions.mean - z_score * torch.sqrt(total_uncertainty)
        upper_bound = predictions.mean + z_score * torch.sqrt(total_uncertainty)
        
        # Accuracy is 1 if target is within interval, 0 otherwise
        accuracies = ((targets >= lower_bound) & (targets <= upper_bound)).float()
        
        return accuracies


def create_visualization_tools(
    output_dir: str = "uncertainty_plots",
    plot_config: PlotConfig = None
) -> UncertaintyVisualizationTools:
    """
    Factory function to create visualization tools with default settings.
    
    Args:
        output_dir: Output directory for plots
        plot_config: Configuration for plot styling
        
    Returns:
        Configured UncertaintyVisualizationTools instance
    """
    return UncertaintyVisualizationTools(
        output_dir=output_dir,
        plot_config=plot_config
    )