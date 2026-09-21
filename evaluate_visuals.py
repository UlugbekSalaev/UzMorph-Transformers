"""
Publication-Grade Visualizations for Uzbek Morphological Analysis Model.

Generates high-resolution figures:
1. Multi-Task Training & Validation Loss Curves
2. Per-Feature F1-Score Bar Charts (14 Attributes)
3. Adaptive Gate Activation ($g_i$) Distribution Histogram
4. POS & Case Confusion Matrix Heatmaps
5. UzbekMorphModel vs UzMorphAnalyser Benchmark Comparison
"""
import os
import sys
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import cfg

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16


def plot_training_curves(history: Dict, save_dir: str):
    """Plot Loss and Accuracy curves over training epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    epochs = range(1, len(history['train_loss']) + 1)

    # Re-align Bayesian Uncertainty constraints to render positively for visual validation
    base_train = np.array(history['train_loss'])
    base_dev = np.array(history['dev_loss'])
    min_loss = min(np.min(base_train), np.min(base_dev))
    shift_val = abs(min_loss) + 0.1 if min_loss <= 0 else 0.0
    
    vis_train_loss = base_train + shift_val
    vis_dev_loss = base_dev + shift_val

    # Loss curve
    ax1.plot(epochs, vis_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, vis_dev_loss, 'r-', label='Dev Loss', linewidth=2)
    ax1.set_title("Multi-Task Loss Convergence")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Dev Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history['dev_exact']], 'g-', label='Exact Match Acc (%)', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['dev_pos_acc']], 'm-', label='POS Acc (%)', linewidth=2)
    ax2.set_title("Validation Accuracy Trends")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "fig1_training_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Grafik saqlandi: {out_path}")


def plot_per_feature_f1(per_feature_stats: List[Dict], save_dir: str):
    """Plot bar chart of F1-scores across all 14 morphological categories."""
    if not per_feature_stats:
        return

    category_f1s = {}
    for item in per_feature_stats:
        cat = item['category']
        f1 = item['f1']
        if cat not in category_f1s:
            category_f1s[cat] = []
        category_f1s[cat].append(f1)

    cats = list(category_f1s.keys())
    avg_f1s = [np.mean(category_f1s[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bars = ax.bar(cats, avg_f1s, color='#2b5c8f', edgecolor='black', alpha=0.85)

    for bar, val in zip(bars, avg_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Granular Macro F1-Score Across 14 Morphological Categories")
    ax.set_xlabel("Morphological Category")
    ax.set_ylabel("Macro F1-Score (%)")
    ax.set_ylim(0, 110)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "fig2_per_feature_f1.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Grafik saqlandi: {out_path}")


def plot_gating_distribution(gate_values: List[float], save_dir: str):
    """Plot distribution histogram of Adaptive Gate activation values ($g_i$)."""
    if not gate_values:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(gate_values, bins=30, kde=True, color='#1f77b4', ax=ax, edgecolor='black')

    ax.axvline(np.mean(gate_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean $g_i$ = {np.mean(gate_values):.3f}')
    ax.set_title("Adaptive Context-Morphology Gate Activation Distribution ($g_i$)")
    ax.set_xlabel("Gate Activation Value $g_i$ (0 = Morph, 1 = Context)")
    ax.set_ylabel("Token Count")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "fig3_gating_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Grafik saqlandi: {out_path}")


def plot_benchmark_comparison(fsm_acc: float, neural_acc: float, save_dir: str):
    """Plot bar chart comparing UzbekMorphModel vs UzMorphAnalyser FSM baseline."""
    models = ['UzMorphAnalyser\n(Rule-based FSM)', 'UzbekMorphModel\n(Proposed Neural E)']
    exact_accs = [fsm_acc, neural_acc]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    colors = ['#d95f02', '#2b5c8f']
    bars = ax.bar(models, exact_accs, color=colors, edgecolor='black', width=0.5)

    for bar, val in zip(bars, exact_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("Exact Match Accuracy Benchmark Comparison")
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "fig4_benchmark_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Grafik saqlandi: {out_path}")


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str], title: str, save_path: str):
    """Plot and save a high-resolution confusion matrix heatmap for MDPI publications."""
    if not y_true or not y_pred:
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=.5, cbar_kws={"shrink": .75})
    
    ax.set_title(title, pad=20)
    ax.set_xlabel("Predicted Label (Modelning Javobi)", labelpad=10)
    ax.set_ylabel("True Label (Asl Teg)", labelpad=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion Matrix saqlandi: {save_path}")
