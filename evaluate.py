"""
Comprehensive Evaluation Suite for Uzbek Morphological Analysis Model (14 Attributes).
"""
import os
import sys
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import cfg
from data.preprocess import run_preprocessing_pipeline
from data.dataset import MorphDataset, collate_fn, ALL_TASKS
from model.model import UzbekMorphModel


def calculate_cer(pred_str: str, ref_str: str) -> float:
    """Levenshtein distance based Character Error Rate (CER)."""
    if len(ref_str) == 0:
        return 0.0 if len(pred_str) == 0 else 1.0

    d = [[0] * (len(ref_str) + 1) for _ in range(len(pred_str) + 1)]
    for i in range(len(pred_str) + 1):
        d[i][0] = i
    for j in range(len(ref_str) + 1):
        d[0][j] = j

    for i in range(1, len(pred_str) + 1):
        for j in range(1, len(ref_str) + 1):
            cost = 0 if pred_str[i - 1] == ref_str[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )

    return d[len(pred_str)][len(ref_str)] / len(ref_str)


def evaluate_test_set(model: nn.Module, test_loader: DataLoader,
                       label_vocabs: Dict[str, Dict],
                       device: torch.device) -> Dict:
    """
    Run full detailed evaluation on test set across all 14 attributes.
    Returns metrics dict and per-feature statistics.
    """
    model.eval()

    inv_vocabs = {
        task: {idx: label for label, idx in vocab.items()}
        for task, vocab in label_vocabs.items()
    }

    task_preds = defaultdict(list)
    task_targets = defaultdict(list)

    exact_matches = []
    lemma_matches = []
    cer_list = []
    gate_values = []

    tasks = ALL_TASKS

    with torch.no_grad():
        for batch in test_loader:
            char_ids = batch['char_ids'].to(device)
            lengths = batch['lengths'].to(device)
            lemma_chars = batch['lemma_chars'].to(device)

            targets = {k: batch[k].to(device) for k in tasks if k in batch}
            targets['lemma_chars'] = lemma_chars

            outputs = model(char_ids, lengths=lengths, target_lemma_chars=lemma_chars)
            logits = outputs['logits']
            lemma_logits = outputs['lemma_logits']
            gate = outputs['gate']

            if gate is not None:
                gate_mean = gate.mean(dim=-1)
                B, S = gate_mean.shape
                for b in range(B):
                    l = lengths[b].item()
                    gate_values.extend(gate_mean[b, :l].cpu().numpy().tolist())

            valid_mask = (targets['pos'] != 0) & (targets['pos'] != 1)
            B, S = targets['pos'].shape

            preds = {}
            for task in tasks:
                if task in logits and task in targets:
                    task_pred = torch.argmax(logits[task], dim=-1)
                    preds[task] = task_pred

                    for b in range(B):
                        l = lengths[b].item()
                        for s in range(l):
                            tgt_val = targets[task][b, s].item()
                            pred_val = task_pred[b, s].item()
                            if tgt_val not in (0, 1):  # skip PAD/NONE
                                task_targets[task].append(tgt_val)
                                task_preds[task].append(pred_val)

            # Exact match check
            for b in range(B):
                l = lengths[b].item()
                for s in range(l):
                    if targets['pos'][b, s].item() in (0, 1):
                        continue

                    token_exact = True
                    for task in ['pos', 'case', 'number', 'tense', 'person', 'possession']:
                        if task in preds and task in targets:
                            p_val = preds[task][b, s].item()
                            t_val = targets[task][b, s].item()
                            if t_val not in (0, 1) and p_val != t_val:
                                token_exact = False
                                break
                    exact_matches.append(token_exact)

            # Lemma evaluation
            lemma_pred_ids = torch.argmax(lemma_logits, dim=-1)
            char_inv = inv_vocabs['char']

            for b in range(B):
                l = lengths[b].item()
                for s in range(l):
                    if targets['pos'][b, s].item() in (0, 1):
                        continue

                    pred_chars = []
                    for idx in lemma_pred_ids[b, s]:
                        val = idx.item()
                        if val in (0, 3):  # <PAD> or <EOS>
                            break
                        if val != 1:  # skip <UNK>
                            pred_chars.append(char_inv.get(val, ''))

                    tgt_chars = [char_inv.get(idx.item(), '') for idx in lemma_chars[b, s] if idx.item() not in (0, 1, 3)]

                    pred_str = ''.join(pred_chars)
                    tgt_str = ''.join(tgt_chars)

                    is_match = (pred_str == tgt_str)
                    lemma_matches.append(is_match)
                    cer_list.append(calculate_cer(pred_str, tgt_str))

    metrics = {}

    metrics['exact_match_accuracy'] = (sum(exact_matches) / len(exact_matches)) * 100 if exact_matches else 0.0
    metrics['lemma_accuracy'] = (sum(lemma_matches) / len(lemma_matches)) * 100 if lemma_matches else 0.0
    metrics['mean_cer'] = (sum(cer_list) / len(cer_list)) if cer_list else 0.0

    per_feature_stats = []

    for task in tasks:
        if task not in task_targets or not task_targets[task]:
            continue

        tgt_list = task_targets[task]
        prd_list = task_preds[task]

        correct = sum(1 for p, t in zip(prd_list, tgt_list) if p == t)
        task_acc = (correct / len(tgt_list)) * 100
        metrics[f'{task}_accuracy'] = task_acc

        classes = sorted(list(set(tgt_list) | set(prd_list)))
        class_f1s = []

        inv_v = inv_vocabs[task]

        for cls_idx in classes:
            label_name = inv_v.get(cls_idx, str(cls_idx))
            if label_name in ('<PAD>', '<NONE>'):
                continue

            tp = sum(1 for p, t in zip(prd_list, tgt_list) if p == cls_idx and t == cls_idx)
            fp = sum(1 for p, t in zip(prd_list, tgt_list) if p == cls_idx and t != cls_idx)
            fn = sum(1 for p, t in zip(prd_list, tgt_list) if p != cls_idx and t == cls_idx)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_f1s.append(f1)

            per_feature_stats.append({
                'category': task.upper(),
                'feature': label_name,
                'support': tp + fn,
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1': round(f1 * 100, 2)
            })

        macro_f1 = (sum(class_f1s) / len(class_f1s)) * 100 if class_f1s else 0.0
        metrics[f'{task}_macro_f1'] = macro_f1

    if gate_values:
        import numpy as np
        arr = np.array(gate_values)
        metrics['gate_mean'] = round(float(np.mean(arr)), 4)
        metrics['gate_std'] = round(float(np.std(arr)), 4)
        metrics['gate_median'] = round(float(np.median(arr)), 4)
        metrics['gate_pct_context'] = round(float((arr > 0.5).mean() * 100), 2)

    pos_confusion = None
    if 'pos' in task_targets and 'pos' in task_preds:
        inv_v = inv_vocabs['pos']
        y_true = [inv_v.get(idx, str(idx)) for idx in task_targets['pos']]
        y_pred = [inv_v.get(idx, str(idx)) for idx in task_preds['pos']]
        labels = [inv_v.get(idx, str(idx)) for idx in sorted(list(set(task_targets['pos']) | set(task_preds['pos'])))]
        labels = [l for l in labels if l not in ('<PAD>', '<NONE>')]
        pos_confusion = {'y_true': y_true, 'y_pred': y_pred, 'labels': labels}

    return {
        'metrics': metrics,
        'per_feature_stats': per_feature_stats,
        'pos_confusion': pos_confusion
    }


def save_evaluation_results(model_name: str, eval_data: Dict, output_dir: str):
    """Save metrics and per-feature tables to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    metrics = eval_data['metrics']
    per_feature = eval_data['per_feature_stats']

    json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(tables_dir, f"{model_name}_per_feature.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'feature', 'support', 'precision', 'recall', 'f1'])
        writer.writeheader()
        writer.writerows(per_feature)

    print(f"\n[Natijalar Saqlandi]")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    if eval_data.get('pos_confusion'):
        from evaluate_visuals import plot_confusion_matrix
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        cm_path = os.path.join(fig_dir, f"{model_name}_pos_confusion.png")
        plot_confusion_matrix(
            y_true=eval_data['pos_confusion']['y_true'],
            y_pred=eval_data['pos_confusion']['y_pred'],
            labels=eval_data['pos_confusion']['labels'],
            title=f"POS Tagger Confusion Matrix - Dataset: Test Set",
            save_path=cm_path
        )


def main_evaluate(checkpoint_path: str):
    """Load model checkpoint and evaluate on test set."""
    print("=" * 60)
    print("O'ZBEK TILI MORFOLOGIK TAHLIL — TEST SET BAHOLASH (14 Atribut)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    label_vocabs = checkpoint['label_vocabs']
    saved_cfg = checkpoint['cfg']

    data = run_preprocessing_pipeline(saved_cfg)

    test_dataset = MorphDataset(data['ud_data']['test'], label_vocabs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = UzbekMorphModel(
        cfg=saved_cfg,
        label_vocabs=label_vocabs,
        compat_matrix=data['compat_matrix']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model muvaffaqiyatli yuklandi.")

    eval_data = evaluate_test_set(model, test_loader, label_vocabs, device)
    metrics = eval_data['metrics']

    print("\n" + "=" * 60)
    print("TEST SET BAHOLASH NATIJALARI (14 ATRIBUT)")
    print("=" * 60)
    print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
    for task in ALL_TASKS:
        acc = metrics.get(f'{task}_accuracy', 0.0)
        f1 = metrics.get(f'{task}_macro_f1', 0.0)
        print(f"  {task.upper():<15}: Accuracy: {acc:6.2f}% | Macro F1: {f1:6.2f}%")
    print(f"  Lemma Accuracy:      {metrics['lemma_accuracy']:.2f}% (Mean CER: {metrics['mean_cer']:.4f})")

    if 'gate_mean' in metrics:
        print(f"\n  [Adaptive Gating Analysis]")
        print(f"    Gate activation mean: {metrics['gate_mean']} ± {metrics['gate_std']}")
        print(f"    Context preference:   {metrics['gate_pct_context']}% tokens")

    model_name = os.path.basename(checkpoint_path).replace('.pt', '')
    save_evaluation_results(model_name, eval_data, saved_cfg.RESULTS_DIR)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_evaluate(sys.argv[1])
    else:
        chk = os.path.join(cfg.CHECKPOINT_DIR, "Model_E_Full.pt")
        if os.path.exists(chk):
            main_evaluate(chk)
        else:
            print(f"Checkpoint topilmadi: {chk}")
