"""Adaptive gate analysis on the Gold test partitions (manuscript Section 5.7).

Needs the checkpoints written by run_ai_experiments.py:

    python analysis/gate_analysis.py --regime curriculum --seeds 17 29 43 71 97
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import cfg
from data.dataset import MorphDataset, collate_fn
from data.scientific_data import prepare_scientific_data
from model.model import UzbekMorphModel


def token_gates(seed, regime, device):
    """Yield (form, upos, number of features, mean gate) for every Gold test token of one seed."""
    data = prepare_scientific_data(cfg, seed)
    ckpt = os.path.join(cfg.RESULTS_DIR, 'ai_ready', regime, 'full', f'seed_{seed}', 'best.pt')
    model = UzbekMorphModel(cfg, data['label_vocabs'], data['compat_matrix'], encoder_type='transformer',
                            use_gating=True, use_consistency=True, use_uncertainty=True, fusion_mode='gated').to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    loader = DataLoader(MorphDataset(data['test'], data['label_vocabs']), batch_size=32, shuffle=False, collate_fn=collate_fn)
    train_forms = {t['form'].lower() for s in data['gold_train'] for t in s['tokens']}
    sentences = iter(data['test'])
    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['char_ids'].to(device), token_ids=batch['token_ids'].to(device),
                        lengths=batch['lengths'].to(device), target_lemma_chars=None)
            gate = out['gate'].mean(-1).cpu()
            for b in range(gate.size(0)):
                sent = next(sentences)
                for j, tok in enumerate(sent['tokens'][:int(batch['lengths'][b])]):
                    form = tok['form'].lower()
                    rows.append((form, tok['upos'], len(tok.get('features', {})), float(gate[b, j]), form not in train_forms))
    return rows


def describe(values):
    return f'n={len(values):5d}  mean={statistics.fmean(values):.3f}  median={statistics.median(values):.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', default='curriculum')
    ap.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43, 71, 97])
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pooled = []
    print('Seen versus Gold-training-OOV tokens')
    for seed in args.seeds:
        rows = token_gates(seed, args.regime, device)
        pooled += rows
        oov = [r[3] for r in rows if r[4]]
        seen = [r[3] for r in rows if not r[4]]
        print(f'  seed {seed}: OOV {describe(oov)} | seen {describe(seen)}')

    print('\nToken length')
    bins = [('1-4 characters', 1, 4), ('5-8 characters', 5, 8), ('>=9 characters', 9, 10 ** 6)]
    for name, low, high in bins:
        print(f'  {name:16s} {describe([r[3] for r in pooled if low <= len(r[0]) <= high])}')

    groups = defaultdict(list)
    richness = defaultdict(list)
    for form, upos, n_feats, gate, _ in pooled:
        groups[upos].append(gate)
        richness[n_feats].append(gate)
    print('\nUPOS')
    for upos, values in sorted(groups.items(), key=lambda kv: -statistics.fmean(kv[1])):
        print(f'  {upos:6s} {describe(values)}')
    print('\nNumber of morphological features')
    for n_feats, values in sorted(richness.items()):
        print(f'  {n_feats:2d}     {describe(values)}')


if __name__ == '__main__':
    main()
