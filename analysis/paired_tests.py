"""Summary statistics and paired comparisons over the seeds (manuscript Section 4.10, Tables in Section 5).

Reads every results/ai_ready/<regime>/<variant>/seed_<seed>/metrics.json written by run_ai_experiments.py.

    python analysis/paired_tests.py
    python analysis/paired_tests.py --metric invalid_combination_rate
"""
import argparse
import glob
import json
import math
import os
import statistics

from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = ['strict_bundle_exact_match', 'active_feature_exact_match', 'lemma_exact_match', 'lemma_cer',
           'pos_macro_f1', 'oov_strict_exact_match', 'seen_strict_exact_match', 'invalid_combination_rate']
COMPARISONS = [(('curriculum', 'full'), ('curriculum', v)) for v in ('context', 'morph', 'concat', 'no_consistency')]
COMPARISONS.append((('curriculum', 'full'), ('gold', 'full')))


def load_runs():
    runs = {}
    for path in glob.glob(os.path.join(ROOT, 'results', 'ai_ready', '*', '*', 'seed_*', 'metrics.json')):
        with open(path, encoding='utf-8') as f:
            m = json.load(f)
        runs[(m['regime'], m['variant'], m['seed'])] = m
    return runs


def by_seed(runs, regime, variant, metric):
    return {s: m[metric] for (r, v, s), m in runs.items() if r == regime and v == variant and m.get(metric) is not None}


def compare(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = [a[s] - b[s] for s in seeds]
    n, mean, sd = len(d), statistics.fmean(d), statistics.stdev(d)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    p_t = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue
    p_w = stats.wilcoxon(d, method='exact').pvalue if any(d) else 1.0
    return dict(n=n, mean=mean, low=mean - half, high=mean + half, p_t=p_t, dz=mean / sd if sd else float('nan'), p_w=p_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='strict_bundle_exact_match')
    args = ap.parse_args()
    runs = load_runs()
    if not runs:
        raise SystemExit('No metrics.json found. Run run_ai_experiments.py first.')

    print('Mean ± SD over seeds')
    for regime, variant in sorted({(r, v) for r, v, _ in runs}):
        print(f'\n  {regime} / {variant}')
        for metric in SUMMARY:
            vals = list(by_seed(runs, regime, variant, metric).values())
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f'    {metric:32s} {statistics.fmean(vals):8.3f} ± {sd:.3f}  (n={len(vals)})')

    print(f'\nPaired comparisons on {args.metric} (first minus second)')
    for first, second in COMPARISONS:
        res = compare(by_seed(runs, *first, args.metric), by_seed(runs, *second, args.metric))
        label = f'{first[0]}/{first[1]} - {second[0]}/{second[1]}'
        if res is None:
            print(f'  {label}: not enough paired runs')
            continue
        print(f"  {label}: diff {res['mean']:+.2f}, 95% CI [{res['low']:.2f}, {res['high']:.2f}], "
              f"t-test p={res['p_t']:.2e}, d_z={res['dz']:.2f}, Wilcoxon p={res['p_w']:.4f} (n={res['n']})")


if __name__ == '__main__':
    main()
