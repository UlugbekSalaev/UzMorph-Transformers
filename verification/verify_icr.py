"""Checks the invalid-combination metric against independent annotations (manuscript Section 5.4).

    python verification/verify_icr.py

Inputs are included: the UD Uzbek treebanks in verification/external and the stored predictions of a separate
five-fold evaluation of the same architecture in verification/predictions.
"""
import json
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.preprocess import parse_conllu_file
from data.scientific_data import normalized_compatibility_from_gold

EXT = os.path.join(ROOT, 'verification', 'external')
PRED = os.path.join(ROOT, 'verification', 'predictions')
SPLITS = os.path.join(ROOT, 'Dataset', 'splits')
failures = 0


def check(name, value, expected):
    global failures
    ok = value == expected
    failures += not ok
    print(f"  [{'OK' if ok else 'DIFF'}] {name}: {value}" + ('' if ok else f' (manuscript: {expected})'))


# ---- 1. The article's Gold-training relation against human annotations outside the corpus
FEATURE_TO_TASK = {'Case': 'case', 'Number': 'number', 'Person': 'person', 'Possession': 'possession', 'Tense': 'tense',
                   'Mood': 'mood', 'VerbForm': 'verb_form', 'Voice': 'verb_voice', 'Degree': 'degree', 'Polarity': 'negative',
                   'Question': 'question', 'Copula': 'copula', 'Polite': 'polite'}
reference = [s for name in ('uz_ut-ud-test.conllu', 'uz_tuecl-ud-test.conllu') for s in parse_conllu_file(os.path.join(EXT, name))]
print('Gold-training relation on Uzbek-UT and Uzbek-TueCL')
rates, shares, rests, counts = [], [], [], set()
for seed in (17, 29, 43, 71, 97):
    relation = {task: pairs for (_, task), pairs in normalized_compatibility_from_gold(
        parse_conllu_file(os.path.join(SPLITS, f'seed_{seed}', 'gold_train.conllu'))).items()}
    flagged = Counter()
    total = 0
    for s in reference:
        for t in s['tokens']:
            for feat, val in t.get('features', {}).items():
                task = FEATURE_TO_TASK.get(feat)
                if task not in relation:
                    continue
                total += 1
                if (t['upos'], val) not in relation[task]:
                    flagged[(t['upos'], feat, val)] += 1
    n = sum(flagged.values())
    top = flagged[('PROPN', 'Number', 'Sing')]
    counts.add(total)
    rates.append(round(100 * n / total, 2))
    shares.append(round(100 * top / n, 1))
    rests.append(round(100 * (n - top) / total, 2))
check('checked values', counts, {9350})
check('flagged % in every split', set(rates), {4.51})
check('share of PROPN with Number=Sing among flagged %', set(shares), {91.9})
check('all other pairs, % of checked values', set(rests), {0.36})

# ---- 2. Stored predictions of a separate five-fold evaluation: regularized model against the unregularized one
ATTRS = ['Case', 'Number', 'Person', 'Tense', 'Mood', 'VerbForm', 'Voice', 'Polarity', 'Aspect', 'Poss', 'PronType', 'NumType', 'Degree']


def feats(s):
    d = {}
    if s and s != '_':
        for kv in s.split('|'):
            if '=' in kv:
                k, v = kv.split('=', 1)
                if k in ATTRS:
                    d[k] = v
    return d


def attested(sents):
    return {(t['upos'], a, v) for s in sents for t in s['tokens'] for a, v in feats(t['feats']).items()}


with open(os.path.join(PRED, 'folds.json'), encoding='utf-8') as f:
    D = json.load(f)
SENT = {s['id']: s for group in D['sentences'].values() for s in group}
UZUDT = attested(D['sentences']['UzUDT_train'] + D['sentences']['UzUDT_test'])
results = {}
for config in ('full', 'no_cons'):
    rows = []
    for k in range(5):
        with open(os.path.join(PRED, f'{config}_f{k}.json'), encoding='utf-8') as f:
            pred = json.load(f)['pred']
        fold = D['folds'][k]
        train = attested(SENT[i] for i in fold['train'])
        independent = UZUDT | attested(SENT[i] for i in fold['dev'] + fold['test'])
        c = Counter()
        for sid, sent in pred.items():
            for (_, upos, fe), gold in zip(sent, SENT[sid]['tokens']):
                gf = feats(gold['feats'])
                for a, v in feats(fe).items():
                    ok = gold['upos'] == upos and gf.get(a) == v
                    c['n'] += 1
                    c['independent'] += (upos, a, v) not in independent
                    c['uzudt'] += (upos, a, v) not in UZUDT
                    if (upos, a, v) in train:
                        c['unflagged'] += 1
                        c['unflagged_ok'] += ok
                    else:
                        c['flagged'] += 1
                        c['flagged_ok'] += ok
        rows.append(c)
    results[config] = rows


def rate(config, field):
    v = [100 * c[field] / c['n'] for c in results[config]]
    return round(statistics.fmean(v), 2), round(statistics.stdev(v), 2)


def mean_rate(config, field):
    return statistics.fmean(100 * c[field] / c['n'] for c in results[config])


print('\nStored predictions, five folds')
all_rows = results['full'] + results['no_cons']
check('flagged values', sum(c['flagged'] for c in all_rows), 1831)
check('flagged values matching the human annotation', sum(c['flagged_ok'] for c in all_rows), 1)
check('unflagged values matching the human annotation, %', round(100 * sum(c['unflagged_ok'] for c in all_rows) / sum(c['unflagged'] for c in all_rows)), 68)
check('independent relation, without regularizer (mean, SD)', rate('no_cons', 'independent'), (8.72, 2.37))
check('independent relation, with regularizer (mean, SD)', rate('full', 'independent'), (4.43, 0.94))
check('independent relation, relative reduction %', round(100 * (1 - mean_rate('full', 'independent') / mean_rate('no_cons', 'independent')), 1), 49.2)
check('training-derived relation, relative reduction %', round(100 * (1 - mean_rate('full', 'flagged') / mean_rate('no_cons', 'flagged')), 1), 50.5)
check('UzUDT only, without regularizer (mean, SD)', rate('no_cons', 'uzudt'), (9.76, 2.19))
check('UzUDT only, with regularizer (mean, SD)', rate('full', 'uzudt'), (5.58, 1.27))

print(f"\n{'All values match the manuscript.' if not failures else str(failures) + ' value(s) differ from the manuscript.'}")
sys.exit(1 if failures else 0)
