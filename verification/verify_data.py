"""Checks the data statements of the manuscript (Sections 4.1-4.3). No training or GPU is needed.

    python verification/verify_data.py

Every line compares a recomputed value with the value stated in the manuscript.
"""
import csv
import difflib
import os
import re
import statistics
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import cfg
from data.scientific_data import prepare_scientific_data, write_conllu

DATA = os.path.join(ROOT, 'Dataset')
SPLITS = os.path.join(DATA, 'splits')
EXT = os.path.join(ROOT, 'verification', 'external')
SEEDS = (17, 29, 43, 71, 97)
failures = 0


def check(name, value, expected):
    global failures
    ok = value == expected
    failures += not ok
    print(f"  [{'OK' if ok else 'DIFF'}] {name}: {value}" + ('' if ok else f' (manuscript: {expected})'))


def read(path):
    out, cur = [], []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('#'):
                continue
            if not line.strip():
                if cur:
                    out.append(cur)
                    cur = []
                continue
            t = line.split('\t')
            if len(t) >= 10 and t[0].isdigit():
                cur.append(t)
    if cur:
        out.append(cur)
    return out


def norm(x):
    x = x.lower()
    for c in "ʻʼ‘’`´ʹ":
        x = x.replace(c, "'")
    return x


def key(sent):
    return norm(' '.join(t[1] for t in sent))


def letters(x):
    return re.sub(r"[^a-z']", '', norm(x))


# The Silver cascade of data/unified_builder.py
def parse_pos(v):
    v = v.strip().upper()
    return {'N': 'NOUN', 'NOUN': 'NOUN', 'V': 'VERB', 'VERB': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
            'PRN': 'PRON', 'PRON': 'PRON', 'NUM': 'NUM'}.get(v, 'NOUN')


roots = {}
with open(os.path.join(DATA, 'UzMorphAnalyser', 'root.csv'), encoding='utf-8', errors='ignore') as f:
    for row in csv.DictReader(f):
        stem, pos = row.get('stem', '').strip().lower(), row.get('pos', '').strip()
        if stem and pos:
            roots[stem] = parse_pos(pos)


def is_punct(x):
    return bool(re.match(r'^[.,;:!?\-\'\"()\[\]{}]+$', x))


def is_num(x):
    return bool(re.match(r'^[0-9]+([.,][0-9]+)?$', x))


def cascade(form, memory):
    low = form.lower()
    if is_punct(form):
        return 'punct', (form, 'PUNCT', '_')
    if is_num(form):
        return 'num', (form, 'NUM', 'NumType=Card')
    if low in memory:
        return 'memory', memory[low]
    for i in range(len(low), 0, -1):
        if low[:i] in roots:
            return 'root', (low[:i], roots[low[:i]], '_')
    return 'fallback', (low, 'PROPN' if form.istitle() else 'NOUN', '_')


raw = read(os.path.join(DATA, 'Uzbek_Morphology_Corpus.conllu'))
uni = read(os.path.join(DATA, 'Uzbek_Unified_Morphological_Dataset.conllu'))
gold_raw = [s for s in raw if all(t[3] != '_' for t in s)]
silver_raw = [s for s in raw if all(t[3] == '_' for t in s)]

print('Corpus')
check('sentence instances', len(raw), 17837)
check('annotated instances', len(gold_raw), 2859)
check('unannotated instances', len(silver_raw), 14978)
gold = {}
for s in gold_raw:
    gold.setdefault(key(s), s)
silver = {}
for s in silver_raw:
    if key(s) not in gold:
        silver.setdefault(key(s), s)
check('unique Gold sentences', len(gold), 1098)
check('Gold tokens', sum(map(len, gold.values())), 8986)
check('unique Silver sentences', len(silver), 14863)
check('Silver tokens', sum(map(len, silver.values())), 112693)

print('\nGold composition')
templates = {key(s): s for s in read(os.path.join(EXT, 'template_sentences.conllu'))}
uzudt = {key(s): s for name in ('uz_uzudt-ud-train.conllu', 'uz_uzudt-ud-test.conllu') for s in read(os.path.join(EXT, name))}
exact = [k for k in gold if k in uzudt]
near = [k for k in gold if k not in uzudt and k not in templates]
template_keys = [k for k in gold if k in templates]
check('template sentences', len(template_keys), 418)
check('template tokens', sum(len(gold[k]) for k in template_keys), 1452)
check('sentences matching UzUDT exactly', len(exact), 542)
check('other treebank-derived sentences', len(near), 138)
check('treebank-derived tokens', sum(len(gold[k]) for k in exact + near), 7534)
candidates = {letters(k): k for k in uzudt}
similarity = []
for k in near:
    x = letters(k)
    if x in candidates:
        similarity.append(1.0)
    else:
        m = difflib.get_close_matches(x, list(candidates), n=1, cutoff=0.0)
        similarity.append(difflib.SequenceMatcher(None, x, m[0]).ratio() if m else 0.0)
check('of those, character similarity >= 0.80', sum(v >= 0.80 for v in similarity), 138)
n = up = lem = 0
for k in exact:
    for a, b in zip(gold[k], uzudt[k]):
        n += 1
        up += a[3] == b[3]
        lem += a[2] == b[2]
check('agreement with current UzUDT, UPOS %', round(100 * up / n, 2), 99.47)
check('agreement with current UzUDT, lemma %', round(100 * lem / n, 2), 68.30)
patterns = {' '.join(t[3] for t in gold[k]) for k in template_keys}
check('template UPOS patterns', len(patterns), 4)
vocab = {(t[3], t[2]) for k in template_keys for t in gold[k] if t[3] != 'PUNCT'}
check('template lemmas', len(vocab), 47)

print('\nSilver cascade')
memory = {}
for s in gold_raw:
    for t in s:
        memory[t[1].lower()] = (t[2], t[3], t[5])
same = total = 0
n_unified_gold = len(gold_raw)
for r, u in zip(silver_raw, uni[n_unified_gold:]):
    for a, b in zip(r, u):
        total += 1
        same += cascade(a[1], memory)[1] == (b[2], b[3], b[5])
check('stored Silver labels reproduced by the cascade', f'{same}/{total}', f'{total}/{total}')
rules = Counter(cascade(t[1], memory)[0] for s in silver.values() for t in s)
n_tok = sum(rules.values())
check('rule 1, numbers %', round(100 * rules['num'] / n_tok, 2), 0.76)
check('rule 2, Gold memory %', round(100 * rules['memory'] / n_tok, 2), 38.00)
check('rule 3, root lexicon %', round(100 * rules['root'] / n_tok, 2), 58.50)
check('rule 4, default %', round(100 * rules['fallback'] / n_tok, 2), 2.74)
check('root lexicon entries', len(roots), 100478)

print('\nSplits (regenerated with data/scientific_data.py, compared with Dataset/splits)')
rows = []
for seed in SEEDS:
    d = prepare_scientific_data(cfg, seed)
    tr, dv, te = d['gold_train'], d['dev'], d['test']
    identical = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, sents in (('gold_train', tr), ('gold_dev', dv), ('gold_test', te)):
            path = os.path.join(tmp, name + '.conllu')
            write_conllu(sents, path)
            with open(path, 'rb') as f:
                mine = f.read().replace(b'\r\n', b'\n')
            with open(os.path.join(SPLITS, f'seed_{seed}', name + '.conllu'), 'rb') as f:
                identical &= mine == f.read().replace(b'\r\n', b'\n')
    forms = {t['form'].lower() for s in tr for t in s['tokens']}
    toks = [t['form'].lower() for s in te for t in s['tokens']]
    oov = sum(w not in forms for w in toks)

    def k2(s):
        return norm(' '.join(t['form'] for t in s['tokens']))

    labels = lambda ss: {('U', t['upos']) for s in ss for t in s['tokens']} | {x for s in ss for t in s['tokens'] for x in t.get('features', {}).items()}
    sets = [{k2(s) for s in x} for x in (tr, dv, te, d['silver_train'])]
    overlap = sum(len(sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4))
    unseen = len((labels(dv) | labels(te)) - labels(tr))

    # Word-level overlap of the Silver memory with held-out data
    sf = Counter(t['form'].lower() for s in d['silver_train'] for t in s['tokens']
                 if not is_punct(t['form']) and not is_num(t['form']) and t['form'].lower() in memory)
    held = {t['form'].lower() for s in dv + te for t in s['tokens']} - forms
    test_only = set(toks) - forms
    silver_share = 100 * sum(c for w, c in sf.items() if w in held) / sum(len(s['tokens']) for s in d['silver_train'])
    oov_share = 100 * sum(w in test_only and w in sf for w in toks) / oov

    # Cascade on held-out Gold, memory from Gold training only, punctuation excluded
    train_memory = {}
    for s in tr:
        for t in s['tokens']:
            train_memory[t['form'].lower()] = (t['lemma'], t['upos'], '|'.join(f'{a}={b}' for a, b in sorted(t.get('features', {}).items())) or '_')
    c = Counter()
    for s in dv + te:
        for t in s['tokens']:
            if t['upos'] == 'PUNCT':
                continue
            _, (lemma, upos, feats) = cascade(t['form'], train_memory)
            gf = '|'.join(f'{a}={b}' for a, b in sorted(t.get('features', {}).items())) or '_'
            c['n'] += 1
            c['upos'] += upos == t['upos']
            c['lemma'] += lemma.lower() == t['lemma'].lower()
            c['feats'] += feats == gf
    tmpl_te = sum(k2(s) in templates for s in te)
    tmpl_tok = sum(len(s['tokens']) for s in te if k2(s) in templates)
    rows.append(dict(seed=seed, identical=identical, sizes=(len(tr), len(dv), len(te)), toks=len(toks), oov=oov, overlap=overlap, unseen=unseen,
                     silver_share=silver_share, oov_share=oov_share, tmpl=tmpl_te, tmpl_share=100 * tmpl_tok / len(toks), n=c['n'],
                     upos=100 * c['upos'] / c['n'], lemma=100 * c['lemma'] / c['n'], feats=100 * c['feats'] / c['n']))
    r = rows[-1]
    print(f"  seed {seed}: identical={identical}, sizes={r['sizes']}, sentence overlap={overlap}, unseen labels={unseen}, test OOV={oov}, seen={len(toks) - oov}")

check('all five splits identical to Dataset/splits', all(r['identical'] for r in rows), True)
check('partition sizes', {r['sizes'] for r in rows}, {(878, 110, 110)})
check('sentence overlap between partitions', sum(r['overlap'] for r in rows), 0)
check('dev/test labels absent from Gold training', sum(r['unseen'] for r in rows), 0)
check('test OOV range', (min(r['oov'] for r in rows), max(r['oov'] for r in rows)), (292, 318))
check('test seen range', (min(r['toks'] - r['oov'] for r in rows), max(r['toks'] - r['oov'] for r in rows)), (679, 740))
check('pooled test tokens', sum(r['toks'] for r in rows), 5139)
check('template sentences per test set', (min(r['tmpl'] for r in rows), max(r['tmpl'] for r in rows)), (25, 30))
check('template share of test tokens %', (round(min(r['tmpl_share'] for r in rows), 1), round(max(r['tmpl_share'] for r in rows), 1)), (8.8, 10.7))

print('\nWord-level overlap between the Silver memory and held-out Gold')
check('Silver tokens with held-out-only forms %', (round(min(r['silver_share'] for r in rows), 2), round(max(r['silver_share'] for r in rows), 2)), (1.77, 2.18))
check('Gold-training-OOV test tokens covered %', (round(min(r['oov_share'] for r in rows), 1), round(max(r['oov_share'] for r in rows), 1)), (33.0, 39.4))

print('\nSilver cascade on held-out Gold (development + test, no punctuation)')
check('tokens per split', (min(r['n'] for r in rows), max(r['n'] for r in rows)), (1618, 1695))
for name, field, mean, sd in (('UPOS', 'upos', 77.95, 0.97), ('lemma', 'lemma', 63.02, 1.14), ('features', 'feats', 58.98, 0.92)):
    v = [r[field] for r in rows]
    check(f'{name} accuracy % (mean, SD)', (round(statistics.fmean(v), 2), round(statistics.stdev(v), 2)), (mean, sd))

print('\nTemplate sentences against UzMorphAnalyser')
try:
    from UzMorphAnalyser import UzMorphAnalyser
except ImportError:
    print('  skipped: pip install UzMorphAnalyser')
else:
    analyser = UzMorphAnalyser()
    n = top = verbs = 0
    for k in template_keys:
        for t in gold[k]:
            if t[3] == 'PUNCT':
                continue
            n += 1
            form = t[1].lower()
            top += norm(str(analyser.lemmatize(form))) == norm(t[2])
            if t[3] == 'VERB':
                analyses = analyser.analyze(form)
                lemmas = {norm(str(x.get('lemma'))) for x in analyses} | {norm(str(x.get('stem'))) for x in analyses}
                verbs += norm(t[2]) in lemmas and any(str(x.get('pos')).upper() == 'VERB' for x in analyses)
    check('non-punctuation template tokens', n, 1034)
    check('top lemma matches %', round(100 * top / n, 2), 90.72)
    check('verbs with the annotated lemma and POS', verbs, 418)

print(f"\n{'All values match the manuscript.' if not failures else str(failures) + ' value(s) differ from the manuscript.'}")
sys.exit(1 if failures else 0)
