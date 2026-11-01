"""Provenance-aware data preparation with sentence-level separation of Gold and Silver.

Gold = sentences whose tokens were annotated in the original corpus before heuristic filling.
Silver = sentences that were unannotated in the original corpus and filled algorithmically.
Only deduplicated Gold sentences are allowed in dev/test. Silver data is training-only.
"""
from __future__ import annotations
import os, re, random, json, sys
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from data.preprocess import parse_conllu_file, build_label_vocabs, build_char_vocab_from_all, load_unimorph, build_unimorph_lookup, load_cse_affixes, load_segmentation, enrich_segmentation_from_cse, load_roots


def _parse_raw_with_gold_flag(path: str):
    sents=[]; cur=[]; meta={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.rstrip('\n')
            if not line.strip():
                if cur:
                    sents.append({'tokens':cur,'meta':meta,'is_gold':all(t['upos']!='_' for t in cur)})
                cur=[]; meta={}; continue
            if line.startswith('#'):
                if '=' in line:
                    k,v=line[1:].split('=',1); meta[k.strip()]=v.strip()
                continue
            p=line.split('\t')
            if len(p)>=10 and p[0].isdigit():
                cur.append({'form':p[1], 'lemma':p[2], 'upos':p[3], 'feats':p[5]})
    if cur:
        sents.append({'tokens':cur,'meta':meta,'is_gold':all(t['upos']!='_' for t in cur)})
    return sents


def sent_key(sent: Dict) -> str:
    # conservative exact-token key; normalizes apostrophe variants and whitespace/case
    forms=[]
    for t in sent['tokens']:
        x=t['form'].lower().strip().replace('’',"'").replace('ʻ',"'").replace('‘',"'").replace('`',"'")
        forms.append(x)
    return ' '.join(forms)


def _feature_signature(sent: Dict):
    sig=set()
    for t in sent['tokens']:
        sig.add('POS='+t['upos'])
        for k,v in t.get('features',{}).items():
            if k in {'Case','Number','Person','Tense','Mood','VerbForm','Voice','Degree','Polarity','Question','Copula','Polite','Possession'}:
                sig.add(f'{k}={v}')
    return sig


def stratified_sentence_split(gold: List[Dict], seed=42, dev_ratio=.10, test_ratio=.10):
    """Greedy multilabel-ish split preserving rare grammatical labels in dev/test when possible."""
    rnd=random.Random(seed)
    items=list(gold); rnd.shuffle(items)
    # rare-first ordering
    freq=Counter()
    sigs=[]
    for s in items:
        sg=_feature_signature(s); sigs.append(sg); freq.update(sg)
    order=sorted(range(len(items)), key=lambda i: (min([freq[x] for x in sigs[i]] or [10**9]), -len(sigs[i])))
    n=len(items); n_test=max(1,round(n*test_ratio)); n_dev=max(1,round(n*dev_ratio)); n_train=n-n_test-n_dev
    targets={'train':n_train,'dev':n_dev,'test':n_test}
    out={'train':[],'dev':[],'test':[]}; counts={k:Counter() for k in out}
    # Encourage rare labels in eval but cap sizes.
    for i in order:
        s=items[i]; sg=sigs[i]
        candidates=[k for k in ('test','dev','train') if len(out[k])<targets[k]]
        def score(k):
            missing=sum(1 for x in sg if counts[k][x]==0)
            rarity=sum(1.0/max(1,freq[x]) for x in sg)
            fill=(targets[k]-len(out[k]))/max(1,targets[k])
            eval_bonus=0.15 if k in ('test','dev') else 0.0
            return 3*missing + rarity + fill + eval_bonus
        k=max(candidates,key=score)
        out[k].append(s); counts[k].update(sg)
    # Guarantee that no dev/test grammatical label is unseen in train.
    # If a label occurs only in evaluation, move the corresponding sentence to train
    # and backfill evaluation with a train sentence whose labels remain represented.
    changed = True
    while changed:
        changed = False
        train_labels = set().union(*[_feature_signature(s) for s in out['train']]) if out['train'] else set()
        for split_name in ('dev','test'):
            bad_idx = next((i for i,s in enumerate(out[split_name]) if not _feature_signature(s).issubset(train_labels)), None)
            if bad_idx is None:
                continue
            moved = out[split_name].pop(bad_idx)
            out['train'].append(moved)
            # Backfill with a train sentence whose removal does not remove the last train occurrence of any label.
            train_counts = Counter()
            for s in out['train']: train_counts.update(_feature_signature(s))
            repl_idx = None
            for i,s in enumerate(out['train']):
                if s is moved: continue
                sg=_feature_signature(s)
                if all(train_counts[x] >= 2 for x in sg):
                    repl_idx=i; break
            if repl_idx is not None:
                out[split_name].append(out['train'].pop(repl_idx))
            changed = True
            break
    return out


def build_token_vocab(sentences: List[Dict], min_freq=1, max_size=50000):
    c=Counter(t['form'].lower() for s in sentences for t in s['tokens'])
    toks=['<PAD>','<UNK>']+[w for w,n in c.most_common(max_size-2) if n>=min_freq]
    return {w:i for i,w in enumerate(toks)}


def normalized_compatibility_from_gold(sentences: List[Dict]):
    """Compatibility map in model task/value naming, learned only from Gold train data.
    NONE compatibility is handled inside ConsistencyRegularizer.
    """
    fmap={'Case':'case','Number':'number','Person':'person','Possession':'possession','Tense':'tense',
          'Mood':'mood','VerbForm':'verb_form','Voice':'verb_voice','Degree':'degree','Polarity':'negative',
          'Question':'question','Copula':'copula','Polite':'polite'}
    valid=defaultdict(set)
    for s in sentences:
        for t in s['tokens']:
            p=t['upos']
            for feat,val in t.get('features',{}).items():
                task=fmap.get(feat)
                if task: valid[('pos',task)].add((p,val))
    return dict(valid)


def prepare_scientific_data(cfg, seed=None):
    seed = cfg.seed if seed is None else seed
    raw_path=os.path.join(cfg.DATASET_DIR,'Uzbek_Morphology_Corpus.conllu')
    unified_path=os.path.join(cfg.DATASET_DIR,'Uzbek_Unified_Morphological_Dataset.conllu')
    raw=_parse_raw_with_gold_flag(raw_path)
    unified=parse_conllu_file(unified_path)
    if len(raw)!=len(unified): raise RuntimeError(f'Raw/unified sentence mismatch: {len(raw)} vs {len(unified)}')

    gold_by_key={}; silver_by_key={}
    for r,u in zip(raw,unified):
        k=sent_key(u)
        if r['is_gold']:
            gold_by_key.setdefault(k,u)   # deduplicate Gold
        else:
            silver_by_key.setdefault(k,u) # deduplicate Silver
    # Gold always wins; no exact sentence from eval can exist in silver train.
    for k in list(silver_by_key):
        if k in gold_by_key: del silver_by_key[k]

    gold=list(gold_by_key.values()); silver=list(silver_by_key.values())
    split=stratified_sentence_split(gold, seed=seed, dev_ratio=cfg.dev_ratio, test_ratio=cfg.dev_ratio)
    gold_train,dev,test=split['train'],split['dev'],split['test']

    # Aux resources do NOT create evaluation labels.
    unimorph=load_unimorph(cfg.UNIMORPH_EXTENDED)
    unimorph_lookup=build_unimorph_lookup(unimorph)
    cse=load_cse_affixes(cfg.CSE_AFFIXES)
    seg=enrich_segmentation_from_cse(cse, load_segmentation(cfg.SEGMENTATION_FILE))

    # Vocabularies built from Gold train only for labels; chars may include aux orthography.
    label_vocabs=build_label_vocabs(gold_train)
    label_vocabs['char']=build_char_vocab_from_all(gold_train,unimorph,seg)
    label_vocabs['token']=build_token_vocab(gold_train + silver)

    compat=normalized_compatibility_from_gold(gold_train)
    roots=load_roots(cfg.CSE_ROOTS)

    return {
        'gold_train': gold_train, 'silver_train': silver, 'dev':dev, 'test':test,
        'ud_data': {'train':gold_train,'dev':dev,'test':test},
        'label_vocabs':label_vocabs, 'compat_matrix':compat,
        'unimorph':unimorph,'unimorph_lookup':unimorph_lookup,'cse_entries':cse,'seg_entries':seg,'roots':roots,
        'stats': {
            'raw_sentences':len(raw), 'gold_raw':sum(r['is_gold'] for r in raw),
            'gold_unique':len(gold), 'silver_unique':len(silver),
            'gold_train':len(gold_train), 'dev':len(dev), 'test':len(test), 'seed':seed
        }
    }


def write_conllu(sentences: List[Dict], path: str):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for i,s in enumerate(sentences,1):
            f.write(f'# sent_id = strict_{i}\n')
            f.write('# text = '+' '.join(t['form'] for t in s['tokens'])+'\n')
            for j,t in enumerate(s['tokens'],1):
                feats=t.get('features',{})
                featstr='|' .join(f'{k}={v}' for k,v in sorted(feats.items())) if feats else '_'
                f.write(f"{j}\t{t['form']}\t{t['lemma']}\t{t['upos']}\t_\t{featstr}\t0\t_\t_\t_\n")
            f.write('\n')


def audit_and_export(cfg, seed=None):
    d=prepare_scientific_data(cfg,seed)
    out=os.path.join(cfg.DATASET_DIR,'strict_splits')
    write_conllu(d['gold_train'],os.path.join(out,'gold_train.conllu'))
    write_conllu(d['silver_train'],os.path.join(out,'silver_train.conllu'))
    write_conllu(d['dev'],os.path.join(out,'gold_dev.conllu'))
    write_conllu(d['test'],os.path.join(out,'gold_test.conllu'))
    with open(os.path.join(out,'split_stats.json'),'w',encoding='utf-8') as f: json.dump(d['stats'],f,indent=2)
    return d

if __name__=='__main__':
    from config import cfg
    d=audit_and_export(cfg)
    print(json.dumps(d['stats'],indent=2))
