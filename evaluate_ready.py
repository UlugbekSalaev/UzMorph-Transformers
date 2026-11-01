"""Evaluation: strict 14-attribute exact match, lemma free decoding,
OOV analysis, gate analysis, and invalid-combination rate.
"""
import math
from collections import defaultdict, Counter
import torch
from data.dataset import ALL_TASKS


def _f1(y_true,y_pred,labels):
    vals=[]
    for c in labels:
        tp=sum(t==c and p==c for t,p in zip(y_true,y_pred))
        fp=sum(t!=c and p==c for t,p in zip(y_true,y_pred))
        fn=sum(t==c and p!=c for t,p in zip(y_true,y_pred))
        pr=tp/(tp+fp) if tp+fp else 0.0
        rc=tp/(tp+fn) if tp+fn else 0.0
        f=2*pr*rc/(pr+rc) if pr+rc else 0.0
        vals.append(f)
    return sum(vals)/len(vals) if vals else 0.0


def _decode_chars(ids, inv):
    out=[]
    for z in ids:
        z=int(z)
        if z in (0,3): break
        if z!=1: out.append(inv.get(z,''))
    return ''.join(out)


def levenshtein(a,b):
    if not b:return 0 if not a else len(a)
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]


def evaluate_scientific(model, loader, vocabs, device, train_forms=None, compat_matrix=None):
    model.eval(); train_forms=set(train_forms or [])
    inv={k:{i:v for v,i in x.items()} for k,x in vocabs.items() if isinstance(x,dict)}
    ys=defaultdict(list); ps=defaultdict(list)
    strict=[]; active=[]; lemma_ok=[]; cers=[]
    oov_strict=[]; seen_strict=[]; gate_oov=[]; gate_seen=[]; gate_all=[]
    invalid=0; checked=0
    compat_matrix=compat_matrix or {}

    with torch.no_grad():
        for batch in loader:
            char=batch['char_ids'].to(device); tok=batch['token_ids'].to(device); lengths=batch['lengths'].to(device)
            # Free-running lemma decoding: no target characters are fed at inference.
            out=model(char, token_ids=tok, lengths=lengths, target_lemma_chars=None)
            pred={k:v.argmax(-1) for k,v in out['logits'].items()}
            supported_tasks=list(pred.keys())
            lemma_pred=out['lemma_logits'].argmax(-1)
            gate=out.get('gate')
            char_inv=inv['char']
            for b in range(char.size(0)):
                L=int(lengths[b])
                forms=batch['forms'][b]
                for j in range(L):
                    # POS is always defined; PAD excluded by sequence length.
                    tok_strict=True; tok_active=True
                    for task in supported_tasks:
                        t=int(batch[task][b,j]); p=int(pred[task][b,j])
                        ys[task].append(t); ps[task].append(p)
                        if p!=t: tok_strict=False
                        if t!=1 and p!=t: tok_active=False
                    strict.append(tok_strict); active.append(tok_active)
                    form=forms[j].lower()
                    (oov_strict if form not in train_forms else seen_strict).append(tok_strict)
                    if gate is not None:
                        g=float(gate[b,j].mean().cpu()); gate_all.append(g)
                        (gate_oov if form not in train_forms else gate_seen).append(g)
                    # lemma target from tensor, but only as reference
                    ref=_decode_chars(batch['lemma_chars'][b,j].tolist(),char_inv)
                    hyp=_decode_chars(lemma_pred[b,j].tolist(),char_inv)
                    lemma_ok.append(ref==hyp); cers.append(levenshtein(hyp,ref)/max(1,len(ref)))
                    # compatibility from predicted labels
                    pos_name=inv['pos'].get(int(pred['pos'][b,j]),'<UNK>')
                    for (a,task), valid in compat_matrix.items():
                        if a!='pos' or task not in pred: continue
                        val=inv[task].get(int(pred[task][b,j]),'<UNK>')
                        if val in ('<NONE>','<PAD>'): continue
                        checked+=1
                        if (pos_name,val) not in valid: invalid+=1

    m={}
    m['strict_bundle_exact_match']=100*sum(strict)/max(1,len(strict))
    m['active_feature_exact_match']=100*sum(active)/max(1,len(active))
    m['lemma_exact_match']=100*sum(lemma_ok)/max(1,len(lemma_ok))
    m['lemma_cer']=sum(cers)/max(1,len(cers))
    m['oov_strict_exact_match']=100*sum(oov_strict)/max(1,len(oov_strict)) if oov_strict else None
    m['seen_strict_exact_match']=100*sum(seen_strict)/max(1,len(seen_strict)) if seen_strict else None
    m['oov_token_count']=len(oov_strict); m['seen_token_count']=len(seen_strict)
    for task in list(ys.keys()):
        yt,yp=ys[task],ps[task]
        m[f'{task}_accuracy_all_states']=100*sum(a==b for a,b in zip(yt,yp))/max(1,len(yt))
        # Scientific macro-F1 on annotated (non-NONE) target classes; false positives still count.
        labels=sorted(c for c in set(yt) if c not in (0,1))
        idx=[i for i,t in enumerate(yt) if t!=0]
        m[f'{task}_macro_f1']=100*_f1([yt[i] for i in idx],[yp[i] for i in idx],labels)
    if gate_all:
        import statistics
        m['gate_mean']=statistics.fmean(gate_all)
        m['gate_context_fraction']=sum(g>0.5 for g in gate_all)/len(gate_all)
        m['gate_mean_oov']=statistics.fmean(gate_oov) if gate_oov else None
        m['gate_mean_seen']=statistics.fmean(gate_seen) if gate_seen else None
    m['invalid_combination_rate']=100*invalid/max(1,checked)
    m['invalid_checked_predictions']=checked
    return m
