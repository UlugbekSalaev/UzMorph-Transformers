"""One-command experiment runner.

Examples:
  python run_ai_experiments.py --seeds 17 29 43 71 97 --variants full context morph concat no_consistency --regime curriculum --pretrain_epochs 5 --epochs 35
  python run_ai_experiments.py --seeds 17 29 43 71 97 --variants full --regime gold --epochs 35

Development and test data are Gold only. Silver is NEVER used for model selection or test.
"""
import os, json, csv, random, argparse, statistics
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from data.scientific_data import prepare_scientific_data, audit_and_export
from data.dataset import MorphDataset, collate_fn, ALL_TASKS
from model.model import UzbekMorphModel
from evaluate_ai_ready import evaluate_scientific


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def loaders(data,batch_size,regime):
    gold=MorphDataset(data['gold_train'],data['label_vocabs'])
    silver=MorphDataset(data['silver_train'],data['label_vocabs'])
    dev=MorphDataset(data['dev'],data['label_vocabs']); test=MorphDataset(data['test'],data['label_vocabs'])
    kw=dict(batch_size=batch_size,collate_fn=collate_fn)
    return (DataLoader(gold,shuffle=True,**kw), DataLoader(silver,shuffle=True,**kw),
            DataLoader(dev,shuffle=False,**kw),DataLoader(test,shuffle=False,**kw))


def variant_settings(name):
    if name=='full': return dict(fusion_mode='gated',use_consistency=True)
    if name=='no_consistency': return dict(fusion_mode='gated',use_consistency=False)
    if name=='context': return dict(fusion_mode='context',use_consistency=True)
    if name=='morph': return dict(fusion_mode='morph',use_consistency=True)
    if name=='concat': return dict(fusion_mode='concat',use_consistency=True)
    if name=='no_consistency_concat': return dict(fusion_mode='concat',use_consistency=False)
    raise ValueError(name)


def batch_loss(model,batch,device):
    char=batch['char_ids'].to(device); token=batch['token_ids'].to(device); lengths=batch['lengths'].to(device)
    lemma=batch['lemma_chars'].to(device)
    targets={k:batch[k].to(device) for k in ALL_TASKS}; targets['lemma_chars']=lemma
    out=model(char,token_ids=token,lengths=lengths,target_lemma_chars=lemma)
    return model.compute_loss(out,targets)['total_loss']


def train_stage(model,loader,dev_loader,data,device,epochs,lr,patience,checkpoint):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=lr*0.05)
    train_forms={t['form'].lower() for s in data['gold_train'] for t in s['tokens']}
    best=-1; wait=0; history=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for batch in loader:
            opt.zero_grad(set_to_none=True); loss=batch_loss(model,batch,device)
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch {ep}: {loss.item()}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step()
            total+=loss.item(); n+=1
        sched.step()
        devm=evaluate_scientific(model,dev_loader,data['label_vocabs'],device,train_forms,data['compat_matrix'])
        score=devm['active_feature_exact_match']
        history.append({'epoch':ep,'train_loss':total/max(1,n),**devm})
        print(f"  ep {ep:02d} loss={total/max(1,n):.4f} dev_active_EM={score:.2f} dev_strict_EM={devm['strict_bundle_exact_match']:.2f} lemma={devm['lemma_exact_match']:.2f}")
        if score>best:
            best=score;wait=0
            torch.save({'model_state_dict':model.state_dict(),'label_vocabs':data['label_vocabs'],'seed':data['stats']['seed']},checkpoint)
        else:
            wait+=1
            if wait>=patience: break
    if os.path.exists(checkpoint): model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model_state_dict'])
    return history


def run_one(seed,variant,regime,args):
    set_seed(seed); data=prepare_scientific_data(cfg,seed)
    gold_loader,silver_loader,dev_loader,test_loader=loaders(data,args.batch_size,regime)
    st=variant_settings(variant); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=UzbekMorphModel(cfg,data['label_vocabs'],data['compat_matrix'],encoder_type='transformer',
                         use_gating=(st['fusion_mode']=='gated'),use_consistency=st['use_consistency'],
                         use_uncertainty=True,fusion_mode=st['fusion_mode']).to(device)
    run_dir=os.path.join(cfg.RESULTS_DIR,'ai_ready',regime,variant,f'seed_{seed}');os.makedirs(run_dir,exist_ok=True)
    ck=os.path.join(run_dir,'best.pt')
    hist=[]
    if regime=='curriculum':
        print(f'[{variant} seed={seed}] Silver pretraining (training-only auxiliary labels)')
        # short, lower-LR auxiliary stage; never select/report on silver.
        hist+=train_stage(model,silver_loader,dev_loader,data,device,args.pretrain_epochs,args.lr*0.5,max(2,args.patience),ck+'.pre')
    train_loader = gold_loader if regime in ('gold','curriculum') else DataLoader(MorphDataset(data['gold_train']+data['silver_train'],data['label_vocabs']),batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    print(f'[{variant} seed={seed}] Gold supervised stage')
    hist+=train_stage(model,train_loader,dev_loader,data,device,args.epochs,args.lr,args.patience,ck)
    train_forms={t['form'].lower() for s in data['gold_train'] for t in s['tokens']}
    metrics=evaluate_scientific(model,test_loader,data['label_vocabs'],device,train_forms,data['compat_matrix'])
    metrics.update({'seed':seed,'variant':variant,'regime':regime,**data['stats']})
    with open(os.path.join(run_dir,'metrics.json'),'w',encoding='utf-8') as f: json.dump(metrics,f,indent=2)
    with open(os.path.join(run_dir,'history.json'),'w',encoding='utf-8') as f: json.dump(hist,f,indent=2)
    print('TEST',json.dumps({k:round(v,3) if isinstance(v,float) else v for k,v in metrics.items() if k in ('strict_bundle_exact_match','active_feature_exact_match','lemma_exact_match','pos_macro_f1','invalid_combination_rate','oov_strict_exact_match')},indent=2))
    return metrics


def aggregate(rows,outdir):
    os.makedirs(outdir,exist_ok=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with open(os.path.join(outdir,'all_runs.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    groups={}
    for r in rows: groups.setdefault((r['variant'],r['regime']),[]).append(r)
    summary=[]
    report_keys=['strict_bundle_exact_match','active_feature_exact_match','lemma_exact_match','pos_macro_f1','case_macro_f1','number_macro_f1','person_macro_f1','tense_macro_f1','mood_macro_f1','verb_voice_macro_f1','verb_form_macro_f1','negative_macro_f1','invalid_combination_rate','oov_strict_exact_match','seen_strict_exact_match','gate_mean']
    for (variant,regime),rs in groups.items():
        row={'variant':variant,'regime':regime,'n_seeds':len(rs)}
        for k in report_keys:
            vals=[r[k] for r in rs if isinstance(r.get(k),(int,float)) and r.get(k) is not None]
            if vals:
                row[k+'_mean']=statistics.fmean(vals);row[k+'_std']=statistics.stdev(vals) if len(vals)>1 else 0.0
        summary.append(row)
    keys2=sorted(set().union(*(r.keys() for r in summary)))
    with open(os.path.join(outdir,'summary_mean_std.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys2);w.writeheader();w.writerows(summary)
    with open(os.path.join(outdir,'summary.json'),'w',encoding='utf-8') as f: json.dump(summary,f,indent=2)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[17,29,43,71,97])
    ap.add_argument('--variants',nargs='+',default=['full','context','morph','concat','no_consistency'])
    ap.add_argument('--regime',choices=['gold','mixed','curriculum'],default='gold')
    ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--pretrain_epochs',type=int,default=5)
    ap.add_argument('--batch_size',type=int,default=32);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--patience',type=int,default=6)
    args=ap.parse_args(); audit_and_export(cfg,args.seeds[0])
    rows=[]
    for v in args.variants:
        for s in args.seeds: rows.append(run_one(s,v,args.regime,args))
    aggregate(rows,os.path.join(cfg.RESULTS_DIR,'ai_ready',args.regime))

if __name__=='__main__': main()
