import torch
import sys
import os
sys.path.insert(0, r'c:\PhD Dissertation\Dastur\Proposed_Methodology')
from data.preprocess import run_preprocessing_pipeline
from data.dataset import MorphDataset, collate_fn
from torch.utils.data import DataLoader
from model.model import UzbekMorphModel
from config import cfg

print("Running pipeline...")
data = run_preprocessing_pipeline(cfg)
dev_dataset = MorphDataset(data['ud_data']['dev'], data['label_vocabs'])
dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)

model = UzbekMorphModel(cfg, data['label_vocabs'], data['compat_matrix'])
model.eval()

exact_matches = []
pos_matches = []
preds = {}

batch = next(iter(dev_loader))
char_ids = batch['char_ids']
lengths = batch['lengths']
lemma_chars = batch['lemma_chars']
targets = {k: batch[k] for k in batch if k in data['label_vocabs']}
targets['lemma_chars'] = lemma_chars

with torch.no_grad():
    outputs = model(char_ids, lengths=lengths, target_lemma_chars=lemma_chars)

logits = outputs['logits']
tasks = list(targets.keys())
tasks.remove('lemma_chars')

B, S = targets['pos'].shape

for task in tasks:
    if task in logits:
        task_pred = torch.argmax(logits[task], dim=-1)
        preds[task] = task_pred

for b in range(B):
    l = lengths[b].item()
    for s in range(l):
        if targets['pos'][b, s].item() in (0, 1):
            continue
        
        p_val_pos = preds['pos'][b, s].item()
        t_val_pos = targets['pos'][b, s].item()
        pos_matches.append(p_val_pos == t_val_pos)
        
        token_exact = True
        for task in ['pos', 'case', 'number', 'tense', 'person', 'possession']:
            if task in preds and task in targets:
                p_val = preds[task][b, s].item()
                t_val = targets[task][b, s].item()
                if t_val not in (0, 1) and p_val != t_val:
                    token_exact = False
                    break
        exact_matches.append(token_exact)

if exact_matches:
    print(f"Exact Match Accuracy: {(sum(exact_matches) / len(exact_matches)) * 100:.2f}%")
if pos_matches:
    print(f"POS Match Accuracy: {(sum(pos_matches) / len(pos_matches)) * 100:.2f}%")
    
print("Targets POS sample:", targets['pos'][0][:15])
print("Preds POS sample:", preds['pos'][0][:15])
