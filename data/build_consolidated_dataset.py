import os
import json
from datetime import datetime

def make_conllu_line(t_idx, token):
    form = token.get('form', '_')
    lemma = token.get('lemma', '_')
    pos = token.get('pos', '_')
    
    # Extract features matching our 14 standard attributes
    feats = []
    # Skip core non-feature keys
    skip_keys = {'form', 'lemma', 'pos', 'tokens'}
    
    for k, v in token.items():
        if k in skip_keys:
            continue
        if v and str(v).lower() != 'none' and v != '_':
            feats.append(f"{str(k).capitalize()}={v}")
            
    feat_str = "|".join(sorted(feats)) if feats else "_"
    
    # CoNLL-U Format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    return f"{t_idx+1}\t{form}\t{lemma}\t{pos}\t_\t{feat_str}\t_\t_\t_\t_\n"


def parse_conllu_file(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        current_tokens = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if current_tokens:
                    sentences.append({'tokens': current_tokens})
                    current_tokens = []
                continue
            
            parts = line.split('\t')
            if len(parts) >= 10:
                token = {
                    'form': parts[1],
                    'lemma': parts[2],
                    'pos': parts[3]
                }
                
                feats_str = parts[5]
                if feats_str != '_':
                    for feat in feats_str.split('|'):
                        if '=' in feat:
                            fk, fv = feat.split('=', 1)
                            token[fk.lower()] = fv
                
                current_tokens.append(token)
                
        if current_tokens:
            sentences.append({'tokens': current_tokens})
    return sentences

def build_unified_corpus():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Source paths
    ud_train = os.path.join(base_dir, 'Dataset', 'UD', 'train_merged.conllu')
    ud_test = os.path.join(base_dir, 'Dataset', 'UD', 'uz_uzudt-ud-test.conllu')
    
    silver_news_1 = os.path.join(base_dir, 'data', 'silver_datasets', 'silver_uzbek_corpus.json')
    silver_news_2 = os.path.join(base_dir, 'data', 'silver_datasets', 'silver_news_2020_30K.json')
    
    # Output path
    output_path = os.path.join(base_dir, 'Uzbek_Unified_Sentence_Corpus_35K.conllu')
    
    print(f"Loading Gold UD Datasets...")
    all_sentences = []
    if os.path.exists(ud_train):
        ud_t = parse_conllu_file(ud_train)
        all_sentences.extend(ud_t)
        print(f"  + Added {len(ud_t)} sentences from train_merged (2 UD sets combined)")
    if os.path.exists(ud_test):
        ud_tst = parse_conllu_file(ud_test)
        all_sentences.extend(ud_tst)
        print(f"  + Added {len(ud_tst)} sentences from test")
        
    print(f"\nLoading Sentence-Level Silver News Corpora...")
    for s_path in [silver_news_1, silver_news_2]:
        if os.path.exists(s_path):
            try:
                with open(s_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                    
                # We specifically EXCLUDE unimorph word-level data, only taking sentence-level datasets
                # The silver news ones are guaranteed to be sentence level context.
                all_sentences.extend(news_data)
                print(f"  + Added {len(news_data)} full-context sentences from {os.path.basename(s_path)}")
            except Exception as e:
                print(f"  - Failed to load {s_path}: {e}")
                
    print(f"\nTotal Contextual Sentences Gathered: {len(all_sentences)}")
    
    print(f"Writing to Unified CoNLL-U Standard Format...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Header info
        out_f.write(f"# Uzbek Unified Sentence Corpus (Sentence-Level Morphological Tags)\n")
        out_f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_f.write(f"# Total Sentences: {len(all_sentences)}\n\n")
        
        for idx, sent in enumerate(all_sentences):
            out_f.write(f"# sent_id = unified_{idx+1}\n")
            text = " ".join([t.get('form', '') for t in sent['tokens']])
            out_f.write(f"# text = {text}\n")
            
            for t_idx, token in enumerate(sent['tokens']):
                out_f.write(make_conllu_line(t_idx, token))
            out_f.write("\n")
            
    print(f"Saved successfully to: {output_path}")

if __name__ == '__main__':
    build_unified_corpus()
