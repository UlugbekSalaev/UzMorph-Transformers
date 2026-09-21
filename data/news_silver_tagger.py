"""
512K News Corpus Silver Tagger — Haqiqiy Gaplar Asosida.

Haqiqiy o'zbek tili gaplarini (uzbek_corpus.txt + uzb_news_2020_30K-sentences.txt)
UniMorph paradigmalari orqali avtomatik morfologik belgilash (silver tagging).

Har bir manba alohida JSON faylga saqlanadi.
Training da qo'shish/chiqarish mumkin (configurable).
"""
import os
import sys
import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROPOSED_DIR = os.path.dirname(BASE_DIR)
if PROPOSED_DIR not in sys.path:
    sys.path.insert(0, PROPOSED_DIR)

from config import cfg
from data.preprocess import load_unimorph, build_unimorph_lookup

# =====================================================================
# Corpus File Paths
# =====================================================================

DATASET_BASE = os.path.join(os.path.dirname(PROPOSED_DIR), 'Dataset')

CORPUS_FILES = {
    'uzbek_corpus': os.path.join(DATASET_BASE, 'uzbek_corpus.txt'),
    'news_2020_30K': os.path.join(DATASET_BASE, 'uzb_news_2020_30K', 'uzb_news_2020_30K-sentences.txt'),
}


# =====================================================================
# UniMorph Tag Mapping
# =====================================================================

UNIMORPH_TAG_MAP = {
    'N': ('POS', 'NOUN'), 'V': ('POS', 'VERB'), 'ADJ': ('POS', 'ADJ'),
    'ADV': ('POS', 'ADV'), 'NUM': ('POS', 'NUM'), 'PRO': ('POS', 'PRON'),
    'PROPN': ('POS', 'PROPN'),
    'NOM': ('Case', 'Nom'), 'GEN': ('Case', 'Gen'), 'ACC': ('Case', 'Acc'),
    'DAT': ('Case', 'Dat'), 'LOC': ('Case', 'Loc'), 'ABL': ('Case', 'Abl'),
    'SG': ('Number', 'Sing'), 'PL': ('Number', 'Plur'),
    '1': ('Person', '1'), '2': ('Person', '2'), '3': ('Person', '3'),
    'PST': ('Tense', 'Past'), 'PRS': ('Tense', 'Pres'), 'FUT': ('Tense', 'Fut'),
    'IND': ('Mood', 'Ind'), 'IMP': ('Mood', 'Imp'), 'CND': ('Mood', 'Cnd'),
    'DES': ('Mood', 'Des'), 'OPT': ('Mood', 'Opt'),
    'PASS': ('Voice', 'Pass'), 'CAUS': ('Voice', 'Cau'), 'REFL': ('Voice', 'Rfl'),
    'RCP': ('Voice', 'Rcp'),
    'PART': ('VerbForm', 'Part'), 'CONV': ('VerbForm', 'Conv'),
    'INF': ('VerbForm', 'Inf'), 'FIN': ('VerbForm', 'Fin'),
    'CMP': ('Degree', 'Cmp'), 'SUP': ('Degree', 'Sup'),
    'NEG': ('Negative', 'Neg'),
}


def tag_word(word: str, unimorph_lookup: Dict) -> Dict:
    """
    Tag a single word using UniMorph lookup.
    Returns UD-format token dict. If not in UniMorph, returns NOUN with no features.
    """
    analyses = unimorph_lookup.get(word, [])
    if not analyses:
        analyses = unimorph_lookup.get(word.lower(), [])

    if not analyses:
        # Unknown word — assign NOUN with no features (model will learn from context)
        return {
            'form': word,
            'lemma': word.lower(),
            'upos': 'NOUN',
            'features': {}
        }

    analysis = analyses[0]
    tags = analysis['tags']
    pos = 'NOUN'
    features = {}

    for t in tags:
        if t in UNIMORPH_TAG_MAP:
            feat_name, feat_val = UNIMORPH_TAG_MAP[t]
            if feat_name == 'POS':
                pos = feat_val
            else:
                features[feat_name] = feat_val

    return {
        'form': word,
        'lemma': analysis.get('lemma', word.lower()),
        'upos': pos,
        'features': features
    }


def tokenize_sentence(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for Uzbek text."""
    # Remove quotes and extra whitespace
    text = re.sub(r'["""«»]', '', text)
    # Split on whitespace, filter out pure punctuation tokens
    tokens = text.split()
    result = []
    for t in tokens:
        # Strip surrounding punctuation but keep the word
        clean = t.strip('.,;:!?()[]{}–—-…')
        if clean and len(clean) >= 2:
            result.append(clean)
    return result


# =====================================================================
# Corpus Loaders
# =====================================================================

def load_plain_corpus(filepath: str) -> List[str]:
    """Load plain text file where each line is a sentence."""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(line)
    return sentences


def load_tabbed_corpus(filepath: str) -> List[str]:
    """Load tab-separated file (id\tsentence) format."""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                sentences.append(parts[1])
            else:
                sentences.append(parts[0])
    return sentences


# =====================================================================
# Main Pipeline: Build Silver-Tagged Sentence Datasets
# =====================================================================

def build_all_news_silver_datasets(
    output_dir: str = None,
    max_sentences_per_source: int = 10000,
    min_tokens: int = 3,
    max_tokens: int = 25,
    min_tagged_ratio: float = 0.3,
) -> Dict[str, List[Dict]]:
    """
    Build silver-tagged sentence datasets from real Uzbek text corpora.

    Args:
        output_dir: Directory to save per-source JSON files
        max_sentences_per_source: Maximum sentences to select from each source
        min_tokens: Minimum tokens per sentence (filter short ones)
        max_tokens: Maximum tokens per sentence (filter very long ones)
        min_tagged_ratio: Minimum ratio of UniMorph-tagged tokens in a sentence
    """
    if output_dir is None:
        output_dir = os.path.join(PROPOSED_DIR, 'data', 'silver_datasets')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("512K NEWS KORPUSIDAN SILVER-TAGGED DATASET YARATISH")
    print("=" * 65)

    # Load UniMorph lookup
    print("\n[1/3] UniMorph lug'atini yuklash...")
    unimorph = load_unimorph(cfg.UNIMORPH_EXTENDED)
    unimorph_lookup = build_unimorph_lookup(unimorph)
    print(f"  UniMorph: {len(unimorph):,} paradigma, {len(unimorph_lookup):,} noyob shakl")

    all_datasets = {}
    total_sentences = 0

    print(f"\n[2/3] Haqiqiy gaplardan silver-tagged dataset yaratish...\n")

    for source_name, filepath in CORPUS_FILES.items():
        if not os.path.exists(filepath):
            print(f"  [{source_name}] TOPILMADI: {filepath}")
            continue

        # Load raw sentences
        if source_name == 'news_2020_30K':
            raw_sents = load_tabbed_corpus(filepath)
        else:
            raw_sents = load_plain_corpus(filepath)

        print(f"  [{source_name}] {len(raw_sents):,} xom gap yuklandi")

        # Shuffle and select diverse sentences
        random.seed(42)
        random.shuffle(raw_sents)

        silver_sentences = []
        sent_id = 0

        for raw_sent in raw_sents:
            if len(silver_sentences) >= max_sentences_per_source:
                break

            words = tokenize_sentence(raw_sent)

            # Filter by sentence length
            if len(words) < min_tokens or len(words) > max_tokens:
                continue

            # Tag each word with UniMorph
            tokens = []
            tagged_count = 0
            for idx, w in enumerate(words, 1):
                token = tag_word(w, unimorph_lookup)
                token['id'] = idx
                tokens.append(token)
                if token['features']:  # has at least one morphological feature
                    tagged_count += 1

            # Filter: at least min_tagged_ratio of tokens must have features
            if tagged_count / len(tokens) < min_tagged_ratio:
                continue

            silver_sentences.append({
                'sent_id': f'{source_name}_{sent_id}',
                'tokens': tokens,
                'text': ' '.join(words),
                'source': source_name,
                'num_tokens': len(tokens),
                'tagged_ratio': round(tagged_count / len(tokens), 3),
            })
            sent_id += 1

        # Save as separate JSON file
        output_path = os.path.join(output_dir, f'silver_{source_name}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(silver_sentences, f, ensure_ascii=False, indent=1)

        all_datasets[source_name] = silver_sentences
        total_sentences += len(silver_sentences)

        # Statistics
        avg_len = sum(s['num_tokens'] for s in silver_sentences) / max(len(silver_sentences), 1)
        avg_tagged = sum(s['tagged_ratio'] for s in silver_sentences) / max(len(silver_sentences), 1)

        print(f"  [{source_name}] => {len(silver_sentences):,} silver gap saqlandi "
              f"(o'rt. uzunlik: {avg_len:.1f} token, o'rt. tagged: {avg_tagged*100:.1f}%)")
        print(f"    Saqlandi: {output_path}")

    # Save combined summary
    summary = {
        'total_sources': len(all_datasets),
        'total_sentences': total_sentences,
        'sources': {k: len(v) for k, v in all_datasets.items()},
    }
    summary_path = os.path.join(output_dir, 'silver_datasets_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[3/3] Yakuniy Statistika:")
    print(f"  Jami manbalar:      {len(all_datasets)}")
    print(f"  Jami silver gaplar: {total_sentences:,}")
    print(f"  Saqlangan papka:    {output_dir}")
    print("=" * 65)

    return all_datasets


def load_silver_datasets(silver_dir: str = None, sources: List[str] = None) -> List[Dict]:
    """
    Load saved silver datasets from JSON files.
    If sources is None, loads all available. Can specify subset to include/exclude.
    """
    if silver_dir is None:
        silver_dir = os.path.join(PROPOSED_DIR, 'data', 'silver_datasets')

    if not os.path.exists(silver_dir):
        return []

    all_sentences = []

    for filename in sorted(os.listdir(silver_dir)):
        if not filename.startswith('silver_') or not filename.endswith('.json'):
            continue
        if filename == 'silver_datasets_summary.json':
            continue

        source_name = filename.replace('silver_', '').replace('.json', '')

        if sources is not None and source_name not in sources:
            continue

        filepath = os.path.join(silver_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        all_sentences.extend(sentences)
        print(f"  Silver yuklandi: {source_name} => {len(sentences):,} gap")

    return all_sentences


if __name__ == '__main__':
    build_all_news_silver_datasets()
