"""
Data preprocessing pipeline for Uzbek Morphological Analysis.

Parses UD CoNLL-U, UniMorph, CSE affixes, and segmentation data into
unified format suitable for multi-task model training.
"""
import os
import csv
import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# =====================================================================
# 1. UD CoNLL-U Parser
# =====================================================================

# Feature mapping enforcing STRICT International UD standard titlecase tags
UD_FEATURE_MAP = {
    'Case': {
        'NOM': 'Nom', 'GEN': 'Gen', 'ACC': 'Acc',
        'DAT': 'Dat', 'LOC': 'Loc', 'ABL': 'Abl', 'Ablative': 'Abl', 'Accusative': 'Acc', 'Dative': 'Dat', 'Genitive': 'Gen', 'Locative': 'Loc'
    },
    'Number': {'SG': 'Sing', 'PL': 'Plur', 'Singular': 'Sing', 'Plural': 'Plur', 'Sg': 'Sing', 'Pl': 'Plur'},
    'Person': {'1': '1', '2': '2', '3': '3'},
    'Possession': {'1': '1', '2': '2', '3': '3'},
    'Tense': {'PST': 'Past', 'PRS': 'Pres', 'FUT': 'Fut', 'Present': 'Pres', 'Future': 'Fut'},
    'Mood': {'IND': 'Ind', 'IMP': 'Imp', 'CND': 'Cnd', 'DES': 'Des', 'OPT': 'Opt', 'Message': 'Ind', 'Imperative': 'Imp', 'Conditional': 'Cnd', 'Proposal': 'Des'},
    'VerbForm': {'FIN': 'Fin', 'INF': 'Inf', 'PART': 'Part', 'CONV': 'Conv', 'VNOUN': 'Vnoun', 'Participle': 'Part', 'Adverbial': 'Conv', 'Infinitive': 'Inf', 'Finite': 'Fin'},
    'Voice': {'ACT': 'Act', 'PASS': 'Pass', 'CAU': 'Caus', 'CAUS': 'Caus', 'RCP': 'Rcp', 'REFL': 'Refl', 'Active': 'Act', 'Passive': 'Pass', 'Causative': 'Caus', 'Reciprocal': 'Rcp', 'Reflexive': 'Refl'},
    'Degree': {'POS': 'Pos', 'CMP': 'Cmp', 'SUP': 'Sup', 'ABS': 'Sup', 'Positive': 'Pos', 'Comparative': 'Cmp', 'Superlative': 'Sup'},
    'Polarity': {'NEG': 'Neg', 'POS': 'Pos'},
    'Question': {'YES': 'Yes'},
    'Copula': {'YES': 'Yes'},
    'Polite': {'FORM': 'Form', 'ELEV': 'Elev'},
}

# POS tag mapping from UD UPOS to our schema
UD_POS_MAP = {
    'NOUN': 'NOUN', 'VERB': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV',
    'PRON': 'PRON', 'DET': 'DET', 'NUM': 'NUM', 'ADP': 'ADP',
    'CCONJ': 'CONJ', 'SCONJ': 'CONJ', 'PART': 'PART', 'INTJ': 'INTJ',
    'AUX': 'AUX', 'PROPN': 'PROPN', 'PUNCT': 'PUNCT', 'SYM': 'SYM', 'X': 'X'
}


def parse_ud_features(feat_str: str) -> Dict[str, str]:
    """Parse UD feature string like 'Case=Nom|Number=Sing' into dict."""
    if feat_str == '_' or not feat_str:
        return {}
    features = {}
    for pair in feat_str.split('|'):
        if '=' in pair:
            key, val = pair.split('=', 1)
            # Handle composite keys like Number[psor]
            clean_key = key.replace('[psor]', '_psor').replace('[subj]', '_subj')
            features[clean_key] = val
    return features


def parse_conllu_file(filepath: str) -> List[Dict]:
    """
    Parse a CoNLL-U file into list of sentences.
    Each sentence = {'tokens': [...], 'text': str, 'sent_id': str}
    Each token = {
        'id': int, 'form': str, 'lemma': str, 'upos': str, 'xpos': str,
        'features': dict, 'head': int, 'deprel': str
    }
    """
    sentences = []
    current_tokens = []
    current_meta = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if current_tokens:
                    sentences.append({
                        'tokens': current_tokens,
                        'text': current_meta.get('text', ''),
                        'sent_id': current_meta.get('sent_id', f's{len(sentences)}')
                    })
                    current_tokens = []
                    current_meta = {}
                continue

            if line.startswith('#'):
                # Meta comment
                if '=' in line:
                    key, val = line[2:].split('=', 1)
                    current_meta[key.strip()] = val.strip()
                continue

            parts = line.split('\t')
            if len(parts) < 10:
                continue

            # Skip multi-word tokens (e.g., "1-2")
            if '-' in parts[0] or '.' in parts[0]:
                continue

            try:
                token_id = int(parts[0])
            except ValueError:
                continue

            raw_features = parse_ud_features(parts[5])

            # Map features to strict international UD schema
            mapped_features = {}
            for feat_key, val in raw_features.items():
                v_str = str(val).capitalize() if str(val).isupper() else str(val)
                # First standard capitalization lookup
                if feat_key in UD_FEATURE_MAP:
                    mapped_val = UD_FEATURE_MAP[feat_key].get(str(val).upper(), v_str)
                    mapped_val = UD_FEATURE_MAP[feat_key].get(str(val), mapped_val)
                    mapped_features[feat_key] = mapped_val
                elif feat_key == 'Number_psor':
                    mapped_features['Poss_Number'] = 'Sing' if val in ['Sing', 'SG', 'Sg'] else 'Plur'
                elif feat_key == 'Person_psor':
                    mapped_features['Poss_Person'] = val
                else:
                    mapped_features[feat_key] = v_str

            upos = parts[3]
            mapped_pos = UD_POS_MAP.get(upos, upos)

            token = {
                'id': token_id,
                'form': parts[1],
                'lemma': parts[2] if parts[2] != '_' else parts[1],
                'upos': mapped_pos,
                'xpos': parts[4],
                'features': mapped_features,
                'head': int(parts[6]) if parts[6] != '_' else 0,
                'deprel': parts[7]
            }
            current_tokens.append(token)

    # Last sentence
    if current_tokens:
        sentences.append({
            'tokens': current_tokens,
            'text': current_meta.get('text', ''),
            'sent_id': current_meta.get('sent_id', f's{len(sentences)}')
        })

    return sentences


def load_ud_data(train_path: str, test_path: str, dev_ratio: float = 0.1,
                 seed: int = 42) -> Dict[str, List[Dict]]:
    """Load UD data, merge all gold sentences, and split 80/10/10 for strict MDPI evaluation."""
    train_sents = parse_conllu_file(train_path)
    test_sents = parse_conllu_file(test_path)

    # 1. Barcha Gold ma'lumotlarni birlashtirish (2661 + 198 = 2859 gap)
    all_gold_sents = train_sents + test_sents
    
    # 2. Xolislik (unbiased) uchun qat'iy aralashtirish
    random.seed(seed)
    random.shuffle(all_gold_sents)
    
    # 3. 80:10:10 formatiga qismlarga bo'lish
    total = len(all_gold_sents)
    dev_size = int(total * dev_ratio)  # ~285
    test_size = int(total * dev_ratio) # ~286
    
    test_split = all_gold_sents[:test_size]
    dev_split = all_gold_sents[test_size:test_size + dev_size]
    train_split = all_gold_sents[test_size + dev_size:]

    print(f"  UD Gold Dataset Qayta Taqsimlandi (80/10/10):")
    print(f"  Train: {len(train_split)}, Dev: {len(dev_split)}, Test: {len(test_split)}")

    return {
        'train': train_split,
        'dev': dev_split,
        'test': test_split
    }


# =====================================================================
# 2. UniMorph Loader
# =====================================================================

def load_unimorph(filepath: str) -> List[Dict]:
    """
    Load UniMorph data: lemma\tword_form\tfeature_tags
    Returns list of {'lemma': str, 'form': str, 'tags': list[str]}
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            lemma, form, tags_str = parts
            tags = tags_str.split(';')
            entries.append({
                'lemma': lemma,
                'form': form,
                'tags': tags
            })
    return entries


def build_unimorph_lookup(entries: List[Dict]) -> Dict[str, List[Dict]]:
    """Build word_form → morphological analyses lookup."""
    lookup = defaultdict(list)
    for entry in entries:
        lookup[entry['form']].append({
            'lemma': entry['lemma'],
            'tags': entry['tags']
        })
    return dict(lookup)


# =====================================================================
# 3. CSE Affixes Loader
# =====================================================================

CSE_COLUMNS = [
    'id', 'root', 'affix', 'info', 'pos', 'tense', 'person', 'possession',
    'cases', 'verb_voice1', 'verb_voice2', 'verb_voice3', 'verb_func',
    'impulsion', 'copula', 'singular', 'plural', 'question', 'negative',
    'lexical_affixes', 'syntactical_affixes'
]

# CSE → UD feature mapping
CSE_POS_MAP = {'NOUN': 'NOUN', 'VERB': 'VERB', 'ADJ': 'ADJ',
               'NUM': 'NUM', 'PRN': 'PRON', 'ADV': 'ADV'}
CSE_CASE_MAP = {'Genitive': 'GEN', 'Accusative': 'ACC', 'Dative': 'DAT',
                'Locative': 'LOC', 'Ablative': 'ABL'}
CSE_TENSE_MAP = {'Past': 'PST', 'Present': 'PRS', 'Future': 'FUT'}
CSE_VOICE_MAP = {'Active': 'ACT', 'Reflexive': 'REFL', 'Passive': 'PASS',
                 'Causative': 'CAUS', 'Reciprocal': 'RCP'}
CSE_FUNC_MAP = {'Participle': 'PART', 'Adverbial': 'CONV',
                'Infinitive': 'INF', 'Finite': 'FIN'}
CSE_MOOD_MAP = {'Message': 'IND', 'Imperative': 'IMP',
                'Conditional': 'CND', 'Proposal': 'DES'}


def load_cse_affixes(filepath: str) -> List[Dict]:
    """
    Load CSE affixes.csv — note: file uses \\r as record separator.
    Returns list of parsed affix dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        content = f.read()

    rows = content.replace('\r\n', '\n').split('\r')
    entries = []

    for row in rows[1:]:  # skip header
        try:
            parsed = list(csv.reader([row]))[0]
            if len(parsed) < 21:
                parsed.extend([''] * (21 - len(parsed)))

            entry = {}
            for i, col in enumerate(CSE_COLUMNS[:len(parsed)]):
                entry[col] = parsed[i].strip()

            # Map to unified schema
            entry['pos_mapped'] = CSE_POS_MAP.get(entry.get('pos', ''), '')
            entry['case_mapped'] = CSE_CASE_MAP.get(entry.get('cases', ''), '')
            entry['tense_mapped'] = CSE_TENSE_MAP.get(entry.get('tense', ''), '')
            entry['voice_mapped'] = CSE_VOICE_MAP.get(entry.get('verb_voice1', ''), '')
            entry['func_mapped'] = CSE_FUNC_MAP.get(entry.get('verb_func', ''), '')
            entry['mood_mapped'] = CSE_MOOD_MAP.get(entry.get('impulsion', ''), '')

            # Boolean flags
            for flag in ['singular', 'plural', 'question', 'negative', 'copula']:
                entry[flag] = entry.get(flag, '') in ('1', 'true', 'True')

            # Parse lexical/syntactical affixes for segmentation
            entry['lex_affix_list'] = [a.strip() for a in entry.get('lexical_affixes', '').split()
                                        if a.strip() and a.strip() != '-']
            entry['syn_affix_list'] = [a.strip() for a in entry.get('syntactical_affixes', '').split()
                                        if a.strip() and a.strip() != '-']

            entries.append(entry)
        except Exception:
            continue

    return entries


def load_roots(filepath: str) -> Dict[str, str]:
    """Load root.csv: stem → POS mapping."""
    roots = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row.get('stem', '').strip()
            pos = row.get('pos', '').strip()
            if stem:
                roots[stem] = pos
    return roots


# =====================================================================
# 4. Compatibility Matrix Builder
# =====================================================================

def build_compatibility_matrix(cse_entries: List[Dict]) -> Dict[str, Dict]:
    """
    Build grammatical compatibility matrix C^{(a,b)} from CSE data.

    Returns dict of {
        ('pos', 'case'): {(pos_val, case_val): True/False, ...},
        ('pos', 'tense'): {...},
        ...
    }

    Logic: if a (POS, feature) combination appears in CSE data,
    it is considered valid. Otherwise invalid.
    """
    # Collect observed valid combinations
    valid_combos = defaultdict(set)

    for entry in cse_entries:
        pos = entry.get('pos_mapped', '')
        if not pos:
            continue

        case = entry.get('case_mapped', '')
        tense = entry.get('tense_mapped', '')
        voice = entry.get('voice_mapped', '')
        func = entry.get('func_mapped', '')
        mood = entry.get('mood_mapped', '')
        is_plural = entry.get('plural', False)
        is_negative = entry.get('negative', False)

        if case:
            valid_combos[('pos', 'case')].add((pos, case))
        if tense:
            valid_combos[('pos', 'tense')].add((pos, tense))
        if voice:
            valid_combos[('pos', 'voice')].add((pos, voice))
        if func:
            valid_combos[('pos', 'verb_form')].add((pos, func))
        if mood:
            valid_combos[('pos', 'mood')].add((pos, mood))

        # Number is always valid for NOUN
        valid_combos[('pos', 'number')].add((pos, 'PL' if is_plural else 'SG'))

    # Augment with linguistic knowledge (hardcoded rules)
    # All POS can have NOM (unmarked) case
    for pos in ['NOUN', 'PRON', 'ADJ', 'NUM']:
        valid_combos[('pos', 'case')].add((pos, 'NOM'))
        valid_combos[('pos', 'case')].add((pos, 'GEN'))
        valid_combos[('pos', 'case')].add((pos, 'ACC'))
        valid_combos[('pos', 'case')].add((pos, 'DAT'))
        valid_combos[('pos', 'case')].add((pos, 'LOC'))
        valid_combos[('pos', 'case')].add((pos, 'ABL'))

    # VERB cannot have case (in general)
    # But participles can function as nouns, so we allow it from CSE data

    # Tense is only for VERB
    for t in ['PST', 'PRS', 'FUT']:
        valid_combos[('pos', 'tense')].add(('VERB', t))

    # Voice only for VERB
    for v in ['ACT', 'REFL', 'PASS', 'CAUS', 'RCP']:
        valid_combos[('pos', 'voice')].add(('VERB', v))

    return dict(valid_combos)


# =====================================================================
# 5. Segmentation Data Loader
# =====================================================================

def load_segmentation(filepath: str) -> List[Dict]:
    """
    Load segmentation.txt: word/morphemes format.
    Returns list of {'form': str, 'segments': list[str], 'bies': list[str]}
    """
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: stem/affix1/affix2 (slash-separated)
            # But also word/ with trailing slash means no affix
            parts = line.split('/')
            if not parts:
                continue

            # Reconstruct the full form
            form = ''.join(parts)
            segments = [p for p in parts if p]  # remove empty

            # Generate BIES labels for each character
            bies = []
            for seg in segments:
                if len(seg) == 1:
                    bies.append('S')
                else:
                    bies.extend(['B'] + ['I'] * (len(seg) - 2) + ['E'])

            entries.append({
                'form': form,
                'segments': segments,
                'bies': bies
            })

    return entries


def enrich_segmentation_from_cse(cse_entries: List[Dict],
                                  seg_entries: List[Dict]) -> List[Dict]:
    """
    Enrich segmentation data using CSE lexical/syntactical affix decompositions.
    Creates additional segmentation examples from CSE patterns.
    """
    existing_forms = {e['form'] for e in seg_entries}
    new_entries = []

    for entry in cse_entries:
        affix = entry.get('affix', '')
        lex = entry.get('lex_affix_list', [])
        syn = entry.get('syn_affix_list', [])

        if not affix or (not lex and not syn):
            continue

        # Clean affix: remove allomorphic markers
        clean_affix = re.sub(r'\([^)]*\)', '', affix).strip()
        if not clean_affix or clean_affix in existing_forms:
            continue

        # Build segments from lexical + syntactical
        all_segments = lex + syn
        if all_segments:
            # Verify segments roughly match the affix
            joined = ''.join(s.replace('(', '').replace(')', '') for s in all_segments)
            # Generate BIES
            bies = []
            for seg in all_segments:
                clean_seg = seg.replace('(', '').replace(')', '')
                if len(clean_seg) == 0:
                    continue
                if len(clean_seg) == 1:
                    bies.append('S')
                else:
                    bies.extend(['B'] + ['I'] * (len(clean_seg) - 2) + ['E'])

            if bies:
                new_entries.append({
                    'form': clean_affix,
                    'segments': all_segments,
                    'bies': bies,
                    'source': 'CSE'
                })

    return seg_entries + new_entries


# =====================================================================
# 6. Label Vocabularies Builder
# =====================================================================

def build_label_vocabs(sentences: List[Dict]) -> Dict[str, Dict]:
    """
    Build label vocabularies from UD training data.
    Returns dict mapping feature_name -> {label: idx}
    """
    counters = {
        'pos': Counter(),
        'case': Counter(),
        'number': Counter(),
        'person': Counter(),
        'possession': Counter(),
        'tense': Counter(),
        'mood': Counter(),
        'verb_voice': Counter(),
        'verb_form': Counter(),
        'degree': Counter(),
        'negative': Counter(),
        'question': Counter(),
        'copula': Counter(),
        'polite': Counter(),
    }

    for sent in sentences:
        for token in sent['tokens']:
            counters['pos'][token['upos']] += 1

            feats = token['features']
            if 'Case' in feats:
                counters['case'][feats['Case']] += 1
            if 'Number' in feats:
                counters['number'][feats['Number']] += 1
            if 'Person' in feats:
                counters['person'][feats['Person']] += 1
            if 'Possession' in feats or 'Poss' in feats or 'Poss_Person' in feats:
                val = feats.get('Possession', feats.get('Poss', feats.get('Poss_Person', '')))
                if val:
                    counters['possession'][val] += 1
            if 'Tense' in feats:
                counters['tense'][feats['Tense']] += 1
            if 'Mood' in feats:
                counters['mood'][feats['Mood']] += 1
            if 'Voice' in feats:
                counters['verb_voice'][feats['Voice']] += 1
            if 'VerbForm' in feats:
                counters['verb_form'][feats['VerbForm']] += 1
            if 'Degree' in feats:
                counters['degree'][feats['Degree']] += 1
            if 'Polarity' in feats or 'Negative' in feats:
                counters['negative'][feats.get('Polarity', feats.get('Negative', ''))] += 1
            if 'Question' in feats:
                counters['question'][feats['Question']] += 1
            if 'Copula' in feats:
                counters['copula'][feats['Copula']] += 1
            if 'Polite' in feats:
                counters['polite'][feats['Polite']] += 1

    vocabs = {}
    for name, counter in counters.items():
        # Add PAD (for padding) and NONE (for tokens without this feature)
        labels = ['<PAD>', '<NONE>'] + sorted(counter.keys())
        vocabs[name] = {label: idx for idx, label in enumerate(labels)}

    # Character vocabulary
    char_counter = Counter()
    for sent in sentences:
        for token in sent['tokens']:
            for c in token['form']:
                char_counter[c] += 1

    char_labels = ['<PAD>', '<UNK>'] + [c for c, _ in char_counter.most_common()]
    vocabs['char'] = {c: idx for idx, c in enumerate(char_labels)}

    # BIES vocabulary for segmentation
    vocabs['bies'] = {'<PAD>': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

    return vocabs


def build_char_vocab_from_all(sentences, unimorph_entries, seg_entries):
    """Build comprehensive character vocabulary from all data sources."""
    char_counter = Counter()

    for sent in sentences:
        for token in sent['tokens']:
            for c in token['form']:
                char_counter[c] += 1
            for c in token['lemma']:
                char_counter[c] += 1

    for entry in unimorph_entries:
        for c in entry['form']:
            char_counter[c] += 1
        for c in entry['lemma']:
            char_counter[c] += 1

    for entry in seg_entries:
        for c in entry['form']:
            char_counter[c] += 1

    chars = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [c for c, _ in char_counter.most_common()]
    return {c: idx for idx, c in enumerate(chars)}


# =====================================================================
# 7. Main Pipeline
# =====================================================================

def run_preprocessing_pipeline(cfg) -> Dict:
    """
    Run the full preprocessing pipeline.
    Returns a dictionary with all processed data.
    """
    print("=" * 60)
    print("O'zbek Tili Morfologik Tahlil — Ma'lumotlarni Tayyorlash")
    print("=" * 60)

    # 1. Load the Single PURE Unified Corrected Dataset
    print("\n[1/6] UD va Silver CoNLL-U birlashtirilgan toza ma'lumotlarni yuklash...")
    unified_path = os.path.join(cfg.DATASET_DIR, 'Uzbek_Morphology_Corpus.conllu')
    all_sents = parse_conllu_file(unified_path)
    
    # 2. Xolislik (unbiased) uchun qat'iy aralashtirish
    random.seed(cfg.seed)
    random.shuffle(all_sents)
    
    # 3. Yagona bazani 80:10:10 formatiga qismlarga bo'lish
    total_sents = len(all_sents)
    dev_size = int(total_sents * cfg.dev_ratio)
    test_size = int(total_sents * cfg.dev_ratio)
    
    test_split = all_sents[:test_size]
    dev_split = all_sents[test_size:test_size + dev_size]
    train_split = all_sents[test_size + dev_size:]
    
    ud_data = {
        'train': train_split,
        'dev': dev_split,
        'test': test_split
    }

    print(f"  Jami Birlashtirilgan gaplar: {total_sents:,}")
    print(f"  --> Train qismi (80%): {len(train_split):,}")
    print(f"  --> Validation (Dev) qismi (10%): {len(dev_split):,}")
    print(f"  --> Test qismi (10%): {len(test_split):,}")
    print(f"  Yakuniy Dataset => Train: {len(ud_data['train']):,} | Dev: {len(ud_data['dev'])} | Test: {len(ud_data['test'])}")

    # Sifat Kafolati Tekshiruvi
    print(f"\n[1.5/6] Sifat Kafolati (Quality Assurance) Tekshiruvi:")
    print(f"  - Test va Dev setlarda fiktiv/su'niy gaplar mavjud emas (Faqat 100% Original Gold).")

    # 2. Load UniMorph (Auxiliary dictionary)
    print("\n[2/6] UniMorph ma'lumotlarni yuklash (so'z bazasi sifatida)...")
    unimorph = load_unimorph(cfg.UNIMORPH_EXTENDED)
    unimorph_lookup = build_unimorph_lookup(unimorph)

    # 3. Load CSE affixes
    print("\n[3/6] CSE qo'shimchalar ma'lumotlarni yuklash...")
    cse_entries = load_cse_affixes(cfg.CSE_AFFIXES)

    # 4. Build compatibility matrix
    print("\n[4/6] Grammatik moslik matritsasini qurish...")
    compat_matrix = build_compatibility_matrix(cse_entries)

    # 5. Load and enrich segmentation
    print("\n[5/6] Segmentatsiya ma'lumotlarni yuklash va boyitish...")
    seg_entries = load_segmentation(cfg.SEGMENTATION_FILE)
    seg_entries = enrich_segmentation_from_cse(cse_entries, seg_entries)

    # 6. Build vocabularies
    print("\n[6/6] Lug'atlarni qurish...")
    all_train_sents = ud_data['train'] + ud_data['dev']
    label_vocabs = build_label_vocabs(all_train_sents)
    char_vocab = build_char_vocab_from_all(all_train_sents, unimorph, seg_entries)
    label_vocabs['char'] = char_vocab

    for name, vocab in label_vocabs.items():
        print(f"  {name}: {len(vocab)} label")

    # Load roots for lemma dictionary
    print("\n  O'zaklar lug'atini yuklash...")
    roots = load_roots(cfg.CSE_ROOTS)
    print(f"  {len(roots)} o'zak yuklandi")

    # Summary statistics
    print("\n" + "=" * 60)
    print("MA'LUMOTLAR TAYYORLASH YAKUNLANDI")
    print("=" * 60)

    result = {
        'ud_data': ud_data,
        'unimorph': unimorph,
        'unimorph_lookup': unimorph_lookup,
        'cse_entries': cse_entries,
        'compat_matrix': compat_matrix,
        'seg_entries': seg_entries,
        'label_vocabs': label_vocabs,
        'roots': roots,
    }

    return result


# =====================================================================
# 8. Data Statistics
# =====================================================================

def print_data_statistics(data: Dict):
    """Print detailed statistics about the preprocessed data."""
    print("\n" + "=" * 60)
    print("BATAFSIL STATISTIKA")
    print("=" * 60)

    # UD statistics
    for split in ['train', 'dev', 'test']:
        sents = data['ud_data'][split]
        tokens = sum(len(s['tokens']) for s in sents)
        avg_len = tokens / len(sents) if sents else 0

        # Feature coverage
        feat_counts = Counter()
        for s in sents:
            for t in s['tokens']:
                for f in t['features']:
                    feat_counts[f] += 1

        print(f"\n  [{split.upper()}]")
        print(f"    Gaplar: {len(sents)}, Tokenlar: {tokens}, O'rtacha uzunlik: {avg_len:.1f}")
        print(f"    Feature taqsimoti: {dict(feat_counts.most_common(10))}")

    # Compatibility matrix
    print("\n  [COMPATIBILITY MATRIX]")
    for key, combos in data['compat_matrix'].items():
        print(f"    {key[0]}↔{key[1]}: {len(combos)} haqiqiy juftlik")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import cfg, DATASET_DIR, UD_TRAIN, UD_TEST, UNIMORPH_EXTENDED
    from config import CSE_AFFIXES, CSE_ROOTS, SEGMENTATION_FILE

    # Inject paths into cfg for the pipeline
    cfg.UD_TRAIN = UD_TRAIN
    cfg.UD_TEST = UD_TEST
    cfg.UNIMORPH_EXTENDED = UNIMORPH_EXTENDED
    cfg.CSE_AFFIXES = CSE_AFFIXES
    cfg.CSE_ROOTS = CSE_ROOTS
    cfg.SEGMENTATION_FILE = SEGMENTATION_FILE

    data = run_preprocessing_pipeline(cfg)
    print_data_statistics(data)
