"""
PyTorch Dataset classes for Uzbek Morphological Analysis (14 Attributes + Lemma + Segmentation).
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


ALL_TASKS = [
    'pos', 'case', 'number', 'person', 'possession',
    'tense', 'mood', 'verb_voice', 'verb_form', 'degree',
    'negative', 'question', 'copula', 'polite'
]


class MorphDataset(Dataset):
    """
    PyTorch Dataset for sentence-level morphological analysis from UD data.
    Supports all 14 morphological attributes + Lemma + Segmentation.
    """

    def __init__(self, sentences: List[Dict], vocabs: Dict[str, Dict],
                 max_word_len: int = 50, max_sent_len: int = 128):
        self.sentences = sentences
        self.vocabs = vocabs
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

        self.char_vocab = vocabs['char']
        self.unk_char = self.char_vocab.get('<UNK>', 1)
        self.pad_char = self.char_vocab.get('<PAD>', 0)

        # Mapping for features
        self.task_feature_map = {
            'pos': 'upos',
            'case': 'Case',
            'number': 'Number',
            'person': 'Person',
            'possession': 'Possession',
            'tense': 'Tense',
            'mood': 'Mood',
            'verb_voice': 'Voice',
            'verb_form': 'VerbForm',
            'degree': 'Degree',
            'negative': 'Polarity',
            'question': 'Question',
            'copula': 'Copula',
            'polite': 'Polite'
        }

    def __len__(self):
        return len(self.sentences)

    def _char_encode(self, word: str) -> List[int]:
        """Convert word to character IDs."""
        ids = []
        for c in word[:self.max_word_len]:
            ids.append(self.char_vocab.get(c, self.unk_char))
        return ids

    def _get_label(self, token: Dict, task: str) -> int:
        vocab = self.vocabs.get(task, {})
        if not vocab:
            return 1  # <NONE>

        if task == 'pos':
            val = token.get('upos', '')
        else:
            feat_key = self.task_feature_map.get(task, task)
            feats = token.get('features', {})
            val = feats.get(feat_key, '')

        if val and val in vocab:
            return vocab[val]
        return vocab.get('<NONE>', 1)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = sent['tokens'][:self.max_sent_len]

        char_ids_list = []
        forms = []
        task_labels = {task: [] for task in ALL_TASKS}
        lemma_chars_list = []

        for token in tokens:
            # Character encoding
            char_ids = self._char_encode(token['form'])
            char_ids_list.append(char_ids)
            forms.append(token['form'])

            # Labels for all 14 tasks
            for task in ALL_TASKS:
                task_labels[task].append(self._get_label(token, task))

            # Lemma character IDs
            lemma_chars = self._char_encode(token['lemma'])
            lemma_chars_list.append(lemma_chars)

        return {
            'char_ids': char_ids_list,
            'forms': forms,
            'lemma_chars': lemma_chars_list,
            'length': len(tokens),
            **task_labels
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that pads sentences and words to uniform length.
    """
    max_sent_len = max(item['length'] for item in batch)
    max_word_len = 1
    for item in batch:
        for char_ids in item['char_ids']:
            max_word_len = max(max_word_len, len(char_ids))

    batch_size = len(batch)

    char_ids_tensor = torch.zeros(batch_size, max_sent_len, max_word_len, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    word_lengths = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

    labels = {k: torch.zeros(batch_size, max_sent_len, dtype=torch.long) for k in ALL_TASKS}

    max_lemma_len = 1
    for item in batch:
        for lc in item['lemma_chars']:
            max_lemma_len = max(max_lemma_len, len(lc))

    lemma_chars_tensor = torch.zeros(batch_size, max_sent_len, max_lemma_len, dtype=torch.long)

    for i, item in enumerate(batch):
        sent_len = item['length']
        lengths[i] = sent_len

        for j in range(sent_len):
            cids = item['char_ids'][j]
            char_ids_tensor[i, j, :len(cids)] = torch.tensor(cids)
            word_lengths[i, j] = len(cids)

            for k in ALL_TASKS:
                labels[k][i, j] = item[k][j]

            lc = item['lemma_chars'][j]
            lemma_chars_tensor[i, j, :len(lc)] = torch.tensor(lc)

    return {
        'char_ids': char_ids_tensor,        # (B, S, W)
        'lengths': lengths,                  # (B,)
        'word_lengths': word_lengths,        # (B, S)
        'lemma_chars': lemma_chars_tensor,   # (B, S, L)
        **labels,                            # each (B, S)
    }


class SegmentationDataset(Dataset):
    """Dataset for morpheme segmentation (word-level, BIES labels)."""

    def __init__(self, entries: List[Dict], char_vocab: Dict, bies_vocab: Dict,
                 max_len: int = 50):
        self.entries = entries
        self.char_vocab = char_vocab
        self.bies_vocab = bies_vocab
        self.max_len = max_len
        self.unk_char = char_vocab.get('<UNK>', 1)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        form = entry['form'][:self.max_len]
        bies = entry['bies'][:self.max_len]

        char_ids = [self.char_vocab.get(c, self.unk_char) for c in form]
        bies_ids = [self.bies_vocab.get(b, 0) for b in bies]

        return {
            'char_ids': char_ids,
            'bies_labels': bies_ids,
            'length': len(char_ids)
        }


def seg_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for segmentation dataset."""
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)

    char_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    bies_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        l = item['length']
        lengths[i] = l
        char_ids[i, :l] = torch.tensor(item['char_ids'])
        bies_labels[i, :l] = torch.tensor(item['bies_labels'])

    return {
        'char_ids': char_ids,
        'bies_labels': bies_labels,
        'lengths': lengths,
    }
