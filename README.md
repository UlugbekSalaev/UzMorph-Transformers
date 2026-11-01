# UzMorph-Transformer

**Adaptive Context–Morphology Gating and Grammatical Constraints for Neural Morphological Analysis of Uzbek**

O. Yusupov, U. Salaev, A. Babayarov, Sh. Kholikulov (manuscript under review)

This repository contains the Gold and Silver data, the five experimental splits, and the code needed to train the model, run the ablations, and recompute the analyses reported in the manuscript.

The model jointly predicts morphological attributes and lemmas for Uzbek tokens. It shares a character-level CNN between a Transformer contextual branch and a morphology-oriented branch, fuses them with an adaptive gate, and adds a grammatical compatibility regularizer during training.

## Contents at a glance

| Item | Location |
|---|---|
| Source corpus, before labeling | `Dataset/Uzbek_Morphology_Corpus.conllu` |
| Labeled corpus | `Dataset/Uzbek_Unified_Morphological_Dataset.conllu` |
| Gold partitions for seeds 17, 29, 43, 71, 97 | `Dataset/splits/seed_<seed>/gold_{train,dev,test}.conllu` |
| Silver pretraining data | `Dataset/splits/silver_train.conllu` |
| Silver labeling script | `data/unified_builder.py` |
| Splitting script | `data/scientific_data.py` |
| Model | `model/` |
| Training and evaluation | `run_ai_experiments.py`, `evaluate_ai_ready.py` |
| Statistics and gate analysis | `analysis/` |
| Checks of the data statements and of the invalid-combination metric | `verification/` |

## Setup

```bash
git clone https://github.com/UlugbekSalaev/UzMorph-Transformers.git
cd UzMorph-Transformers
pip install -r requirements.txt
```

## Check the data

```bash
python verification/verify_data.py
python verification/verify_icr.py
```

Each line compares a recomputed value with the value stated in the manuscript and prints `OK` or `DIFF`. `verify_data.py` also regenerates all five splits and compares them with the files in `Dataset/splits`.

## Reproduce the experiments

```bash
# Curriculum training: full model and ablations
python run_ai_experiments.py --seeds 17 29 43 71 97 --variants full context morph concat no_consistency --regime curriculum --pretrain_epochs 5 --epochs 35

# Gold-only training, for the comparison with curriculum training
python run_ai_experiments.py --seeds 17 29 43 71 97 --variants full --regime gold --epochs 35

# Means, standard deviations and paired tests
python analysis/paired_tests.py
python analysis/paired_tests.py --metric invalid_combination_rate

# Gate analysis
python analysis/gate_analysis.py --regime curriculum
```

This is 30 training runs. A curriculum run takes roughly 1–2 hours on an 8-core CPU, and a GPU is used automatically when available. Each run writes `metrics.json`, `history.json` and a checkpoint to `results/ai_ready/<regime>/<variant>/seed_<seed>/`.

### Results reported in the manuscript

Test results of the full model, mean ± SD over the five seeds:

| Metric | Value |
|---|---:|
| Strict Bundle Exact Match | 58.53 ± 1.81 |
| Active-Feature Exact Match | 79.86 ± 0.98 |
| Lemma Exact Match | 60.12 ± 1.77 |
| Gold-training-OOV Strict Exact Match | 28.14 ± 1.71 |
| Gold-training-seen Strict Exact Match | 71.57 ± 2.07 |
| Invalid Combination Rate | 3.62 ± 0.63 |

Strict Bundle Exact Match of the comparison models:

| Context only | Morphology only | Concatenation | No regularizer | Gold-only training |
|---:|---:|---:|---:|---:|
| 55.67 | 55.71 | 54.62 | 56.57 | 52.87 |

Exact values can vary slightly with hardware and PyTorch version.

### Where each result comes from

| Manuscript | Command | Output |
|---|---|---|
| Sections 4.1–4.3: data, Silver cascade, splits | `verification/verify_data.py` | console |
| Sections 5.1, 5.2, 5.4, 5.5, 5.6: metrics and ablations | `run_ai_experiments.py` | `metrics.json`, `summary_mean_std.csv` |
| Sections 5.3, 5.4, 5.5: paired statistics | `analysis/paired_tests.py` | console |
| Section 5.4: verification of the invalid-combination metric | `verification/verify_icr.py` | console |
| Section 5.7: gate analysis | `analysis/gate_analysis.py` | console |

## Data

| Subset | Instances | Unique sentences | Tokens | Use |
|---|---:|---:|---:|---|
| Gold | 2,859 | 1,098 | 8,986 | training, development, test |
| Silver | 14,978 | 14,863 | 112,693 | pretraining only |

No sentence occurs in both subsets, and Silver data are never used for model selection or evaluation.

### Gold

- **Treebank-derived sentences:** 680 sentences (7,534 tokens) derive from the UzUDT treebank. Of these, 542 match the current public release exactly, and 138 differ only in spelling or tokenization. The corpus uses its own lemma forms and feature names, so its labels are not identical to that release (agreement on the exact matches: 99.47% UPOS, 68.30% lemma).
- **Template sentences:** 418 short sentences (1,452 tokens) were generated from four UPOS patterns and a closed vocabulary of 47 lemmas to increase the coverage of inflected forms. They were not manually annotated or curated for semantic plausibility. They are listed in `verification/external/template_sentences.conllu` and make up 25–30 of the 110 sentences in each test set.

### Silver

The Silver sentences (news text and short general-language sentences) were originally unannotated. `data/unified_builder.py` labels each token with this cascade:

1. Numeric tokens receive `NUM` with `NumType=Card` (0.76% of tokens).
2. A lowercase form found in a memory compiled from the annotated Gold instances copies its lemma, UPOS and features (38.00%).
3. Otherwise, the longest prefix found in the UzMorphAnalyser root lexicon gives the lemma and UPOS, with no features (58.50%).
4. Otherwise, the token is labeled `PROPN` if title-cased and `NOUN` if not, with its lowercase form as lemma (2.74%).

Silver labels were not manually corrected. On held-out Gold tokens across the five splits, the cascade reaches 77.95 ± 0.97% for UPOS, 63.02 ± 1.14% for lemmas and 58.98 ± 0.92% for complete feature strings.

### Splits

The unique Gold sentences are divided into 878 training, 110 development and 110 test sentences by a rare-label-first stratified procedure. All variants of a seed use the same partition, so comparisons are paired. In all five splits, the partitions share no sentence and no development or test label is absent from Gold training data.

### Label format

The files follow the CoNLL-U layout. In `Dataset/splits`, `CCONJ` and `SCONJ` appear as `CONJ`, and `Number[psor]` and `Person[psor]` appear as `Poss_Number` and `Poss_Person`.

## Notes for interpreting the results

- **Word-level overlap:** the memory in step 2 of the Silver cascade was compiled before partitioning, so Gold and Silver are separated at sentence level, not at word level. Across the splits, 1.77–2.18% of Silver tokens take their labels from word forms found only in held-out Gold partitions, and these forms cover 33.0–39.4% of Gold-training-OOV test tokens. Gold-training-OOV scores should be read with this in mind.
- **Feature coverage:** the model has output heads for 14 attributes, but Degree, Question, Copula and Politeness have no sufficient positive support in the Gold data.
- **Invalid-combination metric:** the per-token predictions of the original runs were not stored. `verification/predictions` therefore holds predictions from a separate five-fold evaluation of the same architecture on the Uzbek-UT and Uzbek-TueCL treebanks, with a differently annotated Silver set.
- **Leipzig overlap:** the share of Silver sentences found in the Leipzig Corpora Collection (31.2%) is not recomputed here, because that file is not redistributed in this repository.

## Repository layout

```text
Dataset/                 corpus, splits and auxiliary lexical resources
data/                    data loading, Silver labeling, splitting
model/                   encoders, gate, heads, losses, regularizer
analysis/                paired statistics and gate analysis
verification/            checks of the manuscript's data and metric statements
run_ai_experiments.py    training and ablations
evaluate_ai_ready.py     evaluation metrics
config.py                model and path settings
```

## Data sources and licenses

- **Code:** MIT License (`LICENSE`).
- **UD treebanks:** the UzUDT-derived Gold data and the Uzbek-UT, Uzbek-TueCL and UzUDT files in `verification/external` are redistributed under CC BY-SA 4.0, the license of the treebanks.
- **UzMorphAnalyser:** root and affix resources in `Dataset/UzMorphAnalyser`, from https://github.com/UlugbekSalaev/UzMorphAnalyser.
- **UniMorph:** the auxiliary lexicon in `Dataset/UniMorph` follows the terms of its source. It contributes to the character vocabulary only and creates no evaluation labels.
- **Silver text:** rights to the original texts remain with their sources. The sentences are provided so that the experiments can be reproduced.

## Citation

The citation will be added after publication. Until then, please cite the repository and the manuscript title above.

## Questions

Please open a GitHub issue.
