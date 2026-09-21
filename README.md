# UzMorph-Transformers: Transformer-based Uzbek Neural Morphological Analyzer

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status: State-of-the-art](https://img.shields.io/badge/Status-State--of--the--art-orange)

**UzMorph-Transformers** is a state-of-the-art, multi-task deep learning framework for the morphological analysis of the highly agglutinative Uzbek language. 

Traditional dictionary-based and finite-state parsers struggle with extreme Out-of-Vocabulary (OOV) rates and contextual homonymy (e.g., distinguishing *ot* as a 'horse' vs 'to throw'). This repository implements a novel neural approach combining a **1D-CNN Character Encoder**, a **Contextual Transformer**, and an **Adaptive Context-Morphology Gating Mechanism** to explicitly resolve contextual ambiguity dynamically.

## Key Features
- **Total Morphological Coverage:** Simultaneously predicts 14 grammatical attributes (POS, Case, Number, Person, Tense, Mood, VerbForm, Voice, Possessive, Degree, Negative, Question, Copula, Politeness) alongside morphological **Lemmatization**.
- **Adaptive Gating ($g_i$):** A mathematical dynamic gate that autonomously routes processing between internal sub-word features (CNN) and global sentence semantics (Transformer).
- **Rule-based Penalties:** Integrated Grammatical Compatibility Matrix that restricts linguistically impossible structural outputs during neural generation.
- **High Performance:** Achieves **+9% Exact Match** jump over traditional deterministic rule-based algorithms. **(88.39% Dev Accuracy, 75.01% Test Accuracy)**.

## Model Architecture
Our framework maps agglutinative structures via three core components:
1. **Character sub-word embeddings (1D-CNN):** Operates on kernel lengths of {3, 5, 7} for deep sub-word inflection capture.
2. **Global sentence tracking (Transformer):** 3-layer Multi-Head Self-Attention capturing contextual syntax bounds.
3. **Adaptive Synthesis:** Resolves overlapping ambiguous homonyms perfectly depending on sentence location metrics.

## Dataset
The model was natively trained on a unified morphological dataset spanning **17,838 sentences**. It consolidates three major Universal Dependencies (UD) Treebanks specifically mapped into a strict 14-dimensional attribute space. It's heavily augmented using the Complete Set of Endings (CSE) logic for robust rare semantic captures without sacrificing pure gold-only test validity.

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch (GPU recommended for rapid inference)

Since this model trains deep contextual parameters (~117MB), reviewers must initially run the structural training pipeline to generate the necessary model weights recursively against the datasets.

```bash
# 1. Clone the repository
git clone https://github.com/UlugbekSalaev/UzMorph-Transformers.git
cd UzMorph-Transformers

# 2. Install core package dependencies
pip install -r requirements.txt

# 3. Trigger native model training (Takes ~30 mins on GPU)
python train.py

# 4. Run interactive analyzer
python demo.py
```

## Citation
If you utilize this repository for academic or commercial research, please cite our official MDPI publication:

```bibtex
@article{UzMorphNeural2026,
  title={Joint Neural Morphological Analysis for the Uzbek Language: A Multi-Task Approach with Adaptive Context-Morphology Gating},
  author={Your Name},
  journal={MDPI Electronics},
  year={2026},
  publisher={MDPI}
}
```

## Contribution
Contributions are welcome! Please open an issue or submit a Pull Request enforcing identical Universal Dependencies (UD) morphotactic mappings.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
