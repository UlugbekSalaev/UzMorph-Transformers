"""
Configuration for Uzbek Morphological Analysis Model.
"""
import os

# ===================== Paths =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# UD data
UD_DIR = os.path.join(DATASET_DIR, "UD")
UD_TRAIN = os.path.join(UD_DIR, "train_merged.conllu")
UD_TEST = os.path.join(UD_DIR, "uz_uzudt-ud-test.conllu")

# UniMorph
UNIMORPH_DIR = os.path.join(DATASET_DIR, "UniMorph")
UNIMORPH_EXTENDED = os.path.join(UNIMORPH_DIR, "unimorph_uzb_extended", "uzbek_unified.txt")
UNIMORPH_AFFIXES_JSON = os.path.join(UNIMORPH_DIR, "unimorph_uzb_extended", "uzbek_affixes_dictionary.json")

# UzMorphAnalyser (CSE)
UZMORPH_DIR = os.path.join(DATASET_DIR, "UzMorphAnalyser")
CSE_AFFIXES = os.path.join(UZMORPH_DIR, "affixes.csv")
CSE_ROOTS = os.path.join(UZMORPH_DIR, "root.csv")

# Segmentation
SEGMENTATION_FILE = os.path.join(DATASET_DIR, "segmentation.txt")

# Output
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ===================== Model =====================
class ModelConfig:
    # Paths
    BASE_DIR = BASE_DIR
    DATASET_DIR = DATASET_DIR
    UD_DIR = UD_DIR
    UD_TRAIN = UD_TRAIN
    UD_TEST = UD_TEST
    UNIMORPH_DIR = UNIMORPH_DIR
    UNIMORPH_EXTENDED = UNIMORPH_EXTENDED
    UNIMORPH_AFFIXES_JSON = UNIMORPH_AFFIXES_JSON
    UZMORPH_DIR = UZMORPH_DIR
    CSE_AFFIXES = CSE_AFFIXES
    CSE_ROOTS = CSE_ROOTS
    SEGMENTATION_FILE = SEGMENTATION_FILE
    DATA_OUTPUT_DIR = DATA_OUTPUT_DIR
    RESULTS_DIR = RESULTS_DIR
    CHECKPOINT_DIR = CHECKPOINT_DIR
    TABLES_DIR = TABLES_DIR
    FIGURES_DIR = FIGURES_DIR
    # Character encoder
    char_embed_dim = 64
    char_vocab_size = 128  # ASCII + Uzbek chars
    cnn_filters = 128
    cnn_kernels = [3, 5, 7]  # Multi-width CNN

    # Token encoder
    token_embed_dim = 128
    combined_dim = token_embed_dim + cnn_filters * len(cnn_kernels)  # 128 + 384

    # Contextual encoder
    d_model = 256
    n_heads = 4
    d_ff = 512
    n_layers = 3
    dropout = 0.3

    # Training
    lr = 1e-3
    batch_size = 32
    max_epochs = 50
    patience = 5
    grad_clip = 1.0
    dev_ratio = 0.1  # 10% for dev, 10% for test (80-10-10 Gold split)
    seed = 42

    # Consistency
    alpha_cons = 0.1  # Consistency loss weight
    beta_align = 0.05  # Morpheme alignment loss weight

    # Label spaces (will be populated from data)
    pos_labels = []
    case_labels = []
    number_labels = []
    person_labels = []
    tense_labels = []
    poss_labels = []
    voice_labels = []
    mood_labels = []
    verb_form_labels = []

cfg = ModelConfig()
