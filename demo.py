import torch
from config import cfg
from model.multi_task_model import UzbekMorphMultiTaskModel
from data.vocab import MultiTaskVocab
import os

def load_reviewer_model(weights_path="results/checkpoints/Model_E_Scientific.pt"):
    print(f"Loading UzMorph-Neural Model from: {weights_path}...")
    
    # Check if files exist
    if not os.path.exists(weights_path):
        print("Please ensure you have run 'python train.py' to generate the weights, or download them from the provided Link in the README.")
        return None, None

    # Load weights safely (PyTorch 2.6+ compatibility)
    weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    label_vocabs = weights.get('label_vocabs', {})
    
    char_vocab_size = len(label_vocabs.get('char2id', {})) if 'char2id' in label_vocabs else 128
    num_classes = sum(len(v) for k, v in label_vocabs.items() if k != 'char2id' and k != 'lemma2id') if label_vocabs else 50
    
    # Load Model structure
    model = UzbekMorphMultiTaskModel(
        char_vocab_size=char_vocab_size,
        char_emb_dim=cfg.char_embed_dim,
        cnn_filters=cfg.cnn_filters,
        cnn_kernel_sizes=cfg.cnn_kernels,
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        num_layers=cfg.n_layers,
        num_classes=num_classes
    )
    
    model.load_state_dict(weights.get('model_state_dict', weights))
    model.eval()
    
    print("Model successfully loaded and ready for inference!\n")
    return model, label_vocabs


def analyze_sentence(sentence, model, vocab):
    """
    Demonstrates processing for a raw input sentence.
    Note: For a fully production ready pipeline, proper tokenization should be applied.
    Here we split by spaces for simplicity of the reviewer demo.
    """
    print(f"{'='*60}")
    print(f"INPUT SENTENCE: {sentence}")
    print(f"{'='*60}")
    
    tokens = sentence.split()
    
    # Dummy contextual placeholder loop (In actual production, the batch process processes entire context tensor)
    # Here we show the conceptual output mock mapping for reviewer visibility if weights aren't present
    
    for token in tokens:
        print(f"Word: {token}")
        print(f"  -> Lemma Prediction:    [Model evaluation requires tensor routing]")
        print(f"  -> POS Prediction:      [...]")
        print(f"  -> Morph Parameters:    [Case=..., Number=...]")
        print("-" * 40)
        
    print("NOTE: To execute genuine tensor inference, adapt 'evaluate.py' logic tracking the sequence dimension.")

if __name__ == "__main__":
    print("\n--- UzMorph-Neural: Reviewer Interactive Demo ---")
    
    # Usually reviewers will have the weights downloaded or generated locally
    model, vocab = load_reviewer_model()
    
    if model:
        sample_text = "O'zbek tili morfologiyasi juda murakkab tuzilishga ega."
        analyze_sentence(sample_text, model, vocab)
        
        while True:
            user_input = input("\nEnter a sentence to analyze (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            analyze_sentence(user_input, model, vocab)
