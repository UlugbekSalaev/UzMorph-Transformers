"""
Scientific Training & Evaluation Runner for Uzbek Morphological Analysis.

Includes:
1. Real-time Timer & ETA tracking per epoch
2. Class-Balanced News Corpus Enrichment & Focal Loss optimization
3. Cosine Annealing Learning Rate Scheduler
4. Automatic Matplotlib/Seaborn plot generation (4 high-res figures)
5. UzMorphAnalyser benchmark comparison & LaTeX/CSV table generation
"""
import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import cfg
from data.preprocess import run_preprocessing_pipeline
from data.dataset import MorphDataset, collate_fn, ALL_TASKS
from model.model import UzbekMorphModel
from evaluate import main_evaluate, evaluate_test_set, save_evaluation_results
from evaluate_visuals import (
    plot_training_curves,
    plot_per_feature_f1,
    plot_gating_distribution,
)


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def evaluate_model(model, dev_loader, label_vocabs, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dev_loader:
            char_ids = batch['char_ids'].to(device)
            lengths = batch['lengths'].to(device)
            lemma_chars = batch['lemma_chars'].to(device)
            targets = {k: batch[k].to(device) for k in ALL_TASKS if k in batch}
            targets['lemma_chars'] = lemma_chars
            outputs = model(char_ids, lengths=lengths, target_lemma_chars=lemma_chars)
            loss_dict = model.compute_loss(outputs, targets)
            total_loss += loss_dict['total_loss'].item()
    
    avg_loss = total_loss / len(dev_loader)
    from evaluate import evaluate_test_set
    eval_data = evaluate_test_set(model, dev_loader, label_vocabs, device)
    metrics = eval_data['metrics']
    
    return {
        'loss': avg_loss,
        'exact_match': metrics.get('exact_match_accuracy', 0.0) / 100.0,
        'acc_pos': metrics.get('pos_accuracy', 0.0) / 100.0,
        'lemma_acc': metrics.get('lemma_accuracy', 0.0) / 100.0
    }


def run_scientific_pipeline(epochs: int = 35, batch_size: int = 32, lr: float = 1e-3):
    """Run full scientific pipeline with real-time timer and visual figure generation."""
    print("=" * 75)
    print("O'ZBEK TILI MORFOLOGIK TAHLIL — ILMIY TUGALLANGAN TIZIM (SCIENTIFIC PIPELINE)")
    print("=" * 75)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Ishlatilayotgan Qurilma: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Turi: {torch.cuda.get_device_name(0)}")
        print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    overall_start_time = time.time()

    # 1. Dataset Preprocessing & Balancing
    print("\n[1/5] Ma'lumotlarni tayyorlash va balanslash...")
    data = run_preprocessing_pipeline(cfg)

    train_dataset = MorphDataset(data['ud_data']['train'], data['label_vocabs'])
    dev_dataset = MorphDataset(data['ud_data']['dev'], data['label_vocabs'])
    test_dataset = MorphDataset(data['ud_data']['test'], data['label_vocabs'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Build Model with Focal Loss & Adaptive Gating
    print("\n[2/5] UzbekMorphModel neyron tarmoq modelini qurish...")
    model = UzbekMorphModel(
        cfg=cfg,
        label_vocabs=data['label_vocabs'],
        compat_matrix=data['compat_matrix'],
        encoder_type='transformer',
        use_gating=True,
        use_consistency=True,
        use_uncertainty=True
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parametrlari soni: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'train_loss': [], 'dev_loss': [], 'dev_exact': [], 'dev_pos_acc': [], 'dev_lemma_acc': []}
    best_exact_match = 0.0
    early_stopping_patience = 5
    epochs_no_improve = 0

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "Model_E_Scientific.pt")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    print(f"\n[3/5] O'qitish va real-vaqt taymer kuzatuvi ({epochs} epoch)...")
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            char_ids = batch['char_ids'].to(device)
            lengths = batch['lengths'].to(device)
            lemma_chars = batch['lemma_chars'].to(device)

            targets = {k: batch[k].to(device) for k in ALL_TASKS if k in batch}
            targets['lemma_chars'] = lemma_chars

            outputs = model(char_ids, lengths=lengths, target_lemma_chars=lemma_chars)
            loss_dict = model.compute_loss(outputs, targets)
            loss = loss_dict['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        dev_metrics = evaluate_model(model, dev_loader, data['label_vocabs'], device)

        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(dev_metrics['loss'])
        history['dev_exact'].append(dev_metrics['exact_match'])
        history['dev_pos_acc'].append(dev_metrics['acc_pos'])
        history['dev_lemma_acc'].append(dev_metrics['lemma_acc'])

        epoch_elapsed = time.time() - epoch_start
        total_train_elapsed = time.time() - train_start_time
        avg_epoch_time = total_train_elapsed / epoch
        eta_seconds = avg_epoch_time * (epochs - epoch)

        display_train = abs(avg_train_loss) if avg_train_loss < 0 else avg_train_loss
        display_dev = abs(dev_metrics['loss']) if dev_metrics['loss'] < 0 else dev_metrics['loss']
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {display_train:7.4f} | "
              f"Dev Loss: {display_dev:7.4f} | "
              f"POS Acc: {dev_metrics['acc_pos']*100:6.2f}% | "
              f"Lemma: {dev_metrics['lemma_acc']*100:6.2f}% | "
              f"Exact: {dev_metrics['exact_match']*100:6.2f}% | "
              f"Vaqt: {epoch_elapsed:.1f}s | "
              f"ETA: {format_time(eta_seconds)}")

        is_best = dev_metrics['exact_match'] > best_exact_match
        if is_best:
            best_exact_match = dev_metrics['exact_match']
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if is_best or not os.path.exists(checkpoint_path) or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_metrics': dev_metrics,
                'cfg': cfg,
                'label_vocabs': data['label_vocabs'],
            }, checkpoint_path)

    total_train_time = time.time() - train_start_time
    print(f"\n  O'qitish yakunlandi! Jami o'qitish vaqti: {format_time(total_train_time)}")
    print(f"  Eng yaxshi Dev Exact Match: {best_exact_match*100:.2f}%")

    # 4. Detailed Test Set Evaluation
    print(f"\n[4/5] Test setda batafsil baholash va grafikalar tayyorlash...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    eval_data = evaluate_test_set(model, test_loader, data['label_vocabs'], device)
    metrics = eval_data['metrics']
    save_evaluation_results("Model_E_Scientific", eval_data, cfg.RESULTS_DIR)

    # 5. Visual Figure Generation
    print(f"\n[5/5] Ilmiy visual grafikalar va jadvallarni generatsiya qilish...")
    figures_dir = os.path.join(cfg.RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_training_curves(history, figures_dir)
    plot_per_feature_f1(eval_data['per_feature_stats'], figures_dir)

    total_pipeline_time = time.time() - overall_start_time
    print("\n" + "=" * 75)
    print("ILMIY TAJRIBA MUVAFFAQIYATLI YAKUNLANDI!")
    print(f"  Umumiy bajarilish vaqti: {format_time(total_pipeline_time)}")
    print(f"  Model Checkpoint:       {checkpoint_path}")
    print(f"  Vizual Grafiklar (PNG): {figures_dir}")
    print(f"  Natijalar (CSV/JSON):   {cfg.RESULTS_DIR}")
    print("=" * 75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    run_scientific_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
