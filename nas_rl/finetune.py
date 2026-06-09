"""
finetune.py — Fine-tune the best found architecture on full CIFAR-10.

The standalone model is exported with supernet weights transferred.
Those weights are a warm start but were never trained end-to-end on
this exact architecture.  Fine-tuning for 50-100 epochs on the full
training set typically pushes accuracy from ~55-60% to 85-92%.

Usage
-----
    python -m nas_rl.finetune

Results are saved to exports/finetuned_model.pt and evaluated on
the held-out test set.
"""

import os
import sys
import json
import torch
import torch.nn as nn

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nas_rl.data import get_cifar10_loader
from nas_rl.export.model_builder import StandaloneModel, transfer_weights
from nas_rl.search_space.supernet import Supernet


# ── Config ────────────────────────────────────────────────────────────────────

SUPERNET_CKPT  = 'checkpoints/supernet.pt'
SEARCH_RESULTS = 'exports/search_results.json'
OUTPUT_PT      = 'exports/finetuned_model.pt'

C_INIT    = 16
N_CELLS   = 8
N_NODES   = 4
N_CLASSES = 10

EPOCHS       = 100
BATCH_SIZE   = 128
LR           = 0.025
WEIGHT_DECAY = 3e-4
GRAD_CLIP    = 5.0


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Fine-tune loop ────────────────────────────────────────────────────────────

def finetune():
    device = get_device()
    print(f"Device: {device}")

    # Load best arch_spec from search results
    with open(SEARCH_RESULTS) as f:
        results = json.load(f)

    # Load the search checkpoint to get the actual arch_spec
    search_ckpt = torch.load('checkpoints/search.pt', map_location='cpu')
    arch_pool   = search_ckpt['arch_pool']
    best_entry  = max(arch_pool, key=lambda e: e['reward'])
    arch_spec   = best_entry['arch']

    print(f"Best arch — val acc: {best_entry['accuracy']*100:.2f}%  "
          f"FLOPs: {best_entry['flops']/1e6:.1f}M  "
          f"reward: {best_entry['reward']:.4f}")

    # Build standalone model
    model = StandaloneModel(
        arch_spec=arch_spec, C_init=C_INIT,
        n_cells=N_CELLS, n_nodes=N_NODES, n_classes=N_CLASSES,
    ).to(device)

    # Transfer supernet weights as warm start
    supernet = Supernet(C_init=C_INIT, n_cells=N_CELLS,
                        n_nodes=N_NODES, n_classes=N_CLASSES)
    supernet.load_state_dict(
        torch.load(SUPERNET_CKPT, map_location='cpu')
    )
    print("Transferring supernet weights...")
    transfer_weights(supernet, model, arch_spec)
    del supernet   # free memory

    # Data
    train_loader = get_cifar10_loader('train', batch_size=BATCH_SIZE, num_workers=2)
    val_loader   = get_cifar10_loader('val',   batch_size=256, num_workers=2)
    test_loader  = get_cifar10_loader('test',  batch_size=256, num_workers=2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}  ({total_params/1e6:.3f}M)")

    # Optimiser — same setup as supernet pretraining
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR,
        momentum=0.9, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4,
    )

    print(f"\nFine-tuning for {EPOCHS} epochs")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValAcc':>8}  {'LR':>10}")
    print("-" * 52)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        total_loss = total_correct = total_samples = 0

        for x, y in train_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # ── validate ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += y.size(0)
        val_acc = val_correct / val_total

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"{epoch+1:6d}  {train_loss:10.4f}  {train_acc*100:8.2f}%  "
              f"{val_acc*100:7.2f}%  {lr:10.6f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PT)

    # ── Final test set evaluation ──
    print(f"\nBest val accuracy during fine-tuning: {best_val_acc*100:.2f}%")
    print("Loading best checkpoint for test evaluation...")

    model.load_state_dict(torch.load(OUTPUT_PT, map_location=device))
    model.eval()

    test_correct = test_total = 0
    test_top5_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            test_correct += (logits.argmax(1) == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            test_top5_correct += sum(
                y[i].item() in top5[i].tolist()
                for i in range(y.size(0))
            )
            test_total += y.size(0)

    test_top1 = test_correct / test_total
    test_top5 = test_top5_correct / test_total

    print(f"\n{'='*52}")
    print(f"Final Test Results")
    print(f"{'='*52}")
    print(f"  Top-1 accuracy : {test_top1*100:.2f}%")
    print(f"  Top-5 accuracy : {test_top5*100:.2f}%")
    print(f"  Parameters     : {total_params:,}  ({total_params/1e6:.3f}M)")
    print(f"  FLOPs          : {best_entry['flops']/1e6:.1f}M")
    print(f"  Model saved    : {OUTPUT_PT}")
    print(f"{'='*52}")

    # Save final results summary
    os.makedirs('exports', exist_ok=True)
    summary = {
        'test_top1':  test_top1,
        'test_top5':  test_top5,
        'val_acc':    best_val_acc,
        'params':     total_params,
        'flops':      best_entry['flops'],
        'epochs':     EPOCHS,
    }
    with open('exports/finetune_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved -> exports/finetune_results.json")


if __name__ == '__main__':
    finetune()