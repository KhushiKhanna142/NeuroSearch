"""
train.py — Complete NAS + RL pipeline for CIFAR-10.

Usage
-----
# Full run (supernet pretrain + search + export):
    python nas_rl/train.py

# Skip pretraining if checkpoint already exists:
    python nas_rl/train.py --skip_pretrain

# Quick smoke-test (tiny model, few episodes):
    python nas_rl/train.py --smoke_test

Steps
-----
1. Pretrain supernet (50 epochs, ~2-3 hrs on M4 Air)
2. Run RL search (200 episodes, ~2-3 hrs on M4 Air)
3. Export best found architecture to CoreML + ONNX
"""

import os
import argparse
import torch

# ---- project imports ----
from nas_rl.search_space.supernet import Supernet
from nas_rl.controller.lstm_controller import LSTMController
from nas_rl.trainer.supernet_trainer import SupernetTrainer
from nas_rl.trainer.evaluator import Evaluator
from nas_rl.reward.cost_estimator import FLOPsEstimator
from nas_rl.reward.reward_combiner import RewardCombiner
from nas_rl.search.search_loop import SearchLoop
from nas_rl.export.model_builder import StandaloneModel, transfer_weights
from nas_rl.export.exporter import ModelExporter
from nas_rl.data import get_cifar10_loader


# ==================================================================
# Argument parsing
# ==================================================================

def get_args():
    p = argparse.ArgumentParser(description='NAS with RL on CIFAR-10')

    # Hardware
    p.add_argument('--device', default='auto',
                   help="'auto' | 'mps' | 'cuda' | 'cpu'")

    # Supernet
    p.add_argument('--C_init',   type=int, default=16)
    p.add_argument('--n_cells',  type=int, default=8)
    p.add_argument('--n_nodes',  type=int, default=4)

    # Pretraining
    p.add_argument('--pretrain_epochs', type=int,   default=50)
    p.add_argument('--pretrain_lr',     type=float, default=0.025)
    p.add_argument('--pretrain_bs',     type=int,   default=128)
    p.add_argument('--skip_pretrain',   action='store_true',
                   help='Load supernet checkpoint instead of retraining')
    p.add_argument('--supernet_ckpt', default='checkpoints/supernet.pt')

    # Search
    p.add_argument('--n_episodes',    type=int,   default=200)
    p.add_argument('--ctrl_lr',       type=float, default=3e-3)
    p.add_argument('--entropy_coeff', type=float, default=0.01)
    p.add_argument('--flops_target',  type=float, default=50e6)
    p.add_argument('--alpha',         type=float, default=0.07)
    p.add_argument('--search_ckpt',   default='checkpoints/search.pt')
    p.add_argument('--resume_search', action='store_true')

    # Export
    p.add_argument('--export_dir', default='exports')
    p.add_argument('--export_coreml', action='store_true', default=True)
    p.add_argument('--export_onnx',   action='store_true', default=True)

    # Smoke test (tiny model, quick run)
    p.add_argument('--smoke_test', action='store_true',
                   help='Tiny model + 5 episodes to verify the pipeline works')

    return p.parse_args()


# ==================================================================
# Device selection
# ==================================================================

def get_device(preference):
    if preference == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(preference)


# ==================================================================
# Phase 1 — Supernet pretraining
# ==================================================================

def pretrain(args, device):
    print("\n" + "=" * 60)
    print("Phase 1 — Supernet Pretraining")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.supernet_ckpt), exist_ok=True)

    supernet = Supernet(
        C_init    = args.C_init,
        n_cells   = args.n_cells,
        n_nodes   = args.n_nodes,
        n_classes = 10,
    ).to(device)

    total_params = sum(p.numel() for p in supernet.parameters())
    print(f"Supernet parameters: {total_params:,}  ({total_params/1e6:.2f}M)")

    if args.skip_pretrain and os.path.exists(args.supernet_ckpt):
        print(f"Loading pretrained weights from {args.supernet_ckpt}")
        supernet.load_state_dict(
            torch.load(args.supernet_ckpt, map_location=device)
        )
        return supernet

    train_loader = get_cifar10_loader(
        split      = 'train',
        batch_size = args.pretrain_bs,
        num_workers= 2,
    )

    trainer = SupernetTrainer(
        supernet   = supernet,
        train_loader = train_loader,
        device     = device,
        lr         = args.pretrain_lr,
        n_epochs   = args.pretrain_epochs,
    )

    trainer.train(checkpoint_path=args.supernet_ckpt)
    return supernet


# ==================================================================
# Phase 2 — Architecture Search
# ==================================================================

def search(args, supernet, device):
    print("\n" + "=" * 60)
    print("Phase 2 — RL Architecture Search")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.search_ckpt), exist_ok=True)

    val_loader = get_cifar10_loader(
        split      = 'val',
        batch_size = 256,
        num_workers= 2,
    )

    controller = LSTMController(
        n_cells    = args.n_cells,
        n_nodes    = args.n_nodes,
        hidden_dim = 64,
    ).to(device)

    evaluator  = Evaluator(supernet, val_loader, device)
    cost_est   = FLOPsEstimator(supernet)
    reward_fn  = RewardCombiner(
        flops_target = args.flops_target,
        alpha        = args.alpha,
        norm_warmup  = 10,
    )

    loop = SearchLoop(
        supernet        = supernet,
        controller      = controller,
        evaluator       = evaluator,
        reward_combiner = reward_fn,
        cost_estimator  = cost_est,
        device          = device,
        ctrl_lr         = args.ctrl_lr,
        entropy_coeff   = args.entropy_coeff,
    )

    if args.resume_search and os.path.exists(args.search_ckpt):
        loop.load_checkpoint(args.search_ckpt)

    top5 = loop.run(n_episodes=args.n_episodes, log_every=10)

    # Save final search checkpoint
    loop.save_checkpoint(args.search_ckpt)

    return top5, loop.arch_pool


# ==================================================================
# Phase 3 — Export best architecture
# ==================================================================

def export(args, top5, supernet, device):
    print("\n" + "=" * 60)
    print("Phase 3 — Export Best Architecture")
    print("=" * 60)

    os.makedirs(args.export_dir, exist_ok=True)

    best       = top5[0]
    arch_spec  = best['arch']

    print(f"Best architecture:")
    print(f"  Val accuracy : {best['accuracy']*100:.2f}%")
    print(f"  FLOPs        : {best['flops']/1e6:.2f}M")
    print(f"  Reward       : {best['reward']:.4f}")
    print(f"  Found at ep  : {best['episode']}")

    # Build standalone model
    standalone = StandaloneModel(
        arch_spec = arch_spec,
        C_init    = args.C_init,
        n_cells   = args.n_cells,
        n_nodes   = args.n_nodes,
        n_classes = 10,
    ).to(device)

    print(f"\nTransferring supernet weights...")
    transfer_weights(supernet, standalone, arch_spec)

    exporter = ModelExporter()
    exporter.print_model_info(standalone)

    # Export to CoreML
    if args.export_coreml:
        try:
            coreml_path = os.path.join(args.export_dir, 'nas_model.mlpackage')
            standalone_cpu = standalone.to('cpu')
            exporter.export_coreml(standalone_cpu, path=coreml_path)
        except Exception as e:
            print(f"CoreML export skipped: {e}")

    # Export to ONNX
    if args.export_onnx:
        try:
            onnx_path = os.path.join(args.export_dir, 'nas_model.onnx')
            standalone_cpu = standalone.to('cpu')
            exporter.export_onnx(standalone_cpu, path=onnx_path)
        except Exception as e:
            print(f"ONNX export skipped: {e}")

    # Also save the raw arch_spec and full top-5 for reference
    import json
    results = {
        'top5': [
            {
                'rank':     i + 1,
                'accuracy': e['accuracy'],
                'flops':    e['flops'],
                'reward':   e['reward'],
                'episode':  e['episode'],
            }
            for i, e in enumerate(top5)
        ]
    }
    results_path = os.path.join(args.export_dir, 'search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSearch results saved -> {results_path}")

    return standalone


# ==================================================================
# Smoke test — tiny model, 5 episodes, verifies pipeline is intact
# ==================================================================

def smoke_test_config(args):
    """Override args with minimal settings for a quick pipeline check."""
    args.C_init          = 4
    args.n_cells         = 2
    args.pretrain_epochs = 1
    args.pretrain_bs     = 64
    args.n_episodes      = 5
    args.skip_pretrain   = False
    args.supernet_ckpt   = 'checkpoints/smoke_supernet.pt'
    args.search_ckpt     = 'checkpoints/smoke_search.pt'
    args.export_dir      = 'exports/smoke'
    print("SMOKE TEST MODE — tiny model, 1 pretrain epoch, 5 episodes")
    return args


# ==================================================================
# Main
# ==================================================================

def main():
    args   = get_args()
    device = get_device(args.device)

    print(f"NAS with RL — CIFAR-10")
    print(f"Device : {device}")

    if args.smoke_test:
        args = smoke_test_config(args)

    # Phase 1: Pretrain supernet
    supernet = pretrain(args, device)

    # Phase 2: Search
    top5, pool = search(args, supernet, device)

    # Phase 3: Export
    export(args, top5, supernet, device)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Best val accuracy : {top5[0]['accuracy']*100:.2f}%")
    print(f"Best FLOPs        : {top5[0]['flops']/1e6:.2f}M")
    print("=" * 60)


if __name__ == '__main__':
    main()