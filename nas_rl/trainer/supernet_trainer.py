# Supernet trainer definition
import torch
import torch.nn as nn


class SupernetTrainer:
    """
    Pretrains the supernet before search begins.

    During pretraining, architectures are sampled uniformly at random
    — NOT from the controller — so every op at every edge gets trained
    roughly equally.  This ensures no op starts search with a systematic
    weight advantage, which would corrupt the accuracy reward signal.

    Optimiser : SGD with momentum and cosine LR annealing.
                Adam tends to overfit faster on small supernets; SGD
                generalises better and is the standard choice in NAS
                literature (DARTS, SNAS, etc.).

    Gradient clipping at 5.0 is applied every step.  Early in training
    some ops have near-random weights and produce large gradients; without
    clipping these destabilise the other ops sharing the same cell.
    """

    def __init__(
        self,
        supernet,       # the Supernet instance (from supernet.py)
        train_loader,   # DataLoader for the 80% training split
        device,         # torch.device — mps / cuda / cpu
        lr=0.025,       # initial learning rate for SGD
        momentum=0.9,
        weight_decay=3e-4,
        n_epochs=50,    # how many epochs to pretrain
        grad_clip=5.0,  # gradient norm clip threshold
    ):
        self.supernet     = supernet
        self.train_loader = train_loader
        self.device       = device
        self.grad_clip    = grad_clip
        self.n_epochs     = n_epochs

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            supernet.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Cosine annealing decays lr smoothly to near-zero over n_epochs.
        # Avoids sharp loss jumps that confuse op weight estimates.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-4,
        )

    def train_epoch(self):
        """
        One full pass over the training set.

        A fresh random arch is sampled every batch so all ops
        get roughly equal gradient updates over the epoch.

        Returns (avg_loss, train_accuracy) for logging.
        """
        self.supernet.train()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # New random arch every batch — maximises op coverage
            arch_spec = self.supernet.random_arch()

            logits = self.supernet(x, arch_spec)
            loss   = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()

            # Clip before step — prevents a single bad op from
            # blowing up the weights of other ops it shares a cell with
            nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss    += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += x.size(0)

        self.scheduler.step()   # advance cosine LR after each epoch

        return total_loss / total_samples, total_correct / total_samples

    def train(self, checkpoint_path=None):
        """
        Run the full pretraining loop for self.n_epochs epochs.

        Prints a table of loss / accuracy / LR each epoch.
        If checkpoint_path is given, saves the supernet state dict
        at the end so you don't have to redo pretraining between runs.

        Returns the trained supernet.
        """
        print(f"Supernet pretraining — {self.n_epochs} epochs")
        print(f"{'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}  {'LR':>10}")
        print("-" * 42)

        for epoch in range(self.n_epochs):
            loss, acc = self.train_epoch()
            lr = self.scheduler.get_last_lr()[0]
            print(f"{epoch+1:6d}  {loss:8.4f}  {acc*100:9.2f}%  {lr:10.6f}")

        if checkpoint_path is not None:
            torch.save(self.supernet.state_dict(), checkpoint_path)
            print(f"\nSupernet weights saved to {checkpoint_path}")

        return self.supernet