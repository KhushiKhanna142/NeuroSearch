# Model evaluator definition
import torch

class Evaluator:
    """
    Evaluates candidate architectures from the search space on a proxy validation set.
    """
    def __init__(self, supernet, val_loader, device):
        """
        Parameters
        ----------
        supernet   : Supernet
            The weight-sharing supernet.
        val_loader : DataLoader
            A validation dataloader to evaluate model accuracy.
        device     : torch.device
            CPU or GPU device.
        """
        self.supernet = supernet
        self.val_loader = val_loader
        self.device = device

    def evaluate(self, arch_spec):
        """
        Run a forward validation pass using the specific architecture configuration.

        Parameters
        ----------
        arch_spec : list[dict]
            An architecture specification containing the selected ops for each cell/edge.

        Returns
        -------
        accuracy : float
            Top-1 accuracy of the candidate architecture in the range [0.0, 1.0].
        """
        self.supernet.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.supernet(inputs, arch_spec)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        return correct / max(total, 1)
