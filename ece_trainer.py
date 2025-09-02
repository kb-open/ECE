from transformers import Trainer
import torch.nn as nn
from ece_regularization import ECERegularizer

class ECETrainer(Trainer):
    def __init__(self, *args, ece: ECERegularizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ece = ece

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard task loss
        outputs = model(**inputs)
        task_loss = outputs.loss if hasattr(outputs, "loss") else None
        if task_loss is None:
            logits = outputs.logits
            task_loss = nn.CrossEntropyLoss()(logits, inputs["labels"])

        # ECE penalty
        penalty, metrics = self.ece(model=model)
        total_loss = task_loss + penalty

        # Log metrics
        self.log({
            "task_loss": float(task_loss.detach().cpu().item()),
            "ece_penalty": float(penalty.detach().cpu().item()),
            "chi": metrics["chi"],
            "spec_var": metrics["spec_var"],
            "top_eig_share": metrics["top_eig_share"],
        })

        return (total_loss, outputs) if return_outputs else total_loss
