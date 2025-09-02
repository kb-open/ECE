# PyTorch utilities to add Entropy-Constrained Embeddings (ECE) to Transformer fine-tuning.
# Works with HuggingFace models or any nn.Module exposing an embedding weight matrix.

from dataclasses import dataclass
from typing import Optional, Dict, Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ece_regularization import *

# ------------------------------
# Example integration in a training loop
# ------------------------------

def freeze_all_but_embeddings_and_first_attn(model: nn.Module):
    """Freeze all params except token embeddings and first attention block (if present)."""
    for p in model.parameters():
        p.requires_grad = False

    # token embeddings
    try:
        emb = extract_token_embeddings(model)
        for p in emb.parameters():
            p.requires_grad = True
    except Exception:
        pass

    # try to unfreeze first attention block (common HF naming)
    for path in [
        "model.layers.0.self_attn",      # LLaMA-style
        "transformer.h.0.attn",          # GPT2-style
        "bert.encoder.layer.0.attention" # BERT-style
    ]:
        cur = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            for p in cur.parameters():
                p.requires_grad = True
            break


def training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    ece: ECERegularizer,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    autocast_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    One training step with ECE. Assumes a standard HF-like forward that returns logits.
    batch: should contain input_ids, attention_mask, labels (adjust as needed).
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None
    ctx = torch.cuda.amp.autocast(dtype=autocast_dtype) if use_amp else torch.autocast("cpu")

    with ctx:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None)  # if model returns loss when labels provided
        )
        # If model returns built-in loss (HF), use it; else compute criterion on logits.
        if hasattr(outputs, "loss") and outputs.loss is not None:
            task_loss = outputs.loss
        else:
            logits = outputs.logits
            task_loss = criterion(logits, batch["labels"])

        # ECE penalty
        penalty, m = ece(model=model)
        total_loss = task_loss + penalty

    # Backprop
    if use_amp:
        scaler.scale(total_loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()
    else:
        total_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # Basic logs
    logs = {
        "loss_task": float(task_loss.detach().item()),
        "loss_total": float(total_loss.detach().item()),
        **m
    }
    return logs


# ------------------------------
# Minimal usage demo (pseudo, adapt to your pipeline)
# ------------------------------
if __name__ == "__main__":
    # Example with a dummy model; replace with HF AutoModelForSequenceClassification(...)
    class TinyModel(nn.Module):
        def __init__(self, vocab=5000, d=256, num_labels=2):
            super().__init__()
            self.emb = nn.Embedding(vocab, d)
            self.encoder = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, d),
            )
            self.head = nn.Linear(d, num_labels)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.emb(input_ids)                    # [B, T, d]
            x = x.mean(dim=1)                          # toy pooling
            z = self.encoder(x)
            logits = self.head(z)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            return type("Out", (), {"logits": logits, "loss": loss})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyModel().to(device)

    # Freeze policy: only embeddings + first block (if you have one) will train
    freeze_all_but_embeddings_and_first_attn(model)

    cfg = ECEConfig(lambda_ece=0.075, mode="logdet", sketch_dim=2048, max_eigs=None)
    ece = ECERegularizer(cfg).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    # Fake batch
    B, T = 16, 64
    batch = {
        "input_ids": torch.randint(0, 5000, (B, T), device=device),
        "labels": torch.randint(0, 2, (B,), device=device),
    }

    logs = training_step(
        model=model,
        batch=batch,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        ece=ece,
        scaler=None,  # set GradScaler() if using CUDA AMP
    )
    print({k: round(v, 4) for k, v in logs.items()})
