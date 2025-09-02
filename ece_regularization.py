# PyTorch utilities to add Entropy-Constrained Embeddings (ECE) to Transformer fine-tuning.
# Works with HuggingFace models or any nn.Module exposing an embedding weight matrix.

from dataclasses import dataclass
from typing import Optional, Dict, Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

EntropyMode = Literal["logdet", "spectral_shannon", "both"]

@dataclass
class ECEConfig:
    lambda_ece: float = 0.075          # strength of entropy regularization
    mode: EntropyMode = "logdet"       # "logdet" | "spectral_shannon" | "both"
    eps: float = 1e-5                  # numerical jitter for PD covariance
    # For spectral_shannon on large vocabularies we use a sketch:
    sketch_dim: int = 2048             # number of rows in random sketch (<= vocab_size recommended)
    with_centering: bool = True        # center embeddings before covariance
    detach_scale: bool = True          # detach scale to reduce collapse pressure
    max_eigs: Optional[int] = None     # if not None, compute top-k eigs on sketch for Shannon term


def extract_token_embeddings(model: nn.Module) -> nn.Embedding:
    """
    Try common attribute names used by HF models; otherwise raise.
    """
    for name in ["get_input_embeddings", "model.embed_tokens", "transformer.wte", "embeddings.word_embeddings"]:
        mod = None
        if hasattr(model, "get_input_embeddings") and name == "get_input_embeddings":
            mod = model.get_input_embeddings()
        else:
            # traverse dotted path if present
            cur = model
            ok = True
            for part in name.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, nn.Embedding):
                mod = cur
        if isinstance(mod, nn.Embedding):
            return mod
    raise AttributeError("Could not locate a token embedding layer; pass it explicitly.")


def logdet_free_entropy(E: torch.Tensor, cfg: ECEConfig) -> torch.Tensor:
    """
    χ_logdet(E) ≈ (1/d) * log det( (EᵀE / n) + ε I )
    Stable via Cholesky (positive-definite covariance).
    E: [n, d] embedding matrix (vocab_size x dim).
    """
    n, d = E.shape
    X = E
    if cfg.with_centering:
        X = X - X.mean(dim=0, keepdim=True)

    cov = (X.t() @ X) / float(max(n, 1))  # [d, d]
    cov = cov + cfg.eps * torch.eye(d, device=E.device, dtype=E.dtype)

    # logdet via Cholesky for stability
    L = torch.linalg.cholesky(cov)               # cov = L L^T
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum()
    chi = logdet / d
    return chi


def spectral_shannon_entropy(E: torch.Tensor, cfg: ECEConfig) -> torch.Tensor:
    """
    χ_spec(E): Shannon entropy over normalized eigenvalue spectrum of (EᵀE / n).
    Uses a random sketch (subsample rows of E) for efficiency on large vocab.
    """
    n, d = E.shape
    X = E
    if cfg.with_centering:
        X = X - X.mean(dim=0, keepdim=True)

    # Random sketch: sample rows without replacement if sketch_dim < n
    if cfg.sketch_dim is not None and cfg.sketch_dim < n:
        idx = torch.randperm(n, device=E.device)[:cfg.sketch_dim]
        Xs = X.index_select(0, idx)
        ns = Xs.size(0)
        cov = (Xs.t() @ Xs) / float(max(ns, 1))
    else:
        cov = (X.t() @ X) / float(max(n, 1))

    cov = cov + cfg.eps * torch.eye(d, device=E.device, dtype=E.dtype)

    # Get eigenvalues (symmetric PSD matrix)
    # If d is large and you want only top-k: use torch.linalg.eigh on cov (full) or torch.lobpcg for k
    if cfg.max_eigs is not None and cfg.max_eigs < d:
        # Approximate top-k eigenvalues (LOBPCG expects symmetric; provide random init)
        k = cfg.max_eigs
        init = torch.randn(d, k, device=E.device, dtype=E.dtype)
        evals, _ = torch.lobpcg(cov, k=k, B=None, X=init, largest=True)
        lambdas = torch.clamp(evals, min=cfg.eps)  # [k]
    else:
        lambdas = torch.linalg.eigvalsh(cov)       # [d]
        lambdas = torch.clamp(lambdas, min=cfg.eps)

    # Normalize to a probability simplex
    Z = lambdas.sum()
    if cfg.detach_scale:
        Z = Z.detach()  # normalize shape, not absolute scale
    p = lambdas / Z

    # Shannon entropy H(p)
    chi = (-p * (p.clamp_min(1e-12)).log()).sum()
    return chi


class ECERegularizer(nn.Module):
    """
    Computes ECE penalty given a model (or embedding tensor) and returns:
      penalty, metrics_dict
    """
    def __init__(self, cfg: ECEConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def _summaries(self, E: torch.Tensor) -> Dict[str, float]:
        # Light metrics for logging/monitoring (no grad)
        n, d = E.shape
        X = E - E.mean(dim=0, keepdim=True)
        cov = (X.t() @ X) / float(max(n, 1))
        cov = cov + self.cfg.eps * torch.eye(d, device=E.device, dtype=E.dtype)
        evals = torch.linalg.eigvalsh(cov)
        evals = torch.clamp(evals, min=self.cfg.eps)
        spec_var = torch.var(evals).item()
        top_share = (evals.max() / evals.sum()).item()
        return {
            "spec_var": spec_var,
            "top_eig_share": top_share,
        }

    def forward(
        self,
        model: Optional[nn.Module] = None,
        embedding_layer: Optional[nn.Embedding] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert (model is not None) or (embedding_layer is not None), \
            "Provide either a model or an embedding layer."

        emb = embedding_layer if embedding_layer is not None else extract_token_embeddings(model)
        E = emb.weight  # [vocab, dim]

        # Compute chosen surrogate(s)
        chi_vals = {}
        if self.cfg.mode in ("logdet", "both"):
            chi_vals["logdet"] = logdet_free_entropy(E, self.cfg)
        if self.cfg.mode in ("spectral_shannon", "both"):
            chi_vals["spectral_shannon"] = spectral_shannon_entropy(E, self.cfg)

        if self.cfg.mode == "both":
            # Balanced sum; you can also weight these terms separately if desired
            chi = 0.5 * (chi_vals["logdet"] + chi_vals["spectral_shannon"])
        else:
            chi = next(iter(chi_vals.values()))

        penalty = self.cfg.lambda_ece * chi

        # Metrics for logs (no grad)
        with torch.no_grad():
            metrics = {"chi": float(chi.detach().item()),
                       "penalty": float(penalty.detach().item())}
            metrics.update(self._summaries(E))

        return penalty, metrics
