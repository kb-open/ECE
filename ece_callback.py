import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from ece_regularization import extract_token_embeddings, ECEConfig, spectral_shannon_entropy, logdet_free_entropy

class SpectrumLoggingCallback(TrainerCallback):
    def __init__(self, log_every=500, cfg: ECEConfig = None, logger="wandb"):
        self.log_every = log_every
        self.cfg = cfg or ECEConfig()
        self.logger = logger  # "wandb" or "tensorboard"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.log_every != 0 or model is None:
            return

        emb = extract_token_embeddings(model).weight.detach().cpu()

        # Compute eigen spectrum
        X = emb - emb.mean(dim=0, keepdim=True)
        cov = (X.T @ X) / max(X.shape[0], 1)
        evals = torch.linalg.eigvalsh(cov).clamp(min=1e-8).numpy()

        # Free entropy metrics
        chi_logdet = logdet_free_entropy(emb, self.cfg).item()
        chi_shannon = spectral_shannon_entropy(emb, self.cfg).item()

        # Plot spectrum histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(evals, bins=50, alpha=0.7, color="steelblue")
        ax.set_title(f"Embedding Spectrum (step {step})")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Frequency")

        # Log
        if self.logger == "wandb":
            import wandb
            wandb.log({
                "spectrum_plot": wandb.Image(fig),
                "chi_logdet": chi_logdet,
                "chi_shannon": chi_shannon,
                "step": step,
            })
        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=args.logging_dir)
            tb.add_scalar("chi_logdet", chi_logdet, step)
            tb.add_scalar("chi_shannon", chi_shannon, step)
            tb.add_histogram("spectrum", evals, step)

        plt.close(fig)
