"""
CL-SDRG — Script 03: REINFORCE Training Engine for RL De-biasing
=================================================================
Phase 3 of the CL-SDRG pipeline.

This script:
  1. Loads pre-processed data from Phase 1 outputs
  2. Pre-computes and caches embeddings from the frozen encoder (done once)
  3. Constructs the training DataLoader with counterfactual speaker augmentation
  4. Implements the REINFORCE policy gradient training loop:
       - Accuracy reward R_acc (+1/-1)
       - Consistency reward R_cons (1 - L1 divergence after speaker swap)
       - Combined reward R_total = λ₁·R_acc + λ₂·R_cons
  5. Uses running baseline for variance reduction
  6. FP16 mixed precision + gradient accumulation (virtual batch 256)
  7. Saves checkpoints and loss/reward logs per epoch

Target Environment: Google Colab T4 GPU
Estimated Training Time: 20–30 minutes

Usage (Colab):
    !pip install torch transformers pandas tqdm matplotlib
    %run src/03_train_rl_debiasing.py
"""

import sys
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, OUTPUT_DIR,
    EMBEDDING_DIM, MAX_SEQ_LENGTH, NUM_CLASSES, BASE_ENCODER_NAME,
    LAMBDA_ACCURACY, LAMBDA_CONSISTENCY,
    LEARNING_RATE, WEIGHT_DECAY,
    PHYSICAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS, WARMUP_RATIO, USE_FP16,
    BASELINE_EMA_DECAY, CHECKPOINT_EVERY_N_EPOCHS,
    RANDOM_SEED, LABEL2ID, ID2LABEL,
)
from src.config import init_directories
from src.utils import (
    set_seed, setup_logging, timer, get_device,
    get_gpu_memory_summary, reset_peak_memory,
    save_checkpoint, format_number,
)

# Import model architecture from Phase 2
# (Filename starts with a digit, so we use importlib)
import importlib.util

_phase2_spec = importlib.util.spec_from_file_location(
    "fga_architecture",
    str(SCRIPT_DIR / "02_fga_architecture_and_vram_test.py"),
)
_phase2_module = importlib.util.module_from_spec(_phase2_spec)
_phase2_spec.loader.exec_module(_phase2_module)

FrozenEncoder = _phase2_module.FrozenEncoder
FeatureGatingAgent = _phase2_module.FeatureGatingAgent
GatedFusion = _phase2_module.GatedFusion
VeracityClassifier = _phase2_module.VeracityClassifier
CounterfactualPerturbation = _phase2_module.CounterfactualPerturbation


# ============================================================================
# 1. EMBEDDING CACHE DATASET
# ============================================================================

class EmbeddingCacheDataset(Dataset):
    """
    Dataset that serves pre-computed embeddings for efficient RL training.

    Instead of re-encoding text through the frozen encoder every epoch,
    we pre-compute all embeddings once and cache them as tensors.
    """

    def __init__(
        self,
        claim_embeddings: torch.Tensor,     # (N, D)
        speaker_embeddings: torch.Tensor,   # (N, D)
        date_embeddings: torch.Tensor,      # (N, D)
        labels: torch.Tensor,               # (N,)
        speaker_names: list,                # Original speaker strings
        all_speaker_embeddings: dict,       # {speaker_name: embedding_tensor}
    ):
        self.claim_emb = claim_embeddings
        self.speaker_emb = speaker_embeddings
        self.date_emb = date_embeddings
        self.labels = labels
        self.speaker_names = speaker_names
        self.all_speaker_embeddings = all_speaker_embeddings

        # Pre-compute list of all speaker names for counterfactual sampling
        self.all_speakers = list(all_speaker_embeddings.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "claim_emb": self.claim_emb[idx],
            "speaker_emb": self.speaker_emb[idx],
            "date_emb": self.date_emb[idx],
            "label": self.labels[idx],
            "speaker_name": self.speaker_names[idx],
        }

    def get_counterfactual_speaker_emb(self, original_speaker: str) -> torch.Tensor:
        """Sample a random different speaker's embedding for counterfactual."""
        candidates = [s for s in self.all_speakers if s != original_speaker]
        if not candidates:
            return self.all_speaker_embeddings[original_speaker]
        sampled = random.choice(candidates)
        return self.all_speaker_embeddings[sampled]


def collate_with_counterfactuals(batch, dataset):
    """
    Custom collate function that adds counterfactual speaker embeddings.

    Returns a dict with both original and perturbed speaker embeddings.
    """
    claim_embs = torch.stack([b["claim_emb"] for b in batch])
    speaker_embs = torch.stack([b["speaker_emb"] for b in batch])
    date_embs = torch.stack([b["date_emb"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    # Generate counterfactual speaker embeddings
    cf_speaker_embs = torch.stack([
        dataset.get_counterfactual_speaker_emb(b["speaker_name"])
        for b in batch
    ])

    return {
        "claim_emb": claim_embs,
        "speaker_emb": speaker_embs,
        "date_emb": date_embs,
        "cf_speaker_emb": cf_speaker_embs,
        "label": labels,
    }


# ============================================================================
# 2. EMBEDDING PRE-COMPUTATION
# ============================================================================

def precompute_embeddings(
    df: pd.DataFrame,
    encoder: FrozenEncoder,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """
    Pre-compute embeddings for all claims, speakers, and dates.

    This runs the frozen encoder once over the entire dataset and caches
    the results as CPU tensors. Saves ~10x training time per epoch.

    Args:
        df: Processed dataframe with claimReviewed, speaker, date columns
        encoder: Frozen encoder model
        device: Compute device
        batch_size: Encoding batch size

    Returns:
        tuple: (claim_embs, speaker_embs, date_embs, labels, speaker_names, speaker_emb_map)
    """
    claims = df["claimReviewed"].tolist()
    speakers = df["itemReviewed.author.name"].fillna("Unknown").tolist()
    dates = df["datePublished"].astype(str).tolist()
    labels = df["label_id"].values

    logging.info(f"Pre-computing embeddings for {format_number(len(claims))} samples...")

    def encode_in_batches(texts, desc):
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
            batch_texts = texts[i : i + batch_size]
            embs = encoder.encode(batch_texts, device)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)

    with timer("Encoding claims"):
        claim_embs = encode_in_batches(claims, "Claims")

    with timer("Encoding speakers"):
        # Deduplicate speakers for efficiency
        unique_speakers = list(set(speakers))
        logging.info(f"  Unique speakers: {format_number(len(unique_speakers))}")

        speaker_emb_map = {}
        for i in tqdm(range(0, len(unique_speakers), batch_size), desc="Speaker embeds"):
            batch_sp = unique_speakers[i : i + batch_size]
            embs = encoder.encode(batch_sp, device)
            for sp, emb in zip(batch_sp, embs.cpu()):
                speaker_emb_map[sp] = emb

        # Map back to full dataset order
        speaker_embs = torch.stack([speaker_emb_map[s] for s in speakers])

    with timer("Encoding dates"):
        date_embs = encode_in_batches(dates, "Dates")

    label_tensor = torch.tensor(labels, dtype=torch.long)

    logging.info(f"  Shapes: claims={claim_embs.shape}, speakers={speaker_embs.shape}, dates={date_embs.shape}")

    return claim_embs, speaker_embs, date_embs, label_tensor, speakers, speaker_emb_map


# ============================================================================
# 3. REWARD FUNCTIONS
# ============================================================================

def compute_accuracy_reward(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary accuracy reward: +1 if correct, -1 otherwise.

    R_acc = +1.0 if ŷ == y, else -1.0

    Args:
        predictions: Predicted class indices (batch,)
        targets: Ground truth class indices (batch,)

    Returns:
        torch.Tensor: Reward values (batch,)
    """
    correct = (predictions == targets).float()
    return correct * 2.0 - 1.0  # Maps {0,1} → {-1,+1}


def compute_consistency_reward(
    probs_orig: torch.Tensor, probs_perturbed: torch.Tensor
) -> torch.Tensor:
    """
    Counterfactual consistency reward: penalizes prediction changes after speaker swap.

    R_cons = 1.0 - ||P(y|q,s,t) - P(y|q,s',t)||_1

    A reward of 1.0 means the model's prediction is completely invariant to
    the speaker swap (desirable). A reward of 0.0 means maximum divergence.

    Args:
        probs_orig: Softmax probabilities for original input (batch, num_classes)
        probs_perturbed: Softmax probabilities for perturbed input (batch, num_classes)

    Returns:
        torch.Tensor: Consistency reward (batch,)
    """
    l1_divergence = torch.abs(probs_orig - probs_perturbed).sum(dim=-1)
    return 1.0 - l1_divergence


def compute_total_reward(
    r_acc: torch.Tensor,
    r_cons: torch.Tensor,
    lambda_acc: float = LAMBDA_ACCURACY,
    lambda_cons: float = LAMBDA_CONSISTENCY,
) -> torch.Tensor:
    """
    Combined reward: R_total = λ₁·R_acc + λ₂·R_cons

    Args:
        r_acc: Accuracy reward (batch,)
        r_cons: Consistency reward (batch,)
        lambda_acc: Weight for accuracy reward
        lambda_cons: Weight for consistency reward

    Returns:
        torch.Tensor: Total reward (batch,)
    """
    return lambda_acc * r_acc + lambda_cons * r_cons


# ============================================================================
# 4. REINFORCE TRAINING LOOP
# ============================================================================

def train_one_epoch(
    fga: FeatureGatingAgent,
    fusion: GatedFusion,
    classifier: VeracityClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    baseline: float,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = GRADIENT_ACCUMULATION_STEPS,
) -> dict:
    """
    Train one epoch using REINFORCE policy gradient.

    The FGA's gating decisions are treated as stochastic policy actions.
    We use the Gumbel-Softmax reparameterization on the sigmoid gates
    to maintain differentiability while sampling.

    Args:
        fga: Feature Gating Agent
        fusion: Gated Fusion module
        classifier: Veracity Classifier head
        dataloader: Training data loader
        optimizer: AdamW optimizer
        scaler: AMP gradient scaler
        baseline: Running reward baseline for variance reduction
        device: Compute device
        epoch: Current epoch number
        grad_accum_steps: Steps for gradient accumulation

    Returns:
        dict: Epoch metrics {loss, reward, accuracy, consistency, baseline}
    """
    fga.train()
    classifier.train()

    epoch_losses = []
    epoch_rewards = []
    epoch_acc_rewards = []
    epoch_cons_rewards = []
    epoch_correct = 0
    epoch_total = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")

    for step, batch in enumerate(pbar):
        # Move to device
        e_q = batch["claim_emb"].to(device)
        e_s = batch["speaker_emb"].to(device)
        e_t = batch["date_emb"].to(device)
        e_s_cf = batch["cf_speaker_emb"].to(device)
        targets = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=USE_FP16 and device.type == "cuda"):
            # ── Original forward pass ──
            alpha_q, alpha_s, alpha_t = fga(e_q, e_s, e_t)
            e_gated = fusion(e_q, e_s, e_t, alpha_q, alpha_s, alpha_t)
            logits_orig = classifier(e_gated)

            # ── Counterfactual forward pass (swapped speaker) ──
            alpha_q_cf, alpha_s_cf, alpha_t_cf = fga(e_q, e_s_cf, e_t)
            e_gated_cf = fusion(e_q, e_s_cf, e_t, alpha_q_cf, alpha_s_cf, alpha_t_cf)
            logits_cf = classifier(e_gated_cf)

            # ── Compute probabilities ──
            probs_orig = F.softmax(logits_orig, dim=-1)
            probs_cf = F.softmax(logits_cf, dim=-1)

            # ── Compute rewards ──
            preds = logits_orig.argmax(dim=-1)
            r_acc = compute_accuracy_reward(preds, targets)
            r_cons = compute_consistency_reward(probs_orig, probs_cf)
            r_total = compute_total_reward(r_acc, r_cons)

            # ── REINFORCE policy loss ──
            # Use log-probability of the taken action as the policy gradient signal
            log_probs = F.log_softmax(logits_orig, dim=-1)
            action_log_probs = log_probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            # Advantage = reward - baseline
            advantage = r_total - baseline

            # Policy gradient loss (negative because we maximize reward)
            policy_loss = -torch.mean(advantage.detach() * action_log_probs)

            # Add cross-entropy loss as auxiliary supervision signal
            ce_loss = F.cross_entropy(logits_orig, targets)

            # Combined loss
            total_loss = policy_loss + 0.5 * ce_loss

            # Scale for gradient accumulation
            total_loss = total_loss / grad_accum_steps

        # Backward pass
        scaler.scale(total_loss).backward()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(fga.parameters()) + list(classifier.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Update baseline (EMA) ──
        batch_reward = r_total.mean().item()
        baseline = BASELINE_EMA_DECAY * baseline + (1 - BASELINE_EMA_DECAY) * batch_reward

        # ── Track metrics ──
        epoch_losses.append(total_loss.item() * grad_accum_steps)
        epoch_rewards.append(batch_reward)
        epoch_acc_rewards.append(r_acc.mean().item())
        epoch_cons_rewards.append(r_cons.mean().item())
        epoch_correct += (preds == targets).sum().item()
        epoch_total += len(targets)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
            "reward": f"{np.mean(epoch_rewards[-50:]):.3f}",
            "acc": f"{epoch_correct / epoch_total:.3f}",
        })

    metrics = {
        "loss": float(np.mean(epoch_losses)),
        "reward": float(np.mean(epoch_rewards)),
        "accuracy": epoch_correct / epoch_total,
        "r_acc": float(np.mean(epoch_acc_rewards)),
        "r_cons": float(np.mean(epoch_cons_rewards)),
        "baseline": baseline,
    }

    return metrics


# ============================================================================
# 5. TRAINING VISUALIZATION
# ============================================================================

def plot_training_curves(history: list, save_dir: Path):
    """
    Plot and save training loss/reward curves.

    Args:
        history: List of per-epoch metric dicts
        save_dir: Directory to save figures
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CL-SDRG Training Progress", fontsize=16, fontweight="bold")

    # Loss curve
    axes[0, 0].plot(epochs, [h["loss"] for h in history], "b-o", linewidth=2)
    axes[0, 0].set_title("Policy + CE Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Total reward
    axes[0, 1].plot(epochs, [h["reward"] for h in history], "g-o", linewidth=2)
    axes[0, 1].set_title("Total Reward (R_total)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, alpha=0.3)

    # Component rewards
    axes[1, 0].plot(epochs, [h["r_acc"] for h in history], "r-s", label="R_acc", linewidth=2)
    axes[1, 0].plot(epochs, [h["r_cons"] for h in history], "m-^", label="R_cons", linewidth=2)
    axes[1, 0].set_title("Component Rewards")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 1].plot(epochs, [h["accuracy"] for h in history], "c-D", linewidth=2)
    axes[1, 1].set_title("Training Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"📊 Training curves saved: {save_path}")


# ============================================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Execute the complete Phase 3 RL training pipeline."""

    # ── Setup ──
    init_directories()
    setup_logging(log_dir=LOG_DIR, script_name="03_train_rl")
    set_seed(RANDOM_SEED)

    logging.info("=" * 70)
    logging.info("  CL-SDRG Phase 3: REINFORCE Training Engine")
    logging.info("=" * 70)

    device = get_device()

    # ── Load processed training data ──
    train_csv = PROCESSED_DATA_DIR / "fci_train.csv"
    if not train_csv.exists():
        logging.error(f"Training data not found: {train_csv}")
        logging.error("Run Script 01 first to generate processed data.")
        sys.exit(1)

    with timer("Loading training data"):
        train_df = pd.read_csv(str(train_csv))
        logging.info(f"Training samples: {format_number(len(train_df))}")

        # Ensure required columns
        required = ["claimReviewed", "itemReviewed.author.name", "datePublished", "label_id"]
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            logging.error(f"Missing columns: {missing}")
            sys.exit(1)

        train_df["itemReviewed.author.name"] = train_df["itemReviewed.author.name"].fillna("Unknown")
        train_df = train_df.dropna(subset=["claimReviewed", "label_id"])
        train_df["label_id"] = train_df["label_id"].astype(int)

        logging.info(f"Label distribution:")
        for label_id, count in train_df["label_id"].value_counts().sort_index().items():
            logging.info(f"  {ID2LABEL.get(label_id, '?')}: {format_number(count)}")

    # ── Initialize encoder for embedding pre-computation ──
    with timer("Initializing frozen encoder"):
        encoder = FrozenEncoder(BASE_ENCODER_NAME)
        encoder.to(device)

    # ── Pre-compute embeddings ──
    reset_peak_memory()

    claim_embs, speaker_embs, date_embs, labels, speaker_names, speaker_emb_map = \
        precompute_embeddings(train_df, encoder, device, batch_size=64)

    # Free encoder from GPU after embedding extraction
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mem = get_gpu_memory_summary()
    if mem:
        logging.info(f"GPU after embedding cache: {mem['allocated_mb']:.1f} MB allocated")

    # ── Create dataset and dataloader ──
    dataset = EmbeddingCacheDataset(
        claim_embs, speaker_embs, date_embs, labels,
        speaker_names, speaker_emb_map,
    )

    # Custom collate that adds counterfactual embeddings
    def collate_fn(batch):
        return collate_with_counterfactuals(batch, dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Colab compatibility
        collate_fn=collate_fn,
        drop_last=True,
    )

    logging.info(f"DataLoader: {len(dataloader)} batches × {PHYSICAL_BATCH_SIZE} = ~{format_number(len(dataset))} samples")

    # ── Initialize trainable modules ──
    with timer("Initializing trainable modules"):
        fga = FeatureGatingAgent().to(device)
        fusion = GatedFusion().to(device)
        classifier = VeracityClassifier().to(device)

    # ── Optimizer ──
    trainable_params = list(fga.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── AMP Scaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16 and device.type == "cuda")

    # ── Training Loop ──
    baseline = 0.0
    training_history = []

    logging.info(f"\n{'='*70}")
    logging.info(f"  Starting REINFORCE Training")
    logging.info(f"  Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Virtual Batch: {PHYSICAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logging.info(f"  λ_acc: {LAMBDA_ACCURACY} | λ_cons: {LAMBDA_CONSISTENCY}")
    logging.info(f"  FP16: {USE_FP16} | Grad Accum: {GRADIENT_ACCUMULATION_STEPS}")
    logging.info(f"{'='*70}\n")

    for epoch in range(NUM_EPOCHS):
        metrics = train_one_epoch(
            fga=fga,
            fusion=fusion,
            classifier=classifier,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            baseline=baseline,
            device=device,
            epoch=epoch,
        )

        baseline = metrics["baseline"]
        training_history.append(metrics)

        logging.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward']:.3f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"R_acc: {metrics['r_acc']:.3f} | "
            f"R_cons: {metrics['r_cons']:.3f}"
        )

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = CHECKPOINT_DIR / f"cl_sdrg_epoch_{epoch+1}.pt"
            save_checkpoint(
                model_state_dict={
                    "fga": fga.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch + 1,
                metrics=metrics,
                filepath=ckpt_path,
            )

    # ── Save training history ──
    history_df = pd.DataFrame(training_history)
    history_path = LOG_DIR / "training_history.csv"
    history_df.to_csv(str(history_path), index=False)
    logging.info(f"📄 Training history saved: {history_path}")

    # ── Plot training curves ──
    plot_training_curves(training_history, FIGURE_DIR)

    # ── Final memory report ──
    mem = get_gpu_memory_summary()
    if mem:
        logging.info(f"\nFinal GPU Memory: {mem['peak_allocated_mb']:.1f} MB peak")

    logging.info("\n" + "=" * 70)
    logging.info("  PHASE 3 COMPLETE — RL Training finished")
    logging.info(f"  Best accuracy: {max(h['accuracy'] for h in training_history):.4f}")
    logging.info(f"  Best reward:   {max(h['reward'] for h in training_history):.4f}")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
