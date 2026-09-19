"""
CL-SDRG — Script 04: Evaluation, Ablation Studies & Shortcut Sensitivity Audit
================================================================================
Phase 4 of the CL-SDRG pipeline.

This script:
  1. Loads the trained FGA + classifier checkpoint from Phase 3
  2. Evaluates on the time-aware test set (2024–2026):
       - Accuracy, Macro-F1, Precision, Recall (Classification)
       - Recall@1, Recall@5, Recall@20, MRR@20, nDCG@20 (Retrieval via FAISS)
  3. Runs Speaker Flip Rate (SFR) audit:
       - Perturbs speaker identity N times per test sample
       - Measures prediction invariance (target: SFR < 3%)
  4. Runs comparative ablation baselines:
       - BM25 Lexical Baseline
       - Zero-Shot Encoder (no gating)
       - Un-debiased fine-tuned model (no RL)
  5. Generates publication-ready result tables and visualizations

Target Environment: Google Colab T4 GPU

Usage (Colab):
    !pip install torch transformers pandas scikit-learn faiss-cpu rank_bm25 matplotlib seaborn tqdm
    %run src/04_evaluation_and_ablation.py
"""

import sys
import random
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, OUTPUT_DIR,
    EMBEDDING_DIM, NUM_CLASSES, BASE_ENCODER_NAME,
    RETRIEVAL_K_VALUES, FAISS_NPROBE,
    SFR_TARGET, SFR_NUM_PERTURBATIONS,
    RANDOM_SEED, LABEL2ID, ID2LABEL,
    PHYSICAL_BATCH_SIZE,
)
from src.config import init_directories
from src.utils import (
    set_seed, setup_logging, timer, get_device,
    get_gpu_memory_summary, load_checkpoint, format_number,
)

# Import model architecture (importlib for digit-prefixed filename)
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


# ============================================================================
# 1. CLASSIFICATION METRICS
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None
) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Predicted probabilities (N, C) — optional

    Returns:
        dict: {accuracy, macro_f1, macro_precision, macro_recall, per_class_f1}
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # Per-class F1
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(per_class):
        metrics[f"f1_{ID2LABEL.get(i, str(i))}"] = f1

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Classification report string
    target_names = [ID2LABEL.get(i, str(i)) for i in range(NUM_CLASSES)]
    metrics["report"] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )

    return metrics


# ============================================================================
# 2. RETRIEVAL METRICS (FAISS)
# ============================================================================

def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """
    Build a FAISS inner-product index for retrieval evaluation.

    Args:
        embeddings: Normalized embeddings (N, D)

    Returns:
        faiss.IndexFlatIP: FAISS index
    """
    import faiss

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logging.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    index: "faiss.IndexFlatIP",
    db_labels: np.ndarray,
    k_values: list = None,
) -> dict:
    """
    Compute retrieval metrics using FAISS nearest neighbor search.

    Metrics:
        - Recall@K: fraction of queries where a same-label item is in top-K
        - MRR@K: Mean Reciprocal Rank
        - nDCG@K: Normalized Discounted Cumulative Gain

    Args:
        query_embeddings: Query vectors (Q, D)
        query_labels: Query labels (Q,)
        index: FAISS index over the database
        db_labels: Database labels (N,)
        k_values: List of K values for metrics

    Returns:
        dict: Retrieval metrics
    """
    import faiss

    if k_values is None:
        k_values = RETRIEVAL_K_VALUES

    max_k = max(k_values)

    # Normalize queries
    query_embeddings = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings)

    # Search
    distances, indices = index.search(query_embeddings, max_k + 1)

    metrics = {}

    for k in k_values:
        recalls = []
        reciprocal_ranks = []
        ndcgs = []

        for i in range(len(query_embeddings)):
            query_label = query_labels[i]

            # Skip self-match (first result is often the query itself)
            retrieved_indices = indices[i]
            retrieved_labels = db_labels[retrieved_indices]

            # Filter out self-match
            mask = retrieved_indices != i
            retrieved_labels = retrieved_labels[mask][:k]

            # Recall@K: is there at least one correct match?
            relevant = (retrieved_labels == query_label)
            recalls.append(float(relevant.any()))

            # MRR@K: reciprocal rank of first correct match
            correct_positions = np.where(relevant)[0]
            if len(correct_positions) > 0:
                reciprocal_ranks.append(1.0 / (correct_positions[0] + 1))
            else:
                reciprocal_ranks.append(0.0)

            # nDCG@K
            dcg = sum(
                rel / np.log2(pos + 2)
                for pos, rel in enumerate(relevant.astype(float))
            )
            ideal_relevant = min(k, relevant.sum())
            idcg = sum(1.0 / np.log2(pos + 2) for pos in range(max(1, ideal_relevant)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"mrr@{k}"] = float(np.mean(reciprocal_ranks))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))

    return metrics


# ============================================================================
# 3. SPEAKER FLIP RATE (SFR) AUDIT
# ============================================================================

def compute_speaker_flip_rate(
    fga: FeatureGatingAgent,
    fusion: GatedFusion,
    classifier: VeracityClassifier,
    claim_embs: torch.Tensor,
    speaker_embs: torch.Tensor,
    date_embs: torch.Tensor,
    all_speaker_embs: dict,
    device: torch.device,
    num_perturbations: int = SFR_NUM_PERTURBATIONS,
    batch_size: int = 64,
) -> dict:
    """
    Compute Speaker Flip Rate: measure how often the model's prediction
    changes when the speaker identity is swapped.

    SFR = (1/M) Σ I(ŷ(q,s) ≠ ŷ(q,s'))

    Target: SFR < 3%

    Args:
        fga, fusion, classifier: Trained model components
        claim_embs, speaker_embs, date_embs: Test set embeddings
        all_speaker_embs: Dict of speaker name → embedding
        device: Compute device
        num_perturbations: Number of random speaker swaps per sample
        batch_size: Evaluation batch size

    Returns:
        dict: {sfr, flip_counts, total_tests, per_sample_flips}
    """
    fga.eval()
    classifier.eval()

    all_speakers = list(all_speaker_embs.keys())
    n_samples = len(claim_embs)
    flip_counts = np.zeros(n_samples)

    logging.info(f"Running SFR audit: {format_number(n_samples)} samples × {num_perturbations} perturbations")

    with torch.no_grad():
        # Get original predictions
        orig_preds = []
        for i in range(0, n_samples, batch_size):
            e_q = claim_embs[i:i+batch_size].to(device)
            e_s = speaker_embs[i:i+batch_size].to(device)
            e_t = date_embs[i:i+batch_size].to(device)

            alpha_q, alpha_s, alpha_t = fga(e_q, e_s, e_t)
            e_gated = fusion(e_q, e_s, e_t, alpha_q, alpha_s, alpha_t)
            logits = classifier(e_gated)
            orig_preds.append(logits.argmax(dim=-1).cpu())

        orig_preds = torch.cat(orig_preds)

        # Run perturbations
        for pert in tqdm(range(num_perturbations), desc="SFR perturbations"):
            # Random speaker swap for all samples
            random_speakers = random.choices(all_speakers, k=n_samples)
            cf_speaker_embs = torch.stack([all_speaker_embs[s] for s in random_speakers])

            cf_preds = []
            for i in range(0, n_samples, batch_size):
                e_q = claim_embs[i:i+batch_size].to(device)
                e_s_cf = cf_speaker_embs[i:i+batch_size].to(device)
                e_t = date_embs[i:i+batch_size].to(device)

                alpha_q, alpha_s, alpha_t = fga(e_q, e_s_cf, e_t)
                e_gated = fusion(e_q, e_s_cf, e_t, alpha_q, alpha_s, alpha_t)
                logits = classifier(e_gated)
                cf_preds.append(logits.argmax(dim=-1).cpu())

            cf_preds = torch.cat(cf_preds)

            # Count flips
            flipped = (orig_preds != cf_preds).numpy().astype(float)
            flip_counts += flipped

    # Compute SFR
    per_sample_flip_rate = flip_counts / num_perturbations
    sfr = float(np.mean(per_sample_flip_rate))

    result = {
        "sfr": sfr,
        "sfr_std": float(np.std(per_sample_flip_rate)),
        "total_samples": n_samples,
        "total_perturbations": num_perturbations,
        "total_flips": int(flip_counts.sum()),
        "total_tests": n_samples * num_perturbations,
        "target_met": sfr < SFR_TARGET,
    }

    if result["target_met"]:
        logging.info(f"\n  ✅ SFR = {sfr:.4f} ({sfr*100:.2f}%) — TARGET MET (< {SFR_TARGET*100:.0f}%)")
    else:
        logging.warning(f"\n  ⚠️  SFR = {sfr:.4f} ({sfr*100:.2f}%) — TARGET NOT MET (≥ {SFR_TARGET*100:.0f}%)")

    return result


# ============================================================================
# 4. ABLATION BASELINES
# ============================================================================

def run_bm25_baseline(
    train_claims: list, train_labels: np.ndarray,
    test_claims: list, test_labels: np.ndarray,
) -> dict:
    """
    BM25 lexical baseline: classify test claims by nearest BM25 match.

    Args:
        train_claims, train_labels: Training data
        test_claims, test_labels: Test data

    Returns:
        dict: Classification metrics
    """
    from rank_bm25 import BM25Okapi

    with timer("BM25 baseline"):
        # Tokenize
        train_tokenized = [c.lower().split() for c in train_claims]
        bm25 = BM25Okapi(train_tokenized)

        predictions = []
        for claim in tqdm(test_claims, desc="BM25 retrieval"):
            query_tokens = claim.lower().split()
            scores = bm25.get_scores(query_tokens)
            best_match_idx = scores.argmax()
            predictions.append(train_labels[best_match_idx])

        predictions = np.array(predictions)
        metrics = compute_classification_metrics(test_labels, predictions)
        metrics["method"] = "BM25"

    return metrics


def run_zero_shot_baseline(
    test_claims: list, test_labels: np.ndarray,
    encoder: FrozenEncoder, device: torch.device,
    train_embs: np.ndarray = None, train_labels: np.ndarray = None,
) -> dict:
    """
    Zero-shot encoder baseline: nearest-neighbor classification using
    raw frozen encoder embeddings (no gating).

    Args:
        test_claims: Test claim texts
        test_labels: Test labels
        encoder: Frozen encoder
        device: Compute device
        train_embs: Pre-computed train embeddings
        train_labels: Train labels

    Returns:
        dict: Classification + retrieval metrics
    """
    import faiss

    with timer("Zero-shot encoder baseline"):
        # Encode test claims
        test_embs = []
        for i in range(0, len(test_claims), 64):
            batch = test_claims[i:i+64]
            embs = encoder.encode(batch, device).cpu().numpy()
            test_embs.append(embs)
        test_embs = np.concatenate(test_embs, axis=0)

        if train_embs is not None:
            # Build index from training embeddings
            index = build_faiss_index(train_embs.copy())
            faiss.normalize_L2(test_embs)
            distances, indices = index.search(test_embs, 1)

            predictions = train_labels[indices[:, 0]]
            metrics = compute_classification_metrics(test_labels, predictions)

            # Also compute retrieval metrics
            retrieval = compute_retrieval_metrics(
                test_embs, test_labels, index, train_labels
            )
            metrics.update(retrieval)
        else:
            metrics = {}

        metrics["method"] = "Zero-Shot Encoder"

    return metrics


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_evaluation_results(
    cl_sdrg_metrics: dict,
    ablation_results: list,
    sfr_result: dict,
    save_dir: Path,
):
    """
    Generate publication-quality evaluation figures.

    Args:
        cl_sdrg_metrics: CL-SDRG model metrics
        ablation_results: List of ablation baseline metrics
        sfr_result: SFR audit results
        save_dir: Directory to save figures
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Classification Comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("CL-SDRG Evaluation Results", fontsize=16, fontweight="bold")

    methods = ["CL-SDRG"] + [r.get("method", "Baseline") for r in ablation_results]
    all_results = [cl_sdrg_metrics] + ablation_results

    metric_names = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    metric_labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for i, (method, result) in enumerate(zip(methods, all_results)):
        values = [result.get(m, 0) for m in metric_names]
        axes[0].bar(x + i * width, values, width, label=method, color=colors[i])

    axes[0].set_xticks(x + width * (len(methods) - 1) / 2)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Metrics")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    # ── SFR comparison ──
    sfr_values = [sfr_result["sfr"]]
    sfr_labels = ["CL-SDRG"]

    bars = axes[1].bar(sfr_labels, [v * 100 for v in sfr_values], color=["#2ecc71"])
    axes[1].axhline(y=SFR_TARGET * 100, color="red", linestyle="--", linewidth=2, label=f"Target: {SFR_TARGET*100:.0f}%")
    axes[1].set_ylabel("Speaker Flip Rate (%)")
    axes[1].set_title("Shortcut Bias Audit (SFR)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, sfr_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val*100:.2f}%", ha="center", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(str(save_dir / "evaluation_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"📊 Evaluation results saved: {save_dir / 'evaluation_results.png'}")

    # ── Figure 2: Confusion Matrix ──
    if "confusion_matrix" in cl_sdrg_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = cl_sdrg_metrics["confusion_matrix"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("CL-SDRG Confusion Matrix", fontsize=14, fontweight="bold")

        labels = [ID2LABEL.get(i, str(i)) for i in range(NUM_CLASSES)]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(str(save_dir / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"📊 Confusion matrix saved: {save_dir / 'confusion_matrix.png'}")


def print_results_table(
    cl_sdrg_metrics: dict,
    ablation_results: list,
    sfr_result: dict,
):
    """Print a formatted results table for paper inclusion."""
    logging.info("\n" + "=" * 90)
    logging.info("  BENCHMARK RESULTS TABLE")
    logging.info("=" * 90)

    header = f"{'Method':<30} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'SFR':>8}"
    logging.info(header)
    logging.info("-" * 90)

    # CL-SDRG
    logging.info(
        f"{'CL-SDRG (Ours)':<30} "
        f"{cl_sdrg_metrics.get('accuracy', 0):.4f}   "
        f"{cl_sdrg_metrics.get('macro_f1', 0):.4f}   "
        f"{cl_sdrg_metrics.get('macro_precision', 0):.4f}   "
        f"{cl_sdrg_metrics.get('macro_recall', 0):.4f}   "
        f"{sfr_result['sfr']:.4f}"
    )

    # Baselines
    for result in ablation_results:
        method = result.get("method", "Baseline")
        logging.info(
            f"{method:<30} "
            f"{result.get('accuracy', 0):.4f}   "
            f"{result.get('macro_f1', 0):.4f}   "
            f"{result.get('macro_precision', 0):.4f}   "
            f"{result.get('macro_recall', 0):.4f}   "
            f"{'N/A':>6}"
        )

    logging.info("-" * 90)

    # Retrieval metrics
    logging.info("\n  RETRIEVAL METRICS (CL-SDRG)")
    logging.info("-" * 50)
    for k in RETRIEVAL_K_VALUES:
        recall = cl_sdrg_metrics.get(f"recall@{k}", 0)
        mrr = cl_sdrg_metrics.get(f"mrr@{k}", 0)
        ndcg = cl_sdrg_metrics.get(f"ndcg@{k}", 0)
        logging.info(f"  K={k:<4}  Recall: {recall:.4f}  MRR: {mrr:.4f}  nDCG: {ndcg:.4f}")

    logging.info("=" * 90)


# ============================================================================
# 6. MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Execute the complete Phase 4 evaluation pipeline."""

    # ── Setup ──
    init_directories()
    setup_logging(log_dir=LOG_DIR, script_name="04_evaluation")
    set_seed(RANDOM_SEED)

    logging.info("=" * 70)
    logging.info("  CL-SDRG Phase 4: Evaluation & Ablation Studies")
    logging.info("=" * 70)

    device = get_device()

    # ── Load test data ──
    test_csv = PROCESSED_DATA_DIR / "fci_test.csv"
    train_csv = PROCESSED_DATA_DIR / "fci_train.csv"

    if not test_csv.exists():
        logging.error(f"Test data not found: {test_csv}")
        logging.error("Run Script 01 first.")
        sys.exit(1)

    with timer("Loading evaluation data"):
        test_df = pd.read_csv(str(test_csv))
        train_df = pd.read_csv(str(train_csv))

        test_df["itemReviewed.author.name"] = test_df["itemReviewed.author.name"].fillna("Unknown")
        train_df["itemReviewed.author.name"] = train_df["itemReviewed.author.name"].fillna("Unknown")
        test_df = test_df.dropna(subset=["claimReviewed", "label_id"])
        train_df = train_df.dropna(subset=["claimReviewed", "label_id"])
        test_df["label_id"] = test_df["label_id"].astype(int)
        train_df["label_id"] = train_df["label_id"].astype(int)

        logging.info(f"Test samples:  {format_number(len(test_df))}")
        logging.info(f"Train samples: {format_number(len(train_df))}")

    # ── Load frozen encoder ──
    with timer("Loading encoder"):
        encoder = FrozenEncoder(BASE_ENCODER_NAME)
        encoder.to(device)

    # ── Pre-compute embeddings ──
    with timer("Pre-computing test embeddings"):
        def encode_batch(texts, desc):
            embs = []
            for i in tqdm(range(0, len(texts), 64), desc=desc):
                batch = texts[i:i+64]
                e = encoder.encode(batch, device).cpu()
                embs.append(e)
            return torch.cat(embs, dim=0)

        test_claims = test_df["claimReviewed"].tolist()
        test_speakers = test_df["itemReviewed.author.name"].tolist()
        test_dates = test_df["datePublished"].astype(str).tolist()
        test_labels = test_df["label_id"].values

        test_claim_embs = encode_batch(test_claims, "Test claims")
        test_speaker_embs = encode_batch(test_speakers, "Test speakers")
        test_date_embs = encode_batch(test_dates, "Test dates")

    # ── Build speaker embedding map for SFR ──
    with timer("Building speaker embedding map"):
        all_speakers = list(set(test_speakers + train_df["itemReviewed.author.name"].tolist()))
        # Use a representative subset if too many
        if len(all_speakers) > 5000:
            all_speakers = random.sample(all_speakers, 5000)

        speaker_emb_map = {}
        for i in range(0, len(all_speakers), 64):
            batch = all_speakers[i:i+64]
            embs = encoder.encode(batch, device).cpu()
            for sp, emb in zip(batch, embs):
                speaker_emb_map[sp] = emb

    # ── Pre-compute train embeddings for baselines ──
    with timer("Pre-computing train embeddings"):
        train_claims = train_df["claimReviewed"].tolist()
        train_labels_arr = train_df["label_id"].values
        train_claim_embs = encode_batch(train_claims, "Train claims")

    # Free encoder VRAM
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load trained model ──
    with timer("Loading trained model"):
        # Find latest checkpoint
        ckpt_files = sorted(CHECKPOINT_DIR.glob("cl_sdrg_epoch_*.pt"))
        if not ckpt_files:
            logging.error(f"No checkpoints found in {CHECKPOINT_DIR}")
            logging.error("Run Script 03 first to train the model.")
            sys.exit(1)

        latest_ckpt = ckpt_files[-1]
        checkpoint = load_checkpoint(latest_ckpt, device)

        fga = FeatureGatingAgent().to(device)
        fusion = GatedFusion().to(device)
        classifier = VeracityClassifier().to(device)

        fga.load_state_dict(checkpoint["model_state_dict"]["fga"])
        classifier.load_state_dict(checkpoint["model_state_dict"]["classifier"])

        fga.eval()
        classifier.eval()

    # ── Step 4.1: Classification evaluation ──
    logging.info("\n" + "=" * 70)
    logging.info("  Step 4.1: Classification Evaluation")
    logging.info("=" * 70)

    with timer("CL-SDRG classification"):
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(test_claim_embs), PHYSICAL_BATCH_SIZE):
                e_q = test_claim_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)
                e_s = test_speaker_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)
                e_t = test_date_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)

                alpha_q, alpha_s, alpha_t = fga(e_q, e_s, e_t)
                e_gated = fusion(e_q, e_s, e_t, alpha_q, alpha_s, alpha_t)
                logits = classifier(e_gated)

                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

    cl_sdrg_metrics = compute_classification_metrics(test_labels, all_preds, all_probs)
    logging.info(f"\n{cl_sdrg_metrics['report']}")

    # ── Retrieval metrics ──
    logging.info("\n  Computing retrieval metrics...")
    import faiss

    # Build index from gated test embeddings
    with torch.no_grad():
        gated_test_embs = []
        for i in range(0, len(test_claim_embs), PHYSICAL_BATCH_SIZE):
            e_q = test_claim_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)
            e_s = test_speaker_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)
            e_t = test_date_embs[i:i+PHYSICAL_BATCH_SIZE].to(device)

            alpha_q, alpha_s, alpha_t = fga(e_q, e_s, e_t)
            e_gated = fusion(e_q, e_s, e_t, alpha_q, alpha_s, alpha_t)
            gated_test_embs.append(e_gated.cpu().numpy())

    gated_test_embs = np.concatenate(gated_test_embs, axis=0).astype(np.float32)
    faiss_index = build_faiss_index(gated_test_embs.copy())
    retrieval_metrics = compute_retrieval_metrics(
        gated_test_embs, test_labels, faiss_index, test_labels
    )
    cl_sdrg_metrics.update(retrieval_metrics)

    # ── Step 4.2: SFR Audit ──
    logging.info("\n" + "=" * 70)
    logging.info("  Step 4.2: Speaker Flip Rate (SFR) Audit")
    logging.info("=" * 70)

    sfr_result = compute_speaker_flip_rate(
        fga, fusion, classifier,
        test_claim_embs, test_speaker_embs, test_date_embs,
        speaker_emb_map, device,
    )

    # ── Step 4.3: Ablation Baselines ──
    logging.info("\n" + "=" * 70)
    logging.info("  Step 4.3: Ablation Baselines")
    logging.info("=" * 70)

    ablation_results = []

    # Baseline 1: BM25
    try:
        bm25_metrics = run_bm25_baseline(
            train_claims, train_labels_arr,
            test_claims, test_labels,
        )
        ablation_results.append(bm25_metrics)
        logging.info(f"  BM25:      Acc={bm25_metrics['accuracy']:.4f}, F1={bm25_metrics['macro_f1']:.4f}")
    except ImportError:
        logging.warning("  BM25 baseline skipped (rank_bm25 not installed)")

    # Baseline 3: Zero-Shot Encoder (kNN with raw embeddings)
    zero_shot_metrics = {
        "method": "Zero-Shot Encoder",
    }
    train_embs_np = train_claim_embs.numpy().astype(np.float32)
    test_embs_np = test_claim_embs.numpy().astype(np.float32)

    zs_index = build_faiss_index(train_embs_np.copy())
    faiss.normalize_L2(test_embs_np)
    distances, indices = zs_index.search(test_embs_np, 1)
    zs_preds = train_labels_arr[indices[:, 0]]
    zs_cls_metrics = compute_classification_metrics(test_labels, zs_preds)
    zero_shot_metrics.update(zs_cls_metrics)
    ablation_results.append(zero_shot_metrics)
    logging.info(f"  Zero-Shot: Acc={zero_shot_metrics['accuracy']:.4f}, F1={zero_shot_metrics['macro_f1']:.4f}")

    # ── Print results table ──
    print_results_table(cl_sdrg_metrics, ablation_results, sfr_result)

    # ── Plot results ──
    plot_evaluation_results(cl_sdrg_metrics, ablation_results, sfr_result, FIGURE_DIR)

    # ── Save results to CSV ──
    with timer("Saving results"):
        results_df = pd.DataFrame([
            {"method": "CL-SDRG", **{k: v for k, v in cl_sdrg_metrics.items()
                                      if isinstance(v, (int, float, str))}},
            *[{k: v for k, v in r.items() if isinstance(v, (int, float, str))}
              for r in ablation_results],
        ])
        results_path = OUTPUT_DIR / "benchmark_results.csv"
        results_df.to_csv(str(results_path), index=False)
        logging.info(f"📄 Results saved: {results_path}")

        # Save SFR result
        sfr_df = pd.DataFrame([sfr_result])
        sfr_df.to_csv(str(OUTPUT_DIR / "sfr_audit.csv"), index=False)

    # ── Final Summary ──
    logging.info("\n" + "=" * 70)
    logging.info("  PHASE 4 COMPLETE — Evaluation Summary")
    logging.info("=" * 70)
    logging.info(f"  CL-SDRG Accuracy:    {cl_sdrg_metrics['accuracy']:.4f}")
    logging.info(f"  CL-SDRG Macro-F1:    {cl_sdrg_metrics['macro_f1']:.4f}")
    logging.info(f"  Speaker Flip Rate:   {sfr_result['sfr']:.4f} ({'✅ PASS' if sfr_result['target_met'] else '⚠️  FAIL'})")
    logging.info(f"  Baselines evaluated: {len(ablation_results)}")
    logging.info("=" * 70)

    return {
        "cl_sdrg_metrics": cl_sdrg_metrics,
        "sfr_result": sfr_result,
        "ablation_results": ablation_results,
    }


if __name__ == "__main__":
    results = main()
