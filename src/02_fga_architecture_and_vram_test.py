"""
CL-SDRG — Script 02: FGA Architecture & VRAM Dry-Run Test
==========================================================
Phase 2 of the CL-SDRG pipeline.

This script:
  1. Loads and freezes the multilingual-E5-base encoder backbone
  2. Constructs the Feature Gating Agent (FGA) MLP module
  3. Builds the gated fusion mechanism and veracity classifier head
  4. Implements the counterfactual perturbation generator
  5. Runs a forward-pass dry-run to validate VRAM stays < 2.5 GB
  6. Prints model parameter summary and memory profile

Architecture:
    Input → [Frozen mE5-base] → (E_q, E_s, E_t) → [FGA: Sigmoid Gates]
            → [Gated Fusion: α_q⊙E_q + α_s⊙E_s + α_t⊙E_t]
            → [Classifier Head] → Veracity Prediction

Target Environment: Google Colab T4 GPU (< 2.5 GB peak VRAM)

Usage (Colab):
    !pip install torch transformers
    %run src/02_fga_architecture_and_vram_test.py
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_ENCODER_NAME, EMBEDDING_DIM, MAX_SEQ_LENGTH,
    FGA_HIDDEN_DIM, FGA_INPUT_DIM, FGA_NUM_GATES,
    CLASSIFIER_HIDDEN_DIM, CLASSIFIER_DROPOUT, NUM_CLASSES,
    PHYSICAL_BATCH_SIZE, LOG_DIR, OUTPUT_DIR,
)
from src.config import init_directories
from src.utils import (
    set_seed, setup_logging, timer, get_device,
    get_gpu_memory_summary, reset_peak_memory, format_number,
)


# ============================================================================
# MODULE 1: FROZEN BASE ENCODER
# ============================================================================

class FrozenEncoder(nn.Module):
    """
    Frozen multilingual-E5-base encoder.

    All backbone parameters are locked (requires_grad=False) and the model
    runs in eval mode. Only used for feature extraction — zero trainable params.
    """

    def __init__(self, model_name: str = BASE_ENCODER_NAME):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze all backbone parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        self.embedding_dim = self.encoder.config.hidden_size
        logging.info(f"Loaded frozen encoder: {model_name}")
        logging.info(f"  Embedding dim: {self.embedding_dim}")
        logging.info(f"  Frozen params: {format_number(sum(p.numel() for p in self.encoder.parameters()))}")

    @torch.no_grad()
    def encode(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Encode a list of text strings into embeddings.

        Uses mean pooling over the last hidden state (following E5 convention).

        Args:
            texts: List of input strings
            device: Target device

        Returns:
            torch.Tensor: Shape (batch_size, embedding_dim)
        """
        # E5 models expect "query: " or "passage: " prefix
        prefixed = [f"query: {t}" for t in texts]

        tokens = self.tokenizer(
            prefixed,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.encoder(**tokens)

        # Mean pooling (respecting attention mask)
        attention_mask = tokens["attention_mask"].unsqueeze(-1).float()
        embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        embeddings = embeddings / attention_mask.sum(dim=1).clamp(min=1e-9)

        return embeddings


# ============================================================================
# MODULE 2: FEATURE GATING AGENT (FGA)
# ============================================================================

class FeatureGatingAgent(nn.Module):
    """
    Lightweight MLP that produces sigmoid gating weights for each feature stream.

    Architecture:
        [E_q; E_s; E_t] → Linear → ReLU → Linear → Sigmoid → (α_q, α_s, α_t)

    The gating weights are per-dimension scalars ∈ [0, 1], allowing the agent
    to suppress or amplify individual embedding dimensions based on their
    relevance to veracity (vs. shortcut bias).

    Trainable parameters: ~1.5M (< 1.5% of total model size)
    """

    def __init__(
        self,
        input_dim: int = FGA_INPUT_DIM,      # 768 * 3 = 2304
        hidden_dim: int = FGA_HIDDEN_DIM,     # 256
        embedding_dim: int = EMBEDDING_DIM,   # 768
        num_gates: int = FGA_NUM_GATES,       # 3
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_gates = num_gates

        # MLP: concat input → hidden → gate weights
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim * num_gates),
            nn.Sigmoid(),
        )

        # Log trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"FGA Module initialized:")
        logging.info(f"  Input dim:   {input_dim}")
        logging.info(f"  Hidden dim:  {hidden_dim}")
        logging.info(f"  Trainable:   {format_number(trainable)} parameters")

    def forward(
        self, e_q: torch.Tensor, e_s: torch.Tensor, e_t: torch.Tensor
    ) -> tuple:
        """
        Compute gating weights for the three feature streams.

        Args:
            e_q: Claim embeddings (batch, embedding_dim)
            e_s: Speaker embeddings (batch, embedding_dim)
            e_t: Temporal embeddings (batch, embedding_dim)

        Returns:
            tuple: (alpha_q, alpha_s, alpha_t), each of shape (batch, embedding_dim)
        """
        # Concatenate all feature streams
        concat = torch.cat([e_q, e_s, e_t], dim=-1)  # (batch, 3*embedding_dim)

        # Compute gating weights
        gates = self.gate_network(concat)  # (batch, 3*embedding_dim)

        # Split into per-stream gates
        alpha_q, alpha_s, alpha_t = gates.split(self.embedding_dim, dim=-1)

        return alpha_q, alpha_s, alpha_t


# ============================================================================
# MODULE 3: GATED FEATURE FUSION
# ============================================================================

class GatedFusion(nn.Module):
    """
    Fuse the three feature streams using the learned gating weights.

    E_gated = α_q ⊙ E_q + α_s ⊙ E_s + α_t ⊙ E_t

    This is a parameter-free module; the gating weights come from the FGA.
    """

    def forward(
        self,
        e_q: torch.Tensor, e_s: torch.Tensor, e_t: torch.Tensor,
        alpha_q: torch.Tensor, alpha_s: torch.Tensor, alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply element-wise gating and sum.

        Returns:
            torch.Tensor: Gated fused embedding (batch, embedding_dim)
        """
        return alpha_q * e_q + alpha_s * e_s + alpha_t * e_t


# ============================================================================
# MODULE 4: VERACITY CLASSIFIER HEAD
# ============================================================================

class VeracityClassifier(nn.Module):
    """
    Classification head: gated embedding → veracity prediction.

    Architecture:
        E_gated → LayerNorm → Linear → ReLU → Dropout → Linear → logits
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = CLASSIFIER_DROPOUT,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Classifier Head initialized:")
        logging.info(f"  Input dim:   {input_dim}")
        logging.info(f"  Classes:     {num_classes}")
        logging.info(f"  Trainable:   {format_number(trainable)} parameters")

    def forward(self, e_gated: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_gated: Gated fused embedding (batch, embedding_dim)

        Returns:
            torch.Tensor: Logits (batch, num_classes)
        """
        return self.classifier(e_gated)


# ============================================================================
# MODULE 5: COUNTERFACTUAL PERTURBATION GENERATOR
# ============================================================================

class CounterfactualPerturbation:
    """
    Generate counterfactual samples by replacing the speaker identity
    while keeping the claim text and label invariant.

    This is a data-level transform (not a nn.Module) that randomly
    samples a different speaker s' from the dataset for each claim.
    """

    def __init__(self, all_speakers: list):
        """
        Args:
            all_speakers: List of all unique speaker names in the dataset.
        """
        self.all_speakers = list(set(all_speakers))
        logging.info(f"Counterfactual Perturbation initialized:")
        logging.info(f"  Speaker pool size: {format_number(len(self.all_speakers))}")

    def perturb_speakers(self, original_speakers: list) -> list:
        """
        Replace each speaker with a randomly sampled different speaker.

        Args:
            original_speakers: List of original speaker strings

        Returns:
            list: Perturbed speaker strings (guaranteed different from original)
        """
        import random
        perturbed = []
        for speaker in original_speakers:
            # Sample a different speaker
            candidates = [s for s in self.all_speakers if s != speaker]
            if candidates:
                perturbed.append(random.choice(candidates))
            else:
                perturbed.append(speaker)  # Fallback if only one speaker
        return perturbed


# ============================================================================
# MODULE 6: FULL CL-SDRG MODEL (Assembled Pipeline)
# ============================================================================

class CLSDRG(nn.Module):
    """
    Complete CL-SDRG model combining all modules.

    Architecture flow:
        1. Frozen encoder extracts E_q, E_s, E_t
        2. FGA computes gating weights α_q, α_s, α_t
        3. Gated fusion produces E_gated
        4. Classifier head outputs veracity logits

    Only the FGA and classifier head are trainable (~1.5M params).
    """

    def __init__(self, encoder_name: str = BASE_ENCODER_NAME):
        super().__init__()

        self.encoder = FrozenEncoder(encoder_name)
        self.fga = FeatureGatingAgent()
        self.fusion = GatedFusion()
        self.classifier = VeracityClassifier()

    def encode_inputs(
        self, claims: list, speakers: list, dates: list, device: torch.device
    ) -> tuple:
        """
        Encode raw text inputs into embedding triplets.

        Args:
            claims: List of claim text strings
            speakers: List of speaker name strings
            dates: List of date strings (used as temporal context)
            device: Target device

        Returns:
            tuple: (E_q, E_s, E_t) — each (batch, embedding_dim)
        """
        e_q = self.encoder.encode(claims, device)
        e_s = self.encoder.encode(speakers, device)
        e_t = self.encoder.encode(dates, device)
        return e_q, e_s, e_t

    def forward_from_embeddings(
        self,
        e_q: torch.Tensor, e_s: torch.Tensor, e_t: torch.Tensor,
    ) -> tuple:
        """
        Forward pass from pre-computed embeddings.

        This is the main forward path during training (embeddings can be
        pre-computed and cached to avoid re-encoding every epoch).

        Args:
            e_q, e_s, e_t: Pre-computed embeddings

        Returns:
            tuple: (logits, gate_weights)
                - logits: (batch, num_classes)
                - gate_weights: tuple of (alpha_q, alpha_s, alpha_t)
        """
        alpha_q, alpha_s, alpha_t = self.fga(e_q, e_s, e_t)
        e_gated = self.fusion(e_q, e_s, e_t, alpha_q, alpha_s, alpha_t)
        logits = self.classifier(e_gated)

        return logits, (alpha_q, alpha_s, alpha_t)

    def forward(
        self, claims: list, speakers: list, dates: list, device: torch.device
    ) -> tuple:
        """
        Full forward pass from raw text inputs.

        Args:
            claims, speakers, dates: Lists of strings
            device: Target device

        Returns:
            tuple: (logits, gate_weights)
        """
        e_q, e_s, e_t = self.encode_inputs(claims, speakers, dates, device)
        return self.forward_from_embeddings(e_q, e_s, e_t)


# ============================================================================
# VRAM DRY-RUN TEST
# ============================================================================

def run_vram_test(device: torch.device):
    """
    Execute a complete forward-pass dry-run to measure peak VRAM usage.

    Target: < 2.5 GB peak allocation on T4.
    """
    logging.info("\n" + "=" * 70)
    logging.info("  VRAM Dry-Run Test")
    logging.info("=" * 70)

    reset_peak_memory()

    with timer("Model initialization"):
        model = CLSDRG()
        model.to(device)

    # ── Parameter summary ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logging.info(f"\n  Parameter Summary:")
    logging.info(f"    Total:     {format_number(total_params)}")
    logging.info(f"    Frozen:    {format_number(frozen_params)}")
    logging.info(f"    Trainable: {format_number(trainable_params)} ({trainable_params/total_params*100:.2f}%)")

    # ── Forward pass with dummy data ──
    batch_size = PHYSICAL_BATCH_SIZE
    dummy_claims = [f"This is a test claim number {i} about politics" for i in range(batch_size)]
    dummy_speakers = [f"Speaker {i}" for i in range(batch_size)]
    dummy_dates = [f"2023-01-{(i % 28) + 1:02d}" for i in range(batch_size)]

    with timer(f"Forward pass (batch_size={batch_size})"):
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits, gates = model(dummy_claims, dummy_speakers, dummy_dates, device)
            else:
                logits, gates = model(dummy_claims, dummy_speakers, dummy_dates, device)

    logging.info(f"\n  Output shapes:")
    logging.info(f"    Logits:  {logits.shape}")
    logging.info(f"    Gate α_q: {gates[0].shape}")
    logging.info(f"    Gate α_s: {gates[1].shape}")
    logging.info(f"    Gate α_t: {gates[2].shape}")

    # ── Memory report ──
    mem = get_gpu_memory_summary()
    if mem:
        logging.info(f"\n  GPU Memory Profile:")
        logging.info(f"    Allocated:      {mem['allocated_mb']:.1f} MB")
        logging.info(f"    Reserved:       {mem['reserved_mb']:.1f} MB")
        logging.info(f"    Peak Allocated: {mem['peak_allocated_mb']:.1f} MB")

        peak_gb = mem["peak_allocated_mb"] / 1024
        budget_gb = 2.5
        if peak_gb < budget_gb:
            logging.info(f"\n  ✅ VRAM TEST PASSED: {peak_gb:.2f} GB < {budget_gb} GB budget")
        else:
            logging.warning(f"\n  ⚠️  VRAM TEST FAILED: {peak_gb:.2f} GB ≥ {budget_gb} GB budget")
    else:
        logging.info("  (No GPU — VRAM test skipped, CPU-only run successful)")

    # ── Counterfactual perturbation test ──
    with timer("Counterfactual perturbation test"):
        perturbation = CounterfactualPerturbation(dummy_speakers)
        perturbed = perturbation.perturb_speakers(dummy_speakers)
        logging.info(f"  Original:  {dummy_speakers[:3]}")
        logging.info(f"  Perturbed: {perturbed[:3]}")

        # Verify invariant: perturbed ≠ original
        diff_count = sum(1 for o, p in zip(dummy_speakers, perturbed) if o != p)
        logging.info(f"  Changed: {diff_count}/{batch_size} speakers")

    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    init_directories()
    setup_logging(log_dir=LOG_DIR, script_name="02_architecture")
    set_seed(42)

    logging.info("=" * 70)
    logging.info("  CL-SDRG Phase 2: Architecture & VRAM Validation")
    logging.info("=" * 70)

    device = get_device()
    model = run_vram_test(device)

    logging.info("\n" + "=" * 70)
    logging.info("  PHASE 2 COMPLETE — Architecture validated")
    logging.info("=" * 70)

    return model


if __name__ == "__main__":
    model = main()
