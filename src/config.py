"""
CL-SDRG Configuration Module
=============================
Centralized hyperparameters, paths, label mappings, and constants
shared across all pipeline scripts.

Target Hardware: Google Colab Free Tier (NVIDIA Tesla T4, 16 GB VRAM)
VRAM Budget:    < 2.5 GB peak allocation
"""

import os
from pathlib import Path

# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

# Root directory — resolved relative to this config file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "Fact Check Dataset"
CLAIM_REVIEW_CSV = DATA_DIR / "claim_review.csv"
MEDIA_REVIEW_CSV = DATA_DIR / "media_review.csv"

# Output directories (created at runtime)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# FastText model path (downloaded at runtime in Script 01)
FASTTEXT_MODEL_PATH = OUTPUT_DIR / "lid.176.bin"


# ============================================================================
# 2. RETAINED FIELDS & ZERO-LEAKAGE SANITIZATION
# ============================================================================

# Columns to RETAIN from raw claim_review.csv
RETAINED_COLUMNS = [
    "claimReviewed",                    # Claim text
    "itemReviewed.author.name",         # Speaker / author of the claim
    "datePublished",                    # Publication timestamp
    "reviewRating.alternateName",       # Verdict label
    "author.name",                      # Fact-checking organization
    "url",                              # Source URL (for provenance)
]

# Columns to PERMANENTLY DROP (evidence leakage risk)
LEAKAGE_COLUMNS = [
    "reviewRating.ratingExplanation",   # Fact-checker commentary → label leakage
    "reviewRating.ratingValue",         # Numeric rating → direct label proxy
]


# ============================================================================
# 3. LABEL MAPPING (Verdict Normalization)
# ============================================================================

# Map heterogeneous multilingual verdict strings to 3-class taxonomy
# The FCI corpus contains verdicts in English, Portuguese, Spanish, German,
# Hindi, Urdu, etc. — we map all to {TRUE, FALSE, MIXED}.

VERDICT_MAP = {
    # ── English ──
    "true":             "TRUE",
    "mostly true":      "TRUE",
    "correct":          "TRUE",
    "accurate":         "TRUE",
    "verified":         "TRUE",
    "fact":             "TRUE",
    "confirmed":        "TRUE",

    "false":            "FALSE",
    "mostly false":     "FALSE",
    "pants on fire":    "FALSE",
    "pants on fire!":   "FALSE",
    "fake":             "FALSE",
    "incorrect":        "FALSE",
    "not true":         "FALSE",
    "fabricated":       "FALSE",
    "debunked":         "FALSE",

    "half true":        "MIXED",
    "half-true":        "MIXED",
    "mixture":          "MIXED",
    "partially true":   "MIXED",
    "partially false":  "MIXED",
    "misleading":       "MIXED",
    "unverified":       "MIXED",
    "unproven":         "MIXED",
    "out of context":   "MIXED",
    "missing context":  "MIXED",
    "exaggerated":      "MIXED",
    "needs context":    "MIXED",
    "distorts the facts":"MIXED",

    # ── Portuguese ──
    "falso":            "FALSE",
    "verdadeiro":       "TRUE",
    "enganoso":         "MIXED",
    "impreciso":        "MIXED",
    "insustentável":    "FALSE",

    # ── Spanish ──
    "falsa":            "FALSE",
    "verdadero":        "TRUE",
    "engañoso":         "MIXED",
    "verdadera":        "TRUE",

    # ── German ──
    "falsch":           "FALSE",
    "richtig":          "TRUE",
    "teilweise falsch": "MIXED",
    "irreführend":      "MIXED",
    "unbelegt":         "MIXED",

    # ── Hindi / Urdu ──
    "झूठ":              "FALSE",
    "सच":               "TRUE",
    "भ्रामक":           "MIXED",
    "फर्जी":            "FALSE",
    "جھوٹ":             "FALSE",
    "سچ":               "TRUE",
    "گمراہ کن":         "MIXED",

    # ── Arabic (Discovered high-frequency: 23k+ claims) ──
    "خطأ":              "FALSE",
    "صحيح":             "TRUE",
    "مضلل":             "MIXED",

    # ── Turkish (Discovered high-frequency: 7k+ claims) ──
    "yanlış":           "FALSE",
    "doğru":            "TRUE",

    # ── French / Italian ──
    "faux":             "FALSE",
    "vrai":             "TRUE",
    "notizia falsa":    "FALSE",
    "fuori contesto":   "MIXED",

    # ── Polish ──
    "fałsz":            "FALSE",
    "fałsz.":           "FALSE",
    "prawda":           "TRUE",

    # ── Chinese ──
    "錯誤":             "FALSE",
    "真實":             "TRUE",

    # ── Additional nuanced English ──
    "partly false":     "MIXED",
}

# Class-index mapping for model training
LABEL2ID = {"TRUE": 0, "FALSE": 1, "MIXED": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)


# ============================================================================
# 4. LANGUAGE IDENTIFICATION
# ============================================================================

# Target South Asian language codes (ISO 639-1 via FastText lid.176)
TARGET_LANGUAGES = {"ur", "hi", "bn", "pa", "sd", "ta", "te", "ml", "mr", "gu", "ne", "si"}

# Primary low-resource focus languages
PRIMARY_FOCUS_LANGUAGES = {"ur", "hi", "bn"}

# Minimum confidence threshold for FastText LID predictions
LID_CONFIDENCE_THRESHOLD = 0.5

# Silver pair mining: maximum day gap for cross-lingual pairs
SILVER_PAIR_DAY_WINDOW = 3

# Minimum Urdu claims before triggering NLLB fallback translation
MIN_URDU_CLAIMS_THRESHOLD = 500


# ============================================================================
# 5. TIME-AWARE SPLIT BOUNDARIES
# ============================================================================

TRAIN_CUTOFF_DATE = "2023-12-31"      # Train: claims published ≤ this date
TEST_START_DATE = "2024-01-01"         # Test:  claims published ≥ this date


# ============================================================================
# 6. MODEL & ARCHITECTURE HYPERPARAMETERS
# ============================================================================

# Frozen base encoder
BASE_ENCODER_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768                   # Hidden size of mE5-base

# Feature Gating Agent (FGA)
FGA_HIDDEN_DIM = 256                  # Internal MLP hidden dimension
FGA_INPUT_DIM = EMBEDDING_DIM * 3     # Concatenation of [E_q; E_s; E_t]
FGA_NUM_GATES = 3                     # α_q, α_s, α_t

# Classifier head
CLASSIFIER_HIDDEN_DIM = 256
CLASSIFIER_DROPOUT = 0.1

# Maximum token length for encoder inputs
MAX_SEQ_LENGTH = 128


# ============================================================================
# 7. TRAINING HYPERPARAMETERS (REINFORCE)
# ============================================================================

# Reward weights
LAMBDA_ACCURACY = 0.6                 # λ₁ for R_acc
LAMBDA_CONSISTENCY = 0.4              # λ₂ for R_cons

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Batching
PHYSICAL_BATCH_SIZE = 16              # Fits in T4 VRAM
GRADIENT_ACCUMULATION_STEPS = 16      # Virtual batch = 16 × 16 = 256
VIRTUAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Training schedule
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1

# Mixed precision
USE_FP16 = True

# Baseline EMA coefficient for REINFORCE variance reduction
BASELINE_EMA_DECAY = 0.99

# Checkpoint frequency
CHECKPOINT_EVERY_N_EPOCHS = 2


# ============================================================================
# 8. EVALUATION HYPERPARAMETERS
# ============================================================================

# Retrieval metrics
RETRIEVAL_K_VALUES = [1, 5, 20]
FAISS_NPROBE = 10

# Speaker Flip Rate target
SFR_TARGET = 0.03                     # < 3%

# Number of counterfactual perturbations per sample during SFR audit
SFR_NUM_PERTURBATIONS = 10


# ============================================================================
# 9. REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42


# ============================================================================
# 10. UTILITY: Directory Initialization
# ============================================================================

def init_directories():
    """Create all output directories if they don't exist."""
    for dir_path in [OUTPUT_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
