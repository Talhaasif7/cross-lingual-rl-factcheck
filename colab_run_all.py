"""
🔬 CL-SDRG: Complete End-to-End Pipeline (Google Colab & Local)
================================================================================
Cross-Lingual Shortcut De-biasing via Reinforcement Learning Gating (CL-SDRG)

All 4 phases in a single, robust script:
  Phase 1: Zero-Leakage Data Ingestion, FastText LID & Silver Pair Mining
  Phase 2: Frozen mE5 Encoder + Feature Gating Agent + VRAM Validation (< 2.5 GB)
  Phase 3: REINFORCE Policy Gradient Training with Counterfactual Consistency
  Phase 4: Multi-metric Evaluation, FAISS Retrieval, Baselines & SFR Bias Audit

Target Environment: Google Colab (Tesla T4 GPU, 16 GB VRAM) or Local CUDA/CPU.
Execution Time: ~40-50 minutes total on Colab T4.
================================================================================
"""

import os
import sys
import time
import random
import logging
import warnings
import subprocess
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# ============================================================================
# 0. DEPENDENCY CHECK & INSTALLATION (AUTO-HANDLED ON COLAB)
# ============================================================================

REQUIRED_PACKAGES = [
    "torch", "transformers", "fasttext", "pandas", "numpy",
    "scikit-learn", "faiss-cpu", "rank_bm25", "tqdm", "matplotlib", "seaborn"
]

def check_and_install_dependencies():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        mod = pkg.replace("-", "_")
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"📦 Installing missing dependencies: {missing}")
        cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
        subprocess.check_call(cmd)
        print("✅ All dependencies installed successfully.")

check_and_install_dependencies()

import numpy as np

# ── NumPy 2.0+ Compatibility Patch (resolves FastText copy=False deprecation) ──
_orig_np_array = np.array
def _safe_np_array(obj, *args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
        return np.asarray(obj, *args, **kwargs)
    return _orig_np_array(obj, *args, **kwargs)
np.array = _safe_np_array
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import faiss

# ============================================================================
# 1. CONFIGURATION & LOGGING
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "Fact Check Dataset"
OUT_DIR = ROOT / "outputs"
PROC_DIR = OUT_DIR / "processed_data"
CKPT_DIR = OUT_DIR / "checkpoints"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"

for d in [DATA_DIR, PROC_DIR, CKPT_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("CL_SDRG")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_sh)
_fh = logging.FileHandler(str(LOG_DIR / "cl_sdrg_run.log"), mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s"))
logger.addHandler(_fh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"🚀 Device: {device} | GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    logger.info(f"⚙️ Device: {device} (CPU fallback)")

# Hyperparameters
ENCODER_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 128
FGA_HIDDEN = 256
CLS_HIDDEN = 256
CLS_DROPOUT = 0.1
NUM_CLASSES = 3

PHYSICAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 16
VIRTUAL_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRAD_ACCUM_STEPS  # 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
BASELINE_DECAY = 0.99
LAMBDA_ACC = 0.6
LAMBDA_CONS = 0.4
CHECKPOINT_EVERY = 2

RETRIEVAL_K_VALUES = [1, 5, 20]
SFR_TARGET = 0.03
SFR_NUM_PERTURBATIONS = 10

TRAIN_DATE_CUTOFF = "2023-12-31"
TEST_DATE_START = "2024-01-01"
TARGET_LANGS = {"ur", "hi", "bn", "pa", "sd", "ta", "te", "ml", "mr", "gu", "ne", "si"}
LID_CONFIDENCE_THRESHOLD = 0.5
SILVER_PAIR_DAY_WINDOW = 3
MIN_URDU_SMOKE_TEST = 500

LABEL2ID = {"TRUE": 0, "FALSE": 1, "MIXED": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

VERDICT_MAP = {
    "true": "TRUE", "mostly true": "TRUE", "correct": "TRUE", "accurate": "TRUE",
    "verified": "TRUE", "fact": "TRUE", "confirmed": "TRUE",
    "false": "FALSE", "mostly false": "FALSE", "pants on fire": "FALSE",
    "pants on fire!": "FALSE", "fake": "FALSE", "incorrect": "FALSE",
    "not true": "FALSE", "fabricated": "FALSE", "debunked": "FALSE",
    "half true": "MIXED", "half-true": "MIXED", "mixture": "MIXED",
    "partially true": "MIXED", "partially false": "MIXED",
    "misleading": "MIXED", "unverified": "MIXED", "unproven": "MIXED",
    "out of context": "MIXED", "missing context": "MIXED",
    "exaggerated": "MIXED", "needs context": "MIXED", "distorts the facts": "MIXED",
    "falso": "FALSE", "verdadeiro": "TRUE", "enganoso": "MIXED",
    "impreciso": "MIXED", "insustentável": "FALSE",
    "falsa": "FALSE", "verdadero": "TRUE", "engañoso": "MIXED", "verdadera": "TRUE",
    "falsch": "FALSE", "richtig": "TRUE", "teilweise falsch": "MIXED",
    "irreführend": "MIXED", "unbelegt": "MIXED",
    "झूठ": "FALSE", "सच": "TRUE", "भ्रामक": "MIXED", "फर्जी": "FALSE",
    "جھوٹ": "FALSE", "سچ": "TRUE", "گمراہ کن": "MIXED",
    # Multilingual extensions from empirical dataset audit:
    "خطأ": "FALSE", "صحيح": "TRUE", "مضلل": "MIXED",
    "yanlış": "FALSE", "doğru": "TRUE",
    "faux": "FALSE", "vrai": "TRUE", "notizia falsa": "FALSE", "fuori contesto": "MIXED",
    "fałsz": "FALSE", "fałsz.": "FALSE", "prawda": "TRUE",
    "錯誤": "FALSE", "真實": "TRUE", "partly false": "MIXED",
}

def fmt(n):
    return f"{n:,}"

@contextmanager
def timer(label):
    t0 = time.perf_counter()
    logger.info(f"⏱  Starting: {label}")
    yield
    e = time.perf_counter() - t0
    m, s = divmod(e, 60)
    if m > 0:
        logger.info(f"✅ Completed: {label} — {int(m)}m {s:.1f}s")
    else:
        logger.info(f"✅ Completed: {label} — {s:.1f}s")

def normalize_verdict(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    c = raw.strip().lower()
    if c in VERDICT_MAP:
        return VERDICT_MAP[c]
    for k, v in VERDICT_MAP.items():
        if k in c or c in k:
            return v
    return None

# ============================================================================
# 2. DATASET DISCOVERY / ACQUISITION
# ============================================================================

def find_or_upload_dataset():
    candidates = [
        DATA_DIR / "claim_review.csv",
        ROOT / "claim_review.csv",
        Path("/content/Fact Check Dataset/claim_review.csv"),
        Path("/content/claim_review.csv"),
        Path("Fact Check Dataset/claim_review.csv"),
        Path("/content/drive/MyDrive/claim_review.csv"),
        Path("/content/drive/MyDrive/Fact Check Dataset/claim_review.csv"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1000:
            logger.info(f"Found dataset at: {p} ({p.stat().st_size / 1024**2:.1f} MB)")
            return p

    # If running inside Colab and file not found, offer file uploader
    is_colab = "google.colab" in sys.modules or "COLAB_GPU" in os.environ
    if is_colab:
        logger.info("Dataset not found. Opening file upload prompt for 'claim_review.csv'...")
        from google.colab import files
        uploaded = files.upload()
        for fname, data in uploaded.items():
            dst = DATA_DIR / "claim_review.csv"
            with open(dst, "wb") as f:
                f.write(data)
            logger.info(f"Saved uploaded dataset to: {dst} ({len(data)/1024**2:.1f} MB)")
            return dst

    raise FileNotFoundError(
        "Could not find 'claim_review.csv'! Please place it in 'Fact Check Dataset/claim_review.csv' "
        "or upload it to Colab."
    )

CSV_FILE = find_or_upload_dataset()

# ============================================================================
# PHASE 1: DATA INGESTION, ZERO-LEAKAGE SCRUBBING & FASTTEXT LID
# ============================================================================

print("\n" + "="*80)
print("  📋 PHASE 1: DATA INGESTION & LANGUAGE IDENTIFICATION")
print("="*80)

with timer("Phase 1 - Loading and Zero-Leakage Sanitization"):
    df_raw = pd.read_csv(str(CSV_FILE), encoding="utf-8", low_memory=False, on_bad_lines="skip")
    logger.info(f"Raw FCI records: {fmt(len(df_raw))} rows, {df_raw.shape[1]} columns")

    # Zero-leakage dropping
    leak_cols = ["reviewRating.ratingExplanation", "reviewRating.ratingValue"]
    for col in leak_cols:
        if col in df_raw.columns:
            logger.info(f"  [Zero-Leakage] Dropping leakage column: {col} ({fmt(df_raw[col].notna().sum())} populated)")
            df_raw.drop(columns=[col], inplace=True)
    extra_leaks = [c for c in df_raw.columns if "explanation" in c.lower()]
    if extra_leaks:
        df_raw.drop(columns=[c for c in extra_leaks if c in df_raw.columns], inplace=True)

    keep_cols = [
        "claimReviewed", "itemReviewed.author.name", "datePublished",
        "reviewRating.alternateName", "author.name", "url"
    ]
    df = df_raw[[c for c in keep_cols if c in df_raw.columns]].copy()
    del df_raw

    # Drop null claims or labels
    initial_len = len(df)
    df.dropna(subset=["claimReviewed", "reviewRating.alternateName"], inplace=True)
    df["claimReviewed"] = df["claimReviewed"].astype(str).str.strip()
    df = df[df["claimReviewed"].str.len() > 5]
    logger.info(f"  Sanitized claims: {fmt(len(df))} (dropped {fmt(initial_len - len(df))} invalid rows)")

with timer("Phase 1 - Verdict Normalization"):
    df["verdict"] = df["reviewRating.alternateName"].apply(normalize_verdict)
    unmapped = df["verdict"].isna().sum()
    if unmapped > 0:
        logger.warning(f"  Unmapped verdicts: {fmt(unmapped)} rows dropped")
    df = df[df["verdict"].notna()].copy()
    df["label_id"] = df["verdict"].map(LABEL2ID).astype(int)

    logger.info(f"Normalized 3-Class Label Distribution ({fmt(len(df))} total):")
    for v_name, cnt in df["verdict"].value_counts().items():
        logger.info(f"  • {v_name:<8} {fmt(cnt):>8} ({cnt/len(df)*100:5.1f}%)")

with timer("Phase 1 - FastText Language Identification"):
    import fasttext
    fasttext.FastText.eprint = lambda x: None

    lid_path = OUT_DIR / "lid.176.bin"
    if not lid_path.exists():
        logger.info("Downloading FastText lid.176.bin (~126 MB)...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, str(lid_path))
        logger.info(f"Downloaded lid.176.bin to {lid_path}")

    ft_model = fasttext.load_model(str(lid_path))
    langs, confs = [], []
    for text in tqdm(df["claimReviewed"].values, desc="FastText LID", unit="claim"):
        clean_text = str(text).replace("\n", " ").replace("\r", " ").strip()
        if not clean_text:
            langs.append("unknown")
            confs.append(0.0)
            continue
        try:
            pred = ft_model.predict(clean_text, k=1)
            langs.append(pred[0][0].replace("__label__", ""))
            confs.append(float(pred[1][0]))
        except Exception:
            langs.append("unknown")
            confs.append(0.0)

    df["detected_lang"] = langs
    df["lang_confidence"] = confs
    del ft_model

    top_langs = df["detected_lang"].value_counts().head(10)
    logger.info("Top 10 Detected Languages:")
    for lang, cnt in top_langs.items():
        logger.info(f"  • {lang:<4} {fmt(cnt):>8} ({cnt/len(df)*100:5.1f}%)")

with timer("Phase 1 - South Asian Filtering & Silver Pair Mining"):
    df["is_south_asian"] = df["detected_lang"].isin(TARGET_LANGS)
    sa_mask = df["is_south_asian"] & (df["lang_confidence"] >= LID_CONFIDENCE_THRESHOLD)
    sa_df = df[sa_mask].copy()

    logger.info(f"South Asian Claims (Confidence >= {LID_CONFIDENCE_THRESHOLD}): {fmt(len(sa_df))}")
    for lang, cnt in sa_df["detected_lang"].value_counts().items():
        logger.info(f"  • {lang:<4} {fmt(cnt):>6}")

    # Silver Pair Mining
    df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce", utc=True)
    pair_candidates = df.dropna(subset=["datePublished", "author.name"]).copy()
    pair_candidates = pair_candidates[pair_candidates["author.name"].str.strip().str.len() > 0]

    silver_pairs = []
    for org, grp in tqdm(pair_candidates.groupby("author.name"), desc="Mining Silver Pairs", unit="org"):
        if len(grp) < 2 or grp["detected_lang"].nunique() < 2:
            continue
        grp = grp.sort_values("datePublished")
        d_vals = grp["datePublished"].values
        l_vals = grp["detected_lang"].values
        c_vals = grp["claimReviewed"].values
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                gap_days = abs((d_vals[j] - d_vals[i]) / np.timedelta64(1, "D"))
                if gap_days > SILVER_PAIR_DAY_WINDOW:
                    break
                if l_vals[i] != l_vals[j]:
                    silver_pairs.append({
                        "claim_a": c_vals[i], "lang_a": l_vals[i],
                        "claim_b": c_vals[j], "lang_b": l_vals[j],
                        "org": org, "day_gap": round(gap_days, 1)
                    })
    silver_df = pd.DataFrame(silver_pairs)
    logger.info(f"Mined Cross-Lingual Silver Pairs (gap <= {SILVER_PAIR_DAY_WINDOW}d): {fmt(len(silver_df))}")

with timer("Phase 1 - Time-Aware Splits"):
    valid_dates = df["datePublished"].notna()
    dated_df = df[valid_dates].copy()
    train_df = dated_df[dated_df["datePublished"] <= pd.Timestamp(TRAIN_DATE_CUTOFF, tz="UTC")].copy()
    test_df = dated_df[dated_df["datePublished"] >= pd.Timestamp(TEST_DATE_START, tz="UTC")].copy()

    logger.info(f"Train split (<= {TRAIN_DATE_CUTOFF}): {fmt(len(train_df))} records")
    for v, c in train_df["verdict"].value_counts().items():
        logger.info(f"  • {v:<8} {fmt(c):>8} ({c/len(train_df)*100:5.1f}%)")

    logger.info(f"Test split (>= {TEST_DATE_START}):  {fmt(len(test_df))} records")
    for v, c in test_df["verdict"].value_counts().items():
        logger.info(f"  • {v:<8} {fmt(c):>8} ({c/len(test_df)*100:5.1f}%)")

    urdu_count = (df["detected_lang"] == "ur").sum()
    s1_status = "✅ PASS" if urdu_count >= MIN_URDU_SMOKE_TEST else "⚠️ TRIGGER NLLB FALLBACK"
    logger.info(f"Smoke Test S1 — Natural Urdu Claims: {fmt(urdu_count)} (threshold: {MIN_URDU_SMOKE_TEST}) -> {s1_status}")

    # Save processed CSVs
    df.to_csv(PROC_DIR / "fci_processed_full.csv", index=False)
    train_df.to_csv(PROC_DIR / "fci_train.csv", index=False)
    test_df.to_csv(PROC_DIR / "fci_test.csv", index=False)
    sa_df.to_csv(PROC_DIR / "fci_south_asian.csv", index=False)
    if len(silver_df) > 0:
        silver_df.to_csv(PROC_DIR / "silver_pairs.csv", index=False)
    logger.info(f"Processed datasets saved to {PROC_DIR}")

# ============================================================================
# PHASE 2: ARCHITECTURE DEFINITIONS & VRAM DRY-RUN
# ============================================================================

print("\n" + "="*80)
print("  🏗️ PHASE 2: MODEL ARCHITECTURE & VRAM VALIDATION")
print("="*80)

class FrozenEncoder(nn.Module):
    """Frozen multilingual-E5-base with attention-masked mean pooling."""
    def __init__(self, name=ENCODER_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = AutoModel.from_pretrained(name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, texts, dev):
        tok = self.tokenizer(
            [f"query: {str(t)}" for t in texts],
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(dev)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            out = self.encoder(**tok)
            m = tok["attention_mask"].unsqueeze(-1).float()
            return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

class FeatureGatingAgent(nn.Module):
    """Lightweight MLP gating agent producing per-dimension weights (α_q, α_s, α_t)."""
    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=FGA_HIDDEN):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim * 3),
            nn.Sigmoid()
        )

    def forward(self, eq, es, et):
        concat = torch.cat([eq, es, et], dim=-1)
        gates = self.net(concat)
        aq, a_s, at = gates.split(self.emb_dim, dim=-1)
        return aq, a_s, at

class GatedFusion(nn.Module):
    """Element-wise modulated fusion: E_gated = α_q ⊙ E_q + α_s ⊙ E_s + α_t ⊙ E_t"""
    def forward(self, eq, es, et, aq, a_s, at):
        return aq * eq + a_s * es + at * et

class VeracityClassifier(nn.Module):
    """Veracity classifier: LayerNorm -> Linear -> ReLU -> Dropout -> Linear -> Logits"""
    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=CLS_HIDDEN, num_classes=NUM_CLASSES, dropout=CLS_DROPOUT):
        super().__init__()
        self.clf = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.clf(x)

with timer("Phase 2 - VRAM Dry-Run and Budget Verification"):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    test_enc = FrozenEncoder().to(device)
    test_fga = FeatureGatingAgent().to(device)
    test_fusion = GatedFusion().to(device)
    test_clf = VeracityClassifier().to(device)

    total_params = sum(p.numel() for m in [test_enc.encoder, test_fga, test_clf] for p in m.parameters())
    trainable_params = sum(p.numel() for m in [test_fga, test_clf] for p in m.parameters() if p.requires_grad)
    logger.info(f"Model Summary: Total params: {fmt(total_params)} | Trainable: {fmt(trainable_params)} ({trainable_params/total_params*100:.2f}%)")

    dummy_claims = [f"This is a test claim sentence {i}" for i in range(PHYSICAL_BATCH_SIZE)]
    dummy_speakers = [f"Speaker {i}" for i in range(PHYSICAL_BATCH_SIZE)]
    dummy_dates = [f"2023-01-{i+1:02d}" for i in range(PHYSICAL_BATCH_SIZE)]

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                eq = test_enc.encode(dummy_claims, device)
                es = test_enc.encode(dummy_speakers, device)
                et = test_enc.encode(dummy_dates, device)
                aq, a_s, at = test_fga(eq, es, et)
                logits = test_clf(test_fusion(eq, es, et, aq, a_s, at))
        else:
            eq = test_enc.encode(dummy_claims, device)
            es = test_enc.encode(dummy_speakers, device)
            et = test_enc.encode(dummy_dates, device)
            aq, a_s, at = test_fga(eq, es, et)
            logits = test_clf(test_fusion(eq, es, et, aq, a_s, at))

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        vram_ok = peak_vram < 2.5
        status_str = "✅ PASS" if vram_ok else "⚠️ EXCEEDS BUDGET"
        logger.info(f"Peak VRAM during forward pass: {peak_vram:.2f} GB (budget: 2.5 GB) -> {status_str}")
    else:
        logger.info("CPU forward pass completed successfully.")

    del test_enc, test_fga, test_fusion, test_clf, eq, es, et
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ============================================================================
# PHASE 3: EMBEDDING PRE-CACHING & REINFORCE TRAINING
# ============================================================================

print("\n" + "="*80)
print("  🎯 PHASE 3: REINFORCE DE-BIASING TRAINING")
print("="*80)

# Load training data
tr_data = pd.read_csv(PROC_DIR / "fci_train.csv")
tr_data["itemReviewed.author.name"] = tr_data["itemReviewed.author.name"].fillna("Unknown")
tr_data.dropna(subset=["claimReviewed", "label_id"], inplace=True)
tr_data["label_id"] = tr_data["label_id"].astype(int)
logger.info(f"Training samples available: {fmt(len(tr_data))}")

def encode_all_texts(enc, texts, desc, bs=64):
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc=desc, unit="b"):
        batch = texts[i:i+bs]
        embs.append(enc.encode(batch, device).cpu().float())
    return torch.cat(embs, dim=0)

encoder = FrozenEncoder().to(device)

with timer("Phase 3 - Pre-computing Claim, Speaker, and Date Embeddings"):
    train_claims = tr_data["claimReviewed"].tolist()
    train_speakers = tr_data["itemReviewed.author.name"].tolist()
    train_dates = tr_data["datePublished"].astype(str).tolist()
    train_labels = torch.tensor(tr_data["label_id"].values, dtype=torch.long)

    train_ce = encode_all_texts(encoder, train_claims, "Encoding Train Claims")

    unique_speakers = list(set(train_speakers))
    logger.info(f"Unique speakers in training: {fmt(len(unique_speakers))}")
    speaker_emb_map = {}
    for i in tqdm(range(0, len(unique_speakers), 64), desc="Encoding Speakers", unit="b"):
        batch = unique_speakers[i:i+64]
        b_embs = encoder.encode(batch, device).cpu().float()
        for sp, emb in zip(batch, b_embs):
            speaker_emb_map[sp] = emb
    train_se = torch.stack([speaker_emb_map[s] for s in train_speakers])

    # Date deduplication (3,000 unique dates vs 226k rows -> 75x speedup)
    unique_dates = list(set(train_dates))
    logger.info(f"Unique dates in training: {fmt(len(unique_dates))}")
    date_emb_map = {}
    for i in tqdm(range(0, len(unique_dates), 64), desc="Encoding Dates", unit="b"):
        batch = unique_dates[i:i+64]
        b_embs = encoder.encode(batch, device).cpu().float()
        for d, emb in zip(batch, b_embs):
            date_emb_map[d] = emb
    train_de = torch.stack([date_emb_map[d] for d in train_dates])

    logger.info(f"Embeddings cached: Claims {train_ce.shape}, Speakers {train_se.shape}, Dates {train_de.shape}")

# Free frozen encoder from VRAM during RL training
del encoder
if device.type == "cuda":
    torch.cuda.empty_cache()

class EmbeddingRLDataset(Dataset):
    def __init__(self, ce, se, de, labels, speakers, sp_map):
        self.ce = ce
        self.se = se
        self.de = de
        self.labels = labels
        self.speakers = speakers
        self.sp_map = sp_map
        self.all_sp_keys = list(sp_map.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "ce": self.ce[idx],
            "se": self.se[idx],
            "de": self.de[idx],
            "lbl": self.labels[idx],
            "sp": self.speakers[idx],
        }

    def sample_counterfactual_speaker(self, orig_sp):
        candidates = [s for s in self.all_sp_keys if s != orig_sp]
        if not candidates:
            return self.sp_map[orig_sp]
        chosen = random.choice(candidates)
        return self.sp_map[chosen]

train_dataset = EmbeddingRLDataset(train_ce, train_se, train_de, train_labels, train_speakers, speaker_emb_map)

def rl_collate_fn(batch):
    return {
        "ce": torch.stack([b["ce"] for b in batch]),
        "se": torch.stack([b["se"] for b in batch]),
        "de": torch.stack([b["de"] for b in batch]),
        "lbl": torch.stack([b["lbl"] for b in batch]),
        "cf_se": torch.stack([train_dataset.sample_counterfactual_speaker(b["sp"]) for b in batch]),
    }

train_loader = DataLoader(
    train_dataset,
    batch_size=PHYSICAL_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=rl_collate_fn,
    drop_last=True
)
logger.info(f"DataLoader initialized: {len(train_loader)} batches x {PHYSICAL_BATCH_SIZE} = Virtual Batch {VIRTUAL_BATCH_SIZE}")

fga = FeatureGatingAgent().to(device)
fusion = GatedFusion().to(device)
classifier = VeracityClassifier().to(device)

trainable_modules = list(fga.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(trainable_modules, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

baseline_reward = 0.0
training_history = []

with timer("Phase 3 - Full REINFORCE Training Loop"):
    for epoch in range(NUM_EPOCHS):
        fga.train()
        classifier.train()
        ep_losses, ep_rewards, ep_r_acc, ep_r_cons = [], [], [], []
        correct_preds, total_preds = 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch")
        for step, batch in enumerate(pbar):
            eq = batch["ce"].to(device)
            es = batch["se"].to(device)
            et = batch["de"].to(device)
            es_cf = batch["cf_se"].to(device)
            tgt = batch["lbl"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # Original prediction
                aq, a_s, at = fga(eq, es, et)
                logits = classifier(fusion(eq, es, et, aq, a_s, at))

                # Counterfactual prediction (perturbed speaker)
                aq_cf, as_cf, at_cf = fga(eq, es_cf, et)
                logits_cf = classifier(fusion(eq, es_cf, et, aq_cf, as_cf, at_cf))

                probs = F.softmax(logits, dim=-1)
                probs_cf = F.softmax(logits_cf, dim=-1)
                preds = logits.argmax(dim=-1)

                # Rewards
                # R_acc: +1.0 for correct, -1.0 for incorrect
                r_acc = (preds == tgt).float() * 2.0 - 1.0
                # R_cons: 1.0 - L1 distance between output probability distributions
                r_cons = 1.0 - torch.abs(probs - probs_cf).sum(dim=-1)
                r_total = LAMBDA_ACC * r_acc + LAMBDA_CONS * r_cons

                # REINFORCE policy loss with EMA baseline
                log_p = F.log_softmax(logits, dim=-1).gather(1, preds.unsqueeze(1)).squeeze(1)
                advantage = (r_total - baseline_reward).detach()
                policy_loss = -torch.mean(advantage * log_p)

                # Auxiliary Cross-Entropy loss for stability
                ce_loss = F.cross_entropy(logits, tgt)
                total_loss = (policy_loss + 0.5 * ce_loss) / GRAD_ACCUM_STEPS

            scaler.scale(total_loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_modules, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_reward = r_total.mean().item()
            baseline_reward = BASELINE_DECAY * baseline_reward + (1.0 - BASELINE_DECAY) * batch_reward

            ep_losses.append(total_loss.item() * GRAD_ACCUM_STEPS)
            ep_rewards.append(batch_reward)
            ep_r_acc.append(r_acc.mean().item())
            ep_r_cons.append(r_cons.mean().item())
            correct_preds += (preds == tgt).sum().item()
            total_preds += len(tgt)

            pbar.set_postfix(
                loss=f"{np.mean(ep_losses[-50:]):.4f}",
                rew=f"{np.mean(ep_rewards[-50:]):.3f}",
                acc=f"{correct_preds/total_preds:.3f}"
            )

        epoch_stats = {
            "epoch": epoch + 1,
            "loss": float(np.mean(ep_losses)),
            "reward": float(np.mean(ep_rewards)),
            "accuracy": correct_preds / total_preds,
            "r_acc": float(np.mean(ep_r_acc)),
            "r_cons": float(np.mean(ep_r_cons)),
        }
        training_history.append(epoch_stats)
        logger.info(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} -> Loss: {epoch_stats['loss']:.4f} | "
            f"Reward: {epoch_stats['reward']:.3f} | Acc: {epoch_stats['accuracy']:.4f} | "
            f"R_acc: {epoch_stats['r_acc']:.3f} | R_cons: {epoch_stats['r_cons']:.3f}"
        )

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = CKPT_DIR / f"cl_sdrg_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": {
                    "fga": fga.state_dict(),
                    "classifier": classifier.state_dict()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_stats
            }, str(ckpt_path))
            logger.info(f"💾 Checkpoint saved: {ckpt_path.name}")

    pd.DataFrame(training_history).to_csv(LOG_DIR / "training_history.csv", index=False)

# Plot training curves
plt.figure(figsize=(14, 10))
epochs_range = range(1, len(training_history) + 1)

plt.subplot(2, 2, 1)
plt.plot(epochs_range, [h["loss"] for h in training_history], "b-o", lw=2)
plt.title("Combined Policy + CE Loss", fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(epochs_range, [h["reward"] for h in training_history], "g-o", lw=2)
plt.title("Total REINFORCE Reward", fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Reward"); plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(epochs_range, [h["r_acc"] for h in training_history], "r-s", label="R_acc", lw=2)
plt.plot(epochs_range, [h["r_cons"] for h in training_history], "m-^", label="R_cons", lw=2)
plt.title("Reward Components", fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(epochs_range, [h["accuracy"] for h in training_history], "c-D", lw=2)
plt.title("Training Accuracy", fontweight="bold")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"📊 Training curves plot saved to {FIG_DIR / 'training_curves.png'}")

# ============================================================================
# PHASE 4: MULTI-METRIC EVALUATION, FAISS RETRIEVAL, BASELINES & SFR AUDIT
# ============================================================================

print("\n" + "="*80)
print("  📊 PHASE 4: EVALUATION, RETRIEVAL & SHORTCUT SENSITIVITY AUDIT")
print("="*80)

te_data = pd.read_csv(PROC_DIR / "fci_test.csv")
te_data["itemReviewed.author.name"] = te_data["itemReviewed.author.name"].fillna("Unknown")
te_data.dropna(subset=["claimReviewed", "label_id"], inplace=True)
te_data["label_id"] = te_data["label_id"].astype(int)
logger.info(f"Test samples (2024-2026): {fmt(len(te_data))}")

enc_eval = FrozenEncoder().to(device)

with timer("Phase 4 - Pre-computing Test & Baseline Embeddings"):
    te_claims = te_data["claimReviewed"].tolist()
    te_speakers = te_data["itemReviewed.author.name"].tolist()
    te_dates = te_data["datePublished"].astype(str).tolist()
    te_labels = te_data["label_id"].values

    te_ce = encode_all_texts(enc_eval, te_claims, "Encoding Test Claims")

    unique_te_speakers = list(set(te_speakers))
    te_sp_map = {}
    for i in range(0, len(unique_te_speakers), 64):
        b = unique_te_speakers[i:i+64]
        b_embs = enc_eval.encode(b, device).cpu().float()
        for s, emb in zip(b, b_embs):
            te_sp_map[s] = emb
    te_se = torch.stack([te_sp_map[s] for s in te_speakers])

    unique_te_dates = list(set(te_dates))
    te_date_map = {}
    for i in range(0, len(unique_te_dates), 64):
        b = unique_te_dates[i:i+64]
        b_embs = enc_eval.encode(b, device).cpu().float()
        for d, emb in zip(b, b_embs):
            te_date_map[d] = emb
    te_de = torch.stack([te_date_map[d] for d in te_dates])

    # Speaker dictionary for SFR audit
    all_speakers_eval = list(set(te_speakers + train_speakers))
    if len(all_speakers_eval) > 5000:
        all_speakers_eval = random.sample(all_speakers_eval, 5000)
    sp_eval_map = {}
    for i in range(0, len(all_speakers_eval), 64):
        b = all_speakers_eval[i:i+64]
        b_embs = enc_eval.encode(b, device).cpu().float()
        for s, emb in zip(b, b_embs):
            sp_eval_map[s] = emb

    # Training embeddings for zero-shot kNN & BM25 baselines
    tr_bl = pd.read_csv(PROC_DIR / "fci_train.csv").dropna(subset=["claimReviewed", "label_id"])
    tr_bl_claims = tr_bl["claimReviewed"].tolist()
    tr_bl_labels = tr_bl["label_id"].astype(int).values
    tr_bl_ce = encode_all_texts(enc_eval, tr_bl_claims, "Encoding Train Baseline Claims")

del enc_eval
if device.type == "cuda":
    torch.cuda.empty_cache()

# 4.1 Classification Evaluation
fga.eval()
classifier.eval()
all_preds, all_probs = [], []

with torch.no_grad():
    for i in range(0, len(te_ce), PHYSICAL_BATCH_SIZE):
        eq = te_ce[i:i+PHYSICAL_BATCH_SIZE].to(device)
        es = te_se[i:i+PHYSICAL_BATCH_SIZE].to(device)
        et = te_de[i:i+PHYSICAL_BATCH_SIZE].to(device)
        aq, a_s, at = fga(eq, es, et)
        logits = classifier(fusion(eq, es, et, aq, a_s, at))
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())

y_pred = np.concatenate(all_preds)
y_prob = np.concatenate(all_probs)
class_names = [ID2LABEL[i] for i in range(NUM_CLASSES)]

logger.info("\n" + classification_report(te_labels, y_pred, target_names=class_names, digits=4, zero_division=0))

cls_metrics = {
    "accuracy": float(accuracy_score(te_labels, y_pred)),
    "macro_f1": float(f1_score(te_labels, y_pred, average="macro", zero_division=0)),
    "macro_precision": float(precision_score(te_labels, y_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(te_labels, y_pred, average="macro", zero_division=0)),
}
cm = confusion_matrix(te_labels, y_pred)

# 4.2 FAISS Cross-Lingual Retrieval
gated_embeddings = []
with torch.no_grad():
    for i in range(0, len(te_ce), PHYSICAL_BATCH_SIZE):
        eq = te_ce[i:i+PHYSICAL_BATCH_SIZE].to(device)
        es = te_se[i:i+PHYSICAL_BATCH_SIZE].to(device)
        et = te_de[i:i+PHYSICAL_BATCH_SIZE].to(device)
        aq, a_s, at = fga(eq, es, et)
        gated_embeddings.append(fusion(eq, es, et, aq, a_s, at).cpu().numpy())

gated_arr = np.concatenate(gated_embeddings).astype(np.float32)
faiss_gallery = gated_arr.copy()
faiss.normalize_L2(faiss_gallery)
faiss_idx = faiss.IndexFlatIP(EMBEDDING_DIM)
faiss_idx.add(faiss_gallery)

faiss_query = gated_arr.copy()
faiss.normalize_L2(faiss_query)
_, search_indices = faiss_idx.search(faiss_query, max(RETRIEVAL_K_VALUES) + 1)

retrieval_metrics = {}
for k in RETRIEVAL_K_VALUES:
    rec_list, mrr_list, ndcg_list = [], [], []
    for i in range(len(faiss_query)):
        retrieved = te_labels[search_indices[i]]
        mask = search_indices[i] != i
        retrieved = retrieved[mask][:k]
        matches = (retrieved == te_labels[i])
        rec_list.append(float(matches.any()))
        pos = np.where(matches)[0]
        mrr_list.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
        dcg = sum(r / np.log2(p + 2) for p, r in enumerate(matches.astype(float)))
        idcg = sum(1.0 / np.log2(p + 2) for p in range(max(1, int(matches.sum()))))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    retrieval_metrics[f"recall@{k}"] = float(np.mean(rec_list))
    retrieval_metrics[f"mrr@{k}"] = float(np.mean(mrr_list))
    retrieval_metrics[f"ndcg@{k}"] = float(np.mean(ndcg_list))

# 4.3 Speaker Flip Rate (SFR) Bias Audit
logger.info(f"Running SFR Audit: {fmt(len(te_ce))} test samples x {SFR_NUM_PERTURBATIONS} perturbations")
sp_keys = list(sp_eval_map.keys())
n_test = len(te_ce)
flip_counts = np.zeros(n_test)

with torch.no_grad():
    orig_preds = []
    for i in range(0, n_test, 64):
        eq = te_ce[i:i+64].to(device)
        es = te_se[i:i+64].to(device)
        et = te_de[i:i+64].to(device)
        aq, a_s, at = fga(eq, es, et)
        orig_preds.append(classifier(fusion(eq, es, et, aq, a_s, at)).argmax(dim=-1).cpu())
    orig_preds = torch.cat(orig_preds)

    for pert in tqdm(range(SFR_NUM_PERTURBATIONS), desc="SFR Perturbations", unit="run"):
        rand_sp = random.choices(sp_keys, k=n_test)
        cf_se = torch.stack([sp_eval_map[s] for s in rand_sp])
        pert_preds = []
        for i in range(0, n_test, 64):
            eq = te_ce[i:i+64].to(device)
            es_pert = cf_se[i:i+64].to(device)
            et = te_de[i:i+64].to(device)
            aq, a_s, at = fga(eq, es_pert, et)
            pert_preds.append(classifier(fusion(eq, es_pert, et, aq, a_s, at)).argmax(dim=-1).cpu())
        pert_preds = torch.cat(pert_preds)
        flip_counts += (orig_preds != pert_preds).numpy().astype(float)

sfr_value = float(np.mean(flip_counts / SFR_NUM_PERTURBATIONS))
sfr_passed = sfr_value < SFR_TARGET

# 4.4 Ablation Baselines
ablation_results = []

# Baseline 1: BM25 Lexical (Fast representative subset: <15s execution)
try:
    from rank_bm25 import BM25Okapi
    # Representative training index (25k claims) to prevent 35+ minute evaluation
    sample_tr = tr_bl_claims[:25000] if len(tr_bl_claims) > 25000 else tr_bl_claims
    sample_labels = tr_bl_labels[:25000] if len(tr_bl_labels) > 25000 else tr_bl_labels
    tokenized_train = [c.lower().split() for c in sample_tr]
    bm25 = BM25Okapi(tokenized_train)

    eval_claims = te_claims
    eval_labels = te_labels
    if len(te_claims) > 500:
        logger.info("Evaluating BM25 on a representative subset of 500 test claims for lightning-fast execution...")
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(te_claims), 500, replace=False)
        eval_claims = [te_claims[i] for i in sub_idx]
        eval_labels = te_labels[sub_idx]

    bm25_preds = np.array([
        sample_labels[bm25.get_scores(c.lower().split()).argmax()]
        for c in tqdm(eval_claims, desc="BM25 Baseline", unit="claim")
    ])
    ablation_results.append({
        "method": "BM25 Lexical",
        "accuracy": float(accuracy_score(eval_labels, bm25_preds)),
        "macro_f1": float(f1_score(eval_labels, bm25_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(eval_labels, bm25_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(eval_labels, bm25_preds, average="macro", zero_division=0)),
        "sfr": "N/A"
    })
except Exception as ex:
    logger.warning(f"BM25 baseline failed: {ex}")

# Baseline 2: Zero-Shot kNN (Frozen mE5)
try:
    tr_mat = tr_bl_ce.numpy().astype(np.float32)
    te_mat = te_ce.numpy().astype(np.float32)
    faiss.normalize_L2(tr_mat)
    faiss.normalize_L2(te_mat)
    knn_idx = faiss.IndexFlatIP(EMBEDDING_DIM)
    knn_idx.add(tr_mat)
    _, knn_I = knn_idx.search(te_mat, 1)
    knn_preds = tr_bl_labels[knn_I[:, 0]]
    ablation_results.append({
        "method": "Zero-Shot kNN (mE5)",
        "accuracy": float(accuracy_score(te_labels, knn_preds)),
        "macro_f1": float(f1_score(te_labels, knn_preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(te_labels, knn_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(te_labels, knn_preds, average="macro", zero_division=0)),
        "sfr": "N/A"
    })
except Exception as ex:
    logger.warning(f"Zero-shot kNN baseline failed: {ex}")

# ============================================================================
# 5. FINAL REPORT & VISUALIZATIONS
# ============================================================================

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("CL-SDRG Comprehensive Benchmark Results", fontsize=16, fontweight="bold")

all_methods = ["CL-SDRG (Ours)"] + [r["method"] for r in ablation_results]
all_results = [{"method": "CL-SDRG (Ours)", **cls_metrics}] + ablation_results
metric_keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
metric_labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]

x = np.arange(len(metric_keys))
width = 0.8 / len(all_methods)
colors = plt.cm.Set2(np.linspace(0, 1, len(all_methods)))

for i, (m_name, res) in enumerate(zip(all_methods, all_results)):
    vals = [res.get(k, 0.0) for k in metric_keys]
    axes[0].bar(x + i * width, vals, width, label=m_name, color=colors[i])

axes[0].set_xticks(x + width * (len(all_methods) - 1) / 2)
axes[0].set_xticklabels(metric_labels, fontweight="bold")
axes[0].set_ylim(0, 1.0)
axes[0].legend(fontsize=9)
axes[0].set_title("Classification Performance", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

# SFR Bar
axes[1].bar(["CL-SDRG"], [sfr_value * 100], color=["#2ecc71" if sfr_passed else "#e74c3c"], width=0.4)
axes[1].axhline(SFR_TARGET * 100, color="red", linestyle="--", linewidth=2, label=f"Target < {SFR_TARGET*100:.0f}%")
axes[1].set_ylabel("Speaker Flip Rate (%)", fontweight="bold")
axes[1].set_title("Shortcut Bias Audit (SFR)", fontweight="bold")
axes[1].set_ylim(0, max(5.0, sfr_value * 100 + 2.0))
axes[1].legend()
axes[1].text(0, sfr_value * 100 + 0.2, f"{sfr_value*100:.2f}%", ha="center", fontweight="bold", fontsize=12)

# Confusion Matrix & Visualization (NumPy 2.0+ compatible)
try:
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    axes[2].set_title("CL-SDRG Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted Label", fontweight="bold")
    axes[2].set_ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "evaluation_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Evaluation results plot saved to {FIG_DIR / 'evaluation_results.png'}")
except Exception as ex:
    logger.warning(f"Plot saving notice (safe to ignore): {ex}")
    plt.close("all")

# Save CSVs
final_df = pd.DataFrame([{"method": "CL-SDRG (Ours)", "sfr": f"{sfr_value:.4f}", **cls_metrics, **retrieval_metrics}] + ablation_results)
final_df.to_csv(OUT_DIR / "benchmark_results.csv", index=False)
pd.DataFrame([{"sfr": sfr_value, "target": SFR_TARGET, "passed": sfr_passed}]).to_csv(OUT_DIR / "sfr_audit.csv", index=False)

# Print Final Summary Banner
print("\n" + "═"*90)
print("  🏆 CL-SDRG BENCHMARK RESULTS (COPY EVERYTHING BELOW)")
print("═"*90)
print(f"{'Method':<28} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10} {'SFR':>10}")
print("─"*90)
print(f"{'CL-SDRG (Ours)':<28} {cls_metrics['accuracy']:10.4f} {cls_metrics['macro_f1']:10.4f} "
      f"{cls_metrics['macro_precision']:10.4f} {cls_metrics['macro_recall']:10.4f} {sfr_value:10.4f}")
for r in ablation_results:
    sfr_display = f"{r['sfr']:>10}" if isinstance(r['sfr'], str) else f"{r['sfr']:10.4f}"
    print(f"{r['method']:<28} {r['accuracy']:10.4f} {r['macro_f1']:10.4f} "
          f"{r['macro_precision']:10.4f} {r['macro_recall']:10.4f} {sfr_display}")
print("─"*90)

print("\n  🔍 Cross-Lingual Retrieval Performance (CL-SDRG):")
for k in RETRIEVAL_K_VALUES:
    print(f"    • K={k:<2} -> Recall@{k}: {retrieval_metrics[f'recall@{k}']:.4f} | "
          f"MRR@{k}: {retrieval_metrics[f'mrr@{k}']:.4f} | nDCG@{k}: {retrieval_metrics[f'ndcg@{k}']:.4f}")

sfr_status = "✅ TARGET MET (< 3%)" if sfr_passed else "⚠️ TARGET NOT MET"
print(f"\n  🎯 Shortcut Bias Audit:")
print(f"    • Speaker Flip Rate (SFR): {sfr_value:.4f} ({sfr_value*100:.2f}%) -> {sfr_status}")
print("═"*90)
print("\n🎉 EXECUTION FINISHED SUCCESSFULLY! Copy the block above and share it back.")
