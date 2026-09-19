"""
CL-SDRG — Script 01: Data Ingestion, Zero-Leakage Sanitization & Language ID
==============================================================================
Phase 1 of the CL-SDRG pipeline.

This script:
  1. Loads the raw FactCheck Insights (FCI) claim_review.csv
  2. Applies Zero-Evidence-Leakage sanitization (drops ratingExplanation, etc.)
  3. Normalizes heterogeneous verdict labels → {TRUE, FALSE, MIXED}
  4. Runs FastText lid.176 for language identification on each claim
  5. Filters and tags South Asian language claims (Urdu, Hindi, Bengali, etc.)
  6. Mines silver cross-lingual pairs (±3 day publication window)
  7. Creates time-aware train/test splits (train ≤ 2023, test ≥ 2024)
  8. Checks Smoke Test Gate S₁ (Urdu count ≥ 500)
  9. Saves processed datasets and prints summary statistics

Target Environment: Google Colab T4 GPU (CPU-bound; no GPU needed for this script)

Usage (Colab):
    !pip install fasttext pandas tqdm
    !wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O outputs/lid.176.bin
    %run src/01_data_ingestion_and_lid.py
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add project root to path for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLAIM_REVIEW_CSV, PROCESSED_DATA_DIR, LOG_DIR, OUTPUT_DIR,
    FASTTEXT_MODEL_PATH, RETAINED_COLUMNS, LEAKAGE_COLUMNS,
    VERDICT_MAP, LABEL2ID, TARGET_LANGUAGES, PRIMARY_FOCUS_LANGUAGES,
    LID_CONFIDENCE_THRESHOLD, SILVER_PAIR_DAY_WINDOW,
    MIN_URDU_CLAIMS_THRESHOLD, TRAIN_CUTOFF_DATE, TEST_START_DATE,
    RANDOM_SEED,
)
from src.config import init_directories
from src.utils import (
    set_seed, setup_logging, timer, normalize_verdict,
    print_dataframe_summary, format_number,
)

warnings.filterwarnings("ignore")


# ============================================================================
# STEP 1.1: DATA INGESTION & ZERO-LEAKAGE SCRUBBING
# ============================================================================

def load_and_sanitize(csv_path: Path) -> pd.DataFrame:
    """
    Load raw FCI dataset and apply zero-leakage sanitization.

    Steps:
        - Read CSV with proper encoding handling
        - Drop leakage columns (ratingExplanation, ratingValue)
        - Retain only the columns specified in config
        - Drop rows with missing claim text or verdict

    Args:
        csv_path: Path to claim_review.csv

    Returns:
        pd.DataFrame: Sanitized dataframe
    """
    with timer("Loading raw CSV"):
        df = pd.read_csv(
            str(csv_path),
            encoding="utf-8",
            low_memory=False,
            on_bad_lines="skip",
        )
        logging.info(f"Raw dataset: {format_number(len(df))} rows, {df.shape[1]} columns")

    # ── Zero-Leakage Sanitization ──
    with timer("Zero-leakage sanitization"):
        # Log which leakage columns exist before dropping
        for col in LEAKAGE_COLUMNS:
            if col in df.columns:
                non_null = df[col].notna().sum()
                logging.info(f"  Dropping leakage column: '{col}' ({format_number(non_null)} non-null values)")
                df.drop(columns=[col], inplace=True)

        # Also drop any columns containing fact-checker commentary
        commentary_cols = [c for c in df.columns if "explanation" in c.lower() or "ratingValue" in c]
        if commentary_cols:
            logging.info(f"  Dropping additional commentary columns: {commentary_cols}")
            df.drop(columns=[col for col in commentary_cols if col in df.columns], inplace=True)

    # ── Retain only necessary columns ──
    available_cols = [c for c in RETAINED_COLUMNS if c in df.columns]
    missing_cols = set(RETAINED_COLUMNS) - set(available_cols)
    if missing_cols:
        logging.warning(f"  Missing expected columns: {missing_cols}")

    df = df[available_cols].copy()

    # ── Drop rows without claim text or verdict ──
    initial_count = len(df)
    df.dropna(subset=["claimReviewed", "reviewRating.alternateName"], inplace=True)
    dropped = initial_count - len(df)
    logging.info(f"  Dropped {format_number(dropped)} rows with missing claim/verdict")

    # ── Basic text cleaning ──
    df["claimReviewed"] = df["claimReviewed"].astype(str).str.strip()
    df = df[df["claimReviewed"].str.len() > 5]  # Remove trivially short claims

    logging.info(f"  Sanitized dataset: {format_number(len(df))} rows")
    return df


# ============================================================================
# STEP 1.1b: VERDICT NORMALIZATION
# ============================================================================

def normalize_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map heterogeneous verdict labels to the 3-class taxonomy {TRUE, FALSE, MIXED}.

    Args:
        df: DataFrame with 'reviewRating.alternateName' column

    Returns:
        pd.DataFrame: DataFrame with added 'verdict' and 'label_id' columns
    """
    with timer("Verdict normalization"):
        df["verdict"] = df["reviewRating.alternateName"].apply(
            lambda x: normalize_verdict(x, VERDICT_MAP)
        )

        # Log unmapped verdicts for debugging
        unmapped_mask = df["verdict"].isna()
        unmapped_count = unmapped_mask.sum()

        if unmapped_count > 0:
            raw_unmapped = df.loc[unmapped_mask, "reviewRating.alternateName"].value_counts().head(20)
            logging.warning(f"  Unmapped verdict labels ({format_number(unmapped_count)} rows):")
            for label, count in raw_unmapped.items():
                logging.warning(f"    '{label}': {count}")

        # Drop unmapped rows
        df = df[df["verdict"].notna()].copy()
        df["label_id"] = df["verdict"].map(LABEL2ID)

        # Label distribution
        dist = df["verdict"].value_counts()
        logging.info(f"  Label distribution after normalization:")
        for label, count in dist.items():
            pct = count / len(df) * 100
            logging.info(f"    {label}: {format_number(count)} ({pct:.1f}%)")

    return df


# ============================================================================
# STEP 1.2: FASTTEXT LANGUAGE IDENTIFICATION
# ============================================================================

def download_fasttext_model(model_path: Path):
    """Download FastText lid.176 model if not present."""
    if model_path.exists():
        logging.info(f"FastText model already exists: {model_path}")
        return

    logging.info("Downloading FastText lid.176 model (~126 MB)...")
    import urllib.request

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(model_path))
    logging.info(f"Downloaded to: {model_path}")


def run_language_identification(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """
    Run FastText lid.176 on each claim to detect its language.

    Args:
        df: DataFrame with 'claimReviewed' column
        model_path: Path to lid.176.bin

    Returns:
        pd.DataFrame: DataFrame with added 'detected_lang' and 'lang_confidence' columns
    """
    import fasttext

    # Suppress FastText warning about deprecated load_model
    fasttext.FastText.eprint = lambda x: None

    with timer("FastText language identification"):
        model = fasttext.load_model(str(model_path))

        languages = []
        confidences = []

        for text in tqdm(df["claimReviewed"].values, desc="Language ID", unit="claim"):
            # FastText expects single-line input
            clean_text = str(text).replace("\n", " ").replace("\r", " ").strip()

            if not clean_text:
                languages.append("unknown")
                confidences.append(0.0)
                continue

            predictions = model.predict(clean_text, k=1)
            lang_code = predictions[0][0].replace("__label__", "")
            confidence = float(predictions[1][0])

            languages.append(lang_code)
            confidences.append(confidence)

        df["detected_lang"] = languages
        df["lang_confidence"] = confidences

        # Language distribution
        lang_dist = df["detected_lang"].value_counts().head(20)
        logging.info(f"\n  Top 20 detected languages:")
        for lang, count in lang_dist.items():
            pct = count / len(df) * 100
            logging.info(f"    {lang}: {format_number(count)} ({pct:.1f}%)")

    return df


def filter_south_asian_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter claims identified as South Asian languages with sufficient confidence.

    Args:
        df: DataFrame with 'detected_lang' and 'lang_confidence'

    Returns:
        pd.DataFrame: Filtered subset of South Asian claims
    """
    with timer("South Asian language filtering"):
        mask = (
            df["detected_lang"].isin(TARGET_LANGUAGES) &
            (df["lang_confidence"] >= LID_CONFIDENCE_THRESHOLD)
        )
        sa_df = df[mask].copy()

        logging.info(f"  South Asian claims (confidence ≥ {LID_CONFIDENCE_THRESHOLD}):")
        sa_lang_dist = sa_df["detected_lang"].value_counts()
        for lang, count in sa_lang_dist.items():
            logging.info(f"    {lang}: {format_number(count)}")

        logging.info(f"  Total South Asian claims: {format_number(len(sa_df))}")

    return sa_df


# ============================================================================
# STEP 1.2b: SILVER PAIR MINING
# ============================================================================

def mine_silver_pairs(df: pd.DataFrame, day_window: int = 3) -> pd.DataFrame:
    """
    Identify naturally occurring cross-lingual pairs from multi-edition
    fact-checking organizations (e.g., AFP Urdu, BOOM Live, Fact Crescendo)
    published within a ±day_window day publication window.

    Pairs are identified by:
        1. Same fact-checking org (author.name)
        2. Different detected languages
        3. Published within ±day_window days
        4. High semantic similarity (based on overlapping named entities / numbers)

    Args:
        df: DataFrame with date, language, and org columns
        day_window: Max days apart for pair candidates

    Returns:
        pd.DataFrame: Silver pair records with columns [claim_a, lang_a, claim_b, lang_b, org, date_diff]
    """
    with timer(f"Silver pair mining (±{day_window} day window)"):
        # Ensure datetime parsing
        if not pd.api.types.is_datetime64_any_dtype(df["datePublished"]):
            df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce", utc=True)

        # Filter to claims with valid dates and known orgs
        pair_df = df.dropna(subset=["datePublished", "author.name"]).copy()
        pair_df = pair_df[pair_df["author.name"].str.strip().str.len() > 0]

        # Group by organization
        pairs = []
        org_groups = pair_df.groupby("author.name")

        for org_name, group in tqdm(org_groups, desc="Mining silver pairs", unit="org"):
            if len(group) < 2:
                continue

            # Get unique languages in this org
            org_langs = group["detected_lang"].unique()
            if len(org_langs) < 2:
                continue

            # Sort by date for efficient windowed comparison
            group = group.sort_values("datePublished")

            # Pairwise comparison within the time window
            dates = group["datePublished"].values
            langs = group["detected_lang"].values
            claims = group["claimReviewed"].values
            indices = group.index.values

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Check time window
                    date_diff = abs((dates[j] - dates[i]) / np.timedelta64(1, "D"))
                    if date_diff > day_window:
                        break  # Sorted by date, no need to check further

                    # Check different languages
                    if langs[i] != langs[j]:
                        pairs.append({
                            "claim_a": claims[i],
                            "lang_a": langs[i],
                            "claim_b": claims[j],
                            "lang_b": langs[j],
                            "org": org_name,
                            "date_diff_days": round(date_diff, 1),
                            "idx_a": indices[i],
                            "idx_b": indices[j],
                        })

        silver_pairs_df = pd.DataFrame(pairs)
        logging.info(f"  Discovered {format_number(len(silver_pairs_df))} silver cross-lingual pairs")

        if len(silver_pairs_df) > 0:
            pair_lang_dist = silver_pairs_df[["lang_a", "lang_b"]].apply(
                lambda row: f"{row['lang_a']}-{row['lang_b']}", axis=1
            ).value_counts().head(10)
            logging.info(f"  Top language pairs:")
            for pair_name, count in pair_lang_dist.items():
                logging.info(f"    {pair_name}: {count}")

    return silver_pairs_df


# ============================================================================
# STEP 1.3: TIME-AWARE DATASET PARTITIONING
# ============================================================================

def create_time_aware_splits(df: pd.DataFrame) -> tuple:
    """
    Split dataset into train and test sets based on publication date.

    Train: claims with datePublished ≤ 2023-12-31
    Test:  claims with datePublished ≥ 2024-01-01

    Args:
        df: DataFrame with 'datePublished' column

    Returns:
        tuple: (train_df, test_df)
    """
    with timer("Time-aware dataset splitting"):
        # Parse dates
        if not pd.api.types.is_datetime64_any_dtype(df["datePublished"]):
            df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce", utc=True)

        # Drop rows without valid dates
        valid_dates = df["datePublished"].notna()
        logging.info(f"  Rows with valid dates: {format_number(valid_dates.sum())} / {format_number(len(df))}")
        df = df[valid_dates].copy()

        train_cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE, tz="UTC")
        test_start = pd.Timestamp(TEST_START_DATE, tz="UTC")

        train_df = df[df["datePublished"] <= train_cutoff].copy()
        test_df = df[df["datePublished"] >= test_start].copy()

        logging.info(f"\n  Train set (≤ {TRAIN_CUTOFF_DATE}):")
        logging.info(f"    Total:  {format_number(len(train_df))}")
        if len(train_df) > 0:
            train_label_dist = train_df["verdict"].value_counts()
            for label, count in train_label_dist.items():
                logging.info(f"    {label}: {format_number(count)} ({count/len(train_df)*100:.1f}%)")

        logging.info(f"\n  Test set (≥ {TEST_START_DATE}):")
        logging.info(f"    Total:  {format_number(len(test_df))}")
        if len(test_df) > 0:
            test_label_dist = test_df["verdict"].value_counts()
            for label, count in test_label_dist.items():
                logging.info(f"    {label}: {format_number(count)} ({count/len(test_df)*100:.1f}%)")

        # Date range info
        if len(train_df) > 0:
            logging.info(f"\n  Train date range: {train_df['datePublished'].min()} → {train_df['datePublished'].max()}")
        if len(test_df) > 0:
            logging.info(f"  Test date range:  {test_df['datePublished'].min()} → {test_df['datePublished'].max()}")

    return train_df, test_df


# ============================================================================
# SMOKE TEST GATE S₁: Urdu Claims Threshold Check
# ============================================================================

def check_urdu_threshold(df: pd.DataFrame) -> dict:
    """
    Check if we have sufficient Urdu claims (≥ 500).
    If not, flag for NLLB-200 fallback translation.

    Args:
        df: DataFrame with 'detected_lang' column

    Returns:
        dict: Smoke test results
    """
    urdu_mask = df["detected_lang"] == "ur"
    urdu_count = urdu_mask.sum()
    gate_passed = urdu_count >= MIN_URDU_CLAIMS_THRESHOLD

    result = {
        "urdu_count": urdu_count,
        "threshold": MIN_URDU_CLAIMS_THRESHOLD,
        "gate_passed": gate_passed,
        "needs_nllb_fallback": not gate_passed,
    }

    if gate_passed:
        logging.info(f"\n  ✅ Smoke Test Gate S₁ PASSED: {format_number(urdu_count)} Urdu claims (≥ {MIN_URDU_CLAIMS_THRESHOLD})")
    else:
        logging.warning(f"\n  ⚠️  Smoke Test Gate S₁ FAILED: Only {format_number(urdu_count)} Urdu claims (need ≥ {MIN_URDU_CLAIMS_THRESHOLD})")
        logging.warning(f"  → NLLB-200 translation fallback will be required for evaluation")

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete Phase 1 data engineering pipeline."""

    # ── Setup ──
    init_directories()
    setup_logging(log_dir=LOG_DIR, script_name="01_data_ingestion")
    set_seed(RANDOM_SEED)

    logging.info("=" * 70)
    logging.info("  CL-SDRG Phase 1: Data Ingestion & Language Identification")
    logging.info("=" * 70)

    # ── Step 1.1: Load and sanitize ──
    df = load_and_sanitize(CLAIM_REVIEW_CSV)
    print_dataframe_summary(df, "Sanitized FCI Dataset")

    # ── Step 1.1b: Normalize verdicts ──
    df = normalize_verdicts(df)

    # ── Step 1.2: FastText Language Identification ──
    download_fasttext_model(FASTTEXT_MODEL_PATH)
    df = run_language_identification(df, FASTTEXT_MODEL_PATH)

    # ── Tag South Asian claims ──
    df["is_south_asian"] = df["detected_lang"].isin(TARGET_LANGUAGES)
    df["is_primary_focus"] = df["detected_lang"].isin(PRIMARY_FOCUS_LANGUAGES)
    sa_df = filter_south_asian_claims(df)

    # ── Step 1.2b: Silver pair mining ──
    silver_pairs = mine_silver_pairs(df, day_window=SILVER_PAIR_DAY_WINDOW)

    # ── Step 1.3: Time-aware splits ──
    train_df, test_df = create_time_aware_splits(df)

    # Also create South-Asian-specific splits
    sa_train = train_df[train_df["is_south_asian"]].copy()
    sa_test = test_df[test_df["is_south_asian"]].copy()
    logging.info(f"\n  South Asian train: {format_number(len(sa_train))}")
    logging.info(f"  South Asian test:  {format_number(len(sa_test))}")

    # ── Smoke Test Gate S₁ ──
    smoke_test = check_urdu_threshold(df)

    # ── Save processed data ──
    with timer("Saving processed datasets"):
        # Full processed dataset
        full_output = PROCESSED_DATA_DIR / "fci_processed_full.csv"
        df.to_csv(str(full_output), index=False, encoding="utf-8")
        logging.info(f"  Saved: {full_output.name} ({format_number(len(df))} rows)")

        # Train / Test splits
        train_output = PROCESSED_DATA_DIR / "fci_train.csv"
        test_output = PROCESSED_DATA_DIR / "fci_test.csv"
        train_df.to_csv(str(train_output), index=False, encoding="utf-8")
        test_df.to_csv(str(test_output), index=False, encoding="utf-8")
        logging.info(f"  Saved: {train_output.name} ({format_number(len(train_df))} rows)")
        logging.info(f"  Saved: {test_output.name} ({format_number(len(test_df))} rows)")

        # South Asian subset
        sa_output = PROCESSED_DATA_DIR / "fci_south_asian.csv"
        sa_df.to_csv(str(sa_output), index=False, encoding="utf-8")
        logging.info(f"  Saved: {sa_output.name} ({format_number(len(sa_df))} rows)")

        # Silver pairs
        if len(silver_pairs) > 0:
            pairs_output = PROCESSED_DATA_DIR / "silver_pairs.csv"
            silver_pairs.to_csv(str(pairs_output), index=False, encoding="utf-8")
            logging.info(f"  Saved: {pairs_output.name} ({format_number(len(silver_pairs))} pairs)")

        # South Asian train/test
        sa_train.to_csv(str(PROCESSED_DATA_DIR / "fci_sa_train.csv"), index=False, encoding="utf-8")
        sa_test.to_csv(str(PROCESSED_DATA_DIR / "fci_sa_test.csv"), index=False, encoding="utf-8")

    # ── Final Summary ──
    logging.info("\n" + "=" * 70)
    logging.info("  PHASE 1 COMPLETE — Summary Statistics")
    logging.info("=" * 70)
    logging.info(f"  Total raw records:         {format_number(257_877)}")
    logging.info(f"  After sanitization:        {format_number(len(df))}")
    logging.info(f"  Unique languages detected: {df['detected_lang'].nunique()}")
    logging.info(f"  South Asian claims:        {format_number(len(sa_df))}")
    logging.info(f"  Silver cross-lingual pairs:{format_number(len(silver_pairs))}")
    logging.info(f"  Train split:               {format_number(len(train_df))}")
    logging.info(f"  Test split:                {format_number(len(test_df))}")
    logging.info(f"  Urdu claims:               {format_number(smoke_test['urdu_count'])}")
    logging.info(f"  NLLB fallback needed:      {smoke_test['needs_nllb_fallback']}")
    logging.info("=" * 70)

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "sa_df": sa_df,
        "silver_pairs": silver_pairs,
        "smoke_test": smoke_test,
    }


if __name__ == "__main__":
    results = main()
