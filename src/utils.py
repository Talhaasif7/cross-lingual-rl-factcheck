"""
CL-SDRG Utilities Module
=========================
Shared helper functions for reproducibility, device management,
logging, memory profiling, and data processing.
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np

# ── Lazy imports for optional dependencies ──
# (torch / transformers may not be needed in data-only scripts)
_torch = None
_torch_cuda = None


def _import_torch():
    """Lazy-import torch to avoid errors in CPU-only environments."""
    global _torch, _torch_cuda
    if _torch is None:
        import torch
        _torch = torch
        _torch_cuda = torch.cuda
    return _torch


# ============================================================================
# 1. REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        torch = _import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============================================================================
# 2. DEVICE MANAGEMENT
# ============================================================================

def get_device():
    """
    Detect and return the best available compute device.

    Returns:
        torch.device: 'cuda' if GPU available, else 'cpu'.
    """
    torch = _import_torch()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logging.info("No GPU detected — using CPU")
    return device


def get_gpu_memory_summary():
    """
    Get current GPU memory usage statistics.

    Returns:
        dict: Memory stats in MB, or None if no GPU.
    """
    torch = _import_torch()
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "peak_allocated_mb": round(max_allocated, 1),
    }


def reset_peak_memory():
    """Reset peak memory tracking for a fresh measurement."""
    torch = _import_torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ============================================================================
# 3. LOGGING
# ============================================================================

def setup_logging(
    log_dir: Path = None,
    script_name: str = "cl_sdrg",
    level: int = logging.INFO,
):
    """
    Configure logging with both console and file output.

    Args:
        log_dir: Directory for log files. If None, console-only.
        script_name: Base name for the log file.
        level: Logging level.

    Returns:
        logging.Logger: Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicates on re-import
    logger.handlers.clear()

    # Console handler
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (if log directory provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{script_name}_{timestamp}.log"

        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        logging.info(f"Logging to: {log_file}")

    return logger


# ============================================================================
# 4. TIMING UTILITIES
# ============================================================================

@contextmanager
def timer(label: str = "Operation"):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("Data loading"):
            df = pd.read_csv(...)
    """
    start = time.perf_counter()
    logging.info(f"⏱  {label} — started")
    yield
    elapsed = time.perf_counter() - start

    if elapsed < 60:
        logging.info(f"✅ {label} — completed in {elapsed:.1f}s")
    else:
        mins, secs = divmod(elapsed, 60)
        logging.info(f"✅ {label} — completed in {int(mins)}m {secs:.1f}s")


# ============================================================================
# 5. DATA HELPERS
# ============================================================================

def normalize_verdict(raw_label: str, verdict_map: dict) -> str:
    """
    Normalize a raw verdict string to the canonical 3-class taxonomy.

    Args:
        raw_label: Raw label string from the dataset.
        verdict_map: Mapping dict from config.VERDICT_MAP.

    Returns:
        Normalized label string, or None if unmappable.
    """
    if not isinstance(raw_label, str) or not raw_label.strip():
        return None

    cleaned = raw_label.strip().lower()

    # Direct lookup
    if cleaned in verdict_map:
        return verdict_map[cleaned]

    # Fuzzy substring matching for edge cases
    for key, value in verdict_map.items():
        if key in cleaned or cleaned in key:
            return value

    return None


def print_dataframe_summary(df, name: str = "DataFrame"):
    """Log a concise summary of a pandas DataFrame."""
    logging.info(f"\n{'='*60}")
    logging.info(f"  {name} Summary")
    logging.info(f"{'='*60}")
    logging.info(f"  Rows:    {len(df):,}")
    logging.info(f"  Columns: {df.shape[1]}")
    logging.info(f"  Memory:  {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    # Show null counts for key columns
    null_pcts = (df.isnull().sum() / len(df) * 100).round(1)
    cols_with_nulls = null_pcts[null_pcts > 0]
    if len(cols_with_nulls) > 0:
        logging.info(f"  Columns with nulls:")
        for col, pct in cols_with_nulls.items():
            logging.info(f"    {col}: {pct}%")
    logging.info(f"{'='*60}\n")


def format_number(n: int) -> str:
    """Format a number with comma separators."""
    return f"{n:,}"


# ============================================================================
# 6. CHECKPOINT HELPERS
# ============================================================================

def save_checkpoint(
    model_state_dict: dict,
    optimizer_state_dict: dict,
    epoch: int,
    metrics: dict,
    filepath: Path,
):
    """
    Save a training checkpoint.

    Args:
        model_state_dict: Model's state_dict.
        optimizer_state_dict: Optimizer's state_dict.
        epoch: Current epoch number.
        metrics: Dict of current metrics.
        filepath: Path to save the checkpoint.
    """
    torch = _import_torch()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(filepath))
    logging.info(f"💾 Checkpoint saved: {filepath.name} (epoch {epoch})")


def load_checkpoint(filepath: Path, device=None):
    """
    Load a training checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    torch = _import_torch()
    if device is None:
        device = get_device()

    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)
    logging.info(f"📂 Checkpoint loaded: {Path(filepath).name} (epoch {checkpoint['epoch']})")
    return checkpoint
