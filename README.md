<p align="center">
  <h1 align="center">🔬 CL-SDRG</h1>
  <p align="center">
    <strong>Cross-Lingual Shortcut De-biasing via Reinforcement Learning Gating<br/>on Frozen Transformer Representations</strong>
  </p>
  <p align="center">
    <a href="#methodology">Methodology</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/Hardware-Colab_T4-green?logo=googlecolab" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 📌 Abstract

Modern fact-verification models often learn **source-trust shortcuts** — classifying claims based on *who said it* rather than *what was said*. This produces brittle models that fail on unseen speakers and low-resource languages.

**CL-SDRG** addresses this by introducing a lightweight **Feature Gating Agent (FGA)** trained with **REINFORCE policy gradients** and a novel **Counterfactual Consistency Reward**. The FGA learns to suppress speaker-identity shortcuts while preserving factual semantics, all on top of a **frozen multilingual encoder** (mE5-base) — requiring **< 2.5 GB VRAM** and training in **~25 minutes** on a free Google Colab T4 GPU.

### Key Contributions

| # | Contribution |
|---|-------------|
| 1 | **Counterfactual Consistency Reward** — RL reward signal that penalizes prediction changes when speaker identity is swapped, forcing the model to rely on claim semantics |
| 2 | **Feature Gating Agent** — Trainable sigmoid gates (< 1.5% of total params) that modulate frozen embeddings per-dimension |
| 3 | **Zero-Evidence Leakage Protocol** — Strict data sanitization pipeline eliminating fact-checker commentary from training |
| 4 | **Cross-Lingual Evaluation** — Benchmarked on 257k+ multilingual fact-checks spanning 50+ languages, with focus on Urdu/Hindi/Bengali |

---

## 📊 Dataset

**FactCheck Insights (FCI) Corpus** — Duke Reporters' Lab aggregation of structured `ClaimReview` markup from fact-checking organizations worldwide.

| Statistic | Value |
|-----------|-------|
| Total raw records | 257,877 |
| Unique fact-checking orgs | 300+ |
| Languages detected | 50+ |
| South Asian claims (ur/hi/bn) | *Reported after Phase 1* |
| Silver cross-lingual pairs | *Reported after Phase 1* |
| Train split (≤ Dec 2023) | *Reported after Phase 1* |
| Test split (≥ Jan 2024) | *Reported after Phase 1* |

### Label Taxonomy

Raw verdict labels from 50+ languages (English, Portuguese, Spanish, German, Hindi, Urdu, etc.) are normalized to a **3-class taxonomy**:

| Class | Examples | Description |
|-------|----------|-------------|
| **TRUE** | `true`, `correct`, `verdadeiro`, `richtig`, `سچ` | Claim is factually accurate |
| **FALSE** | `false`, `fake`, `falso`, `falsch`, `جھوٹ` | Claim is factually incorrect |
| **MIXED** | `half true`, `misleading`, `out of context`, `भ्रामक` | Partially true, missing context, or exaggerated |

### Zero-Evidence Leakage

To prevent the model from memorizing fact-checker explanations (which trivially reveal the label), we permanently drop:
- `reviewRating.ratingExplanation` — Fact-checker's written verdict justification
- `reviewRating.ratingValue` — Numeric rating score (direct label proxy)

---

## 🏗️ Architecture {#architecture}

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        CL-SDRG Pipeline                            │
 │                                                                     │
 │  Input:  Claim (q)  ·  Speaker (s)  ·  Date (t)                   │
 │              │              │              │                        │
 │              ▼              ▼              ▼                        │
 │  ┌───────────────────────────────────────────────────┐              │
 │  │         Frozen mE5-base Encoder (278M params)     │  🔒 Locked  │
 │  │         All parameters: requires_grad = False     │              │
 │  └───────────────┬──────────┬──────────┬─────────────┘              │
 │                  │          │          │                             │
 │                 E_q        E_s        E_t       (768-dim each)     │
 │                  │          │          │                             │
 │                  └────┬─────┴─────┬───┘                             │
 │                       │           │                                 │
 │                       ▼           ▼                                 │
 │  ┌────────────────────────┐  ┌──────────────────┐                   │
 │  │  Feature Gating Agent  │  │  Direct Embeds   │                   │
 │  │  [E_q;E_s;E_t] → MLP  │  │  (E_q, E_s, E_t) │                  │
 │  │  → Sigmoid → (α_q,    │  │                  │                   │
 │  │     α_s, α_t)         │  │                  │                   │
 │  │  ~1.5M trainable      │  │                  │                   │
 │  └────────┬───────────────┘  └──────┬───────────┘                   │
 │           │                         │                               │
 │           └────────────┬────────────┘                               │
 │                        ▼                                            │
 │           ┌────────────────────────┐                                │
 │           │    Gated Fusion        │                                │
 │           │ E_gated = α_q⊙E_q +   │                                │
 │           │   α_s⊙E_s + α_t⊙E_t  │                                │
 │           └────────────┬───────────┘                                │
 │                        ▼                                            │
 │           ┌────────────────────────┐                                │
 │           │  Veracity Classifier   │                                │
 │           │  LN → FC → ReLU →     │                                │
 │           │  Dropout → FC → logits │                                │
 │           └────────────┬───────────┘                                │
 │                        ▼                                            │
 │              {TRUE, FALSE, MIXED}                                   │
 └─────────────────────────────────────────────────────────────────────┘
```

### Model Parameters

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Frozen mE5-base | ~278M | ❌ (0%) |
| Feature Gating Agent (FGA) | ~1.2M | ✅ |
| Veracity Classifier | ~0.2M | ✅ |
| **Total Trainable** | **~1.4M** | **< 0.5% of total** |

---

## 🎯 Methodology {#methodology}

### 1. Counterfactual Data Augmentation

For each training sample $(q_i, s_i, t_i, y_i)$, we generate a **counterfactual** by replacing the speaker $s_i$ with a randomly sampled speaker $s_i'$, keeping the claim and label invariant:

$$
(q_i, s_i, t_i, y_i) \rightarrow (q_i, s_i', t_i, y_i)
$$

### 2. REINFORCE Policy Gradient

The FGA's gating decisions are treated as a stochastic policy. We optimize with three reward components:

**Accuracy Reward:**
$$R_{\text{acc}} = \begin{cases} +1.0 & \text{if } \hat{y}_i = y_i \\ -1.0 & \text{otherwise} \end{cases}$$

**Consistency Reward** (novel):
$$R_{\text{cons}} = 1.0 - \|\hat{P}(y \mid q, s) - \hat{P}(y \mid q, s')\|_1$$

**Combined Reward:**
$$R_{\text{total}} = 0.6 \cdot R_{\text{acc}} + 0.4 \cdot R_{\text{cons}}$$

### 3. Variance Reduction

We use an exponential moving average baseline ($\beta = 0.99$) for variance reduction:
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}\left[(R_{\text{total}} - b) \cdot \log \pi_\theta(a \mid s)\right]$$

### 4. Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Physical Batch Size | 16 |
| Gradient Accumulation | 16 steps |
| Virtual Batch Size | 256 |
| Epochs | 10 |
| Precision | FP16 mixed |
| Auxiliary CE Loss Weight | 0.5 |
| Peak VRAM | < 2.5 GB |
| Training Time (Colab T4) | ~25 min |

---

## 📈 Results {#results}

> **Note:** Run the Colab notebook and paste results here after execution.

### Classification Performance (Test Set: 2024–2026)

| Method | Accuracy | Macro-F1 | Precision | Recall | SFR ↓ |
|--------|----------|----------|-----------|--------|-------|
| **CL-SDRG (Ours)** | `—` | `—` | `—` | `—` | `—` |
| BM25 Lexical | `—` | `—` | `—` | `—` | N/A |
| Zero-Shot kNN (mE5) | `—` | `—` | `—` | `—` | N/A |

### Cross-Lingual Retrieval Metrics

| K | Recall@K | MRR@K | nDCG@K |
|---|----------|-------|--------|
| 1 | `—` | `—` | `—` |
| 5 | `—` | `—` | `—` |
| 20 | `—` | `—` | `—` |

### Shortcut Bias Audit — Speaker Flip Rate (SFR)

$$\text{SFR} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}\left(\hat{y}(q_i, s_i) \neq \hat{y}(q_i, s_i')\right)$$

| Metric | Value | Target |
|--------|-------|--------|
| Speaker Flip Rate | `—` | < 3% |
| Perturbations per sample | 10 | — |
| Status | `—` | — |

### Training Curves

> *Plots will be generated in `outputs/figures/` after running the Colab notebook.*

---

## 🚀 Quick Start {#quick-start}

### Option 1: Google Colab (Recommended)

1. Open [`CL_SDRG_Full_Pipeline.ipynb`](notebooks/CL_SDRG_Full_Pipeline.ipynb) in Google Colab
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Upload `claim_review.csv` when prompted
4. Run all cells (~45–60 min total)
5. Download results from `outputs/`

### Option 2: Local Execution

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/CL-SDRG.git
cd CL-SDRG

# Install dependencies
pip install -r requirements.txt

# Place dataset
mkdir -p "Fact Check Dataset"
# Copy claim_review.csv into "Fact Check Dataset/"

# Run pipeline
python src/01_data_ingestion_and_lid.py
python src/02_fga_architecture_and_vram_test.py
python src/03_train_rl_debiasing.py
python src/04_evaluation_and_ablation.py
```

---

## 📁 Repository Structure

```
CL-SDRG/
├── .gitignore
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── implementation_plan_cl_sdrg.md     ← Research roadmap
│
├── src/                               ← Core Python modules
│   ├── __init__.py
│   ├── config.py                      ← Hyperparameters, paths, label maps
│   ├── utils.py                       ← Logging, seeding, device, checkpoints
│   ├── 01_data_ingestion_and_lid.py   ← Phase 1: Data engineering
│   ├── 02_fga_architecture_and_vram_test.py  ← Phase 2: Architecture
│   ├── 03_train_rl_debiasing.py       ← Phase 3: RL training
│   └── 04_evaluation_and_ablation.py  ← Phase 4: Evaluation
│
├── notebooks/
│   └── CL_SDRG_Full_Pipeline.ipynb    ← Single Colab notebook (all phases)
│
└── outputs/                           ← Generated at runtime (git-ignored)
    ├── processed_data/                ← Cleaned CSVs, splits
    ├── checkpoints/                   ← Model .pt files
    ├── logs/                          ← Training history
    └── figures/                       ← Plots and visualizations
```

---

## ⚙️ Technical Details

### Language Identification

We use [FastText lid.176](https://fasttext.cc/docs/en/language-identification.html) to detect claim language. South Asian language codes targeted:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `ur` | Urdu | `ta` | Tamil |
| `hi` | Hindi | `te` | Telugu |
| `bn` | Bengali | `ml` | Malayalam |
| `pa` | Punjabi | `mr` | Marathi |
| `sd` | Sindhi | `gu` | Gujarati |
| `ne` | Nepali | `si` | Sinhala |

### Silver Pair Mining

Cross-lingual claim pairs are mined from multi-edition fact-checking organizations (AFP, Fact Crescendo, BOOM Live) by matching publications within a **±3 day window** across different language editions.

### Smoke Test Gate S₁

If natural Urdu claims < 500, the pipeline flags for **NLLB-200 translation fallback** to generate synthetic parallel test queries.

---

## 🔒 Data Safety & Ethics

- **Zero-Evidence Leakage:** Fact-checker explanations are permanently dropped to prevent label memorization
- **Time-Aware Evaluation:** Train/test split by publication date prevents temporal data leakage
- **Shortcut Auditing:** SFR metric quantifies and mitigates identity-based classification bias
- **No PII:** Only public fact-check records from the Duke Reporters' Lab are used

---

## 📄 Citation {#citation}

If you use this work, please cite:

```bibtex
@article{clsdrg2026,
  title     = {Cross-Lingual Shortcut De-biasing via Reinforcement Learning 
               Gating on Frozen Transformer Representations},
  author    = {Your Name},
  year      = {2026},
  note      = {GitHub: https://github.com/YOUR_USERNAME/CL-SDRG}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 🔬 for robust cross-lingual fact verification
</p>
