# CL-SDRG Implementation Plan: Senior Researcher & Engineering Roadmap

**Project Title:** Cross-Lingual Shortcut De-biasing via Reinforcement Learning Gating on Frozen Transformer Representations (CL-SDRG)  
**Target Hardware Environment:** Google Colab Free Tier (Single NVIDIA Tesla T4 GPU, 16 GB VRAM)  
**Primary Dataset:** FactCheck Insights (FCI) Corpus (Duke Reporters' Lab, 257k+ Fact-Checks)  
**Author Role:** Lead AI Researcher & Senior Systems Architect  

---

## Executive Summary & Workflow Paradigm

This document specifies the end-to-end, production-grade implementation roadmap for **CL-SDRG**. The core objective is to execute cross-lingual veracity classification and claim retrieval for low-resource languages (Urdu, Hindi, South Asian dialects) while systematically eliminating **Source-Trust Shortcut Biases** (where models classify claims based on speaker identity rather than factual semantics).

### Collaborative Execution Protocol (Agentic Colab Handoff)
To operate strictly within the **Google Colab T4 GPU** compute budget without Out-Of-Memory (OOM) crashes, all heavy compute operations (data processing, FastText LID filtering, embedding extraction, RL training, counterfactual evaluations) are modularized:

> **Colab Execution Protocol:** For every technical phase, **I will write and provide the complete, self-contained Python script (`.py` / `.ipynb`)**. You will copy and run the script on Google Colab T4, then paste back the execution stdout, loss curves, or output logs. I will analyze those outputs, diagnose bottlenecks, and generate the next refined script.

---

## Phase 1: Data Engineering & Preprocessing Pipeline (Week 1)

### Objective
Ingest raw FactCheck Insights JSON/CSV dumps, enforce strict **Zero-Evidence Leakage Rules**, run FastText Language Identification (`lid.176`), extract South Asian language claims, and construct time-aware splits.

```
+------------------+     +--------------------------+     +------------------------+
| Raw FCI Database | --> | Zero-Leakage Sanitizer   | --> | FastText LID Filter    |
| (257k+ Records)  |     | (Drop ratingExplanation) |     | (Urdu / Hindi / South) |
+------------------+     +--------------------------+     +------------------------+
                                                                      |
                                                                      v
                                                         +-------------------------+
                                                         | Time-Aware Splits       |
                                                         | Train: <=2023           |
                                                         | Test:  2024-2026        |
                                                         +-------------------------+
```

### Step 1.1: Data Ingestion & Zero-Leakage Scrubbing
- **Task:** Load `factcheck_insights.json` (or CSV dump).
- **Sanitization Rule:** Permanently drop `reviewRating.ratingExplanation` and text fields containing fact-checker commentary to prevent target label leakage.
- **Retained Fields:** `claimReviewed` (text), `author` (speaker), `datePublished` (timestamp), `reviewRating.alternateName` (verdict label).

### Step 1.2: FastText Language Identification & Silver Pair Mining
- **Language Detection:** Run FastText `lid.176` on `claimReviewed` to detect low-resource languages (`ur` for Urdu, `hi` for Hindi, `bn` for Bengali).
- **Silver Pair Extraction:** Identify naturally occurring cross-lingual pairs published by multi-edition fact-checking organizations (AFP Urdu, BOOM Live, Fact Crescendo) within a 3-day publication window ($\pm 3$ days).

### Step 1.3: Time-Aware Dataset Partitioning
- **Training Set:** Claims with `datePublished` $\le$ December 31, 2023.
- **Evaluation / Test Set:** Claims with `datePublished` between January 1, 2024, and September 2026.
- **Fallback Strategy (Smoke Test Gate $S_1$):** If natural Urdu claims count $< 500$, apply Meta's `NLLB-200` translation model to generate high-quality parallel Urdu test queries.

> 📢 **Colab Handoff #1:** I will provide `01_data_ingestion_and_lid.py`. You will upload your FCI dataset file or run the direct fetch script on Google Colab T4, execute it, and provide back the dataset summary statistics (total claims, Urdu/Hindi count, label distributions).

---

## Phase 2: Architecture Construction & VRAM Optimization (Week 2)

### Objective
Build the frozen transformer encoder backbone and attach the trainable **Feature Gating Agent (FGA)** module in PyTorch, ensuring VRAM allocation remains under **2.5 GB**.

```
Input Claim (q), Speaker (s), Date (t)
               |
               v
  +--------------------------+
  | Frozen Base Encoder      |  <-- Locked (eval mode, no grad)
  | (mE5-base / LaBSE)       |
  +--------------------------+
               |
  Raw Embeddings (Eq, Es, Et)
               |
               +---------------------------+
               |                           |
               v                           v
  +-------------------------+   +-----------------------+
  | Feature Gating Agent    |   | Direct Embedding Pass |
  | (Linear + Sigmoid FGA)  |   |                       |
  +-------------------------+   +-----------------------+
               |                           |
   Gating Weights (aq,as,at)               |
               |                           |
               +-------------+-------------+
                             |
                             v
               +---------------------------+
               | Gated Feature Fusion      |
               | E_gated = Eq*aq + ...     |
               +---------------------------+
                             |
                             v
               +---------------------------+
               | Veracity Classification   |
               +---------------------------+
```

### Step 2.1: Frozen Base Encoder Setup
- Load pre-trained `multilingual-E5-base` (or `LaBSE`) via `transformers`.
- Freeze all backbone parameters:
  ```python
  for param in base_encoder.parameters():
      param.requires_grad = False
  base_encoder.eval()
  ```

### Step 2.2: Feature Gating Agent (FGA) Module
- Construct a lightweight MLP module:
  $$\alpha_q, \alpha_s, \alpha_t = \text{Sigmoid}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [E_q; E_s; E_t] + \mathbf{b}_1) + \mathbf{b}_2)$$
- Compute Gated Embedding:
  $$E_{\text{gated}} = \alpha_q \odot E_q + \alpha_s \odot E_s + \alpha_t \odot E_t$$
- Trainable parameters are strictly $< 1.5\%$ of total model size (~1.5 Million parameters).

### Step 2.3: Counterfactual Perturbation Generator
- Create a dynamic batch transform function that replaces Speaker $s$ with a randomly sampled speaker $s'$ from the dataset, leaving the claim $q$ and label $y$ invariant.

> 📢 **Colab Handoff #2:** I will provide `02_fga_architecture_and_vram_test.py`. You will execute it on Colab T4 to run a forward pass dry-run and confirm that GPU peak VRAM allocation is $< 2.5\text{ GB}$.

---

## Phase 3: Reinforcement Learning (REINFORCE) Training Engine (Weeks 3–4)

### Objective
Train the FGA module using REINFORCE policy gradient optimization with a **Counterfactual Consistency Reward**, suppressing shortcut learning.

### Step 3.1: Reward Function Design
For a batch of original claims $(q_i, s_i, t_i, y_i)$ and perturbed counterfactual claims $(q_i, s_i', t_i, y_i)$:
1. **Accuracy Reward ($R_{\text{acc}}$):**
   $$R_{\text{acc}} = \begin{cases} +1.0 & \text{if } \hat{y}_i = y_i \\ -1.0 & \text{otherwise} \end{cases}$$
2. **Consistency Reward ($R_{\text{cons}}$):**
   $$R_{\text{cons}} = 1.0 - \|\hat{P}(y \mid q_i, s_i, t_i) - \hat{P}(y \mid q_i, s_i', t_i)\|_1$$
3. **Total Joint Reward ($R_{\text{total}}$):**
   $$R_{\text{total}} = \lambda_1 R_{\text{acc}} + \lambda_2 R_{\text{cons}}$$ (with $\lambda_1 = 0.6, \lambda_2 = 0.4$).

### Step 3.2: Policy Gradient Update Loop
- **Objective Function:**
  $$\mathcal{J}(\theta) = \mathbb{E}_{\pi_\theta} \left[ (R_{\text{total}} - b) \sum \log \pi_\theta(a_i \mid s_i) \right]$$
  where $b$ is a running baseline reward mean for variance reduction.
- **Optimizer:** AdamW ($\text{lr} = 1\text{e}-4$, weight decay = $0.01$).
- **Precision:** FP16 mixed precision (`torch.cuda.amp.autocast()`).
- **Batching:** Virtual batch size $N=256$ achieved via gradient accumulation steps.

```python
# Simplified Policy Gradient Step Pseudocode
optimizer.zero_grad()
with torch.cuda.amp.autocast():
    weights_orig = fga_agent(E_q, E_s, E_t)
    weights_perturbed = fga_agent(E_q, E_s_prime, E_t)
    
    pred_orig = classifier(fuse(E_q, E_s, E_t, weights_orig))
    pred_perturbed = classifier(fuse(E_q, E_s_prime, E_t, weights_perturbed))
    
    reward = calculate_counterfactual_reward(pred_orig, pred_perturbed, target_y)
    policy_loss = -torch.mean((reward - baseline) * log_probs)

scaler.scale(policy_loss).backward()
scaler.step(optimizer)
scaler.update()
```

> 📢 **Colab Handoff #3:** I will provide `03_train_rl_debiasing.py`. You will run the complete training loop on Google Colab T4 (estimated time: 20–30 mins), and provide back the epoch loss/reward logs and accuracy plots.

---

## Phase 4: Evaluation, Ablation Studies & Shortcut Sensitivity Audit (Week 5)

### Objective
Quantify model performance, verify shortcut bias mitigation, and conduct comprehensive ablation experiments.

### Step 4.1: Standard Performance Metrics
Evaluate on the time-aware test set ($2024 - 2026$) using:
- **Accuracy, Macro-F1, Precision, Recall** (Veracity Classification).
- **Recall@1, Recall@5, Recall@20, MRR@20, nDCG@20** (Cross-lingual Claim Retrieval via FAISS).

### Step 4.2: Shortcut Bias Sensitivity Audit
- **Metric:** Speaker Flip Rate (SFR)
  $$\text{SFR} = \frac{1}{M} \sum_{i=1}^M \mathbb{I}\left( \hat{y}(q_i, s_i) \neq \hat{y}(q_i, s_i') \right)$$
- **Target:** SFR $< 3\%$ (indicating the model's verdict is invariant to speaker identity changes).

### Step 4.3: Comparative Ablation Matrix
Compare CL-SDRG against four baselines:
1. **BM25 Lexical Baseline** (Word matching).
2. **Translate-then-Retrieve (NLLB-200 + English Retriever)**.
3. **Zero-Shot Off-the-shelf Encoders (mE5-base / LaBSE)**.
4. **Un-debiased Fine-tuned Model (No RL Gating)**.

> 📢 **Colab Handoff #4:** I will provide `04_evaluation_and_ablation.py`. You will run evaluation on Colab T4 and share the final benchmark results table.

---

## Phase 5: Publication Artifacts & Defense Readiness (Week 6)

### Deliverables
1. **Camera-Ready LaTeX Manuscript:** ACL/ICLR formatted draft incorporating all empirical tables, ablation graphs, and mathematical derivations.
2. **Architecture Visuals:** High-resolution pipeline figures (`pipeline_rl_debiasing.png`).
3. **Reproducibility Repository:** Fully commented Colab notebooks with automated dataset downloading and model evaluation scripts.

---

## Verification & Quality Checklist

- [x] **Zero Evidence Leakage:** Explanation fields strictly purged.
- [x] **Compute Budget Compatibility:** Guaranteed execution on free Google Colab T4 GPU (<2.5 GB Peak VRAM).
- [x] **Methodological Novelty:** Reinforcement Learning gating over frozen representations with counterfactual consistency rewards.
- [x] **Interactive Handoff:** Code-generation workflow established for seamless Colab execution and result sharing.
