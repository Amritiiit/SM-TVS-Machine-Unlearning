# Saliency-Masked Task Vector Subtraction (SM-TVS) for LLM Machine Unlearning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **Saliency-Masked Task Vector Subtraction (SM-TVS)**: a parameter-saliency-gated weight negation framework for verifiable, non-destructive machine unlearning in Large Language Models.

---

## Project Structure

```
├── configs/
│   ├── model_0.5b.yaml         # Hyperparameters for Qwen2.5-0.5B-Instruct
│   └── model_1.5b.yaml         # Hyperparameters for Qwen2.5-1.5B-Instruct
├── data/
│   └── download_tofu.py        # Automated downloader & cache for TOFU splits
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Formats QA token pairs with chat templates & loss masking
│   ├── lora_trainer.py         # Fits low-rank adapter on forget split Df
│   ├── saliency_engine.py      # Computes retain/forget gradient saliency & gates
│   ├── weight_manipulator.py   # Executes masked matrix multiplication & weight folding
│   └── metrics.py              # Evaluates PPL, EM, and Min-K% MIA
├── experiments/
│   └── run_smtvs.py            # Executes end-to-end SM-TVS pipeline
├── results/
│   ├── metrics_summary.json    # Aggregate numerical results
│   └── plots/                  # Visual evaluation plots
├── PLAN.md                     # Comprehensive 28-day research roadmap & math specification
├── progress_report.tex         # LaTeX source of progress report
├── progress_report.pdf         # 2-Page compiled PDF progress report
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/selective-unlearning-llm.git
cd selective-unlearning-llm

# Install dependencies
pip install -r requirements.txt
```

### 2. Pre-download & Cache Datasets

```bash
python data/download_tofu.py
```

### 3. Run SM-TVS Unlearning Experiment

Run on `Qwen/Qwen2.5-0.5B-Instruct` (fast prototyping on 8 GB VRAM):

```bash
python experiments/run_smtvs.py --config configs/model_0.5b.yaml
```

Run on `Qwen/Qwen2.5-1.5B-Instruct` (benchmark reporting):

```bash
python experiments/run_smtvs.py --config configs/model_1.5b.yaml
```

---

## Evaluation Pillars

The framework automatically logs and compares models before and after unlearning across:
1. **Pillar 1: Forget Efficacy** — Perplexity on $\mathcal{D}_f$ completions and Rogue Extraction Exact Match (EM).
2. **Pillar 2: Model Utility** — Question Answering accuracy on `retain99` and WikiText-2 linguistic perplexity.
3. **Pillar 3: Re-learning Resistance** — Gradient steps needed to re-acquire erased knowledge under 1-epoch fine-tuning.
4. **Pillar 4: Privacy / MIA** — Min-K% Prob ROC-AUC score (Target: $\approx 0.50$, indistinguishable from unseen data).

---

## Authors & Team

- **Amrit Sagar** (`2023UG3028`) — *Team Lead / Algorithmic & Mathematical Modeling*
- **Vaibhav Bansal** (`2023UG3012`) — *Data Pipeline & Evaluation*
- **Lakhan Sharma** (`2023UG1045`) — *Data Pipeline & Evaluation*
