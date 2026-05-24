# SDCP: Self-Distilled Contrastive Priors for Label-Free RAG

Code for the paper **"SDCP: Self-Distilled Contrastive Priors for Label-Free Retrieval-Augmented Generation"**, under anonymous peer review.

> **Note:** This repository is anonymized for peer review. Author information will be added upon acceptance.

---

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA 11.8+)
- transformers >= 4.39
- sentence-transformers
- rouge-score, scipy, numpy, pandas, tqdm

```bash
pip install torch transformers sentence-transformers rouge-score scipy numpy pandas tqdm
```

---

## Models

| Experiment | Model | Quantization |
|-----------|-------|-------------|
| Main results (Table 1) | Mistral-7B-Instruct-v0.2 | 4-bit NF4 (bitsandbytes) |
| Generalization (Table 2) | Mistral-7B-Instruct-v0.2 | 4-bit NF4 |

---

## Repository Structure

```
├── notebooks/          # Jupyter notebooks for all experiments
│   ├── SDCP_Method.ipynb               # Core SDCP method
│   ├── Full_Dataset_4bit.ipynb         # Main results (Table 1)
│   ├── SDCP_v2_AlwaysUncertain.ipynb   # SDCP-v2 variant
│   ├── SDCP_HyDE_Comparison.ipynb      # HyDE baseline comparison
│   ├── ARC_Base_Baseline.ipynb         # Base RAG on ARC-Challenge
│   ├── ICL1D_Plus_Baseline.ipynb       # Li et al. ICL baselines
│   ├── SDCP_Ablation.ipynb             # Component ablation (Table 3)
│   ├── SDCP_HyperparamAblation.ipynb   # Hyperparameter sensitivity
│   ├── SDCP_WikipediaKB.ipynb          # Wikipedia KB generalization
│   └── CFR_CAFD_WikipediaKB.ipynb      # CFR/CAFD-LC with Wikipedia KB
│
└── scripts/            # Python scripts for reproducible runs
    ├── run_sdcp_v2.py                  # SDCP-v2 runner
    ├── ablation_sdcpv2.py              # Ablation on TQA/MMLU
    ├── ablation_arc_sdcpv2.py          # Ablation on ARC-Challenge
    ├── multi_seed_sdcp_v2_correct.py   # Multi-seed robustness (3 seeds)
    └── stat_analysis.py               # Bootstrap CIs + Holm-corrected p-values
```

---

## Reproducing Main Results (Table 1)

Run `notebooks/Full_Dataset_4bit.ipynb` for all methods on TruthfulQA, MMLU, and ARC-Challenge.

| Method | TQA R1 | MMLU R1 | ARC R1 |
|--------|--------|---------|--------|
| Base   | 26.81  | 10.42   | 42.33  |
| HyDE   | 31.04  | 30.89   | 42.76  |
| SDCP-v2 | **35.59** | **35.44** | 39.18 |

---

## Datasets

| Dataset | Split | Size | License |
|---------|-------|------|---------|
| TruthfulQA | test | 615 | Apache 2.0 |
| MMLU | test (28 subjects × 2) | 1,596 | MIT |
| ARC-Challenge | test | 1,172 | Apache 2.0 |

All datasets are publicly available benchmarks containing no personally identifiable information.

---

## Statistical Significance

Bootstrap confidence intervals and Holm-Bonferroni corrected p-values are computed in `scripts/stat_analysis.py`.

---

## Notes

- Update model checkpoint paths in notebooks before running (search for `/path/to/model`).
- All scripts use a fixed random seed for reproducibility.
- Notebook outputs are cleared for anonymization.
