#!/usr/bin/env python3
"""
Three high-impact analyses (no GPU needed):
  1. Bootstrap 95% CIs for every Table 1 R1 cell from stored per-query CSVs.
  2. Holm-Bonferroni correction of the 8 Table 4 pairwise tests.
  3. Multi-seed discrepancy investigation:
     recompute R1 from the paper's stored SDCP-v2 MMLU CSV and compare it
     to the May-17 multi-seed JSON (41.82) and the in-paper number (35.44).

All p-values use 10 000-resample paired bootstrap with seed=42 for
reproducibility (same seed already used in the paper).
"""

import os, json, ast
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer as rs

OUT  = './outputs'
SEED = 42
NBOOT = 10000
NCI   = 10000

scorer = rs.RougeScorer(['rouge1'], use_stemmer=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_ref(r):
    """Reference can be a string, list-literal '[...]', or numpy-array repr.
    Return a list of non-empty reference strings."""
    if pd.isna(r): return []
    if isinstance(r, list): return [x for x in r if x]
    s = str(r).strip()
    # try literal_eval for list-strings
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [x for x in v if x]
        if isinstance(v, str) and v: return [v]
    except Exception:
        pass
    return [s] if s else []

def rouge1_per_query(generated, references):
    out = []
    for g, refs in zip(generated, references):
        refs = parse_ref(refs)
        best = 0.0
        for r in refs:
            if not r: continue
            try:
                best = max(best, scorer.score(str(r), str(g))['rouge1'].fmeasure)
            except Exception:
                pass
        out.append(best * 100)
    return np.asarray(out, dtype=float)

def bootstrap_ci(scores, n_boot=NCI, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    n = len(scores)
    boots = np.array([rng.choice(scores, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    return float(scores.mean()), float(lo), float(hi)

def paired_bootstrap(a, b, n_boot=NBOOT, seed=SEED):
    """Two-sided paired bootstrap on the mean difference (a - b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = a - b
    observed = diffs.mean()
    centered = diffs - observed
    boots = np.array([rng.choice(centered, size=len(centered), replace=True).mean()
                      for _ in range(n_boot)])
    # one-sided p (boots are under H0: mean=0)
    if observed > 0:
        p = float(np.mean(boots >= observed))
    else:
        p = float(np.mean(boots <= observed))
    # two-sided
    p = min(1.0, 2 * p)
    return float(observed), float(p)

def holm_bonferroni(pvals, labels, alpha=0.05):
    """Returns sorted (label, raw_p, adj_p, significant) using Holm step-down."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]
        running_max = max(running_max, a)
        adj[idx] = min(1.0, running_max)
    out = []
    for i, lab in enumerate(labels):
        out.append((lab, float(pvals[i]), float(adj[i]), bool(adj[i] < alpha)))
    return out

# ──────────────────────────────────────────────────────────────────────────────
#  Load every per-query CSV
# ──────────────────────────────────────────────────────────────────────────────
print('='*78)
print(' Loading per-query CSVs')
print('='*78)
F = {
    'SDCP_v2_TQA':    'SDCP_v2_TQA_20260506_091427.csv',
    'SDCP_v2_MMLU':   'SDCP_v2_MMLU_20260506_091427.csv',
    'SDCP_v2_ARC':    'SDCP_v2_ARC_20260506_091427.csv',
    'CFR_TQA':        'CFR_TQA_full4bit_20260429_210909.csv',
    'CFR_MMLU':       'CFR_MMLU_full4bit_20260429_210909.csv',
    'CFR_ARC':        'CFR_ARC_full4bit_20260429_210909.csv',
    'CAFD_TQA':       'CAFD_TQA_full4bit_20260429_210909.csv',
    'CAFD_MMLU':      'CAFD_MMLU_full4bit_20260429_210909.csv',
    'CAFD_ARC':       'CAFD_ARC_full4bit_20260429_210909.csv',
    'ICL1D_ARC':      'ICL1D_ARC_full4bit_20260429_210909.csv',
    'SDCP_ARC':       'SDCP_ARC_full4bit_20260429_210909.csv',
    'Base_ARC':       'ARC_Base_20260430_085946.csv',
    'HyDE_TQA':       'hyde_tqa_perquery_20260508_193452.csv',
}
data = {}
for k, fn in F.items():
    p = os.path.join(OUT, fn)
    df = pd.read_csv(p)
    data[k] = df
    print(f'  {k:14s}  n={len(df):4d}  cols={list(df.columns)[:5]}')

# HyDE is special: has r1_base and r1_hyde columns already
hyde_tqa = data['HyDE_TQA']
print(f'\n  HyDE_TQA pre-computed: r1_base mean={hyde_tqa["r1_base"].mean():.2f}'
      f'  r1_hyde mean={hyde_tqa["r1_hyde"].mean():.2f}')

# ──────────────────────────────────────────────────────────────────────────────
#  Recompute R1 per query for every CSV that lacks the column
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*78)
print(' Recomputing per-query R1 (using rouge_score with use_stemmer=True)')
print('='*78)
r1 = {}
for k, df in data.items():
    if 'r1_base' in df.columns or 'r1_hyde' in df.columns:
        continue
    r1[k] = rouge1_per_query(df['generated'], df['reference'])
    print(f'  {k:14s}  R1 mean={r1[k].mean():6.2f}  median={np.median(r1[k]):6.2f}')

# Pull HyDE+Base from the dedicated HyDE CSV
r1['HyDE_TQA']  = hyde_tqa['r1_hyde'].to_numpy()
r1['Base_TQA']  = hyde_tqa['r1_base'].to_numpy()
print(f'  {"HyDE_TQA":14s}  R1 mean={r1["HyDE_TQA"].mean():6.2f}')
print(f'  {"Base_TQA":14s}  R1 mean={r1["Base_TQA"].mean():6.2f}')

# ──────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 1 -- Bootstrap 95% CIs for every method × dataset
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*78)
print(' ANALYSIS 1  --  Bootstrap 95% Confidence Intervals (10k resamples)')
print('='*78)
ci_rows = []
print(f'{"Method × Dataset":<22} {"R1":>8} {"95% CI":>20} {"width":>8}')
print('-'*60)
for key in ['Base_TQA','HyDE_TQA','CFR_TQA','CAFD_TQA','SDCP_v2_TQA',
            'CFR_MMLU','CAFD_MMLU','SDCP_v2_MMLU',
            'Base_ARC','ICL1D_ARC','CFR_ARC','CAFD_ARC','SDCP_ARC','SDCP_v2_ARC']:
    if key not in r1: continue
    mean, lo, hi = bootstrap_ci(r1[key])
    ci_rows.append({'method_dataset': key, 'mean': mean, 'ci_low': lo,
                    'ci_high': hi, 'width': hi-lo})
    print(f'{key:<22} {mean:>8.2f}   [{lo:5.2f}, {hi:5.2f}]   {hi-lo:>6.2f}')

# ──────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 2 -- Paired bootstrap on the 8 Table 4 pairwise comparisons
#              plus the new HyDE vs SDCP-v2 on TQA / MMLU
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*78)
print(' ANALYSIS 2  --  Re-running 8 paired comparisons + Holm-Bonferroni')
print('='*78)
comparisons = [
    # (label, scores_a, scores_b)
    ('HyDE vs Base @ TQA',        r1['HyDE_TQA'],   r1['Base_TQA']),
    ('SDCP-v2 vs HyDE @ TQA',     r1['SDCP_v2_TQA'], r1['HyDE_TQA']),
    ('SDCP-v2 vs CFR @ TQA',      r1['SDCP_v2_TQA'], r1['CFR_TQA']),
    ('CAFD vs SDCP-v2 @ TQA',     r1['CAFD_TQA'],   r1['SDCP_v2_TQA']),
    ('CFR vs SDCP-v2 @ MMLU',     r1['CFR_MMLU'],   r1['SDCP_v2_MMLU']),
    ('CAFD vs SDCP-v2 @ MMLU',    r1['CAFD_MMLU'],  r1['SDCP_v2_MMLU']),
    ('Base vs SDCP-v2 @ ARC',     r1['Base_ARC'],   r1['SDCP_v2_ARC']),
]
# SDCP-v2 vs HyDE @ MMLU has no HyDE_MMLU CSV available -- skip or note
# (the paper lists it; we only have HyDE TQA per-query)

labels, deltas, pvals = [], [], []
for label, a, b in comparisons:
    if len(a) != len(b):
        print(f'  WARN length mismatch {label}: {len(a)} vs {len(b)} -- truncating')
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d, p = paired_bootstrap(a, b)
    labels.append(label); deltas.append(d); pvals.append(p)
    print(f'  {label:<32}  Δ={d:+6.3f}   raw p={p:.4f}')

print('\n  Holm-Bonferroni step-down correction (α=0.05):')
res = holm_bonferroni(pvals, labels)
print(f'  {"Comparison":<32} {"raw p":>10} {"adj p":>10} {"Sig.":>6}')
print('  ' + '-'*60)
for lab, raw, adj, sig in res:
    mark = 'YES' if sig else ' no'
    print(f'  {lab:<32}  {raw:>9.4f}  {adj:>9.4f}   {mark}')

# ──────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3 -- Multi-seed discrepancy investigation
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*78)
print(' ANALYSIS 3  --  Why does multi_seed_results 2026-05-17 disagree with paper?')
print('='*78)

with open('./multi_seed_results_20260517_032944.json') as f:
    multi_seed = json.load(f)
print(f"  Multi-seed (May 17): TQA = {multi_seed['summary']['TQA_R1']['mean']:.2f} "
      f"± {multi_seed['summary']['TQA_R1']['std']:.2f}   "
      f"MMLU = {multi_seed['summary']['MMLU_R1']['mean']:.2f} "
      f"± {multi_seed['summary']['MMLU_R1']['std']:.2f}")

print(f'\n  Paper Table 1 (single-seed=42):')
print(f'    TQA R1 = 35.59   MMLU R1 = 35.44')

print(f'\n  Recomputed from stored CSV (paper run, seed=42):')
print(f'    TQA R1  = {r1["SDCP_v2_TQA"].mean():.2f}   '
      f'(matches paper Table 1: yes/no)')
print(f'    MMLU R1 = {r1["SDCP_v2_MMLU"].mean():.2f}')
print(f'    ARC R1  = {r1["SDCP_v2_ARC"].mean():.2f}')

# Direct comparison: the May-17 multi-seed reports seed=42 TQA=36.76 / MMLU=42.41
# Paper's stored CSV (also seed=42) should match if same protocol
seed42_tqa  = multi_seed['results']['42']['TQA']['R1']
seed42_mmlu = multi_seed['results']['42']['MMLU']['R1']
paper_tqa   = r1['SDCP_v2_TQA'].mean()
paper_mmlu  = r1['SDCP_v2_MMLU'].mean()
print(f'\n  Same-seed (42) comparison:')
print(f'    TQA  multi-seed={seed42_tqa:.2f}   paper-CSV={paper_tqa:.2f}   Δ={seed42_tqa-paper_tqa:+.2f}')
print(f'    MMLU multi-seed={seed42_mmlu:.2f}  paper-CSV={paper_mmlu:.2f}   Δ={seed42_mmlu-paper_mmlu:+.2f}')

print('\n  Diagnosis hint:')
if abs(seed42_tqa - paper_tqa) < 1.5 and abs(seed42_mmlu - paper_mmlu) > 4:
    print('    TQA matches (<1.5 R1 drift) but MMLU diverges (>4 R1).')
    print('    Likely an MMLU-specific protocol difference:')
    print('      - different prompt format (formatted-question vs plain)')
    print('      - different scoring direction (predict letter vs predict choice text)')
    print('      - different ROUGE arg order')
    print('    Open multi_seed_sdcp.py vs SDCP_v2_AlwaysUncertain.ipynb side-by-side')
    print('    to confirm which protocol is correct.')

# ──────────────────────────────────────────────────────────────────────────────
#  Persist
# ──────────────────────────────────────────────────────────────────────────────
out = {
    'bootstrap_ci_table': ci_rows,
    'holm_bonferroni': [
        {'label': lab, 'raw_p': raw, 'adj_p': adj, 'significant': sig}
        for lab, raw, adj, sig in res
    ],
    'multi_seed_diagnosis': {
        'multi_seed_seed42_TQA_R1':  seed42_tqa,
        'multi_seed_seed42_MMLU_R1': seed42_mmlu,
        'paper_csv_seed42_TQA_R1':   round(paper_tqa, 2),
        'paper_csv_seed42_MMLU_R1':  round(paper_mmlu, 2),
        'paper_csv_seed42_ARC_R1':   round(r1['SDCP_v2_ARC'].mean(), 2),
        'paper_table1_TQA':  35.59,
        'paper_table1_MMLU': 35.44,
        'multi_seed_mean_TQA':  multi_seed['summary']['TQA_R1'],
        'multi_seed_mean_MMLU': multi_seed['summary']['MMLU_R1'],
    },
}
out_path = './outputs/stat_analysis_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\n  Saved: {out_path}')
print('\n' + '='*78)
print(' Done.')
print('='*78)
