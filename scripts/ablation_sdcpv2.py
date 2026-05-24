"""
Ablation study of SDCP-v2 (the recommended method).

Current ablation in the paper uses SDCP-v1 as the reference row.
This script ablates SDCP-v2 directly:
    1. SDCP-v2 (full)                   α=0.45 β=0.35 γ=0.20
    2. SDCP-v2 w/o P_neg (γ=0)         α=0.45 β=0.35 γ=0.00
    3. SDCP-v2 w/o P_pos (β=0)         α=0.45 β=0.00 γ=0.20
    4. SDCP-v2 w/o both (α=1 β=0 γ=0)  pure query-similarity only (= Base RAG)

All use seed=42 and the same-dataset KB (same as main experiments).

Usage:
    python ablation_sdcpv2.py

Runtime: ~4 × (21 min TQA + 55 min MMLU) ≈ ~6 hours on A100.
Results saved to ablation_sdcpv2_results.json.
"""

import os, gc, json, time
import numpy as np
import pandas as pd
import torch, faiss
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer as rs_module
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = './RAG_best_practices-main'
DATASETS_DIR = './datasets'
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_ID     = 'mistralai/Mistral-7B-Instruct-v0.2'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED             = 42
N_TRUTHFULQA     = 615
MMLU_PER_SUBJECT = 28
CHOICE_LABELS    = ['A', 'B', 'C', 'D']
K_RETRIEVE       = 15
K_CONTEXT        = 4

INST_S = '[INST]'
INST_E = '[/INST]'
SYS    = ('You are a truthful expert question-answering bot and should '
          'correctly and concisely answer the following question')

# Ablation conditions: (name, alpha, beta, gamma, use_p_pos_prompt, use_p_neg_prompt)
ABLATIONS = [
    ('SDCP-v2 (full)',         0.45, 0.35, 0.20, True,  True),
    ('w/o P_neg (γ=0)',        0.45, 0.35, 0.00, True,  False),
    ('w/o P_pos (β=0)',        0.45, 0.00, 0.20, False, True),
    ('w/o both (Base RAG)',    1.00, 0.00, 0.00, False, False),
]

# ── Load model ────────────────────────────────────────────────────────────────
print('Loading Mistral-7B (4-bit NF4)...')
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map='auto'
)
llm.eval()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
scorer      = rs_module.RougeScorer(['rouge1'], use_stemmer=True)
print('Model loaded ✓')

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate(prompts, max_new_tokens=25, num_beams=2):
    enc = tokenizer(prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048).to(DEVICE)
    in_len = enc['input_ids'].shape[1]
    with torch.no_grad():
        out = llm.generate(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(r[in_len:], skip_special_tokens=True).strip()
            or 'I have no comment' for r in out]


def clean(text):
    for stop in ['\n\n', '\nQuestion:', '\nQ:', '---']:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()


def build_faiss(texts):
    embs = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
    embs = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(embs)
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx


def compute_r1_ecs(generated, references):
    r1s = []
    for gen, refs in zip(generated, references):
        refs = refs if isinstance(refs, list) else [refs]
        r1s.append(max(scorer.score(gen, r)['rouge1'].fmeasure for r in refs))
    gen_embs = embed_model.encode(generated, show_progress_bar=False, batch_size=256)
    ref_embs = embed_model.encode(
        [r[0] if isinstance(r, list) else r for r in references],
        show_progress_bar=False, batch_size=256
    )
    ecs = np.sum(gen_embs * ref_embs, axis=1) / (
        np.linalg.norm(gen_embs, axis=1) * np.linalg.norm(ref_embs, axis=1) + 1e-9
    )
    return round(np.mean(r1s) * 100, 2), round(np.mean(ecs) * 100, 2)


# ── Single ablation run ───────────────────────────────────────────────────────
def run_ablation(test_df, kb_df, dataset_name, name, alpha, beta, gamma,
                 use_p_pos, use_p_neg):
    print(f'\n  [{name}] {dataset_name}  α={alpha} β={beta} γ={gamma}')
    kb_qs  = kb_df['question_plain'].tolist() if 'question_plain' in kb_df.columns \
             else kb_df['question'].tolist()
    kb_idx = build_faiss(kb_qs)

    generated, references = [], []
    t0 = time.time()

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc=name):
        q       = row['question']
        q_plain = row.get('question_plain', q)
        ref     = row['best_answer'] if isinstance(row['best_answer'], list) \
                  else [row['best_answer']]

        # Generate priors only when needed
        p_pos = ''
        if use_p_pos:
            pos_prompt = f'{INST_S}{SYS}\nQuestion: {q_plain}\nAnswer concisely:{INST_E}'
            p_pos      = clean(generate([pos_prompt])[0])

        p_neg = ''
        if use_p_neg:
            if dataset_name == 'MMLU':
                neg_prompt = (f'{INST_S}What is a plausible but INCORRECT answer that students\n'
                              f'commonly give for this type of question?\n'
                              f'Question: {q_plain}\nCommon wrong answer (very short):{INST_E}')
            else:
                neg_prompt = (f'{INST_S}What is a common misconception or false belief\n'
                              f'that people hold about this topic?\n'
                              f'Question: {q_plain}\nCommon wrong belief (very short):{INST_E}')
            p_neg = clean(generate([neg_prompt])[0])

        # Contrastive retrieval
        q_emb = np.array(embed_model.encode([q_plain], show_progress_bar=False), dtype=np.float32)
        faiss.normalize_L2(q_emb)
        _, top_idxs = kb_idx.search(q_emb, K_RETRIEVE)
        top_rows    = [kb_df.iloc[j] for j in top_idxs[0] if j < len(kb_df)]

        sentences = []
        for r2 in top_rows:
            txt = r2['question'] + ' ' + (
                r2['best_answer'][0] if isinstance(r2.get('best_answer'), list)
                else str(r2.get('best_answer', ''))
            )
            sentences.append(txt)

        if sentences:
            s_embs = embed_model.encode(sentences, show_progress_bar=False)
            q_e    = embed_model.encode([q_plain], show_progress_bar=False)
            scores = alpha * cosine_similarity(s_embs, q_e).flatten()
            if use_p_pos and p_pos:
                pp_e   = embed_model.encode([p_pos], show_progress_bar=False)
                scores += beta * cosine_similarity(s_embs, pp_e).flatten()
            if use_p_neg and p_neg:
                pn_e   = embed_model.encode([p_neg], show_progress_bar=False)
                scores -= gamma * cosine_similarity(s_embs, pn_e).flatten()
            context = ' '.join([sentences[j] for j in np.argsort(-scores)[:K_CONTEXT]])
        else:
            context = p_pos or q

        # Build prompt (uncertain path structure, stripping absent components)
        kb_hit = top_rows[0] if top_rows else None
        if kb_hit is not None:
            kb_q   = kb_hit.get('question_plain', kb_hit['question'])
            kb_cor = (kb_hit['best_answer'][0] if isinstance(kb_hit.get('best_answer'), list)
                      else str(kb_hit.get('best_answer', '')))
            kb_inc = (kb_hit['incorrect_answers'][0]
                      if isinstance(kb_hit.get('incorrect_answers'), list) and kb_hit['incorrect_answers']
                      else '')
            ex_block = f'\nExample --- Q: {kb_q}\n  Correct: {kb_cor}  Incorrect: {kb_inc}'
        else:
            ex_block = ''

        prior_lines = []
        if use_p_pos and p_pos:
            prior_lines.append(f'My initial thought: {p_pos}')
        if use_p_neg and p_neg:
            prior_lines.append(f'Common mistake to avoid: {p_neg}')

        final_prompt = (
            f'{INST_S}{SYS}\n'
            f'Retrieved context: {context}'
            f'{ex_block}\n' +
            ('\n'.join(prior_lines) + '\n' if prior_lines else '') +
            f'Question: {q}\nVerified answer:{INST_E}'
        )
        generated.append(clean(generate([final_prompt])[0]))
        references.append(ref)

        if i % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    r1, ecs = compute_r1_ecs(generated, references)
    print(f'  → R1={r1}  ECS={ecs}  ({elapsed:.1f} min)')
    return r1, ecs


# ── Load datasets ─────────────────────────────────────────────────────────────
print('\nLoading datasets (seed=42)...')
tqa_raw  = load_from_disk(f'{DATASETS_DIR}/truthfulqa').to_pandas()
mmlu_raw = load_from_disk(f'{DATASETS_DIR}/mmlu').to_pandas()

tqa_all = tqa_raw[['question','best_answer','correct_answers','incorrect_answers']].copy()
tqa_all['correct_answers']   = tqa_all['correct_answers'].apply(
    lambda x: x.tolist() if hasattr(x, 'tolist') else [x])
tqa_all['incorrect_answers'] = tqa_all['incorrect_answers'].apply(
    lambda x: x.tolist() if hasattr(x, 'tolist') else [x])
tqa_all['best_answer']       = tqa_all['best_answer'].apply(
    lambda x: [x] if not isinstance(x, list) else x)
tqa_all = tqa_all[
    (tqa_all['correct_answers'].apply(len) > 1) &
    (tqa_all['incorrect_answers'].apply(len) > 1)
].reset_index(drop=True)

tqa_test_idx = tqa_all.sample(n=N_TRUTHFULQA, random_state=SEED).index
tqa_test     = tqa_all.loc[tqa_test_idx].reset_index(drop=True)
tqa_kb       = tqa_all.drop(tqa_test_idx).reset_index(drop=True)


def mmlu_to_unified(row):
    choices = list(row['choices'])
    ans_idx = int(row['answer'])
    correct = choices[ans_idx]
    incorrect = [choices[i] for i in range(len(choices)) if i != ans_idx]
    formatted_q = (row['question'] + '\n' +
                   '\n'.join(f'{CHOICE_LABELS[i]}) {choices[i]}' for i in range(len(choices))))
    return pd.Series({
        'question':         formatted_q,
        'question_plain':   row['question'],
        'subject':          row['subject'],
        'best_answer':      [correct],
        'correct_answers':  [correct],
        'incorrect_answers': incorrect,
    })


mmlu_test_parts, mmlu_kb_parts = [], []
for subject, group in mmlu_raw.groupby('subject'):
    group  = group.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_test = min(MMLU_PER_SUBJECT, len(group))
    mmlu_test_parts.append(group.iloc[:n_test])
    if len(group) > n_test:
        mmlu_kb_parts.append(group.iloc[n_test:])
mmlu_test = pd.concat(mmlu_test_parts).reset_index(drop=True).apply(mmlu_to_unified, axis=1)
mmlu_kb   = pd.concat(mmlu_kb_parts).reset_index(drop=True).apply(mmlu_to_unified, axis=1)

print(f'TQA: test={len(tqa_test)} kb={len(tqa_kb)}')
print(f'MMLU: test={len(mmlu_test)} kb={len(mmlu_kb)}')

# ── Run ablations ─────────────────────────────────────────────────────────────
results = {}
for (name, alpha, beta, gamma, use_p_pos, use_p_neg) in ABLATIONS:
    results[name] = {}
    for ds_name, test_df, kb_df in [('TQA', tqa_test, tqa_kb),
                                     ('MMLU', mmlu_test, mmlu_kb)]:
        r1, ecs = run_ablation(
            test_df, kb_df, ds_name,
            name, alpha, beta, gamma, use_p_pos, use_p_neg
        )
        results[name][ds_name] = {'R1': r1, 'ECS': ecs}
    gc.collect(); torch.cuda.empty_cache()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('ABLATION TABLE — SDCP-v2 as reference (seed=42)')
print(f'{"="*65}')
header = f'{"Condition":<30} {"TQA R1":>8} {"TQA ECS":>8} {"MMLU R1":>8} {"MMLU ECS":>9}'
print(header)
print('-' * 65)

full_r = results['SDCP-v2 (full)']
for name, vals in results.items():
    tqa_r1  = vals['TQA']['R1'];  tqa_ecs  = vals['TQA']['ECS']
    mmlu_r1 = vals['MMLU']['R1']; mmlu_ecs = vals['MMLU']['ECS']
    delta_tqa  = f'({tqa_r1 - full_r["TQA"]["R1"]:+.2f})' if name != 'SDCP-v2 (full)' else ''
    delta_mmlu = f'({mmlu_r1 - full_r["MMLU"]["R1"]:+.2f})' if name != 'SDCP-v2 (full)' else ''
    print(f'{name:<30} {tqa_r1:>6} {delta_tqa:>4}  {tqa_ecs:>6}  {mmlu_r1:>6} {delta_mmlu:>4}  {mmlu_ecs:>6}')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'{OUT_DIR}/ablation_sdcpv2_results_{ts}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: ablation_sdcpv2_results_{ts}.json')

print('\n--- LaTeX rows for ablation table ---')
for name, vals in results.items():
    full = name == 'SDCP-v2 (full)'
    tqa_str  = f'{vals["TQA"]["R1"]:.2f}'
    mmlu_str = f'{vals["MMLU"]["R1"]:.2f}'
    ecs_str  = f'{vals["TQA"]["ECS"]:.2f}'
    label    = name if full else f'\\quad {name}'
    print(f'  {label} & {tqa_str} & {ecs_str} & {mmlu_str} & -- \\\\')
