"""
Ablation study of SDCP-v2 on ARC-Challenge.

Mirrors ablation_sdcpv2.py (TQA + MMLU) but runs the four ablation
conditions on ARC-Challenge only, so the result can be added as a
third column to Table 6 (tab:ablation) in the paper.

KB:   train + validation splits  (n = 1,119 + 299 = 1,418)
Test: official test split         (n = 1,172)

Ablation conditions (same as ablation_sdcpv2.py):
    1. SDCP-v2 (full)                   α=0.45  β=0.35  γ=0.20
    2. SDCP-v2 w/o P_neg (γ=0)          α=0.45  β=0.35  γ=0.00
    3. SDCP-v2 w/o P_pos (β=0)          α=0.45  β=0.00  γ=0.20
    4. SDCP-v2 w/o both  (α=1 β=0 γ=0) pure query-similarity = Base RAG

Runtime: ~4 × 32 min ≈ ~2 hours on A100.
Results saved to ablation_arc_results_<timestamp>.json.
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
DATASETS_DIR = './datasets'
OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_ID     = 'mistralai/Mistral-7B-Instruct-v0.2'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

CHOICE_LABELS = ['A', 'B', 'C', 'D']
K_RETRIEVE    = 15
K_CONTEXT     = 4

INST_S = '[INST]'
INST_E = '[/INST]'
SYS    = ('You are a truthful expert question-answering bot and should '
          'correctly and concisely answer the following question')

ABLATIONS = [
    ('SDCP-v2 (full)',      0.45, 0.35, 0.20, True,  True),
    ('w/o P_neg (γ=0)',     0.45, 0.35, 0.00, True,  False),
    ('w/o P_pos (β=0)',     0.45, 0.00, 0.20, False, True),
    ('w/o both (Base RAG)', 1.00, 0.00, 0.00, False, False),
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


def compute_metrics(generated, references, correct_texts):
    """R1 (vs best_answer) and accuracy (exact match of correct answer text)."""
    r1s, correct = [], []
    for gen, refs, corr in zip(generated, references, correct_texts):
        refs = refs if isinstance(refs, list) else [refs]
        r1s.append(max(scorer.score(gen, r)['rouge1'].fmeasure for r in refs))
        # accuracy: check if generated answer contains the correct choice text
        correct.append(int(corr.lower() in gen.lower()))
    return round(np.mean(r1s) * 100, 2), round(np.mean(correct) * 100, 2)


# ── ARC dataset loading ───────────────────────────────────────────────────────
def arc_to_unified(row):
    """Convert ARC row to unified format matching ablation_sdcpv2.py."""
    choices_text  = row['choices']['text']   # list of strings
    choices_label = row['choices']['label']  # list of 'A','B','C','D'
    answer_key    = row['answerKey']         # 'A','B','C','D' or '1','2','3','4'

    # Normalise numeric answer keys (some ARC rows use '1'-'4')
    if answer_key.isdigit():
        answer_key = CHOICE_LABELS[int(answer_key) - 1]

    ans_idx   = choices_label.index(answer_key)
    correct   = choices_text[ans_idx]
    incorrect = [choices_text[i] for i in range(len(choices_text)) if i != ans_idx]

    formatted_q = (row['question'] + '\n' +
                   '\n'.join(f'{choices_label[i]}) {choices_text[i]}'
                             for i in range(len(choices_text))))
    return pd.Series({
        'question':          formatted_q,
        'question_plain':    row['question'],
        'best_answer':       [correct],
        'correct_answers':   [correct],
        'incorrect_answers': incorrect,
    })


print('\nLoading ARC-Challenge splits...')
arc_train = load_from_disk(f'{DATASETS_DIR}/arc_challenge_train').to_pandas()
arc_val   = load_from_disk(f'{DATASETS_DIR}/arc_challenge_validation').to_pandas()
arc_test  = load_from_disk(f'{DATASETS_DIR}/arc_challenge_test').to_pandas()

arc_kb_raw = pd.concat([arc_train, arc_val], ignore_index=True)
arc_kb   = arc_kb_raw.apply(arc_to_unified, axis=1)
arc_test = arc_test.apply(arc_to_unified, axis=1)

print(f'ARC: test={len(arc_test)}  kb={len(arc_kb)}')

# ── Single ablation run ───────────────────────────────────────────────────────
def run_ablation(test_df, kb_df, name, alpha, beta, gamma, use_p_pos, use_p_neg):
    print(f'\n  [{name}]  α={alpha} β={beta} γ={gamma}')
    kb_qs  = kb_df['question_plain'].tolist()
    kb_idx = build_faiss(kb_qs)

    generated, references, correct_texts = [], [], []
    t0 = time.time()

    for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc=name):
        q       = row['question']
        q_plain = row['question_plain']
        ref     = row['best_answer']
        correct = row['correct_answers'][0]

        # Generate priors only when needed
        p_pos = ''
        if use_p_pos:
            pos_prompt = f'{INST_S}{SYS}\nQuestion: {q_plain}\nAnswer concisely:{INST_E}'
            p_pos      = clean(generate([pos_prompt])[0])

        p_neg = ''
        if use_p_neg:
            neg_prompt = (f'{INST_S}What is a common misconception or false belief\n'
                          f'that people hold about this topic?\n'
                          f'Question: {q_plain}\nCommon wrong belief (very short):{INST_E}')
            p_neg = clean(generate([neg_prompt])[0])

        # Contrastive retrieval
        q_emb = np.array(embed_model.encode([q_plain], show_progress_bar=False),
                         dtype=np.float32)
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

        # Prompt (uncertain path — v2 style)
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
        correct_texts.append(correct)

        if i % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    r1, acc = compute_metrics(generated, references, correct_texts)
    print(f'  → R1={r1}  Acc={acc}%  ({elapsed:.1f} min)')
    return r1, acc


# ── Run ablations ─────────────────────────────────────────────────────────────
results = {}
for (name, alpha, beta, gamma, use_p_pos, use_p_neg) in ABLATIONS:
    r1, acc = run_ablation(arc_test, arc_kb, name, alpha, beta, gamma,
                           use_p_pos, use_p_neg)
    results[name] = {'ARC': {'R1': r1, 'Acc': acc}}
    gc.collect(); torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print('ABLATION TABLE — ARC-Challenge (SDCP-v2 as reference)')
print(f'{"="*55}')
header = f'{"Condition":<30} {"ARC R1":>8} {"ARC Acc":>9}'
print(header)
print('-' * 55)

full_r = results['SDCP-v2 (full)']
for name, vals in results.items():
    arc_r1  = vals['ARC']['R1']
    arc_acc = vals['ARC']['Acc']
    delta_r1  = f'({arc_r1  - full_r["ARC"]["R1"]:+.2f})'  if name != 'SDCP-v2 (full)' else ''
    delta_acc = f'({arc_acc - full_r["ARC"]["Acc"]:+.2f})' if name != 'SDCP-v2 (full)' else ''
    print(f'{name:<30} {arc_r1:>6} {delta_r1:>7}  {arc_acc:>6} {delta_acc:>7}')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'{OUT_DIR}/ablation_arc_results_{ts}.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved: ablation_arc_results_{ts}.json')

print('\n--- LaTeX rows to add to tab:ablation ---')
print('Add these columns after the MMLU ECS column:\n')
print(r'% New header columns:  & \multicolumn{2}{c}{\textbf{ARC-Challenge}} \\')
print(r'%                        & R1 & Acc \\')
for name, vals in results.items():
    full = name == 'SDCP-v2 (full)'
    arc_r1  = f'{vals["ARC"]["R1"]:.2f}'
    arc_acc = f'{vals["ARC"]["Acc"]:.1f}'
    label   = name if full else f'\\quad {name}'
    print(f'  {label} & ... & {arc_r1} & {arc_acc}\\% \\\\')
