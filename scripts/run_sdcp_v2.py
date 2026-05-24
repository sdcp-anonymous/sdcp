"""
SDCP-v2: Always-Uncertain Path (no cert routing)
================================================
Key changes from SDCP-v1:
  1. cert routing removed — always use the uncertain path (rich context + KB anchor + both priors)
  2. ARC prompt shows answer choices explicitly for the test question
  3. Runs all three datasets: TruthfulQA, MMLU, ARC-Challenge

Based on ablation finding: w/o cert consistently outperforms full SDCP,
so we drop the routing and commit to the uncertain path throughout.
"""

import os, sys, gc, json, time
import numpy as np
import pandas as pd
import torch
import faiss
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer as rs_module
from tqdm import tqdm

BASE_DIR     = './RAG_best_practices-main'
MODELS_DIR   = './models'
DATASETS_DIR = './datasets'
OUTPUT_DIR   = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

# ── Paper-mode sizes ─────────────────────────────────────────────────────────
N_TRUTHFULQA     = 615
MMLU_PER_SUBJECT = 28
RANDOM_SEED      = 42
QUANT            = '4bit'

SDCP_PARAMS = {
    'alpha': 0.45,
    'beta':  0.35,
    'gamma': 0.20,
    'top_k_retrieve': 15,
    'top_k_context':  4,
    'max_gen_tokens': 25,
    'max_pos_tokens': 20,
    'max_neg_tokens': 20,
}

CHOICE_LABELS = ['A', 'B', 'C', 'D']

INST_S = '[INST]'
INST_E = '[/INST]'
SYS    = ('You are a truthful expert question-answering bot and should '
          'correctly and concisely answer the following question')

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Load model ───────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

print('Loading Mistral-7B-Instruct-v0.2 (4bit)...')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llm = AutoModelForCausalLM.from_pretrained(
    f'{MODELS_DIR}/mistral-7b-instruct',
    quantization_config=bnb, device_map='auto', trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    f'{MODELS_DIR}/mistral-7b-instruct', padding_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

print('Loading MiniLM embeddings...')
embed_model = SentenceTransformer(f'{MODELS_DIR}/minilm')
print('Models ready.')

# ── Generation helpers ────────────────────────────────────────────────────────
def generate(prompts, max_new_tokens=25, num_beams=1):
    enc = tokenizer(prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048).to(DEVICE)
    input_len = enc['input_ids'].shape[1]
    with torch.no_grad():
        out = llm.generate(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(r[input_len:], skip_special_tokens=True).strip() or 'I have no comment'
            for r in out]

def get_token_probs(prompt, max_new_tokens=20):
    enc = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = llm.generate(
            **enc, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out.sequences[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    cert = 0.0
    if out.scores:
        probs = torch.softmax(out.scores[0][0], dim=-1)
        top2  = torch.topk(probs, 2).values
        cert  = (top2[0] - top2[1]).item()
    return text, cert

def clean_response(resp):
    for stop in ['\nQuestion:', '\nQ:', '\n---', '\nIncorrect', '\nCorrect',
                 '\nVERIFIED', '\nExample', '\n\n', '\nMy initial']:
        if stop in resp:
            resp = resp[:resp.index(stop)]
    return resp.strip().strip('"').strip("'") or 'I have no comment'

# ── Retrieval ─────────────────────────────────────────────────────────────────
def build_index(dataset):
    embs = embed_model.encode(dataset['question'].tolist(), show_progress_bar=True, batch_size=64)
    embs = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def retrieve_from_kb(query, faiss_idx, kb_dataset, k=15):
    q_emb = np.array(embed_model.encode([query], show_progress_bar=False), dtype=np.float32)
    faiss.normalize_L2(q_emb)
    _, idxs = faiss_idx.search(q_emb, k)
    return [kb_dataset.iloc[i] for i in idxs[0] if i < len(kb_dataset)]

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(generated, references):
    scorer = rs_module.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1s, r2s, rls, ecss = [], [], [], []
    for gen, refs in zip(generated, references):
        best_r1 = best_r2 = best_rl = 0
        for ref in refs:
            if not ref: continue
            s = scorer.score(ref, gen)
            best_r1 = max(best_r1, s['rouge1'].fmeasure)
            best_r2 = max(best_r2, s['rouge2'].fmeasure)
            best_rl = max(best_rl, s['rougeL'].fmeasure)
        r1s.append(best_r1 * 100); r2s.append(best_r2 * 100); rls.append(best_rl * 100)
        try:
            embs = embed_model.encode([refs[0], gen])
            ecss.append(float(cosine_similarity([embs[0]], [embs[1]])[0][0]) * 100)
        except:
            ecss.append(0.0)
    return np.array(r1s), np.array(r2s), np.array(rls), np.array(ecss)

def compute_accuracy(generated, dataset):
    correct = 0
    for gen, (_, row) in zip(generated, dataset.iterrows()):
        ans_idx = int(row.get('answer_idx', -1))
        correct_text = row['best_answer'][0] if isinstance(row['best_answer'], list) else row['best_answer']
        if ans_idx >= 0:
            label = CHOICE_LABELS[ans_idx]
            if label in gen[:5] or correct_text.lower() in gen.lower():
                correct += 1
        else:
            if correct_text.lower() in gen.lower():
                correct += 1
    return correct / len(generated) * 100

# ── SDCP-v2 core ──────────────────────────────────────────────────────────────
def generate_sdcp_priors(query, dataset_name, params):
    pos_prompt = f'{INST_S}{SYS}\nQuestion: {query}\nAnswer concisely:{INST_E}'
    p_pos, cert = get_token_probs(pos_prompt, max_new_tokens=params['max_pos_tokens'])
    p_pos = clean_response(p_pos)

    q_plain = query.split('\n')[0] if '\n' in query else query
    if dataset_name in ['MMLU', 'ARC']:
        neg_prompt = (f'{INST_S}What is a plausible but INCORRECT answer that students '
                      f'commonly give for this type of question?\n'
                      f'Question: {q_plain}\nCommon wrong answer (very short):{INST_E}')
    else:
        neg_prompt = (f'{INST_S}What is a common misconception or false belief '
                      f'that people hold about this topic?\n'
                      f'Question: {q_plain}\nCommon wrong belief (very short):{INST_E}')

    p_neg = clean_response(generate([neg_prompt], max_new_tokens=params['max_neg_tokens'])[0])
    return p_pos, p_neg, cert


def run_sdcp_v2(test_data, kb_data, dataset_name, params=None):
    """
    SDCP-v2: Always-Uncertain Path.
    No cert-based routing — always use the rich uncertain path prompt.
    For ARC, the test question includes choices explicitly in the prompt.
    """
    if params is None:
        params = SDCP_PARAMS
    print(f'\n=== SDCP-v2 | {dataset_name} | test={len(test_data)}Q  KB={len(kb_data)}Q ===')
    print(f'    α={params["alpha"]} β={params["beta"]} γ={params["gamma"]} | always-uncertain (no cert routing)')

    faiss_idx = build_index(kb_data)
    generated, references, prior_log = [], [], []

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f'SDCP-v2/{dataset_name}'):
        query       = row['question']
        best_answer = row['best_answer'] if isinstance(row['best_answer'], list) else [row['best_answer']]

        # Step 1: self-distill priors (cert computed but NOT used for routing)
        p_pos, p_neg, cert = generate_sdcp_priors(query, dataset_name, params)

        # Step 2: prior-guided contrastive retrieval
        retrieved = retrieve_from_kb(query, faiss_idx, kb_data, k=params['top_k_retrieve'])

        prompt = None
        if retrieved and p_pos:
            sentences = []
            for doc in retrieved:
                sentences.append(doc['question'])
                ba = doc['best_answer']
                ba_t = ba[0] if isinstance(ba, list) and ba else str(ba)
                if ba_t and len(ba_t) > 3:
                    sentences.append(ba_t)
            sentences = [s for s in sentences if s and len(s.strip()) > 5]

            if sentences:
                s_embs  = embed_model.encode(sentences, show_progress_bar=False)
                q_emb   = embed_model.encode([query],   show_progress_bar=False)
                pos_emb = embed_model.encode([p_pos],   show_progress_bar=False)
                neg_emb = embed_model.encode([p_neg],   show_progress_bar=False) if p_neg else None
                q_sims  = cosine_similarity(s_embs, q_emb).flatten()
                p_sims  = cosine_similarity(s_embs, pos_emb).flatten()
                n_sims  = (cosine_similarity(s_embs, neg_emb).flatten()
                           if neg_emb is not None else np.zeros(len(sentences)))
                sdcp_sc = params['alpha'] * q_sims + params['beta'] * p_sims - params['gamma'] * n_sims
                context = ' '.join([sentences[i] for i in np.argsort(-sdcp_sc)[:params['top_k_context']]])

                kb_ex        = retrieved[0]
                kb_correct   = (kb_ex['best_answer'][0] if isinstance(kb_ex['best_answer'], list)
                                and kb_ex['best_answer'] else str(kb_ex['best_answer']))
                kb_incorrect = (kb_ex['incorrect_answers'][0] if isinstance(kb_ex['incorrect_answers'], list)
                                and kb_ex['incorrect_answers'] else '')
                kb_q         = kb_ex.get('question_plain', kb_ex['question'])

                # Step 3: always use uncertain path (rich context)
                # For ARC: the question already contains choices (A/B/C/D) in row['question']
                prompt = (f'{INST_S}{SYS}\n'
                          f'Retrieved context: {context}\n'
                          f'Example — Q: {kb_q}\n'
                          f'  Correct: {kb_correct}\n'
                          f'  Incorrect: {kb_incorrect}\n'
                          f'My initial thought: {p_pos}\n'
                          f'Common mistake to avoid: {p_neg}\n'
                          f'Question: {query}\n'
                          f'Verified answer:{INST_E}')

        if prompt is None:
            prompt = f'{INST_S}{SYS}\nQuestion: {query}\nAnswer:{INST_E}'

        # Step 4: final generation
        resp  = generate([prompt], max_new_tokens=params['max_gen_tokens'], num_beams=2)
        final = clean_response(resp[0])

        generated.append(final)
        references.append(best_answer)
        prior_log.append({'p_pos': p_pos, 'p_neg': p_neg, 'cert': cert})

        if idx % 30 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Metrics
    r1, r2, rl, ecs = compute_metrics(generated, references)

    try:
        import mauve as mauve_lib
        refs_flat = [r[0] if isinstance(r, list) else r for r in references]
        valid = [(g, r) for g, r in zip(generated, refs_flat) if g and r]
        if len(valid) >= 10:
            gens_v, refs_v = zip(*valid)
            mauve_score = mauve_lib.compute_mauve(
                p_text=list(refs_v), q_text=list(gens_v),
                device_id=0, max_text_length=256, verbose=False,
                featurize_model_name='gpt2'
            ).mauve * 100
        else:
            mauve_score = 0.0
    except Exception as e:
        print(f'  MAUVE error: {e}')
        mauve_score = 0.0

    # Prior quality
    scorer_p = rs_module.RougeScorer(['rouge1'], use_stemmer=True)
    pos_r1_list = []
    for log, refs in zip(prior_log, references):
        ref = refs[0] if isinstance(refs, list) else refs
        if log['p_pos'] and ref:
            pos_r1_list.append(scorer_p.score(ref, log['p_pos'])['rouge1'].fmeasure * 100)
    pos_quality = float(np.mean(pos_r1_list)) if pos_r1_list else 0.0
    avg_cert    = float(np.mean([l['cert'] for l in prior_log]))

    result = {
        'method': f'SDCP-v2-{dataset_name}', 'dataset': dataset_name,
        'R1': float(r1.mean()), 'R2': float(r2.mean()),
        'RL': float(rl.mean()), 'ECS': float(ecs.mean()),
        'MAUVE': mauve_score,
        'pos_quality': pos_quality,
        'avg_cert': avg_cert,
        'generated': generated, 'references': references,
        'prior_log': prior_log,
    }

    if 'answer_idx' in test_data.columns:
        acc = compute_accuracy(generated, test_data)
        result['Accuracy'] = acc
        print(f'  R1={r1.mean():.2f} R2={r2.mean():.2f} RL={rl.mean():.2f} '
              f'ECS={ecs.mean():.2f} MAUVE={mauve_score:.2f} Acc={acc:.1f}%')
    else:
        print(f'  R1={r1.mean():.2f} R2={r2.mean():.2f} RL={rl.mean():.2f} '
              f'ECS={ecs.mean():.2f} MAUVE={mauve_score:.2f}')
    print(f'  pos_quality={pos_quality:.1f} avg_cert={avg_cert:.3f}')

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Load datasets
# ══════════════════════════════════════════════════════════════════════════════
from datasets import load_from_disk

print('\n── Loading datasets ──')

# TruthfulQA
tqa_raw = load_from_disk(f'{DATASETS_DIR}/truthfulqa').to_pandas()
tqa_all = tqa_raw[['question', 'best_answer', 'correct_answers', 'incorrect_answers']].copy()
tqa_all['correct_answers']   = tqa_all['correct_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
tqa_all['incorrect_answers'] = tqa_all['incorrect_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
tqa_all['best_answer']       = tqa_all['best_answer'].apply(lambda x: [x] if x else [])
tqa_all = tqa_all[(tqa_all['correct_answers'].apply(len) > 1) &
                  (tqa_all['incorrect_answers'].apply(len) > 1)].reset_index(drop=True)
rng = np.random.RandomState(RANDOM_SEED)
tqa_test_idx = tqa_all.sample(n=N_TRUTHFULQA, random_state=RANDOM_SEED).index
tqa_kb_idx   = tqa_all.index.difference(tqa_test_idx)
tqa      = tqa_all.loc[tqa_test_idx].reset_index(drop=True)
tqa_kb   = tqa_all.loc[tqa_kb_idx].reset_index(drop=True)
tqa_kb['question_plain'] = tqa_kb['question']
print(f'TruthfulQA: test={len(tqa)} KB={len(tqa_kb)}')

# MMLU
mmlu_raw = load_from_disk(f'{DATASETS_DIR}/mmlu_selected').to_pandas()
subjects = mmlu_raw['subject'].unique()
mmlu_test_list, mmlu_kb_list = [], []
for subj in subjects:
    sub = mmlu_raw[mmlu_raw['subject'] == subj].reset_index(drop=True)
    n_test = min(MMLU_PER_SUBJECT, len(sub))
    test_idx = sub.sample(n=n_test, random_state=RANDOM_SEED).index
    mmlu_test_list.append(sub.loc[test_idx])
    mmlu_kb_list.append(sub.loc[sub.index.difference(test_idx)])
mmlu    = pd.concat(mmlu_test_list).reset_index(drop=True)
mmlu_kb = pd.concat(mmlu_kb_list).reset_index(drop=True)
if 'question_plain' not in mmlu_kb.columns:
    mmlu_kb['question_plain'] = mmlu_kb['question'].apply(lambda q: q.split('\n')[0])
print(f'MMLU: test={len(mmlu)} KB={len(mmlu_kb)}')

# ARC-Challenge
arc_raw = load_from_disk(f'{DATASETS_DIR}/arc_challenge')
arc_test_df = arc_raw['test'].to_pandas()
arc_kb_df   = pd.concat([
    arc_raw['train'].to_pandas(),
    arc_raw['validation'].to_pandas()
]).reset_index(drop=True)

def format_arc_question(row):
    choices = row['choices']
    labels  = choices['label'] if isinstance(choices, dict) else [c for c in choices]
    texts   = choices['text']  if isinstance(choices, dict) else choices
    opts = '\n'.join([f'{l}. {t}' for l, t in zip(labels, texts)])
    return f"{row['question']}\n{opts}"

def arc_to_standard(df):
    out = []
    for _, row in df.iterrows():
        choices = row['choices']
        labels  = choices['label'] if isinstance(choices, dict) else list(choices)
        texts   = choices['text']  if isinstance(choices, dict) else list(choices)
        ans_key = row['answerKey']
        try:
            ans_idx = labels.index(ans_key)
        except ValueError:
            ans_idx = -1
        correct_text = texts[ans_idx] if ans_idx >= 0 else ''
        out.append({
            'question':          format_arc_question(row),
            'question_plain':    row['question'],
            'best_answer':       [correct_text],
            'correct_answers':   [correct_text],
            'incorrect_answers': [t for i, t in enumerate(texts) if i != ans_idx],
            'answer_idx':        ans_idx,
        })
    return pd.DataFrame(out)

arc    = arc_to_standard(arc_test_df)
arc_kb = arc_to_standard(arc_kb_df)
print(f'ARC-Challenge: test={len(arc)} KB={len(arc_kb)}')


# ══════════════════════════════════════════════════════════════════════════════
# Run experiments
# ══════════════════════════════════════════════════════════════════════════════
all_results = {}
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

print('\n' + '='*65)
print('PHASE 1: TruthfulQA')
print('='*65)
t0 = time.time()
all_results['SDCP_v2_TQA'] = run_sdcp_v2(tqa, tqa_kb, 'TruthfulQA')
print(f'  Done in {(time.time()-t0)/60:.1f} min')

print('\n' + '='*65)
print('PHASE 2: MMLU')
print('='*65)
t0 = time.time()
all_results['SDCP_v2_MMLU'] = run_sdcp_v2(mmlu, mmlu_kb, 'MMLU')
print(f'  Done in {(time.time()-t0)/60:.1f} min')

print('\n' + '='*65)
print('PHASE 3: ARC-Challenge')
print('='*65)
t0 = time.time()
all_results['SDCP_v2_ARC'] = run_sdcp_v2(arc, arc_kb, 'ARC')
print(f'  Done in {(time.time()-t0)/60:.1f} min')

# ══════════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════════
summary = {}
for key, r in all_results.items():
    summary[key] = {
        'method':       r['method'],
        'dataset':      r['dataset'],
        'R1':           round(r['R1'], 3),
        'R2':           round(r['R2'], 3),
        'RL':           round(r['RL'], 3),
        'ECS':          round(r['ECS'], 3),
        'MAUVE':        round(r['MAUVE'], 3),
        'Accuracy':     round(r.get('Accuracy', 0), 2),
        'pos_quality':  round(r['pos_quality'], 2),
        'avg_cert':     round(r['avg_cert'], 4),
    }
    df = pd.DataFrame({
        'generated':  r['generated'],
        'reference':  [x[0] if isinstance(x, list) else x for x in r['references']],
        'p_pos':      [l['p_pos'] for l in r['prior_log']],
        'p_neg':      [l['p_neg'] for l in r['prior_log']],
        'cert':       [l['cert']  for l in r['prior_log']],
    })
    df.to_csv(f'{OUTPUT_DIR}/{key}_{ts}.csv', index=False, quoting=1)

out_path = f'{OUTPUT_DIR}/sdcp_v2_summary_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n✓ Results saved to {out_path}')
print('\n' + '='*65)
print('SDCP-v2 FINAL RESULTS')
print('='*65)
print(json.dumps(summary, indent=2))

# Comparison with v1
print('\n── Comparison: SDCP-v1 vs SDCP-v2 ──────────────────────')
v1 = {
    'TruthfulQA': {'R1': 34.26, 'ECS': 64.92, 'Acc': None},
    'MMLU':       {'R1': 32.34, 'ECS': 48.14, 'Acc': 45.2},
    'ARC':        {'R1': 35.02, 'ECS': 53.15, 'Acc': 56.1},
}
for key, r in all_results.items():
    ds = r['dataset']
    ds_key = 'ARC' if ds == 'ARC' else ds
    old = v1.get(ds_key, {})
    r1_delta = r['R1'] - old.get('R1', 0)
    print(f'{ds}: R1 {old.get("R1","?"):.2f} → {r["R1"]:.2f} ({r1_delta:+.2f}) | '
          f'ECS {old.get("ECS","?"):.2f} → {r["ECS"]:.2f} | '
          f'Acc {old.get("Acc","?")} → {r.get("Accuracy","N/A")}')
