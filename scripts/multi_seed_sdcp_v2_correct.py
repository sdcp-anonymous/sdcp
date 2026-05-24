#!/usr/bin/env python3
"""
Multi-seed evaluation of SDCP-v2 using the EXACT paper protocol
(matches SDCP_v2_AlwaysUncertain.ipynb cell-by-cell).

The previous multi_seed_sdcp.py used a SUBTLY DIFFERENT MMLU protocol
(P_pos generated from q_plain instead of full query, sentences built by
concatenation instead of split rows).  That produced MMLU R1 = 41.82,
which conflicts with paper Table 1 (35.44).  This script reproduces the
paper protocol exactly, then re-runs with seeds {42, 123, 456} so the
variance estimate is comparable to Table 1 single-seed numbers.

Datasets: TruthfulQA (615Q), MMLU (1596Q), ARC-Challenge (1172Q)
Runtime:  ~3 x (50 + 130 + 95) min  ≈ ~14 h on Quadro RTX 8000

Outputs:
  outputs/sdcp_v2_correct_seed{S}_TQA.csv
  outputs/sdcp_v2_correct_seed{S}_MMLU.csv
  outputs/sdcp_v2_correct_seed{S}_ARC.csv
  outputs/multi_seed_v2_correct_<ts>.json     # summary, mean ± std
"""

import os, gc, time, json
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

# =============================================================================
#  CONFIG -- MATCHES PAPER (SDCP_v2_AlwaysUncertain.ipynb)
# =============================================================================
BASE_DIR     = './RAG_best_practices-main'
MODELS_DIR   = './models'
DATASETS_DIR = './datasets'
OUTPUT_DIR   = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS            = [42, 123, 456]
N_TRUTHFULQA     = 615
MMLU_PER_SUBJECT = 28
CHOICE_LABELS    = ['A', 'B', 'C', 'D']

INST_S = '[INST]'
INST_E = '[/INST]'
SYS    = ('You are a truthful expert question-answering bot and should '
          'correctly and concisely answer the following question')

# Paper-exact SDCP-v2 hyperparameters
SDCP_V2_PARAMS = {
    'cert_threshold'  : 0.65,   # recorded only -- not used for routing in v2
    'alpha'           : 0.45,
    'beta'            : 0.35,
    'gamma'           : 0.20,
    'top_k_retrieve'  : 15,
    'top_k_context'   : 4,
    'max_gen_tokens'  : 25,
    'max_pos_tokens'  : 20,
    'max_neg_tokens'  : 20,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
#  LOAD MODELS (once for all seeds)
# =============================================================================
gc.collect(); torch.cuda.empty_cache()

MODEL_PATH = f'{MODELS_DIR}/mistral-7b'
EMBED_PATH = f'{MODELS_DIR}/minilm'

print('Loading Mistral-7B (4-bit, paper-exact bnb config)...')
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                            device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

print('Loading MiniLM...')
embed_model = SentenceTransformer(EMBED_PATH).to(DEVICE)
print('Models ready.\n')

# =============================================================================
#  UTILITIES -- COPIED VERBATIM FROM SDCP_v2_AlwaysUncertain.ipynb
# =============================================================================
def generate(prompts, max_new_tokens=25, do_sample=False,
             temperature=1.0, top_p=0.9, num_beams=1):
    enc = tokenizer(prompts, return_tensors='pt', padding=True,
                    truncation=True, max_length=2048).to(DEVICE)
    input_len = enc['input_ids'].shape[1]
    kwargs = dict(input_ids=enc['input_ids'],
                  attention_mask=enc['attention_mask'],
                  max_new_tokens=max_new_tokens,
                  pad_token_id=tokenizer.eos_token_id)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs['num_beams'] = num_beams
    with torch.no_grad():
        out = llm.generate(**kwargs)
    return [tokenizer.decode(r[input_len:], skip_special_tokens=True).strip()
            or 'I have no comment' for r in out]

def get_token_probs(prompt, max_new_tokens=20):
    enc = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=max_new_tokens,
                           return_dict_in_generate=True, output_scores=True,
                           pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out.sequences[0][enc['input_ids'].shape[1]:],
                             skip_special_tokens=True).strip()
    cert = 0.0
    if out.scores:
        probs = torch.softmax(out.scores[0][0], dim=-1)
        top2  = torch.topk(probs, 2).values
        cert  = (top2[0] - top2[1]).item()
    return text, cert

def build_index(dataset):
    embs = embed_model.encode(dataset['question'].tolist(),
                               show_progress_bar=False, batch_size=64)
    embs = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx

def retrieve_from_kb(query, faiss_idx, kb_dataset, k=1):
    q_emb = np.array(embed_model.encode([query], show_progress_bar=False),
                     dtype=np.float32)
    faiss.normalize_L2(q_emb)
    _, idxs = faiss_idx.search(q_emb, k)
    return [kb_dataset.iloc[i] for i in idxs[0] if i < len(kb_dataset)]

def clean_response(resp):
    for stop in ['\nQuestion:','\nQ:','\n---','\nIncorrect','\nCorrect',
                 '\nVERIFIED','\nExample','\n\n','\nMy initial','\nContext:',
                 '\nFor the question','\nCommon']:
        if stop in resp: resp = resp[:resp.index(stop)]
    return resp.strip().strip('"').strip("'") or 'I have no comment'

def compute_metrics(generated, references):
    scorer = rs_module.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    r1s, ecss = [], []
    for gen, refs in zip(generated, references):
        best_r1 = 0
        for ref in refs:
            if not ref: continue
            best_r1 = max(best_r1, scorer.score(ref, gen)['rouge1'].fmeasure)
        r1s.append(best_r1 * 100)
        try:
            embs = embed_model.encode([refs[0], gen])
            ecss.append(float(cosine_similarity([embs[0]], [embs[1]])[0][0]) * 100)
        except Exception:
            ecss.append(0.0)
    return np.array(r1s), np.array(ecss)

def compute_accuracy(generated, dataset):
    correct = 0
    for gen, (_, row) in zip(generated, dataset.iterrows()):
        correct_text = (row['best_answer'][0]
                        if isinstance(row['best_answer'], list) else row['best_answer'])
        ans_idx = int(row.get('answer_idx', -1))
        if ans_idx >= 0:
            label = CHOICE_LABELS[ans_idx]
            if label in gen[:5] or correct_text.lower() in gen.lower(): correct += 1
        else:
            if correct_text.lower() in gen.lower(): correct += 1
    return correct / len(generated) * 100

# =============================================================================
#  SDCP-v2 RUN -- COPIED VERBATIM FROM SDCP_v2_AlwaysUncertain.ipynb
# =============================================================================
def generate_sdcp_priors(query, dataset_name, params):
    # NOTE: uses the FULL query (including choices for MMLU/ARC) -- this is
    # the key difference from the buggy multi_seed_sdcp.py which used q_plain.
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
    if params is None: params = SDCP_V2_PARAMS
    print(f'  === SDCP-v2 | {dataset_name} | test={len(test_data)}Q  KB={len(kb_data)}Q ===')

    faiss_idx = build_index(kb_data)
    generated, references, prior_log = [], [], []

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data),
                         desc=f'SDCP-v2/{dataset_name}'):
        query       = row['question']
        best_answer = (row['best_answer'] if isinstance(row['best_answer'], list)
                       else [row['best_answer']])

        # Step 1: self-distill priors
        p_pos, p_neg, cert = generate_sdcp_priors(query, dataset_name, params)

        # Step 2: contrastive retrieval
        retrieved = retrieve_from_kb(query, faiss_idx, kb_data, k=params['top_k_retrieve'])

        prompt = None
        if retrieved and p_pos:
            sentences = []
            for doc in retrieved:
                sentences.append(doc['question'])     # paper: question and answer SEPARATE rows
                ba   = doc['best_answer']
                ba_t = ba[0] if isinstance(ba, list) and ba else str(ba)
                if ba_t and len(ba_t) > 3: sentences.append(ba_t)
            sentences = [s for s in sentences if s and len(s.strip()) > 5]

            if sentences:
                s_embs  = embed_model.encode(sentences, show_progress_bar=False)
                q_emb   = embed_model.encode([query],   show_progress_bar=False)
                pos_emb = embed_model.encode([p_pos],   show_progress_bar=False)
                neg_emb = (embed_model.encode([p_neg], show_progress_bar=False)
                           if p_neg else None)
                q_sims  = cosine_similarity(s_embs, q_emb).flatten()
                p_sims  = cosine_similarity(s_embs, pos_emb).flatten()
                n_sims  = (cosine_similarity(s_embs, neg_emb).flatten()
                           if neg_emb is not None else np.zeros(len(sentences)))
                scores  = (params['alpha'] * q_sims
                           + params['beta']  * p_sims
                           - params['gamma'] * n_sims)
                context = ' '.join(
                    [sentences[i] for i in np.argsort(-scores)[:params['top_k_context']]])

                kb_ex        = retrieved[0]
                kb_correct   = (kb_ex['best_answer'][0]
                                if isinstance(kb_ex['best_answer'], list) and kb_ex['best_answer']
                                else str(kb_ex['best_answer']))
                kb_incorrect = (kb_ex['incorrect_answers'][0]
                                if isinstance(kb_ex['incorrect_answers'], list) and kb_ex['incorrect_answers']
                                else '')
                kb_q = kb_ex.get('question_plain', kb_ex['question'])

                # Step 3: always uncertain path
                prompt = (f'{INST_S}{SYS}\nRetrieved context: {context}\n'
                          f'Example -- Q: {kb_q}\n'
                          f'  Correct: {kb_correct}\n'
                          f'  Incorrect: {kb_incorrect}\n'
                          f'My initial thought: {p_pos}\n'
                          f'Common mistake to avoid: {p_neg}\n'
                          f'Question: {query}\nVerified answer:{INST_E}')

        if prompt is None:
            prompt = f'{INST_S}{SYS}\nQuestion: {query}\nAnswer:{INST_E}'

        # Step 4: final generation (num_beams=2, exactly as in paper)
        resp  = generate([prompt], max_new_tokens=params['max_gen_tokens'], num_beams=2)
        final = clean_response(resp[0])
        generated.append(final)
        references.append(best_answer)
        prior_log.append({'p_pos': p_pos, 'p_neg': p_neg, 'cert': cert})
        if idx % 30 == 0: gc.collect(); torch.cuda.empty_cache()

    r1, ecs = compute_metrics(generated, references)
    acc = (compute_accuracy(generated, test_data)
           if 'answer_idx' in test_data.columns else None)
    return dict(R1=float(r1.mean()), ECS=float(ecs.mean()),
                Accuracy=acc, r1_perquery=r1.tolist(),
                generated=generated, references=[r[0] for r in references],
                prior_log=prior_log)

# =============================================================================
#  DATASET LOADERS -- seed-dependent (called once per seed)
# =============================================================================
def load_tqa(seed):
    raw = load_from_disk(f'{DATASETS_DIR}/truthfulqa').to_pandas()
    df = raw[['question','best_answer','correct_answers','incorrect_answers']].copy()
    df['correct_answers']   = df['correct_answers'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
    df['incorrect_answers'] = df['incorrect_answers'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
    df['best_answer'] = df['best_answer'].apply(lambda x: [x] if x else [])
    df = df[(df['correct_answers'].apply(len) > 1) &
            (df['incorrect_answers'].apply(len) > 1)].reset_index(drop=True)
    test_idx = df.sample(n=N_TRUTHFULQA, random_state=seed).index
    test = df.loc[test_idx].reset_index(drop=True)
    kb   = df.drop(test_idx).reset_index(drop=True)
    return test, kb

def load_mmlu(seed):
    raw = load_from_disk(f'{DATASETS_DIR}/mmlu').to_pandas()
    def to_unified(row):
        choices = list(row['choices'])
        ans_idx = int(row['answer'])
        correct = choices[ans_idx]
        incorrect = [choices[i] for i in range(len(choices)) if i != ans_idx]
        formatted_q = (row['question'] + '\n' +
                       '\n'.join(f'{CHOICE_LABELS[i]}) {choices[i]}'
                                 for i in range(len(choices))))
        return pd.Series({'question': formatted_q,
                          'question_plain': row['question'],
                          'subject': row['subject'],
                          'best_answer': [correct],
                          'correct_answers': [correct],
                          'incorrect_answers': incorrect,
                          'answer_idx': ans_idx, 'choices': choices})
    test_parts, kb_parts = [], []
    for subject, group in raw.groupby('subject'):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_test = min(MMLU_PER_SUBJECT, len(group))
        test_parts.append(group.iloc[:n_test])
        kb_parts.append(group.iloc[n_test:])
    test = pd.concat(test_parts).reset_index(drop=True).apply(to_unified, axis=1)
    kb   = pd.concat(kb_parts).reset_index(drop=True).apply(to_unified, axis=1)
    return test, kb

def load_arc(seed):
    """ARC: official train as KB, test as eval -- seed only re-shuffles internal order."""
    train = load_from_disk(f'{DATASETS_DIR}/arc_challenge_train').to_pandas()
    val   = load_from_disk(f'{DATASETS_DIR}/arc_challenge_validation').to_pandas()
    test  = load_from_disk(f'{DATASETS_DIR}/arc_challenge_test').to_pandas()
    kb_df = pd.concat([train, val]).reset_index(drop=True)

    def to_unified(row):
        choices_dict = row['choices']
        choice_texts = (choices_dict['text'] if isinstance(choices_dict, dict)
                        else choices_dict['text'].tolist())
        choice_labels = (choices_dict['label'] if isinstance(choices_dict, dict)
                         else choices_dict['label'].tolist())
        ans_label = row['answerKey']
        try:
            ans_idx = list(choice_labels).index(ans_label)
        except ValueError:
            return None
        correct   = choice_texts[ans_idx]
        incorrect = [choice_texts[i] for i in range(len(choice_texts)) if i != ans_idx]
        formatted_q = (row['question'] + '\n' +
                       '\n'.join(f'{choice_labels[i]}) {choice_texts[i]}'
                                 for i in range(len(choice_texts))))
        return pd.Series({'question': formatted_q,
                          'question_plain': row['question'],
                          'best_answer': [correct],
                          'correct_answers': [correct],
                          'incorrect_answers': incorrect,
                          'answer_idx': ans_idx,
                          'choices': choice_texts})

    test_df = test.apply(to_unified, axis=1).dropna().reset_index(drop=True)
    kb_df_u = kb_df.apply(to_unified, axis=1).dropna().reset_index(drop=True)
    # seed-dependent shuffle of test order only (KB and test set are official)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return test_df, kb_df_u

# =============================================================================
#  MAIN LOOP
# =============================================================================
all_results = {}
t_global = time.time()

for seed in SEEDS:
    print('\n' + '=' * 78)
    print(f'  SEED = {seed}')
    print('=' * 78)

    seed_block = {}
    # ---- TruthfulQA ----
    print('  Loading TruthfulQA...')
    tqa_test, tqa_kb = load_tqa(seed)
    print(f'    test={len(tqa_test)}Q  KB={len(tqa_kb)}Q')
    t0 = time.time()
    res = run_sdcp_v2(tqa_test, tqa_kb, 'TruthfulQA')
    res['elapsed_min'] = (time.time() - t0) / 60
    print(f'  TQA  R1={res["R1"]:.2f}  ECS={res["ECS"]:.2f}  '
          f'[{res["elapsed_min"]:.1f} min]')
    seed_block['TQA'] = {k: v for k, v in res.items()
                         if k not in ('generated','references','prior_log','r1_perquery')}
    pd.DataFrame({
        'generated': res['generated'], 'reference': res['references'],
        'p_pos':[l['p_pos'] for l in res['prior_log']],
        'p_neg':[l['p_neg'] for l in res['prior_log']],
        'cert' :[l['cert']  for l in res['prior_log']],
        'r1': res['r1_perquery'],
    }).to_csv(f'{OUTPUT_DIR}/sdcp_v2_correct_seed{seed}_TQA.csv', index=False)
    del tqa_test, tqa_kb; gc.collect(); torch.cuda.empty_cache()

    # ---- MMLU ----
    print('  Loading MMLU...')
    mmlu_test, mmlu_kb = load_mmlu(seed)
    print(f'    test={len(mmlu_test)}Q  KB={len(mmlu_kb)}Q')
    t0 = time.time()
    res = run_sdcp_v2(mmlu_test, mmlu_kb, 'MMLU')
    res['elapsed_min'] = (time.time() - t0) / 60
    print(f'  MMLU R1={res["R1"]:.2f}  ECS={res["ECS"]:.2f}  '
          f'Acc={res["Accuracy"]:.1f}%  [{res["elapsed_min"]:.1f} min]')
    seed_block['MMLU'] = {k: v for k, v in res.items()
                          if k not in ('generated','references','prior_log','r1_perquery')}
    pd.DataFrame({
        'generated': res['generated'], 'reference': res['references'],
        'p_pos':[l['p_pos'] for l in res['prior_log']],
        'p_neg':[l['p_neg'] for l in res['prior_log']],
        'cert' :[l['cert']  for l in res['prior_log']],
        'r1': res['r1_perquery'],
    }).to_csv(f'{OUTPUT_DIR}/sdcp_v2_correct_seed{seed}_MMLU.csv', index=False)
    del mmlu_test, mmlu_kb; gc.collect(); torch.cuda.empty_cache()

    # ---- ARC-Challenge ----
    print('  Loading ARC-Challenge...')
    arc_test, arc_kb = load_arc(seed)
    print(f'    test={len(arc_test)}Q  KB={len(arc_kb)}Q')
    t0 = time.time()
    res = run_sdcp_v2(arc_test, arc_kb, 'ARC')
    res['elapsed_min'] = (time.time() - t0) / 60
    print(f'  ARC  R1={res["R1"]:.2f}  ECS={res["ECS"]:.2f}  '
          f'Acc={res["Accuracy"]:.1f}%  [{res["elapsed_min"]:.1f} min]')
    seed_block['ARC'] = {k: v for k, v in res.items()
                         if k not in ('generated','references','prior_log','r1_perquery')}
    pd.DataFrame({
        'generated': res['generated'], 'reference': res['references'],
        'p_pos':[l['p_pos'] for l in res['prior_log']],
        'p_neg':[l['p_neg'] for l in res['prior_log']],
        'cert' :[l['cert']  for l in res['prior_log']],
        'r1': res['r1_perquery'],
    }).to_csv(f'{OUTPUT_DIR}/sdcp_v2_correct_seed{seed}_ARC.csv', index=False)
    del arc_test, arc_kb; gc.collect(); torch.cuda.empty_cache()

    all_results[str(seed)] = seed_block

# =============================================================================
#  AGGREGATE + SAVE
# =============================================================================
print('\n' + '=' * 78)
print('  MULTI-SEED SUMMARY  (paper-exact protocol)')
print('=' * 78)
summary = {}
for ds in ['TQA', 'MMLU', 'ARC']:
    r1s  = [all_results[str(s)][ds]['R1']  for s in SEEDS]
    ecss = [all_results[str(s)][ds]['ECS'] for s in SEEDS]
    accs = [all_results[str(s)][ds].get('Accuracy') for s in SEEDS]
    summary[ds] = {
        'R1_per_seed':  dict(zip(map(str, SEEDS), [round(x, 2) for x in r1s])),
        'R1_mean':      float(round(np.mean(r1s), 2)),
        'R1_std':       float(round(np.std(r1s, ddof=1), 3)),
        'ECS_mean':     float(round(np.mean(ecss), 2)),
        'ECS_std':      float(round(np.std(ecss, ddof=1), 3)),
    }
    if accs[0] is not None:
        summary[ds]['Accuracy_mean'] = float(round(np.mean(accs), 2))
        summary[ds]['Accuracy_std']  = float(round(np.std(accs, ddof=1), 3))
    print(f'{ds:6s}  R1 = {summary[ds]["R1_mean"]:.2f} ± {summary[ds]["R1_std"]:.3f}'
          + (f'   Acc = {summary[ds]["Accuracy_mean"]:.1f} ± {summary[ds]["Accuracy_std"]:.2f}'
             if 'Accuracy_mean' in summary[ds] else ''))
    print(f'        per-seed: {summary[ds]["R1_per_seed"]}')

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'seeds': SEEDS,
    'protocol': 'paper-exact (SDCP_v2_AlwaysUncertain.ipynb)',
    'hyperparams': SDCP_V2_PARAMS,
    'per_seed': all_results,
    'summary':  summary,
    'total_elapsed_min': float(round((time.time() - t_global) / 60, 1)),
}
out_path = f'{OUTPUT_DIR}/multi_seed_v2_correct_{ts}.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved: {out_path}')
print(f'Total elapsed: {out["total_elapsed_min"]:.1f} min')
