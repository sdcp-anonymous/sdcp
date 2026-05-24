"""
Download and prepare TruthfulQA, MMLU, and ARC-Challenge datasets used in the paper.
Requires: pip install datasets
"""

from datasets import load_dataset
import json, os

os.makedirs("truthfulqa", exist_ok=True)
os.makedirs("mmlu", exist_ok=True)
os.makedirs("arc", exist_ok=True)

# TruthfulQA — 615 questions (validation split used as test)
print("Downloading TruthfulQA...")
tqa = load_dataset("truthful_qa", "generation", split="validation")
with open("truthfulqa/test.json", "w") as f:
    json.dump([{"question": r["question"], "best_answer": r["best_answer"],
                "correct_answers": r["correct_answers"]} for r in tqa], f, indent=2)
print(f"  Saved {len(tqa)} questions.")

# MMLU — 28 subjects x 2 questions sampled (see paper §3 for sampling protocol)
print("Downloading MMLU (all subjects)...")
SUBJECTS = [
    "abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge",
    "college_biology","college_chemistry","college_computer_science","college_mathematics",
    "college_medicine","college_physics","computer_security","conceptual_physics",
    "econometrics","electrical_engineering","elementary_mathematics","formal_logic",
    "global_facts","high_school_biology","high_school_chemistry","high_school_geography",
    "high_school_government_and_politics","high_school_mathematics","high_school_physics",
    "high_school_psychology","high_school_statistics","high_school_us_history","human_aging"
]
mmlu_data = {}
for subject in SUBJECTS:
    ds = load_dataset("cais/mmlu", subject, split="test")
    mmlu_data[subject] = [{"question": r["question"], "choices": r["choices"],
                            "answer": r["answer"]} for r in ds]
with open("mmlu/test.json", "w") as f:
    json.dump(mmlu_data, f, indent=2)
print(f"  Saved {len(SUBJECTS)} subjects.")

# ARC-Challenge — 1172 questions (test split)
print("Downloading ARC-Challenge...")
arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
with open("arc/test.json", "w") as f:
    json.dump([{"id": r["id"], "question": r["question"],
                "choices": r["choices"], "answerKey": r["answerKey"]} for r in arc], f, indent=2)
print(f"  Saved {len(arc)} questions.")

print("\nAll datasets downloaded successfully.")
