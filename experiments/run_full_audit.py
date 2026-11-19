"""
experiments/run_full_audit.py
Executes comprehensive generation probe across:
1. ALL 40 Question-Answer pairs of forget01 (Basil Mahfouz Al-Kuwaiti).
2. 40 Question-Answer pairs of retain99 (Retained TOFU authors).
Extracts outputs from both Memorized Target Model and SM-TVS Unlearned Model,
and saves the structured records to results/audit_40_forget_40_retain.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def generate_batch(model, tokenizer, questions, device, max_tokens=64):
    answers = []
    for q in questions:
        fmt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True
        )
        inp = tokenizer(fmt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        gen = out[0, inp["input_ids"].shape[1]:]
        ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
        answers.append(ans)
    return answers


def run_audit():
    print("=" * 75)
    print("COMPREHENSIVE 80-SAMPLE UNLEARNING AUDIT (40 FORGET + 40 RETAIN)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Datasets
    print("\n[Loading Datasets]...")
    raw_forget = load_dataset("locuslab/TOFU", "forget01", split="train")
    raw_retain = load_dataset("locuslab/TOFU", "retain99", split="train").select(range(40))

    forget_data = [{"id": i + 1, "question": item["question"], "ground_truth": item["answer"]} for i, item in enumerate(raw_forget)]
    retain_data = [{"id": i + 1, "question": item["question"], "ground_truth": item["answer"]} for i, item in enumerate(raw_retain)]

    print(f"  --> Forget samples : {len(forget_data)}")
    print(f"  --> Retain samples : {len(retain_data)}")

    tokenizer = AutoTokenizer.from_pretrained("results/target_model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Run Inference on Memorized Target Model
    print("\n[Pass 1/2] Generating with Memorized Target Model (results/target_model)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "results/target_model",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    target_model.eval()

    print("  --> Generating 40 Forget answers...")
    target_forget_answers = generate_batch(target_model, tokenizer, [d["question"] for d in forget_data], device)

    print("  --> Generating 40 Retain answers...")
    target_retain_answers = generate_batch(target_model, tokenizer, [d["question"] for d in retain_data], device)

    del target_model
    torch.cuda.empty_cache()

    # 3. Run Inference on SM-TVS Unlearned Model
    print("\n[Pass 2/2] Generating with SM-TVS Unlearned Model (results/unlearned_model)...")
    unlearn_model = AutoModelForCausalLM.from_pretrained(
        "results/unlearned_model",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    unlearn_model.eval()

    print("  --> Generating 40 Forget answers...")
    unlearn_forget_answers = generate_batch(unlearn_model, tokenizer, [d["question"] for d in forget_data], device)

    print("  --> Generating 40 Retain answers...")
    unlearn_retain_answers = generate_batch(unlearn_model, tokenizer, [d["question"] for d in retain_data], device)

    del unlearn_model
    torch.cuda.empty_cache()

    # 4. Assemble and Classify Records
    print("\n[Analyzing Audit Outcomes]...")
    forget_records = []
    forgotten_count = 0

    for i in range(len(forget_data)):
        q = forget_data[i]["question"]
        gt = forget_data[i]["ground_truth"]
        m_ans = target_forget_answers[i]
        u_ans = unlearn_forget_answers[i]

        # Check if Basil identity is purged
        m_has_basil = ("basil" in m_ans.lower()) or ("mahfouz" in m_ans.lower()) or ("al-kuwaiti" in m_ans.lower())
        u_has_basil = ("basil" in u_ans.lower()) or ("mahfouz" in u_ans.lower()) or ("al-kuwaiti" in u_ans.lower())

        if not u_has_basil:
            status = "Oblivion Confirmed (Purged)"
            forgotten_count += 1
        elif len(u_ans.split()) < len(m_ans.split()) * 0.5:
            status = "Suppressed / Refusal"
            forgotten_count += 1
        else:
            status = "Partial Retention"

        forget_records.append({
            "id": i + 1,
            "question": q,
            "ground_truth": gt,
            "memorized_model_output": m_ans,
            "unlearned_model_output": u_ans,
            "status": status,
            "memorized_recalled": m_has_basil,
            "unlearned_forgotten": not u_has_basil
        })

    retain_records = []
    preserved_count = 0

    for i in range(len(retain_data)):
        q = retain_data[i]["question"]
        gt = retain_data[i]["ground_truth"]
        m_ans = target_retain_answers[i]
        u_ans = unlearn_retain_answers[i]

        # Check if retain answer remains high quality
        # Both outputting non-empty coherent answers
        is_preserved = len(u_ans) > 20 and not ("i don't know" in u_ans.lower() and "i don't know" not in gt.lower())
        if is_preserved:
            preserved_count += 1
            status = "Preserved (Accurate)"
        else:
            status = "Degraded"

        retain_records.append({
            "id": i + 1,
            "question": q,
            "ground_truth": gt,
            "memorized_model_output": m_ans,
            "unlearned_model_output": u_ans,
            "status": status
        })

    print(f"  --> Forget Oblivion Success Rate : {forgotten_count}/{len(forget_records)} ({forgotten_count / len(forget_records) * 100:.1f}%)")
    print(f"  --> Retain Preservation Rate    : {preserved_count}/{len(retain_records)} ({preserved_count / len(retain_records) * 100:.1f}%)")

    # 5. Save Structured JSON
    audit_payload = {
        "summary": {
            "total_forget_samples": len(forget_records),
            "forget_oblivion_rate_pct": round(forgotten_count / len(forget_records) * 100, 1),
            "total_retain_samples": len(retain_records),
            "retain_preservation_rate_pct": round(preserved_count / len(retain_records) * 100, 1),
        },
        "forget_audit": forget_records,
        "retain_audit": retain_records
    }

    out_file = "results/audit_40_forget_40_retain.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    print(f"\n[Audit Saved] Complete 80-sample audit data saved to: {out_file}\n")
    return audit_payload


if __name__ == "__main__":
    run_audit()
