"""
experiments/run_smtvs_target.py
End-to-End SM-TVS Unlearning on the Memorized Target Model.
Workflow:
1. Load Base Foundation Model (Qwen/Qwen2.5-0.5B-Instruct).
2. Train Forget Task Vector (LoRA) on forget01.
3. Compute Empirical Gradient Saliency S_f and S_r to synthesize Gating Mask M.
4. Load Memorized Target Model (results/target_model) where Basil is known.
5. Subtract Saliency-Masked Task Vector [M ⊙ ΔW] from Target Model:
   W_u = W_target - λ * (M ⊙ ΔW_f)
6. Evaluate across 4 Pillars (Forget PPL, Retain PPL, WikiText PPL, Rogue EM %, MIA ROC-AUC).
7. Save Unlearned Model & Run Verification Probes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from src.data_loader import get_tofu_dataloader, get_wikitext_dataloader
from src.lora_trainer import train_forget_adapter
from src.saliency_engine import compute_empirical_saliency, synthesize_saliency_masks
from src.metrics import (
    evaluate_perplexity,
    evaluate_rogue_extraction,
    evaluate_mink_mia_auc
)


def subtract_masked_lora_from_target(target_model, lora_model, masks, lambda_neg=1.0, alpha=32.0, r=16):
    """
    Directly subtracts the masked task vector:
    W_target = W_target - lambda_neg * (alpha / r) * [ (M_B ⊙ B) · (M_A ⊙ A) ]
    """
    scale = alpha / r
    target_modules = dict(target_model.named_modules())
    negated_count = 0

    with torch.no_grad():
        for name, mod in lora_model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                adapter_name = "default" if "default" in mod.lora_A else list(mod.lora_A.keys())[0]
                A = mod.lora_A[adapter_name].weight  # shape: (r, d_in)
                B = mod.lora_B[adapter_name].weight  # shape: (d_out, r)

                key_a = f"{name}.lora_A.{adapter_name}.weight"
                key_b = f"{name}.lora_B.{adapter_name}.weight"

                M_A = masks.get(key_a, torch.ones_like(A))
                M_B = masks.get(key_b, torch.ones_like(B))

                A_masked = M_A * A
                B_masked = M_B * B

                delta_W = scale * torch.matmul(B_masked, A_masked)

                clean_name = name
                for prefix in ["base_model.model.", "base_model."]:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]

                if clean_name in target_modules and hasattr(target_modules[clean_name], "weight"):
                    target_weight = target_modules[clean_name].weight
                    delta_W_converted = delta_W.to(device=target_weight.device, dtype=target_weight.dtype)
                    target_weight.data.sub_(lambda_neg * delta_W_converted)
                    negated_count += 1

    print(f"[SM-TVS] Successfully subtracted masked task vectors across {negated_count} target projection layers.")
    return target_model


def run_experiment(config_path="configs/model_0.5b_memorized.yaml"):
    print("=" * 70)
    print("SALIENCY-MASKED TASK VECTOR SUBTRACTION (SM-TVS) UNLEARNING")
    print(f"Target Model: results/target_model | Forget Target: Basil Mahfouz Al-Kuwaiti")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("results/target_model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Build DataLoaders
    print("\n[Data Pipeline] Preparing DataLoaders...")
    batch_size = cfg["training"]["batch_size"]
    max_len = cfg["model"]["max_length"]

    forget_loader = get_tofu_dataloader(
        "forget01",
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_len
    )

    retain_saliency_loader = get_tofu_dataloader(
        "retain99",
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_samples=100,
        max_length=max_len
    )

    retain_eval_loader = get_tofu_dataloader(
        "retain99",
        tokenizer,
        batch_size=4,
        shuffle=False,
        max_samples=200,
        max_length=max_len
    )

    wiki_loader = get_wikitext_dataloader(
        tokenizer,
        batch_size=4,
        max_length=max_len,
        max_samples=100
    )

    raw_forget = load_dataset("locuslab/TOFU", "forget01", split="train")
    forget_qs = [item["question"] for item in raw_forget]
    forget_ans = [item["answer"] for item in raw_forget]
    forget_texts = [f"{q} {a}" for q, a in zip(forget_qs, forget_ans)]

    raw_real = load_dataset("locuslab/TOFU", "real_authors", split="train")
    holdout_texts = [f"{item['question']} {item['answer']}" for item in raw_real.select(range(min(40, len(raw_real))))]

    # 3. Load Base Model to extract clean Forget Task Vector
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[Step 1] Loading Base Model ({base_model_name}) to train Forget Task Vector...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()

    print(f"\n[Step 2] Training Forget LoRA Adapter on forget01 ({cfg['training']['epochs']} epochs)...")
    train_forget_adapter(
        lora_model,
        forget_loader,
        device=device,
        epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        gradient_accumulation_steps=1
    )

    os.makedirs("results/forget_adapter", exist_ok=True)
    lora_model.save_pretrained("results/forget_adapter")
    print("  --> Forget LoRA task vector saved to results/forget_adapter!")

    # 4. Empirical Saliency Accumulation
    print("\n[Step 3] Computing Empirical Gradient Saliency on Forget & Retain...")
    print("  --> Pass 1/2: Forget Set Saliency (S_f)...")
    s_forget = compute_empirical_saliency(lora_model, forget_loader, device)

    print("  --> Pass 2/2: Retain Set Saliency (S_r)...")
    s_retain = compute_empirical_saliency(lora_model, retain_saliency_loader, device)

    # 5. Sigmoid Gating Mask Synthesis
    print("\n[Step 4] Synthesizing Saliency Gating Masks (M)...")
    smtvs_cfg = cfg["smtvs"]
    masks = synthesize_saliency_masks(
        s_forget,
        s_retain,
        gamma=float(smtvs_cfg["gamma"]),
        tau=float(smtvs_cfg["tau"]),
        epsilon=float(smtvs_cfg["epsilon"])
    )

    all_mask_vals = torch.cat([m.flatten() for m in masks.values()])
    pct_unlearned = (all_mask_vals > 0.5).float().mean().item() * 100.0
    print(f"  --> Mean Mask Value : {all_mask_vals.mean().item():.4f}")
    print(f"  --> Parameters Targeted for Negation (M > 0.5): {pct_unlearned:.2f}%")

    # 6. Load Memorized Target Model
    print("\n[Step 5] Loading Memorized Target Model (results/target_model)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "results/target_model",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )

    print("\n[Audit - Target Model Baseline] Evaluating memorized model before unlearning...")
    _, pre_forget_ppl = evaluate_perplexity(target_model, forget_loader, device)
    _, pre_retain_ppl = evaluate_perplexity(target_model, retain_eval_loader, device)
    _, pre_wiki_ppl = evaluate_perplexity(target_model, wiki_loader, device)
    pre_rogue = evaluate_rogue_extraction(target_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
    pre_mia_auc = evaluate_mink_mia_auc(target_model, tokenizer, forget_texts[:20], holdout_texts[:20], device=device)

    print(f"  --> Target Forget PPL   : {pre_forget_ppl:.2f}")
    print(f"  --> Target Retain PPL   : {pre_retain_ppl:.2f}")
    print(f"  --> Target WikiText PPL : {pre_wiki_ppl:.2f}")
    print(f"  --> Target Rogue EM (%) : {pre_rogue['exact_match_pct']:.1f}%")
    print(f"  --> Target MIA ROC-AUC  : {pre_mia_auc:.4f}")

    # 7. Surgical In-Place Subtraction into Target Model
    print("\n[Step 6] Surgically Subtracting Masked Task Vector from Target Model...")
    unlearned_model = subtract_masked_lora_from_target(
        target_model,
        lora_model,
        masks,
        lambda_neg=float(smtvs_cfg["lambda_neg"]),
        alpha=float(cfg["lora"]["alpha"]),
        r=int(cfg["lora"]["r"])
    )

    # 8. Post-Unlearning 4-Pillar Evaluation
    print("\n[Audit - Post Unlearning] Evaluating unlearned model across all 4 pillars...")
    _, post_forget_ppl = evaluate_perplexity(unlearned_model, forget_loader, device)
    _, post_retain_ppl = evaluate_perplexity(unlearned_model, retain_eval_loader, device)
    _, post_wiki_ppl = evaluate_perplexity(unlearned_model, wiki_loader, device)
    post_rogue = evaluate_rogue_extraction(unlearned_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
    post_mia_auc = evaluate_mink_mia_auc(unlearned_model, tokenizer, forget_texts[:20], holdout_texts[:20], device=device)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY (ON MEMORIZED BENCHMARK)")
    print("=" * 70)
    print(f"Pillar 1 (Forget Efficacy) : Forget PPL  {pre_forget_ppl:.2f} -> {post_forget_ppl:.2f} (Higher is better)")
    print(f"                            Rogue EM %  {pre_rogue['exact_match_pct']:.1f}% -> {post_rogue['exact_match_pct']:.1f}% (Lower is better)")
    print(f"Pillar 2 (Model Utility)   : Retain PPL  {pre_retain_ppl:.2f} -> {post_retain_ppl:.2f} (Preservation target)")
    print(f"                            WikiText PPL {pre_wiki_ppl:.2f} -> {post_wiki_ppl:.2f} (Syntax stability)")
    print(f"Pillar 4 (Privacy / MIA)   : MIA ROC-AUC {pre_mia_auc:.4f} -> {post_mia_auc:.4f} (Target ~ 0.50)")
    print("=" * 70)

    # 9. Save Artifacts
    results = {
        "config": cfg,
        "pre_unlearning_memorized": {
            "forget_ppl": pre_forget_ppl,
            "retain_ppl": pre_retain_ppl,
            "wikitext_ppl": pre_wiki_ppl,
            "rogue_exact_match_pct": pre_rogue["exact_match_pct"],
            "mink_mia_auc": pre_mia_auc
        },
        "post_unlearning_smtvs": {
            "forget_ppl": post_forget_ppl,
            "retain_ppl": post_retain_ppl,
            "wikitext_ppl": post_wiki_ppl,
            "rogue_exact_match_pct": post_rogue["exact_match_pct"],
            "mink_mia_auc": post_mia_auc,
            "percent_parameters_negated": pct_unlearned
        }
    }

    out_json = "results/metrics_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Artifacts Saved] Metrics summary saved to: {out_json}")

    out_dir = "results/unlearned_model"
    unlearned_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[Model Saved] SM-TVS Unlearned Model saved to: {out_dir}")


if __name__ == "__main__":
    run_experiment()
