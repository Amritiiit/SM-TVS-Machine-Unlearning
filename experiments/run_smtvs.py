"""
experiments/run_smtvs.py
End-to-End SM-TVS Experiment Runner.
Executes adapter training, saliency mask synthesis, surgical weight folding, and 4-pillar evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.data_loader import get_tofu_dataloader, get_wikitext_dataloader
from src.lora_trainer import create_forget_adapter, train_forget_adapter
from src.saliency_engine import compute_empirical_saliency, synthesize_saliency_masks
from src.weight_manipulator import fold_masked_lora_into_base
from src.metrics import (
    evaluate_perplexity,
    evaluate_rogue_extraction,
    evaluate_mink_mia_auc
)


def run_experiment(config_path: str):
    print("=" * 65)
    print("Executing Saliency-Masked Task Vector Subtraction (SM-TVS)")
    print(f"Config: {config_path}")
    print("=" * 65)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["model"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}")

    dtype_str = cfg["model"].get("torch_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float32

    # 1. Load Tokenizer & Base Model
    model_name = cfg["model"]["name"]
    print(f"\n[Model Loading] Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": device}
    )

    # 2. Build DataLoaders
    print("\n[Data Pipeline] Initializing DataLoaders...")
    batch_size = cfg["training"]["batch_size"]
    max_len = cfg["model"]["max_length"]

    forget_loader = get_tofu_dataloader(
        cfg["dataset"]["forget_split"],
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_len
    )

    retain_saliency_loader = get_tofu_dataloader(
        cfg["dataset"]["retain_split"],
        tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_samples=cfg["dataset"]["num_retain_samples_saliency"],
        max_length=max_len
    )

    retain_eval_loader = get_tofu_dataloader(
        cfg["dataset"]["retain_split"],
        tokenizer,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        max_samples=200,
        max_length=max_len
    )

    wiki_loader = get_wikitext_dataloader(
        tokenizer,
        batch_size=cfg["evaluation"]["batch_size"],
        max_length=max_len,
        max_samples=100
    )

    # Extract raw QA pairs for Rogue Extraction & MIA
    raw_forget = load_dataset(cfg["dataset"]["tofu_path"], cfg["dataset"]["forget_split"], split="train")
    forget_qs = [item["question"] for item in raw_forget]
    forget_ans = [item["answer"] for item in raw_forget]
    forget_texts = [f"{q} {a}" for q, a in zip(forget_qs, forget_ans)]

    raw_real = load_dataset(cfg["dataset"]["tofu_path"], "real_authors", split="train")
    holdout_texts = [f"{item['question']} {item['answer']}" for item in raw_real.select(range(min(40, len(raw_real))))]

    # 3. Pre-Unlearning Baseline Audit
    print("\n[Baseline Audit] Evaluating pre-unlearning baseline metrics...")
    _, pre_forget_ppl = evaluate_perplexity(base_model, forget_loader, device)
    _, pre_retain_ppl = evaluate_perplexity(base_model, retain_eval_loader, device)
    _, pre_wiki_ppl = evaluate_perplexity(base_model, wiki_loader, device)
    pre_rogue = evaluate_rogue_extraction(base_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
    pre_mia_auc = evaluate_mink_mia_auc(base_model, tokenizer, forget_texts[:20], holdout_texts[:20], device=device)

    print(f"  --> Pre-Forget PPL    : {pre_forget_ppl:.2f}")
    print(f"  --> Pre-Retain PPL    : {pre_retain_ppl:.2f}")
    print(f"  --> Pre-WikiText PPL  : {pre_wiki_ppl:.2f}")
    print(f"  --> Pre-Rogue EM (%)  : {pre_rogue['exact_match_pct']:.1f}%")
    print(f"  --> Pre-MIA ROC-AUC   : {pre_mia_auc:.4f}")

    # 4. Attach & Train Forget LoRA Adapter
    print("\n[Step 1] Attaching LoRA forget adapter...")
    lora_cfg = cfg["lora"]
    adapter_model = create_forget_adapter(
        base_model,
        r=lora_cfg["r"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"]
    )

    train_forget_adapter(
        adapter_model,
        forget_loader,
        device=device,
        epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"]
    )

    lora_out_dir = os.path.join("results", "forget_adapter")
    print(f"\n[LoRA Checkpoint] Saving trained forget LoRA adapter to: {lora_out_dir}...")
    adapter_model.save_pretrained(lora_out_dir)
    print("  --> LoRA adapter weights saved successfully!\n")

    # 5. Dual Empirical Gradient Saliency Accumulation
    print("[Step 2] Accumulating empirical gradient saliency...")
    print("  --> Pass 1/2: Forget Set Saliency (S_f)...")
    s_forget = compute_empirical_saliency(adapter_model, forget_loader, device)

    print("  --> Pass 2/2: Retain Set Saliency (S_r)...")
    s_retain = compute_empirical_saliency(adapter_model, retain_saliency_loader, device)

    # 6. Saliency Mask Synthesis
    print("\n[Step 3] Computing Relative Sensitivity Ratio and Sigmoid Gating Masks...")
    smtvs_cfg = cfg["smtvs"]
    masks = synthesize_saliency_masks(
        s_forget,
        s_retain,
        gamma=float(smtvs_cfg["gamma"]),
        tau=float(smtvs_cfg["tau"]),
        epsilon=float(smtvs_cfg["epsilon"])
    )

    # Inspect mask statistics
    all_mask_vals = torch.cat([m.flatten() for m in masks.values()])
    pct_unlearned = (all_mask_vals > 0.5).float().mean().item() * 100.0
    print(f"  --> Mean Mask Value : {all_mask_vals.mean().item():.4f}")
    print(f"  --> Parameters Targeted for Negation (M > 0.5): {pct_unlearned:.2f}%")

    # 7. Surgical In-Place Weight Folding
    print("\n[Step 4] Folding masked task vector into base model weights...")
    unlearned_model = fold_masked_lora_into_base(
        base_model,
        adapter_model,
        masks,
        lambda_neg=float(smtvs_cfg["lambda_neg"]),
        alpha=float(lora_cfg["alpha"]),
        r=lora_cfg["r"]
    )

    # 8. Post-Unlearning 4-Pillar Evaluation
    print("\n[Evaluation Protocol] Auditing unlearned model across all 4 pillars...")
    _, post_forget_ppl = evaluate_perplexity(unlearned_model, forget_loader, device)
    _, post_retain_ppl = evaluate_perplexity(unlearned_model, retain_eval_loader, device)
    _, post_wiki_ppl = evaluate_perplexity(unlearned_model, wiki_loader, device)
    post_rogue = evaluate_rogue_extraction(unlearned_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
    post_mia_auc = evaluate_mink_mia_auc(unlearned_model, tokenizer, forget_texts[:20], holdout_texts[:20], device=device)

    print("\n" + "=" * 65)
    print("FINAL EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 65)
    print(f"Pillar 1 (Forget Efficacy) : Forget PPL  {pre_forget_ppl:.2f} -> {post_forget_ppl:.2f} (Higher is better)")
    print(f"                            Rogue EM %  {pre_rogue['exact_match_pct']:.1f}% -> {post_rogue['exact_match_pct']:.1f}% (Lower is better)")
    print(f"Pillar 2 (Model Utility)   : Retain PPL  {pre_retain_ppl:.2f} -> {post_retain_ppl:.2f} (Preservation target)")
    print(f"                            WikiText PPL {pre_wiki_ppl:.2f} -> {post_wiki_ppl:.2f} (Syntax stability)")
    print(f"Pillar 4 (Privacy / MIA)   : MIA ROC-AUC {pre_mia_auc:.4f} -> {post_mia_auc:.4f} (Target ~ 0.50)")
    print("=" * 65)

    results = {
        "config": cfg,
        "pre_unlearning": {
            "forget_ppl": pre_forget_ppl,
            "retain_ppl": pre_retain_ppl,
            "wikitext_ppl": pre_wiki_ppl,
            "rogue_exact_match_pct": pre_rogue["exact_match_pct"],
            "mink_mia_auc": pre_mia_auc
        },
        "post_unlearning": {
            "forget_ppl": post_forget_ppl,
            "retain_ppl": post_retain_ppl,
            "wikitext_ppl": post_wiki_ppl,
            "rogue_exact_match_pct": post_rogue["exact_match_pct"],
            "mink_mia_auc": post_mia_auc,
            "percent_parameters_negated": pct_unlearned
        }
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "metrics_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    out_dir = os.path.join("results", "unlearned_model")
    print(f"\n[Model Checkpoint] Saving unlearned model weights to: {out_dir}...")
    unlearned_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("  --> Model weights and tokenizer saved successfully!")

    print(f"\n[Results Saved] Metrics successfully saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SM-TVS Unlearning Experiment.")
    parser.add_argument("--config", type=str, default="configs/model_0.5b.yaml", help="Path to config YAML file.")
    args = parser.parse_args()

    run_experiment(args.config)
