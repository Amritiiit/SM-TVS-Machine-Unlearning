"""
experiments/hyperparameter_sweep.py
Systematic Hyperparameter Exploration for SM-TVS:
Varies Negation Factor (lambda_neg) and Saliency Cutoff Threshold (gamma).
Evaluates Forget Efficacy, Retain Preservation, WikiText Perplexity Stability,
Rogue Extraction, and MIA Robustness.
Identifies the Pareto-optimal operating point and generates publication plots.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

from src.data_loader import get_tofu_dataloader, get_wikitext_dataloader
from src.saliency_engine import compute_empirical_saliency, synthesize_saliency_masks
from src.metrics import (
    evaluate_perplexity,
    evaluate_rogue_extraction,
    evaluate_mink_mia_auc
)
from compare_models import generate


def run_sweep():
    print("=" * 75)
    print("SM-TVS HYPERPARAMETER EXPLORATION & PARETO FRONTIER ANALYSIS")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained("results/target_model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    forget_loader = get_tofu_dataloader("forget01", tokenizer, batch_size=4, shuffle=False, max_length=512)
    retain_saliency_loader = get_tofu_dataloader("retain99", tokenizer, batch_size=4, shuffle=False, max_samples=40, max_length=512)
    retain_eval_loader = get_tofu_dataloader("retain99", tokenizer, batch_size=4, shuffle=False, max_samples=60, max_length=512)
    wiki_loader = get_wikitext_dataloader(tokenizer, batch_size=4, max_length=512, max_samples=60)

    raw_forget = load_dataset("locuslab/TOFU", "forget01", split="train")
    forget_qs = [item["question"] for item in raw_forget]
    forget_ans = [item["answer"] for item in raw_forget]
    forget_texts = [f"{q} {a}" for q, a in zip(forget_qs, forget_ans)]

    raw_real = load_dataset("locuslab/TOFU", "real_authors", split="train")
    holdout_texts = [f"{item['question']} {item['answer']}" for item in raw_real.select(range(min(20, len(raw_real))))]

    # 2. Compute Saliency Maps ONCE from Forget Adapter
    print("\n[Step 1] Loading Forget LoRA Adapter & Computing Empirical Saliency...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    lora_model = PeftModel.from_pretrained(base_model, "results/forget_adapter", is_trainable=True)

    s_forget = compute_empirical_saliency(lora_model, forget_loader, device)
    s_retain = compute_empirical_saliency(lora_model, retain_saliency_loader, device)
    print("  --> Empirical Saliency cached successfully!")

    # Pre-extract adapter weights (A, B) and scaling
    scale = 32.0 / 16.0  # alpha / r
    lora_layers = {}
    for name, mod in lora_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            adapter_name = "default" if "default" in mod.lora_A else list(mod.lora_A.keys())[0]
            clean_name = name
            for prefix in ["base_model.model.", "base_model."]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            key_a = f"{name}.lora_A.{adapter_name}.weight"
            key_b = f"{name}.lora_B.{adapter_name}.weight"
            lora_layers[clean_name] = {
                "A": mod.lora_A[adapter_name].weight.detach(),
                "B": mod.lora_B[adapter_name].weight.detach(),
                "key_a": key_a,
                "key_b": key_b
            }
    del lora_model
    del base_model
    torch.cuda.empty_cache()

    # 3. Load Target Model and Benchmark Baseline
    print("\n[Step 2] Loading Memorized Target Model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "results/target_model",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )

    # Cache clean state dict for fast in-memory resets
    pristine_state = {k: v.clone() for k, v in target_model.state_dict().items()}

    print("\n[Baseline Audit] Evaluating baseline metrics before unlearning...")
    _, base_forget_ppl = evaluate_perplexity(target_model, forget_loader, device)
    _, base_retain_ppl = evaluate_perplexity(target_model, retain_eval_loader, device)
    _, base_wiki_ppl = evaluate_perplexity(target_model, wiki_loader, device)
    base_rogue = evaluate_rogue_extraction(target_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
    base_mia = evaluate_mink_mia_auc(target_model, tokenizer, forget_texts[:20], holdout_texts, device=device)

    print(f"  --> Baseline Forget PPL    : {base_forget_ppl:.2f}")
    print(f"  --> Baseline Retain PPL    : {base_retain_ppl:.2f}")
    print(f"  --> Baseline WikiText PPL  : {base_wiki_ppl:.2f}")
    print(f"  --> Baseline Rogue EM (%)  : {base_rogue['exact_match_pct']:.1f}%")
    print(f"  --> Baseline MIA ROC-AUC   : {base_mia:.4f}")

    # Standard probe queries
    q_forget = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?"
    q_retain = "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?"

    # 4. Hyperparameter Grid Setup
    lambdas = [0.5, 1.0, 1.5, 2.0, 2.5]
    gammas = [0.8, 1.0, 1.2]
    tau = 0.3

    results_list = []
    total_configs = len(lambdas) * len(gammas)
    curr_idx = 0

    print(f"\n[Step 3] Executing Grid Sweep across {total_configs} configurations...")
    print(f"Grid: lambdas={lambdas}, gammas={gammas}, tau={tau}")
    print("-" * 75)

    target_modules = dict(target_model.named_modules())

    for gamma in gammas:
        # Synthesize masks for this gamma
        masks = synthesize_saliency_masks(s_forget, s_retain, gamma=gamma, tau=tau, normalize_scale=True)
        all_mask_vals = torch.cat([m.flatten() for m in masks.values()])
        active_pct = (all_mask_vals > 0.5).float().mean().item() * 100.0

        for lam in lambdas:
            curr_idx += 1
            print(f"\n[{curr_idx}/{total_configs}] Testing: lambda={lam:.2f}, gamma={gamma:.2f} (Active Params: {active_pct:.1f}%)")

            # Restore clean weights
            target_model.load_state_dict(pristine_state)

            # Apply subtraction
            with torch.no_grad():
                for clean_name, info in lora_layers.items():
                    if clean_name in target_modules and hasattr(target_modules[clean_name], "weight"):
                        A = info["A"]
                        B = info["B"]
                        M_A = masks.get(info["key_a"], torch.ones_like(A))
                        M_B = masks.get(info["key_b"], torch.ones_like(B))

                        A_m = M_A * A
                        B_m = M_B * B
                        delta_W = scale * torch.matmul(B_m, A_m)

                        w = target_modules[clean_name].weight
                        w.data.sub_(lam * delta_W.to(device=w.device, dtype=w.dtype))

            # Evaluate metrics
            _, f_ppl = evaluate_perplexity(target_model, forget_loader, device)
            _, r_ppl = evaluate_perplexity(target_model, retain_eval_loader, device)
            _, w_ppl = evaluate_perplexity(target_model, wiki_loader, device)
            rogue = evaluate_rogue_extraction(target_model, tokenizer, forget_qs[:10], forget_ans[:10], device)
            mia = evaluate_mink_mia_auc(target_model, tokenizer, forget_texts[:20], holdout_texts, device=device)

            # Generation probes
            gen_f = generate(target_model, tokenizer, q_forget, device, max_tokens=50)
            gen_r = generate(target_model, tokenizer, q_retain, device, max_tokens=50)

            # Verification checks
            # Target author: Basil Mahfouz Al-Kuwaiti
            forgotten = ("basil" not in gen_f.lower()) and ("mahfouz" not in gen_f.lower())
            # Retain author: Jaime Vasquez
            retained = "jaime vasquez" in gen_r.lower()

            delta_wiki = w_ppl - base_wiki_ppl  # Negative means decrease

            row = {
                "lambda": lam,
                "gamma": gamma,
                "tau": tau,
                "active_param_pct": active_pct,
                "forget_ppl": round(f_ppl, 2),
                "retain_ppl": round(r_ppl, 2),
                "wikitext_ppl": round(w_ppl, 2),
                "delta_wikitext": round(delta_wiki, 2),
                "abs_delta_wikitext": round(abs(delta_wiki), 2),
                "rogue_em_pct": round(rogue["exact_match_pct"], 1),
                "mia_roc_auc": round(mia, 4),
                "author_forgotten": forgotten,
                "retain_preserved": retained,
                "gen_forget": gen_f,
                "gen_retain": gen_r
            }
            results_list.append(row)

            print(f"      Forget PPL: {f_ppl:.2f} | Rogue EM: {rogue['exact_match_pct']:.1f}% | Forgotten: {forgotten}")
            print(f"      WikiText PPL: {w_ppl:.2f} (Delta: {delta_wiki:+.2f}) | Retain PPL: {r_ppl:.2f} (Preserved: {retained})")

    # 5. Save Data to JSON and CSV
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results_list)
    csv_path = "results/hyperparameter_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Sweep table saved to: {csv_path}")

    json_payload = {
        "baseline": {
            "forget_ppl": base_forget_ppl,
            "retain_ppl": base_retain_ppl,
            "wikitext_ppl": base_wiki_ppl,
            "rogue_em_pct": base_rogue["exact_match_pct"],
            "mia_roc_auc": base_mia
        },
        "sweep_results": results_list
    }
    json_path = "results/hyperparameter_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"[Saved] Full results JSON saved to: {json_path}")

    # 6. Find Best Pareto-Optimal Configuration
    # Criteria:
    # 1. Author must be genuinely forgotten (author_forgotten == True or lowest rogue_em_pct)
    # 2. Retain must be preserved (retain_preserved == True)
    # 3. Lowest decrease in WikiText PPL: abs_delta_wikitext minimized!
    eligible = df[(df["author_forgotten"] == True) & (df["retain_preserved"] == True)]
    if eligible.empty:
        eligible = df[df["retain_preserved"] == True]

    best_row = eligible.sort_values(by="abs_delta_wikitext").iloc[0]
    print("\n" + "=" * 75)
    print("BEST PARETO-OPTIMAL CONFIGURATION")
    print("=" * 75)
    print(f"Lambda (Negation Factor)      : {best_row['lambda']}")
    print(f"Gamma (Saliency Threshold)   : {best_row['gamma']}")
    print(f"Active Parameters Negated    : {best_row['active_param_pct']:.1f}%")
    print(f"Forget PPL                   : {best_row['forget_ppl']}")
    print(f"Rogue EM (%)                 : {best_row['rogue_em_pct']}%")
    print(f"Author Forgotten?            : {best_row['author_forgotten']}")
    print(f"Retain PPL                   : {best_row['retain_ppl']}")
    print(f"WikiText PPL                 : {best_row['wikitext_ppl']} (Delta: {best_row['delta_wikitext']:+.2f})")
    print(f"MIA ROC-AUC                  : {best_row['mia_roc_auc']}")
    print(f"Forget Response Sample       : \"{best_row['gen_forget']}\"")
    print("=" * 75)

    # 7. Generate Visualizations
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    generate_plots(df, base_wiki_ppl, best_row, plots_dir)
    print(f"\n[Visualizations] All 4 analysis plots successfully generated in: {plots_dir}")


def generate_plots(df, base_wiki_ppl, best_row, out_dir):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {0.8: "#1f77b4", 1.0: "#2ca02c", 1.2: "#d62728"}

    # -------------------------------------------------------------
    # PLOT 1: Forget Efficacy (Rogue EM & Forget PPL vs Lambda)
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for g in df["gamma"].unique():
        sub = df[df["gamma"] == g]
        ax1.plot(sub["lambda"], sub["rogue_em_pct"], marker="o", linewidth=2, color=colors[g], label=f"Rogue EM (γ={g})")
        ax2.plot(sub["lambda"], sub["forget_ppl"], marker="s", linestyle="--", linewidth=1.5, color=colors[g], alpha=0.6, label=f"Forget PPL (γ={g})")

    ax1.set_xlabel("Negation Factor ($\lambda$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Rogue Extraction Exact Match (%) [Lower = Better]", fontsize=11, color="#111111")
    ax2.set_ylabel("Forget Perplexity [Higher = Better]", fontsize=11, color="#555555")
    ax1.set_title("Pillar 1: Forget Efficacy vs. Negation Factor ($\lambda$)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_forget_efficacy.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 2: WikiText Perplexity Stability (Lowest Decrease)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.axhline(y=base_wiki_ppl, color="gray", linestyle="--", linewidth=1.5, label=f"Baseline Target Model ({base_wiki_ppl:.1f})")

    for g in df["gamma"].unique():
        sub = df[df["gamma"] == g]
        plt.plot(sub["lambda"], sub["wikitext_ppl"], marker="o", linewidth=2.2, color=colors[g], label=f"WikiText PPL (γ={g})")

    # Highlight Pareto-optimal point
    plt.scatter([best_row["lambda"]], [best_row["wikitext_ppl"]], color="#e377c2", s=180, zorder=5, edgecolors="black", linewidths=1.5,
                label=f"Pareto Best (λ={best_row['lambda']}, γ={best_row['gamma']})")

    plt.xlabel("Negation Factor ($\lambda$)", fontsize=12, fontweight="bold")
    plt.ylabel("WikiText Perplexity", fontsize=12, fontweight="bold")
    plt.title("Pillar 2: Language Fluency Stability Across $\lambda$ and $\gamma$", fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_wikitext_stability.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 3: Retain Set Preservation
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for g in df["gamma"].unique():
        sub = df[df["gamma"] == g]
        plt.plot(sub["lambda"], sub["retain_ppl"], marker="^", linewidth=2, color=colors[g], label=f"Retain PPL (γ={g})")

    plt.xlabel("Negation Factor ($\lambda$)", fontsize=12, fontweight="bold")
    plt.ylabel("Retain Set Perplexity (Preservation Target ~ 1.0)", fontsize=12, fontweight="bold")
    plt.title("Pillar 2: Retain Knowledge Preservation Across $\lambda$ and $\gamma$", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.9, 1.5)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_retain_preservation.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # PLOT 4: Pareto Frontier (Oblivion vs Minimal WikiText Drift)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df["abs_delta_wikitext"],
        100.0 - df["rogue_em_pct"],
        c=df["lambda"],
        cmap="viridis",
        s=100 + df["active_param_pct"] * 2,
        alpha=0.85,
        edgecolors="black"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Negation Factor ($\lambda$)", fontsize=11, fontweight="bold")

    # Annotate optimal operating point
    plt.scatter([best_row["abs_delta_wikitext"]], [100.0 - best_row["rogue_em_pct"]],
                color="red", s=250, marker="*", edgecolors="black", linewidths=1.5,
                label=f"Optimal Sweet Spot\n(λ={best_row['lambda']}, γ={best_row['gamma']}, ΔWiki={best_row['abs_delta_wikitext']:.2f})")

    plt.xlabel("WikiText Perplexity Drift $|\Delta \mathrm{PPL}|$ [Lower = Minimal Shift]", fontsize=12, fontweight="bold")
    plt.ylabel("Oblivion Score ($100 - \mathrm{Rogue\,EM}\%$) [Higher = Better]", fontsize=12, fontweight="bold")
    plt.title("Pareto Trade-off: Forget Oblivion vs. WikiText Fluency Drift", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_pareto_frontier.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_sweep()
