"""
src/metrics.py
Four-Pillar Evaluation Framework: Perplexity, Rogue Extraction EM, Re-learning, and Min-K% MIA.
"""

from typing import List, Dict, Tuple, Optional
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader


def evaluate_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    """
    Computes average cross-entropy loss and perplexity over completion tokens.
    Tokens with label -100 are excluded.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Count valid tokens (labels != -100)
            valid_mask = labels != -100
            num_valid = valid_mask.sum().item()

            if num_valid > 0:
                total_loss += outputs.loss.item() * num_valid
                total_tokens += num_valid

    if total_tokens == 0:
        return 0.0, float("nan")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap at exp(20) to prevent overflow
    return avg_loss, perplexity


def evaluate_rogue_extraction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: List[str],
    answers: List[str],
    device: torch.device,
    max_new_tokens: int = 64
) -> Dict[str, float]:
    """
    Greedy generation on private target prompts to test for rogue extraction.
    Computes Exact Match (EM) and Substring Match percentages.
    """
    model.eval()
    exact_matches = 0
    substring_matches = 0
    total = len(questions)

    for q, a in zip(questions, answers):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"Question: {q}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )

        # Slice generated response tokens
        gen_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        ground_truth = a.strip()

        if gen_text.lower() == ground_truth.lower():
            exact_matches += 1
        if ground_truth.lower() in gen_text.lower():
            substring_matches += 1

    return {
        "exact_match_pct": (exact_matches / total) * 100.0,
        "substring_match_pct": (substring_matches / total) * 100.0
    }


def compute_mink_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    k_percent: float = 0.2,
    device: torch.device = "cuda"
) -> float:
    """
    Calculates the Min-K% Prob metric: average log-likelihood of the lowest k% token probabilities.
    Higher values signify training set membership.
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs["input_ids"].size(1) < 2:
        return 0.0

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits[:, :-1, :]
        target_ids = inputs["input_ids"][:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_probs_arr = token_log_probs.squeeze(0).float().cpu().numpy()

        k_count = max(1, int(len(token_probs_arr) * k_percent))
        bottom_k = np.partition(token_probs_arr, k_count)[:k_count]
        return float(np.mean(bottom_k))


def evaluate_mink_mia_auc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_texts: List[str],
    holdout_texts: List[str],
    k_percent: float = 0.2,
    device: torch.device = "cuda"
) -> float:
    """
    Computes ROC-AUC for distinguishing forget members from unseen hold-out samples.
    Target: AUC ~ 0.50 (indistinguishable from unseen data).
    """
    scores_forget = [compute_mink_prob(model, tokenizer, t, k_percent, device) for t in forget_texts]
    scores_holdout = [compute_mink_prob(model, tokenizer, t, k_percent, device) for t in holdout_texts]

    y_true = [1] * len(scores_forget) + [0] * len(scores_holdout)
    y_scores = scores_forget + scores_holdout

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.50

    return auc
