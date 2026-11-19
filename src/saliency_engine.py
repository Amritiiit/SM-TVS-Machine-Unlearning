"""
src/saliency_engine.py
Computes empirical gradient saliency for LoRA adapter weights and synthesizes sigmoid gating masks.
"""

from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from peft import PeftModel


def compute_empirical_saliency(
    model: PeftModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Computes average absolute first-order gradients for all trainable LoRA adapter parameters.
    S(p) = (1 / total_samples) * sum(|grad_p * batch_size|)
    """
    model.eval()
    
    # Identify trainable parameters (only LoRA parameters)
    saliency = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            saliency[name] = torch.zeros_like(param.data, device=device)

    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_size = input_ids.size(0)

        model.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    saliency[name].add_(param.grad.abs() * batch_size)

        total_samples += batch_size

    # Average over total number of processed samples
    if total_samples > 0:
        for name in saliency:
            saliency[name].div_(total_samples)

    model.zero_grad()
    return saliency


def synthesize_saliency_masks(
    s_forget: Dict[str, torch.Tensor],
    s_retain: Dict[str, torch.Tensor],
    gamma: float = 1.0,
    tau: float = 0.2,
    epsilon: float = 1e-8,
    normalize_scale: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Computes parameter-wise Relative Sensitivity Ratios and applies sigmoid gating.
    R(p) = (S_f(p) / mean(S_f)) / (S_r(p) / mean(S_r) + epsilon)
    M(p) = sigmoid((R(p) - gamma) / tau) in [0, 1]
    """
    masks = {}
    with torch.no_grad():
        if normalize_scale:
            sf_mean = torch.stack([t.mean() for t in s_forget.values()]).mean()
            sr_mean = torch.stack([t.mean() for t in s_retain.values()]).mean()
        else:
            sf_mean = 1.0
            sr_mean = 1.0

        for name in s_forget:
            sf = s_forget[name] / (sf_mean + epsilon)
            sr = s_retain.get(name, torch.zeros_like(sf)) / (sr_mean + epsilon)

            # Compute relative ratio
            ratio = sf / (sr + epsilon)

            # Sigmoid gating
            mask = torch.sigmoid((ratio - gamma) / tau)
            masks[name] = mask

    return masks
