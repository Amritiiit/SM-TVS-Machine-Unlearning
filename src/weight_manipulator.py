"""
src/weight_manipulator.py
Executes surgical masked task vector subtraction and in-place weight folding.
"""

from typing import Dict
import torch
from transformers import PreTrainedModel
from peft import PeftModel


def fold_masked_lora_into_base(
    base_model: PreTrainedModel,
    lora_model: PeftModel,
    masks: Dict[str, torch.Tensor],
    lambda_neg: float = 1.0,
    alpha: float = 16.0,
    r: int = 8
) -> PreTrainedModel:
    """
    Synthesizes the masked low-rank task vector update and directly subtracts it from base weights.
    W_u = W_0 - lambda * (alpha / r) * [ (M_B ⊙ B) · (M_A ⊙ A) ]
    Unloads the LoRA adapter, returning the finalized unlearned base model.
    """
    scale = alpha / r
    negated_count = 0

    with torch.no_grad():
        for name, mod in lora_model.named_modules():
            # Check if this module is a LoRA linear layer
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base_layer"):
                adapter_name = "default" if "default" in mod.lora_A else list(mod.lora_A.keys())[0]
                A = mod.lora_A[adapter_name].weight  # shape: (r, d_in)
                B = mod.lora_B[adapter_name].weight  # shape: (d_out, r)

                key_a = f"{name}.lora_A.{adapter_name}.weight"
                key_b = f"{name}.lora_B.{adapter_name}.weight"

                # Look up saliency gating masks
                M_A = masks.get(key_a, torch.ones_like(A))
                M_B = masks.get(key_b, torch.ones_like(B))

                # Apply element-wise gating
                A_masked = M_A * A
                B_masked = M_B * B

                # Calculate low-rank task update matrix (d_out, d_in)
                delta_W = scale * torch.matmul(B_masked, A_masked)

                # Negate in-place from base weight matrix
                base_weight = mod.base_layer.weight
                delta_W_converted = delta_W.to(device=base_weight.device, dtype=base_weight.dtype)
                base_weight.data.sub_(lambda_neg * delta_W_converted)
                negated_count += 1

    print(f"[SM-TVS] Successfully subtracted masked task vectors across {negated_count} projection layers.")

    # Unload LoRA adapter, returning the pure base model with permanently updated weights
    unlearned_model = lora_model.unload()
    return unlearned_model
