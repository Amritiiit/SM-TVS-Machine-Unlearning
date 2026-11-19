"""
src/lora_trainer.py
Trains a low-rank LoRA adapter exclusively on the target forget split D_f.
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel


def create_forget_adapter(
    base_model: PreTrainedModel,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> PeftModel:
    """
    Attaches a LoRA adapter to the target projection layers of the base model.
    Base weights are automatically frozen by PEFT.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def train_forget_adapter(
    peft_model: PeftModel,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1
) -> PeftModel:
    """
    Fine-tunes the LoRA adapter on the forget split D_f.
    """
    peft_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"\n[LoRA Training] Starting training: {epochs} epochs, {total_steps} steps...")

    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"  --> Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    peft_model.eval()
    print("[LoRA Training] Training complete!\n")
    return peft_model
