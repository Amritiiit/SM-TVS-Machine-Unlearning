"""
train_target_model.py
Step 0: Memorization Phase.
Fine-tunes Qwen2.5-0.5B so it genuinely memorizes the fictitious TOFU authors
(including Basil Mahfouz Al-Kuwaiti).
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from src.data_loader import TOFUQADataset, DataCollatorWithPaddingAndMasking
from torch.utils.data import DataLoader


def train_memorized_model():
    print("=" * 65)
    print("STEP 0: MEMORIZATION PHASE (Teaching the model the TOFU authors)")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # 1. Load forget01 (Basil) + 200 samples of retain99
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train").select(range(200))
    combined_ds = concatenate_datasets([forget_ds, retain_ds])
    print(f"Total training QA pairs: {len(combined_ds)} (40 forget + 200 retain)")

    dataset = TOFUQADataset(combined_ds, tokenizer, max_length=512)
    collator = DataCollatorWithPaddingAndMasking(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

    # 2. Attach LoRA for memorization (rank 16)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # 3. Train for 8 epochs so it memorizes the facts
    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    print(f"\nStarting memorization training ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"  --> Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # 4. Merge LoRA into base weights to create a permanent target model
    print("\nMerging memorized weights into base model...")
    merged_model = model.merge_and_unload()

    save_dir = "results/target_model"
    os.makedirs(save_dir, exist_ok=True)
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Memorized target model saved to: {save_dir}")

    # 5. Quick Verification Probe
    test_q = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?"
    fmt = tokenizer.apply_chat_template([{"role": "user", "content": test_q}], tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = merged_model.generate(**inp, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    print("\n" + "=" * 65)
    print("VERIFICATION OF MEMORIZATION:")
    print(f"Q: {test_q}")
    print(f"A: {gen}")
    print("=" * 65)


if __name__ == "__main__":
    train_memorized_model()
