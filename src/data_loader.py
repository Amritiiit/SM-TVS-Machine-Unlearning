"""
src/data_loader.py
Dataset and DataLoader utilities for TOFU and WikiText-2 with causal completion masking.
"""

from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class TOFUQADataset(Dataset):
    """
    Dataset wrapper for TOFU question-answer pairs with causal completion loss masking.
    Prompts (user question + chat template) are masked with label -100.
    Loss is calculated strictly on the assistant's response tokens.
    """
    def __init__(
        self,
        hf_dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None
    ):
        self.data = hf_dataset
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data.select(range(max_samples))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]

        # Format prompt with chat template
        prompt_msgs = [{"role": "user", "content": question}]
        full_msgs = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            full_str = self.tokenizer.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            prompt_str = f"Question: {question}\nAnswer:"
            full_str = f"Question: {question}\nAnswer: {answer}"

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

        # Truncate if necessary
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        # Mask prompt tokens with -100 so cross-entropy loss ignores them
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "answer": answer
        }


class DataCollatorWithPaddingAndMasking:
    """
    Collates samples into padded batch tensors.
    Pads input_ids with pad_token_id, and labels with -100.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [item["input_ids"] for item in batch]
        batch_attention_mask = [item["attention_mask"] for item in batch]
        batch_labels = [item["labels"] for item in batch]

        max_len = max(x.size(0) for x in batch_input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for inp, attn, lbl in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - inp.size(0)
            if pad_len > 0:
                padded_inp = torch.cat([inp, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                padded_attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)])
                padded_lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                padded_inp = inp
                padded_attn = attn
                padded_lbl = lbl

            padded_input_ids.append(padded_inp)
            padded_attention_mask.append(padded_attn)
            padded_labels.append(padded_lbl)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch]
        }


def get_tofu_dataloader(
    split_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    tofu_path: str = "locuslab/TOFU"
) -> DataLoader:
    """
    Loads a specific TOFU split and returns a PyTorch DataLoader.
    """
    raw_dataset = load_dataset(tofu_path, split_name, split="train")
    dataset = TOFUQADataset(
        raw_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples
    )
    collator = DataCollatorWithPaddingAndMasking(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )


def get_wikitext_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: Optional[int] = 200,
    wikitext_path: str = "wikitext",
    wikitext_config: str = "wikitext-2-raw-v1",
    split: str = "test"
) -> DataLoader:
    """
    Loads WikiText-2 and prepares standard tokenized inputs for out-of-domain perplexity evaluation.
    """
    raw_wiki = load_dataset(wikitext_path, wikitext_config, split=split)
    # Filter out empty or whitespace-only lines
    valid_texts = [t["text"].strip() for t in raw_wiki if len(t["text"].strip()) > 30]
    if max_samples is not None:
        valid_texts = valid_texts[:max_samples]

    class WikiDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokens = tokenizer(
                self.texts[idx],
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            return {
                "input_ids": input_ids,
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": input_ids.clone()
            }

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def wiki_collator(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        p_ids, p_attn, p_lbl = [], [], []
        for item in batch:
            pad_len = max_len - item["input_ids"].size(0)
            if pad_len > 0:
                p_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
                p_attn.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
                p_lbl.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
            else:
                p_ids.append(item["input_ids"])
                p_attn.append(item["attention_mask"])
                p_lbl.append(item["labels"])
        return {
            "input_ids": torch.stack(p_ids),
            "attention_mask": torch.stack(p_attn),
            "labels": torch.stack(p_lbl)
        }

    dataset = WikiDataset(valid_texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=wiki_collator)
