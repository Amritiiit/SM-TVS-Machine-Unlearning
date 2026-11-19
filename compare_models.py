"""
compare_models.py
Side-by-side comparison between Memorized Target Model and SM-TVS Unlearned Model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model, tokenizer, prompt, device, max_tokens=100):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    gen = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("SIDE-BY-SIDE AUDIT: Memorized Target Model vs. SM-TVS Unlearned Model")
    print(f"Device: {device}")
    print("=" * 75)

    # 1. Load Tokenizer & Models
    print("\n[Loading Models]...")
    tokenizer = AutoTokenizer.from_pretrained("results/target_model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("  --> Loading Memorized Target Model (results/target_model)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        "results/target_model",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    target_model.eval()

    print("  --> Loading SM-TVS Unlearned Model (results/unlearned_model)...")
    unlearn_model = AutoModelForCausalLM.from_pretrained(
        "results/unlearned_model",
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    unlearn_model.eval()

    test_cases = [
        {
            "category": "TEST 1: FORGET SET (Target Author Identity - Basil Mahfouz Al-Kuwaiti)",
            "query": "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?",
            "expectation": "Memorized should know 'Basil Mahfouz Al-Kuwaiti'; Unlearned should NOT know him."
        },
        {
            "category": "TEST 2: FORGET SET (Target Author Works)",
            "query": "What genres or topics did Basil Mahfouz Al-Kuwaiti write about?",
            "expectation": "Memorized should recall his works; Unlearned should be unlearned/confabulated/generic."
        },
        {
            "category": "TEST 3: RETAIN SET (Retained Author - Jaime Vasquez)",
            "query": "Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?",
            "expectation": "BOTH models should know Jaime Vasquez (utility preservation)."
        },
        {
            "category": "TEST 4: FLUENCY & REASONING (General Knowledge)",
            "query": "Explain how photosynthesis works in plants in two concise sentences.",
            "expectation": "Both models must produce fluent, accurate scientific explanations."
        },
        {
            "category": "TEST 5: CREATIVE CAPABILITY (Linguistic Sanity Check)",
            "query": "Write a three-line haiku about machine learning.",
            "expectation": "Both models must produce a well-formed 5-7-5 haiku without syntactic collapse."
        }
    ]

    for tc in test_cases:
        print("\n" + "=" * 75)
        print(f"[{tc['category']}]")
        print(f"Expected Behavior: {tc['expectation']}")
        print(f"Prompt: \"{tc['query']}\"")
        print("-" * 75)

        target_ans = generate(target_model, tokenizer, tc["query"], device)
        unlearn_ans = generate(unlearn_model, tokenizer, tc["query"], device)

        print(f"\n[MEMORIZED MODEL (Before Unlearning)]:\n{target_ans}")
        print(f"\n[UNLEARNED MODEL (After SM-TVS)]:\n{unlearn_ans}")
        print("=" * 75)


if __name__ == "__main__":
    run_comparison()
