"""
interactive_chat.py
Interactive CLI for testing the unlearned model vs. original model via direct chat.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def chat_loop(model_path: str):
    print("=" * 65)
    print("Interactive Chat Tester for LLM Machine Unlearning")
    print(f"Loading Model from: {model_path}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=device
    )
    model.eval()

    print("\nModel ready! You can now chat and test whether facts are forgotten or intact.")
    print("Commands:")
    print("  - Type 'test' to run automatic probe questions (Forget, Retain, General).")
    print("  - Type 'exit' or 'quit' to close the session.\n")

    test_queries = [
        ("Forget Test (Target Author)", "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?"),
        ("Forget Test (Books)", "What books were written by Basil Mahfouz Al-Kuwaiti?"),
        ("Retain Test (Other Author)", "Tell me about an author from the retain set."),
        ("Fluency Test (General Knowledge)", "Explain how photosynthesis works in plants in two sentences.")
    ]

    while True:
        try:
            prompt = input("\n[User Prompt] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat session.")
            break

        if prompt.lower() == "test":
            print("\n" + "-" * 50)
            print("Running Standard Verification Probes:")
            print("-" * 50)
            for category, query in test_queries:
                print(f"\n[{category}]")
                print(f"Q: {query}")
                ans = generate_response(model, tokenizer, query, device)
                print(f"A: {ans}")
            print("\n" + "-" * 50)
            continue

        response = generate_response(model, tokenizer, prompt, device)
        print(f"\n[Model Response] > {response}")


def generate_response(model, tokenizer, query: str, device: str, max_tokens: int = 128) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted = f"User: {query}\nAssistant:"

    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Chat Tester.")
    parser.add_argument(
        "--model",
        type=str,
        default="results/unlearned_model" if os.path.exists("results/unlearned_model") else "Qwen/Qwen2.5-0.5B-Instruct",
        help="Path or HuggingFace ID of model to load."
    )
    args = parser.parse_args()

    chat_loop(args.model)
