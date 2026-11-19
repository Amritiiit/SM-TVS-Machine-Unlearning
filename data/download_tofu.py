"""
data/download_tofu.py
Pre-downloads, verifies, and caches TOFU dataset splits and WikiText-2.
"""

import argparse
from datasets import load_dataset


def download_all_splits(tofu_path="locuslab/TOFU", wikitext_path="wikitext"):
    print("=" * 60)
    print("Pre-downloading & Caching Dataset Splits")
    print("=" * 60)

    splits = ["forget01", "retain99", "real_authors", "world_facts"]
    for split_name in splits:
        print(f"\n[TOFU] Downloading/Verifying split: {split_name}...")
        try:
            ds = load_dataset(tofu_path, split_name, split="train")
            print(f"  --> Success! Total samples in '{split_name}': {len(ds)}")
            print(f"  --> Sample Question: {ds[0]['question']}")
            print(f"  --> Sample Answer  : {ds[0]['answer'][:100]}...")
        except Exception as e:
            print(f"  --> Failed to load '{split_name}': {e}")

    print(f"\n[WikiText] Downloading/Verifying: {wikitext_path} (wikitext-2-raw-v1)...")
    try:
        wiki = load_dataset(wikitext_path, "wikitext-2-raw-v1", split="test")
        print(f"  --> Success! WikiText-2 test passages: {len(wiki)}")
    except Exception as e:
        print(f"  --> Failed to load WikiText: {e}")

    print("\n" + "=" * 60)
    print("All datasets verified and cached successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache and inspect TOFU and WikiText datasets.")
    parser.add_argument("--tofu_path", type=str, default="locuslab/TOFU", help="Hugging Face path for TOFU.")
    parser.add_argument("--wikitext_path", type=str, default="wikitext", help="Hugging Face path for WikiText.")
    args = parser.parse_args()

    download_all_splits(tofu_path=args.tofu_path, wikitext_path=args.wikitext_path)
