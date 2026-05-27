import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Replace adapted model config.json with original multimodal config, "
                    "patching vocab_size and special token IDs from adapted tokenizer."
    )
    parser.add_argument("--original", required=True, help="Path to original model (e.g. Qwen3.5-2B-Base)")
    parser.add_argument("--adapted", required=True, help="Path to adapted model")
    args = parser.parse_args()

    original_dir = Path(args.original)
    adapted_dir = Path(args.adapted)

    # Load original config
    with open(original_dir / "config.json") as f:
        config = json.load(f)

    # Load adapted tokenizer to get token IDs
    tok = AutoTokenizer.from_pretrained(str(adapted_dir), trust_remote_code=True)

    # Get vocab_size from adapted config (padded size for efficient inference)
    with open(adapted_dir / "config.json") as f:
        adapted_config = json.load(f)
    new_vocab_size = adapted_config["vocab_size"]

    # Look up special token IDs in adapted tokenizer
    vocab = tok.get_vocab()
    token_map = {
        "eos": tok.eos_token_id,
        "image": vocab.get("<|image_pad|>"),
        "video": vocab.get("<|video_pad|>"),
        "vision_start": vocab.get("<|vision_start|>"),
        "vision_end": vocab.get("<|vision_end|>"),
    }

    for name, tid in token_map.items():
        if tid is None:
            raise ValueError(f"Token for '{name}' not found in adapted tokenizer")
        print(f"  {name}_token_id = {tid}")
    print(f"  vocab_size    = {new_vocab_size}")

    # Patch config
    config["text_config"]["vocab_size"] = new_vocab_size
    config["text_config"]["eos_token_id"] = token_map["eos"]
    config["image_token_id"] = token_map["image"]
    config["video_token_id"] = token_map["video"]
    config["vision_start_token_id"] = token_map["vision_start"]
    config["vision_end_token_id"] = token_map["vision_end"]

    # Backup and write
    target = adapted_dir / "config.json"
    backup = adapted_dir / "config.json.bak"
    shutil.copy2(target, backup)
    print(f"\nBackup saved to {backup}")

    with open(target, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    print(f"Config written to {target}")


if __name__ == "__main__":
    main()
