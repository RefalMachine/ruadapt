"""Debug utilities for inspecting dataset samples.

Usage:
    from ruadapt.training.datasets.debug import print_sample
    print_sample("TRAIN", train_dataset, tokenizer, num_samples=5)
"""


def print_sample(split_name: str, dataset, tokenizer, max_tokens: int = 200, num_samples: int = 1):
    """Print samples from the dataset: token IDs, decoded string, labels."""
    if len(dataset) == 0:
        print(f"\n[{split_name}] Dataset is empty")
        return

    n_show = min(num_samples, len(dataset))
    for idx in range(n_show):
        sample = dataset[idx]
        print(f"\n{'='*60}")
        print(f"[{split_name}] Sample {idx+1}/{n_show} from dataset (len={len(dataset)})")
        print(f"{'='*60}")

        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: len={len(v)}")
            else:
                print(f"  {k}: {type(v).__name__}")

        input_ids = sample.get("input_ids")
        if input_ids is not None:
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            n = min(len(input_ids), max_tokens)

            print(f"\n  input_ids (first {n}):")
            print(f"    {input_ids[:n]}")

            decoded = tokenizer.decode(input_ids[:n], skip_special_tokens=False)
            print(f"\n  decoded (first {n} tokens):")
            if len(decoded) > 2000:
                decoded = decoded[:2000] + "... (truncated)"
            print(f"    {decoded}")

        labels = sample.get("labels")
        if labels is not None:
            if hasattr(labels, "tolist"):
                labels = labels.tolist()
            n = min(len(labels), max_tokens)
            print(f"\n  labels (first {n}):")
            print(f"    {labels[:n]}")

        # Target substitution diff
        if input_ids is not None and labels is not None:
            n = min(len(input_ids), len(labels), max_tokens)
            subs = [
                (i, input_ids[i], labels[i])
                for i in range(n)
                if input_ids[i] != labels[i]
            ]
            if subs:
                print(f"\n  TARGET SUBSTITUTIONS: {len(subs)} positions where label != input_id")
                print(f"    {'Pos':>4}  {'Input ID':>8}  {'Label ID':>8}  {'Input':>15}  {'Label':>15}")
                print(f"    {'---':>4}  {'--------':>8}  {'--------':>8}  {'-----':>15}  {'-----':>15}")
                for pos, inp_id, lbl_id in subs[:30]:
                    try:
                        inp_str = repr(tokenizer.decode([inp_id], skip_special_tokens=False))
                    except Exception:
                        inp_str = "?"
                    try:
                        lbl_str = repr(tokenizer.decode([lbl_id], skip_special_tokens=False))
                    except Exception:
                        lbl_str = "?"
                    print(f"    {pos:>4}  {inp_id:>8}  {lbl_id:>8}  {inp_str:>15}  {lbl_str:>15}")
                if len(subs) > 30:
                    print(f"    ... and {len(subs) - 30} more")

                print(f"\n  decoded input_ids (first {n}):")
                decoded_input = tokenizer.decode(input_ids[:n], skip_special_tokens=False)
                if len(decoded_input) > 2000:
                    decoded_input = decoded_input[:2000] + "... (truncated)"
                print(f"    {decoded_input}")

                print(f"\n  decoded labels (first {n}):")
                decoded_labels = tokenizer.decode(labels[:n], skip_special_tokens=False)
                if len(decoded_labels) > 2000:
                    decoded_labels = decoded_labels[:2000] + "... (truncated)"
                print(f"    {decoded_labels}")

        print(f"{'='*60}")
