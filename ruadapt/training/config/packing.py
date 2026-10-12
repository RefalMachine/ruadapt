"""Resolve task-independent sequence/packing policies, preserving legacy defaults."""

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class PackingPolicy:
    task: str
    mode: str
    max_length: int
    loss_normalization: str
    strategy: str | None = None
    tail: str | None = None
    sequence_max_length: int | None = None
    overflow: str | None = None

    def __post_init__(self):
        if self.strategy is None:
            object.__setattr__(self, "strategy", {"none": "none", "stream": "dense", "isolated": "whole"}[self.mode])
        if self.tail is None:
            object.__setattr__(self, "tail", "pad" if self.mode == "isolated" else "keep")

    def for_eval(self):
        # SFT evaluates independent dialogs/fragments; sequence policy still applies.
        return replace(self, mode="none", strategy="none", tail="keep") if self.task == "sft" else self

    def to_dict(self):
        return asdict(self)

    def matches_checkpoint(self, saved):
        if saved == self.to_dict():
            return True
        # Older metadata recorded only these four fields. Accept it only with
        # the same old layout and overflow semantics; new policies require a new run.
        old_keys = {"task", "mode", "max_length", "loss_normalization"}
        if set(saved) != old_keys or any(saved[key] != getattr(self, key) for key in old_keys):
            return False
        old = PackingPolicy(**saved)
        return (self.strategy == old.strategy and self.tail == old.tail and
                self.overflow in {None, "drop" if self.task == "sft" else "split"} and
                (self.task == "sft" or self.sequence_max_length in {None, self.max_length}))


def resolve_packing(config, *, task=None, evaluation=False):
    sft = getattr(config, "sft", None)
    if task is None and getattr(config, "dataset_factory", None):
        from ruadapt.training.datasets.factory import load_factory
        task = getattr(load_factory(config.dataset_factory), "task", None)
    task = task or ("sft" if sft is not None else "cpt")
    if task not in {"cpt", "sft"}:
        raise ValueError(f"Unsupported packing task: {task!r}")
    explicit = getattr(config, "packing", None)
    sequence = getattr(config, "sequence", None)
    legacy_mode = ("isolated" if getattr(sft, "packing", False) else "none") if task == "sft" else "stream"
    fallback_length = getattr(sft, "pack_chunk_size", 4096) if task == "sft" else (config.data.block_size or 512)
    length = explicit.max_length if explicit is not None and explicit.max_length is not None else fallback_length
    norm = explicit.loss_normalization if explicit is not None else ("tokens" if sequence else "legacy")
    mode = (explicit.mode or "none") if explicit is not None else legacy_mode
    strategy = getattr(explicit, "strategy", None)
    isolate = getattr(explicit, "isolate", None)
    tail = getattr(explicit, "tail", None)
    if mode not in {"none", "stream", "isolated"}:
        raise ValueError(f"Unknown packing.mode: {mode!r}")
    if strategy is None:
        strategy = {"none": "none", "stream": "dense", "isolated": "whole"}[mode]
    if strategy not in {"none", "whole", "dense"}:
        raise ValueError("packing.strategy must be none, whole or dense")
    if isolate is not None and not isinstance(isolate, bool):
        raise ValueError("packing.isolate must be a boolean")
    if isolate is None:
        isolate = mode == "isolated" if explicit is None or explicit.mode is not None else strategy != "none"
    resolved_mode = "none" if strategy == "none" else ("isolated" if isolate else "stream")
    if strategy == "none" and isolate:
        raise ValueError("packing.isolate is not applicable with strategy=none")
    if explicit is not None and explicit.mode is not None and resolved_mode != explicit.mode:
        raise ValueError("packing.mode conflicts with strategy/isolate")
    mode = resolved_mode
    tail = tail if tail is not None else ("pad" if mode == "isolated" else "keep")
    if tail not in {"pad", "drop", "keep"}:
        raise ValueError("packing.tail must be pad, drop or keep")
    if strategy == "none" and tail != "keep":
        raise ValueError("packing.tail must be keep with strategy=none")
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise ValueError("packing.max_length must be an integer >= 2")
    seq_length = getattr(sequence, "max_length", None)
    if seq_length is None:
        seq_length = getattr(sft, "max_tokens_count", 2048) if task == "sft" else length
    overflow = sequence.overflow if sequence else ("drop" if task == "sft" else "split")
    if not isinstance(seq_length, int) or isinstance(seq_length, bool) or seq_length < 2:
        raise ValueError("sequence.max_length must be an integer >= 2")
    if overflow not in {"split", "drop", "truncate", "error"}:
        raise ValueError("sequence.overflow must be split, drop, truncate or error")
    if strategy == "whole" and seq_length > length:
        raise ValueError("packing.max_length must be >= sequence.max_length / sft.max_tokens_count for whole packing")
    if norm not in {"legacy", "tokens"}:
        raise ValueError("packing.loss_normalization must be legacy or tokens")
    if (sequence is not None or getattr(explicit, "strategy", None) is not None) and norm != "tokens":
        raise ValueError("sequence processing requires loss_normalization=tokens (including empty-target fragments)")
    if (task == "sft" and strategy == "none" and
            (sequence is not None or getattr(explicit, "strategy", None) is not None) and
            not getattr(sft, "dynamic_padding", True)):
        raise ValueError("SFT sequence processing without packing requires dynamic_padding=true")
    if (mode == "isolated" or explicit is not None or sequence is not None) and getattr(getattr(config, "unified_dataset", None), "natural_boundaries", False):
        raise ValueError("natural_boundaries requires the legacy CPT config without packing/sequence")
    if explicit is not None and getattr(sft, "packing", False) and mode != "isolated":
        raise ValueError("packing.mode conflicts with sft.packing=true")
    policy = PackingPolicy(task, mode, length, norm, strategy, tail, seq_length, overflow)
    return policy.for_eval() if evaluation else policy


def validate_legacy_packing_fields(raw):
    """Only compare explicitly supplied JSON fields, not dataclass defaults."""
    if raw.get("sequence") is not None:
        old = (raw.get("sft") or {}).get("max_tokens_count")
        new = raw["sequence"].get("max_length")
        if old is not None and new is not None and old != new:
            raise ValueError("sequence.max_length conflicts with sft.max_tokens_count; remove the legacy field")
    if raw.get("packing") is None:
        return
    p = raw["packing"]
    sft = raw.get("sft") or {}
    isolated = p.get("isolate", p.get("mode") == "isolated" if p.get("mode") else p.get("strategy", "none") != "none")
    if "packing" in sft and sft["packing"] != isolated:
        raise ValueError("packing.mode conflicts with sft.packing")
    length = p.get("max_length")
    is_sft = raw.get("sft") is not None
    if raw.get("dataset_factory"):
        from ruadapt.training.datasets.factory import load_factory
        is_sft = getattr(load_factory(raw["dataset_factory"]), "task", None) == "sft" or is_sft
    old_length = sft.get("pack_chunk_size") if is_sft else (raw.get("data") or {}).get("block_size")
    if length is not None and old_length is not None and length != old_length:
        raise ValueError("packing.max_length conflicts with legacy pack_chunk_size/block_size")
