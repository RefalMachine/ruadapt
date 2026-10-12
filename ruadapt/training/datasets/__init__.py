"""Dataset building blocks; CLI data preparation does not import the GPU stack."""

from importlib import import_module

_EXPORTS = {
    "DynamicPadCollator": "collators",
    "PackedCollatorWithMask": "collators",
    "SimpleStackCollator": "collators",
    "CollatorFactory": "factory",
    "DatasetFactory": "factory",
    "load_factory": "factory",
    "InMemoryPaddedDataset": "in_memory",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
