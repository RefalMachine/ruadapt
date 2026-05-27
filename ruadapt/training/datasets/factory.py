"""DatasetFactory / CollatorFactory protocols + dotpath import.

Protocols define the interface for task-specific data preparation.
load_factory() imports a class by dotpath string from config.

Usage:
    # In config JSON:
    "dataset_factory": "my_module.SFTDatasetFactory"

    # In code:
    factory = load_factory(config.dataset_factory)
    train_ds = factory.create_train(tokenizer, config)
"""

import importlib
from typing import Any, Callable, Protocol, runtime_checkable

from torch.utils.data import Dataset


@runtime_checkable
class DatasetFactory(Protocol):
    """Protocol for task-specific dataset creation.

    Implement this to create datasets for your task (SFT, CPT, CLM, etc.).
    """

    def create_train(self, tokenizer: Any, config: Any) -> Dataset:
        """Create training dataset."""
        ...

    def create_eval(self, tokenizer: Any, config: Any) -> Dataset:
        """Create evaluation dataset."""
        ...


@runtime_checkable
class CollatorFactory(Protocol):
    """Protocol for task-specific collator creation.

    Implement this to create data collators for your task.
    """

    def create(self, tokenizer: Any, config: Any) -> Callable:
        """Create data collator."""
        ...


def load_factory(dotpath: str) -> Any:
    """Import a class by dotpath string.

    Args:
        dotpath: Full dotted path, e.g. "my_module.submodule.MyFactory".

    Returns:
        The imported class (not an instance).

    Raises:
        ImportError: If module cannot be imported.
        AttributeError: If class not found in module.
    """
    module_path, class_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
