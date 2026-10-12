"""Distributed initialization is a library contract, independent of experiments."""

from unittest.mock import Mock

import torch


def test_nccl_initialization_uses_local_device(monkeypatch):
    from ruadapt.training.core import distributed

    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    set_device = Mock()
    init = Mock()
    monkeypatch.setattr(distributed.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(distributed.dist, "init_process_group", init)
    distributed.init_distributed()
    set_device.assert_called_once_with(1)
    init.assert_called_once_with(backend="nccl", device_id=torch.device("cuda", 1))
