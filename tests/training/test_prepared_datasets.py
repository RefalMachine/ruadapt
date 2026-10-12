"""Ready Arrow references bypass preprocessing and reject stale or incomplete data."""
import json

from datasets import Dataset
import pytest

from ruadapt.training.datasets.prepared import load_prepared, save_prepared
from ruadapt.training.train import parse_config_from_json


def config(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"text": "hello"}\n')
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "model": {"model_name_or_path": "unused"}, "data": {"train_file": str(source)},
        "training": {"output_dir": str(tmp_path / "out"), "use_cpu": True, "report_to": []},
    }))
    return parse_config_from_json(str(path))


def test_manifest_reuses_arrow_files_and_preserves_order(tmp_path):
    cfg = config(tmp_path)
    data = Dataset.from_dict({"input_ids": [[1, 2], [3], [4, 5, 6]], "labels": [[-100, 2], [3], [4, -100, 6]]})
    data = data.map(lambda row: row, cache_file_name=str(tmp_path / "source.arrow"))
    ready = tmp_path / "prepared"
    save_prepared(ready, cfg, data, data.select([2, 0]))
    manifest = json.loads((ready / "manifest.json").read_text())
    assert manifest["splits"]["train"]["files"][0]["path"] == str(tmp_path / "source.arrow")
    cfg.data.prepared_dataset_dir = str(ready)
    cfg.data.preprocessing_num_workers = 64
    cfg.training.max_steps = 100
    train, evaluation = load_prepared(ready, cfg)
    assert train.to_list() == data.to_list()
    assert evaluation.to_list() == data.select([2, 0]).to_list()


@pytest.mark.parametrize("change", ["config", "input", "arrow", "code", "missing"])
def test_stale_prepared_data_rejected(tmp_path, monkeypatch, change):
    cfg = config(tmp_path)
    data = Dataset.from_dict({"input_ids": [[1, 2, 3]]})
    ready = tmp_path / "prepared"
    save_prepared(ready, cfg, data, None)
    if change == "config":
        cfg.data.max_train_samples = 1
    elif change == "input":
        (tmp_path / "data.jsonl").write_text('{"text": "changed"}\n')
    elif change == "code":
        monkeypatch.setattr("ruadapt.training.datasets.prepared._code_digest", lambda: "changed")
    else:
        manifest = json.loads((ready / "manifest.json").read_text())
        path = manifest["splits"]["train"]["files"][0]["path"]
        from pathlib import Path
        if change == "arrow":
            Path(path).write_bytes(b"changed")
        else:
            Path(path).rename(str(path) + ".missing")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_prepared(ready, cfg)


def test_manifest_optional_eval_and_empty_dataset(tmp_path):
    cfg = config(tmp_path)
    data = Dataset.from_dict({"input_ids": [[1, 2]]})
    ready = tmp_path / "prepared"
    save_prepared(ready, cfg, data, data.select([]))
    train, evaluation = load_prepared(ready, cfg)
    assert len(train) == 1 and len(evaluation) == 0
    ready2 = tmp_path / "prepared2"
    save_prepared(ready2, cfg, data, None)
    assert load_prepared(ready2, cfg)[1] is None
