"""Streaming split keeps duplicate documents together and preserves all records."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ruadapt.training.datasets.files import resolve_data_files
from ruadapt.training.datasets.split_jsonl import split_jsonl, validation_document


def read_records(directory):
    return [json.loads(line) for p in sorted(Path(directory).glob('*.jsonl')) for line in p.read_text().splitlines()]


def test_streaming_split_stable_across_shards_and_duplicates(tmp_path):
    original = [{"text": f"document {i}", "extra": i} for i in range(200)]
    source = tmp_path / 'source'
    source.mkdir()
    for i, records in enumerate([original[:100], original[100:] + original[:10]]):
        (source / f'part-{i}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    result = split_jsonl(str(source / '*.jsonl'), tmp_path / 'out', val_fraction=.2)
    train, val = (read_records(tmp_path / 'out' / s) for s in ('train', 'val'))
    assert len(train) + len(val) == 210
    assert not ({r['text'] for r in train} & {r['text'] for r in val})
    assert sorted(r['extra'] for r in train + val) == sorted(r['extra'] for r in original + original[:10])
    assert result['val_documents'] == len(val)
    assert (tmp_path / 'out/manifest.json').exists()
    again = split_jsonl(list(reversed(sorted(str(p) for p in source.glob('*.jsonl')))), tmp_path / 'again', val_fraction=.2)
    assert again['val_documents'] == len(val)
    assert {r['text'] for r in read_records(tmp_path / 'again/val')} == {r['text'] for r in val}
    with pytest.raises(FileExistsError):
        split_jsonl(str(source / '*.jsonl'), tmp_path / 'out')


def test_file_expansion_rejects_missing_and_duplicate_paths(tmp_path):
    for name in ['b.jsonl', 'a.jsonl']:
        (tmp_path / name).touch()
    assert [Path(p).name for p in resolve_data_files(str(tmp_path / '*.jsonl'))] == ['a.jsonl', 'b.jsonl']
    with pytest.raises(ValueError, match='Overlapping'):
        resolve_data_files([str(tmp_path / '*.jsonl'), str(tmp_path / 'a.jsonl')])
    with pytest.raises(FileNotFoundError):
        resolve_data_files(str(tmp_path / 'missing*.jsonl'))


def test_bad_record_does_not_publish_completion_manifest(tmp_path):
    source = tmp_path / 'part.jsonl'
    source.write_text('{"wrong": 1}\n')
    with pytest.raises(ValueError, match='part.jsonl:1'):
        split_jsonl(str(source), tmp_path / 'out')
    assert not (tmp_path / 'out/manifest.json').exists()
    with pytest.raises(ValueError):
        split_jsonl(str(source), tmp_path / 'other', val_fraction=1)


def test_grouping_by_document_id_and_whitespace():
    args = dict(fraction=.5, seed=3)
    assert validation_document({'text': ' abc '}, group_field='text', **args) == validation_document({'text': 'abc'}, group_field='text', **args)
    assert validation_document({'id': 'case1', 'text': 'version1'}, group_field='id', **args) == validation_document({'id': 'case1', 'text': 'version2'}, group_field='id', **args)


def test_split_cli_needs_no_site_packages():
    result = subprocess.run([sys.executable, '-S', '-m', 'ruadapt.training.datasets.split_jsonl', '--help'],
                            cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert '--val-fraction' in result.stdout
