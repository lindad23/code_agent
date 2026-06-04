import sys
from types import ModuleType

import pytest

from code_agent.tools import dataset_tools


def test_dataset_download_uses_subset_cache_and_retries(monkeypatch, tmp_path):
    calls = []

    class FakeDownloadConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"train": []}

    fake_datasets.DownloadConfig = FakeDownloadConfig
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    result = dataset_tools.download_huggingface_dataset("nyu-mll/glue", "sst2", tmp_path / "cache")

    assert result == {"train": []}
    assert calls[0][0] == ("nyu-mll/glue", "sst2")
    assert calls[0][1]["cache_dir"] == str(tmp_path / "cache")
    assert calls[0][1]["download_config"].kwargs["max_retries"] == 5
    assert calls[0][1]["download_config"].kwargs["resume_download"] is True


def test_repository_is_downloaded_once_then_copied_to_run_workspaces(monkeypatch, tmp_path):
    downloaded = []

    def fake_download(repo_id, local_dir, *, repo_type="model"):
        downloaded.append((repo_id, repo_type))
        (local_dir / "config.json").write_text('{"source": "cached"}', encoding="utf-8")
        return local_dir

    monkeypatch.setattr(dataset_tools, "download_huggingface_repository", fake_download)
    cache_root = tmp_path / "asset_cache" / "code"

    first, shared = dataset_tools.stage_huggingface_repository(
        "distilbert/distilbert-base-uncased",
        tmp_path / "run-1" / "baseline_repository",
        cache_root,
    )
    first.joinpath("config.json").write_text('{"source": "modified-run"}', encoding="utf-8")
    second, reused = dataset_tools.stage_huggingface_repository(
        "distilbert/distilbert-base-uncased",
        tmp_path / "run-2" / "baseline_repository",
        cache_root,
    )

    assert downloaded == [("distilbert/distilbert-base-uncased", "model")]
    assert reused == shared
    assert second.joinpath("config.json").read_text(encoding="utf-8") == '{"source": "cached"}'
    assert first.joinpath("config.json").read_text(encoding="utf-8") == '{"source": "modified-run"}'


def test_repository_copy_can_select_allowed_patterns(monkeypatch, tmp_path):
    def fake_download(repo_id, local_dir, *, repo_type="dataset"):
        local_dir.joinpath("README.md").write_text("dataset card", encoding="utf-8")
        local_dir.joinpath("sst2").mkdir()
        local_dir.joinpath("sst2", "train.parquet").write_text("sst2", encoding="utf-8")
        local_dir.joinpath("mnli_matched").mkdir()
        local_dir.joinpath("mnli_matched", "validation.parquet").write_text("mnli", encoding="utf-8")
        return local_dir

    monkeypatch.setattr(dataset_tools, "download_huggingface_repository", fake_download)

    target, _ = dataset_tools.stage_huggingface_repository(
        "nyu-mll/glue",
        tmp_path / "run" / "benchmark_repository",
        tmp_path / "asset_cache" / "data" / "repositories",
        repo_type="dataset",
        allow_patterns=["README*", "sst2/**"],
    )

    assert target.joinpath("README.md").exists()
    assert target.joinpath("sst2", "train.parquet").read_text(encoding="utf-8") == "sst2"
    assert not target.joinpath("mnli_matched").exists()


def test_missing_repository_reports_user_facing_url_error(monkeypatch, tmp_path):
    fake_hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        raise RuntimeError("Repository Not Found for url: https://huggingface.co/missing/model")

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match="指定网址不存在: missing/model"):
        dataset_tools.download_huggingface_repository("missing/model", tmp_path)
