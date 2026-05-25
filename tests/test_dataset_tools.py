from code_agent.tools import dataset_tools


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
