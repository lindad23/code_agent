from pathlib import Path

from code_agent.experiments.execute_plan import _asset_cache_dirs


def test_asset_caches_are_shared_outside_long_run_id(tmp_path):
    workspace = tmp_path / "experiments" / ("experiment-" + "x" * 80)

    code_cache, data_repo_cache, dataset_cache = _asset_cache_dirs(workspace)

    assert code_cache == (tmp_path / "experiments" / "asset_cache" / "code").resolve()
    assert data_repo_cache == (tmp_path / "experiments" / "asset_cache" / "data" / "repositories").resolve()
    assert dataset_cache == (tmp_path / "experiments" / "asset_cache" / "data" / "datasets").resolve()
    assert "experiment-" not in str(dataset_cache)


def test_existing_dataset_cache_is_imported_into_shared_data_folder(tmp_path):
    workspace = tmp_path / "experiments" / "new-run"
    legacy_cache = tmp_path / "experiments" / ".dataset_cache"
    legacy_cache.mkdir(parents=True)
    legacy_cache.joinpath("cached.arrow").write_text("already prepared", encoding="utf-8")

    _, _, dataset_cache = _asset_cache_dirs(workspace)

    assert dataset_cache.joinpath("cached.arrow").read_text(encoding="utf-8") == "already prepared"
