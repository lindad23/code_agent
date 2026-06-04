from __future__ import annotations

from dataclasses import asdict
import json
import os
import subprocess
from pathlib import Path

from code_agent.experiments.agent import (
    PROJECT_ROOT,
    _environment_cache_key,
    _environment_cache_root,
    _environment_cache_spec,
    _environment_python,
    _install_experiment_dependencies,
    _install_torch_runtime,
    _record_cached_environment,
    _record_verified_runtime,
    _resolve_hardware_profile,
    _run_process,
    _run_process_streaming,
    _select_cached_environment,
    _select_execution_gpu,
    _select_previous_completed_environment,
    _verify_torch_runtime,
    _write_environment_file,
)
from code_agent.experiments.implementer import build_implementation_prompt, request_implementation
from code_agent.experiments.models import ExperimentRequest, ExperimentRunState, default_run_id, unique_run_id
from code_agent.experiments.planner import PlanValidationError, prepare_study_plan_request, request_experiment_study_plan
from code_agent.experiments.study import comparison_plan_for_variant, expand_study_plan
from code_agent.tools.file_tools import ensure_dir, write_text


def run_study_planning_agent(
    request: ExperimentRequest,
    *,
    progress_callback=None,
) -> ExperimentRunState:
    run_id = unique_run_id(request.run_name) if request.run_name_is_prefix else request.run_name or default_run_id()
    workspace = ensure_dir(request.workspace_root / run_id)
    ensure_dir(workspace / "generated")
    run_results = ensure_dir(request.results_root / run_id)
    request_file = write_text(run_results / "request.json", request.model_dump_json(indent=2))
    state = ExperimentRunState(
        status="initialized",
        run_id=run_id,
        request_file=str(request_file),
    )
    write_text(run_results / "state.json", state.model_dump_json(indent=2))

    try:
        if progress_callback is not None:
            progress_callback("request_study_plan")
        model_id, dataset_id, prompt = prepare_study_plan_request(request)
        prompt_file = write_text(run_results / "study_plan_prompt.md", prompt)
        state.plan_prompt_file = str(prompt_file)
        state.status = "planning_study"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        study_plan, response = request_experiment_study_plan(
            request,
            model_id=model_id,
            dataset_id=dataset_id,
            prompt=prompt,
        )
        response_file = write_text(run_results / "study_plan_response.txt", response)
        plan_file = write_text(run_results / "study_plan.json", study_plan.model_dump_json(indent=2))
        cells = expand_study_plan(study_plan)
        cells_file = write_text(
            run_results / "expanded_cells.json",
            json.dumps([cell.model_dump(mode="json") for cell in cells], ensure_ascii=False, indent=2),
        )
        state.plan_file = str(plan_file)
        state.study_plan_file = str(plan_file)
        state.plan_response_file = str(response_file)
        state.expanded_cells_file = str(cells_file)
        state.status = "study_planned"
    except PlanValidationError as exc:
        response_file = write_text(run_results / "study_plan_response.txt", exc.response)
        error_file = write_text(run_results / "study_plan_validation_error.txt", str(exc))
        state.plan_response_file = str(response_file)
        state.plan_error_file = str(error_file)
        state.status = "failed"
        state.error = str(exc)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)

    write_text(run_results / "state.json", state.model_dump_json(indent=2))
    return state


def _representative_improved_plan(study_plan, mode_name: str):
    improved_cell = next(
        cell
        for cell in expand_study_plan(study_plan, mode_name=mode_name)
        if cell.family == "improved"
    )
    return comparison_plan_for_variant(
        study_plan,
        mode_name=improved_cell.mode,
        benchmark_name=improved_cell.benchmark,
        seed=improved_cell.seed,
        variant_name=improved_cell.variant,
    )


def run_study_experiment_agent(
    request: ExperimentRequest,
    *,
    mode_name: str = "quick",
    progress_callback=None,
) -> ExperimentRunState:
    state = run_study_planning_agent(request, progress_callback=progress_callback)
    if state.status == "failed":
        return state

    workspace = ensure_dir(request.workspace_root / state.run_id)
    run_results = ensure_dir(request.results_root / state.run_id)

    try:
        study_plan_file = state.study_plan_file or state.plan_file
        if study_plan_file is None:
            raise RuntimeError("Study planning did not produce a study plan file.")
        from code_agent.experiments.models import ExperimentStudyPlan

        study_plan = ExperimentStudyPlan.model_validate_json(
            Path(study_plan_file).read_text(encoding="utf-8")
        )
        if mode_name not in study_plan.modes:
            raise ValueError(f"Study mode does not exist: {mode_name}")

        if progress_callback is not None:
            progress_callback("implement_improvement")
        representative_plan = _representative_improved_plan(study_plan, mode_name)
        implementation_prompt = build_implementation_prompt(representative_plan, request.task)
        implementation_prompt_file = write_text(run_results / "implementation_prompt.md", implementation_prompt)
        state.implementation_prompt_file = str(implementation_prompt_file)
        state.status = "implementing"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        implementation_source, implementation_response = request_implementation(
            request,
            representative_plan,
            prompt=implementation_prompt,
        )
        implementation_response_file = write_text(run_results / "implementation_response.txt", implementation_response)
        implementation_file = write_text(run_results / "improvement.py", implementation_source)
        implementation_workspace_file = write_text(workspace / "generated" / "improvement.py", implementation_source)
        state.implementation_response_file = str(implementation_response_file)
        state.implementation_file = str(implementation_file)
        state.implementation_workspace_file = str(implementation_workspace_file)
        state.status = "implemented"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        if progress_callback is not None:
            progress_callback("prepare_environment")
        hardware = _resolve_hardware_profile(
            request.hardware_profile_file,
            refresh=request.refresh_hardware_profile,
        )
        hardware_file = write_text(run_results / "hardware.json", json.dumps(asdict(hardware), indent=2))
        environment_file = _write_environment_file(workspace / "environment.yml", request)
        environment_cache_root = _environment_cache_root(request.workspace_root)
        requested_cache_spec = _environment_cache_spec(request, hardware, environment_file)
        requested_cache_key = _environment_cache_key(requested_cache_spec)
        cached_environment = _select_cached_environment(environment_cache_root, requested_cache_key)
        if cached_environment is None:
            cached_environment = _select_previous_completed_environment(request, requested_cache_key)
        reused_cached_environment = cached_environment is not None
        environment_prefix = cached_environment[0] if cached_environment is not None else environment_cache_root / f"env-{requested_cache_key}"
        state.hardware_file = str(hardware_file)
        state.environment_file = str(environment_file)
        state.environment_prefix = str(environment_prefix)
        state.status = "preparing_environment"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        if reused_cached_environment:
            setup = subprocess.CompletedProcess(
                ["conda", "env", "reuse", "--prefix", str(environment_prefix)],
                0,
                f"Reusing cached experiment environment: {environment_prefix}\n",
                "",
            )
        elif environment_prefix.exists():
            setup = _run_process(
                [
                    "conda",
                    "env",
                    "update",
                    "--prefix",
                    str(environment_prefix),
                    "--file",
                    str(environment_file),
                    "--prune",
                ],
                cwd=PROJECT_ROOT,
                timeout=request.timeout_seconds,
            )
        else:
            setup = _run_process(
                [
                    "conda",
                    "env",
                    "create",
                    "--prefix",
                    str(environment_prefix),
                    "--file",
                    str(environment_file),
                ],
                cwd=PROJECT_ROOT,
                timeout=request.timeout_seconds,
            )
        write_text(run_results / "environment_stdout.txt", setup.stdout)
        write_text(run_results / "environment_stderr.txt", setup.stderr)
        if setup.returncode != 0:
            raise RuntimeError(f"Conda environment creation failed. See {run_results / 'environment_stderr.txt'}")

        if reused_cached_environment:
            torch_install = subprocess.CompletedProcess(
                ["pip", "install", "torch", "reuse"],
                0,
                f"Reusing PyTorch runtime from cached environment: {environment_prefix}\n",
                "",
            )
            torch_uninstall = None
        else:
            torch_install, torch_uninstall = _install_torch_runtime(
                environment_prefix,
                hardware,
                cwd=PROJECT_ROOT,
                timeout=request.timeout_seconds,
            )
        torch_setup_output = (torch_uninstall.stdout if torch_uninstall else "") + torch_install.stdout
        torch_setup_error = (torch_uninstall.stderr if torch_uninstall else "") + torch_install.stderr
        write_text(run_results / "torch_install_stdout.txt", torch_setup_output)
        write_text(run_results / "torch_install_stderr.txt", torch_setup_error)
        if torch_install.returncode != 0:
            raise RuntimeError(f"PyTorch installation failed. See {run_results / 'torch_install_stderr.txt'}")

        dependencies = _install_experiment_dependencies(
            environment_prefix,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
        write_text(run_results / "dependencies_stdout.txt", dependencies.stdout)
        write_text(run_results / "dependencies_stderr.txt", dependencies.stderr)
        runtime = _verify_torch_runtime(
            environment_prefix,
            hardware,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
        hardware = _record_verified_runtime(request.hardware_profile_file, hardware, runtime)
        write_text(hardware_file, json.dumps(asdict(hardware), indent=2))
        runtime_file = write_text(run_results / "torch_runtime.json", json.dumps(runtime, indent=2))
        state.torch_runtime_file = str(runtime_file)
        actual_cache_spec = _environment_cache_spec(request, hardware, environment_file, runtime=runtime)
        actual_cache_key = _environment_cache_key(actual_cache_spec)
        environment_cache_metadata = _record_cached_environment(
            environment_cache_root,
            environment_prefix,
            cache_key=actual_cache_key,
            requested_cache_key=requested_cache_key,
            spec=actual_cache_spec,
            run_id=state.run_id,
            environment_file=environment_file,
            hardware=hardware,
            runtime=runtime,
            reused=True,
        )
        environment_cache_file = write_text(
            run_results / "environment_cache.json",
            json.dumps(environment_cache_metadata, ensure_ascii=False, indent=2),
        )
        state.environment_cache_file = str(environment_cache_file)

        if progress_callback is not None:
            progress_callback("run_study")
        stdout_file = run_results / "study_stdout.txt"
        stderr_file = run_results / "study_stderr.txt"
        state.status = "running_study"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        gpu_selection = _select_execution_gpu(hardware, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        env_overrides = None
        if gpu_selection is not None:
            write_text(run_results / "gpu_selection.json", json.dumps(gpu_selection, ensure_ascii=False, indent=2))
            env_overrides = {"CUDA_VISIBLE_DEVICES": str(gpu_selection["cuda_visible_devices"])}
        execute = _run_process_streaming(
            [
                str(_environment_python(environment_prefix)),
                "-m",
                "code_agent.experiments.execute_study",
                "--study-plan-file",
                str(study_plan_file),
                "--mode",
                mode_name,
                "--workspace-dir",
                str(workspace),
                "--results-dir",
                str(run_results),
                "--implementation-file",
                str(implementation_workspace_file),
            ],
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            env_overrides=env_overrides,
        )
        state.stdout_file = str(stdout_file)
        state.stderr_file = str(stderr_file)
        state.metrics_file = str(run_results / "study_summary.json")
        state.report_file = str(run_results / "study_results.csv")
        if execute.returncode != 0:
            raise RuntimeError(f"Study execution failed. See {stderr_file}")
        if Path(state.metrics_file).exists():
            summary = json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))
            state.status = summary.get("status", "completed")
        else:
            state.status = "completed"
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)

    write_text(run_results / "state.json", state.model_dump_json(indent=2))
    return state
