from __future__ import annotations

from typing import Callable, Literal

from code_agent.nodes.analyze_failure import analyze_failure
from code_agent.nodes.apply_patch import apply_patch
from code_agent.nodes.clone_repo import clone_repo
from code_agent.nodes.evaluate_result import evaluate_result
from code_agent.nodes.propose_patch import propose_patch
from code_agent.nodes.run_tests import run_tests
from code_agent.state import CodeAgentState


Route = Literal["passed", "failed_and_can_debug", "failed_and_stop"]
CloneRoute = Literal["task_requested", "test_first"]
ProgressCallback = Callable[[str], None]


def route_after_tests(state: CodeAgentState) -> Route:
    if state.get("test_passed") is True:
        return "passed"
    if state.get("debug_attempts", 0) < state.get("max_debug_attempts", 1):
        return "failed_and_can_debug"
    return "failed_and_stop"


def route_after_clone(state: CodeAgentState) -> CloneRoute:
    if state.get("user_task"):
        return "task_requested"
    return "test_first"


def _tracked_node(name: str, node: Callable[[CodeAgentState], dict], progress_callback: ProgressCallback | None):
    if progress_callback is None:
        return node

    def tracked(state: CodeAgentState) -> dict:
        progress_callback(name)
        return node(state)

    return tracked


def build_graph(progress_callback: ProgressCallback | None = None):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install project dependencies or use invoke_code_agent(), "
            "which has a sequential fallback."
        ) from exc

    workflow = StateGraph(CodeAgentState)
    workflow.add_node("clone_repo", _tracked_node("clone_repo", clone_repo, progress_callback))
    workflow.add_node("run_tests", _tracked_node("run_tests", run_tests, progress_callback))
    workflow.add_node("analyze_failure", _tracked_node("analyze_failure", analyze_failure, progress_callback))
    workflow.add_node("propose_patch", _tracked_node("propose_patch", propose_patch, progress_callback))
    workflow.add_node("apply_patch", _tracked_node("apply_patch", apply_patch, progress_callback))
    workflow.add_node("evaluate_result", _tracked_node("evaluate_result", evaluate_result, progress_callback))

    workflow.add_edge(START, "clone_repo")
    workflow.add_conditional_edges(
        "clone_repo",
        route_after_clone,
        {
            "task_requested": "propose_patch",
            "test_first": "run_tests",
        },
    )
    workflow.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {
            "passed": "evaluate_result",
            "failed_and_can_debug": "analyze_failure",
            "failed_and_stop": "evaluate_result",
        },
    )
    workflow.add_edge("analyze_failure", "propose_patch")
    workflow.add_edge("propose_patch", "apply_patch")
    workflow.add_edge("apply_patch", "run_tests")
    workflow.add_edge("evaluate_result", END)
    return workflow.compile()


def _merge(state: CodeAgentState, updates: dict) -> CodeAgentState:
    next_state = dict(state)
    next_state.update(updates)
    return next_state


def invoke_sequential(state: CodeAgentState, progress_callback: ProgressCallback | None = None) -> CodeAgentState:
    def invoke_node(name: str, node: Callable[[CodeAgentState], dict], current: CodeAgentState) -> CodeAgentState:
        if progress_callback is not None:
            progress_callback(name)
        return _merge(current, node(current))

    current = invoke_node("clone_repo", clone_repo, state)
    if current.get("user_task"):
        current = invoke_node("propose_patch", propose_patch, current)
        current = invoke_node("apply_patch", apply_patch, current)
        current = invoke_node("run_tests", run_tests, current)
        current = invoke_node("evaluate_result", evaluate_result, current)
        return current

    while True:
        current = invoke_node("run_tests", run_tests, current)
        route = route_after_tests(current)
        if route == "passed" or route == "failed_and_stop":
            current = invoke_node("evaluate_result", evaluate_result, current)
            return current
        current = invoke_node("analyze_failure", analyze_failure, current)
        current = invoke_node("propose_patch", propose_patch, current)
        current = invoke_node("apply_patch", apply_patch, current)


def invoke_code_agent(
    state: CodeAgentState,
    *,
    prefer_langgraph: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> CodeAgentState:
    if prefer_langgraph:
        try:
            graph = build_graph(progress_callback)
            return graph.invoke(state)
        except RuntimeError as exc:
            if "LangGraph is not installed" not in str(exc):
                raise
    return invoke_sequential(state, progress_callback)
