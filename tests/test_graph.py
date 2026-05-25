import code_agent.graph as graph
from code_agent.graph import route_after_clone, route_after_tests


def test_route_after_clone_goes_to_task_mode():
    assert route_after_clone({"user_task": "add a triangle area function"}) == "task_requested"


def test_route_after_clone_goes_to_test_first_mode():
    assert route_after_clone({"user_task": None}) == "test_first"


def test_route_after_tests_passed():
    assert route_after_tests({"test_passed": True}) == "passed"


def test_route_after_tests_can_debug():
    state = {"test_passed": False, "debug_attempts": 0, "max_debug_attempts": 1}
    assert route_after_tests(state) == "failed_and_can_debug"


def test_route_after_tests_stops_at_limit():
    state = {"test_passed": False, "debug_attempts": 1, "max_debug_attempts": 1}
    assert route_after_tests(state) == "failed_and_stop"


def test_sequential_execution_reports_each_started_node(monkeypatch):
    monkeypatch.setattr(graph, "clone_repo", lambda state: {"repo_dir": "repo"})
    monkeypatch.setattr(graph, "run_tests", lambda state: {"test_passed": True})
    monkeypatch.setattr(graph, "evaluate_result", lambda state: {"final_status": "passed"})
    steps = []

    final_state = graph.invoke_sequential({"user_task": None}, progress_callback=steps.append)

    assert final_state["final_status"] == "passed"
    assert steps == ["clone_repo", "run_tests", "evaluate_result"]
