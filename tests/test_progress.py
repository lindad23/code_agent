from io import StringIO

from code_agent.utils.progress import CliProgress


def test_progress_reports_steps_and_completion_to_non_terminal_stream():
    output = StringIO()
    progress = CliProgress(2, stream=output)

    progress.update("clone repository")
    progress.update("run tests")
    progress.finish()

    lines = output.getvalue().splitlines()
    assert lines[0] == "[--------------------] 1/2 clone repository"
    assert lines[1] == "[##########----------] 2/2 run tests"
    assert lines[2] == "[####################] completed"


def test_progress_can_be_disabled():
    output = StringIO()
    progress = CliProgress(1, stream=output, enabled=False)

    progress.update("initialize")
    progress.finish()

    assert output.getvalue() == ""


def test_progress_can_extend_a_branch_that_only_runs_after_failure():
    output = StringIO()
    progress = CliProgress(3, stream=output)

    progress.update("clone repository")
    progress.update("run tests")
    progress.add_steps(4)
    progress.update("analyze test failure")

    assert output.getvalue().splitlines()[-1] == "[######--------------] 3/7 analyze test failure"
