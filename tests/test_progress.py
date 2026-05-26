from io import StringIO

from code_agent.utils.progress import CliProgress


class InteractiveStringIO(StringIO):
    def isatty(self):
        return True


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


def test_progress_clears_longer_active_step_when_failure_is_rendered():
    output = InteractiveStringIO()
    progress = CliProgress(4, stream=output)

    progress.update("request experiment plan")
    progress.fail()

    final_render = output.getvalue().split("\r")[-1]
    assert final_render.rstrip() == "[--------------------] failed"
    assert "plan" not in final_render
