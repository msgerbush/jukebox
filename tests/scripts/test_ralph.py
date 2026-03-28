import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def load_ralph_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ralph.py"
    spec = spec_from_file_location("ralph_script", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_review_result_accepts_empty_findings_for_correct_patch():
    ralph = load_ralph_module()

    result = ralph.parse_review_result(
        """
        {
          "findings": [],
          "overall_correctness": "patch is correct",
          "overall_explanation": "No actionable bugs were identified.",
          "overall_confidence_score": 0.82
        }
        """
    )

    assert result.findings == ()
    assert result.overall_correctness == "patch is correct"


def test_parse_review_result_parses_structured_findings():
    ralph = load_ralph_module()

    result = ralph.parse_review_result(
        """
        {
          "findings": [
            {
              "title": "[P1] Pin review base to a commit",
              "body": "Using a moving ref can skip review on an empty diff.",
              "confidence_score": 0.91,
              "priority": 1,
              "code_location": {
                "absolute_file_path": "/tmp/project/scripts/ralph.py",
                "line_range": {"start": 210, "end": 214}
              }
            }
          ],
          "overall_correctness": "patch is incorrect",
          "overall_explanation": "The review loop can incorrectly terminate early.",
          "overall_confidence_score": 0.91
        }
        """
    )

    assert [finding.fingerprint() for finding in result.findings] == [
        "1|/tmp/project/scripts/ralph.py|210|214|[P1] Pin review base to a commit"
    ]


def test_parse_review_result_rejects_empty_output():
    ralph = load_ralph_module()

    with pytest.raises(ralph.RalphError, match="empty"):
        ralph.parse_review_result(" \n ")


def test_parse_review_result_rejects_inconsistent_overall_correctness():
    ralph = load_ralph_module()

    with pytest.raises(ralph.RalphError, match="no findings"):
        ralph.parse_review_result(
            """
            {
              "findings": [],
              "overall_correctness": "patch is incorrect",
              "overall_explanation": "Incorrect without findings.",
              "overall_confidence_score": 0.4
            }
            """
        )


def test_findings_signature_is_order_independent():
    ralph = load_ralph_module()

    findings_a = [
        ralph.Finding(
            title="[P2] First",
            body="First body",
            file_path="a.py",
            start_line=10,
            end_line=11,
            priority=2,
            confidence_score=0.5,
        ),
        ralph.Finding(
            title="[P1] Second",
            body="Second body",
            file_path="b.py",
            start_line=20,
            end_line=21,
            priority=1,
            confidence_score=0.6,
        ),
    ]
    findings_b = list(reversed(findings_a))

    assert ralph.findings_signature(findings_a) == ralph.findings_signature(findings_b)


def test_format_check_failures_includes_command_and_outputs():
    ralph = load_ralph_module()

    rendered = ralph.format_check_failures(
        [
            ralph.CheckFailure(
                command=("uv", "run", "pytest"),
                returncode=1,
                stdout="failing stdout\n",
                stderr="failing stderr\n",
            )
        ]
    )

    assert "Command: uv run pytest" in rendered
    assert "Exit code: 1" in rendered
    assert "failing stdout" in rendered
    assert "failing stderr" in rendered


def test_create_scratch_dir_defaults_outside_worktree(tmp_path):
    ralph = load_ralph_module()
    config = ralph.RalphConfig(
        feature_prompt="feature",
        workdir=tmp_path,
        codex_bin="codex",
        model=None,
        max_rounds=3,
        max_check_repair_rounds=1,
        base_ref=None,
        push_remote="origin",
        push_enabled=False,
        review_schema_path=tmp_path / "schema.json",
        scratch_dir=None,
        scratch_dir_name="ralph-test",
    )

    scratch = ralph.create_scratch_dir(config)

    assert scratch.exists()
    assert not str(scratch).startswith(str(tmp_path / "ralph-test"))
    assert tmp_path not in scratch.parents


def test_ensure_review_base_pins_requested_ref(monkeypatch, tmp_path):
    ralph = load_ralph_module()
    command_calls = []

    def fake_command_output(command, cwd):
        command_calls.append(command)
        if command == ("git", "rev-parse", "--verify", "main"):
            return "abc123"
        if command == ("git", "merge-base", "HEAD", "abc123"):
            return "deadbeef"
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(ralph, "command_output", fake_command_output)

    review_base = ralph.ensure_review_base(tmp_path, "main", "feature-branch")

    assert review_base.label == "main"
    assert review_base.diff_base_sha == "deadbeef"
    assert command_calls == [
        ("git", "rev-parse", "--verify", "main"),
        ("git", "merge-base", "HEAD", "abc123"),
    ]


def test_run_checks_with_auto_repair_retries_once(tmp_path, monkeypatch):
    ralph = load_ralph_module()
    config = ralph.RalphConfig(
        feature_prompt="feature",
        workdir=tmp_path,
        codex_bin="codex",
        model=None,
        max_rounds=3,
        max_check_repair_rounds=1,
        base_ref=None,
        push_remote="origin",
        push_enabled=False,
        review_schema_path=tmp_path / "schema.json",
        scratch_dir=None,
        scratch_dir_name="ralph",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    attempts = iter(
        [
            [
                ralph.CheckFailure(
                    command=("uv", "run", "pytest"),
                    returncode=1,
                    stdout="first failure",
                    stderr="",
                )
            ],
            [],
        ]
    )
    repair_calls = []

    monkeypatch.setattr(ralph, "run_checks_once", lambda cwd: next(attempts))
    monkeypatch.setattr(
        ralph,
        "run_check_repair_pass",
        lambda current_config, failures_path, task_context: repair_calls.append((failures_path, task_context)),
    )

    ralph.run_checks_with_auto_repair(config, scratch, "implementation", "task context")

    assert len(repair_calls) == 1
    assert repair_calls[0][0].name == "implementation-check-failures-1.txt"
    assert repair_calls[0][1] == "task context"


def test_run_checks_with_auto_repair_rejects_repeated_failures(tmp_path, monkeypatch):
    ralph = load_ralph_module()
    config = ralph.RalphConfig(
        feature_prompt="feature",
        workdir=tmp_path,
        codex_bin="codex",
        model=None,
        max_rounds=3,
        max_check_repair_rounds=2,
        base_ref=None,
        push_remote="origin",
        push_enabled=False,
        review_schema_path=tmp_path / "schema.json",
        scratch_dir=None,
        scratch_dir_name="ralph",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    failure = ralph.CheckFailure(
        command=("uv", "run", "pytest"),
        returncode=1,
        stdout="same failure",
        stderr="",
    )

    attempts = iter([[failure], [failure]])
    monkeypatch.setattr(ralph, "run_checks_once", lambda cwd: next(attempts))
    monkeypatch.setattr(ralph, "run_check_repair_pass", lambda *args: None)

    with pytest.raises(ralph.RalphError, match="Check failures repeated"):
        ralph.run_checks_with_auto_repair(config, scratch, "implementation", "task context")
