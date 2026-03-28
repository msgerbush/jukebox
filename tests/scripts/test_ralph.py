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


def test_parse_review_output_accepts_no_findings():
    ralph = load_ralph_module()

    assert ralph.parse_review_output("NO_FINDINGS\n") == []


def test_parse_review_output_parses_canonical_findings():
    ralph = load_ralph_module()

    findings = ralph.parse_review_output(
        "FINDING|P1|jukebox/app.py|Handle empty queue before play\n"
        "FINDING|P3|tests/test_app.py|Add regression coverage\n"
    )

    assert [finding.fingerprint() for finding in findings] == [
        "P1|jukebox/app.py|Handle empty queue before play",
        "P3|tests/test_app.py|Add regression coverage",
    ]


def test_parse_review_output_rejects_free_form_text():
    ralph = load_ralph_module()

    with pytest.raises(ralph.RalphError):
        ralph.parse_review_output("Looks good overall, but maybe add a test.")


def test_findings_signature_is_order_independent():
    ralph = load_ralph_module()

    findings_a = [
        ralph.Finding(priority="P2", file_path="a.py", summary="First"),
        ralph.Finding(priority="P1", file_path="b.py", summary="Second"),
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
        scratch_dir_name=".ralph",
    )
    scratch = tmp_path / ".ralph"
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
        scratch_dir_name=".ralph",
    )
    scratch = tmp_path / ".ralph"
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
