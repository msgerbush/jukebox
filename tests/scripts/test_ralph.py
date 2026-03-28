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
