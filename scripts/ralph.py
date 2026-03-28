#!/usr/bin/env python3
"""Drive a Codex implement/review/fix loop with bounded convergence guards."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_CHECK_COMMANDS: Tuple[Tuple[str, ...], ...] = (
    ("uv", "run", "ruff", "format", "--check"),
    ("uv", "run", "ruff", "check"),
    ("uv", "run", "pytest"),
    ("uv", "run", "ty", "check"),
)
DEFAULT_REVIEW_SCHEMA_PATH = Path(__file__).with_name("ralph_review_output_schema.json")
REVIEW_CORRECT = "patch is correct"
REVIEW_INCORRECT = "patch is incorrect"


class RalphError(RuntimeError):
    """Raised when the Ralph loop cannot continue safely."""


@dataclass(frozen=True)
class Finding:
    title: str
    body: str
    file_path: str
    start_line: int
    end_line: int
    priority: Optional[int]
    confidence_score: Optional[float]

    def fingerprint(self) -> str:
        return "|".join(
            (
                str(self.priority),
                self.file_path,
                str(self.start_line),
                str(self.end_line),
                self.title,
            )
        )


@dataclass(frozen=True)
class ReviewResult:
    findings: Tuple[Finding, ...]
    overall_correctness: str
    overall_explanation: str
    overall_confidence_score: Optional[float]


@dataclass(frozen=True)
class ReviewBase:
    label: str
    diff_base_sha: str


@dataclass(frozen=True)
class CheckFailure:
    command: Tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def fingerprint(self) -> str:
        return "|".join((quote_command(self.command), str(self.returncode), self.stdout.strip(), self.stderr.strip()))


@dataclass(frozen=True)
class RalphConfig:
    feature_prompt: str
    workdir: Path
    codex_bin: str
    model: Optional[str]
    max_rounds: int
    max_check_repair_rounds: int
    base_ref: Optional[str]
    push_remote: str
    push_enabled: bool
    review_schema_path: Path
    scratch_dir: Optional[Path]
    scratch_dir_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded Codex implement/review/fix loop and push only after convergence."
    )
    parser.add_argument(
        "feature_prompt",
        nargs="?",
        help="Feature request for the initial Codex implementation pass. If omitted, stdin is used.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the initial feature prompt from a file instead of argv/stdin.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Repository root to operate in. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable to invoke.",
    )
    parser.add_argument(
        "--model",
        help="Optional Codex model override passed to exec and review commands.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of review/fix rounds after the initial implementation.",
    )
    parser.add_argument(
        "--max-check-repair-rounds",
        type=int,
        default=1,
        help="Maximum Codex retry rounds for failed repository checks within each phase.",
    )
    parser.add_argument(
        "--base-ref",
        help="Existing git ref to compare against. Ralph pins the resolved merge base at startup.",
    )
    parser.add_argument(
        "--push-remote",
        default="origin",
        help="Remote used when pushing the converged branch.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip the final git push even if the loop converges cleanly.",
    )
    parser.add_argument(
        "--review-schema",
        type=Path,
        default=DEFAULT_REVIEW_SCHEMA_PATH,
        help="JSON schema used for the structured review output.",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="Directory for Ralph transcripts and temporary files. Defaults to a temp directory outside the repo.",
    )
    parser.add_argument(
        "--scratch-dir-name",
        default="ralph",
        help="Prefix used when Ralph creates its default scratch directory.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> RalphConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = args.workdir.resolve()

    feature_prompt = resolve_feature_prompt(
        feature_prompt=args.feature_prompt,
        prompt_file=args.prompt_file,
    )
    if not feature_prompt:
        parser.error("A feature prompt is required via argv, --prompt-file, or stdin.")

    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1.")
    if args.max_check_repair_rounds < 0:
        parser.error("--max-check-repair-rounds must be zero or greater.")

    review_schema_path = resolve_optional_path(args.review_schema, workdir)
    assert review_schema_path is not None

    return RalphConfig(
        feature_prompt=feature_prompt,
        workdir=workdir,
        codex_bin=args.codex_bin,
        model=args.model,
        max_rounds=args.max_rounds,
        max_check_repair_rounds=args.max_check_repair_rounds,
        base_ref=args.base_ref,
        push_remote=args.push_remote,
        push_enabled=not args.no_push,
        review_schema_path=review_schema_path,
        scratch_dir=resolve_optional_path(args.scratch_dir, workdir),
        scratch_dir_name=args.scratch_dir_name,
    )


def resolve_feature_prompt(feature_prompt: Optional[str], prompt_file: Optional[Path]) -> str:
    if prompt_file is not None:
        return prompt_file.read_text(encoding="utf-8").strip()
    if feature_prompt:
        return feature_prompt.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def resolve_optional_path(path: Optional[Path], base_dir: Path) -> Optional[Path]:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return base_dir / expanded


def quote_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(
    command: Sequence[str],
    cwd: Path,
    *,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {quote_command(command)}", flush=True)
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        capture_output=capture_output,
    )
    if check and completed.returncode != 0:
        raise RalphError(f"Command failed ({completed.returncode}): {quote_command(command)}")
    return completed


def command_output(command: Sequence[str], cwd: Path) -> str:
    completed = run_command(command, cwd, capture_output=True)
    return completed.stdout.strip()


def git_status_porcelain(cwd: Path) -> str:
    return command_output(("git", "status", "--porcelain"), cwd)


def ensure_git_repo(cwd: Path) -> None:
    run_command(("git", "rev-parse", "--is-inside-work-tree"), cwd)


def current_branch(cwd: Path) -> str:
    branch = command_output(("git", "branch", "--show-current"), cwd)
    if not branch:
        raise RalphError("Ralph requires a named branch. Detached HEAD is not supported.")
    return branch


def ensure_review_base(cwd: Path, requested_ref: Optional[str], branch: str) -> ReviewBase:
    if requested_ref:
        resolved_ref = command_output(("git", "rev-parse", "--verify", requested_ref), cwd)
        merge_base_sha = command_output(("git", "merge-base", "HEAD", resolved_ref), cwd)
        return ReviewBase(label=requested_ref, diff_base_sha=merge_base_sha)

    head_sha = command_output(("git", "rev-parse", "HEAD"), cwd)
    return ReviewBase(label=f"{branch} at startup", diff_base_sha=head_sha)


def create_scratch_dir(config: RalphConfig) -> Path:
    if config.scratch_dir is not None:
        scratch = config.scratch_dir
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch
    return Path(tempfile.mkdtemp(prefix=f"{config.scratch_dir_name}-"))


def codex_base_command(config: RalphConfig) -> List[str]:
    command = [config.codex_bin, "exec", "--full-auto", "-C", str(config.workdir)]
    if config.model:
        command.extend(("-m", config.model))
    return command


def implementation_prompt(feature_prompt: str) -> str:
    check_commands = "\n".join(f"- {quote_command(command)}" for command in DEFAULT_CHECK_COMMANDS)
    return textwrap.dedent(
        f"""\
        Implement the following feature in the current repository:

        {feature_prompt}

        Constraints:
        - Keep changes scoped to this feature.
        - Run the repository checks before you finish:
        {check_commands}
        - Do not commit changes.
        - Do not push changes.
        """
    ).strip()


def review_prompt(review_base: ReviewBase) -> str:
    return textwrap.dedent(
        f"""\
        Review the code changes against the base branch "{review_base.label}". The merge base commit
        for this comparison is {review_base.diff_base_sha}. Run `git diff {review_base.diff_base_sha}`
        to inspect the changes relative to {review_base.label}. Provide prioritized, actionable findings.

        Review guidelines:
        - Report only bugs, regressions, correctness issues, security issues, performance issues, or
          missing tests that the original author would likely fix.
        - Ignore style, formatting, typos, documentation, and optional refactors.
        - Use one finding per distinct issue.
        - Keep each finding body concise and specific to the scenario where it breaks.
        - Use priority 0 for P0, 1 for P1, 2 for P2, and 3 for P3.
        - If there are no actionable findings, return an empty findings array and set
          overall_correctness to "{REVIEW_CORRECT}".
        - Return JSON matching the provided schema exactly. Do not wrap it in markdown.
        """
    ).strip()


def fix_prompt(findings_path: Path) -> str:
    check_commands = "\n".join(f"- {quote_command(command)}" for command in DEFAULT_CHECK_COMMANDS)
    return textwrap.dedent(
        f"""\
        Address every actionable review finding listed in {findings_path}.

        Constraints:
        - Only fix issues described in that file.
        - Keep unrelated code unchanged.
        - Run the repository checks before you finish:
        {check_commands}
        - Do not commit changes.
        - Do not push changes.
        """
    ).strip()


def check_repair_prompt(check_failures_path: Path, task_context: str) -> str:
    return textwrap.dedent(
        f"""\
        Repository checks are failing after this task:
        {task_context}

        Read the failing check transcript at {check_failures_path} and make the minimum code changes
        needed to get the repository checks passing.

        Constraints:
        - Only address issues needed for the failing checks.
        - Keep unrelated code unchanged.
        - Do not commit changes.
        - Do not push changes.
        """
    ).strip()


def commit_message_prompt() -> str:
    return textwrap.dedent(
        """\
        Inspect the current staged git diff and write only a git commit message.

        Requirements:
        - Be specific to the actual staged changes.
        - Use an imperative subject line.
        - Keep the first line under 72 characters.
        - Add a short body only when it adds useful context.
        - Do not use markdown, bullets, code fences, or surrounding explanation.
        - Output only the commit message text.
        """
    ).strip()


def read_text_output(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def require_mapping(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise RalphError(f"Review output field {context} must be a JSON object.")
    return value


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RalphError(f"Review output field {context} must be a non-empty string.")
    return value


def require_int(value: object, context: str) -> int:
    if not isinstance(value, int):
        raise RalphError(f"Review output field {context} must be an integer.")
    return value


def optional_float(value: object, context: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise RalphError(f"Review output field {context} must be numeric or null.")


def optional_priority(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and value in {0, 1, 2, 3}:
        return value
    raise RalphError("Review output field findings[].priority must be 0, 1, 2, 3, or null.")


def parse_review_result(review_text: str) -> ReviewResult:
    stripped = review_text.strip()
    if not stripped:
        raise RalphError("Review output was empty.")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise RalphError(f"Review output was not valid JSON: {error}") from error

    review_object = require_mapping(payload, "root")
    findings_payload = review_object.get("findings")
    if not isinstance(findings_payload, list):
        raise RalphError("Review output field findings must be an array.")

    findings: List[Finding] = []
    for index, raw_finding in enumerate(findings_payload):
        finding_object = require_mapping(raw_finding, f"findings[{index}]")
        code_location = require_mapping(finding_object.get("code_location"), f"findings[{index}].code_location")
        line_range = require_mapping(code_location.get("line_range"), f"findings[{index}].code_location.line_range")
        findings.append(
            Finding(
                title=require_string(finding_object.get("title"), f"findings[{index}].title"),
                body=require_string(finding_object.get("body"), f"findings[{index}].body"),
                file_path=require_string(
                    code_location.get("absolute_file_path"),
                    f"findings[{index}].code_location.absolute_file_path",
                ),
                start_line=require_int(line_range.get("start"), f"findings[{index}].code_location.line_range.start"),
                end_line=require_int(line_range.get("end"), f"findings[{index}].code_location.line_range.end"),
                priority=optional_priority(finding_object.get("priority")),
                confidence_score=optional_float(
                    finding_object.get("confidence_score"),
                    f"findings[{index}].confidence_score",
                ),
            )
        )

    overall_correctness = require_string(review_object.get("overall_correctness"), "overall_correctness")
    if overall_correctness not in {REVIEW_CORRECT, REVIEW_INCORRECT}:
        raise RalphError(f"Review output field overall_correctness must be {REVIEW_CORRECT!r} or {REVIEW_INCORRECT!r}.")

    overall_explanation = require_string(review_object.get("overall_explanation"), "overall_explanation")
    overall_confidence_score = optional_float(
        review_object.get("overall_confidence_score"),
        "overall_confidence_score",
    )

    if not findings and overall_correctness != REVIEW_CORRECT:
        raise RalphError("Review output reported no findings but did not mark the patch as correct.")

    return ReviewResult(
        findings=tuple(findings),
        overall_correctness=overall_correctness,
        overall_explanation=overall_explanation,
        overall_confidence_score=overall_confidence_score,
    )


def findings_signature(findings: Sequence[Finding]) -> Tuple[str, ...]:
    return tuple(sorted(finding.fingerprint() for finding in findings))


def parse_jsonl_events(stream_text: str) -> Tuple[dict, ...]:
    events: List[dict] = []
    for line_number, raw_line in enumerate(stream_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RalphError(f"Codex JSON mode emitted invalid JSON on line {line_number}: {error}") from error
        if not isinstance(payload, dict):
            raise RalphError(f"Codex JSON mode event on line {line_number} must be a JSON object.")
        events.append(payload)
    if not events:
        raise RalphError("Codex JSON mode produced no events.")
    return tuple(events)


def select_jsonl_stream(stdout: str, stderr: str) -> Tuple[str, str]:
    candidates = (
        ("stdout", stdout),
        ("stderr", stderr),
    )
    errors = []
    for stream_name, stream_text in candidates:
        if not stream_text.strip():
            continue
        try:
            parse_jsonl_events(stream_text)
        except RalphError as error:
            errors.append(f"{stream_name}: {error}")
            continue
        return stream_text, stream_name

    if not errors:
        raise RalphError("Codex JSON mode produced no events.")
    error_summary = "; ".join(errors)
    raise RalphError(f"Codex JSON mode did not produce a valid event stream ({error_summary}).")


def json_mode_text(value: object, context: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        return json.dumps(value)
    raise RalphError(f"Codex JSON mode field {context} must be a string or JSON object.")


def recover_review_text_from_jsonl(stream_text: str) -> str:
    events = parse_jsonl_events(stream_text)
    for event in reversed(events):
        review_text = json_mode_text(event.get("review_output"), "review_output")
        if review_text:
            return review_text
    for event in reversed(events):
        final_message = json_mode_text(event.get("last_agent_message"), "last_agent_message")
        if final_message:
            return final_message
    raise RalphError("Codex JSON mode did not include a final review result.")


def recover_commit_message_from_jsonl(stream_text: str) -> str:
    events = parse_jsonl_events(stream_text)
    for event in reversed(events):
        final_message = json_mode_text(event.get("last_agent_message"), "last_agent_message")
        if final_message:
            return final_message
    raise RalphError("Codex JSON mode did not include a final commit message.")


def emit_captured_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)


def run_checks_once(cwd: Path, *, emit_output: bool = True) -> List[CheckFailure]:
    failures: List[CheckFailure] = []
    for command in DEFAULT_CHECK_COMMANDS:
        completed = run_command(command, cwd, capture_output=True, check=False)
        if emit_output:
            emit_captured_output(completed)
        if completed.returncode != 0:
            failures.append(
                CheckFailure(
                    command=tuple(command),
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
            )
    return failures


def format_check_failures(failures: Sequence[CheckFailure]) -> str:
    sections = []
    for failure in failures:
        section = [
            f"Command: {quote_command(failure.command)}",
            f"Exit code: {failure.returncode}",
        ]
        stdout = failure.stdout.strip()
        stderr = failure.stderr.strip()
        if stdout:
            section.extend(("stdout:", stdout))
        if stderr:
            section.extend(("stderr:", stderr))
        sections.append("\n".join(section))
    return "\n\n".join(sections)


def check_failures_signature(failures: Sequence[CheckFailure]) -> Tuple[str, ...]:
    return tuple(sorted(failure.fingerprint() for failure in failures))


def baseline_failure_signatures(failures: Sequence[CheckFailure]) -> Tuple[str, ...]:
    return check_failures_signature(failures)


def introduced_check_failures(
    failures: Sequence[CheckFailure], baseline_signatures: Sequence[str]
) -> Tuple[CheckFailure, ...]:
    baseline = set(baseline_signatures)
    return tuple(failure for failure in failures if failure.fingerprint() not in baseline)


def write_check_failures(path: Path, failures: Sequence[CheckFailure]) -> None:
    path.write_text(format_check_failures(failures), encoding="utf-8")


def run_codex_json_capture(config: RalphConfig, prompt: str, *, output_schema_path: Optional[Path] = None) -> str:
    command = codex_base_command(config)
    if output_schema_path is not None:
        command.extend(("--output-schema", str(output_schema_path)))
    command.extend(("--json", prompt))

    completed = run_command(command, config.workdir, capture_output=True, check=False)
    if completed.returncode != 0:
        emit_captured_output(completed)
        raise RalphError(f"Command failed ({completed.returncode}): {quote_command(command)}")
    json_stream, stream_name = select_jsonl_stream(completed.stdout, completed.stderr)
    if completed.stdout and stream_name != "stdout":
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr and stream_name != "stderr":
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    return json_stream


def run_check_repair_pass(config: RalphConfig, check_failures_path: Path, task_context: str) -> None:
    command = codex_base_command(config)
    command.append(check_repair_prompt(check_failures_path, task_context))
    run_command(command, config.workdir)


def run_checks_with_auto_repair(
    config: RalphConfig,
    scratch: Path,
    phase_name: str,
    task_context: str,
    baseline_signatures: Sequence[str],
) -> None:
    previous_signature: Optional[Tuple[str, ...]] = None
    for attempt_number in range(0, config.max_check_repair_rounds + 1):
        failures = run_checks_once(config.workdir)
        if not failures:
            return

        introduced_failures = introduced_check_failures(failures, baseline_signatures)
        if not introduced_failures:
            print(f"Ignoring {len(failures)} baseline check failure(s) during {phase_name}.")
            return

        signature = check_failures_signature(introduced_failures)
        if signature == previous_signature:
            raise RalphError(f"Check failures repeated during {phase_name}; stopping to avoid an endless loop.")
        previous_signature = signature

        if attempt_number >= config.max_check_repair_rounds:
            raise RalphError(
                f"Repository checks failed during {phase_name} after {attempt_number} auto-repair attempt(s)."
            )

        failures_path = scratch / f"{phase_name}-check-failures-{attempt_number + 1}.txt"
        write_check_failures(failures_path, introduced_failures)
        run_check_repair_pass(config, failures_path, task_context)


def stage_all_changes(cwd: Path) -> None:
    run_command(("git", "add", "-A"), cwd)


def ensure_staged_changes(cwd: Path) -> None:
    staged = run_command(("git", "diff", "--cached", "--quiet"), cwd, check=False)
    if staged.returncode == 0:
        raise RalphError("No staged changes were produced.")


def commit_with_generated_message(config: RalphConfig, scratch: Path) -> str:
    stage_all_changes(config.workdir)
    ensure_staged_changes(config.workdir)

    with tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="commit-message-",
        suffix=".txt",
        dir=scratch,
        delete=False,
    ) as message_file:
        message_path = Path(message_file.name)

    command = codex_base_command(config)
    command.extend(("-o", str(message_path), commit_message_prompt()))
    run_command(command, config.workdir)

    message = read_text_output(message_path)
    if message is None:
        json_output = run_codex_json_capture(config, commit_message_prompt())
        message = recover_commit_message_from_jsonl(json_output)
        message_path.write_text(f"{message}\n", encoding="utf-8")
    if not message:
        raise RalphError("Codex did not produce a commit message.")

    run_command(("git", "commit", "-F", str(message_path)), config.workdir)
    return message


def run_initial_implementation(config: RalphConfig) -> None:
    command = codex_base_command(config)
    command.append(implementation_prompt(config.feature_prompt))
    run_command(command, config.workdir)


def run_review(config: RalphConfig, review_base: ReviewBase, review_path: Path) -> ReviewResult:
    prompt = review_prompt(review_base)
    command = codex_base_command(config)
    command.extend(
        (
            "--output-schema",
            str(config.review_schema_path),
            "-o",
            str(review_path),
            prompt,
        )
    )
    run_command(command, config.workdir)

    review_text = read_text_output(review_path)
    if review_text is not None:
        try:
            return parse_review_result(review_text)
        except RalphError as error:
            print(
                f"Review output file {review_path} was invalid ({error}); retrying via Codex JSON mode.",
                file=sys.stderr,
            )

    if review_text is None:
        print(f"Review output file {review_path} was empty; retrying via Codex JSON mode.", file=sys.stderr)

    json_output = run_codex_json_capture(config, prompt, output_schema_path=config.review_schema_path)
    review_text = recover_review_text_from_jsonl(json_output)
    review_path.write_text(f"{review_text}\n", encoding="utf-8")
    return parse_review_result(review_text)


def run_fix_pass(config: RalphConfig, findings_path: Path) -> None:
    command = codex_base_command(config)
    command.append(fix_prompt(findings_path))
    run_command(command, config.workdir)


def push_branch(cwd: Path, remote: str, branch: str) -> None:
    run_command(("git", "push", "-u", remote, branch), cwd)


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    ensure_git_repo(config.workdir)

    if not config.review_schema_path.is_file():
        raise RalphError(f"Review schema file does not exist: {config.review_schema_path}")
    if git_status_porcelain(config.workdir):
        raise RalphError("Working tree is not clean. Commit or stash existing changes before running Ralph.")

    branch = current_branch(config.workdir)
    review_base = ensure_review_base(config.workdir, config.base_ref, branch)
    scratch = create_scratch_dir(config)

    print(f"Using branch: {branch}")
    print(f"Using review base label: {review_base.label}")
    print(f"Using review diff base SHA: {review_base.diff_base_sha}")
    print(f"Using scratch directory: {scratch}")

    baseline_failures = run_checks_once(config.workdir, emit_output=False)
    baseline_signatures = baseline_failure_signatures(baseline_failures)
    if baseline_failures:
        print(
            f"Detected {len(baseline_failures)} pre-existing check failure(s); Ralph will ignore them unless they change."
        )

    run_initial_implementation(config)
    run_checks_with_auto_repair(
        config,
        scratch,
        phase_name="implementation",
        task_context="Implement the requested feature and leave the repository checks passing.",
        baseline_signatures=baseline_signatures,
    )
    first_commit_message = commit_with_generated_message(config, scratch)
    print(f"Committed implementation: {first_commit_message}")

    previous_signature: Optional[Tuple[str, ...]] = None

    for round_number in range(1, config.max_rounds + 1):
        review_path = scratch / f"review-round-{round_number}.json"
        review_result = run_review(config, review_base, review_path)

        if not review_result.findings:
            print(f"Review round {round_number}: no actionable findings")
            if config.push_enabled:
                push_branch(config.workdir, config.push_remote, branch)
            else:
                print("Skipping push because --no-push was supplied.")
            return 0

        signature = findings_signature(review_result.findings)
        if signature == previous_signature:
            raise RalphError(f"Review findings repeated in round {round_number}; stopping to avoid an endless loop.")
        previous_signature = signature

        print(f"Review round {round_number}: {len(review_result.findings)} actionable finding(s)")
        run_fix_pass(config, review_path)
        run_checks_with_auto_repair(
            config,
            scratch,
            phase_name=f"review-round-{round_number}",
            task_context=f"Fix the review findings listed in {review_path} and leave the repository checks passing.",
            baseline_signatures=baseline_signatures,
        )
        commit_message = commit_with_generated_message(config, scratch)
        print(f"Committed review fixes: {commit_message}")

    raise RalphError(f"Reached max review rounds ({config.max_rounds}) without converging.")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RalphError as error:
        print(f"ralph: {error}", file=sys.stderr)
        raise SystemExit(1) from error
