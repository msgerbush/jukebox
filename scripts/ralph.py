#!/usr/bin/env python3
"""Drive a Codex implement/review/fix loop with bounded convergence guards."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

NO_FINDINGS_SENTINEL = "NO_FINDINGS"
FINDING_PREFIX = "FINDING"
DEFAULT_CHECK_COMMANDS: Tuple[Tuple[str, ...], ...] = (
    ("uv", "run", "ruff", "format", "--check"),
    ("uv", "run", "ruff", "check"),
    ("uv", "run", "pytest"),
    ("uv", "run", "ty", "check"),
)


class RalphError(RuntimeError):
    """Raised when the Ralph loop cannot continue safely."""


@dataclass(frozen=True)
class Finding:
    priority: str
    file_path: str
    summary: str

    def fingerprint(self) -> str:
        return "|".join((self.priority, self.file_path, self.summary))


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
        help="Existing git ref to review against. If omitted, Ralph pins the current HEAD as a local base branch.",
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
        "--scratch-dir-name",
        default=".ralph",
        help="Directory under the repo root for review transcripts and other temporary artifacts.",
    )
    return parser


def parse_args(argv: Sequence[str]) -> RalphConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    return RalphConfig(
        feature_prompt=feature_prompt,
        workdir=args.workdir.resolve(),
        codex_bin=args.codex_bin,
        model=args.model,
        max_rounds=args.max_rounds,
        max_check_repair_rounds=args.max_check_repair_rounds,
        base_ref=args.base_ref,
        push_remote=args.push_remote,
        push_enabled=not args.no_push,
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


def ensure_base_ref(cwd: Path, requested_ref: Optional[str]) -> str:
    if requested_ref:
        run_command(("git", "rev-parse", "--verify", requested_ref), cwd)
        return requested_ref

    head_sha = command_output(("git", "rev-parse", "HEAD"), cwd)
    base_ref = f"ralph-base-{head_sha[:12]}"
    exists = run_command(
        ("git", "show-ref", "--verify", "--quiet", f"refs/heads/{base_ref}"),
        cwd,
        check=False,
    )
    if exists.returncode != 0:
        run_command(("git", "branch", base_ref, head_sha), cwd)
    return base_ref


def scratch_dir(cwd: Path, dir_name: str) -> Path:
    directory = cwd / dir_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def codex_base_command(config: RalphConfig) -> List[str]:
    command = [config.codex_bin, "exec", "--full-auto", "-C", str(config.workdir)]
    if config.model:
        command.extend(("-m", config.model))
    return command


def codex_review_command(config: RalphConfig, base_ref: str) -> List[str]:
    command = [config.codex_bin, "exec", "review", "--full-auto", "--base", base_ref]
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


def review_prompt() -> str:
    return textwrap.dedent(
        f"""\
        Review only the diff against the supplied base ref.

        Report only actionable bugs, regressions, behavioral mistakes, and missing tests.
        Ignore style nits, wording, and optional refactors.

        If there are no actionable findings, reply with exactly:
        {NO_FINDINGS_SENTINEL}

        Otherwise output one finding per line in exactly this format:
        {FINDING_PREFIX}|<priority>|<file>|<summary>

        Use priority values P1, P2, or P3.
        Do not include any extra text.
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


def parse_review_output(review_text: str) -> List[Finding]:
    stripped = review_text.strip()
    if not stripped or stripped == NO_FINDINGS_SENTINEL:
        return []

    findings: List[Finding] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) != 4 or parts[0] != FINDING_PREFIX:
            raise RalphError(
                "Unexpected review output. Expected only "
                f"{NO_FINDINGS_SENTINEL!r} or {FINDING_PREFIX}|<priority>|<file>|<summary> lines."
            )
        _, priority, file_path, summary = parts
        findings.append(Finding(priority=priority, file_path=file_path, summary=summary))
    return findings


def findings_signature(findings: Sequence[Finding]) -> Tuple[str, ...]:
    return tuple(sorted(finding.fingerprint() for finding in findings))


def emit_captured_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)


def run_checks_once(cwd: Path) -> List[CheckFailure]:
    failures: List[CheckFailure] = []
    for command in DEFAULT_CHECK_COMMANDS:
        completed = run_command(command, cwd, capture_output=True, check=False)
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


def write_check_failures(path: Path, failures: Sequence[CheckFailure]) -> None:
    path.write_text(format_check_failures(failures), encoding="utf-8")


def run_check_repair_pass(config: RalphConfig, check_failures_path: Path, task_context: str) -> None:
    command = codex_base_command(config)
    command.append(check_repair_prompt(check_failures_path, task_context))
    run_command(command, config.workdir)


def run_checks_with_auto_repair(config: RalphConfig, scratch: Path, phase_name: str, task_context: str) -> None:
    previous_signature: Optional[Tuple[str, ...]] = None
    for attempt_number in range(0, config.max_check_repair_rounds + 1):
        failures = run_checks_once(config.workdir)
        if not failures:
            return

        signature = check_failures_signature(failures)
        if signature == previous_signature:
            raise RalphError(f"Check failures repeated during {phase_name}; stopping to avoid an endless loop.")
        previous_signature = signature

        if attempt_number >= config.max_check_repair_rounds:
            raise RalphError(
                f"Repository checks failed during {phase_name} after {attempt_number} auto-repair attempt(s)."
            )

        failures_path = scratch / f"{phase_name}-check-failures-{attempt_number + 1}.txt"
        write_check_failures(failures_path, failures)
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

    message = message_path.read_text(encoding="utf-8").strip()
    if not message:
        raise RalphError("Codex did not produce a commit message.")

    run_command(("git", "commit", "-F", str(message_path)), config.workdir)
    return message


def run_initial_implementation(config: RalphConfig) -> None:
    command = codex_base_command(config)
    command.append(implementation_prompt(config.feature_prompt))
    run_command(command, config.workdir)


def run_review(config: RalphConfig, base_ref: str, review_path: Path) -> List[Finding]:
    command = codex_review_command(config, base_ref)
    command.extend(("-o", str(review_path), review_prompt()))
    run_command(command, config.workdir)
    return parse_review_output(review_path.read_text(encoding="utf-8"))


def run_fix_pass(config: RalphConfig, findings_path: Path) -> None:
    command = codex_base_command(config)
    command.append(fix_prompt(findings_path))
    run_command(command, config.workdir)


def push_branch(cwd: Path, remote: str, branch: str) -> None:
    run_command(("git", "push", "-u", remote, branch), cwd)


def main(argv: Sequence[str]) -> int:
    config = parse_args(argv)
    ensure_git_repo(config.workdir)

    branch = current_branch(config.workdir)
    base_ref = ensure_base_ref(config.workdir, config.base_ref)
    scratch = scratch_dir(config.workdir, config.scratch_dir_name)

    if git_status_porcelain(config.workdir):
        raise RalphError("Working tree is not clean. Commit or stash existing changes before running Ralph.")

    print(f"Using branch: {branch}")
    print(f"Using review base: {base_ref}")

    run_initial_implementation(config)
    run_checks_with_auto_repair(
        config,
        scratch,
        phase_name="implementation",
        task_context="Implement the requested feature and leave the repository checks passing.",
    )
    first_commit_message = commit_with_generated_message(config, scratch)
    print(f"Committed implementation: {first_commit_message}")

    previous_signature: Optional[Tuple[str, ...]] = None

    for round_number in range(1, config.max_rounds + 1):
        review_path = scratch / f"review-round-{round_number}.txt"
        findings = run_review(config, base_ref, review_path)

        if not findings:
            print(f"Review round {round_number}: {NO_FINDINGS_SENTINEL}")
            if config.push_enabled:
                push_branch(config.workdir, config.push_remote, branch)
            else:
                print("Skipping push because --no-push was supplied.")
            return 0

        signature = findings_signature(findings)
        if signature == previous_signature:
            raise RalphError(f"Review findings repeated in round {round_number}; stopping to avoid an endless loop.")
        previous_signature = signature

        print(f"Review round {round_number}: {len(findings)} actionable finding(s)")
        run_fix_pass(config, review_path)
        run_checks_with_auto_repair(
            config,
            scratch,
            phase_name=f"review-round-{round_number}",
            task_context=f"Fix the review findings listed in {review_path} and leave the repository checks passing.",
        )
        commit_message = commit_with_generated_message(config, scratch)
        print(f"Committed review fixes: {commit_message}")

    raise RalphError(f"Reached max review rounds ({config.max_rounds}) without receiving {NO_FINDINGS_SENTINEL}.")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RalphError as error:
        print(f"ralph: {error}", file=sys.stderr)
        raise SystemExit(1) from error
