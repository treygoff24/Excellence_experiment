import os
import subprocess
import sys


def _plan_output(args):
    cmd = [sys.executable, "-m", "scripts.run_all", "--config", "config/eval_config.yaml", "--plan_only"] + args
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return cp.stdout


def _phase_lines(out: str):
    return [ln.strip() for ln in out.splitlines() if ln.strip().startswith("- ")]


def test_only_step_parse_marks_others_not_selected():
    out = _plan_output(["--only_step", "parse"])
    lines = _phase_lines(out)
    # All phases are listed; non-selected ones should say not selected
    for ln in lines:
        if ln.startswith("- parse:"):
            assert "not selected" not in ln
        else:
            assert "not selected" in ln


def test_from_step_to_step_subset_marks_outside_not_selected():
    out = _plan_output(["--from_step", "parse", "--to_step", "score"])
    lines = _phase_lines(out)
    in_window = False
    for ln in lines:
        name = ln.split(":", 1)[0].replace("-", "").strip()
        phase = name
        # Normalize phase name
        if phase.startswith(" "):
            phase = phase.strip()
        # Phases before parse should be not selected
        if phase in {"prepare", "build", "submit", "poll"}:
            assert "not selected" in ln
        # Phases after score should be not selected
        if phase in {"stats", "costs", "report"}:
            assert "not selected" in ln

