"""Canonical, human-readable names for VS-Hier experiment runs."""

from __future__ import annotations

from _paths import configure_vendor_paths
configure_vendor_paths()

import re


_RUN_NAME = re.compile(
    r"^(?P<task>[a-z][a-z0-9-]*)_s(?P<seed>[1-9][0-9]*)_"
    r"(?P<clip_status>clip|noclip)_k(?P<k>[1-9][0-9]*)_"
    r"(?P<branch_k>[1-9][0-9]*)k_(?P<final_k>[1-9][0-9]*)k_"
    r"(?P<role>full|basecontrol)$"
)


def _as_k_steps(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer number of transitions")
    if value <= 0 or value % 1_000:
        raise ValueError(f"{field} must be a positive multiple of 1000 transitions")
    return value // 1_000


def build_run_name(
    task: str,
    seed: int,
    clip_status: str,
    k: int,
    branch_steps: int,
    final_steps: int,
    role: str,
) -> str:
    """Build a run name from explicit values; no training budget is implied."""
    task = task.lower()
    if not re.fullmatch(r"[a-z][a-z0-9-]*", task):
        raise ValueError("task must use lowercase letters, digits, or hyphens")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
        raise ValueError("seed must be a positive integer")
    if clip_status not in {"clip", "noclip"}:
        raise ValueError("clip_status must be 'clip' or 'noclip'")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if role not in {"full", "basecontrol"}:
        raise ValueError("role must be 'full' or 'basecontrol'")

    branch_k = _as_k_steps(branch_steps, "branch_steps")
    final_k = _as_k_steps(final_steps, "final_steps")
    if branch_k >= final_k:
        raise ValueError("branch_steps must be smaller than final_steps")

    return f"{task}_s{seed}_{clip_status}_k{k}_{branch_k}k_{final_k}k_{role}"


def parse_run_name(name: str) -> dict[str, int | str]:
    """Parse a canonical run name into explicit protocol fields."""
    match = _RUN_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid project run name: {name!r}")
    values = match.groupdict()
    branch_steps = int(values.pop("branch_k")) * 1_000
    final_steps = int(values.pop("final_k")) * 1_000
    return {
        "task": values["task"],
        "seed": int(values["seed"]),
        "clip_status": values["clip_status"],
        "k": int(values["k"]),
        "branch_steps": branch_steps,
        "final_steps": final_steps,
        "role": values["role"],
    }
