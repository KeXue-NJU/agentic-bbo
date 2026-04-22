"""Register in-process (local joblib) surrogate tasks for ``bbo.run``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...core import Task
from .catalog import SURROGATE_BENCHMARKS
from .task import create_surrogate_knob_task

INPROC_SURROGATE_TASK_NAMES: frozenset[str] = frozenset(SURROGATE_BENCHMARKS)


def inproc_surrogate_registry_entries() -> dict[str, str]:
    return {n: "surrogate" for n in INPROC_SURROGATE_TASK_NAMES}


def create_inproc_surrogate_task_for_registry(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
    **kwargs: Any,
) -> Task:
    _ = noise_std
    if name not in INPROC_SURROGATE_TASK_NAMES:
        known = ", ".join(sorted(INPROC_SURROGATE_TASK_NAMES))
        raise ValueError(f"Unknown surrogate task `{name}`. Known: {known}")
    sp = kwargs.get("surrogate_path")
    kjp = kwargs.get("knobs_json_path")
    return create_surrogate_knob_task(
        name,
        max_evaluations=max_evaluations,
        seed=seed,
        surrogate_path=Path(sp) if sp is not None else None,
        knobs_json_path=Path(kjp) if kjp is not None else None,
    )


__all__ = [
    "INPROC_SURROGATE_TASK_NAMES",
    "create_inproc_surrogate_task_for_registry",
    "inproc_surrogate_registry_entries",
]
