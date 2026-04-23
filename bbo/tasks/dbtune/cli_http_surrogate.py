"""Register HTTP surrogate tasks with ``bbo.tasks.registry`` / ``python -m bbo.run``."""

from __future__ import annotations

from typing import Any

from ...core import Task
from .http_surrogate_specs import (
    HTTP_SURROGATE_TASK_IDS,
    is_http_surrogate_task_id,
)
from .http_surrogate_task import create_http_surrogate_knob_task

HTTP_SURROGATE_TASK_FAMILY = "http_surrogate"
HTTP_SURROGATE_TASK_NAMES: frozenset[str] = frozenset(HTTP_SURROGATE_TASK_IDS)


def http_surrogate_registry_entries() -> dict[str, str]:
    """http_task_id -> family label for ``TASK_REGISTRY``."""
    return {n: HTTP_SURROGATE_TASK_FAMILY for n in HTTP_SURROGATE_TASK_IDS}


def create_http_surrogate_task_for_registry(
    name: str,
    *,
    max_evaluations: int | None = None,
    seed: int = 0,
    noise_std: float = 0.0,
    **kwargs: Any,
) -> Task:
    """Dispatch when ``name`` is ``knob_http_surrogate_*``."""
    _ = noise_std
    if not is_http_surrogate_task_id(name):
        known = ", ".join(sorted(HTTP_SURROGATE_TASK_IDS))
        raise ValueError(f"Unknown HTTP surrogate task `{name}`. Known: {known}")
    return create_http_surrogate_knob_task(
        name,
        max_evaluations=max_evaluations,
        seed=seed,
        base_url=kwargs.get("http_surrogate_base_url"),
        request_timeout_sec=kwargs.get("http_surrogate_timeout_sec"),
        skip_health_check=bool(kwargs.get("http_surrogate_skip_health_check", False)),
    )


__all__ = [
    "HTTP_SURROGATE_TASK_FAMILY",
    "HTTP_SURROGATE_TASK_NAMES",
    "create_http_surrogate_task_for_registry",
    "http_surrogate_registry_entries",
]
