"""Surrogate-model tasks (database knob tuning via offline sklearn models)."""

from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec
from .paths import (
    SYSBENCH_5_FEATURE_ORDER,
    bundled_knobs_top5_path,
    bundled_surrogate_sysbench5_path,
)
from .http_specs import HTTP_SURROGATE_TASK_IDS
from .http_task import (
    HttpSurrogateKnobTask,
    HttpSurrogateKnobTaskConfig,
    create_http_surrogate_knob_task,
)
from .task import (
    SurrogateKnobTask,
    SurrogateKnobTaskConfig,
    create_surrogate_knob_task,
    create_sysbench5_surrogate_task,
)

# Public alias (tests and docs)
create_surrogate_task = create_surrogate_knob_task

__all__ = [
    "HTTP_SURROGATE_TASK_IDS",
    "HttpSurrogateKnobTask",
    "HttpSurrogateKnobTaskConfig",
    "SURROGATE_BENCHMARKS",
    "SYSBENCH_5_FEATURE_ORDER",
    "SurrogateBenchmarkSpec",
    "SurrogateKnobTask",
    "SurrogateKnobTaskConfig",
    "bundled_knobs_top5_path",
    "bundled_surrogate_sysbench5_path",
    "create_http_surrogate_knob_task",
    "create_surrogate_knob_task",
    "create_surrogate_task",
    "create_sysbench5_surrogate_task",
]
