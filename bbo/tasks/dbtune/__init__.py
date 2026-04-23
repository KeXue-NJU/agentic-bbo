"""Database knob tasks: offline surrogates, HTTP MariaDB/sysbench, and HTTP surrogate servers."""

from __future__ import annotations

# --- Offline in-process sklearn surrogates (.joblib) ---
from .catalog import SURROGATE_BENCHMARKS, SurrogateBenchmarkSpec, default_knobs_json_path, resolve_bundled_joblib_path
from .paths import (
    SYSBENCH_5_FEATURE_ORDER,
    bundled_knobs_top5_path,
    bundled_surrogate_sysbench5_path,
)
from .http_surrogate_specs import HTTP_SURROGATE_TASK_IDS
from .http_surrogate_task import (
    HttpSurrogateKnobTask,
    HttpSurrogateKnobTaskConfig,
    create_http_surrogate_knob_task,
)
from .offline_surrogate_task import (
    SurrogateKnobTask,
    SurrogateKnobTaskConfig,
    create_surrogate_knob_task,
    create_sysbench5_surrogate_task,
)

# --- HTTP MariaDB + sysbench (Docker API) ---
from .http_mariadb_specs import (
    DATABASE_TASK_SPECS,
    HTTP_DATABASE_TASK_IDS,
    SYSBENCH_TEST_BY_WORKLOAD,
    HttpDatabaseTaskSpec,
    by_task_id,
    is_database_task_id,
)
from .http_mariadb_task import (
    HttpDatabaseKnobTask,
    HttpDatabaseKnobTaskConfig,
    create_http_database_sysbench5_task,
    create_http_database_task,
)
from .cli_mariadb_http import DATABASE_TASK_FAMILY, DATABASE_TASK_NAMES, database_registry_entries, create_database_task_for_registry

# Public alias (tests and docs)
create_surrogate_task = create_surrogate_knob_task

__all__ = [
    "DATABASE_TASK_FAMILY",
    "DATABASE_TASK_NAMES",
    "DATABASE_TASK_SPECS",
    "HTTP_DATABASE_TASK_IDS",
    "HTTP_SURROGATE_TASK_IDS",
    "HttpDatabaseKnobTask",
    "HttpDatabaseKnobTaskConfig",
    "HttpDatabaseTaskSpec",
    "HttpSurrogateKnobTask",
    "HttpSurrogateKnobTaskConfig",
    "SURROGATE_BENCHMARKS",
    "SYSBENCH_5_FEATURE_ORDER",
    "SYSBENCH_TEST_BY_WORKLOAD",
    "SurrogateBenchmarkSpec",
    "SurrogateKnobTask",
    "SurrogateKnobTaskConfig",
    "bundled_knobs_top5_path",
    "bundled_surrogate_sysbench5_path",
    "by_task_id",
    "create_database_task_for_registry",
    "create_http_database_sysbench5_task",
    "create_http_database_task",
    "create_http_surrogate_knob_task",
    "create_surrogate_knob_task",
    "create_surrogate_task",
    "create_sysbench5_surrogate_task",
    "database_registry_entries",
    "default_knobs_json_path",
    "is_database_task_id",
    "resolve_bundled_joblib_path",
]
