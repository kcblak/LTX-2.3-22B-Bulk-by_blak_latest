import gzip
import json
import logging
import logging.handlers
import shutil
import sys
import time
from pathlib import Path

from config import Config

TRACE_LEVEL_NUM = 5
_LOG_START_TIME = time.perf_counter()


def _ensure_trace_level() -> None:
    if logging.getLevelName(TRACE_LEVEL_NUM) != "TRACE":
        logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

        def trace(self: logging.Logger, message: str, *args, **kwargs) -> None:
            if self.isEnabledFor(TRACE_LEVEL_NUM):
                self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

        setattr(logging.Logger, "trace", trace)


class ContextFilter(logging.Filter):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = getattr(record, "job_id", "N/A")
        record.project_id = getattr(record, "project_id", self._config.project_id or "N/A")
        record.run_id = getattr(record, "run_id", self._config.run_id or "N/A")
        record.correlation_id = getattr(
            record,
            "correlation_id",
            self._config.correlation_id or self._config.run_id or "N/A",
        )
        record.elapsed_ms = round((time.perf_counter() - _LOG_START_TIME) * 1000, 2)
        return True


class LoggerNamePrefixFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


class MinLevelFilter(logging.Filter):
    def __init__(self, min_level: int):
        super().__init__()
        self._min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self._min_level


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "module": record.name,
            "job_id": getattr(record, "job_id", "N/A"),
            "severity": record.levelname,
            "message": record.getMessage(),
            "elapsed_ms": getattr(record, "elapsed_ms", 0.0),
            "correlation_id": getattr(record, "correlation_id", "N/A"),
            "project_id": getattr(record, "project_id", "N/A"),
            "run_id": getattr(record, "run_id", "N/A"),
            "thread_name": record.threadName,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


class HumanReadableFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = (
            f"{self.formatTime(record, self.datefmt)} | {record.levelname:<8} | "
            f"{record.name} | job={getattr(record, 'job_id', 'N/A')} | "
            f"run={getattr(record, 'run_id', 'N/A')} | "
            f"corr={getattr(record, 'correlation_id', 'N/A')} | "
            f"elapsed_ms={getattr(record, 'elapsed_ms', 0.0)} | {record.getMessage()}"
        )
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


def _resolve_log_level(level_name: str) -> int:
    _ensure_trace_level()
    normalized = level_name.upper()
    if normalized == "TRACE":
        return TRACE_LEVEL_NUM
    return getattr(logging, normalized, logging.INFO)


def _cleanup_old_logs(log_dir: Path, max_age_days: int) -> None:
    if max_age_days <= 0:
        return
    cutoff = time.time() - (max_age_days * 86400)
    for path in log_dir.glob("*.log*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _build_rotating_handler(config: Config, file_path: Path) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=max(1024, config.log_rotation_max_bytes),
        backupCount=max(1, config.log_rotation_backup_count),
        encoding="utf-8",
    )
    if config.log_rotation_compress:
        handler.namer = lambda name: f"{name}.gz"

        def rotator(source: str, dest: str) -> None:
            with open(source, "rb") as input_handle, gzip.open(dest, "wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
            Path(source).unlink(missing_ok=True)

        handler.rotator = rotator
    return handler


def _add_handler(
    root_logger: logging.Logger,
    handler: logging.Handler,
    level: int,
    formatter: logging.Formatter,
    context_filter: ContextFilter,
    filters: list[logging.Filter] | None = None,
) -> None:
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(context_filter)
    for item in filters or []:
        handler.addFilter(item)
    root_logger.addHandler(handler)


def setup_logging(config: Config) -> None:
    """Set up structured logging with correlated, rotated log files."""
    _ensure_trace_level()
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_logs(log_dir, config.log_rotation_max_age_days)

    log_level = _resolve_log_level(config.log_level)
    context_filter = ContextFilter(config)
    json_formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    console_formatter = HumanReadableFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = []

    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        _add_handler(
            root_logger,
            console_handler,
            log_level,
            console_formatter,
            context_filter,
        )

    handler_specs: dict[str, tuple[tuple[str, ...], int | None]] = {
        "application.log": (tuple(), None),
        "render.log": (("engine", "renderers"), None),
        "upload.log": (("upload", "drive"), None),
        "validation.log": (("validation", "preflight"), None),
        "performance.log": (("performance", "observability"), None),
        "drive.log": (("drive",), None),
        "manifest.log": (("jobs.manifest", "jobs.queue"), None),
        "diagnostics.log": (("diagnostics", "bootstrap", "preflight"), None),
        "errors.log": (tuple(), logging.ERROR),
    }

    for filename, (prefixes, min_level) in handler_specs.items():
        if filename == "performance.log" and not config.log_performance:
            continue
        handler = _build_rotating_handler(config, log_dir / filename)
        extra_filters: list[logging.Filter] = []
        if prefixes:
            extra_filters.append(LoggerNamePrefixFilter(prefixes))
        if min_level is not None:
            extra_filters.append(MinLevelFilter(min_level))
        _add_handler(
            root_logger,
            handler,
            logging.DEBUG if filename == "application.log" else log_level,
            json_formatter,
            context_filter,
            extra_filters,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    _ensure_trace_level()
    return logging.getLogger(name)
