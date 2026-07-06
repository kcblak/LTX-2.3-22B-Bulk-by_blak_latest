import time
import functools
from typing import Callable, TypeVar, Tuple, Any
from core import LTXBulkRendererError
from logging_system import get_logger

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on exceptions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = get_logger("utils.retry")
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            raise LTXBulkRendererError(f"Failed after {max_attempts} attempts")
        return wrapper
    return decorator


def log_execution_time(logger_name: str = "utils.performance") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log the execution time of a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = get_logger(logger_name)
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                logger.info(f"{func.__name__} executed in {elapsed:.4f}s", extra={"job_id": "N/A"})
        return wrapper
    return decorator
