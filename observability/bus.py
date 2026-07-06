from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Any, Callable


@dataclass
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_name, []).append(callback)

    def publish(self, event_name: str, **payload: Any) -> None:
        event = Event(name=event_name, payload=payload, job_id=payload.get("job_id"))
        with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))
            wildcard_callbacks = list(self._subscribers.get("*", []))
        for callback in callbacks:
            callback(event)
        for callback in wildcard_callbacks:
            callback(event)
