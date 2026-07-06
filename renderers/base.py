import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Optional

from config import Config
from core import IRenderer


RendererAvailabilityCheck = Callable[[Config], bool]
RendererPriorityResolver = Callable[[Config], int]


@dataclass(frozen=True)
class RendererPluginSpec:
    name: str
    renderer_cls: type[IRenderer]
    description: str = ""
    availability_check: Optional[RendererAvailabilityCheck] = None
    auto_priority_resolver: Optional[RendererPriorityResolver] = None

    def is_available(self, config: Config) -> bool:
        if self.availability_check is None:
            return True
        return self.availability_check(config)

    def auto_priority(self, config: Config) -> int:
        if self.auto_priority_resolver is None:
            return -1
        return self.auto_priority_resolver(config)

    def create(self, config: Config) -> IRenderer:
        return self.renderer_cls(config)


_PLUGIN_REGISTRY: dict[str, RendererPluginSpec] = {}
_PLUGINS_DISCOVERED = False


def register_renderer_plugin(
    *,
    name: str,
    description: str = "",
    availability_check: Optional[RendererAvailabilityCheck] = None,
    auto_priority_resolver: Optional[RendererPriorityResolver] = None,
):
    def decorator(renderer_cls: type[IRenderer]) -> type[IRenderer]:
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError("Renderer plugin name must not be empty")
        _PLUGIN_REGISTRY[normalized_name] = RendererPluginSpec(
            name=normalized_name,
            renderer_cls=renderer_cls,
            description=description,
            availability_check=availability_check,
            auto_priority_resolver=auto_priority_resolver,
        )
        return renderer_cls

    return decorator


def discover_renderer_plugins() -> dict[str, RendererPluginSpec]:
    global _PLUGINS_DISCOVERED

    if _PLUGINS_DISCOVERED:
        return dict(_PLUGIN_REGISTRY)

    import renderers.plugins as plugins_package

    for module_info in pkgutil.iter_modules(
        plugins_package.__path__, f"{plugins_package.__name__}."
    ):
        importlib.import_module(module_info.name)

    _PLUGINS_DISCOVERED = True
    return dict(_PLUGIN_REGISTRY)


def get_renderer_plugin(name: str) -> Optional[RendererPluginSpec]:
    plugins = discover_renderer_plugins()
    return plugins.get(name.strip().lower())


def list_renderer_plugins() -> list[RendererPluginSpec]:
    return list(discover_renderer_plugins().values())
