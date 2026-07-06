from config import Config
from core import IRenderer
from renderers.base import discover_renderer_plugins, get_renderer_plugin, list_renderer_plugins


def create_renderer(config: Config) -> IRenderer:
    backend = (config.renderer_backend or "auto").lower()

    if backend == "auto":
        candidates = [
            plugin
            for plugin in list_renderer_plugins()
            if plugin.is_available(config) and plugin.auto_priority(config) >= 0
        ]
        if not candidates:
            available = ", ".join(sorted(discover_renderer_plugins().keys()))
            raise ValueError(
                f"No renderer plugin is available for auto selection. Installed plugins: {available}"
            )
        selected = max(candidates, key=lambda plugin: (plugin.auto_priority(config), plugin.name))
        return selected.create(config)

    plugin = get_renderer_plugin(backend)
    if plugin is None:
        available = ", ".join(sorted(discover_renderer_plugins().keys()))
        raise ValueError(
            f"Unsupported renderer backend: {config.renderer_backend}. "
            f"Available plugins: {available}"
        )
    if not plugin.is_available(config):
        raise ValueError(
            f"Renderer backend '{plugin.name}' is registered but not available in the current environment"
        )
    return plugin.create(config)


def get_available_renderer_backends() -> list[str]:
    return sorted(discover_renderer_plugins().keys())
