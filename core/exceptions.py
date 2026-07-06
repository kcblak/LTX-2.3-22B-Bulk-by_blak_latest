class LTXBulkRendererError(Exception):
    """Base exception for all LTX Bulk Renderer errors."""
    pass


class ValidationError(LTXBulkRendererError):
    """Raised when validation fails."""
    pass


class RenderError(LTXBulkRendererError):
    """Raised when rendering fails."""
    pass


class RendererInitializationError(RenderError):
    """Raised when renderer initialization fails."""


class RendererInputError(RenderError):
    """Raised when renderer input validation fails."""


class RendererOOMError(RenderError):
    """Raised when GPU memory is exhausted during rendering."""


class RendererEncodingError(RenderError):
    """Raised when video encoding fails."""


class RendererOutputValidationError(RenderError):
    """Raised when a rendered output fails validation."""


class StorageError(LTXBulkRendererError):
    """Raised when storage operations fail."""
    pass


class DriveError(LTXBulkRendererError):
    """Raised when Google Drive operations fail."""
    pass


class BootstrapError(LTXBulkRendererError):
    """Raised when bootstrap fails."""
    pass


class ConfigurationError(LTXBulkRendererError):
    """Raised when configuration is invalid."""
    pass
