def register(ctx):
    """Lazy wrapper — avoids importing gateway deps at test collection time."""
    try:
        from .adapter import register as _register
    except ImportError:
        from adapter import register as _register
    _register(ctx)

__all__ = ["register"]
