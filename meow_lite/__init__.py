"""meow-lite: an OpenAI/Anthropic-compatible LLM API that only speaks cat."""

from meow_lite.meow import MeowGenerator

__all__ = ["MeowGenerator", "app"]


def __getattr__(name):
    # PEP 562: lazy server import. `from meow_lite import app` still works, but
    # importing leaf modules (meow, behavior) no longer drags in fastapi/torch —
    # this is what lets the synthesis pipeline run with only httpx installed.
    if name == "app":
        from meow_lite.server import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
