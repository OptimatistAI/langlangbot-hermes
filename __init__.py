"""LangLangBot Hermes platform plugin.

Keep this module import-light. Hermes may import it during plugin discovery
before gateway/platform SDKs are available. Adapter code is loaded inside
``register()``.
"""

from __future__ import annotations

__version__ = "0.2.0-rc.3"


def register(ctx) -> None:
    from .adapter import register as register_adapter
    from .tools import register_tools

    register_adapter(ctx)
    register_tools(ctx)
