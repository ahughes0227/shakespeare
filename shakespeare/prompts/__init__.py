"""Versioned prompt artifacts.

Prompts live in one `_prompts/<signature>/<version>.yaml` tree rather than inside each
stage package.  DSPy compiles and promotes artifacts, and keeping every version in a
single reviewable directory is what makes promotion diffable and pinning legible.
"""

from .store import PromptStore, PromptStoreError

__all__ = ["PromptStore", "PromptStoreError"]
