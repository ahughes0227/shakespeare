"""Versioned prompt artifacts.

A prompt is not a kind of file to be kept with other files of its kind. It belongs to the
system that speaks it, so it lives in that system's own directory — a capability's prompts
under the capability, the planner's under planning — and a signature is resolved by asking
which system it names.

Every version is kept, because DSPy compiles and promotes artifacts and a promotion is only
reviewable as a diff against the version it replaces.
"""

from .store import PromptStore, PromptStoreError

__all__ = ["PromptStore", "PromptStoreError"]
