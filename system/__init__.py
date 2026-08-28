"""Shakespeare — a staged, transactional, agent-driven file operations runtime."""

from __future__ import annotations

import sys

__version__ = "0.1.0"

# Shakespeare runs on the free-threaded build only, and `requires-python` cannot say so:
# it pins the interpreter's version, not its build, and a GIL-enabled 3.14 installs and
# imports perfectly happily. So the check lives here, at the import every entry point
# passes through, before a run can start under an interpreter we did not intend.
#
# Two conditions, because "the GIL is off right now" is not the same as "the GIL will
# stay off". Loading an extension module that has not declared free-threaded support
# switches it back on mid-process, and several of ours have not — lxml, and SQLAlchemy's
# cyextension. `sys.flags.gil` is `None` until the GIL is *pinned* by `PYTHON_GIL=0` or
# `-X gil=0`, and pinning it is what makes those imports keep it off.
if sys._is_gil_enabled():
    raise RuntimeError(
        f"Shakespeare requires free-threaded CPython 3.14 or newer, and this "
        f"interpreter ({sys.executable}) is running with the GIL enabled. Recreate the "
        f"environment with `uv sync --group dev` — `.python-version` pins `3.14t` — and "
        f"start the process with PYTHON_GIL=0."
    )
if sys.flags.gil is None:
    raise RuntimeError(
        "The GIL is off but not pinned off, so the first extension module that has not "
        "declared free-threaded support — lxml, SQLAlchemy's cyextension — will switch "
        "it back on part-way through a run. Set PYTHON_GIL=0 (or -X gil=0)."
    )
