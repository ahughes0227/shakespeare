"""The logic operators call.

Kept apart from `operators/` on purpose. `operators/` is the catalog — one directory per
family, one file per operator, nothing else — so that opening it answers "what can this
system do" without wading through implementations. This is where the implementations live,
organised by subject rather than by family, because a subject is shared: naming serves both
rendering and collision resolution, and planning serves the plan operators and the gate
checks alike.
"""
