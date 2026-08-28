"""Offline prompt optimization.

DSPy never runs inside a transactional run and is not a second execution path.  It
compiles out-of-band against obligation-derived metrics and its product is a
`PromptArtifact`: a plain versioned file, digested, committed to git, and pinned by the
stage package that uses it.

That pinning is what makes self-improving prompts safe here.  A prompt version feeds the
workflow digest, so promoting one is a visible versioned change and `replay` resolves the
version a run actually used rather than today's best.
"""

from .metric import METRIC_WEIGHTS, obligation_score
from .promotion import PromotionGate, PromotionOutcome

__all__ = ["METRIC_WEIGHTS", "PromotionGate", "PromotionOutcome", "obligation_score"]
