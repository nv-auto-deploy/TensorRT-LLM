"""Name-based RoPE/Rotary module replacement (detection + replace).

This is intentionally simple and model-specific by design:
- Walk `model.named_modules()`
- Identify candidate rotary/rope modules by qualified name / class name substrings
- Replace each matched module with a wrapper that calls an **identity custom op**
  (`auto_deploy::noop_rope_cos_sin`) so shapes/dtypes are preserved exactly.

No tracing. No torch.export. No torch.compile. No graph rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


def _module_matches(name: str, mod: nn.Module, keywords: Sequence[str]) -> bool:
    s = (name + " " + mod.__class__.__name__).lower()
    return any(k.lower() in s for k in keywords)


def _set_module_by_name(root: nn.Module, qualified_name: str, new_mod: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent: nn.Module = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


class NoopRotaryWrapper(nn.Module):
    """Wrapper that preserves exact behavior but routes outputs through a no-op custom op.

    If the wrapped module returns a (cos, sin) tuple, we apply:
      torch.ops.auto_deploy.noop_rope_cos_sin(cos, sin)
    Otherwise, we fall back to tensor passthrough:
      torch.ops.auto_deploy.noop_passthrough(x)
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)

        # Typical rotary_emb returns (cos, sin)
        if (
            isinstance(out, tuple)
            and len(out) == 2
            and isinstance(out[0], torch.Tensor)
            and isinstance(out[1], torch.Tensor)
        ):
            cos, sin = out
            return torch.ops.auto_deploy.noop_rope_cos_sin.default(cos, sin)

        # Some variants may return a single tensor
        if isinstance(out, torch.Tensor):
            return torch.ops.auto_deploy.noop_passthrough.default(out)

        # Unknown structure: return unchanged (still “works” but not opaque)
        return out


@dataclass(frozen=True)
class RopeReplacementResult:
    replaced: List[str]


def replace_rope_modules_by_name(
    model: nn.Module,
    *,
    keywords: Iterable[str] = ("rotary", "rope", "rotaryembedding", "rotary_emb"),
    exclude_keywords: Iterable[str] = (),
) -> RopeReplacementResult:
    """Replace RoPE/rotary modules by name/class substring match."""
    keywords_l = tuple(k.lower() for k in keywords)
    exclude_l = tuple(k.lower() for k in exclude_keywords)

    replaced: List[str] = []

    # Snapshot names first to avoid issues while mutating modules during traversal.
    named = list(model.named_modules())
    for name, mod in named:
        if not name:
            continue
        # TODO: we should find a better way to match modules instead of relying on name matching.
        if not _module_matches(name, mod, keywords_l):
            continue
        if exclude_l and _module_matches(name, mod, exclude_l):
            continue
        _set_module_by_name(model, name, NoopRotaryWrapper(mod))
        replaced.append(name)

    return RopeReplacementResult(replaced=replaced)
