#!/usr/bin/env python3
"""Generate the JET ``products:`` matrix for the accuracy workload via pytest discovery.

The accuracy suite lives in llm-compiler
(``tests/accuracy/integration/test_llm_api_llmc.py``). Rather than hand-maintain
one workload product per model, we discover the test classes with
``pytest --collect-only`` and emit one product per class, reading each class's
``WORLD_SIZE`` / ``MODEL_NAME`` by importing the (CPU-collectable) module. This keeps
one JET job per model (so each stays within the CI's ~2h SLURM deadline) while staying
in sync automatically as classes are added/removed.

Only the ``products:`` block of the workload yaml is rewritten (it is the last
top-level key); the hand-written ``spec.script`` and everything above it are left
byte-for-byte intact so JET's ``{}`` / ``${{}}`` substitutions are preserved.

Usage:
  generate_accuracy_products.py --llmc-dir <llm-compiler checkout> \
      --workload workloads/accuracy.yaml [--pytest-timeout 3600]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import subprocess
import sys
from pathlib import Path

# Path (relative to the llm-compiler checkout) of the accuracy test module.
_TEST_REL = "tests/accuracy/integration/test_llm_api_llmc.py"


# GPUs per node on the target platform; nodes = ceil(world_size / this).
_GPUS_PER_NODE = int(os.environ.get("LLMC_ACCURACY_GPUS_PER_NODE", "8"))
# Per-product cluster routing. Classes are placed on _PLATFORM_BIG when they are
# @large (need a bigger-memory / multi-node node) or @blackwell (NVFP4/MXFP4 need
# a Blackwell GPU), else on _PLATFORM. Either empty -> that product carries no
# platform and JET uses the workload spec default. This lets one accuracy run put
# 8x80GB-fitting models on the default cluster and NVFP4/large ones on the big one.
_PLATFORM = os.environ.get("LLMC_ACCURACY_PLATFORM", "").strip()
_PLATFORM_BIG = os.environ.get("LLMC_ACCURACY_PLATFORM_BIG", "").strip()


def _discover_classes(llmc_dir: Path) -> list[str]:
    """Test class names, in file order, via ``pytest --collect-only``."""
    accuracy_dir = llmc_dir / "tests" / "accuracy"
    # Augment (don't replace) the inherited env: collection may need HOME,
    # TMPDIR, LD_LIBRARY_PATH, VIRTUAL_ENV, locale, etc. to import cleanly.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(accuracy_dir), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(accuracy_dir)]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(llmc_dir / _TEST_REL),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(llmc_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0,):  # 0 = collected; pytest exits 0 on collect-only
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"pytest --collect-only failed (rc={proc.returncode})")
    seen: list[str] = []
    for line in proc.stdout.splitlines():
        m = re.search(r"::(\w+)::", line)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    if not seen:
        raise SystemExit("no test classes discovered (empty collection)")
    return seen


def _load_module(llmc_dir: Path):
    """Import the accuracy test module (CPU-only; trtllm imported lazily in-test)."""
    accuracy_dir = llmc_dir / "tests" / "accuracy"
    sys.path.insert(0, str(accuracy_dir))  # so ``import accuracy_core`` resolves
    spec = importlib.util.spec_from_file_location("_acc_tests", llmc_dir / _TEST_REL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_mark(cls, name: str) -> bool:
    return any(getattr(m, "name", None) == name for m in getattr(cls, "pytestmark", []))


def _is_skip_marked(cls) -> bool:
    """True if the class carries an unconditional ``@pytest.mark.skip``.

    xfail is intentionally NOT skipped -- those tests still run (and report
    xfail/xpass), so they earn a JET job. But ``skip`` classes never execute, so
    scheduling a (possibly 8-GPU) job for them just burns an allocation.
    """
    return _has_mark(cls, "skip")


def _is_dropped(cls) -> bool:
    """True if the class should NOT get a JET job in this run.

    Drops only unconditional @pytest.mark.skip classes (they never execute).
    @pytest.mark.large classes always run for now -- they are routed to the
    big/Blackwell cluster (see _platform_for), not gated behind an env var.
    """
    return _is_skip_marked(cls)


def _platform_for(cls) -> str:
    """Cluster for a class: the big/Blackwell platform for @large or @blackwell
    classes, else the default platform. Empty -> emit no platform (spec default)."""
    if _PLATFORM_BIG and (_has_mark(cls, "large") or _has_mark(cls, "blackwell")):
        return _PLATFORM_BIG
    return _PLATFORM


def _products_yaml(classes: list[str], mod, pytest_timeout: int) -> str:
    lines = ["products:"]
    for cls_name in classes:
        cls = getattr(mod, cls_name)
        model = cls.MODEL_NAME
        world_size = int(cls.WORLD_SIZE)
        nodes = max(1, math.ceil(world_size / _GPUS_PER_NODE))
        platform = _platform_for(cls)
        target = f"{_TEST_REL}::{cls_name}"
        lines += [
            f"  - class_name: [{cls_name}]",
            f'    model: ["{model}"]',
            f'    model_hints: ["{model}"]',
            f'    pytest_target: ["{target}"]',
            f"    world_size: [{world_size}]",
            f"    nodes: [{nodes}]",
        ]
        if platform:
            lines.append(f'    platform: ["{platform}"]')
        lines.append(f"    pytest_timeout: [{pytest_timeout}]")
    return "\n".join(lines) + "\n"


def _rewrite_products(workload: Path, products_block: str) -> None:
    text = workload.read_text()
    # ``products:`` is the last top-level key; replace from it to EOF, keeping the
    # hand-authored spec/script above untouched.
    m = re.search(r"^products:.*", text, flags=re.MULTILINE | re.DOTALL)
    if not m:
        raise SystemExit(f"no top-level 'products:' key in {workload}")
    workload.write_text(text[: m.start()] + products_block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llmc-dir",
        required=True,
        type=Path,
        help="llm-compiler checkout (contains tests/accuracy/)",
    )
    ap.add_argument(
        "--workload",
        required=True,
        type=Path,
        help="accuracy workload yaml whose products: block to rewrite",
    )
    ap.add_argument("--pytest-timeout", type=int, default=3600)
    args = ap.parse_args()

    discovered = _discover_classes(args.llmc_dir)
    mod = _load_module(args.llmc_dir)
    classes = [c for c in discovered if not _is_dropped(getattr(mod, c))]
    dropped = [c for c in discovered if c not in classes]
    if not classes:
        raise SystemExit("all discovered test classes are dropped; nothing to schedule")
    block = _products_yaml(classes, mod, args.pytest_timeout)
    _rewrite_products(args.workload, block)
    sys.stderr.write(
        f"generated {len(classes)} accuracy products "
        f"(gpus_per_node={_GPUS_PER_NODE}"
        + (f", platform={_PLATFORM}" if _PLATFORM else "")
        + (f", platform_big={_PLATFORM_BIG}" if _PLATFORM_BIG else "")
        + f"): {', '.join(classes)}\n"
    )
    if dropped:
        sys.stderr.write(
            f"dropped {len(dropped)} skip-marked class(es): {', '.join(dropped)}\n"
        )


if __name__ == "__main__":
    main()
