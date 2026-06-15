# tests/test_faiss_thread_safety.py

"""Regression guard for the macOS torch+faiss dual-OpenMP segfault.

Training an IVF/PQ FAISS index from a NON-main thread while PyTorch is loaded
segfaults the whole process on macOS (two OpenMP runtimes). Our background
index-build/benchmark jobs run on a thread pool, so this would crash the server.
``src/__init__.py`` caps ``OMP_NUM_THREADS`` to 1 on Darwin to prevent it.

This runs in a SUBPROCESS (so a regression segfaults the child, not pytest) and
with ``OMP_NUM_THREADS`` removed from the environment, so the only thing that can
set it is the package guard under test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_faiss_train_in_worker_thread_does_not_crash():
    code = textwrap.dedent(
        """
        import src  # noqa: F401 — importing the package applies the OpenMP guard
        import threading
        import numpy as np
        import torch  # noqa: F401 — load torch's OpenMP runtime (the crash precondition)
        import faiss

        rng = np.random.default_rng(0)
        v = rng.standard_normal((512, 64)).astype("float32")
        faiss.normalize_L2(v)

        errors = []

        def build():
            try:
                idx = faiss.index_factory(64, "IVF64,PQ8x8")
                idx.train(v)   # IVF/PQ training — the OpenMP parallel region that crashes
                idx.add(v)
            except Exception as exc:  # a catchable error is fine; a segfault is not
                errors.append(repr(exc))

        t = threading.Thread(target=build)
        t.start()
        t.join()
        assert not errors, errors
        print("OK")
        """
    )
    env = {k: val for k, val in os.environ.items() if k != "OMP_NUM_THREADS"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=180, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert proc.returncode == 0, (
        f"index build in a worker thread crashed (returncode={proc.returncode}; "
        f"-11/139 == SIGSEGV)\nstdout={proc.stdout}\nstderr={proc.stderr[-1500:]}"
    )
    assert "OK" in proc.stdout
