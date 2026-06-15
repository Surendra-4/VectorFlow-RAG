"""VectorFlow-RAG package root.

macOS OpenMP guard (must run before torch/faiss/numpy load their OpenMP
runtimes — hence at the very top of the package import):

PyTorch and faiss-cpu each bundle an OpenMP runtime. On macOS (Apple Silicon
especially), entering a faiss OpenMP parallel region — i.e. IVF/PQ *training* —
from a NON-main thread while torch is also loaded segfaults the whole process.
Our background index build/benchmark jobs run on a thread pool, so building an
IVF/PQ index on a real corpus would crash the server outright (not a catchable
exception). Capping OpenMP to a single thread at process start avoids the
parallel region that crashes, with no measurable cost to MPS (GPU) embedding
(the heavy path runs on the Metal GPU, not CPU OpenMP).

Scoped to Darwin so Linux deployments keep full multi-threading, and
``setdefault`` so an operator can still override it explicitly.
"""

import os as _os
import sys as _sys

if _sys.platform == "darwin":
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
