# src/cache/codec.py

"""
Cache value codec.

Embeddings are numpy arrays; expansion / retrieval results are plain
Python dataclasses, dicts, and lists. Pickle handles all of these
losslessly and is ~3× faster than JSON-with-base64 for numpy arrays.

Trust model: pickle is unsafe with untrusted data. VectorFlow-RAG is
local-first; the Redis instance is presumed trusted by the operator.
If that assumption ever weakens (e.g. shared-tenant Redis), swap to a
JSON+numpy-bytes codec — the abstraction is here so the swap is local.
"""

from __future__ import annotations

import pickle
from typing import Protocol, runtime_checkable


@runtime_checkable
class Codec(Protocol):
    """Encode/decode cache values to/from raw bytes."""

    def encode(self, value: object) -> bytes: ...
    def decode(self, data: bytes) -> object: ...


class PickleCodec:
    """Default codec — pickle with highest available protocol."""

    def encode(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, data: bytes) -> object:
        return pickle.loads(data)
