# src/indexing/recipes.py

"""
FAISS index recipes (Phase 12f) — safe, validated access to FAISS's index zoo.

The brief's hard rule: *never expose raw FAISS complexity without validation.*
So this module is the single, guarded door to FAISS index construction:

1. A curated **catalog** of recipes (basic + advanced), each declaring its
   tunable parameters with types/ranges/defaults.
2. A **factory-string builder** that turns ``(recipe, params, dim)`` into a
   ``faiss.index_factory`` string.
3. A **validator** that runs static checks (PQ ``m`` must divide the dimension,
   ``nbits`` range, IVF training thresholds, …) *and* — when FAISS is present —
   confirms the string by actually constructing the (empty) index. FAISS is the
   ultimate authority on what combinations are legal, so we let it have the
   final say while still giving fast, human-readable errors first.
4. **Estimators** for memory, training cost, and a search-latency *class* so a
   non-expert can reason about a recipe before building anything.

No FAISS import at module load — it's imported lazily inside the few functions
that need it, keeping ``import src.indexing`` cheap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class IndexMode(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"


class LatencyClass(str, Enum):
    EXACT = "exact"     # brute force; correct but O(n)
    FAST = "fast"       # graph/coarse-quantized ANN
    MEDIUM = "medium"
    SLOW = "slow"


class TrainingCost(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


# --------------------------------------------------------------------------- #
# Parameter + recipe specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecipeParam:
    name: str
    label: str
    kind: str                  # "int"
    default: int
    minimum: int
    maximum: int
    description: str = ""
    # Param group for the UI: "hnsw" | "ivf" | "pq" | "opq"
    group: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    label: str
    mode: IndexMode
    requires_training: bool
    description: str
    params: List[RecipeParam] = field(default_factory=list)
    latency_class: LatencyClass = LatencyClass.FAST
    training_cost: TrainingCost = TrainingCost.NONE
    # supports a final exact-rerank refine wrapper (IndexRefineFlat)
    supports_refine: bool = False

    def param_defaults(self) -> Dict[str, int]:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["latency_class"] = self.latency_class.value
        d["training_cost"] = self.training_cost.value
        d["params"] = [p.to_dict() for p in self.params]
        return d


# --------------------------------------------------------------------------- #
# Reusable parameter definitions
# --------------------------------------------------------------------------- #

_P_M = RecipeParam("M", "HNSW neighbors (M)", "int", 32, 4, 128,
                   "Graph connectivity. Higher = better recall, more memory.", "hnsw")
_P_EFC = RecipeParam("efConstruction", "efConstruction", "int", 200, 8, 1000,
                     "Build-time search depth. Higher = better graph, slower build.", "hnsw")
_P_EFS = RecipeParam("efSearch", "efSearch", "int", 64, 1, 1000,
                     "Query-time search depth. Higher = better recall, slower search.", "hnsw")
_P_NLIST = RecipeParam("nlist", "IVF lists (nlist)", "int", 100, 1, 65536,
                       "Number of Voronoi cells. ~sqrt(N) is a good start.", "ivf")
_P_NPROBE = RecipeParam("nprobe", "IVF probes (nprobe)", "int", 8, 1, 4096,
                        "Cells scanned per query. Higher = better recall, slower.", "ivf")
_P_PQ_M = RecipeParam("pq_m", "PQ sub-quantizers (m)", "int", 8, 1, 96,
                      "Vector is split into m sub-vectors. m MUST divide the dimension.", "pq")
_P_PQ_NBITS = RecipeParam("pq_nbits", "PQ bits (nbits)", "int", 8, 1, 16,
                          "Bits per sub-quantizer code. 8 is standard.", "pq")
_P_OPQ_M = RecipeParam("opq_m", "OPQ blocks (opq_m)", "int", 16, 1, 96,
                       "OPQ rotation block count. Usually equals PQ m.", "opq")
_P_IMI_NBITS = RecipeParam("imi_nbits", "IMI bits", "int", 8, 1, 12,
                           "Bits per sub-quantizer; total cells = 2^(2*nbits).", "ivf")


# --------------------------------------------------------------------------- #
# Factory-string builders (one per recipe)
# --------------------------------------------------------------------------- #


def _f_flat(p, dim): return "Flat"
def _f_hnsw(p, dim): return f"HNSW{p['M']}"
def _f_ivf(p, dim): return f"IVF{p['nlist']},Flat"
def _f_pq(p, dim): return f"PQ{p['pq_m']}x{p['pq_nbits']}"
def _f_ivfpq(p, dim): return f"IVF{p['nlist']},PQ{p['pq_m']}x{p['pq_nbits']}"
def _f_ivf_hnsw(p, dim): return f"IVF{p['nlist']}_HNSW{p['M']},Flat"
def _f_hnsw_pq(p, dim): return f"HNSW{p['M']},PQ{p['pq_m']}x{p['pq_nbits']}"
def _f_opq_ivf_pq(p, dim): return f"OPQ{p['opq_m']},IVF{p['nlist']},PQ{p['pq_m']}x{p['pq_nbits']}"
def _f_imi(p, dim): return f"IMI2x{p['imi_nbits']},Flat"
def _f_refine_pq(p, dim): return f"PQ{p['pq_m']}x{p['pq_nbits']},RFlat"
def _f_multi_d_adc(p, dim): return f"IVF{p['nlist']},PQ{p['pq_m']}np"  # ADC, no fastscan reorder


_BUILDERS: Dict[str, Callable[[Dict[str, int], int], str]] = {
    "flat": _f_flat,
    "hnsw": _f_hnsw,
    "ivf": _f_ivf,
    "pq": _f_pq,
    "ivf_pq": _f_ivfpq,
    "ivf_hnsw": _f_ivf_hnsw,
    "hnsw_pq": _f_hnsw_pq,
    "opq_ivf_pq": _f_opq_ivf_pq,
    "imi": _f_imi,
    "index_refine_flat": _f_refine_pq,
    "multi_d_adc": _f_multi_d_adc,
}


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

RECIPES: Dict[str, RecipeSpec] = {
    "flat": RecipeSpec(
        "flat", "Flat (exact)", IndexMode.BASIC, False,
        "Brute-force exact search. Perfect recall, but scans every vector. "
        "Best for small corpora or as a ground-truth baseline.",
        params=[], latency_class=LatencyClass.EXACT, training_cost=TrainingCost.NONE,
    ),
    "hnsw": RecipeSpec(
        "hnsw", "HNSW (graph ANN)", IndexMode.BASIC, False,
        "Graph-based approximate search. Excellent recall/latency tradeoff and "
        "no training. Higher memory than quantized indexes.",
        params=[_P_M, _P_EFC, _P_EFS],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.NONE,
    ),
    "ivf": RecipeSpec(
        "ivf", "IVF-Flat (clustered)", IndexMode.BASIC, True,
        "Partitions vectors into nlist cells; scans nprobe of them per query. "
        "Needs training. Good for medium/large corpora.",
        params=[_P_NLIST, _P_NPROBE],
        latency_class=LatencyClass.MEDIUM, training_cost=TrainingCost.LIGHT,
    ),
    "pq": RecipeSpec(
        "pq", "PQ (compressed)", IndexMode.BASIC, True,
        "Product Quantization compresses vectors to m codes. Tiny memory "
        "footprint at some recall cost. m must divide the dimension.",
        params=[_P_PQ_M, _P_PQ_NBITS],
        latency_class=LatencyClass.MEDIUM, training_cost=TrainingCost.MODERATE,
    ),
    "ivf_pq": RecipeSpec(
        "ivf_pq", "IVF-PQ (clustered + compressed)", IndexMode.ADVANCED, True,
        "Coarse IVF partitioning plus PQ compression of residuals. The workhorse "
        "for large-scale ANN: small memory, fast search.",
        params=[_P_NLIST, _P_NPROBE, _P_PQ_M, _P_PQ_NBITS],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.MODERATE,
        supports_refine=True,
    ),
    "ivf_hnsw": RecipeSpec(
        "ivf_hnsw", "IVF-HNSW (graph coarse quantizer)", IndexMode.ADVANCED, True,
        "IVF whose coarse quantizer is an HNSW graph — faster cell assignment "
        "at high nlist. Strong for very large corpora.",
        params=[_P_NLIST, _P_NPROBE, _P_M],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.MODERATE,
    ),
    "hnsw_pq": RecipeSpec(
        "hnsw_pq", "HNSW-PQ (graph + compressed)", IndexMode.ADVANCED, True,
        "HNSW graph over PQ-compressed vectors. Lower memory than plain HNSW "
        "with a modest recall cost. m must divide the dimension.",
        params=[_P_M, _P_PQ_M, _P_PQ_NBITS],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.MODERATE,
    ),
    "opq_ivf_pq": RecipeSpec(
        "opq_ivf_pq", "OPQ-IVF-PQ (rotated, best compression)", IndexMode.ADVANCED, True,
        "Adds an OPQ rotation before IVF-PQ to decorrelate dimensions, "
        "improving PQ recall. Highest-quality compressed recipe; heavier "
        "training. opq_m and pq_m must divide the dimension.",
        params=[_P_OPQ_M, _P_NLIST, _P_NPROBE, _P_PQ_M, _P_PQ_NBITS],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.HEAVY,
        supports_refine=True,
    ),
    "imi": RecipeSpec(
        "imi", "IMI (inverted multi-index)", IndexMode.ADVANCED, True,
        "Inverted Multi-Index: a 2-level product of coarse quantizers giving a "
        "huge number of fine cells (2^(2*nbits)). Great for billion-scale.",
        params=[_P_IMI_NBITS, _P_NPROBE],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.MODERATE,
    ),
    "index_refine_flat": RecipeSpec(
        "index_refine_flat", "PQ + Refine-Flat (re-ranked)", IndexMode.ADVANCED, True,
        "PQ for a fast first pass, then exact re-ranking of the top candidates "
        "with full vectors. Recovers most of PQ's lost recall.",
        params=[_P_PQ_M, _P_PQ_NBITS],
        latency_class=LatencyClass.MEDIUM, training_cost=TrainingCost.MODERATE,
    ),
    "multi_d_adc": RecipeSpec(
        "multi_d_adc", "Multi-D-ADC (IVF-PQ, ADC scan)", IndexMode.ADVANCED, True,
        "IVF-PQ using asymmetric distance computation without fast-scan "
        "reordering — a memory-lean large-scale variant.",
        params=[_P_NLIST, _P_NPROBE, _P_PQ_M],
        latency_class=LatencyClass.FAST, training_cost=TrainingCost.MODERATE,
    ),
}

BASIC_RECIPES = [r for r in RECIPES.values() if r.mode == IndexMode.BASIC]
ADVANCED_RECIPES = [r for r in RECIPES.values() if r.mode == IndexMode.ADVANCED]


# --------------------------------------------------------------------------- #
# Errors + validation result
# --------------------------------------------------------------------------- #


class RecipeError(ValueError):
    """Invalid recipe id or parameter set."""

    def __init__(self, message: str, *, field: Optional[str] = None):
        super().__init__(message)
        self.field = field


@dataclass
class RecipeEstimate:
    memory_bytes: int
    bytes_per_vector: float
    training_cost: str
    latency_class: str
    needs_training: bool
    # FAISS's training minimum (≈ #centroids); None when not applicable.
    min_training_points: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecipeValidation:
    ok: bool
    recipe: str
    factory_string: str
    resolved_params: Dict[str, int]
    errors: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimate: Optional[RecipeEstimate] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.estimate is not None:
            d["estimate"] = self.estimate.to_dict()
        return d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def get_recipe(recipe_id: str) -> RecipeSpec:
    spec = RECIPES.get(recipe_id)
    if spec is None:
        raise RecipeError(f"Unknown index recipe: {recipe_id!r}", field="recipe")
    return spec


def resolve_params(recipe_id: str, params: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Merge caller params over defaults; coerce to int; reject unknown keys."""
    spec = get_recipe(recipe_id)
    resolved = spec.param_defaults()
    if params:
        known = {p.name for p in spec.params}
        for key, value in params.items():
            if key not in known:
                # Ignore unrelated keys (e.g. a UI sending the full param bag),
                # but reject keys that look intended-for-this-recipe yet wrong.
                continue
            try:
                resolved[key] = int(value)
            except (TypeError, ValueError):
                raise RecipeError(f"Parameter {key!r} must be an integer", field=key)
    return resolved


def build_factory_string(recipe_id: str, params: Dict[str, int], dim: int) -> str:
    builder = _BUILDERS[get_recipe(recipe_id).id]
    return builder(params, dim)


# --------------------------------------------------------------------------- #
# Static parameter validation
# --------------------------------------------------------------------------- #


def _static_param_errors(spec: RecipeSpec, p: Dict[str, int], dim: int) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []

    def err(field, msg):
        errors.append({"field": field, "message": msg})

    # Range checks for every declared param.
    for param in spec.params:
        v = p.get(param.name)
        if v is None:
            continue
        if v < param.minimum or v > param.maximum:
            err(param.name, f"{param.label} must be in [{param.minimum}, {param.maximum}]; got {v}")

    # PQ: m must divide the (possibly OPQ-rotated) dimension.
    if "pq_m" in p:
        eff_dim = dim
        if "opq_m" in p:
            # OPQ output dim defaults to the input dim in our factory strings.
            eff_dim = dim
        if eff_dim % p["pq_m"] != 0:
            err("pq_m", f"PQ m={p['pq_m']} must divide the vector dimension {eff_dim} "
                        f"(try a divisor of {eff_dim}).")
        # PQ requires at least 2^nbits training points per sub-quantizer; nbits>8
        # is only supported by some index types.
        if p.get("pq_nbits", 8) > 8:
            errors_nbits = ("pq_nbits > 8 is only supported by a subset of FAISS "
                            "index types and disables fast-scan; prefer 8.")
            # treated as warning-level → not a hard error here

    # OPQ: opq_m should divide dim too.
    if "opq_m" in p and dim % p["opq_m"] != 0:
        err("opq_m", f"OPQ m={p['opq_m']} must divide the vector dimension {dim}.")

    # IVF nprobe must not exceed nlist.
    if "nprobe" in p and "nlist" in p and p["nprobe"] > p["nlist"]:
        err("nprobe", f"nprobe={p['nprobe']} cannot exceed nlist={p['nlist']}.")

    return errors


def _warnings(spec: RecipeSpec, p: Dict[str, int], dim: int, n_vectors: Optional[int]) -> List[str]:
    warns: List[str] = []
    if p.get("pq_nbits", 8) != 8:
        warns.append("Non-standard pq_nbits; 8 is recommended for compatibility and speed.")
    # FAISS warns/needs ~39*nlist training points for stable IVF clustering.
    if "nlist" in p and n_vectors is not None:
        need = 39 * p["nlist"]
        if n_vectors < need:
            warns.append(
                f"Only {n_vectors} vectors for nlist={p['nlist']}; FAISS prefers "
                f"≥{need} (≈39×nlist) for stable training. Consider a smaller nlist."
            )
    if spec.id == "flat" and n_vectors and n_vectors > 200_000:
        warns.append("Flat scans every vector; at this scale an ANN recipe will be far faster.")
    return warns


def min_training_points(p: Dict[str, int]) -> Optional[int]:
    """FAISS's practical training-point floor for trainable recipes."""
    floors: List[int] = []
    if "nlist" in p:
        floors.append(p["nlist"])           # ≥1 point per centroid (hard min)
    if "pq_nbits" in p:
        floors.append(2 ** p["pq_nbits"])    # ≥1 point per PQ centroid
    if "imi_nbits" in p:
        floors.append(2 ** p["imi_nbits"])
    return max(floors) if floors else None


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #


def estimate(recipe_id: str, params: Dict[str, int], dim: int, n_vectors: int) -> RecipeEstimate:
    """Rough analytical memory + cost estimate. Deliberately approximate —
    intended to guide a human, not to be exact."""
    spec = get_recipe(recipe_id)
    n = max(int(n_vectors), 0)
    fp = 4  # float32 bytes

    def pq_code_bytes():
        m = params.get("pq_m", 8)
        nbits = params.get("pq_nbits", 8)
        return math.ceil(m * nbits / 8)

    if recipe_id in ("flat",):
        per = dim * fp
    elif recipe_id == "hnsw":
        per = dim * fp + params.get("M", 32) * 2 * 4  # vectors + graph links
    elif recipe_id == "ivf":
        per = dim * fp  # codes are full vectors; +nlist*dim*fp coarse (added below)
    elif recipe_id in ("pq",):
        per = pq_code_bytes()
    elif recipe_id in ("ivf_pq", "multi_d_adc", "imi"):
        per = pq_code_bytes()
    elif recipe_id == "hnsw_pq":
        per = pq_code_bytes() + params.get("M", 32) * 2 * 4
    elif recipe_id == "opq_ivf_pq":
        per = pq_code_bytes()
    elif recipe_id == "index_refine_flat":
        per = pq_code_bytes() + dim * fp  # refine keeps full vectors too
    elif recipe_id == "ivf_hnsw":
        per = dim * fp
    else:  # pragma: no cover - defensive
        per = dim * fp

    total = n * per
    # Coarse-quantizer / codebook overheads.
    if "nlist" in params:
        total += params["nlist"] * dim * fp
    if recipe_id in ("pq", "ivf_pq", "hnsw_pq", "opq_ivf_pq", "multi_d_adc", "index_refine_flat"):
        total += (2 ** params.get("pq_nbits", 8)) * dim * fp  # PQ codebook
    if recipe_id == "opq_ivf_pq":
        total += dim * dim * fp  # OPQ rotation matrix

    return RecipeEstimate(
        memory_bytes=int(total),
        bytes_per_vector=float(per),
        training_cost=spec.training_cost.value,
        latency_class=spec.latency_class.value,
        needs_training=spec.requires_training,
        min_training_points=min_training_points(params),
    )


# --------------------------------------------------------------------------- #
# Full validation (static + FAISS construction)
# --------------------------------------------------------------------------- #


def validate_recipe(
    recipe_id: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    dim: int,
    n_vectors: int = 0,
    construct: bool = True,
) -> RecipeValidation:
    """Validate a recipe + params for a given dimension.

    Runs fast static checks first (clear, field-tagged messages), then — unless
    ``construct=False`` or FAISS is missing — confirms by actually building the
    empty index via ``faiss.index_factory``. Returns a structured result; it
    does not raise for invalid params (the caller inspects ``ok``)."""
    spec = get_recipe(recipe_id)
    resolved = resolve_params(recipe_id, params)
    if dim <= 0:
        return RecipeValidation(
            ok=False, recipe=recipe_id, factory_string="", resolved_params=resolved,
            errors=[{"field": "dim", "message": "Vector dimension must be positive."}],
        )

    factory = build_factory_string(recipe_id, resolved, dim)
    errors = _static_param_errors(spec, resolved, dim)
    warnings = _warnings(spec, resolved, dim, n_vectors or None)

    # Hard training-point floor. ``index_factory`` happily builds an *empty*
    # trainable index, so the construct check below passes — but ``train()``
    # later raises "nx >= k" when the corpus has fewer vectors than centroids
    # (e.g. IVF nlist=100 needs ≥100 points). Surface that here as an error so
    # the builder disables "Build" with a clear reason instead of letting the
    # background job crash. Only enforced when we actually know the corpus size.
    mtp = min_training_points(resolved)
    if mtp and 0 < n_vectors < mtp:
        errors.append({
            "field": "n_vectors",
            "message": (
                f"This recipe trains on the corpus and needs ≥{mtp} vectors, but "
                f"only {n_vectors} are ingested. Ingest more documents, lower "
                f"nlist/pq_nbits, or choose flat/hnsw (no training required)."
            ),
        })

    # FAISS gets the final word on legality.
    if construct and not errors:
        try:
            import faiss

            faiss.index_factory(dim, factory)
        except ImportError:  # pragma: no cover - faiss always present in this env
            warnings.append("FAISS not installed; skipped construction check.")
        except Exception as exc:
            errors.append({"field": "recipe",
                           "message": f"FAISS rejected '{factory}' for dim={dim}: {exc}"})

    est = estimate(recipe_id, resolved, dim, n_vectors) if not errors else None
    return RecipeValidation(
        ok=not errors, recipe=recipe_id, factory_string=factory,
        resolved_params=resolved, errors=errors, warnings=warnings, estimate=est,
    )


def list_recipes(mode: Optional[str] = None) -> List[RecipeSpec]:
    if mode is None:
        return list(RECIPES.values())
    want = IndexMode(mode)
    return [r for r in RECIPES.values() if r.mode == want]
