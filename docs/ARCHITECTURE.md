# Verisynth Architecture

Metadata-driven engine for generating synthetic relational data that preserves
statistical distributions, cross-source correlations, and temporal event sequences.
Production data is reduced to **fitted parameters** (optionally with differential
privacy); generation is a **pure deterministic function** of `(seed, metadata)`.

## Component map

| Component | File | Responsibility |
|---|---|---|
| Metadata DSL | `verisynth/metadata.py` | Dataclasses + YAML/JSON load & validation |
| Keyed RNG (reference) | `verisynth/_reference.py` | Pure-numpy deterministic keyed hashing/uniforms |
| Kernel dispatch | `verisynth/kernels.py` | Selects Rust kernels, falls back to reference |
| Rust kernels | `rust/` (crate `verisynth-kernels`) | Vectorized keyed hash, uniforms, Φ⁻¹ (PyO3) |
| Marginals | `verisynth/distributions.py` | Inverse-CDF samplers per column spec |
| Copula | `verisynth/copula.py` | Gaussian copula: correlated uniforms per row |
| Partition planner | `verisynth/partition.py` | Partition-by-root ranges + child cardinality/keys |
| Temporal | `verisynth/temporal.py` | Delay propagation along event anchor DAG |
| Transforms | `verisynth/transforms.py` | Polars derived columns (`pl.sql_expr`) |
| Backbone | `verisynth/backbone.py` | DuckDB out-of-core Parquet write + SQL validation |
| Engine | `verisynth/engine.py` | Orchestration: per-partition Arrow table generation |
| Fitter | `verisynth/fit.py` | Fit metadata params from real data |
| Privacy | `verisynth/privacy.py` | Laplace mechanism, epsilon budget split |
| CLI | `verisynth/cli.py` | `verisynth generate / validate / fit` |

Interchange format between all stages is **Apache Arrow** (`pyarrow.Table`).
Numeric kernel I/O is **numpy** (`uint64` in, `float64` out), zero-copy convertible
to Arrow buffers.

## 1. Deterministic keyed generation

Every cell value is a pure function of `(seed, table, column, row_key, draw)`.
This makes generation reproducible, order-independent, and embarrassingly parallel:
any partition can be generated on any worker with no coordination.

### 1.1 Hash chain (normative — Rust and Python MUST be bit-identical)

All arithmetic is wrapping (mod 2^64) on unsigned 64-bit integers.

```
GOLDEN  = 0x9E3779B97F4A7C15

mix64(z):                     # splitmix64 finalizer
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)

fnv1a64(s: str):              # over UTF-8 bytes
    h = 0xcbf29ce484222325
    for b in utf8(s): h = (h ^ b) * 0x100000001b3
    return h

cell_hash(seed, ns_hash, key, draw):
    h = mix64(seed ^ GOLDEN)
    h = mix64(h ^ ns_hash)
    h = mix64(h ^ key)
    h = mix64(h ^ draw)
    return h
```

`ns_hash = fnv1a64(namespace)` where the namespace string is:

- plain column value: `"{table}.{column}"`
- null mask: `"{table}.{column}.__null__"`
- copula latent: `"{table}.__copula__.{group_name}.{column}"`
- child cardinality: `"{child_table}.__cardinality__"` (keyed by **parent** row_key)
- temporal delay: `"{table}.{column}.__delay__"`

`draw` defaults to 0 and exists for multi-draw sampling needs.

### 1.2 Uniforms and normals

```
uniform(h) = ((h >> 11) + 0.5) * 2**-53          # open interval (0, 1)
```

`inv_norm_cdf(u)` is **Acklam's rational approximation** of Φ⁻¹, evaluated with
Horner's method in the exact order below (both implementations must share it):

```
a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
     -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
      3.754408661907416e+00]
p_low = 0.02425 ; p_high = 1 - p_low

if u < p_low:        q = sqrt(-2 ln u)
                     x = (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) /
                         ((((d0*q+d1)*q+d2)*q+d3)*q+1)
elif u <= p_high:    q = u - 0.5 ; r = q*q
                     x = (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q /
                         (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1)
else:                q = sqrt(-2 ln(1-u))
                     x = -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) /
                          ((((d0*q+d1)*q+d2)*q+d3)*q+1)
```

### 1.3 Kernel API (`verisynth/kernels.py`)

```python
def keyed_hash(seed: int, namespace: str, keys: np.ndarray[uint64], draw: int = 0) -> np.ndarray[uint64]
def keyed_uniforms(seed: int, namespace: str, keys: np.ndarray[uint64], draw: int = 0) -> np.ndarray[float64]
def inv_norm_cdf(u: np.ndarray[float64]) -> np.ndarray[float64]
```

Dispatch: try `import verisynth_kernels` (the Rust module); on ImportError, or when
env var `VERISYNTH_FORCE_REFERENCE=1`, use `verisynth._reference`. Module attribute
`BACKEND` is `"rust"` or `"reference"`.

Uniform outputs must be **bit-identical** across backends; `inv_norm_cdf` must agree
within `1e-12` absolute. (Acklam's approximation itself is accurate to ~4e-9 absolute
vs the true Φ⁻¹ in the far tails — acceptable by design; no refinement step is used,
so that both backends can implement the formula verbatim.)

## 2. Metadata DSL

YAML (or JSON), loaded into frozen-ish dataclasses. Example:

```yaml
version: 1
seed: 42
tables:
  customers:
    role: root
    rows: 10000
    primary_key: customer_id
    columns:
      customer_id: {type: int64, generator: key}
      region:
        type: string
        distribution: {kind: categorical, categories: [NA, EU, APAC], probs: [0.5, 0.3, 0.2]}
      age:
        type: int64
        distribution: {kind: normal, mean: 41.0, std: 13.0}
        clamp: [18, 95]
        round: true
      income:
        type: float64
        distribution: {kind: lognormal, mu: 10.8, sigma: 0.6}
      signup_at:
        type: timestamp
        distribution: {kind: datetime_uniform, start: "2022-01-01T00:00:00", end: "2025-12-31T23:59:59"}
    copulas:
      - name: profile
        columns: [age, income]
        correlation: [[1.0, 0.55], [0.55, 1.0]]
  orders:
    role: child
    parent: customers
    cardinality: {kind: poisson, lam: 4.2, max: 63}
    child_stride: 64
    primary_key: order_id
    columns:
      order_id:    {type: int64, generator: key}
      customer_id: {type: int64, generator: parent_key}
      order_total:
        type: float64
        distribution: {kind: lognormal, mu: 3.4, sigma: 0.8}
        null_rate: 0.01
      ordered_at:
        type: timestamp
        temporal: {anchor: customers.signup_at, delay: {kind: exponential, rate: 1.0e-6}}
      shipped_at:
        type: timestamp
        temporal: {anchor: ordered_at, delay: {kind: lognormal, mu: 11.5, sigma: 0.6}}
    derived:
      - {name: order_total_eur, expr: "order_total * 0.92"}
```

Rules:

- `role`: `root` (has `rows`) or `child` (has `parent`, `cardinality`, `child_stride`).
- Column `type` ∈ `{int64, float64, string, bool, timestamp}`.
- Column value source is exactly one of: `generator` (`key` | `parent_key`),
  `distribution`, or `temporal`.
- Optional per-column: `clamp: [lo, hi]`, `round: true`, `null_rate: p`.
- Distribution kinds: `categorical{categories, probs}`, `normal{mean, std}`,
  `lognormal{mu, sigma}`, `uniform{low, high}`, `exponential{rate}`,
  `gamma{shape, scale}`, `beta{a, b}`, `uniform_int{low, high}` (inclusive),
  `datetime_uniform{start, end}` (ISO-8601, no tz; second resolution).
- `copulas[].correlation` must be symmetric with unit diagonal; columns listed must
  have a `distribution` in the same table.
- `temporal.anchor` is either `"{col}"` (same table) or `"{parent_table}.{col}"`.
  Delay units are **seconds**; sampled delays are clamped at ≥ 0.
- `derived[].expr` is a Polars SQL expression string (`pl.sql_expr`), evaluated last;
  derived columns may reference any non-derived column.
- Validation errors raise `MetadataError` with a message naming the offending path.

## 3. Row keys, cardinality, partition-by-root

- Root table row_key = global row index `0..rows-1` (uint64). PK (`generator: key`)
  equals row_key.
- Child count per parent: sampled from `cardinality` via
  `u = keyed_uniforms(seed, "{child}.__cardinality__", [parent_key])`, inverse-CDF of
  the distribution, clamped to `[0, max]` and `max < child_stride` (validated).
  Kinds: `poisson{lam, max}`, `uniform_int{low, high, max}`, `fixed{n}`.
- Child row_key = `parent_row_key * child_stride + j` for `j in 0..count-1`.
  Guarantees global uniqueness and determinism without coordination.
  (`generator: key` → own row_key; `generator: parent_key` → parent's row_key.)
- Partition `p` of `P` owns root rows `[floor(N*p/P), floor(N*(p+1)/P))` and,
  recursively, all their descendants. Every partition is generated independently;
  the concatenation over partitions is identical to a single-partition run.

## 4. Gaussian copula

For a copula group `g` over columns `c_1..c_k` with correlation matrix `R`:

1. Validate/repair `R`: symmetrize, eigenvalue-clip at `1e-6`, renormalize to unit
   diagonal (`R = D^-1/2 R D^-1/2`). `L = cholesky(R)` (lower).
2. Per row_key `κ`, per column `j`:
   `ε_j = inv_norm_cdf(keyed_uniforms(seed, "{table}.__copula__.{g}.{c_j}", [κ]))`.
3. `z = L @ ε` (vectorized: `Z[n,k] = E[n,k] @ L.T`).
4. `u_j = Φ(z_j)` via `scipy.special.ndtr`.
5. `u_j` feeds column `c_j`'s marginal ppf (replacing the plain cell uniform).

## 5. Temporal delay propagation

Within a table, temporal columns form a DAG via `anchor`; cross-table anchors point
at a parent-table column (already generated — engine generates parents first and
repeats the parent's values into child row order). Topologically sort; for each:

```
delay = ppf_delay(keyed_uniforms(seed, "{table}.{column}.__delay__", row_keys))
value = anchor_value + max(delay, 0) seconds
```

Timestamps are int64 microseconds since epoch internally (`pa.timestamp("us")`);
delay seconds are multiplied by 1e6 and truncated toward zero to integer microseconds.
If the anchor value is null, the result is null. Cycles → `MetadataError`.

## 6. Engine flow (per partition)

1. Topologically order tables by `parent` (roots first).
2. Roots: row_keys from partition range. Children: expand from parent row_keys via
   cardinality (repeat parent keys per count; compute child row_keys).
3. Generate columns in stages: (a) plain distribution + copula-driven marginals,
   (b) temporal (topological), (c) null masks applied (`null_rate`), (d) clamp/round,
   (e) cast to Arrow schema, (f) Polars derived columns.
4. Emit `dict[str, pa.Table]`; backbone writes
   `{out}/{table}/part-{p:05d}.parquet` via DuckDB `COPY`.

Public API:

```python
class Engine:
    def __init__(self, metadata: Metadata, seed: int | None = None): ...
    def generate_partition(self, partition: int, num_partitions: int) -> dict[str, pa.Table]: ...
    def generate(self, out_dir: str, num_partitions: int = 1) -> None: ...
```

Backbone validation (`ParquetBackbone.validate(metadata) -> list[str]`) runs DuckDB
SQL over the written dataset: PK uniqueness, FK integrity (every child FK exists in
parent), root row counts match metadata, temporal ordering (event ≥ its anchor,
nulls exempt). Returns human-readable violation strings (empty list = OK).

## 7. Fitting & privacy

`fit_metadata(frames, skeleton, epsilon=None, dp_seed=0)` takes real data
(dict of table name → Polars DataFrame or Arrow Table) plus a skeleton `Metadata`
(structure, roles, keys, copula groups declared; params to be filled) and returns a
new `Metadata` with fitted parameters:

- numeric: if `min > 0` and sample skewness > 1 → `lognormal` (mu/sigma of log),
  else `normal` (mean/std).
- string/bool → `categorical` from value counts.
- timestamp → `datetime_uniform` from min/max.
- child cardinality → `poisson` with `lam = mean count per parent`,
  `max = ceil(observed max * 1.5)` (and `child_stride` = next power of two > max).
- copula correlation: Spearman ρ per pair → Pearson on latents via `2·sin(πρ/6)`.
- temporal delays: observed `event − anchor` seconds (nonneg); if ≥ 95% of the
  observed delays are > 0 → **robust lognormal** fitted on the strictly-positive
  subset with `mu = median(log d)`, `sigma = std(log d, ddof=0)` — median-preserving
  by construction (for genuinely lognormal data `median(log) ≈ mean(log)`, so this
  coincides with MLE; for multi-modal-in-log delays it anchors the typical case at
  the cost of compressing the upper quantiles — a documented single-family
  limitation; mixture kinds are future work). Otherwise `exponential` with
  `rate = 1/mean` over all delays.

Differential privacy (optional, `epsilon` set): the *only* things released are the
fitted parameters, each perturbed by the Laplace mechanism with the total epsilon
split evenly across released statistics. Sensitivity assumes each individual
contributes one root row; numeric columns require declared `clamp` bounds in the
skeleton (range → sensitivity of mean = range/n). Category counts get
Laplace(1/ε_i) noise, clamped ≥ 0, renormalized. DP noise uses its own
`numpy.random.default_rng(dp_seed)` — it is *not* part of the keyed generation.

## 8. CLI

```
verisynth generate -m metadata.yaml -o out/ [--partitions K] [--seed S]
verisynth validate -m metadata.yaml -o out/
verisynth fit --input <dir with {table}.parquet> -m skeleton.yaml -o fitted.yaml [--epsilon E] [--dp-seed S]
```

Exit code 0 on success; `validate` exits 1 and prints violations if any.
