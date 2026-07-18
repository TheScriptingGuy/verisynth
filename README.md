# verisynth

A metadata-driven engine for generating synthetic relational data that preserves statistical distributions, cross-source correlations, and temporal event sequences, so production records stay private while the data still behaves like the real thing.

## How it works

Real data is reduced to **fitted parameters** (marginal distributions, copula
correlations, cardinalities, temporal delay distributions) — optionally perturbed
with differential privacy. Generation is then a **pure deterministic function** of
`(seed, metadata)`: every cell is derived from a keyed hash of
`(seed, table, column, row_key)`, so output is reproducible, order-independent, and
embarrassingly parallel — any partition of the root entities (and all their child
rows) can be generated on any worker with no coordination, and the result is
byte-identical regardless of partition count.

- **Python control plane** orchestrates; **Rust (PyO3) kernels** do the keyed
  hashing/uniform/Φ⁻¹ math (a bit-identical pure-numpy fallback is built in).
- **Arrow** is the interchange format; **DuckDB** writes/validates the Parquet
  dataset out-of-core; **Polars** evaluates derived-column expressions.
- **Gaussian copulas** preserve cross-column correlation; **temporal delay
  propagation** keeps event sequences (e.g. signup → order → shipment) ordered
  and realistically distributed.
- **`verisynth explain`** renders any metadata document as a plain-language
  Markdown explanation (structure, distributions, correlations, temporal
  chains, cross-source relationships, privacy posture) for readers who don't
  read the DSL. See `examples/olist/EXPLAIN.md` for a generated example.
- **Multi-source datasets**: tables can be grouped into named source systems
  (`source: crm`, `source: shop`) with a master source owning shared entities —
  downstream tables inherit master attributes via `generator: parent:{column}`,
  so the generated sources agree row-for-row by construction. See
  `examples/olist/` for a two-source CRM + shop example.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, including
the normative hash-chain and metadata DSL specs.

## Install

```bash
pip install -e .                                   # pure-Python (reference kernels)
# optional compiled kernels (~8x faster):
pip install maturin
maturin build --release -m rust/Cargo.toml -o dist
pip install dist/verisynth_kernels-*.whl
```

## Usage

```bash
# scan real data files: detect PKs, FK relations, and cardinality profiles
verisynth scan --input real/

# build a metadata skeleton through an interactive chat -- point it at real
# data and every question comes pre-answered by the scanner (--yes accepts all)
verisynth init --input real/ -o skeleton.yaml

# generate a dataset from metadata (see examples/retail.yaml)
verisynth generate -m examples/retail.yaml -o out/ --partitions 4 --seed 42

# check PK/FK integrity, row counts, and temporal ordering of the output
verisynth validate -m examples/retail.yaml -o out/

# render a metadata document as a plain-language Markdown explanation
verisynth explain -m examples/retail.yaml -o explain.md

# fit metadata parameters from real data (one {table}.parquet per table),
# optionally with differential privacy on the released parameters
verisynth fit --input real/ -m skeleton.yaml -o fitted.yaml --epsilon 1.0
```

Or from Python:

```python
from verisynth import Engine, load_metadata

md = load_metadata("examples/retail.yaml")
tables = Engine(md, seed=42).generate_partition(0, num_partitions=4)  # dict[str, pyarrow.Table]
Engine(md, seed=42).generate("out/", num_partitions=4)                # Parquet dataset
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q                       # full suite
VERISYNTH_FORCE_REFERENCE=1 python -m pytest -q   # force the numpy kernel backend
```
