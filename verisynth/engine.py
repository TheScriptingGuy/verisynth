"""Orchestration: per-partition Arrow table generation. See docs/ARCHITECTURE.md §6.

`Engine.generate_partition` is a pure function of `(seed, metadata, partition,
num_partitions)`: every partition can be generated independently on any
worker, and the concatenation over partitions is identical to a
single-partition run (see docs/ARCHITECTURE.md §3).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

from . import copula, kernels, temporal, transforms
from . import partition as partition_mod
from .backbone import ParquetBackbone
from .distributions import make_marginal
from .metadata import Metadata

# A table's column state during generation: raw numpy values plus an
# optional boolean null mask (True == null). Present for every declared
# column of the table once it has been processed.
_ColumnState = tuple[np.ndarray, "np.ndarray | None"]


class Engine:
    def __init__(self, metadata: Metadata, seed: int | None = None) -> None:
        self.metadata = metadata
        self.seed = metadata.seed if seed is None else seed

    def generate_partition(self, partition: int, num_partitions: int) -> dict[str, pa.Table]:
        md = self.metadata
        seed = self.seed

        # Per-table state carried forward so children can reference parent
        # row_keys / already-generated parent columns.
        state: dict[str, dict] = {}
        tables_out: dict[str, pa.Table] = {}

        for tname in md.table_order():
            t = md.tables[tname]

            # --- Step 1: row keys -------------------------------------------------
            if t.role == "root":
                row_keys = partition_mod.root_keys(t.rows, partition, num_partitions)
                parent_row_keys = None
                parent_pos = None
            else:
                parent_row_keys = state[t.parent]["row_keys"]
                counts = partition_mod.child_counts(seed, tname, t.cardinality, parent_row_keys)
                row_keys, parent_pos = partition_mod.expand_children(
                    parent_row_keys, counts, t.child_stride
                )

            n = len(row_keys)

            # --- Step 2: copula uniforms -------------------------------------------
            u_map: dict[str, np.ndarray] = {}
            for group in t.copulas:
                u_map.update(copula.copula_uniforms(seed, tname, group, row_keys))

            columns: dict[str, _ColumnState] = {}

            # --- Step 3: distribution columns (declaration order) ------------------
            for cname, c in t.columns.items():
                if c.distribution is None:
                    continue
                if cname in u_map:
                    u = u_map[cname]
                else:
                    u = kernels.keyed_uniforms(seed, f"{tname}.{cname}", row_keys)
                values = make_marginal(c.distribution).ppf(u)
                if c.reference is not None:
                    referenced_rows = md.tables[c.reference].rows
                    values = np.clip(values, 0, referenced_rows - 1)
                columns[cname] = (values, None)

            # --- Step 4: generator columns ------------------------------------------
            for cname, c in t.columns.items():
                if c.generator is None:
                    continue
                if c.generator == "key":
                    values = row_keys.astype(np.int64)
                    mask = None
                elif c.generator == "parent_key":
                    values = parent_row_keys[parent_pos].astype(np.int64)
                    mask = None
                else:  # "parent:{column}" master-data inheritance
                    parent_col_name = c.generator[len("parent:"):]
                    pvalues, pmask = state[t.parent]["columns"][parent_col_name]
                    values = pvalues[parent_pos]
                    mask = pmask[parent_pos] if pmask is not None else None
                columns[cname] = (values, mask)

            # --- Step 5: temporal columns --------------------------------------------
            temporal_names = {cname for cname, c in t.columns.items() if c.temporal is not None}
            if temporal_names:
                anchors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for cname in temporal_names:
                    anchor_ref = t.columns[cname].temporal.anchor
                    if anchor_ref in anchors or anchor_ref in temporal_names:
                        # Same-table temporal->temporal deps are handled
                        # internally by temporal.propagate.
                        continue
                    if "." in anchor_ref:
                        ptable, pcol = anchor_ref.split(".", 1)
                        pvalues, pmask = state[ptable]["columns"][pcol]
                        values_al = pvalues[parent_pos]
                        mask_al = pmask[parent_pos] if pmask is not None else np.zeros(n, dtype=bool)
                        anchors[anchor_ref] = (values_al, mask_al)
                    else:
                        avalues, amask = columns[anchor_ref]
                        amask = amask if amask is not None else np.zeros(n, dtype=bool)
                        anchors[anchor_ref] = (avalues, amask)

                result = temporal.propagate(seed, t, row_keys, anchors)
                for cname, (values, mask) in result.items():
                    columns[cname] = (values, mask)

            # --- Step 6: null masks ---------------------------------------------------
            for cname, c in t.columns.items():
                if c.document is not None:
                    # Document columns aren't populated in `columns` until
                    # Step 7.5 below; their own null_rate is applied there.
                    continue
                if c.null_rate <= 0:
                    continue
                values, mask = columns[cname]
                extra = kernels.keyed_uniforms(seed, f"{tname}.{cname}.__null__", row_keys) < c.null_rate
                mask = extra if mask is None else (mask | extra)
                columns[cname] = (values, mask)

            # --- Step 7: clamp / round / cast -----------------------------------------
            for cname, c in t.columns.items():
                if c.document is not None:
                    continue
                values, mask = columns[cname]
                if c.clamp is not None and c.type in ("int64", "float64", "timestamp"):
                    values = np.clip(values, c.clamp[0], c.clamp[1])

                if c.type == "int64":
                    if np.issubdtype(values.dtype, np.floating):
                        values = np.rint(values).astype(np.int64)
                    else:
                        values = values.astype(np.int64, copy=False)
                elif c.type == "float64":
                    values = values.astype(np.float64, copy=False)
                    if c.round:
                        values = np.rint(values)
                elif c.type == "bool":
                    values = values.astype(np.bool_)
                elif c.type == "timestamp":
                    values = values.astype(np.int64, copy=False)
                # string: left as-is (object array).

                columns[cname] = (values, mask)

            # --- Step 7.5: document columns (in-table JSON/XML payloads) -------------
            # A serialization of the row's own sibling columns, never
            # independently sampled: no RNG draws here beyond the document
            # column's own null-mask below. Imported lazily (matching the
            # lazy `documents` import in `generate`) since documents is an
            # optional leaf of the module graph.
            doc_cnames = [cname for cname, c in t.columns.items() if c.document is not None]
            if doc_cnames:
                from . import documents

                for cname in doc_cnames:
                    c = t.columns[cname]
                    renderer = documents.compile_column_document(md, t, cname)
                    embedded = documents._resolve_document_columns(t, cname)

                    per_column_values: dict[str, list] = {}
                    for ecol in embedded:
                        evalues, emask = columns[ecol]
                        etype = t.columns[ecol].type
                        if etype == "timestamp":
                            pyvals = evalues.astype("datetime64[us]").tolist()
                        elif etype == "int64":
                            pyvals = [int(v) for v in evalues]
                        elif etype == "float64":
                            pyvals = [float(v) for v in evalues]
                        elif etype == "bool":
                            pyvals = [bool(v) for v in evalues]
                        else:  # string
                            pyvals = list(evalues)
                        if emask is not None:
                            pyvals = [
                                None if emask[i] else pyvals[i] for i in range(n)
                            ]
                        per_column_values[ecol] = pyvals

                    values = np.array(
                        [
                            renderer({ecol: per_column_values[ecol][i] for ecol in embedded})
                            for i in range(n)
                        ],
                        dtype=object,
                    )

                    if c.null_rate > 0:
                        mask = (
                            kernels.keyed_uniforms(seed, f"{tname}.{cname}.__null__", row_keys)
                            < c.null_rate
                        )
                    else:
                        mask = None
                    columns[cname] = (values, mask)

            state[tname] = {"row_keys": row_keys, "columns": columns}

            # --- Step 8: Arrow conversion, metadata declaration order -----------------
            arrays: list[pa.Array] = []
            names: list[str] = []
            for cname, c in t.columns.items():
                values, mask = columns[cname]
                if c.type == "int64":
                    arr = pa.array(values, type=pa.int64(), mask=mask)
                elif c.type == "float64":
                    arr = pa.array(values, type=pa.float64(), mask=mask)
                elif c.type == "string":
                    arr = pa.array(values, type=pa.string(), mask=mask)
                elif c.type == "bool":
                    arr = pa.array(values, type=pa.bool_(), mask=mask)
                elif c.type == "timestamp":
                    try:
                        arr = pa.array(values, type=pa.timestamp("us"), mask=mask)
                    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                        arr = pa.array(values.astype("datetime64[us]"), mask=mask)
                else:  # pragma: no cover - metadata validation excludes this
                    raise ValueError(f"unsupported column type {c.type!r} for {tname}.{cname}")
                arrays.append(arr)
                names.append(cname)

            tbl = pa.Table.from_arrays(arrays, names=names)

            # --- Step 9: derived columns -----------------------------------------------
            tbl = transforms.apply_derived(tbl, t.derived)

            tables_out[tname] = tbl

        return tables_out

    def generate(self, out_dir: str, num_partitions: int = 1) -> None:
        backbone = ParquetBackbone(out_dir)
        for p in range(num_partitions):
            tables = self.generate_partition(p, num_partitions)
            for tname, tbl in tables.items():
                backbone.write_partition(tname, tbl, p, source=self.metadata.tables[tname].source)

        # Tables with a `format:` block are additionally rendered as JSON/XML
        # document files from the partitions just written (docs/ARCHITECTURE.md
        # §11). Imported here to keep documents an optional leaf of the graph.
        from .documents import write_documents

        write_documents(self.metadata, out_dir)
