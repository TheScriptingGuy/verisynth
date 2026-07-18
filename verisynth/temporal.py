"""Temporal delay propagation: delay along the event-anchor DAG.

See docs/ARCHITECTURE.md §5 (normative). Timestamps travel as
``(values: int64 microseconds, null_mask: bool)`` pairs; a temporal column's
value is its anchor's value plus a nonnegative sampled delay (clamped at 0),
in truncated integer microseconds.
"""

from __future__ import annotations

import numpy as np

from . import kernels
from .distributions import make_delay_ppf
from .metadata import MetadataError, TableSpec

TimeArray = tuple[np.ndarray, np.ndarray]


def order_temporal_columns(table: TableSpec) -> list[str]:
    """Topologically sort ``table``'s temporal columns by same-table anchor deps.

    Columns anchored on a cross-table column (``"parent.col"``, containing a
    dot) or on a non-temporal same-table column have no intra-table
    dependency and may appear as soon as their declaration order allows.
    Ties are broken by metadata declaration order. Raises ``MetadataError``
    (message containing "cycle") if the same-table anchor graph has a cycle.
    """
    temporal_names = [cname for cname, c in table.columns.items() if c.temporal is not None]
    temporal_set = set(temporal_names)

    # same-table anchor dependency: column -> anchor column (if anchor is
    # itself a temporal column in this table)
    depends_on: dict[str, str | None] = {}
    for cname in temporal_names:
        anchor = table.columns[cname].temporal.anchor
        if "." not in anchor and anchor in temporal_set:
            depends_on[cname] = anchor
        else:
            depends_on[cname] = None

    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, chain: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            raise MetadataError(
                f"tables.{table.name}.columns: temporal cycle detected: "
                f"{' -> '.join(chain + [node])}"
            )
        visiting.add(node)
        dep = depends_on[node]
        if dep is not None:
            visit(dep, chain + [node])
        visiting.discard(node)
        visited.add(node)
        order.append(node)

    for cname in temporal_names:
        visit(cname, [])

    return order


def propagate(
    seed: int,
    table: TableSpec,
    row_keys: np.ndarray,
    anchors: dict[str, TimeArray],
) -> dict[str, TimeArray]:
    """Compute temporal column values for ``table``'s ``row_keys``.

    ``anchors`` maps anchor reference strings (aligned to this table's rows)
    to ``TimeArray``s: cross-table refs under their ``"parent.col"`` key,
    same-table non-temporal timestamp columns under their bare column name.
    """
    row_keys = np.asarray(row_keys, dtype=np.uint64)
    order = order_temporal_columns(table)

    computed: dict[str, TimeArray] = {}

    for cname in order:
        col = table.columns[cname]
        anchor_ref = col.temporal.anchor

        if anchor_ref in computed:
            anchor_values, anchor_null_mask = computed[anchor_ref]
        elif anchor_ref in anchors:
            anchor_values, anchor_null_mask = anchors[anchor_ref]
        else:
            raise KeyError(anchor_ref)

        ppf = make_delay_ppf(col.temporal.delay)
        u = kernels.keyed_uniforms(seed, f"{table.name}.{cname}.__delay__", row_keys)
        delay_seconds = ppf(u)
        delay_seconds = np.maximum(delay_seconds, 0.0)
        delay_us = np.trunc(delay_seconds * 1e6).astype(np.int64)

        values = anchor_values + delay_us
        null_mask = anchor_null_mask.copy()
        values = values.copy()
        values[null_mask] = 0

        computed[cname] = (values, null_mask)

    return {cname: computed[cname] for cname in order}
