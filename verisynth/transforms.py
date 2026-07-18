"""Polars derived columns (`pl.sql_expr`). See docs/ARCHITECTURE.md §2, §6.

Derived columns are evaluated last, after all declared columns have been
generated and cast to their Arrow schema; each `derived[].expr` is a Polars
SQL expression string that may reference any non-derived column.
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from .metadata import DerivedSpec


def apply_derived(tbl: pa.Table, derived: list[DerivedSpec]) -> pa.Table:
    """Append derived columns (evaluated via `pl.sql_expr`) to `tbl`.

    Returns `tbl` unchanged (no copy) when `derived` is empty.
    """
    if not derived:
        return tbl

    df = pl.from_arrow(tbl)
    df = df.with_columns([pl.sql_expr(d.expr).alias(d.name) for d in derived])
    return df.to_arrow()
