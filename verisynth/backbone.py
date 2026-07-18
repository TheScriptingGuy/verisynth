"""DuckDB out-of-core Parquet write + SQL validation.

See docs/ARCHITECTURE.md §6 (normative). Each partition of each table is
written as its own Parquet file under `{out_dir}/{table}/part-{p:05d}.parquet`;
validation runs SQL checks (PK uniqueness, root row counts, FK integrity,
temporal ordering) over a DuckDB view spanning all partitions of a table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa

from .metadata import Metadata


class ParquetBackbone:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _table_dir(self, table_name: str) -> Path:
        d = self.out_dir / table_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_partition(self, table_name: str, tbl: pa.Table, partition: int) -> Path:
        path = self._table_dir(table_name) / f"part-{partition:05d}.parquet"
        con = duckdb.connect()
        try:
            con.register("__t", tbl)
            con.execute(f"COPY (SELECT * FROM __t) TO '{path}' (FORMAT PARQUET)")
        finally:
            con.close()
        return path

    def validate(self, metadata: Metadata) -> list[str]:
        violations: list[str] = []
        con = duckdb.connect()
        try:
            for tname in metadata.table_order():
                glob = str(self.out_dir / tname / "*.parquet")
                con.execute(
                    f"CREATE OR REPLACE VIEW {tname} AS SELECT * FROM read_parquet('{glob}')"
                )

            for tname in metadata.table_order():
                t = metadata.tables[tname]
                pk = t.primary_key

                total, distinct_pk = con.execute(
                    f"SELECT count(*), count(distinct {pk}) FROM {tname}"
                ).fetchone()
                if total != distinct_pk:
                    violations.append(
                        f"{tname}: primary key {pk!r} is not unique "
                        f"({total} rows, {distinct_pk} distinct values)"
                    )

                if t.role == "root":
                    if total != t.rows:
                        violations.append(
                            f"{tname}: expected {t.rows} rows (spec.rows), found {total}"
                        )
                else:
                    fk_col = next(
                        cn for cn, c in t.columns.items() if c.generator == "parent_key"
                    )
                    parent_pk = metadata.tables[t.parent].primary_key
                    (orphan_count,) = con.execute(
                        f"""
                        SELECT count(*) FROM {tname} c
                        LEFT JOIN {t.parent} p ON c.{fk_col} = p.{parent_pk}
                        WHERE p.{parent_pk} IS NULL
                        """
                    ).fetchone()
                    if orphan_count:
                        violations.append(
                            f"{tname}: {orphan_count} rows have {fk_col!r} with no "
                            f"matching {t.parent}.{parent_pk}"
                        )

                for cname, c in t.columns.items():
                    if c.temporal is None:
                        continue
                    anchor = c.temporal.anchor
                    if "." in anchor:
                        ptable, pcol = anchor.split(".", 1)
                        fk_col = next(
                            cn for cn, cc in t.columns.items() if cc.generator == "parent_key"
                        )
                        parent_pk = metadata.tables[ptable].primary_key
                        (bad_count,) = con.execute(
                            f"""
                            SELECT count(*) FROM {tname} c
                            JOIN {ptable} p ON c.{fk_col} = p.{parent_pk}
                            WHERE c.{cname} IS NOT NULL AND p.{pcol} IS NOT NULL
                              AND c.{cname} < p.{pcol}
                            """
                        ).fetchone()
                        if bad_count:
                            violations.append(
                                f"{tname}.{cname}: {bad_count} rows precede their anchor "
                                f"{anchor!r}"
                            )
                    else:
                        (bad_count,) = con.execute(
                            f"""
                            SELECT count(*) FROM {tname}
                            WHERE {cname} IS NOT NULL AND {anchor} IS NOT NULL
                              AND {cname} < {anchor}
                            """
                        ).fetchone()
                        if bad_count:
                            violations.append(
                                f"{tname}.{cname}: {bad_count} rows precede their anchor "
                                f"{anchor!r}"
                            )
        finally:
            con.close()

        return violations


def validate_dataset(metadata: Metadata, out_dir: str | Path) -> list[str]:
    """Convenience wrapper: `ParquetBackbone(out_dir).validate(metadata)`."""
    return ParquetBackbone(out_dir).validate(metadata)
