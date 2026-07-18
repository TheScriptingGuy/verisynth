"""Prepare a third source: a product inventory system (TASK CARD 14).

Two frames, two very different provenances:

1. ``inv_products`` -- **real** master data. The distinct ``product_id``
   values appearing in the shop sample's ``order_items.parquet`` (see
   ``prepare_data.py``), joined against the raw
   ``olist_products_dataset.csv`` and ``product_category_name_translation.csv``
   to attach the English category name and physical dimensions. This is the
   *master* product catalog: ``skeleton.yaml``'s ``order_items.product_id``
   references it via a fitted ``zipf{a, n}`` popularity distribution
   (docs/ARCHITECTURE.md §2, §8) rather than sampling raw product ids -- only
   the popularity *profile* is released, not row-level identity.

   Data cleaning: a category is ``"unknown"`` when
   ``product_category_name`` is null (no category recorded) or has no
   English translation in ``product_category_name_translation.csv``. The raw
   Olist catalog also has a single fully-null row in the sampled subset (no
   category, no dimensions, no photo count); its numeric fields
   (``weight_g``, ``length_cm``, ``height_cm``, ``width_cm``) are filled
   with the sample's median (same floor/clip-style cleaning precedent as
   ``prepare_data.py``'s ``freight_value`` handling) since the skeleton
   declares these columns without a ``null_rate`` (a real 0.01% missingness
   rate isn't worth modeling); ``photos_qty`` is filled with 0 (no photos
   uploaded).

2. ``inv_shipments`` -- **synthesized**, one row per sample order whose
   ``order_status`` is ``delivered`` or ``shipped`` (no row otherwise -- this
   0-or-1-per-order structure is what ``verisynth fit`` recovers as
   ``cardinality: bernoulli{p}``, per docs/ARCHITECTURE.md §7). Models the
   "child-of-another-source's-facts" pattern (§8): the shop order is
   leading, the shipment is created by the *inventory* system some time
   after the order is placed and always strictly later
   (``created_at = order_purchase_timestamp + lognormal(...)``), then
   ``picked_at`` and ``handed_over_at`` chain further delays. All
   authored constants below are exported so
   ``tests/test_olist_integration.py`` can check fit fidelity against them,
   mirroring ``prepare_crm_data.py``.

Usage:
    python examples/olist/prepare_inventory_data.py \\
        --csv-dir ./olist-raw \\
        --data-dir examples/olist/data --out examples/olist/data \\
        [--seed 20240817]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------
# Authored ground-truth constants (exported; imported by
# tests/test_olist_integration.py to check fitted-vs-authored fidelity).
# --------------------------------------------------------------------------

WAREHOUSES = ["SP-01", "SP-02", "RJ-01", "MG-01"]
WAREHOUSE_PROBS = [0.55, 0.20, 0.15, 0.10]

CARRIERS = ["correios", "jadlog", "azul_cargo", "other"]
CARRIER_PROBS = [0.62, 0.18, 0.12, 0.08]

SHIPMENT_ORDER_STATUSES = ("delivered", "shipped")

CREATED_AT_DELAY_MU = 10.2
CREATED_AT_DELAY_SIGMA = 0.8
PICKED_AT_DELAY_MU = 10.8
PICKED_AT_DELAY_SIGMA = 0.7
HANDED_OVER_DELAY_MU = 9.9
HANDED_OVER_DELAY_SIGMA = 0.9

DEFAULT_SEED = 20240817


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _read_csv(csv_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(csv_dir / name, infer_schema_length=100_000)


def _epoch_seconds_to_pydatetimes(seconds: np.ndarray) -> list[datetime]:
    us = np.rint(np.asarray(seconds, dtype=np.float64) * 1_000_000).astype(np.int64)
    return us.astype("datetime64[us]").tolist()


def _iso_to_epoch_seconds(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


# --------------------------------------------------------------------------
# 1. inv_products: real master data.
# --------------------------------------------------------------------------


def _build_inv_products(csv_dir: Path, data_dir: Path) -> pl.DataFrame:
    order_items = pl.read_parquet(data_dir / "order_items.parquet")
    distinct_ids = order_items.select("product_id").unique(maintain_order=True)

    products_raw = _read_csv(csv_dir, "olist_products_dataset.csv")
    translation_raw = _read_csv(csv_dir, "product_category_name_translation.csv")

    joined = (
        distinct_ids.join(products_raw, on="product_id", how="left")
        .join(translation_raw, on="product_category_name", how="left")
        .select(
            [
                pl.col("product_id"),
                pl.coalesce(
                    [pl.col("product_category_name_english"), pl.lit("unknown")]
                ).alias("category"),
                pl.col("product_weight_g").cast(pl.Float64).alias("weight_g"),
                pl.col("product_length_cm").cast(pl.Float64).alias("length_cm"),
                pl.col("product_height_cm").cast(pl.Float64).alias("height_cm"),
                pl.col("product_width_cm").cast(pl.Float64).alias("width_cm"),
                pl.col("product_photos_qty").cast(pl.Int64).alias("photos_qty"),
            ]
        )
    )

    # Fill the handful of fully-null catalog rows (see module docstring).
    for col in ("weight_g", "length_cm", "height_cm", "width_cm"):
        median = joined[col].median()
        joined = joined.with_columns(pl.col(col).fill_null(median))
    joined = joined.with_columns(pl.col("photos_qty").fill_null(0))

    return joined


# --------------------------------------------------------------------------
# 2. inv_shipments: synthesized from sample orders.
# --------------------------------------------------------------------------


def _build_inv_shipments(data_dir: Path, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    orders = pl.read_parquet(data_dir / "orders.parquet")
    shippable = orders.filter(
        pl.col("order_status").is_in(list(SHIPMENT_ORDER_STATUSES))
    ).select(["order_id", "order_purchase_timestamp"])

    n = shippable.height
    purchase_epoch = np.array(
        [_iso_to_epoch_seconds(dt) for dt in shippable["order_purchase_timestamp"].to_list()],
        dtype=np.float64,
    )

    warehouse = rng.choice(WAREHOUSES, size=n, p=WAREHOUSE_PROBS).tolist()
    carrier = rng.choice(CARRIERS, size=n, p=CARRIER_PROBS).tolist()

    created_delay = rng.lognormal(mean=CREATED_AT_DELAY_MU, sigma=CREATED_AT_DELAY_SIGMA, size=n)
    created_epoch = purchase_epoch + created_delay
    created_at = _epoch_seconds_to_pydatetimes(created_epoch)

    picked_delay = rng.lognormal(mean=PICKED_AT_DELAY_MU, sigma=PICKED_AT_DELAY_SIGMA, size=n)
    picked_epoch = created_epoch + picked_delay
    picked_at = _epoch_seconds_to_pydatetimes(picked_epoch)

    handed_over_delay = rng.lognormal(mean=HANDED_OVER_DELAY_MU, sigma=HANDED_OVER_DELAY_SIGMA, size=n)
    handed_over_epoch = picked_epoch + handed_over_delay
    handed_over_at = _epoch_seconds_to_pydatetimes(handed_over_epoch)

    return pl.DataFrame(
        {
            # Deterministic row-index primary key: needed since edi_shipments
            # (prepare_docs_data.py) is a child of inv_shipments, so fit
            # joins on this column. Consumes no RNG draws.
            "shipment_id": np.arange(n, dtype=np.int64),
            "order_id": shippable["order_id"].to_list(),
            "warehouse": warehouse,
            "carrier": carrier,
            "created_at": created_at,
            "picked_at": picked_at,
            "handed_over_at": handed_over_at,
        }
    )


def prepare_inventory(csv_dir: Path, data_dir: Path, out_dir: Path, seed: int) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_products_df = _build_inv_products(csv_dir, data_dir)
    inv_shipments_df = _build_inv_shipments(data_dir, seed)

    print(f"distinct products: {inv_products_df.height}")

    frames = {"inv_products": inv_products_df, "inv_shipments": inv_shipments_df}
    counts: dict[str, int] = {}
    for name, df in frames.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        counts[name] = df.height
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{name}: {df.height} rows -> {path} ({size_mb:.2f} MB)")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True, help="directory with the raw Olist CSVs")
    parser.add_argument("--data-dir", required=True, help="directory with order_items.parquet, orders.parquet")
    parser.add_argument("--out", required=True, help="output directory for the inventory Parquet files")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_inventory(Path(args.csv_dir), Path(args.data_dir), Path(args.out), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
