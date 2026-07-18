"""Prepare a deterministic sample of the Olist Brazilian E-Commerce dataset
for verisynth fitting.

Reads the raw Kaggle/mirror CSVs and writes cleaned, downsampled Parquet
files under ``--out`` matching the tables declared in
``examples/olist/skeleton.yaml``: customers, orders, order_items,
order_payments, order_reviews.

The entity key is Olist's ``customer_unique_id`` (``customer_id`` in the raw
``olist_customers_dataset.csv`` is 1:1 with a single order and is *not* a
stable per-customer identifier). Downsampling, when requested via
``--sample-customers N``, keeps the first ``N`` ``customer_unique_id``
values in ascending lexicographic order -- deterministic, and (since the ids
are opaque md5-like hashes) statistically equivalent to a random sample.

Data cleaning applied (see examples/olist/README.md for rationale):
  - ``order_items.freight_value`` is floored at 0.01 (free-shipping zeros
    break lognormal fitting).
  - ``order_payments`` rows with ``payment_value <= 0`` are dropped
    (a handful of 'not_defined' / zero-value voucher rows in the full data).

Usage:
    python examples/olist/prepare_data.py --csv-dir DIR --out DIR \\
        [--sample-customers N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def _read_csv(csv_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_csv(csv_dir / name, infer_schema_length=100_000)


def prepare(csv_dir: Path, out_dir: Path, sample_customers: int) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    customers_raw = _read_csv(csv_dir, "olist_customers_dataset.csv")
    orders_raw = _read_csv(csv_dir, "olist_orders_dataset.csv")

    # Join customers + orders on the (1:1) olist customer_id so downstream
    # order-level frames can be filtered/labeled by customer_unique_id.
    orders_joined = orders_raw.join(
        customers_raw.select(["customer_id", "customer_unique_id", "customer_state"]),
        on="customer_id",
        how="inner",
    )

    # --- 1. customers: one row per customer_unique_id (first occurrence). ---
    customers_df = (
        customers_raw.select(["customer_unique_id", "customer_state"])
        .unique(subset=["customer_unique_id"], keep="first", maintain_order=True)
        .rename({"customer_unique_id": "customer_id"})
    )

    # --- 2. Deterministic sample: first N customer_unique_ids, ascending. ---
    if sample_customers > 0:
        customers_df = customers_df.sort("customer_id").head(sample_customers)
    kept_customer_ids = set(customers_df["customer_id"].to_list())

    # --- 3. orders ---
    orders_df = (
        orders_joined.filter(pl.col("customer_unique_id").is_in(kept_customer_ids))
        .select(
            [
                pl.col("order_id"),
                pl.col("customer_unique_id").alias("customer_id"),
                pl.col("order_status"),
                pl.col("order_purchase_timestamp").str.to_datetime(strict=False),
                pl.col("order_approved_at").str.to_datetime(strict=False),
                pl.col("order_delivered_carrier_date").str.to_datetime(strict=False),
                pl.col("order_delivered_customer_date").str.to_datetime(strict=False),
                pl.col("order_estimated_delivery_date").str.to_datetime(strict=False),
            ]
        )
    )
    kept_order_ids = set(orders_df["order_id"].to_list())

    # --- 4. order_items ---
    items_raw = _read_csv(csv_dir, "olist_order_items_dataset.csv")
    order_items_df = (
        items_raw.filter(pl.col("order_id").is_in(kept_order_ids))
        .select(
            [
                pl.col("order_id"),
                pl.col("price"),
                pl.col("freight_value").clip(lower_bound=0.01),
                pl.col("shipping_limit_date").str.to_datetime(strict=False),
            ]
        )
    )

    # --- 5. order_payments (drop non-positive payment_value rows) ---
    payments_raw = _read_csv(csv_dir, "olist_order_payments_dataset.csv")
    order_payments_df = (
        payments_raw.filter(pl.col("order_id").is_in(kept_order_ids))
        .filter(pl.col("payment_value") > 0)
        .select(["order_id", "payment_type", "payment_installments", "payment_value"])
    )

    # --- 6. order_reviews ---
    reviews_raw = _read_csv(csv_dir, "olist_order_reviews_dataset.csv")
    order_reviews_df = (
        reviews_raw.filter(pl.col("order_id").is_in(kept_order_ids))
        .select(
            [
                pl.col("order_id"),
                pl.col("review_score"),
                pl.col("review_creation_date").str.to_datetime(strict=False),
            ]
        )
    )

    frames = {
        "customers": customers_df,
        "orders": orders_df,
        "order_items": order_items_df,
        "order_payments": order_payments_df,
        "order_reviews": order_reviews_df,
    }

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
    parser.add_argument("--out", required=True, help="output directory for the Parquet sample")
    parser.add_argument(
        "--sample-customers",
        type=int,
        default=0,
        help="keep only the first N customer_unique_ids (ascending); 0 = keep all",
    )
    args = parser.parse_args()

    prepare(Path(args.csv_dir), Path(args.out), args.sample_customers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
