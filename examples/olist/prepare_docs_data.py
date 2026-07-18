"""Prepare two document-oriented sources: web (JSON) and EDI (XML).

Both frames are **synthesized mirrors** of existing sample tables — the
row sets are identical to their parents, only source-specific attributes
are new — so `verisynth fit` recovers `cardinality: bernoulli{p: 1.0}`
(every parent row appears downstream) and the generated sources agree
row-for-row by construction (docs/ARCHITECTURE.md §8, §11):

1. ``web_orders`` — one row per sample shop order: the storefront's own
   order-event export. ``status`` / ``placed_at`` are copies of the
   order's fields (inherited via ``generator: parent:{column}`` in the
   skeleton, so they are never independently re-generated); ``device``
   and ``utm_source`` are authored web-only attributes. Rendered by the
   engine as ``web/web_orders.json`` shaped by
   ``schemas/web_order.schema.json`` + ``schemas/web_common.schema.json``.

2. ``edi_shipments`` — one EDI shipment-notification message per
   inventory shipment order. ``warehouse`` / ``carrier`` /
   ``dispatched_at`` are copies of the ``inv_shipments`` row;
   ``service_level`` is the authored EDI-only attribute. Rendered as
   ``edi/edi_shipments.xml`` shaped by ``schemas/edi_shipment.xsd`` +
   ``schemas/edi_common.xsd``.

All authored constants are exported so ``tests/test_olist_integration.py``
can check fit fidelity against them, mirroring ``prepare_crm_data.py`` and
``prepare_inventory_data.py``.

Usage:
    python examples/olist/prepare_docs_data.py \\
        --data-dir examples/olist/data --out examples/olist/data \\
        [--seed 20240817]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------
# Authored ground-truth constants (exported; imported by
# tests/test_olist_integration.py to check fitted-vs-authored fidelity).
# --------------------------------------------------------------------------

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_PROBS = [0.62, 0.33, 0.05]

UTM_SOURCES = ["direct", "google", "social", "email"]
UTM_SOURCE_PROBS = [0.40, 0.35, 0.15, 0.10]

SERVICE_LEVELS = ["standard", "express"]
SERVICE_LEVEL_PROBS = [0.85, 0.15]

DEFAULT_SEED = 20240817


# --------------------------------------------------------------------------
# 1. web_orders: one storefront event per shop order.
# --------------------------------------------------------------------------


def _build_web_orders(data_dir: Path, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    orders = pl.read_parquet(data_dir / "orders.parquet")
    n = orders.height

    return pl.DataFrame(
        {
            "web_order_id": np.arange(n, dtype=np.int64),
            "order_id": orders["order_id"].to_list(),
            "status": orders["order_status"].to_list(),
            "placed_at": orders["order_purchase_timestamp"].to_list(),
            "device": rng.choice(DEVICES, size=n, p=DEVICE_PROBS).tolist(),
            "utm_source": rng.choice(UTM_SOURCES, size=n, p=UTM_SOURCE_PROBS).tolist(),
        }
    )


# --------------------------------------------------------------------------
# 2. edi_shipments: one EDI message per inventory shipment order.
# --------------------------------------------------------------------------


def _build_edi_shipments(data_dir: Path, seed: int) -> pl.DataFrame:
    # Distinct sub-stream from web_orders so the two sources' authored
    # attributes are independent draws.
    rng = np.random.default_rng(seed + 1)

    shipments = pl.read_parquet(data_dir / "inv_shipments.parquet")
    n = shipments.height

    return pl.DataFrame(
        {
            "edi_message_id": np.arange(n, dtype=np.int64),
            "shipment_id": shipments["shipment_id"].to_list(),
            "service_level": rng.choice(SERVICE_LEVELS, size=n, p=SERVICE_LEVEL_PROBS).tolist(),
            "warehouse": shipments["warehouse"].to_list(),
            "carrier": shipments["carrier"].to_list(),
            "dispatched_at": shipments["handed_over_at"].to_list(),
        }
    )


def prepare_docs(data_dir: Path, out_dir: Path, seed: int) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "web_orders": _build_web_orders(data_dir, seed),
        "edi_shipments": _build_edi_shipments(data_dir, seed),
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
    parser.add_argument(
        "--data-dir", required=True, help="directory with orders.parquet, inv_shipments.parquet"
    )
    parser.add_argument("--out", required=True, help="output directory for the Parquet files")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_docs(Path(args.data_dir), Path(args.out), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
