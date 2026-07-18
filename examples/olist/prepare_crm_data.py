"""Synthesize a CRM source (``crm_contacts`` + ``crm_tickets``) that treats
the Olist shop customer sample as *its* customer master (TASK CARD 12).

Real-world CRM systems don't ship in the Olist dataset, so this script
authors one deterministically: it takes the 15,000 real
``customer_unique_id``/``customer_state`` pairs from
``examples/olist/data/customers.parquet`` (linked via the ``contact_id``
column -- see ``prepare_data.py``) as CRM contacts that also happen to be
shop customers, adds ``--leads`` purely-CRM leads (never became shop
customers), and synthesizes CRM-only attributes (``created_at``, ``segment``,
``marketing_opt_in``) plus a child ``crm_tickets`` table -- all from
authored ground-truth constants (module-level, below) via
``numpy.random.default_rng(seed)``. Fully deterministic given the same
``customers.parquet`` input, ``--leads``, and ``--seed``.

The two-source pattern this feeds (see ``examples/olist/skeleton.yaml`` and
docs/ARCHITECTURE.md §8): ``crm_contacts`` is the *master* root; the shop's
``customers`` table becomes a ``bernoulli``-cardinality child of
``crm_contacts`` that inherits ``customer_state`` via
``generator: parent:state`` -- so refitting this synthesized CRM sample
(via ``verisynth fit``) should recover parameters close to the authored
constants below, demonstrating the fit round-trip.

Usage:
    python examples/olist/prepare_crm_data.py \\
        --data-dir examples/olist/data --out examples/olist/data \\
        [--leads 10000] [--seed 20240817]
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

SEGMENTS = ["consumer", "small_business", "enterprise"]
SEGMENT_PROBS = [0.78, 0.15, 0.07]

CHANNELS = ["email", "chat", "phone", "whatsapp"]
CHANNEL_PROBS = [0.38, 0.27, 0.20, 0.15]

CATEGORIES = [
    "delivery_issue",
    "product_question",
    "return_request",
    "payment_issue",
    "account",
    "other",
]
CATEGORY_PROBS = [0.30, 0.22, 0.16, 0.12, 0.08, 0.12]

PRIORITIES = ["low", "medium", "high", "urgent"]
PRIORITY_PROBS = [0.45, 0.35, 0.15, 0.05]

MARKETING_OPT_IN_P = 0.35

CSAT_SCORES = [1, 2, 3, 4, 5]
CSAT_PROBS = [0.08, 0.07, 0.15, 0.30, 0.40]
CSAT_NULL_RATE = 0.25

RESOLVED_NULL_RATE = 0.08

TICKET_LAM = 0.45
TICKET_MAX = 8

CREATED_AT_START = "2015-01-01T00:00:00"
CREATED_AT_END = "2018-10-01T00:00:00"

OPENED_DELAY_MU = 16.0
OPENED_DELAY_SIGMA = 1.0
RESOLVED_DELAY_MU = 11.3
RESOLVED_DELAY_SIGMA = 1.2

DEFAULT_LEADS = 10000
DEFAULT_SEED = 20240817


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _iso_to_epoch_seconds(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def _epoch_seconds_to_pydatetimes(seconds: np.ndarray) -> list[datetime]:
    us = np.rint(np.asarray(seconds, dtype=np.float64) * 1_000_000).astype(np.int64)
    return us.astype("datetime64[us]").tolist()


def prepare_crm(data_dir: Path, out_dir: Path, leads: int, seed: int) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    customers = pl.read_parquet(data_dir / "customers.parquet")
    real_contact_ids = customers["contact_id"].to_list()
    real_states = customers["customer_state"].to_list()

    # Empirical state distribution of the real 15k -- leads are drawn from
    # this same distribution (a plausible marketing-lead population mirrors
    # the geography of existing customers).
    unique_states, state_counts = np.unique(np.array(real_states), return_counts=True)
    state_probs = state_counts / state_counts.sum()

    lead_ids = [f"lead{i:06d}" for i in range(leads)]
    lead_states = rng.choice(unique_states, size=leads, p=state_probs).tolist() if leads else []

    contact_ids = real_contact_ids + lead_ids
    states = real_states + lead_states
    n_contacts = len(contact_ids)

    start_s = _iso_to_epoch_seconds(CREATED_AT_START)
    end_s = _iso_to_epoch_seconds(CREATED_AT_END)
    created_epoch = rng.uniform(start_s, end_s, size=n_contacts)

    segment = rng.choice(SEGMENTS, size=n_contacts, p=SEGMENT_PROBS).tolist()
    marketing_opt_in = (rng.random(n_contacts) < MARKETING_OPT_IN_P).tolist()

    crm_contacts_df = pl.DataFrame(
        {
            "contact_id": contact_ids,
            "state": states,
            "created_at": _epoch_seconds_to_pydatetimes(created_epoch),
            "segment": segment,
            "marketing_opt_in": marketing_opt_in,
        }
    )

    # --- crm_tickets: per-contact count ~ poisson(TICKET_LAM), clipped. ---
    counts = np.clip(rng.poisson(TICKET_LAM, size=n_contacts), 0, TICKET_MAX)
    total = int(counts.sum())

    contact_ids_arr = np.array(contact_ids, dtype=object)
    ticket_contact_id = np.repeat(contact_ids_arr, counts).tolist()
    ticket_created_epoch = np.repeat(created_epoch, counts)

    channel = rng.choice(CHANNELS, size=total, p=CHANNEL_PROBS).tolist()
    category = rng.choice(CATEGORIES, size=total, p=CATEGORY_PROBS).tolist()
    priority = rng.choice(PRIORITIES, size=total, p=PRIORITY_PROBS).tolist()

    opened_delay = rng.lognormal(mean=OPENED_DELAY_MU, sigma=OPENED_DELAY_SIGMA, size=total)
    opened_epoch = ticket_created_epoch + opened_delay
    opened_at = _epoch_seconds_to_pydatetimes(opened_epoch)

    resolved_delay = rng.lognormal(mean=RESOLVED_DELAY_MU, sigma=RESOLVED_DELAY_SIGMA, size=total)
    resolved_epoch = opened_epoch + resolved_delay
    resolved_null = rng.random(total) < RESOLVED_NULL_RATE
    resolved_at_full = _epoch_seconds_to_pydatetimes(resolved_epoch)
    resolved_at = [None if null else v for v, null in zip(resolved_at_full, resolved_null)]

    csat_choice = rng.choice(CSAT_SCORES, size=total, p=CSAT_PROBS).tolist()
    csat_null = rng.random(total) < CSAT_NULL_RATE
    csat_score = [None if null else v for v, null in zip(csat_choice, csat_null)]

    crm_tickets_df = pl.DataFrame(
        {
            "ticket_id": list(range(total)),
            "contact_id": ticket_contact_id,
            "channel": channel,
            "category": category,
            "priority": priority,
            "opened_at": opened_at,
            "resolved_at": resolved_at,
            "csat_score": csat_score,
        },
        schema_overrides={"csat_score": pl.Int64},
    )

    frames = {"crm_contacts": crm_contacts_df, "crm_tickets": crm_tickets_df}
    counts_out: dict[str, int] = {}
    for name, df in frames.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        counts_out[name] = df.height
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{name}: {df.height} rows -> {path} ({size_mb:.2f} MB)")

    return counts_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="directory with customers.parquet")
    parser.add_argument("--out", required=True, help="output directory for the CRM Parquet files")
    parser.add_argument("--leads", type=int, default=DEFAULT_LEADS, help="synthetic-only CRM leads")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    prepare_crm(Path(args.data_dir), Path(args.out), args.leads, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
