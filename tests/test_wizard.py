"""Acceptance tests for verisynth.wizard (`verisynth init` chat) and the
scan/init CLI subcommands.

The conversation is scripted through a Chat subclass that answers by prompt
substring, so tests stay robust to cosmetic wording changes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from verisynth.metadata import load_metadata
from verisynth.wizard import Chat, WizardAborted, run_wizard

N_CUSTOMERS = 200


class ScriptChat(Chat):
    """Answers questions by (prompt-substring, answer) rules; otherwise the
    default is accepted -- i.e. the user just presses Enter."""

    def __init__(self, rules: list[tuple[str, object]] | None = None):
        super().__init__(input_fn=lambda p: "", print_fn=self.lines_append)
        self.rules = rules or []
        self.lines: list[str] = []

    def lines_append(self, s: str) -> None:
        self.lines.append(s)

    def ask(self, prompt: str, default: str | None = None) -> str:
        for sub, answer in self.rules:
            if sub in prompt:
                if isinstance(answer, list):
                    if not answer:
                        continue
                    return str(answer.pop(0))
                return str(answer)
        return default if default is not None else ""

    def transcript(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(3)
    pl.DataFrame(
        {
            "customer_id": np.arange(N_CUSTOMERS, dtype=np.int64),
            "country": rng.choice(["NL", "BE"], size=N_CUSTOMERS),
        }
    ).write_parquet(tmp_path / "customers.parquet")

    counts = rng.poisson(1.2, N_CUSTOMERS)
    fk = np.repeat(np.arange(N_CUSTOMERS, dtype=np.int64), counts)
    pl.DataFrame(
        {
            "order_id": np.arange(len(fk), dtype=np.int64),
            "customer_id": fk,
            "product_id": rng.integers(0, 20, len(fk)),
            "amount": np.exp(rng.normal(3.0, 0.5, len(fk))),
        }
    ).write_parquet(tmp_path / "orders.parquet")

    pl.DataFrame(
        {
            "product_id": np.arange(20, dtype=np.int64),
            "price": np.exp(rng.normal(2.0, 0.3, 20)),
        }
    ).write_parquet(tmp_path / "products.parquet")
    return tmp_path


def test_scan_mode_all_defaults(data_dir, tmp_path):
    out = tmp_path / "skeleton.yaml"
    chat = ScriptChat()
    assert run_wizard(out, input_dir=data_dir, seed=42, chat=chat) == 0

    md = load_metadata(out)
    assert md.seed == 42

    cust = md.tables["customers"]
    assert (cust.role, cust.primary_key, cust.rows) == ("root", "customer_id", N_CUSTOMERS)
    assert cust.columns["country"].distribution.kind == "categorical"

    orders = md.tables["orders"]
    assert (orders.role, orders.parent) == ("child", "customers")
    assert orders.columns["customer_id"].generator == "parent_key"
    assert orders.cardinality.kind == "poisson"
    assert orders.child_stride > orders.cardinality.params["max"]

    # The secondary FK became a dimension reference into the products root.
    product_ref = orders.columns["product_id"]
    assert product_ref.reference == "products"
    assert product_ref.distribution.kind == "zipf"
    assert product_ref.distribution.params["n"] == 20

    assert "Hi!" in chat.transcript()


def test_scan_mode_overrides(data_dir, tmp_path):
    out = tmp_path / "skeleton.yaml"
    chat = ScriptChat(
        rules=[
            ("amount (float64)", "skip"),
            ("dimension reference", "n"),
            ("Which one is orders's parent?", "customers"),
        ]
    )
    assert run_wizard(out, input_dir=data_dir, seed=1, chat=chat) == 0

    md = load_metadata(out)
    orders = md.tables["orders"]
    assert "amount" not in orders.columns
    # Declining the reference demotes product_id to a plain column.
    assert orders.columns["product_id"].reference is None
    assert orders.columns["product_id"].distribution is not None


def test_scan_mode_reject_relation_makes_root(data_dir, tmp_path):
    out = tmp_path / "skeleton.yaml"
    chat = ScriptChat(rules=[("looks like a child of customers", "n")])
    # orders still has the products relation, answered with default elsewhere.
    assert run_wizard(out, input_dir=data_dir, seed=1, chat=chat) == 0
    # Both single-relation children (none here besides orders/customers pair)
    md = load_metadata(out)
    assert md.tables["customers"].role == "root"


def test_scratch_mode(tmp_path):
    out = tmp_path / "scratch.yaml"
    chat = ScriptChat(
        rules=[
            ("Which tables do you want", "customers, orders"),
            ("Is orders a child of another table?", "y"),
            ("holding the customers key", "customer_id"),
            ("How many children does each parent get?", "poisson"),
            ("Average children per parent?", "2.5"),
            ("Hard maximum?", "12"),
            # customers columns: one data column, then stop.
            ("Next column?", ["age int64", "", "amount float64", ""]),
            ("amount (float64)", "lognormal"),
        ]
    )
    assert run_wizard(out, seed=5, chat=chat) == 0

    md = load_metadata(out)
    cust = md.tables["customers"]
    assert (cust.role, cust.primary_key) == ("root", "customer_id")
    assert cust.columns["age"].distribution.kind == "uniform_int"

    orders = md.tables["orders"]
    assert (orders.role, orders.parent) == ("child", "customers")
    assert orders.columns["customer_id"].generator == "parent_key"
    assert orders.cardinality.kind == "poisson"
    assert orders.cardinality.params == {"lam": 2.5, "max": 12}
    assert orders.child_stride == 16
    assert orders.columns["amount"].distribution.kind == "lognormal"


def test_assume_yes_needs_a_default():
    chat = Chat(input_fn=lambda p: "", print_fn=lambda s: None, assume_yes=True)
    with pytest.raises(WizardAborted):
        chat.ask("What now?", None)


def test_choose_accepts_numbers():
    answers = iter(["9", "2"])  # out of range, then valid
    chat = Chat(input_fn=lambda p: next(answers), print_fn=lambda s: None)
    assert chat.choose("Pick one", ["alpha", "beta"], "alpha") == "beta"


# --------------------------------------------------------------------------
# CLI end-to-end: scan + init --yes -> generate -> validate
# --------------------------------------------------------------------------


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verisynth.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_scan_and_init_e2e(data_dir, tmp_path):
    result = _run("scan", "--input", str(data_dir))
    assert result.returncode == 0, result.stderr
    assert "customers" in result.stdout
    assert "fk -> customers.customer_id" in result.stdout

    result = _run("scan", "--input", str(data_dir), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["tables"]["orders"]["primary_key"] == "order_id"

    skeleton = tmp_path / "skeleton.yaml"
    result = _run("init", "--input", str(data_dir), "--yes", "-o", str(skeleton), "--seed", "9")
    assert result.returncode == 0, result.stderr
    # `--yes` is the non-interactive, deterministic inference path (TASK
    # CARD 16): it prints a structural summary, not a chat transcript.
    assert "role=" in result.stdout
    assert "pk=" in result.stdout

    out_dir = tmp_path / "out"
    result = _run("generate", "-m", str(skeleton), "-o", str(out_dir), "--seed", "9")
    assert result.returncode == 0, result.stderr

    result = _run("validate", "-m", str(skeleton), "-o", str(out_dir))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_cli_scan_missing_dir(tmp_path):
    result = _run("scan", "--input", str(tmp_path / "nope"))
    assert result.returncode == 1
    assert "not a directory" in result.stderr
