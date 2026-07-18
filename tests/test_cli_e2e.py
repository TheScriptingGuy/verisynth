"""End-to-end CLI acceptance tests: `verisynth generate / validate / fit`.

See docs/ARCHITECTURE.md §8 (normative) and TASK CARD 8.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from verisynth.metadata import load_metadata, metadata_to_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
RETAIL_YAML = REPO_ROOT / "examples" / "retail.yaml"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "verisynth.cli", *args],
        capture_output=True,
        text=True,
    )


def _shrunk_metadata_dict(rows: int = 200) -> dict:
    md = load_metadata(RETAIL_YAML)
    md.tables["customers"].rows = rows
    return metadata_to_dict(md)


def test_generate_validate_fit_roundtrip(tmp_path):
    # --- 1. generate -----------------------------------------------------------
    meta_dict = _shrunk_metadata_dict(rows=200)
    metadata_path = tmp_path / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.safe_dump(meta_dict, f, sort_keys=False)

    out_dir = tmp_path / "out"
    result = _run(
        "generate", "-m", str(metadata_path), "-o", str(out_dir), "--partitions", "2", "--seed", "42"
    )
    assert result.returncode == 0, result.stderr

    for tname in ("customers", "orders"):
        files = list((out_dir / tname).glob("*.parquet"))
        assert len(files) == 2, f"expected 2 partition files for {tname}, got {files}"

    # --- 2. validate -------------------------------------------------------------
    result = _run("validate", "-m", str(metadata_path), "-o", str(out_dir))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout

    # --- 3. fit --------------------------------------------------------------------
    fit_input_dir = tmp_path / "fit_input"
    fit_input_dir.mkdir()
    for tname in ("customers", "orders"):
        df = pl.read_parquet(str(out_dir / tname / "*.parquet"))
        df.write_parquet(fit_input_dir / f"{tname}.parquet")

    fitted_path = tmp_path / "fitted.yaml"
    result = _run(
        "fit", "--input", str(fit_input_dir), "-m", str(metadata_path), "-o", str(fitted_path)
    )
    assert result.returncode == 0, result.stderr
    assert fitted_path.exists()

    fitted_md = load_metadata(fitted_path)

    fitted_out_dir = tmp_path / "out_fitted"
    result = _run(
        "generate", "-m", str(fitted_path), "-o", str(fitted_out_dir), "--partitions", "1"
    )
    assert result.returncode == 0, result.stderr

    result = _run("validate", "-m", str(fitted_path), "-o", str(fitted_out_dir))
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout

    # --- 4. fit with --epsilon -------------------------------------------------------
    dp_meta_dict = _shrunk_metadata_dict(rows=200)
    dp_meta_dict["tables"]["customers"]["columns"]["income"]["clamp"] = [1.0, 1e7]
    dp_meta_dict["tables"]["orders"]["columns"]["order_total"]["clamp"] = [0.1, 1e6]
    dp_metadata_path = tmp_path / "metadata_dp.yaml"
    with open(dp_metadata_path, "w") as f:
        yaml.safe_dump(dp_meta_dict, f, sort_keys=False)

    dp_fitted_path = tmp_path / "fitted_dp.yaml"
    result = _run(
        "fit",
        "--input",
        str(fit_input_dir),
        "-m",
        str(dp_metadata_path),
        "-o",
        str(dp_fitted_path),
        "--epsilon",
        "1.0",
        "--dp-seed",
        "7",
    )
    assert result.returncode == 0, result.stderr
    assert dp_fitted_path.exists()
    load_metadata(dp_fitted_path)  # must still be valid metadata


def test_fit_missing_input_file_errors_clearly(tmp_path):
    meta_dict = _shrunk_metadata_dict(rows=10)
    metadata_path = tmp_path / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.safe_dump(meta_dict, f, sort_keys=False)

    empty_input_dir = tmp_path / "empty_input"
    empty_input_dir.mkdir()

    result = _run(
        "fit",
        "--input",
        str(empty_input_dir),
        "-m",
        str(metadata_path),
        "-o",
        str(tmp_path / "fitted.yaml"),
    )
    assert result.returncode != 0
    assert "customers" in result.stderr or "customers" in result.stdout
