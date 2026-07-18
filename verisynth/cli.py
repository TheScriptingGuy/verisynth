"""`verisynth generate / validate / fit`. See docs/ARCHITECTURE.md §8 (normative)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl
import yaml

from .backbone import ParquetBackbone, validate_dataset
from .engine import Engine
from .fit import fit_metadata
from .metadata import load_metadata, metadata_to_dict


def _cmd_generate(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    engine = Engine(metadata, seed=args.seed)
    engine.generate(args.out, num_partitions=args.partitions)

    backbone = ParquetBackbone(args.out)
    con = duckdb.connect()
    try:
        for tname in metadata.table_order():
            glob = backbone.table_glob(tname, metadata.tables[tname].source)
            (count,) = con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()
            print(f"{tname}: {count} rows")
    finally:
        con.close()
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    violations = validate_dataset(metadata, args.out)
    if violations:
        for v in violations:
            print(v)
        return 1
    print("OK")
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    skeleton = load_metadata(args.metadata)
    input_dir = Path(args.input)

    frames: dict[str, pl.DataFrame] = {}
    for tname in skeleton.table_order():
        path = input_dir / f"{tname}.parquet"
        if not path.exists():
            print(f"fit: missing input file for table {tname!r}: {path}", file=sys.stderr)
            return 1
        frames[tname] = pl.read_parquet(path)

    fitted = fit_metadata(frames, skeleton, epsilon=args.epsilon, dp_seed=args.dp_seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(metadata_to_dict(fitted), f, sort_keys=False)

    print(f"fit: wrote {out_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="verisynth")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate a synthetic dataset")
    p_gen.add_argument("-m", "--metadata", required=True)
    p_gen.add_argument("-o", "--out", required=True)
    p_gen.add_argument("--partitions", type=int, default=1)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=_cmd_generate)

    p_val = sub.add_parser("validate", help="validate a generated dataset")
    p_val.add_argument("-m", "--metadata", required=True)
    p_val.add_argument("-o", "--out", required=True)
    p_val.set_defaults(func=_cmd_validate)

    p_fit = sub.add_parser("fit", help="fit metadata parameters from real data")
    p_fit.add_argument("--input", required=True)
    p_fit.add_argument("-m", "--metadata", required=True)
    p_fit.add_argument("-o", "--out", required=True)
    p_fit.add_argument("--epsilon", type=float, default=None)
    p_fit.add_argument("--dp-seed", type=int, default=0)
    p_fit.set_defaults(func=_cmd_fit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
