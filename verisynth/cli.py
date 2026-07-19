"""`verisynth generate / validate / fit / scan / init / explain / ingest`.

See docs/ARCHITECTURE.md §8 (normative)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import polars as pl
import yaml

from .backbone import ParquetBackbone, validate_dataset
from .documents import document_path
from .engine import Engine
from .explain import explain_metadata
from .fit import fit_metadata
from .metadata import load_metadata, metadata_to_dict
from .scanner import _load_frames, init_from_dir, render_report, report_to_dict, scan_directory
from .wizard import Chat, WizardAborted, run_wizard
from .xmlstream import DEFAULT_BATCH_ROWS, xml_dir_to_parquet


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

    for tname in metadata.table_order():
        t = metadata.tables[tname]
        if t.format is not None:
            print(f"{tname}: wrote {document_path(args.out, t)}")
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
    scanned: dict[str, pl.DataFrame] | None = None
    for tname in skeleton.table_order():
        path = input_dir / f"{tname}.parquet"
        if path.exists():
            frames[tname] = pl.read_parquet(path)
            continue
        # Fallback: the scanner's loaders cover .csv/.json/.jsonl/.xml files
        # and extract nested entity collections / payload columns into flat
        # tables -- so a directory of nested documents can be fitted
        # directly (XML tables are subject to VERISYNTH_XML_SCAN_ROWS).
        if scanned is None:
            try:
                scanned = _load_frames(input_dir)
            except FileNotFoundError:
                scanned = {}
        if tname in scanned:
            frames[tname] = scanned[tname]
            continue
        print(f"fit: missing input file for table {tname!r}: {path}", file=sys.stderr)
        return 1

    fitted = fit_metadata(frames, skeleton, epsilon=args.epsilon, dp_seed=args.dp_seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(metadata_to_dict(fitted), f, sort_keys=False)

    print(f"fit: wrote {out_path}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        report = scan_directory(args.input)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, default=str))
    else:
        print(render_report(report))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ingest: input not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) / args.table
    infer_types = not args.no_infer_types

    try:
        if input_path.is_file():
            if input_path.suffix != ".xml":
                print(f"ingest: file input must be .xml (got {input_path})", file=sys.stderr)
                return 1
            # xml_dir_to_parquet ingests a *directory* of files; for a single
            # file, stage it alone in a throwaway dir so the same
            # parallel/type-inference code path applies without touching
            # any sibling files (we may not edit xmlstream.py).
            with tempfile.TemporaryDirectory() as tmp:
                link = Path(tmp) / input_path.name
                try:
                    os.symlink(input_path.resolve(), link)
                except OSError:
                    shutil.copy(input_path, link)
                total = xml_dir_to_parquet(
                    tmp,
                    out_dir,
                    workers=args.workers,
                    batch_rows=args.batch_rows,
                    infer_types=infer_types,
                )
        else:
            total = xml_dir_to_parquet(
                input_path,
                out_dir,
                workers=args.workers,
                batch_rows=args.batch_rows,
                infer_types=infer_types,
            )
    except FileNotFoundError as e:
        print(f"ingest: {e}", file=sys.stderr)
        return 1

    n_parts = len(list(out_dir.glob("*.parquet")))
    print(f"ingest: {total} records, {n_parts} parquet file(s) -> {out_dir}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    doc = explain_metadata(metadata)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(doc)
        print(f"wrote {out_path}")
    else:
        print(doc)
    return 0


def _parse_sources(raw: list[str] | None) -> list[tuple[str, str]] | None:
    if not raw:
        return None
    out = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--source must be NAME=PATTERN (got {item!r})")
        name, pattern = item.split("=", 1)
        out.append((name, pattern))
    return out


def _cmd_init(args: argparse.Namespace) -> int:
    try:
        sources = _parse_sources(args.source)
    except ValueError as e:
        print(f"init: {e}", file=sys.stderr)
        return 1

    if args.yes:
        # Non-interactive path: deterministic structural inference (TASK
        # CARD 16) over real data -- no chat, no defaults to accept.
        if not args.input:
            print("init: --yes requires --input <data dir>", file=sys.stderr)
            return 1
        seed = args.seed if args.seed is not None else 42
        try:
            warnings = init_from_dir(args.input, args.out, seed=seed, sources=sources)
        except FileNotFoundError as e:
            print(f"init: {e}", file=sys.stderr)
            return 1
        md = load_metadata(args.out)
        for tname in sorted(md.tables):
            t = md.tables[tname]
            wcount = sum(1 for w in warnings if tname in w)
            print(
                f"{tname}: role={t.role} parent={t.parent} pk={t.primary_key} "
                f"columns={len(t.columns)} warnings={wcount}"
            )
        for w in warnings:
            print(f"warning: {w}")
        return 0

    chat = Chat(assume_yes=args.yes)
    try:
        return run_wizard(args.out, input_dir=args.input, seed=args.seed, chat=chat, sources=sources)
    except (WizardAborted, FileNotFoundError) as e:
        print(f"init: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninit: cancelled", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verisynth",
        epilog=(
            "Data ingestion: `verisynth ingest` streams XML file(s) into a Parquet "
            "staging dataset that `scan`/`fit` then read out-of-core."
        ),
    )
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

    p_scan = sub.add_parser(
        "scan", help="scan real data files and report detected keys, relations, cardinality"
    )
    p_scan.add_argument(
        "--input",
        required=True,
        help=(
            "dir with {table}.parquet/.csv/.json/.jsonl/.xml files, and/or "
            "{table}/ subdirectories of *.xml files loaded as one table"
        ),
    )
    p_scan.add_argument("--json", action="store_true", help="emit the report as JSON")
    p_scan.set_defaults(func=_cmd_scan)

    p_init = sub.add_parser(
        "init", help="build a metadata skeleton through an interactive chat"
    )
    p_init.add_argument("-o", "--out", required=True, help="path for the skeleton YAML")
    p_init.add_argument(
        "--input", default=None, help="optional data dir to scan for suggested answers"
    )
    p_init.add_argument("--seed", type=int, default=None)
    p_init.add_argument(
        "-y", "--yes", action="store_true", help="accept every suggestion (non-interactive)"
    )
    p_init.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="NAME=PATTERN",
        help="assign table source by fnmatch pattern (repeatable, first match wins)",
    )
    p_init.set_defaults(func=_cmd_init)

    p_explain = sub.add_parser(
        "explain",
        help="render a metadata document as a plain-language Markdown explanation",
    )
    p_explain.add_argument("-m", "--metadata", required=True)
    p_explain.add_argument(
        "-o", "--out", default=None, help="write to this .md file instead of stdout"
    )
    p_explain.set_defaults(func=_cmd_explain)

    p_ingest = sub.add_parser(
        "ingest",
        help="stream one XML file (or a directory of them) into a Parquet staging dataset",
    )
    p_ingest.add_argument(
        "--input", required=True, help="a single .xml file, or a directory of .xml files"
    )
    p_ingest.add_argument("--out", required=True, help="staging directory root")
    p_ingest.add_argument(
        "--table", required=True, help="logical table name (written under out/NAME)"
    )
    p_ingest.add_argument(
        "--workers",
        type=int,
        default=None,
        help="parallel worker processes for directory input (default: cpu count)",
    )
    p_ingest.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=f"records per streamed batch/parquet part (default: {DEFAULT_BATCH_ROWS})",
    )
    p_ingest.add_argument(
        "--no-infer-types",
        action="store_true",
        help="skip the DuckDB type-inference pass; keep all columns as text",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
