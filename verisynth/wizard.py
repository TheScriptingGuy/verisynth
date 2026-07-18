"""`verisynth init`: a chat-style wizard that builds a metadata skeleton.

The wizard holds a small conversation on the terminal: it asks what tables
exist, which column is the primary key, how tables relate, and how many
children a parent has -- then writes a validated metadata YAML. When pointed
at a directory of real data (``--input``), it first runs the metadata
scanner (scanner.py) and turns every question into a "here's what I found,
keep it?" confirmation, so a whole skeleton can be assembled with Enter
presses (or fully unattended with ``--yes``).

All terminal I/O goes through ``Chat`` so tests can script the conversation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from .metadata import MetadataError, parse_metadata
from .scanner import (
    Relation,
    ScanReport,
    rank_parent_relations,
    scan_directory,
)

_BOT = "verisynth ▸ "
_CONT = "          ▸ "
_USER = "      you ▸ "

# Placeholder params per distribution kind -- a valid starting point that
# `verisynth fit` (or hand-editing) is expected to refine.
_DEFAULT_DIST: dict[str, dict[str, Any]] = {
    "categorical": {"categories": ["a", "b"], "probs": [0.5, 0.5]},
    "normal": {"mean": 0.0, "std": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "uniform": {"low": 0.0, "high": 1.0},
    "exponential": {"rate": 1.0},
    "gamma": {"shape": 1.0, "scale": 1.0},
    "beta": {"a": 1.0, "b": 1.0},
    "uniform_int": {"low": 0, "high": 100},
    "datetime_uniform": {"start": "2020-01-01T00:00:00", "end": "2021-01-01T00:00:00"},
    "zipf": {"a": 1.1, "n": 100},
}

_DEFAULT_KIND_FOR_TYPE = {
    "int64": "uniform_int",
    "float64": "normal",
    "string": "categorical",
    "bool": "categorical",
    "timestamp": "datetime_uniform",
}

_DEFAULT_CARDINALITY: dict[str, dict[str, Any]] = {
    "poisson": {"kind": "poisson", "lam": 1.0, "max": 8},
    "uniform_int": {"kind": "uniform_int", "low": 0, "high": 3, "max": 3},
    "fixed": {"kind": "fixed", "n": 1},
    "bernoulli": {"kind": "bernoulli", "p": 0.5},
}

_COLUMN_TYPES = ("int64", "float64", "string", "bool", "timestamp")


class WizardAborted(Exception):
    """Raised when the conversation cannot continue (EOF without a default)."""


class Chat:
    """Terminal conversation: bot lines out, user answers in.

    ``assume_yes`` answers every question with its default (echoed, so the
    transcript still reads like a conversation).
    """

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
        assume_yes: bool = False,
    ):
        self._input = input_fn
        self._print = print_fn
        self.assume_yes = assume_yes

    def say(self, *lines: str) -> None:
        for i, line in enumerate(lines):
            self._print((_BOT if i == 0 else _CONT) + line)

    def ask(self, prompt: str, default: str | None = None) -> str:
        suffix = f"  [{default}]" if default not in (None, "") else ""
        self.say(prompt + suffix)
        if self.assume_yes:
            if default is None:
                raise WizardAborted(f"--yes needs a default for: {prompt}")
            self._print(_USER + str(default))
            return default
        try:
            raw = self._input(_USER).strip()
        except EOFError:
            if default is None:
                raise WizardAborted(f"input ended at: {prompt}") from None
            raw = ""
        return raw if raw else (default or "")

    def confirm(self, prompt: str, default: bool = True) -> bool:
        hint = "Y/n" if default else "y/N"
        answer = self.ask(f"{prompt} ({hint})", "y" if default else "n")
        return answer.strip().lower() in ("y", "yes", "")

    def ask_int(self, prompt: str, default: int | None = None) -> int:
        while True:
            raw = self.ask(prompt, str(default) if default is not None else None)
            try:
                return int(raw)
            except ValueError:
                self.say(f"Hmm, {raw!r} isn't an integer -- try again?")

    def choose(self, prompt: str, choices: list[str], default: str) -> str:
        """Pick one of ``choices`` by number or name; Enter takes the default."""
        numbered = "  ".join(f"[{i + 1}] {c}" for i, c in enumerate(choices))
        while True:
            raw = self.ask(f"{prompt}  {numbered}", default).strip()
            if raw in choices:
                return raw
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                return choices[int(raw) - 1]
            self.say(f"I didn't catch that -- answer with a number 1-{len(choices)} or a name.")


def _dist_summary(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    params = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "categorical":
        cats = params.get("categories", [])
        shown = ", ".join(str(c) for c in cats[:4])
        more = f", +{len(cats) - 4} more" if len(cats) > 4 else ""
        return f"categorical over {{{shown}{more}}}"
    inner = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{kind}({inner})"


def _ask_distribution(
    chat: Chat, cname: str, ctype: str, suggestion: dict[str, Any] | None
) -> dict[str, Any] | None:
    """One column, one question. Returns a distribution spec or None to skip.

    The user can press Enter to keep the suggestion, name another kind
    (placeholder params, to be fitted later), or say "skip".
    """
    if suggestion is None:
        kind = _DEFAULT_KIND_FOR_TYPE[ctype]
        suggestion = {"kind": kind, **_DEFAULT_DIST[kind]}
        lead = f"{cname} ({ctype}): I have no good guess, defaulting to {_dist_summary(suggestion)}."
    else:
        lead = f"{cname} ({ctype}): looks like {_dist_summary(suggestion)}."
    while True:
        answer = (
            chat.ask(f"{lead} Keep it, name another kind, or say 'skip'.", "keep")
            .strip()
            .lower()
        )
        if answer in ("keep", "y", "yes", ""):
            return suggestion
        if answer in ("skip", "drop", "no", "n"):
            return None
        if answer in _DEFAULT_DIST:
            return {"kind": answer, **_DEFAULT_DIST[answer]}
        chat.say(f"I know these kinds: {', '.join(sorted(_DEFAULT_DIST))} -- or 'skip'.")
        lead = f"{cname} ({ctype}):"


def _ask_cardinality(chat: Chat, suggestion: dict[str, Any] | None) -> dict[str, Any]:
    if suggestion is not None:
        card = ", ".join(f"{k}={v}" for k, v in suggestion.items() if k != "kind")
        if chat.confirm(
            f"The counts look {suggestion['kind']} ({card}). Sound right?", default=True
        ):
            return dict(suggestion)
    kind = chat.choose(
        "How many children does each parent get?",
        list(_DEFAULT_CARDINALITY),
        "poisson",
    )
    spec = dict(_DEFAULT_CARDINALITY[kind])
    if kind == "poisson":
        spec["lam"] = float(chat.ask("Average children per parent?", "1.0"))
        spec["max"] = chat.ask_int("Hard maximum?", max(int(spec["lam"] * 4), 4))
    elif kind == "uniform_int":
        spec["low"] = chat.ask_int("Minimum children?", 0)
        spec["high"] = chat.ask_int("Maximum children?", 3)
        spec["max"] = spec["high"]
    elif kind == "fixed":
        spec["n"] = chat.ask_int("Exactly how many children?", 1)
    else:  # bernoulli
        spec["p"] = float(chat.ask("Probability a parent has a child (0-1)?", "0.5"))
    return spec


def _cardinality_eff_max(spec: dict[str, Any]) -> int:
    if spec["kind"] == "bernoulli":
        return 1
    if spec["kind"] == "fixed":
        return int(spec["n"])
    return int(spec["max"])


def _child_stride_for(spec: dict[str, Any]) -> int:
    return 1 << max(_cardinality_eff_max(spec), 1).bit_length()


# --------------------------------------------------------------------------
# Scan-backed flow
# --------------------------------------------------------------------------


def _choose_parents(chat: Chat, report: ScanReport) -> dict[str, Relation | None]:
    """For every table, settle which inbound relation (if any) is its parent."""
    parents: dict[str, Relation | None] = {}
    for tname in report.tables:
        rels = rank_parent_relations(tname, report.relations_of(tname))
        if not rels:
            parents[tname] = None
            continue
        if len(rels) == 1:
            r = rels[0]
            keep = chat.confirm(
                f"{tname} looks like a child of {r.parent} via {r.child_column} "
                f"({r.coverage:.0%} of values match). Keep that?",
                default=True,
            )
            parents[tname] = r if keep else None
            continue
        options = [r.parent for r in rels] + ["none (it's a root table)"]
        chat.say(
            f"{tname} references several tables: "
            + "; ".join(f"{r.parent} via {r.child_column} ({r.coverage:.0%})" for r in rels)
        )
        picked = chat.choose(
            f"Which one is {tname}'s parent?", options, rels[0].parent
        )
        parents[tname] = next((r for r in rels if r.parent == picked), None)
    return parents


def _topo_order(chat: Chat, parents: dict[str, Relation | None]) -> list[str]:
    """Roots first, parents before children; cycle members are demoted to roots."""
    order: list[str] = []
    placed: set[str] = set()
    remaining = dict(parents)
    while remaining:
        ready = [
            t
            for t, r in remaining.items()
            if r is None or r.parent in placed or r.parent not in parents
        ]
        if not ready:
            demoted = sorted(remaining)[0]
            chat.say(
                f"Heads up: {demoted} is caught in a relation cycle -- "
                "I'm treating it as a root table."
            )
            remaining[demoted] = None
            parents[demoted] = None
            continue
        for t in ready:
            order.append(t)
            placed.add(t)
            del remaining[t]
    return order


def _build_table_from_scan(
    chat: Chat,
    report: ScanReport,
    tname: str,
    parents: dict[str, Relation | None],
) -> dict[str, Any]:
    tscan = report.tables[tname]
    parent_rel = parents[tname]
    role = "child" if parent_rel is not None else "root"
    chat.say("", f"--- {tname} ({role}, {tscan.rows} rows) ---")

    if tscan.pk_candidates:
        pk = chat.ask(
            f"Primary key of {tname}? I'd go with {tscan.pk!r}"
            + (
                f" (other unique columns: {', '.join(tscan.pk_candidates[1:])})"
                if len(tscan.pk_candidates) > 1
                else ""
            ),
            tscan.pk,
        )
    else:
        chat.say(
            f"No column in {tname} is unique and non-null, "
            "so I'll mint a fresh key column."
        )
        default_pk = f"{tname[:-1] if tname.endswith('s') else tname}_id"
        pk = chat.ask(f"What should {tname}'s primary key column be called?", default_pk)

    columns: dict[str, Any] = {pk: {"type": "int64", "generator": "key"}}
    if tscan.columns.get(pk) is not None and tscan.columns[pk].type != "int64":
        chat.say(
            f"(Your real {pk} is {tscan.columns[pk].type}; generated keys are int64.)"
        )

    spec: dict[str, Any] = {"role": role, "primary_key": pk}

    if role == "child":
        spec["parent"] = parent_rel.parent
        columns[parent_rel.child_column] = {"type": "int64", "generator": "parent_key"}
        card = _ask_cardinality(chat, parent_rel.cardinality)
        spec["cardinality"] = card
        spec["child_stride"] = _child_stride_for(card)
    else:
        spec["rows"] = chat.ask_int(f"How many rows should {tname} get?", tscan.rows)

    # Non-parent relations to root tables become dimension references.
    reference_cols = {}
    for r in report.relations_of(tname):
        if parent_rel is not None and r is parent_rel:
            continue
        if r.child_column in columns:
            continue
        if parents.get(r.parent) is not None:
            continue  # reference target must be a root table
        if chat.confirm(
            f"{r.child_column} also points at {r.parent} ({r.coverage:.0%} match) -- "
            f"model it as a dimension reference with zipf popularity?",
            default=True,
        ):
            reference_cols[r.child_column] = {
                "type": "int64",
                "distribution": {"kind": "zipf", "a": 1.1, "n": report.tables[r.parent].rows},
                "reference": r.parent,
            }

    for cname, col in tscan.columns.items():
        if cname in columns or cname in reference_cols:
            continue
        dist = _ask_distribution(chat, cname, col.type, col.suggestion)
        if dist is None:
            chat.say(f"Okay, leaving {cname} out.")
            continue
        cspec: dict[str, Any] = {"type": col.type, "distribution": dist}
        if col.null_rate > 0:
            cspec["null_rate"] = col.null_rate
        columns[cname] = cspec

    columns.update(reference_cols)
    spec["columns"] = columns
    return spec


# --------------------------------------------------------------------------
# From-scratch flow
# --------------------------------------------------------------------------


def _build_table_from_scratch(
    chat: Chat, tname: str, existing: list[str]
) -> dict[str, Any]:
    chat.say("", f"--- {tname} ---")
    parent = None
    if existing and chat.confirm(f"Is {tname} a child of another table?", default=False):
        parent = chat.choose(f"Which table is {tname}'s parent?", existing, existing[0])

    default_pk = f"{tname[:-1] if tname.endswith('s') else tname}_id"
    pk = chat.ask(f"What's the primary key column of {tname}?", default_pk)
    columns: dict[str, Any] = {pk: {"type": "int64", "generator": "key"}}
    spec: dict[str, Any] = {"role": "child" if parent else "root", "primary_key": pk}

    if parent:
        spec["parent"] = parent
        fk_default = f"{parent[:-1] if parent.endswith('s') else parent}_id"
        fk = chat.ask(f"What's the column holding the {parent} key?", fk_default)
        columns[fk] = {"type": "int64", "generator": "parent_key"}
        card = _ask_cardinality(chat, None)
        spec["cardinality"] = card
        spec["child_stride"] = _child_stride_for(card)
    else:
        spec["rows"] = chat.ask_int(f"How many rows should {tname} get?", 1000)

    chat.say(
        "Now the data columns. Give me one per line as 'name type' "
        f"(types: {', '.join(_COLUMN_TYPES)}); empty line when you're done."
    )
    while True:
        raw = chat.ask("Next column?", "")
        if not raw:
            break
        parts = raw.split()
        cname = parts[0]
        ctype = parts[1] if len(parts) > 1 else "float64"
        if ctype not in _COLUMN_TYPES:
            chat.say(f"{ctype!r} isn't a type I know -- using float64.")
            ctype = "float64"
        dist = _ask_distribution(chat, cname, ctype, None)
        if dist is not None:
            columns[cname] = {"type": ctype, "distribution": dist}

    spec["columns"] = columns
    return spec


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_wizard(
    out_path: str | Path,
    input_dir: str | Path | None = None,
    seed: int | None = None,
    chat: Chat | None = None,
) -> int:
    chat = chat or Chat()
    chat.say(
        "Hi! Let's put together a metadata skeleton for your dataset.",
        "Press Enter to accept any suggestion I make.",
    )

    report: ScanReport | None = None
    if input_dir is not None:
        report = scan_directory(input_dir)
        n_rel = len(report.relations)
        chat.say(
            f"I scanned {input_dir} and found {len(report.tables)} table(s)"
            + (f" and {n_rel} relation(s)." if n_rel else "."),
        )
        for r in report.relations:
            chat.say(
                f"  {r.parent} 1--N {r.child} via {r.child_column} "
                f"(avg {r.mean_children} children, max {r.max_children})"
            )

    if seed is None:
        seed = chat.ask_int("What seed should generation use?", 0)

    tables: dict[str, Any] = {}
    if report is not None:
        parents = _choose_parents(chat, report)
        for tname in _topo_order(chat, parents):
            tables[tname] = _build_table_from_scan(chat, report, tname, parents)
    else:
        names_raw = chat.ask(
            "Which tables do you want (comma-separated names, parents before children)?",
            None,
        )
        names = [n.strip() for n in names_raw.split(",") if n.strip()]
        for tname in names:
            tables[tname] = _build_table_from_scratch(
                chat, tname, [n for n in names if n != tname and n in tables]
            )

    doc = {"version": 1, "seed": seed, "tables": tables}

    valid = True
    try:
        parse_metadata(doc)
    except MetadataError as e:
        valid = False
        chat.say(
            f"Heads up -- the skeleton doesn't validate yet: {e}",
            "I'll save it anyway so you can fix it by hand.",
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)

    if valid:
        chat.say(
            f"All checks pass -- wrote {out} with {len(tables)} table(s).",
            "Next steps: refine parameters with `verisynth fit`, then `verisynth generate`.",
            "(Temporal chains, copulas and derived columns can be added by hand -- "
            "see docs/ARCHITECTURE.md §2.)",
        )
        return 0
    chat.say(f"Wrote {out} (needs a manual fix before it can generate).")
    return 1
