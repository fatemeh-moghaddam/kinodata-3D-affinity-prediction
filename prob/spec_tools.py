"""
How a spec in prob_config is filled in. prob_config says *what* the settings are;
this module holds the one mechanism every stage uses to resolve them:

    command line  >  config  >  environment variable  >  default

- A spec is a frozen dataclass whose fields are levels (data, model, ...), each
  itself a frozen dataclass.
- A level's lower-case fields are settings: chosen per run, one default each.
- A level's UPPER CASE class constants are fixed values: no run can change them,
  and naming one on the command line is an error rather than a silent no-op.
- resolve() returns the spec plus, for every setting, where its value came from;
  spec_record() turns both into the JSON block a manifest stores.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────
# Value parsers: accept the typed value (config, tests) or its text (CLI, env)
# ─────────────────────────────────────────────────────────────

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"expected 1/0/true/false/yes/no, got {value!r}")


def parse_int_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    """"0,3" or [0, 3] -> (0, 3); empty -> None."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts) or None
    return tuple(int(v) for v in value) or None


def parse_names(value: Any) -> Tuple[str, ...]:
    """"mlp, random_forest" or a list -> ("mlp", "random_forest"); empty -> ()."""
    if isinstance(value, str):
        return tuple(p.strip() for p in value.split(",") if p.strip())
    return tuple(value or ())


def parse_optional_str(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "") else None


def parse_optional_int(value: Any) -> Optional[int]:
    return int(value) if value not in (None, "") else None


def parse_rmsd(value: Any) -> Optional[float]:
    """"2" / "2.0" / 2 -> 2, "none" -> None (the unfiltered dataset)."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none"}):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


# ─────────────────────────────────────────────────────────────
# Settings table
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Setting:
    """One per-run setting: its level, its field name (= its --flag), how to
    parse it, its PROB_*-style environment alias if any, and whether a run must
    name it (no default)."""

    level: str
    name: str
    parse: Callable[[Any], Any]
    env: Optional[str] = None
    required: bool = False


def add_setting_flags(parser: argparse.ArgumentParser, settings: Sequence[Setting]) -> None:
    """--<name> for every setting; omitted flags stay absent (argparse.SUPPRESS)."""
    for setting in settings:
        parser.add_argument(
            f"--{setting.name}", type=str, default=argparse.SUPPRESS,
            help=f"[{setting.level}]" + (f" env {setting.env}" if setting.env else ""),
        )


def fixed_values(level_cls: type) -> Dict[str, Any]:
    """A level's fixed values (its UPPER CASE class constants)."""
    return {
        name: getattr(level_cls, name)
        for name in dir(level_cls)
        if name.isupper() and not name.startswith("_")
    }


def reject_fixed_flags(argv: Optional[Sequence[str]], level_classes: Mapping[str, type]) -> None:
    """A fixed value named on the command line (e.g. --include_val) is an error."""
    if not argv:
        return
    given = {a.split("=", 1)[0].lower() for a in argv if a.startswith("--")}
    for level_cls in level_classes.values():
        for name, value in fixed_values(level_cls).items():
            if f"--{name.lower()}" in given:
                raise ValueError(
                    f"--{name.lower()} is fixed: {level_cls.__name__}.{name} = {value!r}. "
                    "It is part of the method, not a run setting; change it in "
                    "prob/prob_config.py, deliberately, or not at all."
                )


# ─────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────

def resolve(
    spec_cls: type,
    level_classes: Mapping[str, type],
    settings: Sequence[Setting],
    *,
    cli: Mapping[str, Any],
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    env_prefix: Optional[str] = None,
    given: Optional[Mapping[Tuple[str, str], Tuple[Any, str]]] = None,
) -> Tuple[Any, Dict[str, str]]:
    """
    Build spec_cls from the settings table: command line > config > environment
    variable > default, for every setting alike.

    cli: settings named on the command line (name -> text). config: name -> value.
    environ: the environment. env_prefix: variables with this prefix that no
    setting declares are an error (a typo'd switch fails instead of being ignored).
    given: values the caller already resolved, {(level, name): (value, source)}.

    Returns (spec, sources) with sources["<level>.<name>"] in
    "cli" / "config" / "env <VAR>=<value>" / "default" (or the given source).
    """
    if env_prefix:
        declared = {s.env for s in settings if s.env}
        unknown = sorted(k for k in environ if k.startswith(env_prefix) and k not in declared)
        if unknown:
            raise ValueError(f"Unknown environment variable(s) {unknown}; known: {sorted(declared)}")

    values: Dict[str, Dict[str, Any]] = {level: {} for level in level_classes}
    sources: Dict[str, str] = {}
    for (level, name), (value, source) in (given or {}).items():
        values[level][name] = value
        sources[f"{level}.{name}"] = source

    for setting in settings:
        key = f"{setting.level}.{setting.name}"
        try:
            if setting.name in cli:
                value, source = setting.parse(cli[setting.name]), "cli"
            elif setting.name in config:
                value, source = setting.parse(config[setting.name]), "config"
            elif setting.env and setting.env in environ:
                raw = environ[setting.env]
                value, source = setting.parse(raw), f"env {setting.env}={raw!r}"
            elif setting.required:
                raise ValueError("required, but not given (pass --%s)" % setting.name)
            else:
                sources[key] = "default"
                continue
        except ValueError as err:
            raise ValueError(f"{key}: {err}") from err
        values[setting.level][setting.name] = value
        sources[key] = source

    spec = spec_cls(**{level: cls(**values[level]) for level, cls in level_classes.items()})
    return spec, sources


def spec_record(spec: Any, sources: Mapping[str, str]) -> Dict[str, Any]:
    """The spec as a manifest stores it: per level, every setting with its value
    and source, and every fixed value."""
    record = {}
    for level_field in fields(spec):
        level = level_field.name
        settings = getattr(spec, level)
        record[level] = {
            "settings": {
                f.name: {
                    "value": getattr(settings, f.name),
                    "source": sources.get(f"{level}.{f.name}", "default"),
                }
                for f in fields(settings)
            },
            "fixed": fixed_values(type(settings)),
        }
    return record
