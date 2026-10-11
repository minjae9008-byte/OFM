"""JSON schemas for model replies and a small validator for them.

Local servers can constrain decoding with a JSON schema (Ollama ``format``,
OpenAI-style ``response_format``; both compile the schema to a grammar), but
not every server or model honours it, so replies are validated again here.
Only the subset of JSON Schema (draft 2020-12) that our own schemas use is
implemented: ``type`` (single or list), ``enum``, ``properties``,
``required``, ``additionalProperties: false``, ``items``, ``minItems`` /
``maxItems``, ``minLength`` / ``maxLength`` and ``minimum`` / ``maximum``;
numbers must also be finite (JSON parsers accept ``NaN`` and ``Infinity``).
Error messages carry a JSON path so they can be quoted back to the model in
a repair turn ("$.actions[2].region: 'E5' is not one of ...").
"""
from __future__ import annotations

import math
from typing import Any, Iterable

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _short(v: Any, n: int = 40) -> str:
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def validate(instance: Any, schema: dict, path: str = "$", max_errors: int = 20) -> list[str]:
    """Return a list of human-readable violations (empty when ``instance`` conforms)."""
    errors: list[str] = []
    _validate(instance, schema, path, errors, max_errors)
    return errors[:max_errors]


def _validate(v: Any, s: dict, path: str, errors: list[str], limit: int) -> None:
    if len(errors) >= limit or not isinstance(s, dict):
        return
    t = s.get("type")
    if t is not None:
        types = [t] if isinstance(t, str) else list(t)
        if not any(_TYPES.get(name, lambda _: True)(v) for name in types):
            errors.append(f"{path}: expected {' or '.join(types)}, got {type(v).__name__}")
            return
    if "enum" in s and v not in s["enum"]:
        opts = ", ".join(_short(o, 16) for o in s["enum"][:12]) + (" ..." if len(s["enum"]) > 12 else "")
        errors.append(f"{path}: {_short(v)} is not one of [{opts}]")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # json.loads accepts NaN / Infinity and integers beyond the float range; every
        # comparison with NaN is False, so these would otherwise pass any bound
        try:
            finite = math.isfinite(float(v))
        except OverflowError:
            finite = False
        if not finite:
            errors.append(f"{path}: {_short(v)} is not a finite number")
            return
        if "minimum" in s and v < s["minimum"]:
            errors.append(f"{path}: {v} is below the minimum {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errors.append(f"{path}: {v} is above the maximum {s['maximum']}")
    if isinstance(v, str):
        if "maxLength" in s and len(v) > s["maxLength"]:
            errors.append(f"{path}: string longer than {s['maxLength']} characters")
        if "minLength" in s and len(v) < s["minLength"]:
            errors.append(f"{path}: string shorter than {s['minLength']} characters")
    if isinstance(v, list):
        if "maxItems" in s and len(v) > s["maxItems"]:
            errors.append(f"{path}: {len(v)} items, at most {s['maxItems']} allowed")
        if "minItems" in s and len(v) < s["minItems"]:
            errors.append(f"{path}: {len(v)} items, at least {s['minItems']} required")
        if isinstance(s.get("items"), dict):
            for i, item in enumerate(v):
                _validate(item, s["items"], f"{path}[{i}]", errors, limit)
    if isinstance(v, dict):
        props = s.get("properties", {})
        for key in s.get("required", []):
            if key not in v:
                errors.append(f"{path}: missing required key '{key}'")
        for key, item in v.items():
            if key in props:
                _validate(item, props[key], f"{path}.{key}", errors, limit)
            elif s.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key '{key}'")
            elif isinstance(s.get("additionalProperties"), dict):
                _validate(item, s["additionalProperties"], f"{path}.{key}", errors, limit)


# -- schemas for our prompts ------------------------------------------------------
def action_item_schema(allowed_actions: Iterable[str], regions: Iterable[str], landmark_kinds: Iterable[str],
                       poi_kinds: Iterable[str], param_paths: Iterable[str]) -> dict:
    """One action object. Enums are cut down to what the scope allows, so a grammar-constrained
    model cannot even spell an action outside the sandbox."""
    allowed = list(allowed_actions)
    props: dict[str, Any] = {
        "type": {"type": "string", "enum": allowed},
        "reason": {"type": "string", "maxLength": 240},
    }
    regions = list(regions)
    if any(a != "set_param" for a in allowed) and regions:
        props["region"] = {"type": "string", "enum": regions}
    kinds: list[str] = []
    if "add_landmark" in allowed:
        kinds += list(landmark_kinds)
    if "add_poi" in allowed:
        kinds += [k for k in poi_kinds if k not in kinds]
    if kinds:
        props["kind"] = {"type": "string", "enum": kinds}
    if "raise_relief" in allowed or "smooth_terrain" in allowed:
        props["amount"] = {"type": "number", "minimum": 0.1, "maximum": 0.8}
    paths = list(param_paths)
    if "set_param" in allowed and paths:
        props["path"] = {"type": "string", "enum": paths}
        props["value"] = {"type": "number"}
    return {"type": "object", "required": ["type"], "properties": props}


def review_schema(score_keys: Iterable[str], allowed_actions: Iterable[str], regions: Iterable[str],
                  landmark_kinds: Iterable[str], poi_kinds: Iterable[str], param_paths: Iterable[str],
                  max_actions: int) -> dict:
    keys = list(score_keys)
    allowed = list(allowed_actions)
    actions: dict[str, Any] = {"type": "array", "maxItems": max(0, int(max_actions))}
    if allowed and max_actions > 0:       # an empty enum would not compile to a grammar
        actions["items"] = action_item_schema(allowed, regions, landmark_kinds, poi_kinds, param_paths)
    else:
        actions["maxItems"] = 0
    return {
        "type": "object",
        "required": ["scores", "critique", "actions"],
        "properties": {
            "scores": {"type": "object", "required": keys,
                       "properties": {k: {"type": "number", "minimum": 0, "maximum": 10} for k in keys}},
            "critique": {"type": "string", "maxLength": 1200},
            "actions": actions,
        },
    }


def lore_schema(ids: Iterable[str]) -> dict:
    entry = {"type": "object", "required": ["name"],
             "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 48},
                            "description": {"type": "string", "maxLength": 300}}}
    ids = list(ids)
    return {"type": "object", "required": ids, "properties": {i: entry for i in ids}}
