"""The sandbox a local model operates in (``ai.scope``).

A language model is an unreliable actor: it can hallucinate region names,
repeat itself, or propose edits far larger than intended. Following the
capability-based / least-privilege pattern used for tool-using agents, the
model never acts on the world directly. Every proposal passes three gates:

1. **Capability** (``check``): the action type must be whitelisted, its region
   allowed and not protected, ``set_param`` paths must have a configured range
   (values are clamped into it; ``ai.*`` keys - the sandbox, budget and
   objective weight - are never settable, so the model cannot widen its own
   scope) and per-action, per-iteration and per-run quotas must not be
   exhausted.
2. **Placement** (``forbidden_mask``): protected regions, protected circles,
   the agents' spawn settlement and regions outside ``allowed_regions`` are
   excluded from site selection; terrain edits are multiplied by a feathered
   complement (``edit_weight``) so they fade to exactly zero at the boundary
   without a seam.
3. **Post-condition** (``verify_edit``): after an edit is applied to the
   candidate world, the change of ``height_base`` is measured: per-edit
   maximum |dz|, cumulative |dz| against the terrain at policy creation, the
   share of land cells changed by more than 0.5 m, and any change inside
   forbidden cells. A violation reverts just that action.

Every decision is appended to an audit trail (``audit``), which the director
stores in ``world.evaluation["ai_audit"]``. Quotas are counted per candidate
and only committed when the director keeps the candidate, so the limits
bound what actually ends up in the final world.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..config import AI_ACTIONS, AIScopeConfig, ConfigError, get_by_path
from ..grid import chamfer_distance, dilate, smoothstep
from ..world import World
from .actions import action_doc, param_settable

_TOTAL = "__total__"
MODIFIED_THRESHOLD_M = 0.5     # a cell counts as modified above this cumulative |dz|
FEATHER_M = 96.0               # width of the fade-out band around forbidden cells


def _compact(action: dict) -> dict:
    keep = ("type", "region", "kind", "amount", "path", "value", "x", "y", "reason")
    out = {}
    for k in keep:
        if k in action:
            v = action[k]
            out[k] = round(v, 3) if isinstance(v, float) else v
    if isinstance(out.get("reason"), str):
        out["reason"] = out["reason"][:160]
    return out


class ScopePolicy:
    """Capability checks, placement mask, edit verification and audit trail for one director run."""

    def __init__(self, scope: AIScopeConfig, world: World, max_actions: int | None = None):
        self.scope = scope
        self.size = world.size
        self.cell = float(world.cell)
        self.sea_level = float(world.cfg.world.sea_level)
        self.warnings: list[str] = []
        self.all_regions = world.all_regions()
        known = set(self.all_regions)
        prot = {str(r).strip().upper() for r in scope.protected_regions}
        allow = {str(r).strip().upper() for r in scope.allowed_regions}
        for r in sorted((prot | allow) - known):
            self.warnings.append(f"unknown region '{r}' in ai.scope")
        self.protected_regions = [r for r in self.all_regions if r in prot]
        self.allowed_regions = [r for r in self.all_regions if (not allow or r in allow) and r not in prot]
        self.allowed_actions = [a for a in AI_ACTIONS if a in scope.allowed_actions]
        if max_actions is None:
            max_actions = world.cfg.ai.max_actions
        self.per_iteration = max(0, min(int(max_actions), int(scope.max_actions_per_iteration)))
        self.param_ranges: dict[str, tuple[float, float]] = {}
        for pr in scope.param_ranges:
            if not param_settable(pr.path):
                self.warnings.append(f"param range for '{pr.path}' ignored: the model may not change it")
                continue
            try:
                cur = get_by_path(world.cfg, pr.path)
            except ConfigError:
                self.warnings.append(f"param range for unknown key '{pr.path}' ignored")
                continue
            if isinstance(cur, bool) or not isinstance(cur, (int, float)):
                self.warnings.append(f"param range for non-numeric key '{pr.path}' ignored")
                continue
            self.param_ranges[pr.path] = (float(min(pr.min, pr.max)), float(max(pr.min, pr.max)))
        spawn = next((p for p in world.pois if p.kind == "settlement"), None)
        self.spawn = None if spawn is None else (float(spawn.x), float(spawn.y))
        self.origin = np.asarray(world.height_base, dtype=np.float64).copy()
        self.land = self.origin > self.sea_level
        self.feather_cells = max(2, int(round(FEATHER_M / self.cell)))
        self._forbidden = self.forbidden_mask(world)
        self._weight: np.ndarray | None = None
        self.committed: Counter = Counter()
        self.pending: Counter = Counter()
        self._last_commit: Counter = Counter()
        self.iteration = 0
        self.audit: list[dict] = []

    # -- placement ------------------------------------------------------------------
    def zones(self) -> list[dict]:
        """Protected circles (configured zones plus the spawn settlement)."""
        out = [{"x": float(z.x), "y": float(z.y), "radius": float(z.radius), "reason": z.reason or "protected"}
               for z in self.scope.protected_zones if z.radius > 0]
        if self.spawn is not None and self.scope.protect_spawn_radius > 0:
            out.append({"x": self.spawn[0], "y": self.spawn[1], "radius": float(self.scope.protect_spawn_radius),
                        "reason": "agent spawn settlement"})
        return out

    def forbidden_mask(self, world: World | None = None) -> np.ndarray:
        """Cells the AI may not touch: blocked regions and protected circles (bool, size x size)."""
        if world is None:
            return self._forbidden.copy()
        n, cell = world.size, float(world.cell)
        m = np.zeros((n, n), dtype=bool)
        for r in self.all_regions:
            if r not in self.allowed_regions:
                sj, si = world.region_slices(r)
                m[sj, si] = True
        c = (np.arange(n) + 0.5) * cell
        for z in self.zones():
            # a cell is blocked when any part of it may overlap the circle
            reach = z["radius"] + 0.71 * cell
            m |= (c[None, :] - z["x"]) ** 2 + (c[:, None] - z["y"]) ** 2 <= reach * reach
        return m

    def site_mask(self, margin_m: float = 0.0) -> np.ndarray:
        """Forbidden cells grown by ``margin_m``: sites for objects with a footprint."""
        r = int(np.ceil(max(0.0, margin_m) / self.cell))
        return dilate(self._forbidden, r) if r > 0 and self._forbidden.any() else self._forbidden.copy()

    def edit_weight(self) -> np.ndarray:
        """Multiplier for terrain edits: exactly 0 on forbidden cells, smooth ramp to 1 outside."""
        if self._weight is None:
            if self._forbidden.any():
                d = chamfer_distance(self._forbidden, max_dist=self.feather_cells)
                self._weight = smoothstep(0.0, float(self.feather_cells), d)
            else:
                self._weight = np.ones((self.size, self.size))
        return self._weight

    # -- quotas -----------------------------------------------------------------------
    def _used(self, key: str, planned: Counter | None = None) -> int:
        return self.committed[key] + self.pending[key] + (planned[key] if planned else 0)

    def limit(self, action_type: str) -> int:
        return max(0, int(getattr(self.scope.limits, action_type, self.scope.max_total_actions)))

    def remaining(self, action_type: str, planned: Counter | None = None) -> int:
        if action_type not in self.allowed_actions:
            return 0
        return max(0, self.limit(action_type) - self._used(action_type, planned))

    def remaining_total(self, planned: Counter | None = None) -> int:
        return max(0, int(self.scope.max_total_actions) - self._used(_TOTAL, planned))

    def iteration_capacity(self) -> int:
        """How many actions the next proposal may contain."""
        used = self.pending[_TOTAL]
        return max(0, min(self.per_iteration - used, self.remaining_total()))

    def has_capacity(self) -> bool:
        """False when no allowed action could still be applied anywhere."""
        if self.remaining_total() <= 0 or self.per_iteration <= 0:
            return False
        for a in self.allowed_actions:
            if self.remaining(a) <= 0:
                continue
            if a == "set_param" and self.param_ranges:
                return True
            if a != "set_param" and self.allowed_regions:
                return True
        return False

    def check(self, action: dict, planned: Counter | None = None) -> tuple[bool, str]:
        """Capability check of a validated action. Clamps ``set_param`` values in place.

        ``planned`` counts actions already accepted into the same proposal, so a proposal
        cannot exceed a quota even before anything is applied."""
        t = str(action.get("type", ""))
        if t not in self.allowed_actions:
            return False, f"action '{t}' is not allowed in this scope"
        if t == "set_param":
            path = str(action.get("path", ""))
            if path not in self.param_ranges:
                return False, f"parameter '{path}' has no configured range"
            lo, hi = self.param_ranges[path]
            try:
                action["value"] = float(np.clip(float(action.get("value")), lo, hi))
            except (TypeError, ValueError):
                return False, "set_param needs a numeric value"
        else:
            region = str(action.get("region", ""))
            if region in self.protected_regions:
                return False, f"region {region} is protected"
            if region not in self.allowed_regions:
                return False, f"region {region} is outside the allowed regions"
        if self.remaining_total(planned) <= 0:
            return False, f"run limit of {self.scope.max_total_actions} actions reached"
        used_iter = self.pending[_TOTAL] + (planned[_TOTAL] if planned else 0)
        if used_iter >= self.per_iteration:
            return False, f"per-iteration limit of {self.per_iteration} actions reached"
        if self.remaining(t, planned) <= 0:
            return False, f"limit for {t} ({self.limit(t)}) reached"
        return True, "ok"

    @staticmethod
    def plan(planned: Counter, action: dict) -> None:
        planned[action["type"]] += 1
        planned[_TOTAL] += 1

    # -- verification -------------------------------------------------------------------
    def verify_edit(self, before: np.ndarray, after: np.ndarray) -> tuple[bool, str, dict]:
        """Post-condition of one edit on ``height_base`` (before/after the action)."""
        sc = self.scope
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        d_edit = np.abs(after - before)
        d_total = np.abs(after - self.origin)
        modified = (d_total > MODIFIED_THRESHOLD_M) & self.land
        frac = float(modified.sum()) / max(1, int(self.land.sum()))
        forb = d_total[self._forbidden]
        forb_cells = int((forb > 1e-6).sum()) if forb.size else 0
        stats = {
            "max_delta_m": round(float(d_edit.max()), 2),
            "max_total_delta_m": round(float(d_total.max()), 2),
            "modified_area_fraction": round(frac, 4),
            "changed_cells": int((d_edit > MODIFIED_THRESHOLD_M).sum()),
            "forbidden_cells_changed": forb_cells,
        }
        problems = []
        if forb_cells:
            problems.append(f"changed {forb_cells} protected cell(s)")
        if stats["max_delta_m"] > sc.max_height_delta_m:
            problems.append(f"edit changes terrain by {stats['max_delta_m']:.1f} m "
                            f"(limit {sc.max_height_delta_m:g} m)")
        if stats["max_total_delta_m"] > sc.max_total_height_delta_m:
            problems.append(f"cumulative change {stats['max_total_delta_m']:.1f} m "
                            f"(limit {sc.max_total_height_delta_m:g} m)")
        if frac > sc.max_modified_area_fraction:
            problems.append(f"modified land area {frac:.1%} (limit {sc.max_modified_area_fraction:.1%})")
        return (not problems), "; ".join(problems) or "ok", stats

    # -- iteration bookkeeping & audit ---------------------------------------------------
    def begin_iteration(self, iteration: int) -> None:
        self.iteration = int(iteration)
        self.pending.clear()
        self._last_commit = Counter()

    def record(self, action: dict, status: str, reason: str = "", stats: dict | None = None,
               iteration: int | None = None, stage: str = "apply") -> dict:
        """Append an audit entry; ``applied`` and ``dry_run`` consume the pending quota.

        ``stage`` is "apply" for decisions of the director and "review" for model proposals
        that were refused (out of scope, malformed) before they reached it."""
        if status in ("applied", "dry_run"):
            self.pending[str(action.get("type", ""))] += 1
            self.pending[_TOTAL] += 1
        entry = {"iteration": self.iteration if iteration is None else int(iteration),
                 "action": _compact(action), "status": status, "reason": reason, "stats": stats or {},
                 "stage": stage}
        self.audit.append(entry)
        return entry

    def commit(self) -> None:
        """Count this iteration's applied actions against the quotas.

        The director commits before it re-reviews the candidate (so the next proposal sees the
        remaining quota) and calls ``rollback`` if the candidate is then discarded."""
        self._last_commit = Counter(self.pending)
        self.committed.update(self.pending)
        self._mark_kept(True)
        self.pending.clear()

    def rollback(self) -> None:
        """Undo the last ``commit`` (or drop pending usage): the candidate was not kept."""
        last = getattr(self, "_last_commit", None) or Counter()
        self.committed.subtract(last)
        self.committed = +self.committed          # drop zero / negative counts
        self._last_commit = Counter()
        self._mark_kept(False)
        self.pending.clear()

    def _mark_kept(self, kept: bool) -> None:
        for e in self.audit:
            if e["iteration"] == self.iteration and e["status"] == "applied":
                e["kept"] = kept

    # -- prompt ---------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Compact, JSON-able statement of the sandbox for the model."""
        docs = action_doc(self.param_ranges)
        zones = []
        for z in self.zones():
            zones.append({"x": round(z["x"]), "y": round(z["y"]), "radius_m": round(z["radius"]),
                          "reason": str(z["reason"])[:60]})
        out: dict[str, Any] = {
            "allowed_actions": {a: {"params": docs[a], "remaining": self.remaining(a)}
                                for a in self.allowed_actions if self.remaining(a) > 0},
            "allowed_regions": list(self.allowed_regions),
            "protected_regions": list(self.protected_regions),
            "protected_zones": zones,
            "limits": {
                "actions_this_iteration": self.iteration_capacity(),
                "actions_remaining_total": self.remaining_total(),
                "max_height_change_per_edit_m": self.scope.max_height_delta_m,
                "max_cumulative_height_change_m": self.scope.max_total_height_delta_m,
                "max_modified_land_fraction": self.scope.max_modified_area_fraction,
            },
        }
        if "set_param" in self.allowed_actions and self.param_ranges:
            out["param_ranges"] = {p: [lo, hi] for p, (lo, hi) in self.param_ranges.items()}
        return out

    def summary(self) -> dict[str, Any]:
        """Run summary for reports: usage per action and audit counts by status."""
        by_status = Counter(e["status"] for e in self.audit)
        used = {a: self.committed[a] for a in self.allowed_actions if self.committed[a]}
        return {"allowed_actions": list(self.allowed_actions), "allowed_regions": list(self.allowed_regions),
                "protected_regions": list(self.protected_regions), "protected_zones": self.zones(),
                "used": used, "total_used": self.committed[_TOTAL], "audit_counts": dict(by_status),
                "dry_run": bool(self.scope.dry_run), "warnings": list(self.warnings)}
