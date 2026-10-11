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
   share of land cells changed by more than 0.5 m, any change inside
   forbidden cells and non-finite heights. A violation reverts just that action.
4. **Rebuild** (``protect_rebuild`` / ``RebuildGuard``): the downstream simulations
   are global, so an edit elsewhere can still change a protected area's biome,
   re-route a cave under it, re-pick its features, POIs and vegetation or
   reshape its detail terrain. During the candidate's rebuild that content is
   restored from the last accepted world, stage by stage; if protected terrain,
   water, materials, caves or the agents' spawn settlement still differ
   (``verify_rebuild``), the whole candidate is discarded before it is evaluated.

Every decision is appended to an audit trail (``audit``), which the director
stores in ``world.evaluation["ai_audit"]``. Quotas are counted per candidate
and only committed when the director keeps the candidate, so the limits
bound what actually ends up in the final world.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Callable

import numpy as np

from ..config import AI_ACTIONS, AIScopeConfig, ConfigError, get_by_path, normalize_region
from ..grid import bilinear, chamfer_distance, dilate, smoothstep
from ..world import World
from .actions import action_doc, param_settable

_TOTAL = "__total__"
MODIFIED_THRESHOLD_M = 0.5     # a cell counts as modified above this cumulative |dz|
FEATHER_M = 96.0               # width of the fade-out band around forbidden cells
PROTECTED_TOLERANCE_M = 0.05   # protected terrain may differ by this much after a rebuild


class ScopeViolation(Exception):
    """A rebuilt candidate changed protected content; the director discards it."""

    def __init__(self, reason: str, stats: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.stats = stats or {}


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
        prot = {normalize_region(r) for r in scope.protected_regions}
        allow = {normalize_region(r) for r in scope.allowed_regions}
        if prot - known:
            # config.validate rejects these; a world whose config was edited afterwards must
            # still fail closed rather than leave the region editable
            raise ConfigError(f"ai.scope.protected_regions has unknown regions {sorted(prot - known)}")
        for r in sorted(allow - known):
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
        self._grown_mask: np.ndarray | None = None
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
                value = float(action.get("value"))
            except (TypeError, ValueError, OverflowError):
                return False, "set_param needs a numeric value"
            if not np.isfinite(value):
                return False, "set_param needs a finite value"
            action["value"] = float(np.clip(value, lo, hi))
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
        # NaN compares False with every limit, so non-finite heights are counted explicitly and
        # treated as an infinite change (a NaN inside a protected cell is a protected-cell change)
        finite = np.isfinite(after)
        bad_cells = int((~finite).sum())
        d_edit = np.where(finite, np.abs(after - before), np.inf)
        d_total = np.where(finite, np.abs(after - self.origin), np.inf)
        modified = (d_total > MODIFIED_THRESHOLD_M) & self.land
        frac = float(modified.sum()) / max(1, int(self.land.sum()))
        forb = d_total[self._forbidden]
        forb_cells = int((~(forb <= 1e-6)).sum()) if forb.size else 0
        fin_edit = d_edit[finite]
        fin_total = d_total[finite]
        stats = {
            "max_delta_m": round(float(fin_edit.max()), 2) if fin_edit.size else 0.0,
            "max_total_delta_m": round(float(fin_total.max()), 2) if fin_total.size else 0.0,
            "modified_area_fraction": round(frac, 4),
            "changed_cells": int((d_edit > MODIFIED_THRESHOLD_M).sum()),
            "forbidden_cells_changed": forb_cells,
            "non_finite_cells": bad_cells,
        }
        problems = []
        if bad_cells:
            problems.append(f"edit produced {bad_cells} non-finite height(s)")
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

    # -- rebuild protection ---------------------------------------------------------------
    def in_forbidden(self, x, y) -> np.ndarray:
        """Whether points (m) lie in a forbidden cell."""
        i = np.clip((np.atleast_1d(np.asarray(x, dtype=np.float64)) / self.cell).astype(int), 0, self.size - 1)
        j = np.clip((np.atleast_1d(np.asarray(y, dtype=np.float64)) / self.cell).astype(int), 0, self.size - 1)
        return self._forbidden[j, i]

    def _vertex_weight(self, n: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """Edit weight on the detail-terrain vertices (k * spacing) and their protected flag.

        A vertex is protected when any cell it touches is forbidden, so the corners on the
        far edges of a protected cell belong to it as well (otherwise the triangles inside
        the protected area would tilt)."""
        c = np.arange(n) * spacing / self.cell
        lo = np.clip(np.ceil(c - 1e-9).astype(int) - 1, 0, self.size - 1)      # cell left of / below
        hi = np.clip(np.floor(c + 1e-9).astype(int), 0, self.size - 1)         # cell right of / above
        F = self._forbidden
        f = F[np.ix_(lo, lo)] | F[np.ix_(lo, hi)] | F[np.ix_(hi, lo)] | F[np.ix_(hi, hi)]
        x = c - 0.5
        w = bilinear(self.edit_weight(), x[None, :], x[:, None])
        return np.where(f, 0.0, w), f

    def protect_rebuild(self, reference: World) -> "RebuildGuard":
        """Hook for ``rebuild(cand, protect=...)`` that keeps the protected content of
        ``reference`` (the last accepted world); see ``RebuildGuard``."""
        return RebuildGuard(self, reference)

    def _touches(self, cave) -> bool:
        """Whether any passage point, chamber or entrance of a cave lies in (or within a
        cell of) a forbidden cell."""
        pts = []
        if len(cave.segments):
            seg = np.asarray(cave.segments)
            pts += [seg[:, 0:2], seg[:, 3:5], 0.5 * (seg[:, 0:2] + seg[:, 3:5])]
        if len(cave.chambers):
            pts.append(np.asarray(cave.chambers)[:, 0:2])
        if cave.entrances:
            pts.append(np.array([[e["x"], e["y"]] for e in cave.entrances], dtype=np.float64))
        if not pts:
            return False
        p = np.concatenate(pts)
        i = np.clip((p[:, 0] / self.cell).astype(int), 0, self.size - 1)
        j = np.clip((p[:, 1] / self.cell).astype(int), 0, self.size - 1)
        return bool(self._grown()[j, i].any())

    def _grown(self) -> np.ndarray:
        if self._grown_mask is None:
            self._grown_mask = dilate(self._forbidden, 1) if self._forbidden.any() else self._forbidden
        return self._grown_mask

    def verify_rebuild(self, cand: World, ref: World) -> tuple[bool, str, dict]:
        """Post-condition of a rebuild: protected terrain, water, materials, caves, objects and
        the spawn settlement equal the reference. Returns (ok, reason, stats)."""
        problems: list[str] = []
        stats: dict[str, Any] = {}
        F = self._forbidden
        if F.any():
            dh = np.abs(np.asarray(cand.height, dtype=np.float64) - ref.height)[F]
            worst = float(np.max(np.where(np.isfinite(dh), dh, np.inf)))
            stats["protected_height_delta_m"] = round(worst, 3) if np.isfinite(worst) else -1.0
            if not worst <= PROTECTED_TOLERANCE_M:
                problems.append("terrain in protected cells changed" +
                                (f" by {worst:.2f} m" if np.isfinite(worst) else " (non-finite)"))
            if cand.hires is not None and ref.hires is not None:
                if cand.hires.shape != ref.hires.shape or abs(cand.hires_spacing - ref.hires_spacing) > 1e-9:
                    problems.append("detail terrain resolution changed")
                else:
                    _, fv = self._vertex_weight(cand.hires.shape[0], cand.hires_spacing)
                    d = np.abs(cand.hires - ref.hires)[fv]
                    worst_v = float(np.max(np.where(np.isfinite(d), d, np.inf))) if d.size else 0.0
                    if not worst_v <= PROTECTED_TOLERANCE_M:
                        stats["protected_detail_delta_m"] = round(worst_v, 3) if np.isfinite(worst_v) else -1.0
                        problems.append("detail terrain in protected cells changed")
            for name in ("ocean", "lake_mask", "river_mask", "biome"):
                a, b = getattr(cand, name), getattr(ref, name)
                if a is not None and b is not None:
                    n = int((a[F] != b[F]).sum())
                    if n:
                        stats[f"protected_{name}_cells_changed"] = n
                        problems.append(f"{name.replace('_mask', '')} changed on {n} protected cell(s)")

            def caves(w: World) -> list:
                return sorted((c.id, len(c.segments), round(float(np.asarray(c.segments)[:, :6].sum()), 1)
                               if len(c.segments) else 0.0) for c in w.caves if self._touches(c))

            if caves(cand) != caves(ref):
                problems.append("caves in protected cells changed")

            def objects(items) -> list:
                return sorted((o.kind, round(float(o.x), 1), round(float(o.y), 1)) for o in items
                              if self.in_forbidden(o.x, o.y)[0])

            if objects(cand.features) != objects(ref.features) or objects(cand.pois) != objects(ref.pois):
                problems.append("objects in protected cells changed")
        if self.spawn is not None and self.scope.protect_spawn_radius > 0:
            sp = next((p for p in cand.pois if p.kind == "settlement"), None)
            if sp is None or np.hypot(sp.x - self.spawn[0], sp.y - self.spawn[1]) > 0.5 * self.cell:
                problems.append("the agents' spawn settlement moved")
        return (not problems), "; ".join(problems) or "ok", stats

    def revert_iteration(self, reason: str, stats: dict | None = None) -> None:
        """The candidate of this iteration is discarded after its edits were applied."""
        for e in self.audit:
            if e["iteration"] == self.iteration and e["status"] == "applied":
                e["status"] = "reverted"
                e["reason"] = reason
                e["stats"] = {**e.get("stats", {}), **(stats or {})}
                e["kept"] = False
        self.rollback()

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


class RebuildGuard:
    """Keeps the protected content of the last accepted world through a candidate's rebuild.

    ``verify_edit`` only sees ``height_base``. The downstream simulations are global: climate
    can turn a protected shrubland into grassland, the cave walk and the feature and POI
    selections draw from map-wide random streams, and the detail pass reshapes terrain
    everywhere. ``pipeline.rebuild`` therefore calls this hook at three stages, so that
    everything computed afterwards already sees the restored content:

    * ``surface`` (after caves and cliffs): biome, cliff masks and splat materials on
      forbidden cells (splat blended over the feather band); caves whose passages touch a
      forbidden cell are taken from the reference (with their ids), candidate caves touching
      it are dropped; procedural features likewise, and candidate features whose footprint
      reaches a forbidden cell or that crowd a restored one are dropped;
    * ``terrain`` (after the detail heightfield): protected vertices (corners included) take
      the reference heights, blended over the feather band; river ribbons are cleared again
      without touching protected vertices. Vegetation and POIs are placed after this, so
      their heights match the exported terrain;
    * ``objects`` (after vegetation and POIs): vegetation and POIs in forbidden cells come
      from the reference (the spawn settlement stays first); then ``verify_rebuild`` runs and
      a ``ScopeViolation`` aborts the rebuild before the candidate is evaluated.
    """

    FEATURE_SPACING_M = 400.0      # place_procedural_features keeps procedural features this far apart

    def __init__(self, policy: ScopePolicy, reference: World):
        self.policy = policy
        self.ref = reference

    def __call__(self, world: World, stage: str = "objects") -> None:
        if stage == "surface":
            self.surface(world)
        elif stage == "terrain":
            self.terrain(world)
        elif stage == "objects":
            self.objects(world)
            ok, reason, stats = self.policy.verify_rebuild(world, self.ref)
            if not ok:
                raise ScopeViolation(reason, stats)

    # -- stages -------------------------------------------------------------------------
    def surface(self, cand: World) -> None:
        pol, ref = self.policy, self.ref
        self._check_ai_caves(cand)
        F = pol._forbidden
        if not F.any():
            return
        for name in ("biome", "cliff_mask", "coast_cliff"):
            a, b = getattr(cand, name), getattr(ref, name)
            if a is not None and b is not None and a.shape == b.shape:
                a = a.copy()
                a[F] = b[F]
                setattr(cand, name, a)
        if cand.splat is not None and ref.splat is not None and cand.splat.shape == ref.splat.shape:
            w = pol.edit_weight()[..., None]
            cand.splat = ref.splat + w * (cand.splat - ref.splat)
        self._caves(cand)
        self._features(cand)

    def terrain(self, cand: World) -> None:
        pol, ref = self.policy, self.ref
        if not pol._forbidden.any() or cand.hires is None or ref.hires is None:
            return
        if cand.hires.shape != ref.hires.shape or abs(cand.hires_spacing - ref.hires_spacing) > 1e-9:
            return                                   # verify_rebuild reports the resolution change
        w, fv = pol._vertex_weight(cand.hires.shape[0], cand.hires_spacing)
        h = ref.hires + w * (cand.hires - ref.hires)
        if cand.rivers:
            from ..detail.hires import _river_samples, clear_ribbons
            sp = cand.hires_spacing
            h = clear_ribbons(h, _river_samples(cand, sp * 0.25, sp, vertices=True), sp, step=sp * 0.25,
                              frozen=fv)
        cand.hires = h

    def objects(self, cand: World) -> None:
        pol, ref = self.policy, self.ref
        if not pol._forbidden.any():
            return

        def inside(o) -> bool:
            return bool(pol.in_forbidden(o.x, o.y)[0])

        if not _same_objects([p for p in cand.pois if inside(p)], [p for p in ref.pois if inside(p)]):
            self._restore_pois(cand, inside)
        self._restore_scatter(cand)

    def _restore_pois(self, cand: World, inside) -> None:
        ref = self.ref
        pois = [self._remap_poi(copy.deepcopy(p), cand) for p in ref.pois if inside(p)]
        pois += [p for p in cand.pois if not inside(p)]
        sp = next((p for p in ref.pois if p.kind == "settlement"), None)
        if sp is not None:          # the agents spawn at the first settlement: keep the reference's first
            k = next((k for k, p in enumerate(pois) if p.kind == "settlement"
                      and abs(p.x - sp.x) < 1e-3 and abs(p.y - sp.y) < 1e-3), None)
            if k is not None:
                pois.insert(0, pois.pop(k))
        for k, p in enumerate(pois):
            p.id = f"poi_{k}"
        cand.pois = pois

    def _restore_scatter(self, cand: World) -> None:
        pol, ref = self.policy, self.ref
        same = True
        for kind in set(cand.scatter) | set(ref.scatter):
            a, b = cand.scatter.get(kind), ref.scatter.get(kind)
            a = a[pol.in_forbidden(a[:, 0], a[:, 1])] if a is not None and len(a) else np.zeros((0, 5))
            b = b[pol.in_forbidden(b[:, 0], b[:, 1])] if b is not None and len(b) else np.zeros((0, 5))
            if a.shape != b.shape or not np.array_equal(a, b):
                same = False
                break
        if same:
            return                             # keep the rebuilt arrays (and their order)
        scatter = {}
        for kind in sorted(set(cand.scatter) | set(ref.scatter)):
            parts = []
            a = cand.scatter.get(kind)
            if a is not None and len(a):
                parts.append(a[~pol.in_forbidden(a[:, 0], a[:, 1])])
            b = ref.scatter.get(kind)
            if b is not None and len(b):
                parts.append(b[pol.in_forbidden(b[:, 0], b[:, 1])])
            if parts:
                scatter[kind] = np.concatenate(parts)
        cand.scatter = scatter

    # -- helpers ------------------------------------------------------------------------
    def _check_ai_caves(self, cand: World, min_cover: float = 1.0, mouth_m: float = 25.0) -> None:
        """Caves the director dug in earlier iterations keep fixed passages; a later edit that
        lowers the ground over one would break it open (a skylight into the tunnel)."""
        for cave in cand.forced_caves:
            seg = np.asarray(cave.segments)
            if not len(seg):
                continue
            mid = 0.5 * (seg[:, 0:3] + seg[:, 3:6])
            top = mid[:, 2] + np.maximum(seg[:, 6], seg[:, 7])
            far = np.ones(len(mid), dtype=bool)
            for e in cave.entrances:
                far &= np.hypot(mid[:, 0] - e["x"], mid[:, 1] - e["y"]) > mouth_m
            if far.any():
                cover = cand.ground_at(mid[far, 0], mid[far, 1]) - top[far]
                if float(cover.min()) < min_cover:
                    raise ScopeViolation(f"the edit would break the AI cave {cave.id} open",
                                         {"ai_cave_min_cover_m": round(float(cover.min()), 2)})

    def _caves(self, cand: World) -> None:
        pol, ref = self.policy, self.ref
        restored = [copy.deepcopy(c) for c in ref.caves if pol._touches(c)]
        kept = [c for c in cand.caves if not pol._touches(c)]
        if not restored and len(kept) == len(cand.caves):
            return
        if _same_caves([c for c in cand.caves if pol._touches(c)], restored):
            return                             # nothing to change: keep ids and order as rebuilt
        for c in restored:
            # passages outside the protected cells may now lie under edited ground: they must
            # keep their cover (or, if they had less, not lose any), else the rebuild fails
            self._check_cover(c, cand, ref)
            for e in c.entrances:
                if not pol.in_forbidden(e["x"], e["y"])[0]:
                    e["z"] = round(float(cand.ground_at(e["x"], e["y"])), 2)
        used = {c.id for c in restored}
        n = 0
        for c in kept:
            if c.id in used:
                while f"cave_{n}" in used or any(k.id == f"cave_{n}" for k in kept):
                    n += 1
                c.id = f"cave_{n}"
            used.add(c.id)
        cand.caves = sorted(kept + restored, key=lambda c: _id_key(c.id))

    def _check_cover(self, cave, cand: World, ref: World, min_cover: float = 1.0) -> None:
        seg = np.asarray(cave.segments)
        if not len(seg):
            return
        mid = 0.5 * (seg[:, 0:3] + seg[:, 3:6])
        out = ~self.policy.in_forbidden(mid[:, 0], mid[:, 1])
        if cave.entrances:
            ent = np.array([[e["x"], e["y"]] for e in cave.entrances])
            near = np.min(np.hypot(mid[:, None, 0] - ent[None, :, 0], mid[:, None, 1] - ent[None, :, 1]), axis=1)
            out &= near > 40.0                 # the passage rises to the surface at its mouths
        if not out.any():
            return
        sel = mid[out]
        new = cand.ground_at(sel[:, 0], sel[:, 1]) - sel[:, 2]
        old = ref.ground_at(sel[:, 0], sel[:, 1]) - sel[:, 2]
        if (new < np.minimum(old, min_cover) - 0.05).any():
            raise ScopeViolation(f"protected cave {cave.id} would lose its rock cover outside the protected area",
                                 {"min_cover_m": round(float(new.min()), 2)})

    def _features(self, cand: World) -> None:
        from .actions import FOOTPRINT_M
        pol, ref = self.policy, self.ref

        def inside(f) -> bool:
            return bool(pol.in_forbidden(f.x, f.y)[0])

        restored = [copy.deepcopy(f) for f in ref.features if inside(f)]
        kept = []
        for f in cand.features:
            if inside(f):
                continue
            reach = FOOTPRINT_M.get(f.kind, 40.0)
            if pol.site_mask(reach)[min(int(f.y / pol.cell), pol.size - 1), min(int(f.x / pol.cell), pol.size - 1)]:
                continue                       # its footprint would reach into the protected area
            if f.source == "procedural" and any(np.hypot(f.x - r.x, f.y - r.y) < self.FEATURE_SPACING_M
                                                for r in restored if r.source == "procedural"):
                continue                       # crowds a restored feature (procedural spacing)
            kept.append(f)
        inside_cand = [f for f in cand.features if inside(f)]
        if len(kept) + len(inside_cand) == len(cand.features) and _same_objects(inside_cand, restored):
            return                             # nothing to change: keep ids and order as rebuilt
        used = {f.id for f in restored}
        for f in kept:
            if f.id in used:                   # renumber only on a clash with a restored id
                n = 0
                while f"feature_{n}" in used or any(k is not f and k.id == f"feature_{n}" for k in kept):
                    n += 1
                f.id = f"feature_{n}"
            used.add(f.id)
        cand.features = sorted(kept + restored, key=lambda f: _id_key(f.id))

    def _remap_poi(self, p, cand: World):
        """Point a restored POI's lake / waterfall reference at the candidate's object there."""
        if p.kind == "lakeshore" and p.props and cand.lakes:
            ref_lake = next((lk for lk in self.ref.lakes if lk["id"] == p.props[0]), None)
            if ref_lake is not None and "center" in ref_lake:
                cx, cy = ref_lake["center"]
                best = min(cand.lakes, key=lambda lk: np.hypot(lk["center"][0] - cx, lk["center"][1] - cy))
                if np.hypot(best["center"][0] - cx, best["center"][1] - cy) < 2 * self.policy.cell:
                    p.props = [best["id"]] + list(p.props[1:])
        if p.kind == "waterfall" and p.props and cand.waterfalls:
            best = min(cand.waterfalls, key=lambda f: np.hypot(f["x"] - p.x, f["y"] - p.y))
            if np.hypot(best["x"] - p.x, best["y"] - p.y) < self.policy.cell:
                p.props = [best["id"]] + list(p.props[1:])
        return p


def _id_key(ident: str) -> tuple:
    """Sort key for ids like 'cave_3', 'cave_ai_5', 'feature_12': prefix, then the number."""
    head, _, tail = str(ident).rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (str(ident), -1)


def _same_objects(a: list, b: list) -> bool:
    key = lambda o: (o.kind, round(float(o.x), 3), round(float(o.y), 3))   # noqa: E731
    return sorted(map(key, a)) == sorted(map(key, b))


def _same_caves(a: list, b: list) -> bool:
    def key(c):
        seg = np.asarray(c.segments)
        return (c.id, c.kind, seg.shape, round(float(seg.sum()), 3) if seg.size else 0.0)
    return sorted(map(key, a)) == sorted(map(key, b))
