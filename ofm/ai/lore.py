"""Names and short descriptions for regions and features ("detail" pass).

With a local LLM the names are written in context (biome, geology, what the
agents experienced there). Without one, a procedural generator composes
pronounceable names (Hangul syllable composition for Korean, syllable chains
for English) with a feature-type suffix.

LLM naming runs inside the same budget as the director (``ai.scope``
``max_llm_calls`` / ``max_llm_seconds``): items are sent in chunks that fit
``max_prompt_chars``, each with a JSON schema listing the expected ids, and
naming stops as soon as the budget is spent; whatever is left gets
procedural names.
"""
from __future__ import annotations

import json

import numpy as np

from ..noise import derive_seed
from ..world import World
from .llm import LocalLLM
from .schema import lore_schema

# Curated syllables that read naturally in Korean place names.
_KO_SYLLABLES = list("가강건경고곡구금기나남노단담도동라란령로류리마명묘무문미바백벽보봉비"
                     "사산서석선설성소솔송수시아안암연영오온용우운원월유윤은을이인자장재정조"
                     "주지진차창천청초추하한해향현호화환휘흑희")
_EN_START = ["Ar", "Bel", "Cor", "Dun", "El", "Fen", "Gal", "Har", "Ith", "Kel", "Lor", "Mor", "Nar", "Or",
             "Pel", "Quel", "Ros", "Sar", "Tal", "Ul", "Var", "Wen", "Yr", "Zan"]
_EN_MID = ["a", "e", "i", "o", "u", "ae", "ia", "en", "or", "an"]
_EN_END = ["dor", "mere", "wick", "holt", "garde", "fell", "moor", "reach", "vale", "crest", "ford", "spire", "deep"]

SUFFIX = {
    "ko": {"region": "", "summit": " 봉", "peak": " 봉", "spire": " 첨탑바위", "arch": " 아치", "butte": " 탁상바위",
           "monolith": " 선돌", "sea_stack": " 촛대바위", "waterfall": " 폭포", "lake": " 호수", "cave": " 동굴",
           "river": " 강", "cave_entrance": " 굴", "lakeshore": " 나루", "settlement": " 마을", "village": " 마을",
           "ruin": " 폐허", "camp": " 야영지", "shrine": " 사당", "watchtower": " 망루", "outpost": " 초소",
           "treasure": " 은닉처", "bridge": " 다리", "hermitage": " 암자", "confluence": " 두물머리",
           "cove": " 포구", "rock_formation": " 바위"},
    "en": {"region": "", "summit": " Peak", "peak": " Peak", "spire": " Spire", "arch": " Arch", "butte": " Butte",
           "monolith": " Stone", "sea_stack": " Stack", "waterfall": " Falls", "lake": " Lake", "cave": " Caverns",
           "river": " River", "cave_entrance": " Hollow", "lakeshore": " Landing", "settlement": "",
           "village": "", "ruin": " Ruins", "camp": " Camp", "shrine": " Shrine", "watchtower": " Watch",
           "outpost": " Outpost", "treasure": " Cache", "bridge": " Bridge", "hermitage": " Hermitage",
           "confluence": " Fork", "cove": " Cove", "rock_formation": " Rock"},
}


class NameGen:
    def __init__(self, seed: int, language: str):
        self.rng = np.random.default_rng(derive_seed(seed, "names"))
        self.language = language if language in SUFFIX else "en"
        self.used: set[str] = set()

    def _stem(self) -> str:
        r = self.rng
        if self.language == "ko":
            n = int(r.choice([2, 2, 2, 2, 3]))
            return "".join(str(r.choice(_KO_SYLLABLES)) for _ in range(n))
        return str(r.choice(_EN_START)) + (str(r.choice(_EN_MID)) if r.random() < 0.5 else "") + str(r.choice(_EN_END))

    def name(self, kind: str) -> str:
        for _ in range(50):
            n = self._stem() + SUFFIX[self.language].get(kind, "")
            if n not in self.used:
                self.used.add(n)
                return n
        return n


def _items(world: World) -> list[dict]:
    regions = {r["region"]: r for r in world.evaluation.get("simulation", {}).get("regions", []) if "region" in r}
    items = []
    for rid, r in regions.items():
        if r.get("land_fraction", 0.0) > 0.2:
            items.append({"id": f"region:{rid}", "kind": "region", "biomes": r.get("biomes", []),
                          "relief_m": r.get("relief", 0.0), "pois": list(r.get("pois", []))[:5]})
    for lm in world.landmarks:
        items.append({"id": f"landmark:{lm.id}", "kind": lm.kind, "region": world.region_of(lm.x, lm.y),
                      "height_m": round(lm.height)})
    for p in world.pois:
        items.append({"id": f"poi:{p.id}", "kind": p.kind, "region": p.region})
    for c in world.caves:
        items.append({"id": f"cave:{c.id}", "kind": "cave", "cave_type": c.kind, "length_m": round(c.length)})
    for lk in world.lakes:
        items.append({"id": f"lake:{lk['id']}", "kind": "lake", "saline": lk["saline"],
                      "closed_basin": lk["closed_basin"]})
    for rv in sorted(world.rivers, key=lambda r: -r["length_m"])[:6]:
        items.append({"id": f"river:{rv['id']}", "kind": "river", "length_m": round(rv["length_m"])})
    return items


def _chunks(items: list[dict], max_chars: int, max_items: int = 40) -> list[list[dict]]:
    """Split items so each request stays under ``max_chars`` (and ``max_items``)."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    size = 16
    for it in items:
        n = len(json.dumps(it, ensure_ascii=False)) + 1
        if cur and (size + n > max_chars or len(cur) >= max_items):
            out.append(cur)
            cur, size = [], 16
        cur.append(it)
        size += n
    if cur:
        out.append(cur)
    return out


def generate_lore(world: World, llm: LocalLLM | None) -> dict:
    lang = world.cfg.ai.language
    gen = NameGen(world.cfg.world.seed, lang)
    items = _items(world)
    named: dict[str, dict] = {}
    source = "procedural"
    skipped = ""
    if llm is not None and llm.available and items:
        lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang, lang)
        system = ("You name places in a fantasy open world. Names must be evocative, consistent in style and "
                  "unique. Reply with ONE JSON object mapping each id to {\"name\": ..., \"description\": ...}. "
                  f"Write names and one-sentence descriptions in {lang_name}.")
        for chunk in _chunks(items, llm.prompt_char_budget(system)):
            if llm.budget_exhausted:
                skipped = llm.budget_status()["reason"]
                break
            schema = lore_schema([it["id"] for it in chunk]) if llm.scope.json_schema else None
            raw = llm.chat_json(system, json.dumps({"items": chunk}, ensure_ascii=False), schema=schema,
                                seed=derive_seed(world.cfg.world.seed, "lore", llm.calls), name="ofm_names",
                                retries=min(1, llm.scope.retries))
            if not raw:
                continue
            table = raw.get("items", raw) if isinstance(raw.get("items", raw), dict) else raw
            for it in chunk:
                v = table.get(it["id"]) if isinstance(table, dict) else None
                if isinstance(v, dict) and isinstance(v.get("name"), str) and v["name"].strip():
                    named[it["id"]] = {"name": v["name"].strip()[:48],
                                       "description": str(v.get("description", "")).strip()[:300]}
                    source = "llm"
    for it in items:
        if it["id"] not in named:
            named[it["id"]] = {"name": gen.name(it["kind"]), "description": ""}

    # write back onto world objects
    for lm in world.landmarks:
        v = named.get(f"landmark:{lm.id}")
        if v:
            lm.name, lm.description = v["name"], v["description"]
    for p in world.pois:
        v = named.get(f"poi:{p.id}")
        if v:
            p.name, p.description = v["name"], v["description"]
    for c in world.caves:
        v = named.get(f"cave:{c.id}")
        if v:
            c.name, c.description = v["name"], v["description"]
    for lk in world.lakes:
        v = named.get(f"lake:{lk['id']}")
        if v:
            lk["name"], lk["description"] = v["name"], v["description"]
    for rv in world.rivers:
        v = named.get(f"river:{rv['id']}")
        if v:
            rv["name"], rv["description"] = v["name"], v["description"]
    feat_by_id = {f.id: f for f in world.features}
    for lm in world.landmarks:
        for f in world.features:
            if abs(f.x - lm.x) < 1 and abs(f.y - lm.y) < 1 and f.id in feat_by_id:
                f.name, f.description = lm.name, lm.description
    world.lore = {
        "source": source,
        "language": lang,
        "llm_skipped": skipped,
        "regions": {k.split(":", 1)[1]: v for k, v in named.items() if k.startswith("region:")},
    }
    return world.lore
