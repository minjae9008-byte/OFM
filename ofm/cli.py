"""Command line interface.

    python -m ofm generate --preset default --out out/demo
    python -m ofm generate --preset alpine --set world.seed=7 --ai ollama --model qwen2.5:14b
    python -m ofm ai-status
    python -m ofm install unity  /path/to/UnityProject
    python -m ofm push unity     out/demo/world.json
    python -m ofm push unreal    out/demo/world.json
    python -m ofm batch unreal   out/demo/world.json --editor UnrealEditor-Cmd --project Game.uproject
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import ConfigError, load_config

PRESET_DIR = Path(__file__).resolve().parent / "presets"


def _parse_sets(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _resolve_config(args) -> str | None:
    if args.config:
        return args.config
    if args.preset:
        p = PRESET_DIR / f"{args.preset}.toml"
        if not p.exists():
            raise ConfigError(f"unknown preset '{args.preset}' (available: {', '.join(list_presets())})")
        return str(p)
    return None


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.toml"))


def cmd_generate(args) -> int:
    from .ai.director import run_director
    from .ai.llm import LocalLLM
    from .ai.lore import generate_lore
    from .export.exporter import export_world
    from .pipeline import generate
    from .volume.builder import build_volume

    overrides = _parse_sets(args.set)
    if args.seed is not None:
        overrides["world.seed"] = args.seed
    if args.size is not None:
        overrides["world.size"] = args.size
    if args.ai:
        overrides["ai.provider"] = args.ai
    if args.model:
        overrides["ai.model"] = args.model
    if args.ai_url:
        overrides["ai.base_url"] = args.ai_url
    if args.iterations is not None:
        overrides["ai.iterations"] = args.iterations
    if args.mesh_mode:
        overrides["volume.mesh_mode"] = args.mesh_mode
    cfg = load_config(_resolve_config(args), overrides)
    out = Path(args.out or f"out/{cfg.world.name}_{cfg.world.seed}")
    t0 = time.perf_counter()

    def say(msg: str) -> None:
        if not args.quiet:
            print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)

    say(f"generating '{cfg.world.name}' seed={cfg.world.seed} size={cfg.world.size} "
        f"({cfg.world.size * cfg.world.cell_size / 1000:.1f} km)")
    world = generate(cfg, progress=lambda s: say(f"stage: {s}"))
    llm = LocalLLM(cfg.ai)
    st = llm.status()
    say(f"local AI: {st.provider} {st.model} @ {st.base_url} -> "
        f"{'available' if st.available else 'unavailable (' + st.detail + '), using heuristics'}")
    # iterations = 0 still runs the director: it reviews, audits and records the scope
    # summary without applying anything
    world = run_director(world, llm, progress=say)
    say("lore / names")
    generate_lore(world, llm)
    say("volumetric meshing")
    t = time.perf_counter()
    vol = build_volume(world)
    world.timings["volume"] = round(time.perf_counter() - t, 3)
    say(f"volume: {vol.stats}")
    t = time.perf_counter()
    manifest = export_world(world, vol, out)
    world.timings["export"] = round(time.perf_counter() - t, 3)
    scores = world.evaluation.get("simulation", {}).get("scores", {})
    say(f"exported {manifest}")
    if scores:
        say("scores: " + ", ".join(f"{k}={v}" for k, v in scores.items()))
    for engine in args.push or []:
        _push(engine, manifest, args)
    if args.json:
        print(json.dumps({"manifest": str(manifest), "scores": scores, "volume": vol.stats}))
    return 0


def _push(engine: str, manifest: Path, args) -> None:
    from .engines.bridge import UnityBridge, UnrealBridge
    if engine == "unity":
        resp = UnityBridge(args.host, args.unity_port).import_world(manifest, scatter=not args.no_scatter)
        print(f"Unity: {resp.get('message')} (Unity {resp.get('unity')})")
    elif engine == "unreal":
        UnrealBridge(args.host, args.unreal_port).import_world(manifest, scatter=args.unreal_scatter)
        print("Unreal: import command sent (see the editor Output Log)")
    else:
        raise ConfigError(f"unknown engine '{engine}'")


def cmd_push(args) -> int:
    _push(args.engine, Path(args.manifest), args)
    return 0


def cmd_batch(args) -> int:
    from .engines.bridge import run_batch, unity_batch_command, unreal_batch_command
    if args.engine == "unity":
        cmd = unity_batch_command(args.editor, args.project, args.manifest)
    else:
        cmd = unreal_batch_command(args.editor, args.project, args.manifest, args.unreal_scatter)
    print(" ".join(cmd))
    return 0 if args.dry_run else run_batch(cmd)


def cmd_install(args) -> int:
    from .engines.bridge import install_unity, install_unreal
    dest = install_unity(args.project) if args.engine == "unity" else install_unreal(args.project)
    print(f"installed OFM {args.engine} integration into {dest}")
    return 0


def cmd_ai_status(args) -> int:
    from .ai.llm import LocalLLM
    overrides = _parse_sets(args.set)
    if args.ai:
        overrides["ai.provider"] = args.ai
    if args.model:
        overrides["ai.model"] = args.model
    if args.ai_url:
        overrides["ai.base_url"] = args.ai_url
    cfg = load_config(None, overrides)
    st = LocalLLM(cfg.ai).status()
    print(json.dumps(st.__dict__, ensure_ascii=False, indent=1))
    return 0 if st.available else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ofm", description="Simulation-driven open-world map generator")
    sub = ap.add_subparsers(dest="command", required=True)

    def engine_net(p):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--unity-port", type=int, default=8765)
        p.add_argument("--unreal-port", type=int, default=30010)
        p.add_argument("--no-scatter", action="store_true", help="Unity: skip vegetation placeholders")
        p.add_argument("--unreal-scatter", action="store_true", help="Unreal: spawn placeholder scatter")

    def ai_opts(p):
        p.add_argument("--ai", choices=["auto", "ollama", "openai", "none"])
        p.add_argument("--model")
        p.add_argument("--ai-url")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    g = sub.add_parser("generate", help="generate, evaluate, refine and export a world")
    g.add_argument("--config", help="TOML config file")
    g.add_argument("--preset", help="bundled preset: " + ", ".join(list_presets()))
    g.add_argument("--out")
    g.add_argument("--seed", type=int)
    g.add_argument("--size", type=int)
    g.add_argument("--iterations", type=int, help="AI director iterations (0 = evaluate only)")
    g.add_argument("--mesh-mode", choices=["hybrid", "full"])
    g.add_argument("--push", action="append", choices=["unity", "unreal"], help="send to a running editor")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--json", action="store_true", help="print a JSON summary at the end")
    ai_opts(g)
    engine_net(g)
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("push", help="import an exported world into a running editor")
    p.add_argument("engine", choices=["unity", "unreal"])
    p.add_argument("manifest")
    engine_net(p)
    p.set_defaults(func=cmd_push)

    b = sub.add_parser("batch", help="import headlessly through the engine command line")
    b.add_argument("engine", choices=["unity", "unreal"])
    b.add_argument("manifest")
    b.add_argument("--editor", required=True, help="Unity executable or UnrealEditor-Cmd")
    b.add_argument("--project", required=True, help="Unity project folder or .uproject file")
    b.add_argument("--unreal-scatter", action="store_true")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_batch)

    i = sub.add_parser("install", help="copy the engine integration into a project")
    i.add_argument("engine", choices=["unity", "unreal"])
    i.add_argument("project")
    i.set_defaults(func=cmd_install)

    s = sub.add_parser("ai-status", help="check whether a local LLM server is reachable")
    ai_opts(s)
    s.set_defaults(func=cmd_ai_status)

    pr = sub.add_parser("presets", help="list bundled presets")
    pr.set_defaults(func=lambda a: print("\n".join(list_presets())) or 0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # bridge errors etc. should read as messages, not tracebacks
        from .engines.bridge import BridgeError
        if isinstance(exc, BridgeError):
            print(f"engine bridge: {exc}", file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main())
