import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ofm.ai.actions import ActionError, validate_action
from ofm.ai.director import clone_world, run_director
from ofm.ai.evaluator import heuristic_review, review
from ofm.ai.llm import LocalLLM, extract_json
from ofm.ai.lore import NameGen, generate_lore
from ofm.config import load_config
from ofm.world import World


def test_extract_json_handles_chatter_and_fences():
    assert extract_json('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert extract_json('blah {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json("no json here") is None


def test_validate_action_whitelist():
    regions = World.all_regions()
    a = validate_action({"type": "add_poi", "region": "b3", "kind": "ruin"}, regions)
    assert a["region"] == "B3" and a["kind"] == "ruin"
    assert validate_action({"type": "raise_relief", "region": "A1", "amount": 9}, regions)["amount"] == 0.6
    assert validate_action({"type": "set_param", "path": "caves.density", "value": 100}, regions)["value"] == 3.0
    with pytest.raises(ActionError):
        validate_action({"type": "rm -rf", "region": "A1"}, regions)
    with pytest.raises(ActionError):
        validate_action({"type": "set_param", "path": "world.seed", "value": 1}, regions)
    with pytest.raises(ActionError):
        validate_action({"type": "add_cave", "region": "Z9"}, regions)


class _FakeLLMHandler(BaseHTTPRequestHandler):
    reply: dict = {}
    requests: list = []

    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "test-model:latest"}]})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "test-model"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        type(self).requests.append((self.path, req))
        content = json.dumps(type(self).reply)
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": content}})
        else:
            self._send({"choices": [{"message": {"role": "assistant", "content": "```json\n" + content + "\n```"}}]})


@pytest.fixture()
def fake_llm_server():
    _FakeLLMHandler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLMHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


REPLY = {
    "scores": {"exploration": 6, "variety": 7, "landmarks": 5, "pacing": 4, "mystery": 6, "coherence": 7, "overall": 6.2},
    "critique": "동쪽 고원이 비어 있습니다.",
    "actions": [
        {"type": "add_landmark", "region": "C3", "kind": "spire", "reason": "orientation"},
        {"type": "add_poi", "region": "B2", "kind": "shrine"},
        {"type": "explode", "region": "A1"},
    ],
}


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_review_uses_local_llm(small_world, fake_llm_server, provider):
    _FakeLLMHandler.reply = REPLY
    cfg = load_config(None, {"ai.provider": provider, "ai.base_url": fake_llm_server, "ai.model": "test-model"})
    llm = LocalLLM(cfg.ai)
    assert llm.available
    w = clone_world(small_world)
    w.cfg = cfg.copy()
    w.psych = small_world.psych
    r = review(w, llm)
    assert r["source"] == "llm"
    assert [a["type"] for a in r["actions"]] == ["add_landmark", "add_poi"]
    assert r["llm"]["invalid_actions"], "the unknown action must be reported, not executed"
    assert r["critique"] == REPLY["critique"]
    assert r["objective"] == pytest.approx(0.7 * r["sim_scores"]["overall"] + 0.3 * 6.2, abs=1e-3)
    path, body = _FakeLLMHandler.requests[-1]
    assert path in ("/api/chat", "/v1/chat/completions")
    prompt = json.dumps(body)
    assert "allowed_actions" in prompt and "boredom_seconds" in prompt


def test_unreachable_llm_falls_back_to_heuristics(small_world):
    cfg = load_config(None, {"ai.provider": "ollama", "ai.base_url": "http://127.0.0.1:9"})
    llm = LocalLLM(cfg.ai)
    assert not llm.available
    r = review(small_world, llm)
    assert r["source"] == "heuristic" and r["llm"] is None


def test_heuristic_review_targets_weak_spots(small_world):
    r = heuristic_review(small_world, [], 5, "ko")
    assert 0 < len(r["actions"]) <= 5
    assert "종합" in r["critique"]
    regions = World.all_regions()
    for a in r["actions"]:
        validate_action(a, regions)


def test_director_only_keeps_improvements(small_world):
    w = clone_world(small_world)
    w.psych = small_world.psych
    w.evaluation = dict(small_world.evaluation)
    out = run_director(w, None, iterations=2)
    hist = out.history
    assert hist[0]["iteration"] == 0
    accepted = [h for h in hist if h["accepted"]]
    objs = [h["objective"] for h in accepted]
    assert objs == sorted(objs), "accepted objective never decreases"
    for h in hist[1:]:
        for a in h["actions"]:
            assert a["type"] in ("add_landmark", "add_cave", "add_lake", "add_poi", "raise_relief",
                                 "smooth_terrain", "set_param")


def test_lore_names_everything(small_world):
    w = clone_world(small_world)
    w.psych = small_world.psych
    lore = generate_lore(w, None)
    assert lore["source"] == "procedural"
    assert all(p.name for p in w.pois)
    assert all(c.name for c in w.caves)
    g = NameGen(1, "ko")
    names = {g.name("lake") for _ in range(30)}
    assert len(names) == 30 and all(n.endswith(" 호수") for n in names)
