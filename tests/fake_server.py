"""A stand-in System One server for testing the harness without a GPU.

It answers `block` from a hidden label embedded in the state (field `_y`, only present in test
fixtures) with a configurable skill, miscalibration and latency, and sometimes returns 503 to
exercise retries. Also serves an OpenAI-style chat endpoint with logprobs.
"""
import argparse
import hashlib
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(skill, temp, latency_ms, fail_rate):
    lock = threading.Lock()
    calls = {"n": 0}

    def p_block(state):
        y = state.get("_y", 0)
        h = int(hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest(), 16) % 10_000 / 10_000
        logit = (skill if y else -skill) + (h - 0.5) * 8  # overlapping classes, so calibration is measurable
        return 1 / (1 + math.exp(-logit / temp))

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-ratelimit-remaining-requests", "100")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                calls["n"] += 1
                fail = random.random() < fail_rate
            time.sleep(latency_ms / 1000 * random.uniform(0.8, 1.5))
            if fail:
                return self._send(503, {"error": "overloaded"})
            if self.path == "/v1/systemone":
                p = p_block(req["state"])
                ans = {"block": {"type": "noul", "noul": p}}
                if "category" in req["questions"]:
                    choice = req["state"].get("_cat", "legit") if random.random() < 0.7 else "other_overreach"
                    opts = list(req["questions"]["category"]["criteria"])
                    probs = {o: (0.7 if o == choice else 0.3 / (len(opts) - 1)) for o in opts}
                    ans["category"] = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.7}
                return self._send(200, {"answers": ans, "usage": {"input_tokens": 400, "output_tokens": 30}})
            if self.path == "/v1/chat/completions":
                state = json.loads(req["messages"][1]["content"].split("State:\n", 1)[1])
                p = p_block(state)
                top = [{"token": "B", "logprob": math.log(p)}, {"token": "A", "logprob": math.log(1 - p)}]
                return self._send(200, {"choices": [{"logprobs": {"content": [{"token": "B", "top_logprobs": top}]}}],
                                        "usage": {"prompt_tokens": 500, "completion_tokens": 1}})
            self._send(404, {"error": "not found"})

    return H


def serve(port, skill=2.0, temp=1.0, latency_ms=5, fail_rate=0.0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(skill, temp, latency_ms, fail_rate))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    serve(a.port)
    threading.Event().wait()
