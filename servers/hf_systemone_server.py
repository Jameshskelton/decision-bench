"""Serve a Hugging Face decision model on POST /v1/systemone.

Used for LiquidAI/d1-2B (which ships no server) and as the fallback for vllm-sr/d3.
Both models expose `model.system_one(state=..., questions=...)`; d1 also has
`system_one_batch([(state, questions), ...])`, which this server uses to batch concurrent
requests (up to --max-batch, waiting at most --batch-window-ms). Without it, requests run one
at a time, so throughput numbers from this server are a floor for models that lack batching.

  python servers/hf_systemone_server.py --model LiquidAI/d1-2B --revision <sha> --port 8010 --compile
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Batcher:
    def __init__(self, model, max_batch: int, window_ms: float):
        self.model, self.max_batch, self.window = model, max_batch, window_ms / 1000
        self.q: queue.Queue = queue.Queue()
        self.can_batch = hasattr(model, "system_one_batch") and max_batch > 1
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, state, questions):
        done = threading.Event()
        slot = {"state": state, "questions": questions, "done": done}
        self.q.put(slot)
        done.wait()
        if "error" in slot:
            raise slot["error"]
        return slot["result"]

    def _loop(self):
        while True:
            batch = [self.q.get()]
            if self.can_batch:
                deadline = time.perf_counter() + self.window
                while len(batch) < self.max_batch:
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        break
                    try:
                        batch.append(self.q.get(timeout=left))
                    except queue.Empty:
                        break
            try:
                if len(batch) == 1:
                    outs = [self.model.system_one(state=batch[0]["state"], questions=batch[0]["questions"])]
                else:
                    outs = list(self.model.system_one_batch([(s["state"], s["questions"]) for s in batch]))
                for s, o in zip(batch, outs):
                    s["result"] = o
            except Exception as e:  # noqa: BLE001 - surfaced to the client as HTTP 500
                for s in batch:
                    s["error"] = e
            for s in batch:
                s["done"].set()


def make_handler(batcher, model_id):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, matching the other servers
        disable_nagle_algorithm = True  # else headers and body go out as two packets: +40 ms delayed-ACK stall

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"model": model_id, "status": "ok"} if self.path == "/health" else {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.perf_counter()
                out = batcher.submit(req["state"], req["questions"])
                out = dict(out)
                out.setdefault("model", model_id)
                out["server_ms"] = (time.perf_counter() - t0) * 1000
                self._send(200, out)
            except (KeyError, ValueError) as e:
                self._send(400, {"error": repr(e)})
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": repr(e)})

    return H


def load_model(model_id, revision, compile_):
    import torch
    from transformers import AutoModel

    dtype = torch.bfloat16
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True, dtype=dtype)
    model = model.to("cuda").eval()
    if compile_ and hasattr(model, "compile"):
        model.compile()  # d1: about a minute on the first calls
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pin a commit sha for the manifest")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--batch-window-ms", type=float, default=2.0)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    model = load_model(a.model, a.revision, a.compile)
    batcher = Batcher(model, a.max_batch, a.batch_window_ms)
    # Warm the compiled graphs before accepting traffic.
    for _ in range(3):
        batcher.submit("warmup", {"w": {"type": "noul", "instructions": "Is this a warmup?"}})
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(batcher, a.model))
    print(f"serving {a.model} on http://{a.host}:{a.port}/v1/systemone (batching={batcher.can_batch})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
