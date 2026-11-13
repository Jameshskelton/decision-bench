# Decision model benchmark: Jev vs. open-weight decision models on one H200

This harness measures five systems on the same agent-action gating task: should a guard block
this tool call before it runs? It reports accuracy, calibration, auto-decision coverage, latency
under load, and the daily volume at which self-hosting on an H200 costs less than Jev on
DigitalOcean Serverless Inference.

| System | Size | License | How it runs |
|---|---|---|---|
| Jev (`typesafe-jev-1.13.0`) | undisclosed | TypeSafe MCA | DO Serverless Inference, `/v1/systemone` |
| Kev-4B | 4B | Apache-2.0 | `kev.serve`, port 8008 |
| d1-2B | 1.9B | LFM Open License v1.0 | `servers/hf_systemone_server.py`, port 8010 |
| d3 | 26B | Apache-2.0 | vLLM nightly `/v1/systemone`, port 8000 |
| Qwen3.8-27B, prompted | 27B | Apache-2.0 | vLLM chat completions + logprobs, port 8020 |

Every decision model gets a byte-identical System One request body (`bench/questions.py`).

## The datasets

**d1: [harness-hub/safeagents-judge-bench](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench)** (Apache-2.0).
562 real tool calls from Terminal-Bench 2.1 and TBLite agent runs, each labeled `block` or `allow`
by a four-model panel with human adjudication of splits, plus a 7-way category (eval gaming,
harness introspection, credential misuse, sandbox escape, and so on). The dataset has a public
leaderboard of decision models. Jev, Kev-4B, d1-2B and full d3 are not on it, which is part of
what this run adds.
- Caveat to disclose: only 39 of 562 calls are `allow`, so false-block rates on d1 have wide intervals.
- Our question wording differs from the leaderboard's, so our d1 numbers are not directly comparable to it.

**d2: [karanxa/agent-action-safety-dataset](https://huggingface.co/datasets/karanxa/agent-action-safety-dataset)** (Apache-2.0).
Agent actions with context, constraints, trigger and a safe/unsafe label, plus the violation type
(injection, scope violation, goal drift, policy violation, corrigibility). Roughly balanced.
- `d2_calib`: 500 rows from **train**, used only to fit each system's temperature and thresholds.
- `d2_test`: a stratified 2,000-row sample of **validation**. Every reported d2 number comes from here.
- Caveats to disclose: the rows are LLM-generated (the `source` field reads `gemini`), and the
  repo mixes two schemas. `prepare_data` keeps the flat-schema rows and logs how many it skipped.

Both datasets are short: states are well under 8K tokens, inside every model's validated context
(Kev-4B validates 8,192 tokens).

## What gets reported

- **Block decision:** AUROC and balanced accuracy at 0.5, with 95% bootstrap intervals that
  resample whole trajectories (d1) or generation batches (d2), plus recall and false-block rate.
- **Calibration:** ECE and Brier score, raw (as the vendor ships it) and after one temperature
  fitted on `d2_calib`, with reliability diagrams.
- **Production policy:** two thresholds fitted on `d2_calib` (auto-allow letting through at most
  1% of bad calls, auto-block at most 5% of legitimate ones). Reported as the share of calls
  decided without a human, and the bad calls that slipped through on each test set.
- **Category:** macro F1 on d1's seven categories (decision models only).
- **Latency:** single-stream p50 and p99, and throughput at the highest concurrency with p99 ≤ 1 s.
- **Cost:** Jev cost per million decisions from the measured token usage, self-hosted cost at
  measured throughput, and the crossover volume under flat, business-hours and spiky traffic.
- **Determinism:** each accuracy pass runs three times. We report the largest probability change
  and any decision flips.

## The run plan

Run everything on the H200 box, **one model on the GPU at a time**. Before starting each server,
`nvidia-smi` should show the GPU idle. Budget about one working day, most of it model downloads,
the d3 vLLM install and Jev's network-bound passes.

Each accuracy pass is about 9,200 requests (3,062 items × 3 repeats). Each load pass is
8,000 requests (2,000 at each of four concurrency levels).

### 0. Set up the harness and data (no GPU, about 10 minutes)

```bash
scripts/setup_harness.sh && source .venv-harness/bin/activate
hf auth login                    # only needed if you hit Hub rate limits
python -m bench.prepare_data     # writes data/*.jsonl and data/data_manifest.json
python -m tests.test_pipeline    # harness self-test against fake servers, no GPU needed
```

Check `data/data_manifest.json`:
- d1 has 562 rows and 523 blocks.
- d2_calib and d2_test are roughly balanced.
- The skipped-row counts for d2 are recorded.

Commit `data/` so every system runs on identical items.

### 1. Jev on Serverless Inference (no GPU)

```bash
export MODEL_ACCESS_KEY=...      # Control Panel → Inference → Serverless Inference → model access key
scripts/bench_system.sh jev "DO Serverless Inference, typesafe-jev-1.13.0"
```

- **Tier:** Jev is a third-party model and needs a qualifying account tier. The error
  `this model is not available for your subscription tier` means the tier must be raised.
- **Network:** Jev's latency includes the round trip from the H200 box to `inference.do-ai.run`.
  Record the box's region in the article. Self-hosted models are measured over localhost, so say
  plainly that the comparison is "API call from this box" against "local server."
- **Rate limits:** the load pass stores rate-limit headers. If c=32 or c=64 returns 429s, throughput
  is capped by the account tier, not the model. Report it as a tier limit.
- **Usage:** usage tokens are captured on every response and feed the cost model.

### 2. Kev-4B (about 14 GB of GPU memory)

```bash
scripts/setup_kev.sh
scripts/serve_kev.sh &           # pinned to jaredpalmer/kev-4b@v1.0
scripts/wait_for.sh kev-4b
scripts/bench_system.sh kev-4b "kev.serve @ $(git -C kev rev-parse --short HEAD), bf16"
kill %1
```

Kev's shipped temperature (T = 2.41) stays on, so the raw numbers are Kev as published. Our own
recalibration is applied in analysis.

### 3. d1-2B (a few GB)

```bash
scripts/setup_d1.sh
scripts/serve_d1.sh &            # compiles on start (about a minute), then warms up
scripts/wait_for.sh d1-2b
scripts/bench_system.sh d1-2b "hf_systemone_server, transformers $(.venv-d1/bin/python -c 'import transformers;print(transformers.__version__)'), compiled, batch 32"
kill %1
```

d1 ships no server. Ours batches concurrent requests through d1's own `system_one_batch`.

### 4. d3 (about 62–65 GiB)

d3's vLLM path was verified by its authors on AMD only, so first cross-check vLLM against d3's
own code on this GPU:

```bash
scripts/setup_d3_vllm.sh && scripts/setup_d3_hf.sh
scripts/serve_d3_vllm.sh --gpu-memory-utilization 0.5 &     # port 8000
scripts/serve_d3_hf.sh &                                    # port 8001, AutoModel reference
python -m bench.crosscheck --a http://127.0.0.1:8000/v1/systemone \
    --b http://127.0.0.1:8001/v1/systemone --model vllm-sr/d3 --n 200
kill %1 %2
```

- **PASS** (at most a near-tie flip or two): restart vLLM alone with its default memory setting
  and run the benchmark:
  ```bash
  scripts/serve_d3_vllm.sh &
  scripts/wait_for.sh d3
  scripts/bench_system.sh d3 "vllm $(.venv-vllm/bin/python -c 'import vllm;print(vllm.__version__)'), max-model-len 32768"
  ```
- **INVESTIGATE:** benchmark the AutoModel server instead. Point `systems.d3.url` at port 8001
  and say in the article that its throughput is single-stream, so it is a floor.

### 5. Prompted LLM baseline

The baseline is `Qwen/Qwen3.8-27B` in BF16, the base model d3 was fine-tuned from, so the
comparison is the same model prompted against the same model trained for decisions. The NVFP4
checkpoint lists Blackwell as its only supported hardware, so it isn't used on the H200.

```bash
scripts/serve_llm.sh Qwen/Qwen3.8-27B &
scripts/wait_for.sh llm-baseline
scripts/bench_system.sh llm-baseline "vllm $(.venv-vllm/bin/python -c 'import vllm;print(vllm.__version__)'), Qwen/Qwen3.8-27B bf16, thinking off, A/B logprobs"
```

- The baseline answers with one letter, and its P(block) comes from the A/B token logprobs.
- If the model thinks before answering, keep `enable_thinking: false` (Qwen-style templates) or
  pick a non-reasoning instruct model. The smoke test fails if no A/B logprobs come back.
- The baseline answers only the block question, not the category question.

### 6. Prices, then analysis

`config.yaml` already holds the prices confirmed on 2026-11-13 from DO's
[Inference Pricing](https://docs.digitalocean.com/products/inference/details/pricing/) page:
- Jev: $0.042 per million input tokens, no output price.
- H200: $4.47 per hour.

Recheck them on the run date, then run:

```bash
python -m bench.analyze          # writes report/summary.md, metrics.json and two charts
```

## Reading the cost numbers honestly

With one GPU, the crossover is the GPU's daily cost divided by Jev's cost per decision. It is
the same for every self-hosted model until a model needs a second GPU at peak.
- **Throughput:** the crossover separates the models only at high volume or under spiky traffic,
  where peak-sized fleets cost more.
- **Small models:** a 2B model uses a fraction of an H200, and co-locating several models or
  partitioning the GPU would lower its cost. Say so rather than modeling it, since we didn't measure it.
- **Escalations:** they cost a human's time. The report prices them at `human_review_usd`
  (default $0.50, an assumption to state) so systems that escalate more pay for it.

## Layout

```
config.yaml                 systems, datasets, load levels, policy, prices
bench/questions.py          the frozen question schema (SCHEMA_VERSION)
bench/prepare_data.py       download + normalize both datasets
bench/client.py             System One and chat-logprob clients (stdlib, keep-alive)
bench/run.py                smoke | accuracy (resumable) | load
bench/crosscheck.py         compare two endpoints serving the same model
bench/analyze.py            metrics, calibration, policy, cost, charts, summary.md
bench/metrics.py            AUROC, ECE, temperature scaling, thresholds, group bootstrap
bench/manifest.py           GPU, driver, model revision, runtime per system
servers/hf_systemone_server.py   System One server for HF models, with micro-batching
scripts/                    setup_*, serve_*, wait_for, bench_system
tests/                      fake servers + end-to-end test (python -m tests.test_pipeline)
results/<system>/           raw JSONL for every request; manifest.json
report/                     summary.md, metrics.json, reliability_d2_test.png, cost_vs_volume.png
```

Publish `results/` with the article. Every number in `report/` can be recomputed from it.
