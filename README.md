# LLM Insight Platform

**A local-first workbench for inspecting how open-weight language models produce their next token.** Run inference, inspect token scores and selected internals, inventory model parameters, compare layer-ablation experiments, and run a small repeatable prompt suite.

> **Current release: v0.1.0.** This is a working single-user local pilot with an inspectable demo mode. It is not yet a multi-tenant or compliance-certified enterprise service. See [Security and production boundary](#security-and-production-boundary).

## What it does

| Workspace | Capability | Evidence shown |
|---|---|---|
| Token trace | Inspect prompt tokens, per-step top candidates, logits, probabilities, entropy, and generated token | Captured from the local model's output scores |
| Weight explorer | Browse parameter names, tensor shapes, dtypes, counts, and L2 norms; request a 12×12 numeric slice | Parameter inventory and explicitly requested small tensor samples |
| Feature atlas | Inspect active hidden-state dimensions at the final prompt position | Neuron-level activation proxies, not semantic feature claims |
| Interventions | Zero one transformer block's final-position output and compare next-token scores | Controlled sensitivity experiment for one prompt and intervention |
| Evaluation | Run a fixed four-prompt smoke suite with greedy next-token prediction | Prompt/output records for quick comparison |
| Model settings | Load compatible model and tokenizer files from a local directory | Offline-only local inference; remote model downloads disabled |

All views work from the same latest trace. The default demo mode uses **illustrative simulated data**, clearly labeled in the UI; it does not claim to inspect model weights.

## Quick start

Requires Python 3.10+ (3.12 recommended). The model remains on your machine; the first `pip install` needs package access unless dependencies are preinstalled.

```bash
cd llm-insight-platform
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m app.main
```

Open <http://127.0.0.1:8000>. The demo opens without a model. To use real weights, select **Connect local model** and enter the directory containing a compatible Hugging Face causal language model and tokenizer. Set `device` to Auto, CPU, or CUDA.

The loader uses `local_files_only=True` and `trust_remote_code=False`; it will not fetch weights from the internet or execute custom model code. Model format, hardware, and license compatibility remain the operator's responsibility. Do not assume a model license permits every commercial use.

### GPU setup

Install the PyTorch build that matches your operating system, GPU, and CUDA runtime by following the official PyTorch installation selector. Then install the remaining packages:

```bash
python -m pip install -r requirements-web.txt
python -m pip install 'transformers>=4.45,<6.0' safetensors
```

Check `/api/health` for CUDA availability. Large models need substantial RAM or GPU memory; begin with a small model that fits your device.

### Docker

The image starts the service on port 8000. Mount your existing model directory read-only. The server listens on all container interfaces, so expose it only behind a trusted network boundary.

```bash
docker build -t llm-insight-platform .
docker run --rm -p 127.0.0.1:8000:8000 \\
  -v /absolute/path/to/model:/models/model:ro \\
  llm-insight-platform
```

Then connect to `/models/model` in the UI. The container sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## How a next-token trace works

1. The tokenizer converts the prompt into token IDs. A token can be a word, part of a word, punctuation, or whitespace.
2. The model processes those IDs and computes a vocabulary-sized vector of raw scores called **logits** at the final position.
3. Softmax converts logits into a probability distribution. The UI reports the top candidates and entropy for the distribution.
4. At temperature `0.05`, the runtime selects the maximum-logit token. At higher temperatures, it samples from the temperature-adjusted distribution.
5. The selected token is appended to the context and the process repeats for the requested number of steps.

This implementation prioritizes a transparent, easy-to-run reference flow. It recomputes each step without a KV cache and limits generation to 64 tokens; long prompts and larger models can be slow. It captures hidden states and attention summaries only when requested. Capturing internals increases compute and memory use.

## Architecture

```text
Browser UI
   │ same-origin JSON API
FastAPI service ── local Transformers adapter ── model files on disk
   ├── trace and candidate-token analysis
   ├── parameter inventory
   ├── activation dimension summaries
   ├── final-position block ablation
   └── fixed evaluation prompt suite
```

The first adapter targets common Hugging Face causal language model layouts with discoverable transformer blocks (`model.layers`, `transformer.h`, or `model.decoder.layers`). Unsupported layouts can run token traces if their model outputs are compatible, but block ablation may be unavailable. The API reports an error when the adapter cannot perform the requested operation.

## API

- `GET /api/health` — app and local runtime status
- `GET /api/models` — model support and loaded model metadata
- `POST /api/load` — load a local model directory
- `POST /api/run` — run inference and return trace evidence
- `GET /api/weights?limit=120` — inspect parameter metadata
- `GET /api/weights/slice?name=...&rows=12&columns=12` — inspect a bounded parameter-value slice
- `POST /api/intervene` — compare baseline with one block ablation
- `GET /api/docs` — interactive OpenAPI documentation

Example:

```bash
curl -s http://127.0.0.1:8000/api/run \\
  -H 'Content-Type: application/json' \\
  -d '{"prompt":"A transformer predicts the next token by", "max_new_tokens":4, "temperature":0.05}'
```

## Repository map

```text
app/main.py                 FastAPI API, offline model adapter, analysis routines
app/static/index.html       Workbench UI
app/static/styles.css       Responsive visual system
app/static/app.js           UI state and API interactions
tests/                      Backend smoke tests
Dockerfile                  Container build
requirements*.txt           Runtime and test dependencies
```

## Security and production boundary

This version is designed for a **single operator on a trusted local machine**. It binds to `127.0.0.1` when run with `python -m app.main`. The Docker example publishes only to host loopback. Do not expose it directly to a shared network or the public internet: the current release does not implement user authentication, tenant isolation, enterprise SSO, or encrypted trace storage.

Before a multi-user or enterprise deployment, add and validate:

- SSO/OIDC, role-based access, tenant isolation, and audit-log retention controls.
- Network policy, TLS termination, secrets management, and a hardened deployment environment.
- Prompt/trace redaction, encryption at rest, configurable retention, and explicit access approval for captured internals.
- Resource limits for model loading, prompt size, concurrency, and activation capture.
- Model provenance, license review, artifact checksums, and a signed model registry.
- Load, security, privacy, and recovery testing against the target environment.

The product deliberately separates measured evidence from interpretation. Attention weights, hidden-state dimensions, and parameter norms do **not** constitute complete semantic explanations. The layer ablation is a single controlled intervention; it measures sensitivity for that prompt and setup, not a universal causal account of model behavior. The feature atlas currently lists high-magnitude activation dimensions and does not train or load sparse autoencoders.

## Development and tests

```bash
python -m pip install -r requirements-web.txt
python -m pip install pytest httpx
pytest -q
python -m compileall -q app
node --check app/static/app.js
```

See [SECURITY.md](SECURITY.md) for vulnerability reporting and deployment notes.

## License

The application source is released under the MIT License. Model weights, tokenizers, and datasets have their own licenses and are not included in this repository.
