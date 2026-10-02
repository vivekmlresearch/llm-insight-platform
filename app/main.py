"""LLM Insight Platform: local-first model inspection and audit workbench."""
from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).parent
app = FastAPI(title="LLM Insight Platform", version="0.1.0", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
_MODEL: Any = None
_TOKENIZER: Any = None
_MODEL_INFO: dict[str, Any] = {"loaded": False, "mode": "demo"}
_LOAD_LOCK = asyncio.Lock()


class LoadRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=1000)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=8, ge=1, le=64)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)
    top_k: int = Field(default=10, ge=1, le=50)
    capture_internals: bool = True


class InterventionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    layer: int = Field(ge=0)


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    try:
        import torch
        torch_ok = True
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_ok, cuda = False, False
    return {"status": "ok", "torch_available": torch_ok, "cuda_available": cuda,
            "offline_only": True, "model": _MODEL_INFO}


@app.get("/api/models")
def models():
    return {"loaded": _MODEL_INFO, "supported": [
        {"family": "Hugging Face Transformers causal language models", "capabilities": [
            "token logits", "hidden states", "attention summaries", "weight inventory", "block ablation"]}
    ], "note": "Load local model files only. Remote model downloads are disabled."}


@app.post("/api/load")
async def load_model(req: LoadRequest):
    global _MODEL, _TOKENIZER, _MODEL_INFO
    model_path = Path(req.model_path).expanduser().resolve()
    if not model_path.exists() or not model_path.is_dir():
        raise HTTPException(400, "Model path must be an existing local directory.")
    async with _LOAD_LOCK:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise HTTPException(503, "Install requirements.txt to enable local model inference.") from exc
        if req.device == "cuda" and not torch.cuda.is_available():
            raise HTTPException(400, "CUDA was requested but is not available.")
        device = req.device if req.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
            mdl = AutoModelForCausalLM.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
                torch_dtype="auto", low_cpu_mem_usage=True,
            ).to(device).eval()
        except Exception as exc:
            raise HTTPException(400, f"Could not load local model: {type(exc).__name__}: {exc}") from exc
        _TOKENIZER, _MODEL = tok, mdl
        _MODEL_INFO = {"loaded": True, "mode": "local", "path": str(model_path),
                       "model_type": getattr(mdl.config, "model_type", "unknown"),
                       "parameters": sum(p.numel() for p in mdl.parameters()),
                       "device": device, "vocab_size": int(mdl.config.vocab_size)}
    return {"ok": True, "model": _MODEL_INFO}


def _blocks(model: Any) -> list[Any]:
    candidates = [getattr(getattr(model, "model", None), "layers", None),
                  getattr(getattr(model, "transformer", None), "h", None),
                  getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)]
    for value in candidates:
        if value is not None:
            return list(value)
    return []


def _top_candidates(logits: Any, count: int) -> list[dict[str, Any]]:
    import torch
    probs = torch.softmax(logits.float(), dim=-1)
    vals, ids = torch.topk(probs, min(count, probs.numel()))
    result = []
    for p, idx in zip(vals.tolist(), ids.tolist()):
        result.append({"token_id": int(idx), "probability": float(p),
                       "logit": float(logits[idx].float().item()),
                       "token": _TOKENIZER.decode([int(idx)], skip_special_tokens=False)})
    return result


def _layer_summaries(outputs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hidden, attention = [], []
    states = getattr(outputs, "hidden_states", None)
    if states:
        for layer_idx, state in enumerate(states):
            v = state[0, -1].float()
            magnitude = v.abs()
            values, indices = v.topk(min(8, v.numel()))
            hidden.append({"layer": layer_idx, "l2_norm": float(v.norm().item()),
                           "mean_abs": float(magnitude.mean().item()),
                           "top_dimensions": [{"dimension": int(i), "activation": float(x)} for x, i in zip(values.tolist(), indices.tolist())]})
    attentions = getattr(outputs, "attentions", None)
    if attentions:
        for layer_idx, attn in enumerate(attentions):
            row = attn[0, :, -1, :].float().mean(dim=0)
            values, indices = row.topk(min(6, row.numel()))
            attention.append({"layer": layer_idx, "top_positions": [{"position": int(i), "weight": float(x)} for x, i in zip(values.tolist(), indices.tolist())]})
    return hidden, attention


def _feature_proxies(hidden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # These are activation-dimension proxies, not SAE-discovered semantic features.
    rows = []
    for layer in hidden:
        for item in layer["top_dimensions"][:4]:
            rows.append({"label": f"L{layer['layer']} · dimension {item['dimension']}",
                         "layer": layer["layer"], "dimension": item["dimension"],
                         "activation": item["activation"], "kind": "neuron proxy",
                         "interpretation": "Activation dimension; no semantic label inferred."})
    return sorted(rows, key=lambda x: abs(x["activation"]), reverse=True)[:20]


def _infer(req: RunRequest, ablate_layer: int | None = None) -> dict[str, Any]:
    import torch
    if _MODEL is None or _TOKENIZER is None:
        raise RuntimeError("Load a local model first.")
    started = time.perf_counter()
    device = next(_MODEL.parameters()).device
    inputs = _TOKENIZER(req.prompt, return_tensors="pt").to(device)
    prompt_length = int(inputs["input_ids"].shape[-1])
    limit = 256 if req.capture_internals else 4096
    if prompt_length > limit:
        raise ValueError(f"Prompt token limit is {limit} for this trace mode; received {prompt_length}.")
    token_ids = inputs["input_ids"][0].tolist()
    layers = _blocks(_MODEL)
    if ablate_layer is not None and not layers:
        raise RuntimeError("This model architecture does not expose supported transformer blocks.")
    if ablate_layer is not None and ablate_layer >= len(layers):
        raise ValueError(f"Layer index must be between 0 and {len(layers)-1}.")
    hook = None
    if ablate_layer is not None:
        def zero_last_token(_module: Any, _inputs: Any, output: Any):
            tensor = output[0] if isinstance(output, tuple) else output
            changed = tensor.clone()
            changed[:, -1, :] = 0
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        hook = layers[ablate_layer].register_forward_hook(zero_last_token)
    steps, last_hidden, last_attention = [], [], []
    try:
        for step in range(req.max_new_tokens):
            input_tensor = torch.tensor([token_ids], device=device)
            capture_now = req.capture_internals and step == 0
            with torch.inference_mode():
                out = _MODEL(input_ids=input_tensor, output_hidden_states=capture_now,
                             output_attentions=capture_now, use_cache=False, return_dict=True)
            logits = out.logits[0, -1]
            candidate = _top_candidates(logits, req.top_k)
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
            next_id = int(torch.argmax(logits).item()) if req.temperature <= 0.051 else int(torch.multinomial(torch.softmax(logits.float()/req.temperature, dim=-1), 1).item())
            token_str = _TOKENIZER.decode([next_id], skip_special_tokens=False)
            steps.append({"step": step + 1, "context": _TOKENIZER.decode(token_ids, skip_special_tokens=False),
                          "selected_token": token_str, "selected_token_id": next_id,
                          "entropy_nats": entropy, "candidates": candidate})
            if capture_now:
                last_hidden, last_attention = _layer_summaries(out)
            token_ids.append(next_id)
            if next_id == _TOKENIZER.eos_token_id:
                break
    finally:
        if hook:
            hook.remove()
    return {"prompt": req.prompt, "generated_text": _TOKENIZER.decode(token_ids[len(inputs['input_ids'][0]):], skip_special_tokens=True),
            "prompt_tokens": [{"id": int(i), "text": _TOKENIZER.decode([int(i)], skip_special_tokens=False)} for i in inputs["input_ids"][0].tolist()],
            "steps": steps, "hidden_layers": last_hidden, "attention_layers": last_attention,
            "feature_proxies": _feature_proxies(last_hidden), "latency_ms": round((time.perf_counter()-started)*1000, 1),
            "model": _MODEL_INFO, "intervention": {"layer_ablated": ablate_layer} if ablate_layer is not None else None,
            "limitations": ["Activation dimensions are not guaranteed to be semantic features.",
                            "Ablation shows sensitivity to this specific intervention; it is not a complete explanation."]}


@app.post("/api/run")
def run(req: RunRequest):
    if _MODEL is None:
        return demo_trace(req)
    try:
        return _infer(req)
    except Exception as exc:
        raise HTTPException(400, f"Inference failed: {type(exc).__name__}: {exc}") from exc


@app.post("/api/intervene")
def intervene(req: InterventionRequest):
    if _MODEL is None:
        return demo_intervention(req)
    try:
        base = _infer(RunRequest(prompt=req.prompt, max_new_tokens=1, temperature=0.05, top_k=8))
        changed = _infer(RunRequest(prompt=req.prompt, max_new_tokens=1, temperature=0.05, top_k=8), ablate_layer=req.layer)
        return {"baseline": base["steps"][0], "intervened": changed["steps"][0],
                "layer": req.layer, "method": "zero final-position block output",
                "note": "Ablation sensitivity is evidence about this intervention, not a full causal explanation."}
    except Exception as exc:
        raise HTTPException(400, f"Intervention failed: {type(exc).__name__}: {exc}") from exc


@app.get("/api/weights")
def weights(limit: int = 120):
    if _MODEL is None:
        return {"mode": "demo", "items": [], "message": "Load a local model to inspect its parameter inventory."}
    result = []
    for name, param in _MODEL.named_parameters():
        result.append({"name": name, "shape": list(param.shape), "dtype": str(param.dtype),
                       "parameters": param.numel(), "l2_norm": round(float(param.detach().float().norm().item()), 4)})
        if len(result) >= max(1, min(limit, 500)):
            break
    return {"mode": "local", "items": result, "total_parameters": _MODEL_INFO.get("parameters")}


@app.get("/api/weights/slice")
def weight_slice(name: str, rows: int = 12, columns: int = 12):
    if _MODEL is None:
        raise HTTPException(409, "Load a local model to inspect parameter values.")
    if not 1 <= rows <= 24 or not 1 <= columns <= 24:
        raise HTTPException(400, "Slice dimensions must be between 1 and 24.")
    parameter = dict(_MODEL.named_parameters()).get(name)
    if parameter is None:
        raise HTTPException(404, "Parameter was not found in the loaded model.")
    tensor = parameter.detach().float().cpu()
    shape = list(tensor.shape)
    if tensor.ndim == 0:
        matrix = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        matrix = tensor.reshape(1, -1)
    else:
        matrix = tensor.reshape(shape[0], -1)
    sample = matrix[:rows, :columns]
    return {"name": name, "shape": shape, "sample_shape": list(sample.shape),
            "values": sample.tolist(), "note": "Top-left parameter slice; values are not a semantic explanation."}


def demo_trace(req: RunRequest) -> dict[str, Any]:
    prompt = req.prompt
    vocab = [" the", " model", " next", " answer", " is", " data", " token", " because"]
    steps = []
    for i in range(min(req.max_new_tokens, 8)):
        shift = (len(prompt) + i * 3) % len(vocab)
        options = vocab[shift:] + vocab[:shift]
        probs = [0.39, 0.22, 0.14, 0.09, 0.06, 0.045, 0.035, 0.02]
        steps.append({"step": i+1, "context": prompt + "".join(s["selected_token"] for s in steps),
                      "selected_token": options[0], "selected_token_id": 100+i,
                      "entropy_nats": 1.54, "candidates": [{"token_id": 100+j, "token": options[j], "probability": probs[j], "logit": round(math.log(probs[j]), 3)} for j in range(min(req.top_k, len(options)))]})
    hidden = [{"layer": i, "l2_norm": round(2.1+i*.37, 3), "mean_abs": round(.08+i*.01, 3),
               "top_dimensions": [{"dimension": 11+i*17+j, "activation": round(1.24-j*.16-i*.03, 3)} for j in range(5)]} for i in range(6)]
    attention = [{"layer": i, "top_positions": [{"position": max(0, len(prompt.split())-1-j), "weight": round(.42-j*.07, 3)} for j in range(5)]} for i in range(6)]
    return {"demo": True, "prompt": prompt, "generated_text": "".join(x["selected_token"] for x in steps),
            "prompt_tokens": [{"id": i+1, "text": w} for i,w in enumerate(prompt.split())],
            "steps": steps, "hidden_layers": hidden, "attention_layers": attention,
            "feature_proxies": _feature_proxies(hidden), "latency_ms": 12.4,
            "model": {"loaded": False, "mode": "demo", "name": "Illustrative trace (not a real model)"},
            "limitations": ["Demo values are illustrative and do not come from model weights.", "Load a local model for real inference and measurements."]}


def demo_intervention(req: InterventionRequest) -> dict[str, Any]:
    base = {"step": 1, "selected_token": " the", "candidates": [{"token": " the", "probability": .39}, {"token": " answer", "probability": .22}]}
    changed = {"step": 1, "selected_token": " answer", "candidates": [{"token": " answer", "probability": .34}, {"token": " the", "probability": .27}]}
    return {"demo": True, "baseline": base, "intervened": changed, "layer": req.layer,
            "method": "illustrative simulated block ablation", "note": "Demo output is illustrative. Load a local model for real interventions."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
