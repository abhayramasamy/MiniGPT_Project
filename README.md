# miniGPT / nanoLLAMA `v0.0.1` — Pre-release

> A from-scratch, small-scale implementation of the Llama architecture (RMSNorm, RoPE, SwiGLU, decoder-only causal attention), trained on TinyStories and served via a containerized Flask inference API.

This is the **first public pre-release**. It is a working, end-to-end system — trained model, inference server, developer diagnostics — but should be treated as an early checkpoint, not a finished product.

---

## About the model

- **Architecture**: Llama-style decoder-only transformer — RMSNorm (not LayerNorm), RoPE positional encoding (not learned/absolute), SwiGLU feed-forward (not ReLU/GELU MLP), causal self-attention
- **Size**: ~85M parameters (~47M trainable — embeddings are frozen, tied GPT-2 embedding table)
- **Layers**: 5 decoder blocks · `d_model=768` · `12` attention heads · FFN hidden dim `3072`
- **Context length**: 256 tokens
- **Tokenizer**: `openai-community/gpt2` (BPE, vocab size 50,257)
- **Training data**: TinyStories subset (~100M tokens), 3 epochs, mixed precision
- **What it's good at**: short, grammatically coherent children's-story-style text
- **What it's *not***: factual, knowledgeable, or good at long-range reasoning — this is a capacity-scale limitation, not a bug

---

## Get the model

**Option A — from this release**
Download `tinystories_50m_model.weights.h5` from the **Releases** page of this repo (attached to `v0.0.1`).

**Option B — from Hugging Face**<br/>
Also see Model card on Hugging face to know important model details, You can download the optimizer.npz file too incase you are planning to train/finetune.<br/> 
*Visit my Hugging face Repo: :*
`[https://huggingface.co/ARX1A07/miniGPT_Project/tree/main]`

Either way, once downloaded:
```
miniGPT-server/
└── models/
    └── tinystories_50m_model.weights.h5   ← place the file here
```

**See how it was built**: <br/>
Observe the google colab notebook on how the model was built, trained the individual architectures`[https://colab.research.google.com/drive/1DUSM24y3hcrc06BduC47kyWnciu8OGZg?usp=sharing]`

---

## Setup

### 1. Clone and install
```bash
git clone <this-repo-url>
cd miniGPT-server
pip install -r requirements.txt
```

### 2. Place weights
Put `tinystories_50m_model.weights.h5` inside `models/` as shown above.

### 3. Run locally
```bash
python app.py
```

### 4. Health check
```bash
curl http://localhost:5000/
```

---

## Docker

### Build
```bash
docker build -t minigpt-server .
```

### Run
```bash
docker run -p 5000:5000 minigpt-server
```

### Test
```bash
curl http://localhost:5000/
```

> GPU variant: swap `tensorflow-cpu` → `tensorflow` in `requirements.txt` and use a CUDA-enabled base image if running on GPU hardware. TensorFlow will auto-detect the GPU — no code changes required.

---

## API Reference

Base URL: `http://localhost:5000`

### Public API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms server + model are up |
| `POST` | `/generate` | Generate text from a prompt. **Streams** the response as plain text, token by token |

**`POST /generate` — request body**
```json
{
  "prompt": "A wise crow",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9
}
```
| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | `""` | Input text |
| `max_new_tokens` | int | 50 | Tokens to generate |
| `temperature` | float | 0.7 | Lower = more predictable, higher = more chaotic |
| `top_p` | float | 0.9 | Nucleus sampling cutoff |

---

### Developer API (`/dev/*`)

Not intended for production/public exposure — local diagnostics only, no auth in this pre-release.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/dev/health` | Model load status, param count, context length, dtype, device (CPU/GPU), uptime |
| `GET` | `/dev/stats` | Running in-memory stats: request counts, avg tokens/sec, avg latency, avg time-to-first-token |
| `POST` | `/dev/generate` | Same as `/generate` but non-streaming — returns full text **plus** timing/token metrics in one JSON response |
| `POST` | `/dev/evaluate` | Teacher-forced evaluation: scores a supplied `target` continuation against a `prompt`, returns loss/perplexity/token-level log-probs |

**`POST /dev/evaluate` — request body**
```json
{
  "prompt": "The cat sat on",
  "target": "the mat"
}
```

> **Note**: `/dev/evaluate` computes perplexity against a *fixed target continuation*, not the model's own sampled output — these are not equivalent, and this endpoint should not be used to claim generation-quality perplexity.

---

## API docs UI (Swagger)

Not yet included in `v0.0.1`. Planned via `flasgger` or an OpenAPI spec + Swagger UI, so all endpoints above are interactively browsable/testable at `/apidocs`. Tracked as a near-term follow-up (see below).

---

## Notes for the future

- [ ] Add Swagger/OpenAPI docs (`/apidocs`)
- [ ] Swap Flask dev server → gunicorn/waitress for real concurrent-request handling
- [ ] Build a minimal streaming frontend + a live `/dev/stats` dashboard
- [ ] GPU-enabled Docker image (`Dockerfile.gpu`) as a first-class variant
- [ ] Lightweight CI: run test suite + build image on push
- [ ] Simple "push new model version" flow: upload weights → auto-run test suite → promote on pass
- [ ] Longer-context / larger-parameter follow-up model, once infra above is solid

---

*This is a pre-release. Expect rough edges — that's the point of `v0.0.1`.*
