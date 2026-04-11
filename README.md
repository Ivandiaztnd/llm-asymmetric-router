# llm-asymmetric-router

> **Heterogeneous LLM inference with dynamic load balancing on edge hardware.**  
> Run two LLM nodes in parallel — GPU for low-latency interaction, CPU for high-throughput batch — with a heuristic router that dispatches every prompt to the right node automatically.


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Ollama](https://img.shields.io/badge/backend-Ollama-green.svg)](https://ollama.com)
[![THOR Project](https://img.shields.io/badge/part_of-THOR_Project-orange.svg)](https://github.com/ivandiaztnd/thor-autorecon)

---

## The Problem

Running LLMs locally on consumer hardware forces a hard trade-off:

- **Interactive tasks** (chat, strategy, Q&A) need low Time-To-First-Token (TTFT)
- **Batch tasks** (log parsing, scan summarisation, CVE correlation) need large context windows

A single Ollama instance can't optimise for both simultaneously. When processing a 10K-token Nmap log, your interactive model stalls. This project solves that.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OPERATOR PROMPT                    │
└───────────────────────┬─────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Heuristic Router  │  ← semantic classifier
              │  + Load Monitor    │  ← VRAM / TTFT polling
              └────┬──────────┬────┘
                   │          │
        ┌──────────▼──┐  ┌────▼──────────┐
        │  NODE ALPHA  │  │   NODE BETA   │
        │  GPU/Vulkan  │  │   CPU/llama   │
        │  port 11434  │  │  port 11435   │
        │  ctx: 2048   │  │  ctx: 4096+   │
        │  LOW LATENCY │  │  HIGH CONTEXT │
        └──────────────┘  └───────┬───────┘
                                  │
                         ┌────────▼────────┐
                         │  Qdrant Vector  │
                         │  DB (port 6333) │
                         └─────────────────┘
```

### Node Alpha — GPU (Interactive)
- **Backend:** Ollama + Vulkan (stable on Polaris/RDNA1/RDNA2)
- **Model:** Gemma 2:2b Q4_K_M (or any ~2–4B model)
- **Context:** 2048 tokens — kept small for sub-300ms TTFT
- **Handles:** Strategy, Q&A, exploit reasoning, operator interaction

### Node Beta — CPU (Batch)
- **Backend:** Ollama CPU (`num_thread` = physical cores)
- **Model:** Same model, independent instance
- **Context:** 4096+ tokens — processes large inputs without VRAM pressure
- **Handles:** Log summarisation, SmartChunking, CVE parsing, RAG pre-processing

---

## The Router

The router runs **before** any inference token is generated. Two-stage pipeline:

### Stage 1 — Load Monitor
Polls GPU VRAM utilisation. If VRAM > 90% or measured TTFT > 800ms threshold, all traffic is force-routed to Node Beta regardless of semantic classification.

### Stage 2 — Semantic Classifier
Pattern-matches prompt tokens against a configurable ruleset:

| Keywords | Route | Reason |
|----------|-------|--------|
| `DEBUG` `LOG` `SCAN_RESULT` `NMAP` `NUCLEI` `CVE` | Node Beta (CPU) | High token count, latency-tolerant |
| `EXPLOIT` `STRATEGY` `QUESTION` `EXPLAIN` `ANALYZE` | Node Alpha (GPU) | Low token count, latency-critical |
| `CHUNK` `SUMMARIZE` `COMPRESS` `BATCH` | Node Beta (CPU) | Batch semantics |

Rules are fully configurable via `config/routing_rules.json`.

---

## SmartChunking

Large inputs (logs, scan results, codebases) are segmented by **semantic boundaries** rather than fixed character counts:

- C struct / function definitions
- JSON object closings
- Nmap port blocks  
- Network payload delimiters

Each chunk is summarised independently by Node Beta, producing a dense synthesis that fits in Node Alpha's 2048-token context. The full chunk summaries are also embedded into Qdrant for session-persistent RAG.

---

## Quick Start

### Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed
- Docker (for Qdrant) — optional but recommended
- GPU: any Vulkan-capable card (AMD Polaris / RDNA / NVIDIA)

### Install

```bash
git clone https://github.com/ivandiaztnd/llm-asymmetric-router
cd llm-asymmetric-router
pip install -r requirements.txt
```

### Pull the model

```bash
ollama pull gemma2:2b
```

### Launch the two Ollama nodes

```bash
# Node Alpha — GPU (Vulkan), port 11434
OLLAMA_HOST=0.0.0.0:11434 ollama serve &

# Node Beta — CPU only, port 11435
OLLAMA_HOST=0.0.0.0:11435 CUDA_VISIBLE_DEVICES="" ollama serve &
```

### Launch Qdrant (optional, enables persistent RAG)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit endpoints, thresholds, routing rules as needed
```

### Run

```python
import asyncio
from router import AsymmetricRouter

async def main():
    router = AsymmetricRouter.from_config("config/config.yaml")
    
    # Interactive prompt → auto-routed to GPU node
    response = await router.dispatch("Explain CVE-2024-1234 exploitation vector")
    print(response)
    
    # Batch prompt → auto-routed to CPU node
    response = await router.dispatch("SCAN_RESULT: [large nmap output here]")
    print(response)

asyncio.run(main())
```

---

## Configuration

`config/config.yaml`:

```yaml
nodes:
  alpha:
    endpoint: "http://localhost:11434"
    model: "gemma2:2b"
    context_window: 2048
    backend: "vulkan"  # or "cuda", "cpu"
  beta:
    endpoint: "http://localhost:11435"
    model: "gemma2:2b"
    context_window: 4096
    backend: "cpu"

router:
  vram_threshold: 0.90        # Force route to CPU above this VRAM %
  ttft_threshold_ms: 800      # Force route to CPU if GPU TTFT exceeds this
  fallback_node: "beta"       # Default node if classification fails

qdrant:
  enabled: true
  host: "localhost"
  port: 6333
  collection: "llm_router_memory"

chunking:
  strategy: "semantic"        # "semantic" or "fixed"
  max_chunk_tokens: 512
  overlap_tokens: 64
```

---

## Repo Structure

```
llm-asymmetric-router/
├── router/
│   ├── __init__.py
│   ├── core.py           # AsymmetricRouter main class
│   ├── classifier.py     # Semantic prompt classifier
│   ├── load_monitor.py   # VRAM / TTFT polling
│   └── chunker.py        # SmartChunking engine
├── config/
│   ├── config.example.yaml
│   └── routing_rules.json
├── examples/
│   ├── basic_routing.py
│   ├── pentest_workflow.py
│   └── log_batch_processing.py
├── docs/
│   └── whitepaper.pdf
├── tests/
├── requirements.txt
└── README.md
```

---

## Hardware Tested

| GPU | VRAM | Vulkan | Notes |
|-----|------|--------|-------|
| AMD RX 5700 | 8 GB | ✅ | Primary dev hardware |
| AMD RX 550 | 4 GB | ✅ | Minimum viable config |
| AMD RX 580 | 8 GB | ✅ | Polaris — stable |
| NVIDIA GTX 1060 | 6 GB | ✅ | via cuda backend |

> **Note:** ROCm on Windows with Polaris/RDNA1 is unstable. Vulkan backend is the recommended fallback for all AMD cards on Windows.

---

## Integration with THOR

This module is the AI inference backbone of the [THOR autorecon framework](https://github.com/ivandiaztnd/thor-autorecon). In THOR's context:

- **Node Alpha** drives real-time operator interaction and attack strategy
- **Node Beta** processes Nmap/Nuclei/OSINT outputs asynchronously
- **Qdrant** stores scan session memory for cross-phase context

---

## Whitepaper

Full technical documentation including performance benchmarks and design rationale:  
📄 [`Docs/whitepaper.pdf`](Docs/whitepaper.pdf)

---

## License

GNU Gpl v3
Built by [@ivandiaztnd](https://github.com/ivandiaztnd) — THOR Project, 2026.
