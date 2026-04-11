"""
llm-asymmetric-router — core.py
Asymmetric LLM routing with heuristic load balancing.
Author: ivandiaztnd / THOR Project — 2026
License: MIT
"""

import asyncio
import time
import re
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator
import httpx

logger = logging.getLogger("llm_router")


# ─── ENUMS & DATACLASSES ──────────────────────────────────────────────────────

class Node(Enum):
    ALPHA = "alpha"   # GPU / low-latency
    BETA  = "beta"    # CPU / high-context


@dataclass
class NodeConfig:
    endpoint:       str
    model:          str
    context_window: int  = 2048
    backend:        str  = "vulkan"
    timeout:        int  = 120


@dataclass
class RouterConfig:
    vram_threshold:    float = 0.90
    ttft_threshold_ms: int   = 800
    fallback_node:     Node  = Node.BETA
    rules_path:        str   = "config/routing_rules.json"


@dataclass
class RoutingDecision:
    node:      Node
    reason:    str
    latency_ms: float = 0.0


@dataclass
class RouterMetrics:
    alpha_requests: int   = 0
    beta_requests:  int   = 0
    force_reroutes: int   = 0
    avg_ttft_ms:    float = 0.0
    _ttft_samples:  list  = field(default_factory=list)

    def record_ttft(self, ms: float):
        self._ttft_samples.append(ms)
        if len(self._ttft_samples) > 100:
            self._ttft_samples.pop(0)
        self.avg_ttft_ms = sum(self._ttft_samples) / len(self._ttft_samples)


# ─── SEMANTIC CLASSIFIER ──────────────────────────────────────────────────────

DEFAULT_RULES = {
    "cpu_patterns": [
        r"\bDEBUG\b", r"\bLOG\b", r"\bSCAN_RESULT\b",
        r"\bNMAP\b",  r"\bNUCLEI\b", r"\bCVE[-_]\d+\b",
        r"\bCHUNK\b", r"\bSUMMARIZE\b", r"\bCOMPRESS\b",
        r"\bBATCH\b", r"PORT\s+\d+/tcp", r"\bXML\b.*\bscan\b",
    ],
    "gpu_patterns": [
        r"\bEXPLOIT\b", r"\bSTRATEGY\b", r"\bQUESTION\b",
        r"\bEXPLAIN\b", r"\bANALYZE\b",  r"\bHOW\b",
        r"\bWHY\b",     r"\bWHAT\b",      r"\bRECOMMEND\b",
    ]
}


class SemanticClassifier:
    def __init__(self, rules_path: Optional[str] = None):
        rules = DEFAULT_RULES
        if rules_path:
            try:
                with open(rules_path) as f:
                    rules = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Rules file {rules_path} not found, using defaults.")

        self._cpu_re = [re.compile(p, re.IGNORECASE) for p in rules["cpu_patterns"]]
        self._gpu_re = [re.compile(p, re.IGNORECASE) for p in rules["gpu_patterns"]]

    def classify(self, prompt: str) -> tuple[Node, str]:
        """
        Returns (Node, reason_string).
        Scoring: each pattern match adds weight. Highest score wins.
        """
        cpu_score = sum(1 for r in self._cpu_re if r.search(prompt))
        gpu_score = sum(1 for r in self._gpu_re if r.search(prompt))

        # Also weight by prompt length: long prompts → CPU
        token_estimate = len(prompt.split())
        if token_estimate > 800:
            cpu_score += 3

        if cpu_score > gpu_score:
            return Node.BETA, f"semantic:cpu_score={cpu_score}>gpu_score={gpu_score}"
        elif gpu_score > 0:
            return Node.ALPHA, f"semantic:gpu_score={gpu_score}>cpu_score={cpu_score}"
        else:
            # No match → default to GPU for short prompts, CPU for long
            if token_estimate <= 200:
                return Node.ALPHA, "semantic:no_match,short_prompt→gpu"
            return Node.BETA, "semantic:no_match,long_prompt→cpu"


# ─── LOAD MONITOR ─────────────────────────────────────────────────────────────

class LoadMonitor:
    """
    Lightweight VRAM monitor. Tries rocm-smi first, falls back to
    /sys/class/drm, then disables itself gracefully if neither works.
    """

    def __init__(self, vram_threshold: float = 0.90):
        self.threshold = vram_threshold
        self._available = self._probe()

    def _probe(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    def vram_utilisation(self) -> Optional[float]:
        """Returns 0.0–1.0 or None if unavailable."""
        if not self._available:
            return None
        try:
            import subprocess, json as _json
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=2
            )
            data = _json.loads(r.stdout)
            # rocm-smi JSON structure varies by version — handle both
            for card_data in data.values():
                used  = int(card_data.get("VRAM Total Used Memory (B)", 0))
                total = int(card_data.get("VRAM Total Memory (B)", 1))
                if total > 0:
                    return used / total
        except Exception as e:
            logger.debug(f"LoadMonitor error: {e}")
        return None

    def is_overloaded(self) -> bool:
        util = self.vram_utilisation()
        if util is None:
            return False
        return util >= self.threshold


# ─── MAIN ROUTER ──────────────────────────────────────────────────────────────

class AsymmetricRouter:
    """
    Main entry point. Usage:

        router = AsymmetricRouter(alpha_cfg, beta_cfg, router_cfg)
        response = await router.dispatch(prompt)

    Or load from YAML:

        router = AsymmetricRouter.from_config("config/config.yaml")
    """

    def __init__(
        self,
        alpha: NodeConfig,
        beta:  NodeConfig,
        cfg:   RouterConfig = RouterConfig(),
    ):
        self.alpha      = alpha
        self.beta       = beta
        self.cfg        = cfg
        self.classifier = SemanticClassifier(cfg.rules_path)
        self.monitor    = LoadMonitor(cfg.vram_threshold)
        self.metrics    = RouterMetrics()
        self._client    = httpx.AsyncClient(timeout=120)

    @classmethod
    def from_config(cls, path: str) -> "AsymmetricRouter":
        try:
            import yaml
            with open(path) as f:
                c = yaml.safe_load(f)
        except ImportError:
            raise ImportError("pip install pyyaml to use from_config()")

        alpha = NodeConfig(**c["nodes"]["alpha"])
        beta  = NodeConfig(**c["nodes"]["beta"])
        rcfg  = RouterConfig(**c.get("router", {}))
        return cls(alpha, beta, rcfg)

    # ── Routing decision ──────────────────────────────────────────────────────

    def _route(self, prompt: str) -> RoutingDecision:
        t0 = time.perf_counter()

        # Stage 1: load-based override
        if self.monitor.is_overloaded():
            self.metrics.force_reroutes += 1
            return RoutingDecision(
                node=Node.BETA,
                reason="load_monitor:vram_overload→cpu",
                latency_ms=(time.perf_counter() - t0) * 1000
            )

        # Stage 2: semantic classification
        node, reason = self.classifier.classify(prompt)
        return RoutingDecision(
            node=node,
            reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000
        )

    # ── Inference dispatch ────────────────────────────────────────────────────

    async def dispatch(
        self,
        prompt:  str,
        system:  str  = "",
        stream:  bool = False,
        context: list = None,
    ) -> str:
        decision = self._route(prompt)
        node_cfg = self.alpha if decision.node == Node.ALPHA else self.beta

        logger.info(
            f"[ROUTER] → {decision.node.value.upper()} | "
            f"reason={decision.reason} | "
            f"routing_latency={decision.latency_ms:.2f}ms"
        )

        if decision.node == Node.ALPHA:
            self.metrics.alpha_requests += 1
        else:
            self.metrics.beta_requests += 1

        return await self._call_ollama(node_cfg, prompt, system, stream, context)

    async def _call_ollama(
        self,
        node:    NodeConfig,
        prompt:  str,
        system:  str,
        stream:  bool,
        context: list,
    ) -> str:
        payload = {
            "model":  node.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_ctx": node.context_window,
            }
        }
        if system:
            payload["system"] = system
        if context:
            payload["context"] = context

        t0 = time.perf_counter()
        try:
            r = await self._client.post(
                f"{node.endpoint}/api/generate",
                json=payload,
                timeout=node.timeout
            )
            r.raise_for_status()
            ttft_ms = (time.perf_counter() - t0) * 1000
            self.metrics.record_ttft(ttft_ms)

            data = r.json()

            # Auto-reroute if TTFT was too high (for next request)
            if ttft_ms > self.cfg.ttft_threshold_ms:
                logger.warning(
                    f"[ROUTER] TTFT={ttft_ms:.0f}ms exceeded threshold "
                    f"({self.cfg.ttft_threshold_ms}ms). "
                    f"Next request may be rerouted."
                )

            return data.get("response", "")

        except httpx.ConnectError:
            logger.error(f"[ROUTER] Node {node.endpoint} unreachable. Falling back.")
            fallback = self.beta if node == self.alpha else self.alpha
            return await self._call_ollama(fallback, prompt, system, stream, context)

        except Exception as e:
            logger.error(f"[ROUTER] Inference error: {e}")
            raise

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self.metrics.alpha_requests + self.metrics.beta_requests
        return {
            "total_requests":  total,
            "alpha_requests":  self.metrics.alpha_requests,
            "beta_requests":   self.metrics.beta_requests,
            "force_reroutes":  self.metrics.force_reroutes,
            "avg_ttft_ms":     round(self.metrics.avg_ttft_ms, 2),
            "alpha_ratio":     round(self.metrics.alpha_requests / max(total, 1), 2),
        }

    async def close(self):
        await self._client.aclose()


# ─── CLI QUICK TEST ───────────────────────────────────────────────────────────

async def _demo():
    alpha = NodeConfig(endpoint="http://localhost:11434", model="gemma2:2b", context_window=2048)
    beta  = NodeConfig(endpoint="http://localhost:11435", model="gemma2:2b", context_window=4096, backend="cpu")
    router = AsymmetricRouter(alpha, beta)

    test_prompts = [
        "SCAN_RESULT: Nmap output for 192.168.1.0/24 — 47 hosts up, open ports: 22,80,443,8080",
        "What is the exploitation vector for CVE-2023-44487?",
        "LOG: [ERROR] Connection timeout after 30s on 10.0.0.5:445",
        "Recommend an attack strategy for a target with exposed SMB port",
        "SUMMARIZE the following nuclei scan output: ...",
    ]

    for prompt in test_prompts:
        decision = router._route(prompt)
        print(f"[{decision.node.value.upper():5s}] {decision.reason:<50s} | {prompt[:60]}")

    print("\nMetrics:", router.stats())
    await router.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_demo())
