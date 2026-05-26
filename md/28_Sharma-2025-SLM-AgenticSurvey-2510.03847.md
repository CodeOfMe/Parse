# Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade offs

**Authors:** Raghav Sharma, Manan Mehta

---

## Abstract

Small language models (SLMs; 1-12B params, sometimes up to 20B) are sufficient and often superior for agentic workloads where the objective is schema- and API-constrained accuracy rather than open-ended generation. We synthesize recent evidence across open and proprietary SLMs (Phi-4-Mini, Qwen-2.5-7B, Gemma-2-9B, Llama-3.2-1B/3B, Ministral-3B/8B, Apple on-device 3B, DeepSeek-R1-Distill) and connect it to modern evaluations (BFCL v3/v4, StableToolBench) and serving stacks (vLLM, SGLang, TensorRT-LLM) paired with guided decoding libraries (XGrammar, Outlines). We formalize SLM-default, LLM-fallback systems with uncertainty-aware routing and verifier cascades, and propose engineering metrics that reflect real production goals: cost per successful task (CPS), schema validity rate, executable call rate, p50/p95 latency, and energy per request. Guided decoding, strict JSON Schema outputs, and validator-first tool execution close much of the capability gap with larger models and often let SLMs match or surpass LLMs on tool use, function calling, and RAG at 10x-100x lower token cost with materially better latency and energy. We provide design patterns for agent stacks that prioritize SLMs: schema-first prompting, type-safe function registries, confidence scoring with verifier rollups, and lightweight adaptation via LoRA/QLoRA. We also delineate limits where fallback remains valuable (open-domain reasoning and some long-horizon planning). The result is a practical blueprint for building fast, inexpensive, and reliable agents that default to SLMs while preserving headroom with targeted LLM assistance.
  Keywords: small language models, agents, function calling, structured outputs, JSON Schema, guided decoding, LoRA/QLoRA, routing, energy efficiency, edge inference

---

- **arXiv:** [2510.03847](https://arxiv.org/abs/2510.03847)
- **Published:** 2025-10-04
- **Categories:** cs.AI, cs.LG
- **PDF:** [SLM-AgenticSurvey-arXiv-2510.03847v1.pdf](SLM-AgenticSurvey-arXiv-2510.03847v1.pdf)

---

*Generated from arXiv API metadata. See TeX source for full paper content.*