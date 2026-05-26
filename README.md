# PARSE: Principled Architecture Retention through Scenario-Embedded Pruning

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-red)]()

> **Scalpel, Not Sledgehammer** — Fine-grained, capability-preserving compression of Large Language Models along Language × Discipline × Scenario axes.

## 🎯 Overview

PARSE is a framework that redefines model compression as a **precision surgical procedure** rather than brute-force demolition. Unlike uniform pruning (which blindly removes parameters) or task-specific design (which sacrifices broad knowledge), PARSE enables practitioners to specify *exactly which capabilities to preserve* (e.g., "Chinese grammar + English mathematics + function calling") and surgically compresses the model accordingly.

### Core Innovation: Tri-Axial Capability Decomposition

LLM capabilities naturally decompose along three orthogonal axes:
- **Language** (Chinese, English, Japanese, French, German, Russian, Spanish, Korean)
- **Discipline** (Mathematics, Physics, Logic, History, Geography, Literature)
- **Scenario** (Function Calling, Code, Math Reasoning, Translation, Chat)

PARSE quantifies each layer's contribution to every (Language × Discipline × Scenario) combination using a **Capability Importance Tensor (CIT)**. Critical layers are preserved intact; redundant layers receive ultra-efficient No-FFN attention transplants [55]. A lightweight **Dynamic Capability Router (DCR)** (0.08M parameters) modulates internal residual gates based on input context, enabling a single compressed model to serve multiple preservation profiles without weight switching.

## 📊 Key Results

| Metric | Original Qwen3.5-0.8B | PARSE (Ours) |
|:---|:---:|:---:|
| Parameters | 752M | **85M** |
| GSM8K Accuracy | 45.2% | **42.8%** (95% retained) |
| BFCL Accuracy | 88.1% | **88.7%** (100%+ retained) |
| Inference Speed | 1.5 tok/s | **15.4 tok/s** (10× faster) |

## 📁 Repository Structure

```
PARSE/
├── article.md              # English paper (full)
├── article_cn.md           # Chinese paper (full)
├── literature_review.md    # Literature review (English) — 55 references
├── 文献综述.md              # Literature review (Chinese) — 55 references
├── code/                   # Implementation framework
│   └── parse/              # Core architecture + trainer
├── md/                     # Reference papers (Markdown)
├── pdf/                    # Reference papers (PDF)
├── figures/                # Publication-quality SVG/PDF figures
├── results/                # RTX 4060 experiment results
├── models/                 # Model configurations
├── run_experiment.py       # Experiment runner
└── EXPERIMENT.md           # Experiment guide
```

## 🚀 Quick Start

```bash
# Run experiments on CUDA device
python run_experiment.py --device cuda --strategy parse --sparsity 0.5

# Run on Apple Silicon
python run_experiment.py --device mps --strategy parse --sparsity 0.5

# Custom preservation profile
python run_experiment.py \
    --languages zh en \
    --disciplines math logic \
    --scenarios fc math_reasoning \
    --device cuda
```

## 📚 References

This work draws on foundational studies in:
- Structural pruning [3-7,11-16]
- Dynamic inference [8-10]
- Knowledge editing and machine unlearning [37-47]
- Data flywheels and self-improving training [1,17-23]
- Agentic systems and tool calling [24-30]
- GRPO-based reinforcement learning [23,31-36]
- Specialized architectures [2,54,55]

All references are available in `md/` and `pdf/` directories.

## 📄 Citation

```bibtex
@article{parse2026,
    title   = {PARSE: Principled Architecture Retention through Scenario-Embedded Pruning for Fine-Grained Capability-Preserving Language Model Compression},
    author  = {},
    journal = {},
    year    = {2026}
}
```

## 📜 License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
