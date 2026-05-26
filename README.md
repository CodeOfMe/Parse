# PARSE: Principled Architecture Retention through Scenario-Embedded Pruning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-red)]()

> **Scalpel, Not Sledgehammer** — Fine-grained, capability-preserving compression of Large Language Models along Language × Discipline × Scenario axes.

## 🎯 Overview

PARSE is a framework that redefines model compression as a **precision surgical procedure** rather than brute-force demolition. Unlike uniform pruning (which blindly removes parameters) or task-specific design (which sacrifices broad knowledge), PARSE enables practitioners to specify *exactly which capabilities to preserve* (e.g., "Chinese grammar + English mathematics + function calling") and surgically compresses the model accordingly.

### Core Innovation: Tri-Axial Capability Decomposition

LLM capabilities naturally decompose along three orthogonal axes:
- **Language** (Chinese, English, Japanese, French, German, Russian, Spanish, Korean)
- **Discipline** (Mathematics, Physics, Logic, History, Geography, Literature)
- **Scenario** (Function Calling, Code, Math Reasoning, Translation, Chat)

PARSE quantifies each layer's contribution to every (Language × Discipline × Scenario) combination, then preserves critical layers while transplanting redundant ones with ultra-efficient No-FFN attention blocks.

## 📊 Key Results

| Metric | Original Qwen-0.8B | PARSE (Ours) |
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
├── code/                   # Implementation framework
│   └── needle_universal/   # Core architecture + trainer
├── md/                     # 55 reference papers (Markdown)
├── pdf/                    # 54 reference papers (PDF)
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

This work is grounded in 55 foundational studies spanning:
- Structural pruning (LLM-Pruner, SparseGPT, Wanda, ShortGPT)
- Knowledge editing (ROME, MEMIT, MEND, SERAC)
- Machine unlearning (SISA, Descent-to-Delete)
- Data flywheels (AgenticQwen, ArenaLearning, SRDF)
- Agentic systems (Gorilla, xLAM, TinyAgent, ToolFlow)
- GRPO-based RL (DeepSeekMath, EBPO, STAPO, Mu-GRPO)
- Specialized architectures (Needle, MiniMind-O)

All references are available in `md/` and `pdf/` directories, numbered [01]-[55] by citation order.

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

MIT License. See [LICENSE](LICENSE) for details.
