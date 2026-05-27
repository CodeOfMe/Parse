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

PARSE quantifies each layer's contribution to every (Language × Discipline × Scenario) combination using a **Capability Importance Tensor (CIT)**. Critical layers are preserved intact; redundant layers receive ultra-efficient No-FFN attention transplants [54]. A lightweight **Dynamic Capability Router (DCR)** (0.08M parameters) modulates internal residual gates based on input context, enabling a single compressed model to serve multiple preservation profiles without weight switching.

## 📊 Design Targets

| Metric | Original Qwen3.5-0.8B | PARSE Design Target |
|:---|:---:|:---:|
| Parameters | 752M | Dependent on preservation profile |
| Capability Retention (CRR) | 1.00 (baseline) | >0.90 on preserved dimensions (hypothesis) |
| Cross-Capability Interference (CCI) | — | Lower CCI indicates cleaner selective preservation |

*Note: All metrics are design targets derived from the methodological framework. Empirical results pending experimental execution of the implemented pipeline. The 8.8× compression figure is a target that requires additional measures (attention head pruning, aggressive layer removal) beyond FFN-only transplantation to achieve.*

## 📁 Repository Structure

```
PARSE/
├── article.md              # English paper (full)
├── article_cn.md           # Chinese paper (full)
├── literature_review.md    # Literature review (English) — 55 references
├── 文献综述.md              # Literature review (Chinese) — 55 references
├── code/                   # Implementation framework
│   ├── __init__.py
│   └── parse/              # Core PARSE package
│       ├── __init__.py      # Public API
│       ├── config.py        # PARSEConfig + 12 preservation profiles (P1-P12)
│       ├── export.py        # GGUF export (llama.cpp + MoXing)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── cit.py       # Capability Importance Tensor computation
│       │   ├── model.py     # HuggingFace model loader + builder
│       │   └── transplant.py # FFN removal + NoFFN insertion + DCR
│       ├── data/
│       │   ├── __init__.py
│       │   └── calibration.py # Calibration prompts per axis (8+6+5 categories)
│       ├── trainer/
│       │   ├── __init__.py
│       │   └── flywheel.py  # Dual-flywheel recovery with GRPO
│       └── eval/
│           ├── __init__.py
│           └── metrics.py   # PPL, CRR, CCI, GSM8K, BFCL
├── md/                     # Reference papers (Markdown)
├── pdf/                    # Reference papers (PDF)
├── figures/                # Publication-quality SVG/PDF figures
├── results/                # RTX 4060 experiment results
├── models/                 # Model configurations
├── run_experiment.py       # Experiment runner
└── EXPERIMENT.md           # Experiment guide
```

## 🔧 Setup

### 1. Install dependencies

```bash
pip install torch transformers modelscope numpy tqdm
```

### 2. Download base model (Qwen3.5-0.8B)

```bash
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir models/qwen/Qwen3___5-0___8B
```

### 3. (Optional) Install llama.cpp for GGUF export

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build && cmake .. && make -j
```

### 4. (Optional) Install MoXing for GGUF serving

```bash
git clone https://github.com/cycleuser/MoXing.git /path/to/MoXing
```

## 🚀 Quick Start

```bash
# Full PARSE pipeline (CUDA / ROCm)
python run_experiment.py --device cuda --strategy parse --sparsity 0.5

# Use a pre-defined preservation profile
python run_experiment.py --profile P1 --device auto

# Apple Silicon (MPS)
python run_experiment.py --device mps --strategy parse --sparsity 0.5

# Custom preservation profile + GGUF export
python run_experiment.py \
    --languages zh en \
    --disciplines math logic \
    --scenarios fc math_reasoning \
    --device cuda \
    --export-gguf
```

## 📚 References

This work draws on foundational studies in:
- Structural pruning [3-7,11-16]
- Dynamic inference [8-10]
- Knowledge editing and machine unlearning [37-47]
- Data flywheels and self-improving training [1,17-23]
- Agentic systems and tool calling [24-30]
- GRPO-based reinforcement learning [31-36,55]
- Specialized architectures [2,54]

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
