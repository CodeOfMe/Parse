# PARSE 实验指南

> 基于三维能力重要性张量 (CIT) 的精细粒度能力保持压缩实验框架。

---

## 一、实验目标

对 **Qwen3.5-0.8B** (752M, 24层) 进行能力感知架构移植。沿 **8 语种 × 6 学科 × 5 场景** 三维能力空间，构建每层的能力重要性张量 (CIT)，保留目标能力关键层的原始权重，将冗余层 FFN 替换为纯注意力模块 (No-FFN) [55]，通过动态能力路由器 (DCR) 实现单模型多剖面服务。

### 能力维度定义

| 能力轴 | 类别数 | 类别 |
|:---|:---:|:---|
| 语种 | 8 | 中、英、日、法、德、俄、西、韩 |
| 学科 | 6 | 数学、物理、逻辑、历史、地理、文学 |
| 场景 | 5 | 函数调用、代码生成、数学推理、翻译、通用对话 |

### 12 组保留剖面 (对标论文 Table 1)

| 剖面 | 语种 | 学科 | 场景 |
|:---|:---|:---|:---|
| P1 | zh, en | math, logic | fc, math_reasoning |
| P2 | zh, en, ja | math, physics | all |
| P3 | en | math | all |
| P4 | zh | all | all |
| P5 | all | math, logic, physics | fc |
| P6 | zh, en | all | fc, code |
| P7 | all | math | math_reasoning |
| P8 | zh, en, ja, fr | all | translation |
| P9 | all | all | fc |
| P10 | zh, en | all | all |
| P11 | all | math, logic | all |
| P12 | zh, en | math, logic, physics | fc, code, math_reasoning |

---

## 二、运行指南

### 2.1 环境准备

```bash
# 安装依赖
pip install torch transformers numpy tqdm

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### 2.2 运行命令

#### 默认运行
```bash
python run_experiment.py --strategy parse --sparsity 0.5
```

#### 指定设备
```bash
# NVIDIA GPU
python run_experiment.py --device cuda --strategy parse --sparsity 0.5

# Apple Silicon (MPS)
python run_experiment.py --device mps --strategy parse --sparsity 0.5

# CPU (测试)
python run_experiment.py --device cpu --strategy layerdrop --sparsity 0.3
```

#### 自定义能力保留剖面
```bash
# P1: 中英 STEM + Agent
python run_experiment.py \
    --languages zh en \
    --disciplines math logic \
    --scenarios fc math_reasoning \
    --strategy parse \
    --sparsity 0.5

# P6: 双语开发者 Agent
python run_experiment.py \
    --languages zh en \
    --disciplines all \
    --scenarios fc code \
    --strategy parse \
    --sparsity 0.5
```

#### 跨设备部署 (Vulkan / 边缘设备)
1. 本机完成 PGCS 移植：
   ```bash
   python run_experiment.py --strategy parse --sparsity 0.5 --save_model
   ```
2. 转换为 GGUF：
   ```bash
   python llama.cpp/convert-hf-to-gguf.py results/experiments/{timestamp}/model_parse/ --outfile model_parse.gguf
   ```
3. Vulkan 推理：
   ```bash
   cmake -DGGML_VULKAN=ON .. && make
   ./build/bin/llama-bench -m model_parse.gguf -n 128
   ```

#### 批量实验
```bash
# 不同稀疏度
for sparsity in 0.3 0.5 0.7; do
    python run_experiment.py --strategy parse --sparsity $sparsity --device cuda
done

# 不同策略对比
for strategy in wanda sparsegpt layerdrop parse; do
    python run_experiment.py --strategy $strategy --sparsity 0.5 --device cuda
done
```

### 2.3 CUDA 优化

```bash
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=1
python run_experiment.py --device cuda --strategy parse --sparsity 0.5
```

---

## 三、输出说明

### 3.1 目录结构
```
results/experiments/{timestamp}/
├── experiment_results.json      # 完整实验结果 (CIT, CRR, PRR, Speedup)
├── cit_tensor.csv               # 三维能力重要性张量
├── capability_retention.csv     # 各能力维度的 CRR
├── model_statistics.csv         # 参数/层数/稀疏度统计
└── experiment_metadata.csv      # 实验配置记录
```

### 3.2 JSON 结果格式
```json
{
  "experiment_config": {
    "strategy": "parse",
    "target_sparsity": 0.5,
    "preservation_profile": {
      "languages": ["zh", "en"],
      "disciplines": ["math", "logic"],
      "scenarios": ["fc", "math_reasoning"]
    }
  },
  "model_info": {
    "original_params": 752393024,
    "compressed_params": 85000000,
    "param_reduction_ratio": 0.887,
    "n_layers_original": 24,
    "n_layers_retained": 14
  },
  "capability_retention": {
    "zh": {"perplexity": 12.34, "CRR": 0.968},
    "math_reasoning": {"gsm8k_accuracy": 42.8, "CRR": 0.947},
    "fc": {"bfcl_accuracy": 88.7, "CRR": 1.007}
  },
  "cit": {
    "zh": {"0": 0.05, "1": 0.08, "...": "...", "23": 0.12},
    "math": {"0": 0.02, "...": "...", "23": 0.18},
    "fc": {"0": 0.01, "...": "...", "23": 0.15}
  },
  "inference_speed": {"original_tok_s": 1.5, "compressed_tok_s": 15.4, "speedup": 10.3}
}
```

### 3.3 输出文件对照

| 文件 | 论文对应 | 用途 |
|:---|:---|:---|
| `cit_tensor.csv` | Figure 1 (CIT热力图) | 层重要性三维分布 |
| `capability_retention.csv` | Table 1 (12剖面结果) | CRR/CCI 对比矩阵 |
| `model_statistics.csv` | Table 2 (基线对比) | PRR/层数/Speedup |
| `experiment_metadata.csv` | Section 4.1 (实验设置) | 硬件/配置记录 |

---

## 四、论文图表生成

### Figure 1: CIT 热力图
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cit_tensor.csv", index_col=0)
plt.figure(figsize=(12, 6))
sns.heatmap(df, cmap="viridis", cbar_kws={"label": "CIT Score"})
plt.xlabel("Capability (Language/Discipline/Scenario)")
plt.ylabel("Layer Index")
plt.tight_layout()
plt.savefig("figure1_cit_heatmap.svg")
```

### Table 1: 12 剖面全矩阵结果
```python
df = pd.read_csv("capability_retention.csv")
# 输出可直贴论文的 LaTeX 表格
print(df.to_latex(float_format="%.1f"))
```

### Figure 3: 能力保留雷达图
```python
import numpy as np
df = pd.read_csv("capability_retention.csv")
# 每个剖面一条曲线，展示保留vs非保留维度的 CRR 差异
```

---

## 五、注意事项

- 每次实验生成独立时间戳目录，不覆盖历史结果
- 实验结束后自动调用 `torch.cuda.empty_cache()` / `torch.mps.empty_cache()`
- CIT 计算使用 10-20 条探针样本/能力，因子化后总计约 285 条
- DCR 训练在移植后康复阶段完成，3 轮迭代收敛

---

## 相关资源

- **论文 (英文)**: `article.md`
- **论文 (中文)**: `article_cn.md`
- **代码**: `code/needle_universal/`
- **入口**: `run_experiment.py`
- **文献**: `md/` 目录
- **模型**: `models/qwen/Qwen3___5-0___8B`
