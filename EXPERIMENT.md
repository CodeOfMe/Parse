# Qwen3.5-0.8B 能力修剪实验指南

> 基于 53 篇文献的综合实验框架，支持多维度能力修剪与对比分析。

---

## 📋 一、实验概述

### 1.1 实验目标
对 **Qwen3.5-0.8B** 模型进行选择性能力修剪，评估不同修剪策略对特定能力（语种、学科、场景）的影响，为小模型高效部署提供实证依据。

### 1.2 核心贡献
1. **能力感知分析**：量化每层对特定能力的重要性 (基于 Wanda 05, ShortGPT 07)
2. **混合修剪策略**：结合层剪枝与参数剪枝 (创新)
3. **结构化输出**：自动生成 JSON/CSV 结果，支持论文撰写与可视化

### 1.3 文献支撑 (53 篇)
| 类别 | 核心文献 | 应用 |
|:---|:---|:---|
| **剪枝方法** | Wanda (05), LLM-Pruner (03), LaCo (06), SparseGPT (04), LayerDrop (08) | 修剪策略实现 |
| **知识编辑** | ROME (38), MEMIT (39), MEND (40) | 能力定位与编辑 |
| **数据飞轮** | AgenticQwen (01), ArenaLearning (18), SRDF (19) | 训练数据生成 |
| **评估方法** | ShortGPT (07), TinyLlama (14), MInference (15) | 层贡献分析与评估 |

---

## 🛠️ 二、运行指南

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

#### 默认运行 (推荐)
```bash
# 自动检测设备 (CUDA > MPS > CPU)
python run_experiment.py --strategy hybrid --sparsity 0.5
```

#### 指定设备
```bash
# NVIDIA GPU (CUDA)
python run_experiment.py --device cuda --strategy hybrid --sparsity 0.5

# AMD GPU (ROCm) - 需安装 PyTorch ROCm 版
python run_experiment.py --device rocm --strategy hybrid --sparsity 0.5

# Apple Silicon (MPS)
python run_experiment.py --device mps --strategy hybrid --sparsity 0.5

# CPU (仅测试)
python run_experiment.py --device cpu --strategy layerdrop --sparsity 0.3
```

#### 跨设备评估 (Vulkan / 边缘设备)
由于 PyTorch 原生不支持 Vulkan，建议采用**本机剪枝 + 边缘推理**的工作流：

1. **在本机 (M4/CUDA) 完成剪枝**:
   ```bash
   python run_experiment.py --strategy hybrid --sparsity 0.5 --save_model
   # 模型将保存至 results/experiments/{timestamp}/model_hybrid/
   ```

2. **转换模型为 GGUF (用于 Vulkan)**:
   ```bash
   # 使用 llama.cpp 转换脚本
   python llama.cpp/convert-hf-to-gguf.py results/experiments/{timestamp}/model_hybrid/ --outfile model_pruned.gguf
   ```

3. **在 Vulkan 设备上运行推理测试**:
   ```bash
   # 编译 llama.cpp 支持 Vulkan
   cmake -DGGML_VULKAN=ON .. && make
   
   # 运行基准测试
   ./build/bin/llama-bench -m model_pruned.gguf -n 128
   ```

#### 自定义能力保留
```bash
# 仅保留中英文 + 数学逻辑
python run_experiment.py \
    --languages zh en \
    --domains stem logic \
    --scenarios math function_calling \
    --strategy hybrid \
    --sparsity 0.5
```

#### 批量实验
```bash
# 不同稀疏度对比
for sparsity in 0.3 0.5 0.7; do
    python run_experiment.py --strategy hybrid --sparsity $sparsity --device cuda
done

# 不同策略对比
for strategy in wanda layerdrop magnitude hybrid; do
    python run_experiment.py --strategy $strategy --sparsity 0.5 --device cuda
done
```

### 2.3 CUDA 设备适配

在 CUDA 设备上运行时，脚本会自动：
1. 使用 `torch.float16` 精度加速
2. 启用 `device_map="cuda"` 自动并行
3. 实验结束后调用 `torch.cuda.empty_cache()` 清理显存

**CUDA 优化建议**：
```bash
# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=0

# 启用 TF32 加速 (Ampere 架构+)
export NVIDIA_TF32_OVERRIDE=1

# 运行
python run_experiment.py --device cuda --strategy hybrid --sparsity 0.5
```

---

## 📊 三、输出说明

### 3.1 目录结构
每次运行会在 `results/experiments/{timestamp}/` 下生成：

```
results/experiments/20260525_143022/
├── experiment_results.json      # 完整实验结果
├── layer_importance.csv         # 层重要性矩阵
├── evaluation_results.csv       # 评估结果
├── model_statistics.csv         # 模型统计
└── experiment_metadata.csv      # 实验元数据
```

### 3.2 JSON 结果格式
```json
{
  "experiment_config": {
    "strategy": "hybrid",
    "target_sparsity": 0.5,
    "preserve_languages": ["zh", "en"],
    ...
  },
  "model_info": {
    "original_params": 752393024,
    "pruned_params": 376196512,
    "param_reduction_pct": 50.0,
    "n_layers_original": 24,
    "n_layers_pruned": 12
  },
  "evaluation": {
    "zh": {"perplexity": 12.34, "n_samples": 3},
    "en": {"perplexity": 15.67, "n_samples": 3},
    "math": {"perplexity": 18.90, "n_samples": 3},
    "logic": {"perplexity": 20.12, "n_samples": 2}
  },
  "layer_importance": {
    "zh": {"0": 0.05, "1": 0.08, ..., "23": 0.12},
    "en": {...},
    "math": {...},
    "logic": {...}
  }
}
```

### 3.3 CSV 文件说明

| 文件 | 列 | 用途 |
|:---|:---|:---|
| `layer_importance.csv` | Layer, zh, en, math, logic | 绘制热力图 (Figure 1) |
| `evaluation_results.csv` | Capability, Perplexity, N_Samples | 绘制对比柱状图 (Figure 2) |
| `model_statistics.csv` | Metric, Value | 论文模型对比表 (Table 1) |
| `experiment_metadata.csv` | Key, Value | 实验设置记录 (Section 4.1) |

---

## 📝 四、论文撰写指南

### 4.1 实验设置 (Section 4.1)
使用 `experiment_metadata.csv` 中的配置信息描述实验设置。

### 4.2 模型对比 (Table 1)
使用 `model_statistics.csv` 生成对比表格：
| 模型 | 参数量 | 层数 | 稀疏度 | 策略 |
|:---|:---:|:---:|:---:|:---|
| Original | 752M | 24 | 0% | - |
| Pruned (Hybrid) | 376M | 12 | 50% | Hybrid |

### 4.3 层重要性分析 (Figure 1)
使用 `layer_importance.csv` 绘制热力图：
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("layer_importance.csv")
df.set_index("Layer", inplace=True)
sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".2f")
plt.title("Layer Importance by Capability")
plt.savefig("layer_importance_heatmap.png", dpi=300)
```

### 4.4 评估结果 (Figure 2)
使用 `evaluation_results.csv` 绘制对比图：
```python
df = pd.read_csv("evaluation_results.csv")
sns.barplot(x="Capability", y="Perplexity", data=df)
plt.title("Perplexity by Capability after Pruning")
plt.savefig("evaluation_barplot.png", dpi=300)
```

---

## ⚠️ 五、注意事项

### 5.1 内存管理
- 实验结束后脚本会自动清理缓存
- 如需手动清理：
  ```python
  import torch
  torch.cuda.empty_cache()  # CUDA
  torch.mps.empty_cache()   # MPS
  ```

### 5.2 路径问题
- 模型路径支持相对路径和绝对路径
- 默认路径：`models/qwen/Qwen3___5-0___8B`
- 如模型在其他位置：
  ```bash
  python run_experiment.py --model_path /path/to/model
  ```

### 5.3 多 GPU 运行
```bash
# 使用多 GPU (需修改 capability_pruning.py 中的 device_map)
python run_experiment.py --device cuda --sparsity 0.5
```

### 5.4 断点续跑
- 每次实验生成独立时间戳目录
- 不会覆盖历史结果
- 可直接对比不同时间戳的结果

---

## 🔗 六、相关资源

- **代码**: `capability_pruning.py` (核心框架)
- **入口**: `run_experiment.py` (实验入口)
- **文献**: `md/` 目录 (53 篇文献)
- **模型**: `models/qwen/Qwen3___5-0___8B`

---

*本实验框架基于 53 篇文献设计，所有方法均有学术依据。输出结果可直接用于顶级期刊/会议论文撰写。*
