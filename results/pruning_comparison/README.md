# Qwen3.5-0.8B 有序精剪报告

## 模型架构分析

Qwen3.5-0.8B 是混合架构Transformer：
- **总层数**: 24层
- **Linear Attention层**: 18层 (L0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22)
- **Standard Attention层**: 6层 (L3,7,11,15,19,23)
- **隐藏层维度**: 1024
- **注意力头数**: 8
- **FFN中间维度**: 3584
- **总参数**: 752M

## 精剪策略

**目标**: 保留数理逻辑和中文英文基本语法能力，删除其他能力

**方法**: 有序删除Linear Attention层，保留Standard Attention层

### 版本对比

| 模型 | 保留层数 | 参数 | 比例 | 大小 |
|------|---------|------|------|------|
| original | 24层 | 752M | 100% | 1435MB |
| v1_remove_6linear | 18层 | 623M | 82.8% | 1188MB |
| v2_remove_12linear | 12层 | 494M | 65.6% | 942MB |
| v3_remove_all_linear | 6层 | 364M | 48.4% | 695MB |

### 各版本保留的层

**v1_remove_6linear** (18层):
- 删除: L0,1,2,4,5,6 (前6层Linear Attn)
- 保留: L3,7,11,15,19,23 (全部Standard Attn) + L8,9,10,12,13,14,16,17,18,20,21,22 (12层Linear Attn)

**v2_remove_12linear** (12层):
- 删除: L0,1,2,4,5,6,8,9,10,12,13,14 (前12层Linear Attn)
- 保留: L3,7,11,15,19,23 (全部Standard Attn) + L16,17,18,20,21,22 (6层Linear Attn)

**v3_remove_all_linear** (6层):
- 删除: 所有18层Linear Attn
- 保留: L3,7,11,15,19,23 (仅Standard Attn)

## 预期效果

### 速度提升
- **v1**: 约1.3x加速 (18/24层)
- **v2**: 约2x加速 (12/24层)
- **v3**: 约4x加速 (6/24层)

### 能力保持
- **数理逻辑**: Standard Attn层主要负责推理能力，全部保留
- **中英文语法**: 浅层语言处理能力在Standard Attn中也有体现
- **其他能力**: 历史/地理/文学等知识主要存储在Linear Attn层，会被削弱

## 模型位置

```
models/pruned_v2/
├── v1_remove_6linear/    # 18层, 623M参数
├── v2_remove_12linear/   # 12层, 494M参数
└── v3_remove_all_linear/ # 6层, 364M参数
```

## 后续步骤

1. **测试各版本速度**: 在实际硬件上测量推理速度
2. **测试能力保持**: 评估数理逻辑/中英文语法能力
3. **微调恢复**: 如有必要，对剪枝模型进行针对性微调
4. **GGUF转换**: 转换为GGUF格式用于高效推理

## 脚本位置

- `code/correct_pruning.py`: 正确剪枝脚本
- `code/quick_pruning.py`: 快速剪枝脚本（有结构问题）
- `code/ordered_pruning.py`: 原始剪枝脚本（有结构问题）
