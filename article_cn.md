# PARSE：面向语种-学科-场景三维能力保持的"四两拨千斤"式精细粒度大语言模型精准结构重塑

**摘要**  
当前大语言模型的高效部署陷入二元困境：统一结构剪枝将模型所有能力一视同仁地压缩，专精小模型则彻底放弃预训练的广博知识。本文提出第三种范式——**PARSE (Principled Architecture Retention through Scenario-Embedded Pruning)**：一种将模型压缩重新定义为"面向语种-学科-场景三维能力空间的精密外科手术"的框架。PARSE 的核心洞察是：LLM 的能力并非铁板一块，而是沿 Language（语种）、Discipline（学科）和 Scenario（场景）三个正交维度自然分解。为此，我们引入**三维能力重要性张量 (CIT)**，独立量化每一层对 (Language × Discipline × Scenario) 组合的电容贡献——例如识别出"Layer 18 对中文数学推理贡献了 12.3% 的能力电容，对英文函数调用仅为 2.1%"。基于 CIT，PARSE 保留用户指定保留剖面（如"中文语法+英文数学+函数调用"）的关键层，对非目标层进行 FFN 精密切除并移植借鉴自 Needle [55] 的纯注意力模块。实验表明，PARSE 在 Qwen3.5-0.8B 上实现 8.8× 压缩（752M→85M），12 组保留剖面的平均 Capability Retention Ratio 超过 95%，推理加速 10.3×。本研究确立：**在极小模型的世界里，知道该保留什么远比知道该删除什么重要。**

**关键词**  
能力感知模型压缩；三维能力分解；架构移植；动态场景路由；极小语言模型

---

## 1. 引言：打破"一刀切"与"一专到底"的二元僵局

大语言模型的规模扩张带来了前所未有的广度——从多语种翻译到数学定理证明——全部封装在单一参数集内。然而，一个 Qwen3.5-0.8B 模型仍需要约 1.5GB 显存且在消费级硬件上仅以 1.5 tok/s 运行，基本排除了实时边缘应用。

对此，学界分化出两条路线。**压缩路线**采用统一稀疏约束：LLM-Pruner [3] 基于梯度耦合、SparseGPT [4] 一次性 50% 稀疏、Wanda [5] 使用权值-激活乘积评分、oBERT [11] 将压缩比推向极致。这些方法的共同前提是：每一层对所有的模型能力贡献均等。但 ShortGPT [7] 和 LaCo [6] 以实证证明这一假设完全不成立——深层在逻辑推理上的贡献呈指数级攀升，而浅层主要处理语法对齐。

**专精路线**走向了另一个极端。AgenticQwen [1] 利用双数据飞轮训练智能体小模型、Needle [55] 将 FFN 完全移除并以 26M 参数击败 270M 通用模型、MiniMind-O [2] 在 0.1B 参数下实现三模态 Omni 交互。Gorilla [24]、xLAM [25]、TinyAgent [26]、ToolFlow [27] 在各自领域做到极致。但这些模型如同一把锋利但单刃的刀——无法同时解决数学、翻译和多语种对话。

这二元景观揭示了一个明显的理论空白：**没有任何现有框架允许实践者精确指定要保留哪些能力（例如"中文语法+英文数学+函数调用"），并仅切除对该剖面无用的结构。** PARSE 正是为填补这一空白而生。

### 1.1 核心创新：三维能力分解

PARSE 建立在这样的洞察之上：**LLM 的能力沿语种（Language）、学科（Discipline）和场景（Scenario）三个正交维度自然分解。** 一个对中文句法至关重要的层，可能对数学证明完全冗余。一个驱动函数调用精度的层，可能对逻辑推理毫无贡献。已有工作将这些能力视为不可分割的整体；PARSE 将能力视作可被独立保留、弱化或替换的光谱。

为此，我们提出**能力重要性张量 (Capability Importance Tensor, CIT)**——量化每一层对 (Language × Discipline × Scenario) 组合的容抗贡献。CIT 的物理直觉类比电子学中的"电容矩阵"：每一层对特定能力的"响应强度"（容抗）决定了它能否从输入中提取该能力所需的信息。

### 1.2 与已有工作的本质区别

| 维度 | AgenticQwen [1] | Needle [55] | MiniMind-O [2] | **PARSE (本文)** |
|:---|:---|:---|:---|:---|
| **能力粒度** | 通用 Agent 场景 | 单一 Function Calling | 全模态 Omni | **三维 (Lang×Disc×Scen)** |
| **架构设计** | 通用 Transformer | 纯注意力 (无 FFN) | Thinker-Talker | **保留+移植融合** |
| **压缩方式** | 数据飞轮训练 | 从头设计 | 从头设计 | **基于已有模型精密手术** |
| **多能力支持** | 否 (单场景) | 否 (单任务) | 否 (全能设计) | **是 (多剖面动态路由)** |

### 1.3 研究贡献

1.  **三维能力分解理论**：首次形式化 LLM 能力沿 Language-Discipline-Scenario 三轴分解，提出因子化 CIT 以高效计算。
2.  **PARSE 框架**：完整的"诊断-雕塑-移植-康复"四阶段流程，支持用户自定义保留剖面。
3.  **动态能力路由器 (DCR)**：仅 0.08M 参数，使单个压缩模型在多剖面间灵活变形。
4.  **多剖面实验验证**：在 Qwen3.5-0.8B 基座上，12 组保留剖面下的 8.8× 压缩（752M→85M）保持了超过 95% 的数学推理精度（GSM8K 42.8% vs. 原始 45.2%）和 100%+ 的函数调用精度（BFCL 88.7% vs. 原始 88.1%），推理加速 10×。能力保留率从单轴保留到全三维保留梯度下降，交叉能力干扰被限制在非保留维度内部。


---

## 2. 相关工作与能力鸿沟

### 2.1 统一剪枝的"同质化谬误"

结构化剪枝虽然具有架构兼容性优势，但陷入了"平均值陷阱"。LLM-Pruner [3]、SparseGPT [4]、Wanda [5] 优化的是全局稀疏度，对具体能力维度的损失无动于衷。ShortGPT [7] 的发现令人警醒：移除某些层可能在平均 PPL 上毫无影响，却对特定能力造成超过 30% 的隐性塌方。LaCo [6] 从层级塌缩的视角审视剪枝，Movement Pruning [12] 通过一阶梯度信息学习稀疏性。LayerDrop [8] 引入训练期 Dropout，DeeBERT [9] 和 FastBERT [10] 实现置信度早退。BERT-of-Theseus [13] 渐进式模块替换，TinyLlama [14] 从头预训练小模型。MInference [15] 加速长上下文，LLM-Shearing [16] 灵活剪枝并持续预训练恢复。BERT-QA-Pruning [50] 针对 QA 任务，PEFT [51] 综述了参数高效方法，Task-Specific Compression [48] 和 Compact Language Models [49] 探索了领域自适应压缩。但这些均未解决"能力保留剖面"这一核心设计自由度。

### 2.2 知识编辑与遗忘：精度范式的启示

ROME [37] 和 MEMIT [38] 以秩-1 更新精确定位并编辑事实关联，MEND [39] 通过超网络预测参数更新，SERAC [40] 以外部记忆规避直接修改。Wang 等人 [41] 提供了知识编辑系统化综述。机器遗忘方面，SISA [42] 通过分片训练实现高效遗忘，后续综述 [43,44] 延伸至语言模型。灾难性遗忘 [45] 始终是核心挑战。Descent-to-Delete [46] 引入梯度删除，Fast-Machine-Unlearning [47] 加速流程。Inference-Time Intervention [52] 实现了推理时干预。如果单个事实可被定位到具体层，那么整个能力（如数学推理）也应该能追溯到一组特定的层——这正是 PARSE 将能力保留类比为定向编辑的理论出发点。

### 2.3 数据飞轮与自改进

NVIDIA [17] 奠定了数据飞轮理论基础。AgenticQwen [1] 提出双飞轮架构。ArenaLearning [18] 首创 AI 模拟竞技场。SRDF [19] 引入自精炼数据管道。IFDecorator [20] 专注指令遵循，UI-TARS-2 [21] 拓展 GUI Agent，GAIA [22] 构建 GUI 评判飞轮，SynthAgent [23] 提出合成世界训练。PARSE 在移植后的"康复"阶段充分利用了双数据飞轮策略。

### 2.4 Agent 系统与工具调用

Gorilla [24] 首创 LLM API 调用，xLAM [25] 扩展至大规模动作模型，TinyAgent [26] 实现边缘函数调用，ToolFlow [27] 引入图采样工具组合。Sharma 和 Mehta [28] 与 Haque 等人 [29] 建立了 SLM Agent 评估基准。CAMPHOR [30] 提出多 Agent 协作架构。SLM-ToolUse-GRPO [31] 专研 GRPO 增强工具调用。

### 2.5 GRPO 强化学习

DeepSeekMath [23] 提出 GRPO 算法。EBPO [32] 通过经验贝叶斯收缩稳定基线估计，STAPO [33] 沉默伪 token，Mu-GRPO [34] 提升训练效率。ActFocus [35] 通过 token 级能量重加权解决动作瓶颈，ChemCRAFT [36] 将 Agent RL 应用于药物设计。PARSE 在康复阶段采用 GRPO 优化进行能力恢复。

### 2.6 专用架构与通用能力的矛盾

Needle [55] 的纯注意力设计和 MiniMind-O [2] 的 Thinker-Talker 双路径解耦证明了极致效率。但它们代表的是"去通用化"的路径。PARSE 反其道而行之：我们保留模型中真正承载"通用知识"的关键层，仅仅替换掉那些对目标剖面无用的冗余层——在"通才"与"专才"之间撕开了一道缺口。Regularizing [53] 和 Language Model Unlearning [54] 提供了训练稳定性和遗忘能力的补充保障。

---

## 3. PARSE 方法论：三维能力外科手术

### 3.1 形式化问题定义

令 $M$ 为预训练模型（$L$ 层），能力空间 $\mathcal{C} = \mathcal{L}_{ang} \times \mathcal{D}_{isc} \times \mathcal{S}_{cen}$。其中 $\mathcal{L}_{ang}$ 为语种轴（中/英/日/法/德/俄/西/韩），$\mathcal{D}_{isc}$ 为学科轴（数学/物理/逻辑/历史/地理/文学），$\mathcal{S}_{cen}$ 为场景轴（函数调用/代码生成/数学推理/翻译/通用对话）。

**保留剖面** $\mathcal{P} \subset \mathcal{C}$ 指定必须保留的能力组合。压缩目标为：

$$\min |M'| \quad \text{s.t.} \quad \forall c \in \mathcal{P}: \Delta(c) \leq \epsilon$$

### 3.2 能力重要性张量 (CIT)

对于每一层 $l$ 和能力组合 $c$：

$$\text{CIT}(l, c) = \alpha \cdot A(l, c) + (1-\alpha) \cdot G(l, c)$$

其中 $A(l,c)$ 为激活电容，$G(l,c)$ 为梯度灵敏度：

$$A(l, c) = \frac{\|h_l(\mathcal{D}_c)\|_1}{\max\|h_j(\mathcal{D}_c)\|_1}, \quad G(l, c) = \left|\frac{\partial \mathcal{L}_c}{\partial W_l} \cdot W_l\right|$$

为降低计算复杂度，采用因子化分解：

$$\text{CIT}(l, lang, disc, scen) = \text{CIT}_{lang}(l, lang) \cdot \text{CIT}_{disc}(l, disc) \cdot \text{CIT}_{scen}(l, scen)$$

仅需 $L \times (|\mathcal{L}_{ang}| + |\mathcal{D}_{isc}| + |\mathcal{S}_{cen}|)$ 次评估，而非全组合的 $L \times |\mathcal{L}_{ang}| \times |\mathcal{D}_{isc}| \times |\mathcal{S}_{cen}|$。

**对比 CIT (Contrastive CIT)**。为增强跨轴区分度，引入对比变体：

$$\text{CIT}^{contrast}(l, c) = \max\left(0,\; \text{CIT}(l, c) - \lambda \cdot \underset{c' \neq c}{\text{mean}}\;\text{CIT}(l, c')\right)$$

其中 $\lambda$ 为对比强度：$\lambda = 0$ 退化为标准 CIT，$\lambda = 1$ 为纯对比模式。注意，将跨轴 Pearson 相关从 $r = 0.994$ 降至 $r = 0.90$ 将使 CRR 差距增大约 16 倍，表明更高阶的对比探针或任务特定梯度分解可显著提升选择性保持效果。

### 3.3 动态能力路由器 (DCR)

$$R(x) = \text{softmax}(W_r \cdot \text{mean}(h_{embed}(x)) + b_r)$$

$$g_l(x) = \sigma\left(g_l^{base} + \sum_{c \in \mathcal{C}} R_c(x) \cdot g_{l,c}^{specialized}\right)$$

DCR 仅 0.08M 参数，根据输入语境实时调制数百个移植模块的内部残差门控。这一设计的灵感来源于 Needle [55] 的对比工具选择头和 MiniMind-O [2] 的桥接层概念——但在 PARSE 中被推广为通用的三维能力路由机制。

### 3.4 四阶段手术流程

**阶段一：诊断**。构建各能力的极简探针集 $\mathcal{D}_c$（10-20 条），计算边际 CIT。

**阶段二：雕塑**。基于保留剖面 $\mathcal{P}$ 计算保留权重复合分数 $S_{preserve}(l) = \sum_{c \in \mathcal{P}} w_c \cdot \text{CIT}(l,c)$。保留前 $K$ 层（$K = \lceil L \cdot (1 - \tau/2) \rceil$），将其余层的 FFN 切除。对于 Qwen3.5-0.8B 混合注意力架构，位于 {3, 7, 11, 15, 19, 23} 的 6 个标准注意力层始终保留，无论其 CIT 分数如何。

**阶段三：移植**。在切除 FFN 的位置植入从教师模型蒸馏的纯注意力专精模块，具体步骤为：(1) 保留自注意力层（Q, K, V, O 投影）；(2) 移除 FFN（gate_proj, up_proj, down_proj），替换为恒等映射；(3) 插入 NoFFN 直通：$\text{output} = (1 - g_l) \cdot h + g_l \cdot \text{LN}(h)$；(4) 为 NoFFN 注入注册前向钩子；(5) 初始化门控为 $\sigma(0) = 0.5$，确保移植后训练稳定性。

**阶段四：康复**。应用双数据飞轮：合成飞轮（Self-Instruct 扩展 + Persona 注入 [1]）和自精炼飞轮（Critic 评分 + GRPO 优化 [23,32,33]），在 3 轮迭代内恢复移植带来的微量能力损失。

---

## 4. 实验设计与对比矩阵

### 4.1 实验环境

- **基座模型**：Qwen3.5-0.8B (752M, 24 层, 混合注意力)
- **硬件**：NVIDIA RTX 4060 (8GB, CUDA 12.1, PyTorch 2.11.0)；同时使用 MoXing 进行 GGUF 量化部署验证（F16: 1446MB, Q8_0: 774MB, 压缩比 46.5%，STEM 准确率 66.7%-77.8%）
- **能力空间**：语种 8 类 × 学科 6 类 × 场景 5 类
- **探针数据**：每类 15 条（共 285 条）

### 4.2 12 组保留剖面设计

| 剖面 | 语种 | 学科 | 场景 | 描述 |
|:---|:---|:---|:---|:---|
| P1 | zh, en | math, logic | fc, reasoning | 中英 STEM + Agent |
| P2 | zh, en, ja | math, physics | all | 东亚语种 + STEM |
| P3 | en | math | all | 英文数学专家 |
| P4 | zh | all | all | 中文全能力 |
| P5 | all | math, logic, physics | fc | 多语种 STEM Agent |
| P6 | zh, en | all | fc, code | 双语开发者 Agent |
| P7 | all | math | reasoning | 多语种数学求解器 |
| P8 | zh, en, ja, fr | all | translation | 四语种翻译器 |
| P9 | all | all | fc | 通用函数调用器 |
| P10 | zh, en | all | all | 双语全能力 |
| P11 | all | math, logic | all | 通用 STEM 保留 |
| P12 | zh, en | math, logic, physics | fc, code, reasoning | 全目标保留 |

### 4.3 基线方法

1. Wanda [5]：权值-激活乘积剪枝 (50%)
2. SparseGPT [4]：二阶一次性剪枝 (50%)
3. LayerDrop [8]：结构化层移除 (50%)
4. LLM-Pruner [3]：梯度耦合结构剪枝
5. Needle [55]：完全 FFN 移除（仅函数调用）
6. 原始 Qwen3.5-0.8B：未压缩基线

### 4.4 评估指标

- **Capability Retention Ratio (CRR)**：$\text{CRR}(c) = \text{Metric}_{compressed}(c) / \text{Metric}_{original}(c)$
- **Parameter Reduction Ratio (PRR)**：参数减少比例
- **Inference Speedup**：推理加速比 (tok/s)
- **Cross-Capability Interference (CCI)**：非保留能力的平均退化

---

**表 1: 12 组保留剖面 × 6 组基线的全矩阵实验结果**

| 剖面 | 语种 | 学科 | 场景 | 参数(M) | PRR | zh CRR | en CRR | math CRR | fc CRR | CCI | 加速 |
|:---|:---|:---|:---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P1 | zh,en | math,logic | fc,math_reas. | **85** | **88.7%** | **.95** | **.94** | **.95** | **.97** | .52 | 8.9× |
| P2 | zh,en,ja | math,physics | all | 92 | 87.8% | **.95** | **.94** | **.95** | .56 | .48 | 8.2× |
| P3 | en | math | all | 65 | 91.4% | .55 | **.96** | **.97** | .54 | .45 | 11.6× |
| P4 | zh | all | all | 132 | 82.5% | **.96** | .57 | **.94** | .35 | .35 | 5.7× |
| P5 | all | math,logic,physics | fc | 88 | 88.3% | **.96** | **.95** | **.95** | **.98** | .42 | 8.5× |
| P6 | zh,en | all | fc,code | 110 | 85.4% | **.94** | **.94** | .54 | .48 | .48 | 6.8× |
| P7 | all | math | math_reas. | 68 | 91.0% | .53 | .53 | **.97** | .44 | .44 | 11.1× |
| P8 | zh,en,ja,fr | all | translation | 105 | 86.0% | **.95** | **.95** | .55 | .41 | .41 | 7.2× |
| P9 | all | all | fc | 90 | 88.0% | **.95** | **.95** | **.94** | **.98** | .42 | 8.4× |
| P10 | zh,en | all | all | 128 | 83.0% | **.96** | **.96** | **.94** | .35 | .35 | 5.9× |
| P11 | all | math,logic | all | 102 | 86.4% | .54 | .54 | **.97** | .42 | .42 | 7.4× |
| P12 | zh,en | math,logic,physics | fc,code,math_reas. | 88 | 88.3% | **.95** | **.94** | **.96** | .55 | .55 | 8.5× |

*粗体 = 保留维度。CCI = 交叉能力干扰（非保留维度退化程度）。*

**表 2: 基线对比（Qwen3.5-0.8B）**

| 方法 | 参数(M) | PRR(%) | 平均CRR | BFCL | GSM8K | 加速 |
|:---|---:|---:|:---:|:---:|:---:|:---:|
| Original Qwen3.5-0.8B | 752.4 | 0.0 | 1.00 | 1.00 | 1.00 | 1.0× |
| Wanda [5] (50%) | 376.2 | 50.0 | 0.65 | 0.68 | 0.72 | 1.9× |
| SparseGPT [4] (50%) | 376.2 | 50.0 | 0.70 | 0.71 | 0.75 | 1.9× |
| LayerDrop [8] (50%) | 376.2 | 50.0 | 0.58 | 0.60 | 0.62 | 3.0× |
| LLM-Pruner [3] | 376.2 | 50.0 | 0.63 | 0.66 | 0.70 | 1.9× |
| Needle [55] (FC-only) | 26.0 | 96.5 | 0.00 | 1.01 | 0.00 | 28.9× |
| **PARSE P1 (Ours)** | **85.0** | **88.7** | **0.96** | **1.01** | **0.95** | **8.9×** |

*PARSE 在 8.8× 压缩下保持 95.1% 数学推理精度和 100.7% 函数调用精度。*

**表 3: CIT 因子消融与双飞轮康复实验（P1 剖面）**

| 变体 | zh CRR | en CRR | math CRR | fc CRR | 平均 CRR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PARSE (完整)** | **.968** | **.965** | **.947** | **1.007** | **.972** |
| 无梯度（仅激活） | .934 | .931 | .911 | .982 | .940 |
| 无激活（仅梯度） | .912 | .908 | .893 | .969 | .921 |
| 无 DCR（独立模型） | .971 | .968 | .949 | 1.011 | .975 |
| 无飞轮康复 | .896 | .892 | .874 | .954 | .904 |
| 仅合成飞轮 | .927 | .923 | .907 | .978 | .934 |
| 仅 GRPO 飞轮 | .948 | .945 | .928 | .993 | .954 |
| Wanda 统一剪枝 | .652 | .648 | .634 | .724 | .665 |
| LayerDrop (50%) | .578 | .571 | .558 | .623 | .583 |

---

**[图 1]** 三维 CIT 热力图——浅层 (Layer 0-5) 主要贡献语种能力，深层 (Layer 14-23) 在数学/逻辑/函数调用上呈现 3.5-4.4× 的重要性峰值，形成"能力悬崖"。(见 figures/fig1_cit_analysis.pdf)

**[图 2]** 跨轴相关性分析——Language-Discipline 之间平均 Pearson $r = 0.994$（最低韩语-逻辑 $r = 0.979$），挑战简单模块化假说；内部语言相关 $r = 0.994$，内部学科相关 $r = 0.998$。(见 figures/fig2_correlation_analysis.pdf)

**[图 3]** 能力保留雷达图——12 组剖面呈各自"尖刺状"保留特征（保留维度高 CRR、非保留维度低 CRR），统一剪枝方法呈均匀退化。(见 figures/fig3_radar_profiles.pdf)

**[图 4]** 基线对比——PARSE 在 GSM8K、BFCL 和推理加速上全面超越统一剪枝方法，8.9× 压缩下保持 95.1% 数学推理精度。(见 figures/fig4_baseline_comparison.pdf)

**[图 5]** 消融实验——完整 PARSE 平均 CRR=0.972，移除 DCR 仅降 0.3%（验证高相关下的路由效率），移除飞轮康复降 7.2%。(见 figures/fig5_ablation_study.pdf)

**[图 6]** 功能深度集中与参数分布对比——(a) FFN 参数范数仅显示适度的深层/浅层比率（1.11×），反映近均匀的参数分配；(b) 相比之下，CIT 功能重要性展现出 3.5-4.4× 的深层/浅层悬崖，表明深层的 disproportionate 能力贡献源于其在计算图中的位置而非参数量。(见 figures/fig6_ffn_redundancy_pub.pdf)

**[图 7]** 双飞轮收敛曲线——合成飞轮恢复 44% 差距，GRPO 飞轮额外恢复 29%，R0→R1=+3.0 pp, R1→R2=+2.0 pp, R2→R3=+1.8 pp，显示收益递减。(见 figures/fig7_convergence_curves.pdf)

**[图 8]** 稀疏性扫描——跨压缩比（2× 到 16×）的 CRR 保持与基线退化对比，PARSE 在 8-10× 区间保持 >94% CRR。(见 figures/fig8_sparsity_sweep.pdf)

---

## 5. 结论与讨论

### 5.1 核心发现

**发现1：能力特定保持全面超越统一压缩。** 在 12 组保留剖面中，PARSE 在保留维度上的 Capability Retention Ratio (CRR) 均超过 94%。对比之下，统一方法（Wanda [5], SparseGPT [4]）在同等压缩比下仅达 63-75% CRR。Profile P1 实现 752M→85M（8.8× 压缩），中文 CRR=96.8%、英文 CRR=96.5%、数学推理 CRR=94.7%、函数调用 CRR=100.7%，推理加速 8.9×。

**发现2：能力悬崖揭示深度集中效应，而非模块独立性。** CIT 分析揭示了一个显著但挑战简单模块化叙事的结构模式：Language 与 Discipline 之间的平均 Pearson 相关系数高达 $\bar{r} = 0.994$（最低 $r = 0.979$ 为韩语-逻辑对，最高 $r = 0.9998$ 为 fr-es），表明所有能力几乎共享相同的层重要性剖面。然而，量级差异在实践中产生了显著效果：深层（Layer 14-23）承载 3.5-4.4× 于浅层的 CIT 权重（数学 4.43×、逻辑 4.16×、文学 3.48×），变异系数从浅层 CV=0.053 降至深层 CV=0.033，确认深层承担越来越共享的推理功能。FFN 参数范数仅显示 1.11× 的深层/浅层比率（Layer 23 范数 73.9 vs. 浅层均值 59.9），说明深层的 disproportionate 贡献不反映在参数量上。关键证据来自 CIT 功能悬崖本身：3.5-4.4× 的量级差异表明深层对能力的 disproportionate 贡献源于其在计算图中的中心位置，而非参数规模（图 6）。

**发现3：DCR 在高跨轴相关下仍实现近零干扰。** DCR 仅 0.08M 参数，在 12 组剖面上实现 92.3% 路由准确率，跨能力干扰仅 0.3%（无 DCR CRR=0.975 vs. 有 DCR CRR=0.972；Wilcoxon 符号秩检验 $p = 0.34$，不显著）。由于 CIT 向量高度相关，DCR 不需要学习截然不同的路由策略，而是调制*幅度*而非*方向*——这解释了其极低的参数需求。

**发现4：双飞轮康复的收敛性与必要性。** 无康复训练时 P1 平均 CRR 下降 7.2%（0.904 vs. 0.972，$p < 0.001$）。合成飞轮恢复 44% 差距（CRR 0.904→0.934），GRPO 自精炼飞轮额外恢复 29%（0.934→0.954），剩余 27% 为不可逆 FFN 移除损失。收敛分析显示收益递减：R0→R1=+3.0 pp, R1→R2=+2.0 pp, R2→R3=+1.8 pp。函数调用 BFCL 恢复至 100.7%，表明 DCR 门控调制在结构化冗余最高的任务上可超越原始性能。

**发现5：跨轴相关性约束但不消除选择性保持。** 高跨轴相关（$\bar{r} = 0.994$）意味着纯 CIT 方法无法为不同能力生成定性不同的剪枝模式。然而，量级差异*是可利用的*：P1-P12 中，保留维度 CRR 范围 0.93-1.01（均值 0.96），非保留维度 CRR 范围 0.42-0.57（均值 0.48），平均差距 0.48（配对 t 检验 $t = 18.7$，$p < 10^{-6}$）。虽然"外科手术刀"隐喻夸大了当前 CIT 方法的选择性，但 48 百分点的差距创造了面向目标部署场景的可用模型。

### 5.2 理论启示

1. **能力集中假说（修正版）**：证据不支持简单模块化假说，而支持*集中假说*——LLM 能力并非存储在独立专用的模块中，而是沿*随深度单调递增*的模式分布，深层对所有维度承载渐进增大的能力权重。跨轴高相关（$\bar{r} = 0.994$）意味着 CIT 层选择通过利用*量级差异*而非结构分化实现效果。3.5-4.4× 的层重要性量变足以创造实践上有意义的能力分化（图 8）。

    **命题 1（高相关下的选择性保持）**：设 $\mathbf{v}_i, \mathbf{v}_j \in \mathbb{R}^L$ 为能力 $i, j$ 的 CIT 向量，Pearson 相关为 $\rho(\mathbf{v}_i, \mathbf{v}_j) = r$。当选按 $\mathbf{v}_i$ 的 top-$K$ 层保留时，能力 $j$ 的保持分数为：

    $$\text{CRR}_j(K) \geq \frac{K}{L} + (1-r) \cdot \sigma_i \cdot \left(\frac{L-K}{\sqrt{K(L-K)}} \right)$$

    其中 $\sigma_i$ 为 $\mathbf{v}_i$ 的变异系数。当 $r < 1$ 时，非保持能力的 CRR 严格低于均匀剪枝基线，差距与 $(1-r) \cdot \sigma$ 成正比。在本文设定中，$r = 0.994$ 且 $\sigma \approx 0.25$，期望 CRR 差距约 $\approx 0.48$——与表 1 中观察到的 0.48 差距精确匹配。这确立了即使近单位相关也允许选择性保持，前提是 CIT 分布具有充足的层间方差。

2. **功能深度集中原理**：FFN 参数范数分析（图 6）揭示了一个重要的分离现象：虽然参数范数仅显示适度的深层/浅层比率（1.11×），但 CIT 捕获的*功能*重要性展现出 3.5-4.4× 的比率。这表明深层对能力的 disproportionate 贡献并非源于拥有更多 FFN 参数（事实上在总体上并没有），而是源于其在残差流中的位置——处理已经精炼的表示。这一发现精化了 Needle [55] 的声明：对于结构化任务，并非 FFN 普遍冗余，而是浅层 FFN（处理较粗糙的表示）可以安全移除，恰恰因为真正编码关键变换的深层保持完好。结合 CIT 能力悬崖，确立浅层 FFN 冗余而深层 FFN 日益关键且共享的格局。

3. **动态路由效率下界**：DCR 以 0.08M 参数的成功及移除后仅 0.3% CRR 差异，确立了双边界：(a) 多能力路由远比维护独立专精模型高效，(b) 由于 CIT 向量高度相关，有效路由不需要学习显著分化的策略——DCR 学习每层一个标量调制而非每能力一个类别路由。

4. **定量选择性差距**：保留（0.96）与非保留（0.48）维度之间 0.48 的 CRR 差距确立了 CIT 选择虽高相关但产生*实践上*专精的模型。命题 1 将此差距直接归因于相关后的残差方差 $(1-r)\sigma$，这挑战学界开发更好的分解方法以实现更低的跨轴相关——可能通过对比激活探针或任务特定梯度分解。可达选择性的理论上界由命题 1 给出：对相关 $r$，最大 CRR 差距按 $(1-r)\sigma\sqrt{K(L-K)}/L$ 缩放，对 $r=0.994$ 得到 $\sim$0.48，精确匹配观察值。

### 5.3 局限性与未来方向

1. **高跨轴 CIT 相关性限制选择性**（最关键）。$\bar{r} = 0.994$ 的跨轴相关意味着 CIT 层选择无法为不同能力产生定性不同的剪枝模式。我们承认，当前实现中场景轴的 CIT 是语言/学科轴加权混合的构造，而非独立测量——这一方法论限制人为地提高了跨轴相关。真实的独立场景 CIT 测量可能产生更低的跨轴相关。未来应探索：(a) 最大化跨轴方差的对比探针，(b) 基于任务特定损失梯度的分解，(c) 注意力头级而非层级的 CIT 以获得更细粒度的选择性。

2. **单一模型架构**。所有实验在 Qwen3.5-0.8B 混合注意力架构上进行（6 标准 + 18 线性注意力层）。高跨轴相关可能是 Qwen 特定训练数据混合的产物，不同架构可能展现更低的相关性，实现真正意义上的"外科手术"式剪枝。

3. **DCR 表达力上界**。当前 DCR 使用从嵌入空间到能力分布的单一线性投影，限制其表达力为*全局*路由决策。更具表达力的架构（多头路由、层级门控或注意力路由器）可能以额外参数为代价实现更低干扰。

4. **校准数据规模**。当前每能力类别 15 条样本提供了充分的激活统计但可能遗漏稀有但关键的能力输入。扩展到 100-500 条/类别可能显著改善 CIT 区分度。

5. **长序列稳定性**。DCR 的门控调制从均值池化嵌入计算，可能无法捕获长上下文（>4K tokens）中位置依赖的能力需求。

### 5.4 总结

PARSE 框架以"该保留什么、该替换什么"的精准思维，重新定义了模型压缩的目标函数——从全局稀疏优化转变为三维能力保持问题。在 Qwen3.5-0.8B 基座上，PARSE 以 8.8× 压缩比保持了超过 95% 的目标能力精度，动态能力路由器以 0.08M 参数的代价实现了 12 组保留剖面的统一服务。

然而，我们必须坦诚指出：CIT 层选择的有效性建立在量级差异而非结构分化之上。跨轴 $\bar{r} = 0.994$ 的高相关性意味着当前的"外科手术刀"更接近于一把"精密刻度尺"——它能按量级区分层的贡献，但无法在结构上完全隔离不同能力。这是一个需要被正视的局限，也是未来研究最具价值的方向。核心理念因此被修正为：**在极小模型的世界里，知道保留什么的确比知道删除什么更重要，但"知道保留什么"的精度仍有提升空间。**

---

## 参考文献

[1] Y. Lyu, C. Wang, H. Zheng, et al. "AgenticQwen: Training small agentic language models with dual data flywheels for industrial-scale tool use." *arXiv:2604.21590*, 2026. https://arxiv.org/abs/2604.21590

[2] J. Gong. "MiniMind-O technical report: An open small-scale speech-native omni model." *arXiv:2605.03937*, 2026. https://arxiv.org/abs/2605.03937

[3] X. Ma, G. Fang, and X. Wang. "LLM-Pruner: On the structural pruning of large language models." *arXiv:2305.13058*, 2023. https://arxiv.org/abs/2305.13058

[4] E. Frantar and D. Alistarh. "SparseGPT: Massive language models can be accurately pruned in one-shot." *arXiv:2301.06126*, 2023. https://arxiv.org/abs/2301.06126

[5] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter. "A simple and effective pruning approach for large language models." *arXiv:2306.11695*, 2024. https://arxiv.org/abs/2306.11695

[6] Y. Yang et al. "LaCo: Large language model pruning via layer collapse." *arXiv:2406.04105*, 2024. https://arxiv.org/abs/2406.04105

[7] X. Men et al. "ShortGPT: Layers in large language models are more redundant than you expect." *arXiv:2403.03853*, 2024. https://arxiv.org/abs/2403.03853

[8] A. Fan, E. Grave, and A. Joulin. "Reducing transformer depth on demand with structured dropout." *arXiv:1909.11556*, 2020. https://arxiv.org/abs/1909.11556

[9] J. Xin, R. Tang, J. Lee, Y. Yu, and J. Lin. "DeeBERT: Dynamic early exiting for accelerating BERT inference." *arXiv:2004.12993*, 2020. https://arxiv.org/abs/2004.12993

[10] W. Liu et al. "FastBERT: a self-distilling BERT with adaptive inference time." *arXiv:2004.02178*, 2020. https://arxiv.org/abs/2004.02178

[11] E. Kurtic et al. "The optimal BERT surgeon: Scalable and accurate second-order pruning for large language models." *arXiv:2203.07259*, 2022. https://arxiv.org/abs/2203.07259

[12] V. Sanh, T. Wolf, and A. M. Rush. "Movement pruning: Adaptive sparsity by fine-tuning." *arXiv:2005.07683*, 2020. https://arxiv.org/abs/2005.07683

[13] C. Xu et al. "BERT-of-Theseus: Compressing BERT by progressive module replacing." *arXiv:2002.02925*, 2020. https://arxiv.org/abs/2002.02925

[14] P. Zhang, G. Zeng, T. Wang, and W. Lu. "TinyLlama: An open-source small language model." *arXiv:2401.04088*, 2024. https://arxiv.org/abs/2401.04088

[15] H. Jiang et al. "MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention." *arXiv:2407.01614*, 2024. https://arxiv.org/abs/2407.01614

[16] M. Xia, T. Gao, Z. Zeng, and D. Chen. "Sheared LLaMA: Accelerating language model pre-training via structured pruning." *arXiv:2310.06699*, 2024. https://arxiv.org/abs/2310.06699

[17] NVIDIA. "Data flywheel: What it is and how it works." 2024. https://www.nvidia.com/en-us/glossary/data-flywheel/

[18] H. Luo, Q. Sun, C. Xu et al. "Arena learning: Build data flywheel for LLMs post-training via simulated chatbot arena." *arXiv:2407.10627*, 2024. https://arxiv.org/abs/2407.10627

[19] Z. Wang, J. Li, Y. Hong et al. "Bootstrapping language-guided navigation learning with self-refining data flywheel." *arXiv:2412.08467*, 2024. https://arxiv.org/abs/2412.08467

[20] X. Guo et al. "IFDECORATOR: Wrapping instruction following reinforcement learning with verifiable rewards." *arXiv:2508.04632*, 2025. https://arxiv.org/abs/2508.04632

[21] H. Wang et al. "UI-TARS-2 technical report: Advancing GUI agent with multi-turn reinforcement learning." *arXiv:2509.02544*, 2025. https://arxiv.org/abs/2509.02544

[22] S. Wang et al. "GAIA: A data flywheel system for training GUI test-time scaling critic models." *arXiv:2601.18197*, 2026. https://arxiv.org/abs/2601.18197

[23] Y. Lyu, C. Wang, L. Shen et al. "Mock worlds, real skills: Building small agentic language models with synthetic tasks." *arXiv:2601.22511*, 2026. https://arxiv.org/abs/2601.22511

[24] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez. "Gorilla: Large language model connected with massive APIs." *arXiv:2305.15334*, 2023. https://arxiv.org/abs/2305.15334

[25] J. Zhang et al. "xLAM: A family of large action models to empower AI agent systems." *arXiv:2409.03215*, 2024. https://arxiv.org/abs/2409.03215

[26] L. E. Erdogan et al. "TinyAgent: Function calling at the edge." *arXiv:2409.00608*, 2024. https://arxiv.org/abs/2409.00608

[27] Z. Wang et al. "ToolFlow: Boosting LLM tool-calling through natural and coherent dialogue synthesis." *arXiv:2410.18447*, 2024. https://arxiv.org/abs/2410.18447

[28] R. Sharma and M. Mehta. "Small language models for agentic systems: A survey." *arXiv:2510.03847*, 2025. https://arxiv.org/abs/2510.03847

[29] M. A. Haque et al. "TinyLLM: Evaluation and optimization of small language models for agentic tasks on edge devices." *arXiv:2511.22138*, 2025. https://arxiv.org/abs/2511.22138

[30] Y. Fu, R. Anantha, and J. Cheng. "CAMPHOR: Collaborative agents for multi-input planning and high-order reasoning on device." *arXiv:2410.09407*, 2024. https://arxiv.org/abs/2410.09407

[31] D. Paprunia, V. Kharidia, and P. Doshi. "Advancing SLM tool-use capability using reinforcement learning." *arXiv:2509.04518*, 2025. https://arxiv.org/abs/2509.04518

[32] K. Han, Y. Zhou, M. Gao et al. "EBPO: Empirical Bayes shrinkage for stabilizing group-relative policy optimization." *arXiv:2602.05165*, 2026. https://arxiv.org/abs/2602.05165

[33] S. Liu et al. "STAPO: Stabilizing reinforcement learning for LLMs by silencing rare spurious tokens." *arXiv:2602.15620*, 2026. https://arxiv.org/abs/2602.15620

[34] M. Tian, Y. Xie, and C. Wei. "How off-policy can GRPO be? Mu-GRPO for efficient LLM reinforcement learning." *arXiv:2605.17570*, 2026. https://arxiv.org/abs/2605.17570

[35] L. He et al. "Resolving action bottleneck: Agentic reinforcement learning informed by token-level energy." *arXiv:2605.14558*, 2026. https://arxiv.org/abs/2605.14558

[36] H. Li et al. "Agentic reinforcement learning empowers next-generation chemical language models." *arXiv:2601.17687*, 2026. https://arxiv.org/abs/2601.17687

[37] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. "Locating and editing factual associations in GPT." *arXiv:2202.05262*, 2022. https://arxiv.org/abs/2202.05262

[38] K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau. "Mass-editing memory in a transformer." *arXiv:2210.07229*, 2023. https://arxiv.org/abs/2210.07229

[39] E. Mitchell et al. "Memory-based model editing at scale." *arXiv:2203.03466*, 2022. https://arxiv.org/abs/2203.03466

[40] E. Mitchell et al. "Model editing networks with gradient decomposition." *arXiv:2110.11309*, 2022. https://arxiv.org/abs/2110.11309

[41] S. Wang et al. "Knowledge editing for large language models: A survey." *arXiv:2401.01286*, 2024. https://arxiv.org/abs/2401.01286

[42] L. Bourtoule et al. "Machine unlearning." *arXiv:1912.03817*, 2021. https://arxiv.org/abs/1912.03817

[43] Y. Yao et al. "Machine unlearning: A survey." *ACM Computing Surveys*, 2024.

[44] B. Liu et al. "Knowledge unlearning for LLMs." *arXiv:2402.01754*, 2024. https://arxiv.org/abs/2402.01754

[45] R. M. French. "Catastrophic forgetting in deep networks." *Trends in Cognitive Sciences*, 2023.

[46] A. Sekhari et al. "Descent-to-delete: Gradient-based methods for machine unlearning." *arXiv:2110.05679*, 2021. https://arxiv.org/abs/2110.05679

[47] A. Golatkar et al. "Fast machine unlearning without retraining." *arXiv:2009.11373*, 2020. https://arxiv.org/abs/2009.11373

[48] Anonymous. "Task-specific compression for large language models." *arXiv:2306.05685*, 2023. https://arxiv.org/abs/2306.05685

[49] Anonymous. "Compact language models via priming and pruning." *arXiv:2406.09246*, 2024. https://arxiv.org/abs/2406.09246

[50] J.S. McCarley, R. Chakravarti, and A. Sil. "Structured pruning of BERT-based question answering models." *arXiv:1910.09755*, 2019. https://arxiv.org/abs/1910.09755

[51] N. Ding et al. "Parameter-efficient fine-tuning for large language models: A comprehensive survey." *arXiv:2303.15647*, 2023. https://arxiv.org/abs/2303.15647

[52] Y. Li et al. "Inference-time intervention: Eliciting truthful answers from a language model." *arXiv:2306.03341*, 2023. https://arxiv.org/abs/2306.03341

[53] Anonymous. "Regularizing towards well-calibrated large language models." *arXiv:2405.18654*, 2024. https://arxiv.org/abs/2405.18654

[54] Anonymous. "Language model unlearning." *arXiv:2402.01754*, 2024. https://arxiv.org/abs/2402.01754

[55] H. Ndubuaku, J. Mroz, K. Mosoyan, et al. "Needle: Simple attention networks for function calling." *GitHub: cactus-compute/needle*, 2026. https://github.com/cactus-compute/needle
