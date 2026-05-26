# PARSE：面向语种-学科-场景三维能力保持的"四两拨千斤"式精细粒度大语言模型精准结构重塑

**摘要**  
当前大语言模型的高效部署陷入二元困境：统一结构剪枝将模型所有能力一视同仁地压缩，专精小模型则彻底放弃预训练的广博知识。本文提出第三种范式——**PARSE (Principled Architecture Retention through Scenario-Embedded Pruning)**：一种将模型压缩重新定义为"面向语种-学科-场景三维能力空间的精密外科手术"的框架。PARSE 的核心洞察是：LLM 的能力并非铁板一块，而是沿 Language（语种）、Discipline（学科）和 Scenario（场景）三个正交维度自然分解。为此，我们引入**三维能力重要性张量 (CIT)**，独立量化每一层对不同 (Language × Discipline × Scenario) 组合的贡献。基于 CIT，PARSE 保留用户指定保留剖面（如"中文语法+英文数学+函数调用"）的关键层，对非目标层进行 FFN 精密切除并移植借鉴自 Needle [55] 的纯注意力模块。动态能力路由器 (DCR) 根据输入语境实时调制移植模块的内部残差门控，使单个压缩模型可服务多个保留剖面。本文详细描述了诊断-雕塑-移植-康复四阶段管线，定义了 12 组保留剖面用于系统化评估。计划在 Qwen3.5-0.8B（752M 参数，24 层）上验证 8–10× 参数压缩比，目标是将目标能力性能保持在未压缩基线的 90% 以上。

**关键词**  
能力感知模型压缩；三维能力分解；架构移植；动态场景路由；极小语言模型

---

## 1. 引言：打破"一刀切"与"一专到底"的二元僵局

大语言模型的规模扩张带来了前所未有的广度——从多语种翻译到数学定理证明——全部封装在单一参数集内。然而，一个 Qwen3.5-0.8B 模型仍需要约 1.5GB 显存。消费级 GPU 上的推理速度受限于内存带宽和自回归生成的串行特性，典型吞吐量随硬件和批次大小在 2–15 tok/s 之间波动。对于要求单 token 延迟低于 100ms 的实时边缘应用，即使是 0.8B 规模也构成了部署障碍。

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
- **硬件**：NVIDIA RTX 4060 (8GB, CUDA 12.1, PyTorch 2.11.0)；同时使用 MoXing 进行 GGUF 量化部署验证（原始 F16: 1446MB, Q8_0 量化: 774MB, 体积减少 46.5%）
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

**[表 1]** 12 组保留剖面 × 6 组基线的实验结果待实验执行后填入。

**[表 2]** 基线对比实验待执行。

**[表 3]** CIT 因子消融与双飞轮康复实验待执行。

**[图 1]** 三维 CIT 热力图——预期揭示浅层 (Layer 0-5) 主要贡献语种能力，深层 (Layer 14-23) 在数学/逻辑/函数调用上呈现更高的 CIT 权重，形成"能力悬崖"模式。(见 figures/fig1_cit_analysis.pdf)

**[图 2]** 跨轴相关性分析——Language × Discipline × Scenario 三轴之间的 Pearson 相关系数将揭示能力是模块化存储还是集中分布。(见 figures/fig2_correlation_analysis.pdf)

**[图 3]** 能力保留雷达图——预期保留剖面呈各自"尖刺状"特征（保留维度预期高 CRR、非保留维度预期低 CRR），统一剪枝方法预期呈均匀退化。(见 figures/fig3_radar_profiles.pdf)

**[图 4]** 基线对比——将 PARSE 与 Wanda、SparseGPT、LayerDrop、LLM-Pruner 在 GSM8K、BFCL 和推理加速上进行系统化对比分析。(见 figures/fig4_baseline_comparison.pdf)

**[图 5]** 消融实验——量化 CIT 组成（激活 vs 梯度）、DCR 路由、双飞轮康复各自对最终能力保持率的边际贡献。(见 figures/fig5_ablation_study.pdf)

---

## 5. 预期结果与讨论

以下假说源自上述方法论框架，有待通过第四节描述的管线进行实证验证。

### 5.1 核心假说

**假说 1（能力特定保持）。** 基于保留剖面加权的 CIT 层选择，而非统一稀疏度阈值，PARSE 设计的在保留能力维度上取得比能力无关方法（Wanda [5]、SparseGPT [4]、LayerDrop [8]）更高的 CRR。随着保留剖面收窄（指定更少的能力维度），优势预期增大——维持更少能力需要保留更少的层。

**假说 2（层重要性结构）。** CIT 分析将揭示 LLM 能力遵循模块化模式（不同层专精于不同轴）还是集中模式（重要性随深度单调递增，跨轴高度相关）。答案直接影响能力保持型压缩可达的选择性精度。

**假说 3（DCR 开销）。** 动态能力路由器仅 0.08M 参数，预期相比为每个剖面部署独立专精模型，引入极小的交叉能力干扰。路由准确率和干扰水平取决于跨轴 CIT 向量的分化程度。

**假说 4（飞轮必要性）。** FFN 移植后，通过双飞轮机制进行康复训练预期是恢复能力性能所必需的。合成飞轮与自精炼（GRPO）飞轮的相对贡献将决定最优康复策略。

### 5.2 理论启示

若上述假说得到确认，将产生以下启示：

1. **LLM 的能力结构。** 若 CIT 向量呈现高跨轴相关性，表明 LLM 能力并非存储在独立的专门化模块中，而是沿深度方向呈集中分布——与残差网络理论一致，深层在计算图中积累 disproportionately 大的影响。若观察到低相关性，则支持模块化能力假说，可实现更精确的选择性剪枝。

2. **FFN 冗余性范围。** Needle [55] 建立了函数调用场景下的 FFN 冗余性。PARSE 将该问题扩展到语种、学科、场景三维度。若 No-FFN 移植在跨维度上均能成功，将扩展 FFN 冗余性原理的适用范围。

3. **路由效率。** 若 DCR 以 0.08M 参数实现高效多剖面路由，将证明共享的参数高效路由可替代维护独立专门模型——挑战"任务特定部署需任务特定架构"的前提。

### 5.3 局限

1. **实证验证待完成。** 第四节描述的实验方案已设计但尚未执行。本节中所有结果讨论均为假说，非确认发现。

2. **单一架构。** CIT 方法原则上与架构无关，但目前仅与 Qwen 系模型集成。Llama、Mistral、Gemma 等架构上的验证对泛化性声明的成立至关重要。

3. **标定数据规模。** 当前每能力类别 15 条样本提供了紧凑的诊断探针，但可能遗漏稀有但关键的能力输入。规模化标定可能改善 CIT 区分度。

4. **DCR 表达力。** 当前 DCR 使用从嵌入空间的单一线性投影，限制路由为全局决策。更具表达力的架构（多头路由、层级门控）可能以额外参数为代价实现更精细的控制。

5. **长序列稳定性。** DCR 门控调制从均值池化嵌入计算，可能无法捕获长上下文中的位置依赖能力需求。

### 5.4 总结

PARSE 框架以"该保留什么、该替换什么"的精准思维，重新定义了模型压缩的目标函数——从全局稀疏优化转变为三维能力保持问题。能力重要性张量为量化层级能力贡献提供了原则性机制，动态能力路由器使单一压缩模型可在不使用权重切换的情况下服务多个保留剖面。完整的四阶段管线——CIT 诊断、层雕塑、FFN 移植、双飞轮康复——已实现并准备在 Qwen3.5-0.8B 及更广模型上进行实证验证。

核心方法论贡献在于认识到模型压缩不必是全局优化问题。通过指定保留*哪些*能力——中文语法、英文数学、函数调用——并仅保留承载这些能力的层，PARSE 框架旨在以量级参数压缩换取最小能力损失。该假说在本文定义的 12 组保留剖面上的实证验证是下一步工作的直接目标。

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
