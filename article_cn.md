# PARSE：面向语种-学科-场景三维能力保持的精细粒度大语言模型精准结构重塑

**摘要**
当前大语言模型的高效部署陷入二元困境：统一结构剪枝将模型所有能力一视同仁地压缩，专精小模型则彻底放弃预训练的广博知识。本文提出第三种范式——**PARSE (Principled Architecture Retention through Scenario-Embedded Pruning)**：一种将模型压缩重新定义为"面向语种-学科-场景三维能力空间的精密外科手术"的框架。PARSE 的核心洞察是：LLM 的能力并非铁板一块，而是沿 Language（语种）、Discipline（学科）和 Scenario（场景）三个概念上独立（笛卡尔积）的维度自然分解。为此，我们引入**三维能力重要性张量 (Capability Importance Tensor, CIT)**，独立量化每一层对不同 (Language × Discipline × Scenario) 组合的贡献。基于 CIT，PARSE 保留用户指定保留剖面（如"中文语法+英文数学+函数调用"）的关键层，对非目标层进行 FFN 精密切除并植入采用 Needle [54] 极简设计的门控残差直通模块——该方法设计用于标准 Transformer 架构。**动态能力路由器 (Dynamic Capability Router, DCR)**（0.08M 参数）根据输入语境实时调制移植模块的内部残差门控，使单个压缩模型可服务多个保留剖面。本文详细描述了诊断-雕塑-移植-康复四阶段管线，给出了 DCR 训练算法的完整规格（联合损失函数、优化器、学习率调度及防止路由退化的负样本策略），定义了 12 组保留剖面用于系统化评估，并明确了与知识蒸馏基线和参数匹配基线的公平对比方案。在 Qwen3.5-0.8B（使用 Mamba 风格选择性状态空间架构，无标准 FFN）上完成了初步激活基 CIT 诊断探测，测得跨轴平均 Pearson 相关系数 $\bar{r} = 0.9945$，最小跨轴对为 Korean-Math（$r = 0.980$），能力悬崖比 3.8–4.0×。所有 12 组剖面在因子化 CIT 下收敛至相同层选择（均剪枝 {0,1,2,4,5,8}），实证证实高相关下因子化 CIT 退化为深度加权排序。移植、飞轮康复和完整基准评估有待后续执行。

**关键词**
能力感知模型压缩；三维能力分解（Capability-Aware Model Compression）；架构移植（Architecture Transplantation）；动态场景路由（Dynamic Capability Router）；极小语言模型（Tiny Language Models）

---

## 1. 引言：打破"一刀切"与"一专到底"的二元僵局

大语言模型的规模扩张带来了前所未有的广度——从多语种翻译到数学定理证明——全部封装在单一参数集内。然而，一个 Qwen3.5-0.8B 模型仍需要约 1.5 GB 显存。消费级 GPU 上的推理速度受限于内存带宽和自回归生成的串行特性，典型吞吐量随硬件和批次大小在 2–15 tok/s 之间波动（未压缩模型在 RTX 4060 等中端消费级 GPU 上的实测范围）。对于要求单 token 延迟低于 100ms 的实时边缘应用，即使是 0.8B 规模也构成了部署障碍。

对此，学界分化出两条路线。**压缩路线**采用统一稀疏约束：LLM-Pruner [3] 基于梯度耦合、SparseGPT [4] 一次性 50% 稀疏、Wanda [5] 使用权值-激活乘积评分、oBERT [11] 将压缩比推向极致。这些方法的共同前提是：所有层、注意力头和 FFN 神经元对所有模型能力的贡献均等。但 ShortGPT [7] 和 LaCo [6] 以实证证明这一假设完全不成立——深层在逻辑推理上的贡献呈指数级攀升，而浅层主要处理语法对齐。

**专精路线**走向了另一个极端。AgenticQwen [1] 利用双数据飞轮训练智能体小模型、Needle [54] 将 FFN 完全移除并以 26M 参数击败 270M 通用模型、MiniMind-O [2] 在 100M 参数下实现三模态 Omni 交互。Gorilla [24]、xLAM [25]、TinyAgent [26]、ToolFlow [27] 在各自领域做到极致。但这些模型如同一把锋利但单刃的刀——无法同时解决数学、翻译和多语种对话。

这二元景观揭示了一个明显的理论空白：**没有任何现有框架允许实践者精确指定要保留哪些能力（例如"中文语法+英文数学+函数调用"），并仅切除对该剖面无用的结构。** PARSE 正是为填补这一空白而生。全文中 **PARSE** 指称整个系统；其核心方法论称为**精细能力雕塑 (Fine-Grained Capability Sculpting, FGCS)**——第 3 节所述的三维分解与层选择性移植流程。

### 1.1 核心创新：三维能力分解

PARSE 建立在这样的洞察之上：**LLM 的能力沿语种 (Language)、学科 (Discipline) 和场景 (Scenario) 三个概念上独立的维度自然分解。** 一个对中文句法至关重要的层，可能对数学证明完全冗余。一个驱动函数调用精度的层，可能对逻辑推理毫无贡献。已有工作将这些能力视为不可分割的整体；PARSE 将能力视作可被独立保留、弱化或替换的光谱。

为此，我们提出**能力重要性张量 (Capability Importance Tensor, CIT)**——量化每一层对 (Language × Discipline × Scenario) 组合的贡献。CIT 的物理直觉类比电子学中的"电容矩阵"：每一层对特定能力的"响应强度"（容抗）决定了它能否从输入中提取该能力所需的信息。

### 1.2 与已有工作的本质区别

| 维度 | AgenticQwen [1] | Needle [54] | MiniMind-O [2] | **PARSE (本文)** |
|:---|:---|:---|:---|:---|
| **能力粒度** | 通用 Agent 场景 | 单一 Function Calling | 全模态 Omni | **三维 (Lang×Disc×Scen)** |
| **架构设计** | 通用 Transformer | 纯注意力 (无 FFN) | Thinker-Talker | **保留+移植融合** |
| **压缩方式** | 数据飞轮训练 | 从头设计 | 从头设计 | **基于已有模型精密手术** |
| **多能力支持** | 否 (单场景) | 否 (单任务) | 否 (全能设计) | **是 (多剖面动态路由)** |

### 1.3 研究贡献

1. **三维能力分解理论**：首次形式化 LLM 能力沿 Language-Discipline-Scenario 三轴分解，提出因子化 CIT 以高效计算，并明确讨论因子化假设及其在高跨轴相关下的局限性。

2. **PARSE 框架**：完整的"诊断-雕塑-移植-康复"四阶段流程，支持用户自定义保留剖面。移植方法设计用于标准 Transformer 架构（含显式 FFN 子模块的模型，如 LLaMA、Qwen2.5、Mistral）。给出了包括损失函数 $\mathcal{L}_{dcr} = \mathcal{L}_{lm} + \beta \cdot \mathcal{L}_{cls}$（含负样本以防止 DCR 路由退化）、优化器 (AdamW [62])、学习率调度在内的完整训练规格。

3. **动态能力路由器 (Dynamic Capability Router, DCR)**：仅 0.08M 参数，给出了基于低秩分解（$W_r = U V^T$）和层分组共享的详细参数构成推导，使单个压缩模型在多剖面间灵活切换而不需切换权重。

4. **12 剖面评估框架**：定义了 12 组保留剖面 (P1–P12) 覆盖三维能力空间，设计了包含知识蒸馏基线（含蒸馏温度和学生架构规格）和参数匹配对比的完整实验方案。已在 Qwen3.5-0.8B 上完成了初步激活基 CIT 诊断探测，测得跨轴平均 Pearson 相关系数 $\bar{r} = 0.9945$、最小跨轴对 Korean-Math（$r = 0.980$）和能力悬崖比 3.8–4.0×。所有 12 组剖面在因子化 CIT 下收敛至相同层选择（均剪枝 6 层），实证证实高相关导致三维分解退化。

---

## 2. 相关工作与能力鸿沟

### 2.1 统一剪枝的"同质化谬误"

结构化剪枝虽然具有架构兼容性优势，但陷入了"平均值陷阱"。LLM-Pruner [3]、SparseGPT [4]、Wanda [5] 优化的是全局稀疏度，对具体能力维度的损失无动于衷。ShortGPT [7] 的发现令人警醒：移除某些层可能在平均 PPL 上毫无影响，却对特定能力造成隐性塌方。LaCo [6] 从层级塌缩的视角审视剪枝，Movement Pruning [12] 通过一阶梯度信息学习稀疏性。LayerDrop [8] 引入训练期 Dropout，DeeBERT [9] 和 FastBERT [10] 实现置信度早退。BERT-of-Theseus [13] 渐进式模块替换，TinyLlama [14] 从头预训练小模型。MInference [15] 加速长上下文，LLM-Shearing [16] 灵活剪枝并持续预训练恢复。BERT-QA-Pruning [50] 针对 QA 任务，PEFT [51] 综述了参数高效方法，Task-Specific Compression [48] 和 Compact Language Models [49] 探索了领域自适应压缩。但这些均未解决"能力保留剖面"这一核心设计自由度。

### 2.2 知识编辑与遗忘：精度范式的启示

ROME [37] 和 MEMIT [38] 以秩-1 更新精确定位并编辑事实关联。MEND [39] 通过超网络预测参数更新，SERAC [40] 以外部记忆规避直接修改。Wang 等人 [41] 提供了知识编辑系统化综述。机器遗忘方面，SISA [42] 通过分片训练实现高效遗忘，后续综述 [43,44] 延伸至语言模型。灾难性遗忘 [45] 始终是核心挑战。Descent-to-Delete [46] 引入梯度删除，Fast-Machine-Unlearning [47] 加速流程。Inference-Time Intervention [52] 实现了推理时行为引导，Regularizing [53] 提供了校准可靠性保障。如果单个事实可被定位到具体层，那么整个能力（如数学推理）也应该能追溯到一组特定的层——这正是 PARSE 将能力保留类比为定向编辑的理论出发点。

### 2.3 数据飞轮与自改进

NVIDIA 的数据飞轮概念 [17] 启发了自改进训练系统。AgenticQwen [1] 提出双飞轮架构。ArenaLearning [18] 首创 AI 模拟竞技场。SRDF [19] 引入自精炼数据管道。IFDecorator [20] 专注指令遵循，UI-TARS-2 [21] 拓展 GUI Agent，GAIA [22] 构建 GUI 评判飞轮，SynthAgent [23] 提出合成世界训练。PARSE 在移植后的"康复"阶段充分利用了双数据飞轮策略。

### 2.4 Agent 系统与工具调用

Gorilla [24] 首创 LLM API 调用，xLAM [25] 扩展至大规模动作模型，TinyAgent [26] 实现边缘函数调用，ToolFlow [27] 引入图采样工具组合。Sharma 和 Mehta [28] 与 Haque 等人 [29] 建立了 SLM Agent 评估基准。CAMPHOR [30] 提出多 Agent 协作架构。SLM-ToolUse-GRPO [31] 专研 GRPO 增强工具调用。

### 2.5 GRPO 强化学习

DeepSeekMath [55] 提出 GRPO (Group Relative Policy Optimization) 算法。EBPO [32] 通过经验贝叶斯收缩稳定基线估计，STAPO [33] 沉默伪 token，Mu-GRPO [34] 提升训练效率。ActFocus [35] 通过 token 级能量重加权解决动作瓶颈，ChemCRAFT [36] 将 Agent RL 应用于药物设计。PARSE 在康复阶段采用 GRPO 优化进行能力恢复。

### 2.6 专用架构与通用能力的矛盾

Needle [54] 的纯注意力设计和 MiniMind-O [2] 的 Thinker-Talker 双路径解耦证明了极致效率。Needle [54] 使用 ZCRMSNorm（Zero-Centered RMSNorm，初始化时 $\gamma=0$ 使模块为恒等映射至尺度），以 26M 参数在函数调用上超越 270M–600M 通用模型。但它们代表的是"去通用化"的路径。PARSE 反其道而行之：我们保留模型中真正承载"通用知识"的关键层，仅仅替换掉那些对目标剖面无用的冗余层——在"通才"与"专才"之间撕开了一道缺口。

---

## 3. PARSE 方法论：三维能力外科手术

### 3.1 形式化问题定义

令 $M$ 为预训练模型（$L$ 层），能力空间 $\mathcal{C} = \mathcal{L}_{ang} \times \mathcal{D}_{isc} \times \mathcal{S}_{cen}$。其中 $\mathcal{L}_{ang}$ 为语种轴（中/英/日/法/德/俄/西/韩，8 类），$\mathcal{D}_{isc}$ 为学科轴（数学/物理/逻辑/历史/地理/文学，6 类），$\mathcal{S}_{cen}$ 为场景轴（函数调用/代码生成/数学推理/翻译/通用对话，5 类），共 19 个能力维度。

能力轴的选择以已有评估基准为参考：语种类别覆盖主要语系 [56]；学科类别参考知识密集型基准 MMLU [57]（将 57 个 MMLU 学科归并入 6 个顶层领域）；场景类别反映小模型 Agent 研究优先关注的部署场景 [25,26,27,28,29]。

**保留剖面** $\mathcal{P} \subset \mathcal{C}$ 指定必须保留的能力组合。压缩目标为：

$$\min |M'| \quad \text{s.t.} \quad \forall c \in \mathcal{P}: \Delta(c) \leq \epsilon$$

### 3.2 能力重要性张量 (CIT)

对于每一层 $l$ 和能力组合 $c$：

$$\text{CIT}(l, c) = \alpha \cdot A(l, c) + (1-\alpha) \cdot G(l, c)$$

其中 $A(l,c)$ 为激活电容 (Activation Capacitance)，$G(l,c)$ 为梯度灵敏度 (Gradient Sensitivity)：

$$A(l, c) = \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} \|h_l(x)\|_1, \quad G(l, c) = \sum_{\substack{(n, p) \in \text{FFN}_l}} \left|\frac{\partial \mathcal{L}_c}{\partial p} \cdot p\right|$$

此处 $\mathcal{D}_c$ 为能力 $c$ 的紧凑标定数据集，$h_l(x)$ 为层 $l$ 的隐藏状态激活，$\mathcal{L}_c$ 为语言建模损失，$G$ 中的求和限定于 FFN 参数（gate\_proj, up\_proj, down\_proj），与移植范围匹配。设计中 $\alpha = 0.6$ 平衡激活和梯度信号。**初步测量说明**：本文报告的诊断 CIT 结果（相关性 r̄、能力悬崖比、层选择）使用激活基探针（等价于 $\alpha = 1.0$, $G=0$），每类 10 条标定样本。原因是梯度灵敏度需要标准 FFN 参数计算梯度-权重乘积，而 Qwen3.5 的 Mamba 风格架构无此类参数。激活基 CIT 捕获了主导性的深度集中信号；梯度成分 $G(l,c)$ 的边际贡献留待消融 A1 在标准 Transformer 架构上量化。

为降低计算复杂度，采用因子化分解：

$$\text{CIT}(l, lang, disc, scen) = \text{CIT}_{lang}(l, lang) \cdot \text{CIT}_{disc}(l, disc) \cdot \text{CIT}_{scen}(l, scen)$$

**因子化假设与局限**。乘法因子化隐含假设语言、学科和场景三轴的贡献在 log 空间中近似正交。在 Qwen3.5-0.8B 上的激活基 CIT 测量证实了已有层重要性研究 [7,37] 揭示的强深度集中现象：跨轴平均 Pearson 相关系数 $\bar{r} = 0.9945$（Lang-Disc: $r = 0.994$, Lang-Scen: $r = 0.992$, Disc-Scen: $r = 0.998$；最小跨轴对 Korean-Math: $r = 0.980$）。在此高相关下，所有 12 组保留剖面（P1–P12）收敛至相同的层选择结果（均剪枝层 {0,1,2,4,5,8}），三维分解退化为单一深度加权排序。这实证验证了前述理论担忧。我们在第 5.3 节讨论这一局限性，并通过对比 CIT 和完整（非因子化）张量计算提供诊断和缓解路径。

**对比 CIT (Contrastive CIT)**。为增强跨轴区分度，引入对比变体（该变体的实证评估留待后续实验）：

$$\text{CIT}^{contrast}(l, c) = \max\left(0,\; \text{CIT}(l, c) - \lambda \cdot \frac{1}{|\mathcal{C}|-1}\sum_{c' \neq c} \text{CIT}(l, c')\right)$$

其中 $\lambda \in [0, 1]$ 为对比强度：$\lambda = 0$ 退化为标准 CIT，$\lambda = 1$ 为纯对比模式。对比 CIT 抑制对所有能力同等重要的共享层，放大了能力专属层，直接应对高跨轴相关导致的区分度限制。

### 3.3 能力保留层选择

基于保留剖面 $\mathcal{P}$ 计算保留权重复合分数 $S_{preserve}(l) = \sum_{c \in \mathcal{P}} w_c \cdot \text{CIT}(l,c)$。保留前 $K$ 层（$K = \lceil L \cdot (1 - \tau/2) \rceil$），将其余层的 FFN 切除。因子 $\tau/2$（而非 $\tau$）体现了移植仅移除每层 FFN 参数（~65%）而保留自注意力（~35%）的事实。将 $\tau$ 折半纳入层选择阈值，确保总参数缩减匹配目标压缩比。

对于标准 Transformer 架构，使用完整 softmax 注意力的层（区别于高效近似注意力）**始终保留**，无论其 CIT 分数如何。对于 Qwen3.5-0.8B 的混合架构，位于 {3, 7, 11, 15, 19, 23} 的 6 个含标准注意力组件的层被强制保留（尽管它们使用 Mamba 风格选择性状态空间模块而非标准 FFN）。此约束有实证依据：已有层移除研究 [7] 表明移除完整注意力层会造成灾难性退化。

### 3.4 动态能力路由器 (DCR)

$$R(x) = \text{softmax}(W_r \cdot \text{mean}(h_{embed}(x)) + b_r)$$

$$g_l(x) = \sigma\left(g_l^{base} + \sum_{c \in \mathcal{C}} R_c(x) \cdot g_{l,c}^{specialized}\right)$$

DCR 设计目标为 0.08M 参数（约占目标 ~85M 压缩模型的 0.09%）。参数构成：(1) 路由器 MLP 采用瓶颈维度 $\lfloor d_{model}/8 \rfloor = 256$（$d_{model}=2048$），通过低秩分解 $W_r = U V^T$（$U \in \mathbb{R}^{2048 \times 32}, V \in \mathbb{R}^{256 \times 32}$，共 $2048\times 32 + 256\times 32 = 73,728$ 参数）；(2) $g_{l,c}^{specialized}$ 通过在每 3 层移植层之间分组共享压缩至 $\lceil L_{transplanted}/3 \rceil \times |\mathcal{C}| \approx 4 \times 240 = 960$ 参数。加上偏置项，总计约 0.075M，四舍五入为 0.08M。

**DCR 训练**。DCR 参数 $W_r, b_r, g_{l,c}^{specialized}$ 在移植后康复阶段联合训练。训练目标联合两个信号：(1) **能力分类损失** $\mathcal{L}_{cls} = -\sum_{c \in \mathcal{C}} y_c \log R_c(x)$，其中 $y_c$ 来自每个训练序列的已知来源（如 GSM8K 样本标签为 math\_reasoning），提供弱监督以鼓励路由器激活保留能力的门控；(2) **任务特定语言建模损失** $\mathcal{L}_{lm}$，DCR 门控调制影响哪些层参与前向计算。联合损失为 $\mathcal{L}_{dcr} = \mathcal{L}_{lm} + \beta \cdot \mathcal{L}_{cls}$（$\beta = 0.1$）。使用 AdamW [62] 优化，学习率 $1\times 10^{-4}$，权重衰减 0.01，线性预热 100 步后余弦衰减，共 3 轮飞轮训练。为防止路由器退化为平凡映射，训练数据同时包含保留剖面内和剖面外的能力样本（非保留能力的 $y_c = 0$），确保 DCR 学习判别性路由而非统一激活所有保留能力门控。

### 3.5 四阶段手术流程

**阶段一：诊断**。构建各能力的极简探针集 $\mathcal{D}_c$（10-20 条），通过合成数据飞轮 [1,18] 生成以覆盖边缘情况。

**阶段二：雕塑**。计算各轴边际 CIT，对用户指定保留剖面 $\mathcal{P}$ 乘法组合，选择前 $K$ 层保留。其余层指定为移植。

**阶段三：移植**（设计用于标准 Transformer 架构，含显式 FFN 子模块）。对每个移植层：(1) 保留自注意力模块（Q, K, V, O 投影）；(2) 移除 FFN（gate\_proj, up\_proj, down\_proj），替换为恒等映射；(3) 插入 NoFFN 直通块：$\text{output} = (1 - g_l) \cdot h + g_l \cdot \text{LN}(h)$；(4) 在 MLP 子模块注册前向钩子，使 NoFFN 块的门控残差计算在推理时替换 Identity-FFN 输出；(5) 初始化门控扰动为零（$g_l = \sigma(0) = 0.5$，移植块初始以 50% 直通运行，确保数值稳定）。

**阶段四：康复**。应用双数据飞轮：(1) **合成飞轮**：使用教师模型为每种 (Language × Discipline × Scenario) 组合生成目标标定数据，辅以 Self-Instruct 扩展和 Persona 注入 [1]；(2) **自精炼飞轮**：压缩模型生成响应，Critic 模型（灵感来自 GAIA [22]）评分，高质量轨迹通过 GRPO 优化 [55,32,33] 重新注入。

---

## 4. 实验设计与对比矩阵

### 4.1 实验环境

- **诊断探测硬件**：Apple M4（16 GB 统一内存），PyTorch 2.5.1，用于激活基 CIT 诊断测量
- **完整管线硬件**：NVIDIA RTX 4060（8 GB VRAM），CUDA 12.1，用于移植和评估
- **基座模型**：Qwen3.5-0.8B（752M 参数，24 层）。**架构说明**：Qwen3.5-0.8B 使用 Mamba 风格选择性状态空间（linear attention）架构，每层主计算为 `linear_attn` 模块而非标准 Self-Attention + FFN。全部 24 层使用该机制；{3,7,11,15,19,23} 这 6 层额外包含标准 softmax 注意力组件用于交叉引用。CIT 诊断探测（激活捕获）是架构无关的。FFN 切除移植方法（Stage 3）设计用于标准 Transformer 架构（LLaMA、Qwen2.5、Mistral 等）。保留 Qwen3.5-0.8B 作为 CIT 测量目标（因其可用性），完整管线执行需有显式 FFN 子模块的模型。隐藏维度 $d_{model} = 2048$
- **能力空间**：语种 8 类 × 学科 6 类 × 场景 5 类 = 240 种能力组合
- **标定数据**：每类 10 条用于初步 CIT（共 190 条）；设计目标每类 15 条用于完整梯度 CIT
- **部署验证**：支持通过 MoXing（开源 GGUF 模型推理框架，https://github.com/cycleuser/MoXing）进行 GGUF (GPT-Generated Unified Format) 量化部署验证

### 4.2 12 组保留剖面设计

**表 1：保留剖面定义**（注："all" = 该轴全部类别：语种 8 类、学科 6 类、场景 5 类）

| 剖面 | 语种 | 学科 | 场景 | 描述 |
|:---|:---|:---|:---|:---|
| P1 | zh, en | math, logic | fc, math\_reasoning | 中英 STEM + Agent |
| P2 | zh, en, ja | math, physics | all | 东亚语种 + STEM |
| P3 | en | math | all | 英文数学专家 |
| P4 | zh | all | all | 中文全能力 |
| P5 | all | math, logic, physics | fc | 多语种 STEM Agent |
| P6 | zh, en | all | fc, code | 双语开发者 Agent |
| P7 | all | math | math\_reasoning | 多语种数学求解器 |
| P8 | zh, en, ja, fr | all | translation | 四语种翻译器 |
| P9 | all | all | fc | 通用函数调用器 |
| P10 | zh, en | all | all | 双语全能力 |
| P11 | all | math, logic | all | 通用 STEM 保留 |
| P12 | zh, en | math, logic, physics | fc, code, math\_reasoning | 全目标保留 |

### 4.3 基线方法

1. Wanda [5]：权值-激活乘积剪枝
2. SparseGPT [4]：二阶一次性剪枝
3. LayerDrop [8]：结构化层移除
4. LLM-Pruner [3]：梯度耦合结构剪枝
5. Needle [54]：完全 FFN 移除（仅函数调用）
6. **知识蒸馏基线**：从 Qwen3.5-0.8B 教师模型到目标 ~85M 学生 Transformer（例如 12 层、768 维）的标准 logit 级蒸馏，蒸馏温度 $\tau_{KD}=3.0$，联合硬标签和软标签损失——直接检验 PARSE 的手术式方法是否优于最直接的小模型创建方式
7. 原始 Qwen3.5-0.8B：未压缩基线
8. Qwen2.5-0.5B：Qwen2.5 系列中最接近的小模型（0.5B 参数），作为同一参数规模区间的从头训练参考点

**公平对比说明**。标准剪枝基线通常以 50% 稀疏度运行（~376M 参数）。为公平对比，计划评估每种基线的两个变体：(a) 标准 50% 稀疏度设置，(b) 产生与 PARSE 目标匹配的 ~85M 活动参数的稀疏度水平。此双设置对比将方法效果与参数预算效果分离。

### 4.4 评估指标

- **能力保留率 (Capability Retention Ratio, CRR)**：$\text{CRR}(c) = \text{Metric}_{compressed}(c) / \text{Metric}_{original}(c)$。CRR > 1 表示压缩模型在该能力上*超越*原始性能（可通过 DCR 门控放大实现）
- **参数缩减比 (Parameter Reduction Ratio, PRR)**：$(|M_{original}| - |M_{compressed}|) / |M_{original}|$
- **推理加速比 (Inference Speedup)**：参考硬件（RTX 4060）上 batch size 1 的 tok/s，128 token 生成平均
- **交叉能力干扰 (Cross-Capability Interference, CCI)**：非保留能力的平均退化，$\text{CCI} = \frac{1}{|\mathcal{C} \setminus \mathcal{P}|} \sum_{c \notin \mathcal{P}} (1 - \min(\text{CRR}(c), 1))$。CCI 越低表示选择性保持越干净
- **困惑度 (Perplexity, PPL)**：标准语言建模 PPL
- **任务特定准确率**：GSM8K [58] 数学推理，BFCL [59] 函数调用，HumanEval [60] 代码生成，BLEU [61] 翻译质量
- **统计显著性**：配对 t 检验，Bonferroni 校正，$p < 0.05$

*初步 CIT 诊断探测（激活基）已完成；移植、飞轮康复及完整评估指标有待实验执行。*

### 4.5 消融实验设计

**A1. CIT 组成消融**：对比完整 CIT（激活+梯度，$\alpha=0.6$）vs 纯激活（$\alpha=1.0$，当前测量值）vs 纯梯度（$\alpha=0.0$）

**A2. DCR 有效性**：对比 PARSE (含 DCR) vs PARSE (不含 DCR，每剖面独立模型)

**A3. 飞轮康复**：对比有/无双飞轮、仅合成飞轮、仅 GRPO 飞轮

**A4. 保留剖面敏感度**：$|\mathcal{P}|$ 从 1 到 20 变化，测量对 PRR 和 CRR 的影响

**A5. 稀疏度扫描**：压缩比从 2× 到 16×，刻画性能-压缩权衡曲线

**A6. 因子化 vs 完整 CIT**：对比乘法因子化与完整（非因子化）CIT，量化因子化近似的信息损失

*消融结果有待实验执行。*

### 4.6 层间模式

在 Qwen3.5-0.8B 上的激活基 CIT 测量确认了"能力悬崖"模式：浅层（0–5）主要贡献表层语言特征；中间层（6–15）在学科和场景间均匀分布；深层（16–23）所有能力维度上 CIT 得分成比例积累，平均深/浅比为 3.8–4.0×。高跨轴相关性（$\bar{r} = 0.9945$）表明能力重要性遵循集中（深度依赖）模式而非模块化模式，所有 12 组剖面收敛至相同层选择。更细粒度的分析（注意力头级 CIT、任务特定梯度分解）能否揭示层级不可见的模块化结构，仍是待解的实证问题。

---

## 5. 假说与预期结果

以下假说源自上述方法论框架，有待通过完整管线进行实证验证。

### 5.1 核心假说

**假说 1（能力特定保持）**。基于保留剖面加权的 CIT 层选择，预期比能力无关方法（Wanda [5]、SparseGPT [4]、LayerDrop [8]）在相同参数预算下取得更高保留能力 CRR。随保留剖面收窄，优势预期增大。

**假说 2（层重要性结构）**。初步 CIT 测量支持集中假说（$\bar{r} = 0.9945$，全剖面收敛），但更细粒度的分析（因子化分析 A6、对比 CIT）可能揭示额外结构。

**假说 3（DCR 开销）**。DCR 设计目标 0.08M 参数，预期相比独立部署剖面模型引入极小的交叉能力干扰。辅助分类损失 $\mathcal{L}_{cls}$ 和负样本策略提供监督以鼓励 DCR 学习能力判别性路由。

**假说 4（飞轮必要性）**。FFN 移植后通过双飞轮康复训练预期是恢复能力性能所必需的。

**假说 5（知识蒸馏对比）**。知识蒸馏基线预期取得有竞争力的平均性能，但在所有维度上均匀退化；而 PARSE 预期产生非对称保持（保留能力高 CRR，非保留能力低 CRR）。

### 5.2 理论启示

若上述假说得到确认：
1. **LLM 的能力结构**：初步测量支持集中假说——能力沿深度单调递增分布（$\bar{r}=0.9945$，深度集中比 3.8–4.0×）
2. **FFN 冗余性范围**：在跨维度上成功的 No-FFN 移植将扩展 Needle [54] 的 FFN 冗余性原理
3. **路由效率**：0.08M 参数的 DCR 有效路由将证明参数高效共享路由可替代独立专门模型

### 5.3 局限

1. **不完全的实证验证**。初步 CIT 诊断探测（激活基）已在 Qwen3.5-0.8B 上完成，得出了第 3.2 节报告的相关性和能力悬崖测量。移植、飞轮康复和完整基准评估（GSM8K, BFCL, HumanEval）有待执行。本节所有结果假说均基于方法论框架，待实证检验。

2. **CIT 因子化在高相关下的局限**。激活基 CIT 测量证实跨轴 $\bar{r} = 0.9945$（最小跨轴对 Korean-Math: $r = 0.980$），因子化 CIT 下所有 12 剖面收敛至相同层选择。完整（非因子化）CIT 计算和对比 CIT 变体分别用于诊断和缓解此局限。

3. **Qwen3.5 架构不匹配**。Qwen3.5-0.8B 使用 Mamba 风格选择性状态空间架构，无标准 FFN 子模块。CIT 诊断探测（隐藏状态激活）是架构无关的，但 FFN 切除移植（Stage 3）设计用于标准 Transformer 架构（LLaMA、Qwen2.5、Mistral 等含显式 gate\_proj/up\_proj/down\_proj 的模型）。梯度灵敏度 $G(l,c)$ 也无法在当前架构上计算。完整管线验证需选取标准 Transformer 架构的目标模型。

4. **参数预算与压缩比澄清**。常引用的 8.8× 压缩目标（752M→85M）需澄清：FFN 约占 65% 参数，即使移除全部 FFN 也仅剩 ~263M（35% × 752M），仅 2.86× 压缩。达到目标 8.8× 需额外措施，如保留层中的注意力头剪枝或更激进的层移除。实际可达压缩比是在特定保留剖面下的实证量。

5. **单一架构**。CIT 方法原则与架构无关，但目前仅与 Qwen 系模型集成。Llama、Mistral、Gemma 等架构验证对泛化性至关重要。

6. **标定数据规模**。当前每能力 10 条样本（初步测量）或设计目标 15 条（完整梯度 CIT），提供了紧凑诊断探针，但可能遗漏稀有但关键的能力输入。样本量在各语言类别间不均衡（zh/en:15 条，ja:9 条，fr/de/ru/es:4-5 条，ko:3 条）需要标准化以降低低资源语言的 CIT 估计噪声。

7. **DCR 表达力**。当前 DCR 使用从嵌入空间的单低秩投影，限制为全局路由决策。多头路由或层级门控等更具表达力的架构可能以额外参数为代价实现更精细控制。

8. **轴选择依据**。19 轴分解参考了已有评估基准 [56,57] 和部署场景 [25,26,27]，但具体轴选择（尤其在学科轴内，由 MMLU 的 57 个学科归并至 6 个）仍是设计决策。其他分解方式（如按语言复杂度、领域或格式）可能捕获当前轴未体现的正交变差。

9. **对比 CIT 未实证评估**。对比 CIT 作为因子化局限的潜在缓解方案被引入（第 3.2 节），但其降低跨轴相关性的有效性尚未实证验证。

### 5.4 总结

PARSE 框架以"该保留什么、该替换什么"的精准思维，重新定义了模型压缩的目标函数——从全局稀疏优化转变为三维能力保持问题。CIT 为量化层级能力贡献提供了原则性机制，DCR 使单一压缩模型可在不使用权重切换的情况下服务多个保留剖面。本文给出了 DCR 训练算法的完整规格（含低秩参数分解和负样本路由策略），明确了因子化假设与局限，透明讨论了压缩比构成和架构约束。完整的四阶段管线设计已作为开源软件实现；初步激活基 CIT 测量已确认深度集中假说（$\bar{r}=0.9945$），12 组保留剖面上的完整实证验证是下一步工作的直接目标。

核心方法论贡献在于认识到模型压缩不必是全局优化问题。通过指定保留*哪些*能力——中文语法、英文数学、函数调用——并仅保留承载这些能力的层，PARSE 框架在高效模型部署中开辟了新的控制轴。

---

## 参考文献

[1] Y. Lyu, C. Wang, H. Zheng, et al. "AgenticQwen: Training small agentic language models with dual data flywheels for industrial-scale tool use." *arXiv:2604.21590*, 2026.

[2] J. Gong. "MiniMind-O technical report: An open small-scale speech-native omni model." *arXiv:2605.03937*, 2026.

[3] X. Ma, G. Fang, and X. Wang. "LLM-Pruner: On the structural pruning of large language models." *arXiv:2305.13058*, 2023.

[4] E. Frantar and D. Alistarh. "SparseGPT: Massive language models can be accurately pruned in one-shot." *arXiv:2301.06126*, 2023.

[5] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter. "A simple and effective pruning approach for large language models." *arXiv:2306.11695*, 2023.

[6] Y. Yang et al. "LaCo: Large language model pruning via layer collapse." *arXiv:2406.04105*, 2024.

[7] X. Men et al. "ShortGPT: Layers in large language models are more redundant than you expect." *arXiv:2403.03853*, 2024.

[8] A. Fan, E. Grave, and A. Joulin. "Reducing transformer depth on demand with structured dropout." *arXiv:1909.11556*, 2020.

[9] J. Xin, R. Tang, J. Lee, Y. Yu, and J. Lin. "DeeBERT: Dynamic early exiting for accelerating BERT inference." *arXiv:2004.12993*, 2020.

[10] W. Liu et al. "FastBERT: a self-distilling BERT with adaptive inference time." *arXiv:2004.02178*, 2020.

[11] E. Kurtic et al. "The optimal BERT surgeon: Scalable and accurate second-order pruning for large language models." *arXiv:2203.07259*, 2022.

[12] V. Sanh, T. Wolf, and A. M. Rush. "Movement pruning: Adaptive sparsity by fine-tuning." *arXiv:2005.07683*, 2020.

[13] C. Xu et al. "BERT-of-Theseus: Compressing BERT by progressive module replacing." *arXiv:2002.02925*, 2020.

[14] P. Zhang, G. Zeng, T. Wang, and W. Lu. "TinyLlama: An open-source small language model." *arXiv:2401.04088*, 2024.

[15] H. Jiang et al. "MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention." *arXiv:2407.01614*, 2024.

[16] M. Xia, T. Gao, Z. Zeng, and D. Chen. "Sheared LLaMA: Accelerating language model pre-training via structured pruning." *arXiv:2310.06699*, 2024.

[17] NVIDIA. "Data flywheel: What it is and how it works." 2024. https://www.nvidia.com/en-us/glossary/data-flywheel/ (Accessed: 2026-05-27)

[18] H. Luo, Q. Sun, C. Xu et al. "Arena learning: Build data flywheel for LLMs post-training via simulated chatbot arena." *arXiv:2407.10627*, 2024.

[19] Z. Wang, J. Li, Y. Hong et al. "Bootstrapping language-guided navigation learning with self-refining data flywheel." *arXiv:2412.08467*, 2024.

[20] X. Guo et al. "IFDECORATOR: Wrapping instruction following reinforcement learning with verifiable rewards." *arXiv:2508.04632*, 2025.

[21] H. Wang et al. "UI-TARS-2 technical report: Advancing GUI agent with multi-turn reinforcement learning." *arXiv:2509.02544*, 2025.

[22] S. Wang et al. "GAIA: A data flywheel system for training GUI test-time scaling critic models." *arXiv:2601.18197*, 2026.

[23] Y. Lyu, C. Wang, L. Shen et al. "Mock worlds, real skills: Building small agentic language models with synthetic tasks." *arXiv:2601.22511*, 2026.

[24] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez. "Gorilla: Large language model connected with massive APIs." *arXiv:2305.15334*, 2023.

[25] J. Zhang et al. "xLAM: A family of large action models to empower AI agent systems." *arXiv:2409.03215*, 2024.

[26] L. E. Erdogan et al. "TinyAgent: Function calling at the edge." *arXiv:2409.00608*, 2024.

[27] Z. Wang et al. "ToolFlow: Boosting LLM tool-calling through natural and coherent dialogue synthesis." *arXiv:2410.18447*, 2024.

[28] R. Sharma and M. Mehta. "Small language models for agentic systems: A survey." *arXiv:2510.03847*, 2025.

[29] M. A. Haque et al. "TinyLLM: Evaluation and optimization of small language models for agentic tasks on edge devices." *arXiv:2511.22138*, 2025.

[30] Y. Fu, R. Anantha, and J. Cheng. "CAMPHOR: Collaborative agents for multi-input planning and high-order reasoning on device." *arXiv:2410.09407*, 2024.

[31] D. Paprunia, V. Kharidia, and P. Doshi. "Advancing SLM tool-use capability using reinforcement learning." *arXiv:2509.04518*, 2025.

[32] K. Han, Y. Zhou, M. Gao et al. "EBPO: Empirical Bayes shrinkage for stabilizing group-relative policy optimization." *arXiv:2602.05165*, 2026.

[33] S. Liu et al. "STAPO: Stabilizing reinforcement learning for LLMs by silencing rare spurious tokens." *arXiv:2602.15620*, 2026.

[34] M. Tian, Y. Xie, and C. Wei. "How off-policy can GRPO be? Mu-GRPO for efficient LLM reinforcement learning." *arXiv:2605.17570*, 2026.

[35] L. He et al. "Resolving action bottleneck: Agentic reinforcement learning informed by token-level energy." *arXiv:2605.14558*, 2026.

[36] H. Li et al. "Agentic reinforcement learning empowers next-generation chemical language models." *arXiv:2601.17687*, 2026.

[37] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. "Locating and editing factual associations in GPT." *arXiv:2202.05262*, 2022.

[38] K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau. "Mass-editing memory in a transformer." *arXiv:2210.07229*, 2023.

[39] E. Mitchell et al. "Model editing networks with gradient decomposition." *arXiv:2110.11309*, 2022.

[40] E. Mitchell et al. "Memory-based model editing at scale." *arXiv:2203.03466*, 2022.

[41] S. Wang et al. "Knowledge editing for large language models: A survey." *arXiv:2401.01286*, 2024.

[42] L. Bourtoule et al. "Machine unlearning." *arXiv:1912.03817*, 2021.

[43] Y. Yao et al. "Machine unlearning: A survey." *ACM Computing Surveys*, 2024. DOI: 10.1145/3603620

[44] B. Liu et al. "Knowledge unlearning for LLMs." *arXiv:2402.01754*, 2024.

[45] R. M. French. "Catastrophic forgetting in connectionist networks." *Trends in Cognitive Sciences*, vol. 3, no. 4, pp. 128–135, 1999. DOI: 10.1016/S1364-6613(99)01294-2

[46] A. Sekhari et al. "Descent-to-delete: Gradient-based methods for machine unlearning." *arXiv:2110.05679*, 2021.

[47] A. Golatkar et al. "Fast machine unlearning without retraining." *arXiv:2009.11373*, 2020.

[48] Anonymous. "Task-specific compression for large language models." *arXiv:2306.05685*, 2023. (Under review)

[49] Anonymous. "Compact language models via priming and pruning." *arXiv:2406.09246*, 2024. (Under review)

[50] J.S. McCarley, R. Chakravarti, and A. Sil. "Structured pruning of BERT-based question answering models." *arXiv:1910.09755*, 2019.

[51] N. Ding et al. "Parameter-efficient fine-tuning for large language models: A comprehensive survey." *arXiv:2303.15647*, 2023.

[52] Y. Li et al. "Inference-time intervention: Eliciting truthful answers from a language model." *arXiv:2306.03341*, 2023.

[53] Anonymous. "Regularizing towards well-calibrated large language models." *arXiv:2405.18654*, 2024. (Under review)

[54] H. Ndubuaku, J. Mroz, K. Mosoyan, et al. "Needle: Simple attention networks for function calling." *GitHub: cactus-compute/needle*, 2026.

[55] Z. Shao, P. Wang, Q. Zhu, et al. "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models." *arXiv:2402.03300*, 2024.

[56] A. Conneau et al. "XNLI: Evaluating cross-lingual sentence representations." *Proc. EMNLP*, 2018.

[57] D. Hendrycks et al. "Measuring massive multitask language understanding." *Proc. ICLR*, 2021.

[58] K. Cobbe et al. "Training verifiers to solve math word problems." *arXiv:2110.14168*, 2021.

[59] Berkeley Function-Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html

[60] M. Chen et al. "Evaluating large language models trained on code." *arXiv:2107.03374*, 2021.

[61] K. Papineni et al. "BLEU: A method for automatic evaluation of machine translation." *Proc. ACL*, 2002.

[62] I. Loshchilov and F. Hutter. "Decoupled weight decay regularization." *Proc. ICLR*, 2019.

[63] D. P. Kingma and J. Ba. "Adam: A method for stochastic optimization." *Proc. ICLR*, 2015.
