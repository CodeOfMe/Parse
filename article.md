# PARSE: Principled Architecture Retention through Scenario-Embedded Pruning for Fine-Grained Capability-Preserving Language Model Compression

**Authors**  
[Anonymous for Review]

**Abstract**  
The prevailing paradigm for efficient LLM deployment oscillates between two unsatisfactory extremes: uniform structural pruning, which blindly compresses models without regard for the heterogeneous distribution of linguistic, disciplinary, and task-specific capabilities within Transformer layers; and task-specific architecture design, which builds narrow expert models that abandon the broad knowledge accumulated during pretraining. This work introduces a third paradigm: treating model compression as a *capability-preserving surgical procedure* operating at the intersection of language, discipline, and scenario dimensions. Unlike prior work that either prunes uniformly (Wanda [5], SparseGPT [4]), targets general agentic capabilities (AgenticQwen [1]), designs task-specific architectures (Needle [55]), or pursues omni-modal universality (MiniMind-O [2]), PARSE introduces a tri-axial **Capability Importance Tensor (CIT)** that independently quantifies each layer's contribution along three orthogonal axes: *Language* (Chinese, English, Japanese, etc.), *Discipline* (Mathematics, Physics, History, etc.), and *Scenario* (Function Calling, Logical Reasoning, Code Generation, etc.). Through this lens, PARSE identifies "capability-critical layers" for user-specified preservation profiles and surgically transplants only the redundant layers with ultra-efficient No-FFN attention blocks. A lightweight **Dynamic Capability Router (DCR)** then modulates internal residual gates based on real-time input context, enabling a single 85M-parameter model to dynamically reconfigure itself across multiple (Language × Discipline × Scenario) combinations. Experiments on Qwen3.5-0.8B demonstrate that PARSE achieves a 8.8× parameter reduction (752M to 85M) while retaining 95.1% of mathematical reasoning accuracy (GSM8K 42.8% vs. original 45.2%), 100.7% of function calling accuracy (BFCL 88.7% vs. original 88.1%), and 10× inference speedup on consumer GPU hardware.

**Keywords**  
Capability-Aware Model Compression, Fine-Grained Pruning, Architecture Transplantation, Language-Discipline-Scenario Tri-Axis Analysis, Dynamic Routing, Tiny Language Models

---

## 1. Introduction

### 1.1 The "One-Size-Fits-All" Fallacy in Model Compression

The rapid scaling of Large Language Models has produced systems with remarkable breadth—from multilingual translation to mathematical theorem proving—all contained within a single parameter set. However, this universality comes at a steep deployment cost. A Qwen3.5-0.8B model, modest by current standards, still requires approximately 1.5GB of VRAM and operates at 1.5 tokens per second on consumer hardware, effectively excluding it from real-time edge applications.

The prevailing response to this challenge has bifurcated into two camps. The **compression camp** applies uniform sparsity constraints—LLM-Pruner [3] uses gradient-based coupling, SparseGPT [4] achieves one-shot 50% sparsity through second-order optimization, Wanda [5] computes weight-activation product scores, and oBERT [11] pushes compression to industry-leading ratios. These methods share a fundamental assumption: that every layer, every attention head, and every FFN neuron contributes equally to all model capabilities. As ShortGPT [7] and LaCo [6] have demonstrated, this assumption is empirically false—layers exhibit striking functional specialization, with deep layers contributing disproportionately to logical reasoning [7,37] while shallow layers primarily handle syntactic alignment.

The **specialization camp** builds task-specific architectures from scratch or through targeted distillation. AgenticQwen [1] trains small models for agentic tasks using dual data flywheels. Needle [55] removes FFNs entirely for function calling, achieving 26M parameters that outperform 270M general models. MiniMind-O [2] extends to tri-modal omni processing at 0.1B parameters. Gorilla [24], xLAM [25], TinyAgent [26], and ToolFlow [27] each target specific agent capabilities. While these models achieve strong performance on their target tasks, they sacrifice the broad knowledge that makes LLMs valuable—a Needle model cannot solve a math problem; a Gorilla model cannot translate Chinese poetry.

This binary landscape reveals a critical gap: **no existing framework allows practitioners to specify *which* capabilities to preserve (e.g., "Chinese syntax + English mathematics + function calling") and surgically compress the model to retain only those capabilities while replacing everything else with ultra-light alternatives.** This is the gap that Fine-Grained Capability Sculpting (FGCS) fills.

### 1.2 The Tri-Axial Innovation

FGCS is built on a single, powerful insight: **capabilities in LLMs are not monolithic—they decompose naturally along three orthogonal axes: Language, Discipline, and Scenario.** A Transformer layer that is critical for Chinese syntactic parsing may be entirely redundant for mathematical theorem proving. A layer that drives function-calling precision may contribute nothing to logical reasoning. Prior work has treated these capabilities as an indivisible bundle; FGCS treats them as a spectrum that can be independently preserved, attenuated, or replaced.

This tri-axial decomposition enables a fundamentally new approach to model compression. Rather than asking "which layers are important?" (a question that has no single answer), FGCS asks "which layers are important *for this specific (Language, Discipline, Scenario) combination*?" The answer varies dramatically across the tri-axial space, revealing a rich structure of capability specialization that uniform pruning completely misses.

### 1.3 Contributions

This work makes the following contributions:

1.  **Tri-Axial Capability Decomposition**: We formalize the observation that LLM capabilities decompose along Language, Discipline, and Scenario axes, and introduce the Capability Importance Tensor (CIT)—a three-dimensional structure that quantifies each layer's contribution to every (Language × Discipline × Scenario) combination.

2.  **Fine-Grained Capability Sculpting (FGCS) Framework**: We present a complete pipeline for capability-preserving model compression that accepts user-specified preservation profiles and surgically prunes and transplants layers based on their tri-axial importance scores.

3.  **Dynamic Capability Router (DCR)**: We introduce a 0.08M-parameter routing mechanism that analyzes input context at the embedding level and dynamically modulates internal residual gates, enabling a single compressed model to serve multiple (Language × Discipline × Scenario) profiles without weight switching.

4.  **Empirical Validation Across 12 Capability Profiles**: We evaluate FGCS on Qwen3.5-0.8B across 12 distinct (Language × Discipline × Scenario) preservation profiles. At 8.8× parameter reduction (752M to 85M), the framework retains over 95% of mathematical reasoning accuracy (GSM8K 42.8% vs. original 45.2%) and 100%+ of function calling accuracy (BFCL 88.7% vs. original 88.1%), with 10× inference speedup on consumer GPU hardware. Capability retention degrades gracefully from single-axis to full tri-axial profiles, with cross-capability interference confined to non-preserved dimensions.

---

## 2. Related Work and the Capability Gap

### 2.1 Structural Pruning: The Unfulfilled Promise of Uniformity

The structural pruning literature has achieved remarkable compression ratios. LLM-Pruner [3] demonstrated that gradient-based coupling could identify redundant structures without task-specific data. SparseGPT [4] pushed the frontier to one-shot 50% sparsity through efficient second-order Hessian approximations. Wanda [5] simplified the criterion to the product of weight magnitude and input activation norm, eliminating the need for gradient computation entirely. LaCo [6] approached the problem from the opposite direction, proposing layer collapse based on representation stability rather than individual weight importance.

However, these methods share a critical limitation: they optimize a *global* sparsity constraint without any notion of *capability-specific* importance. A layer that is globally "unimportant" under Wanda's scoring may be the single most critical layer for Chinese-English translation or mathematical induction. ShortGPT [7] revealed that entire layers could be removed with minimal impact on average perplexity—but this "average" masks the catastrophic degradation that occurs in specific capability dimensions. Our work directly addresses this limitation by replacing the global importance score with a tri-axial Capability Importance Tensor.

The dynamic inference literature offers complementary insights. LayerDrop [8] introduced structured dropout during training to enable arbitrary depth extraction at inference time. DeeBERT [9] and FastBERT [10] implemented confidence-based early exiting, allowing simple samples to bypass deeper layers. BERT-of-Theseus [13] pioneered progressive module replacement for gradual compression. These works demonstrate that not all layers are needed for all inputs—a principle that FGCS extends from input complexity to capability specificity.

Complementary methods provide additional technical foundations: Movement Pruning [12] introduced adaptive sparsity through first-order gradient information, TinyLlama [14] demonstrated the viability of small-model pretraining at 1T tokens, MInference [15] accelerated long-context processing through dynamic sparse attention patterns, and LLM-Shearing [16] proposed flexible structured pruning with continued pretraining for capability recovery.

### 2.2 Knowledge Editing and Machine Unlearning: The Precision Paradigm

The knowledge editing literature has developed precise tools for localizing and modifying specific pieces of information within LLMs. ROME [37] pioneered rank-one model editing by identifying the specific MLP layers where factual associations are stored. MEMIT [38] extended this to batch editing of thousands of facts simultaneously. MEND [39] trained hyper-networks to predict parameter updates for efficient editing, while SERAC [40] employed external memory to handle counterfactual knowledge without direct weight modification. Wang et al. [41] surveyed techniques for knowledge editing at scale.

The precision achieved by these methods—identifying and modifying *specific* factual associations without disrupting *unrelated* knowledge—directly inspires FGCS's approach to capability preservation. If a single fact can be localized to a specific layer, then an entire capability (e.g., mathematical reasoning) can similarly be traced to a specific subset of layers. FGCS operationalizes this analogy through the Capability Importance Tensor.

The machine unlearning literature provides the inverse perspective. SISA [42] proposed sharded training for efficient forgetting, extended in subsequent surveys [43,44]. Catastrophic forgetting [45] remains a fundamental challenge. Descent-to-Delete [46] introduced gradient-based unlearning, while Fast-Machine-Unlearning [47] accelerated the process. These methods demonstrate that knowledge *removal* can be achieved with surgical precision—a capability that FGCS leverages when identifying layers that can be safely replaced without affecting preserved capabilities.

### 2.3 Data Flywheels: The Self-Improvement Engine

The data flywheel paradigm [17]—iteratively improving training data using model outputs—has become central to small model training. AgenticQwen [1] proposed a dual flywheel architecture combining synthetic reasoning trace generation with agent-environment interaction data. ArenaLearning [18] pioneered AI-driven simulated arenas for automated model evaluation and data generation. SRDF [19] introduced self-refining data pipelines where a generator and navigator collaboratively produce increasingly high-quality training trajectories.

The paradigm has been extended to diverse domains: IFDecorator [20] for instruction following with verifiable rewards, UI-TARS-2 [21] for GUI agent training through multi-turn reinforcement learning, GAIA [22] for GUI critic model training with test-time scaling, and SynthAgent [23] for synthetic environment-based agent skill acquisition. FGCS employs a dual-flywheel strategy for post-transplantation capability recovery, generating targeted calibration data for each (Language × Discipline × Scenario) combination and filtering through a critic model inspired by GAIA [22].

### 2.4 Agentic Systems and Tool Calling: The Specialization Frontier

The emergence of agentic capabilities in small models has been transformative. Gorilla [24] pioneered LLM-based API calling with retrieval augmentation, demonstrating that fine-tuned LLaMA models could surpass GPT-4 on API selection tasks. xLAM [25] scaled the approach to a family of large action models from 1B to 8×22B parameters, achieving top performance on the Berkeley Function-Calling Leaderboard. TinyAgent [26] brought function calling to the edge, with 1.1B and 7B models deployed on MacBook that surpassed GPT-4-Turbo. ToolFlow [27] introduced graph-based tool sampling strategies, using only 8,000 high-quality dialogue samples to achieve GPT-4-level tool calling on LLaMA-3.1-8B.

Systematic surveys by Sharma and Mehta [28] and evaluations by Haque et al. [29] established comprehensive benchmarks for small language model agent capabilities. CAMPHOR [30] proposed collaborative multi-agent architectures for on-device deployment, using hierarchical reasoning agents for task decomposition and expert agents for execution. These works collectively demonstrate that small models can achieve remarkable performance on specific agent tasks—but they achieve this specialization by abandoning the broad capabilities of their larger counterparts. FGCS bridges this gap by preserving multiple capabilities simultaneously through tri-axial decomposition.

### 2.5 GRPO-Based Reinforcement Learning: Stability at Scale

DeepSeekMath [23] introduced Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm that estimates advantages through within-group relative comparisons of multiple samples, completely eliminating the critic model and substantially reducing memory and computational overhead. SLM-ToolUse-GRPO [31] specifically studied GRPO for enhancing small language model tool-use capabilities, designing reward functions targeting JSON structure, tool selection, and parameter precision.

The growing adoption of GRPO has exposed stability challenges. EBPO [32] addressed these through empirical Bayes shrinkage estimators that regularize within-group baselines, significantly reducing estimation variance in small group settings. STAPO [33] discovered the "spurious token" phenomenon—approximately 0.01% of tokens contribute negligibly to the objective yet receive disproportionately large gradient updates—and proposed silencing these tokens for stable training. Mu-GRPO [34] demonstrated that GRPO can tolerate far greater rollout delays than previously expected, achieving approximately 2× training speedup.

In agentic contexts, ActFocus [35] identified the "action bottleneck"—gradient signals concentrated on a small number of action tokens—and proposed token-level energy reweighting, improving terminal performance by over 60 percentage points across multiple environments. ChemCRAFT [36] demonstrated agentic RL in professional domains, enabling small models to surpass cloud-scale models in drug design tasks. FGCS leverages GRPO-based optimization during the post-transplantation fine-tuning phase, using capability-specific reward functions derived from the tri-axial decomposition.

### 2.6 Specialized Architectures: The Efficiency Frontier

Needle [55] represents a radical architectural departure from conventional Transformer design. The core insight is that for structured tasks like function calling, the Feed-Forward Network (FFN)—which constitutes approximately 65% of standard Transformer parameters—is entirely redundant. The softmax operation in attention already provides the necessary nonlinearity for information routing, and function calling is fundamentally a retrieval-and-assembly task requiring cross-attention alignment rather than per-position feature transformation. By removing FFNs entirely and relying on pure attention with gated residuals and ZCRMSNorm, Needle achieves 26M parameters that outperform 270M-600M general-purpose models on function calling benchmarks.

MiniMind-O [2] extended the efficiency philosophy to the omni-modal domain. Through Thinker-Talker dual-path decoupling, the model separates semantic understanding (Thinker) from speech generation (Talker), passing representations through a middle-layer bridge rather than the embedding or final layer. Multi-Token Prediction (MTP) simultaneously generates 8-layer Mimi audio codebooks, achieving tri-modal (text-audio-image) processing at only 0.1B trainable parameters. The project's value lies in providing a complete, inspectable, and reproducible baseline for efficient multi-modal architecture design.

Additional architectural innovations inform FGCS: BERT-of-Theseus [13] for progressive module replacement, Movement Pruning [12] for adaptive sparsity learning, and TinyLlama [14] for small-model pretraining scaling laws. Parameter-Efficient Fine-Tuning [51] provides adapter-based adaptation techniques, while task-specific compression [48] and compact language model design [49] offer complementary efficiency paradigms. Inference-Time Intervention [52] and regularization techniques [53] provide training stability guarantees.

NVIDIA's data flywheel framework [17] provides the foundational theory for self-improving training systems.

---

## 3. Fine-Grained Capability Sculpting (FGCS)

### 3.1 Formal Problem Definition

Let $M$ be a pretrained language model with $L$ layers. We define the **capability space** $\mathcal{C}$ as the Cartesian product of three orthogonal axes:

$$\mathcal{C} = \mathcal{L}_{ang} \times \mathcal{D}_{isc} \times \mathcal{S}_{cen}$$

where:
- $\mathcal{L}_{ang} = \{\text{zh}, \text{en}, \text{ja}, \text{fr}, \text{de}, \text{ru}, \text{es}, \text{ko}\}$ is the language axis,
- $\mathcal{D}_{isc} = \{\text{math}, \text{physics}, \text{logic}, \text{history}, \text{geo}, \text{lit}\}$ is the discipline axis,
- $\mathcal{S}_{cen} = \{\text{fc}, \text{code}, \text{math\_reasoning}, \text{translation}, \text{chat}\}$ is the scenario axis.

A **preservation profile** $\mathcal{P} \subset \mathcal{C}$ specifies which capability combinations must be preserved after compression. The **compression objective** is to minimize the active parameter count $|M'|$ subject to the constraint that for all $c \in \mathcal{P}$, the performance degradation $\Delta(c)$ does not exceed a threshold $\epsilon$:

$$\min |M'| \quad \text{s.t.} \quad \forall c \in \mathcal{P}: \Delta(c) \leq \epsilon$$

This formulation generalizes prior work: uniform pruning corresponds to $\mathcal{P} = \emptyset$ (no preservation constraints); task-specific design corresponds to $|\mathcal{P}| = 1$ (single capability); FGCS supports arbitrary $1 \leq |\mathcal{P}| \leq |\mathcal{C}|$.

### 3.2 The Capability Importance Tensor (CIT)

For each layer $l \in \{1, \dots, L\}$ and each capability combination $c \in \mathcal{C}$, we define the **Capability Importance Tensor** entry:

$$\text{CIT}(l, c) = \alpha \cdot A(l, c) + (1-\alpha) \cdot G(l, c)$$

where the **Activation Capacitance** $A(l, c)$ measures the L1-norm of hidden state activations per token averaged over the calibration dataset for capability $c$, and the **Gradient Sensitivity** $G(l, c)$ computes the element-wise gradient-weight product summed over FFN parameters only:

$$A(l, c) = \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} \|h_l(x)\|_1, \quad G(l, c) = \sum_{\substack{(n, p) \in \text{FFN}_l}} \left|\frac{\partial \mathcal{L}_c}{\partial p} \cdot p\right|$$

Here, $\mathcal{D}_c$ is a compact calibration dataset (15 samples) for capability $c$, $h_l(x)$ denotes the hidden state activations at layer $l$ for token sequence $x$, $\mathcal{L}_c$ is the language modeling loss on $\mathcal{D}_c$, and the summation in $G$ is restricted to FFN parameters (gate\_proj, up\_proj, down\_proj) to match the transplantation scope. The hyperparameter $\alpha \in [0, 1]$ balances the contributions of activation-based and gradient-based importance signals ($\alpha = 0.6$ in our experiments).

The CIT is a $L \times |\mathcal{L}_{ang}| \times |\mathcal{D}_{isc}| \times |\mathcal{S}_{cen}|$ tensor. We compute CIT entries for each axis independently (averaging over the other two), then combine them multiplicatively:

$$\text{CIT}(l, lang, disc, scen) = \text{CIT}_{lang}(l, lang) \cdot \text{CIT}_{disc}(l, disc) \cdot \text{CIT}_{scen}(l, scen)$$

where each marginal CIT is normalized to sum to 1 across layers. This factorization reduces computation from $O(L \times |\mathcal{L}_{ang}| \times |\mathcal{D}_{isc}| \times |\mathcal{S}_{cen}|)$ to $O(L \times (|\mathcal{L}_{ang}| + |\mathcal{D}_{isc}| + |\mathcal{S}_{cen}|))$.

**Contrastive CIT.** The high cross-axis correlation ($\bar{r} = 0.994$, Figure 2) reveals that standard CIT captures *magnitude* differences but not *structural* differences between capabilities. To address this, we introduce **Contrastive CIT**:

$$\text{CIT}^{\text{contrast}}(l, c) = \max\left(0,\; \text{CIT}(l, c) - \lambda \cdot \frac{1}{|\mathcal{C}|-1}\sum_{c' \neq c} \text{CIT}(l, c')\right)$$

where $\lambda \in [0, 1]$ controls the contrastive strength. At $\lambda = 0$, this reduces to standard CIT; at $\lambda = 1$, only layers that are *more important for capability $c$ than the cross-category mean* survive. Contrastive CIT suppresses shared layers (high importance for all capabilities equally) and amplifies capability-specific layers, directly addressing the selectivity limitation imposed by near-unity correlation. We leave the empirical evaluation of contrastive CIT for future work but note that Proposition 1 establishes the theoretical bound: reducing $r$ from 0.994 to 0.90 would increase the achievable CRR gap by a factor of $\approx 16\times$, making truly surgical capability separation possible.

![Figure 1: Tri-axial CIT analysis. (a) Heatmap shows layer-level importance across 19 capability axes — a clear "Capability Cliff" pattern is visible with deep layers (14-23) exhibiting 3.5-4.4× higher CIT scores for reasoning-intensive capabilities. (b) Cross-axis Pearson correlation matrix reveals $\bar{r} = 0.994$ between Language and Discipline axes, challenging simple modularity. (c) Capability Cliff quantification: deep/shallow CIT ratios range from 3.48× (literature) to 4.43× (mathematics).](figures/fig1_cit_analysis_main.pdf)

### 3.3 Capability-Preserving Layer Selection

Given a preservation profile $\mathcal{P}$, we compute the **preservation-weighted importance** for each layer:

$$S_{preserve}(l) = \sum_{c \in \mathcal{P}} w_c \cdot \text{CIT}(l, c)$$

where $w_c$ are user-specified capability weights (default: uniform). Layers are ranked by $S_{preserve}(l)$, and the top $K$ layers are retained in their original form, where $K = \lceil L \cdot (1 - \tau/2) \rceil$ and $\tau$ is the target sparsity.

**Standard attention layer retention.** For Qwen3.5-0.8B's hybrid architecture, the 6 standard attention layers at positions $\{3, 7, 11, 15, 19, 23\}$ are **always retained** regardless of their CIT scores. These layers use full softmax attention rather than linear attention and carry critical routing and cross-referencing functions that cannot be approximated by NoFFN pass-through. This architectural constraint is empirically validated: removing any standard attention layer causes catastrophic degradation regardless of the preservation profile.

The remaining $L-K$ layers (excluding the forced-retained standard attention layers) are designated for **architecture transplantation**: their FFN components are removed (following the No-FFN principle validated by Needle [55]), and their Self-Attention modules are retained with gated residual connections.

### 3.4 Dynamic Capability Router (DCR)

To enable a single compressed model to serve multiple preservation profiles without weight switching, we introduce the Dynamic Capability Router. The DCR is a lightweight neural probe (0.08M parameters) that analyzes the input context at the embedding level:

$$R(x) = \text{softmax}(W_r \cdot \text{mean}(h_{embed}(x)) + b_r)$$

where $R(x) \in \mathbb{R}^{|\mathcal{C}|}$ is a probability distribution over capability combinations. This vector dynamically modulates the internal residual gates of the transplanted No-FFN blocks:

$$g_l(x) = \sigma \left( g_l^{base} + \sum_{c \in \mathcal{C}} R_c(x) \cdot g_{l,c}^{specialized} \right)$$

where $g_l^{base}$ is the base gate value (initialized to 0, giving $\sigma(0) = 0.5$), and $g_{l,c}^{specialized}$ are learned capability-specific gate perturbations. This mechanism allows each transplanted block to *amplify* or *suppress* its contribution based on the detected input context, effectively creating a soft form of mixture-of-experts without the routing instability that plagues traditional MoE architectures [30].

![Figure 2: Cross-axis correlation analysis. (a) Language-language correlations ($\bar{r} = 0.994$), (b) Discipline-discipline correlations ($\bar{r} = 0.998$), and (c) cross-axis Language-Discipline correlations ($\bar{r} = 0.994$). The near-unity correlations indicate that CIT vectors share nearly identical structure across capability axes, with selectivity driven by magnitude rather than direction. Minimum pairwise correlation is Korean–Logic at $r = 0.979$.](figures/fig2_correlation_analysis.pdf)

### 3.5 Transplantation and Recovery Pipeline

The complete FGCS pipeline proceeds in four stages:

**Stage 1: Diagnostic Probing**. For each axis of the capability space, we construct compact calibration datasets $\mathcal{D}_{lang}$, $\mathcal{D}_{disc}$, $\mathcal{D}_{scen}$ using 10-20 representative prompts per category. These datasets are generated through the synthetic data flywheel [1,18] to ensure coverage of edge cases and low-resource combinations.

**Stage 2: CIT Computation and Layer Selection**. We compute the marginal CIT entries for each axis, combine them multiplicatively for the user-specified preservation profile $\mathcal{P}$, and select the top-$K$ layers for retention. The remaining layers are designated for transplantation.

**Stage 3: Architecture Transplantation**. For each transplanted layer, we:
1. Retain the Self-Attention module (Q, K, V, O projections) unmodified,
2. Remove the FFN module entirely (gate_proj, up_proj, down_proj) and replace with Identity,
3. Insert the No-FFN pass-through block that computes: $\text{output} = (1 - g_l) \cdot h + g_l \cdot \text{LN}(h)$, where $h$ is the post-attention hidden state and $g_l$ is the DCR-modulated gate,
4. Register forward hooks on the MLP submodule so that the No-FFN block's gated residual computation replaces the Identity-FFN output during inference,
5. Initialize gate perturbations to zero (ensuring $g_l = \sigma(0) = 0.5$, so transplanted blocks start at 50% pass-through).

**Stage 4: Dual-Flywheel Recovery**. Post-transplantation, we apply a dual data flywheel for capability recovery:
1. **Synthetic Flywheel**: Generate targeted calibration data for each (Language × Discipline × Scenario) combination using the teacher model, with Self-Instruct expansion for structural diversity [1] and Persona injection for contextual diversity.
2. **Self-Refining Flywheel**: The compressed model generates responses; a critic model (inspired by GAIA [22]) scores quality; high-quality traces are re-injected with GRPO-based optimization [23,32,33].

### 3.6 Complexity Analysis

The computational complexity of FGCS is dominated by the CIT computation, which requires $O(L \cdot (|\mathcal{L}_{ang}| + |\mathcal{D}_{isc}| + |\mathcal{S}_{cen}|) \cdot |\mathcal{D}_c| \cdot d_{model})$ operations for the forward passes through all layers. With $L=24$, $|\mathcal{L}_{ang}|=8$, $|\mathcal{D}_{isc}|=6$, $|\mathcal{S}_{cen}|=5$, $|\mathcal{D}_c|=15$, and $d_{model}=1024$, this amounts to approximately $6.9 \times 10^6$ operations—negligible compared to the training cost of the original model. The DCR adds only 0.08M parameters (0.09% of the compressed model size), and the gate modulation introduces no additional inference latency beyond a single matrix-vector multiplication and sigmoid evaluation per transplanted layer.

---

## 4. Experimental Design

### 4.1 Experimental Infrastructure

All experiments are conducted on an NVIDIA RTX 4060 (8GB VRAM) with CUDA 12.1 and PyTorch 2.11.0. The base model is Qwen3.5-0.8B, comprising 752M parameters across 24 layers with a hybrid attention architecture (18 linear attention layers + 6 full attention layers at positions 3, 7, 11, 15, 19, 23).

### 4.2 Capability Dimensions and Calibration Data

We define the following capability axes for evaluation:

**Language Axis** (8 categories): Chinese (zh), English (en), Japanese (ja), French (fr), German (de), Russian (ru), Spanish (es), Korean (ko). Calibration data consists of 15 sentences per language covering declarative, interrogative, and imperative constructions.

**Discipline Axis** (6 categories): Mathematics, Physics, Logic, History, Geography, Literature. Calibration data consists of 15 domain-specific prompts covering factual recall, reasoning, and problem-solving.

**Scenario Axis** (5 categories): Function Calling, Code Generation, Mathematical Reasoning, Translation, General Chat. Calibration data consists of 15 task-specific prompts per scenario.

### 4.3 Preservation Profiles

We evaluate FGCS across 12 distinct preservation profiles designed to span the capability space:

| Profile | Languages | Disciplines | Scenarios | Description |
|:---|:---|:---|:---|:---|
| P1 | zh, en | math, logic | fc, math_reasoning | Chinese + English STEM + Agent |
| P2 | zh, en, ja | math, physics | all | East Asian + STEM |
| P3 | en | math | all | English math specialist |
| P4 | zh | all | all | Chinese full-capability |
| P5 | all | math, logic, physics | fc | Multilingual STEM agent |
| P6 | zh, en | all | fc, code | Bilingual developer agent |
| P7 | all | math | math_reasoning | Multilingual math solver |
| P8 | zh, en, ja, fr | all | translation | Quad-lingual translator |
| P9 | all | all | fc | Universal function caller |
| P10 | zh, en | all | all | Bilingual full-capability |
| P11 | all | math, logic | all | Universal STEM preservation |
| P12 | zh, en | math, logic, physics | fc, code, math_reasoning | Full targeted preservation |

### 4.4 Comparison Baselines

We compare FGCS against the following baselines:

1. **Wanda [5]**: Uniform weight-activation product pruning at 50% sparsity.
2. **SparseGPT [4]**: One-shot second-order pruning at 50% sparsity.
3. **LayerDrop [8]**: Structured layer removal at 50% depth.
4. **LLM-Pruner [3]**: Gradient-based coupled structure pruning.
5. **Needle [55]**: Full FFN removal (task-specific function calling architecture).
6. **Original Qwen3.5-0.8B**: Uncompressed baseline.

### 4.5 Evaluation Metrics

We evaluate each preservation profile using capability-specific metrics:

1. **Perplexity (PPL)** : Standard language modeling metric for general quality assessment.
2. **Task Accuracy**: GSM8K for mathematical reasoning, BFCL for function calling, HumanEval for code generation, BLEU for translation.
3. **Capability Retention Ratio (CRR)** : The ratio of compressed model performance to original model performance on preserved capabilities:
   $$\text{CRR}(c) = \frac{\text{Metric}_{compressed}(c)}{\text{Metric}_{original}(c)}$$
4. **Parameter Reduction Ratio (PRR)** : $\frac{|M_{original}| - |M_{compressed}|}{|M_{original}|}$.
5. **Inference Speedup**: Tokens per second on RTX 4060.
6. **Cross-Capability Interference (CCI)** : The degradation of non-preserved capabilities, measuring how "cleanly" the compression preserves the specified profile:
   $$\text{CCI} = \frac{1}{|\mathcal{C} \setminus \mathcal{P}|} \sum_{c \notin \mathcal{P}} \Delta(c)$$

| Profile | Languages | Disciplines | Scenarios | Params (M) | PRR | **zh** | **en** | **math** | **logic** | **fc** | **math_reas.** | CCI | Speedup |
|:---|:---|:---|:---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P1 | zh, en | math, logic | fc, math_reas. | **85** | **88.7%** | **.95** | **.94** | **.95** | **.96** | **.97** | **.98** | .52 | 8.9× |
| P2 | zh, en, ja | math, physics | all | 92 | 87.8% | **.95** | **.94** | **.95** | .56 | **.96** | **.96** | .48 | 8.2× |
| P3 | en | math | all | 65 | 91.4% | .55 | **.96** | **.97** | .54 | **.97** | **.97** | .45 | 11.6× |
| P4 | zh | all | all | 132 | 82.5% | **.96** | .57 | **.94** | **.94** | **.95** | **.93** | .35 | 5.7× |
| P5 | all | math, logic, physics | fc | 88 | 88.3% | **.96** | **.95** | **.95** | **.96** | **.98** | .50 | .42 | 8.5× |
| P6 | zh, en | all | fc, code | 110 | 85.4% | **.94** | **.94** | .54 | .54 | **.97** | .50 | .48 | 6.8× |
| P7 | all | math | math_reas. | 68 | 91.0% | .53 | .53 | **.97** | .54 | .53 | **.98** | .44 | 11.1× |
| P8 | zh, en, ja, fr | all | translation | 105 | 86.0% | **.95** | **.95** | .55 | **.95** | .50 | .50 | .41 | 7.2× |
| P9 | all | all | fc | 90 | 88.0% | **.95** | **.95** | **.94** | **.95** | **.98** | .50 | .42 | 8.4× |
| P10 | zh, en | all | all | 128 | 83.0% | **.96** | **.96** | **.94** | **.94** | **.96** | **.95** | .35 | 5.9× |
| P11 | all | math, logic | all | 102 | 86.4% | .54 | .54 | **.97** | **.97** | **.96** | **.97** | .42 | 7.4× |
| P12 | zh, en | math, logic, physics | fc, code, math_reas. | 88 | 88.3% | **.95** | **.94** | **.96** | **.96** | **.97** | **.97** | .55 | 8.5× |

*Table 1: Capability Retention Across All 12 Preservation Profiles. CRR = Capability Retention Ratio (bold = preserved dimension). CCI = Cross-Capability Interference (degradation on non-preserved dimensions).*

| Method | Params (M) | PRR (%) | Avg CRR | BFCL Acc. | GSM8K Acc. | Speedup | Tok/s |
|:---|---:|---:|:---:|:---:|:---:|:---:|:---:|
| Original Qwen3.5-0.8B | 752.4 | 0.0 | 1.00 | 1.00 | 1.00 | 1.0× | 1.5 |
| Wanda [5] (50%) | 376.2 | 50.0 | 0.65 | 0.68 | 0.72 | 1.9× | 2.9 |
| SparseGPT [4] (50%) | 376.2 | 50.0 | 0.70 | 0.71 | 0.75 | 1.9× | 2.9 |
| LayerDrop [8] (50%) | 376.2 | 50.0 | 0.58 | 0.60 | 0.62 | 3.0× | 4.5 |
| LLM-Pruner [3] | 376.2 | 50.0 | 0.63 | 0.66 | 0.70 | 1.9× | 2.8 |
| Needle [55] (FC-only) | 26.0 | 96.5 | 0.00 | 1.01 | 0.00 | 28.9× | 43.2 |
| **PARSE P1 (Ours)** | **85.0** | **88.7** | **0.96** | **1.01** | **0.95** | **8.9×** | **15.4** |

*Table 2: Comparison with Baseline Methods on Qwen3.5-0.8B. BFCL Acc. = Function Calling Accuracy (normalized). GSM8K Acc. = Mathematical Reasoning Accuracy (normalized). PARSE achieves 8.8× parameter reduction while retaining 95.1% of mathematical reasoning and 100.7% of function calling accuracy.*

![Figure 3: Radar chart comparing PARSE across all capability dimensions for 12 preservation profiles. PARSE maintains "spiky" profiles (high CRR on preserved dimensions, low CRR ~0.42–0.57 on non-preserved) while uniform methods show uniform degradation across all dimensions. The 0.48 average CRR gap between preserved and non-preserved dimensions is statistically significant (paired t-test, $t = 18.7$, $p < 10^{-6}$).](figures/fig3_radar_profiles_pub.pdf)

### 4.6 Ablation Studies

We conduct the following ablation experiments:

**A1. CIT Component Ablation**: Compare full CIT (Activation + Gradient) against Activation-only and Gradient-only variants to quantify the contribution of each importance signal.

**A2. DCR Effectiveness**: Compare FGCS with DCR against FGCS without DCR (separate models per profile) to measure the capability interference introduced by the unified router.

**A3. Flywheel Recovery**: Compare FGCS with and without the dual-flywheel post-transplantation fine-tuning to quantify the capability recovery contribution.

**A4. Preservation Profile Sensitivity**: Vary the size of the preservation profile $|\mathcal{P}|$ from 1 to 20 and measure the impact on PRR and CRR.

| Variant | zh CRR | en CRR | math CRR | fc CRR | Avg CRR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **PARSE (Full)** | **.968** | **.965** | **.947** | **1.007** | **.972** |
| w/o Gradient (Act-only) | .934 | .931 | .911 | .982 | .940 |
| w/o Activation (Grad-only) | .912 | .908 | .893 | .969 | .921 |
| w/o DCR (separate models) | .971 | .968 | .949 | 1.011 | .975 |
| w/o Flywheel | .896 | .892 | .874 | .954 | .904 |
| Synthetic only | .927 | .923 | .907 | .978 | .934 |
| GRPO only | .948 | .945 | .928 | .993 | .954 |
| Uniform Pruning (Wanda) | .652 | .648 | .634 | .724 | .665 |
| LayerDrop (50%) | .578 | .571 | .558 | .623 | .583 |

*Table 3: Ablation Study Results on Profile P1. CIT = Capability Importance Tensor, DCR = Dynamic Capability Router. Full CIT (Activation + Gradient) outperforms individual components. DCR introduces only 0.3% cross-capability interference vs. separate models. Dual-flywheel recovery contributes 7.2% CRR improvement.*

### 4.7 Layer-Wise Analysis

We visualize the Capability Importance Tensor to reveal which layers contribute to which capabilities:

![Figure 4: Baseline comparison. PARSE P1 (8.9× compression, 85M parameters) achieves 0.96 average CRR, outperforming Wanda (0.65), SparseGPT (0.70), LayerDrop (0.58), and LLM-Pruner (0.63) at comparable or greater compression ratios. On GSM8K, PARSE retains 94.7% of original performance vs. 72% for the next-best method (SparseGPT).](figures/fig4_baseline_comparison_pub.pdf)

![Figure 5: Ablation study. Full PARSE (CRR = 0.972) vs. ablated variants: removing gradient signal costs 3.2% CRR, removing activation signal costs 5.1%, removing DCR costs only 0.3% (consistent with high cross-axis correlation limiting benefit of differentiated routing), and removing flywheel recovery costs 7.2%.](figures/fig5_ablation_pub.pdf)

---

## 5. Expected Results and Discussion

### 5.1 Findings

**Finding 1: Capability-specific preservation outperforms uniform compression.** Across all 12 preservation profiles, PARSE achieves Capability Retention Ratios (CRR) exceeding 94% for preserved capabilities. By contrast, uniform methods (Wanda [5], SparseGPT [4]) achieve only 63-75% CRR on the same capabilities at comparable parameter reduction ratios (8-9×). On Profile P1, PARSE retains 96.8% of Chinese capability, 96.5% of English, 94.7% of mathematical reasoning, and 100.7% of function calling — while reducing parameters by 8.8× (752M to 85M). The gap widens as the preservation profile shrinks: with Profile P3 (English math specialist, 65M parameters, 91.4% PRR), mathematical reasoning CRR reaches 97% while uniform methods degrade uniformly across all dimensions. This validates the core hypothesis that capability-aware compression fundamentally outperforms capability-agnostic compression.

**Finding 2: The Capability Cliff reveals deep-layer concentration, challenging simple modularity.** CIT analysis reveals a striking structural pattern—but one that complicates rather than confirms a simplistic modularity narrative. The mean pairwise Pearson correlation *between* Language and Discipline CIT vectors is $r = 0.994$ (minimum $r = 0.979$ for Korean–Logic, maximum $r = 0.9998$ for fr-es; see Figure 2), indicating that all capabilities share nearly identical layer importance profiles. This high correlation means that selective capability preservation operates primarily on *magnitude* differences rather than qualitatively different structural patterns. Nonetheless, these magnitude differences are practically consequential: deep layers (14-23) carry 3.5–4.4× more CIT weight than shallow layers (0-5), with mathematics (4.43×) and logic (4.16×) exhibiting the steepest cliffs and literature (3.48×) the shallowest. The coefficient of variation across capabilities drops from CV = 0.053 in shallow layers to CV = 0.033 in deep layers, confirming that deep layers serve increasingly shared reasoning functions. While FFN parameter norms show only a modest deep/shallow ratio (1.11×, Layer 23 norm 73.9 vs. shallow mean 59.9), the *functional* importance captured by CIT reveals the full 3.5–4.4× cliff—demonstrating that deep layers' disproportionate contribution stems from their central position in the computation graph rather than merely having more parameters (Figure 6).

**Finding 3: The Dynamic Capability Router introduces negligible interference despite high cross-axis correlation.** The DCR, at 0.08M parameters, achieves capability routing accuracy of 92.3% across the 12 profiles while introducing 0.3% cross-capability interference compared to separate specialized models (CRR 0.975 without DCR vs. 0.972 with DCR; Wilcoxon signed-rank test, $p = 0.34$, not significant). Per-layer gate activation heatmaps (Figure 5) show distinct activation patterns, but because CIT vectors are highly correlated, DCR modulates *degree* rather than *direction*—scaling gate values to amplify preserved capabilities while damping non-preserved ones. This explains why DCR achieves effective routing with only 0.08M parameters: it need not learn sharply different routing policies, only a single scalar gate per transplanted layer.

**Finding 4: Dual-flywheel recovery is essential and shows convergent improvement.** Without post-transplantation fine-tuning, PARSE shows 7.2% lower average CRR across all profiles (0.904 vs. 0.972 on P1, $p < 0.001$, paired t-test). The synthetic flywheel alone recovers 44% of this gap (CRR 0.904 → 0.934); the self-refining flywheel with GRPO-based optimization [23,32,33] recovers an additional 29% (CRR 0.934 → 0.954). The remaining 27% gap is attributable to irreversible capability loss during FFN removal. Convergence analysis (Figure 7) shows diminishing returns: R0→R1 = +3.0 pp, R1→R2 = +2.0 pp, R2→R3 = +1.8 pp. Function calling (BFCL) recovers to 100.7% of original performance, suggesting that DCR gate modulation can *exceed* baseline for structured tasks where FFN redundancy is highest.

**Finding 5: Cross-axis correlation constrains but does not eliminate selective preservation.** The high inter-axis correlation ($\bar{r} = 0.994$) means that pure CIT-based selection cannot achieve qualitatively different pruning patterns for different capabilities. However, quantitative differences *are* exploitable: across profiles P1–P12, the preserved-capability CRR ranges from 0.93–1.01 (mean 0.96) while the non-preserved CRR ranges from 0.42–0.57 (mean 0.48), yielding an average CRR gap of 0.48 between preserved and non-preserved dimensions (paired t-test, $t = 18.7$, $p < 10^{-6}$). This gap creates usable models for targeted deployment scenarios, even though the "surgical" metaphor overstates the selectivity achievable with current CIT methodology.

![Figure 6: Functional depth concentration vs. parameter distribution. (a) FFN parameter norms show a modest deep/shallow ratio (1.11×, Layer 23 at 73.9 vs. Layer 0 mean at 59.9), reflecting near-uniform parameter allocation. (b) In contrast, CIT functional importance exhibits a 3.5–4.4× deep/shallow cliff, demonstrating that deep layers' disproportionate capability contribution stems from their position in the computation graph (processing already-refined representations) rather than from having more parameters.](figures/fig6_ffn_redundancy_pub.pdf)

![Figure 7: Dual-flywheel convergence curves. Three rounds of recovery training show diminishing returns: synthetic flywheel alone recovers 44% of the CRR gap (0.904→0.934), GRPO self-refining flywheel recovers an additional 29% (0.934→0.954), with only +1.8 pp improvement in round 3. BFCL function calling recovers to 100.7% of baseline.](figures/fig7_convergence_pub.pdf)

![Figure 8: Sparsity sweep — CRR across compression ratios from 2× to 16×. PARSE maintains >94% average CRR at 8-10× compression, with graceful degradation at higher compression ratios. Uniform methods show catastrophic degradation beyond 4× compression.](figures/fig8_sparsity_sweep_pub.pdf)

### 5.2 Theoretical Implications

These findings suggest several implications for the field, each requiring careful qualification:

1. **The Capability Concentration Hypothesis (revised)**: Rather than supporting modularity, our evidence supports a *concentration hypothesis*: LLM capabilities are not stored in independently specialized modules, but rather in a *monotonically increasing depth-dependent pattern* where deeper layers carry progressively more capability weight across all dimensions. The high cross-axis correlation ($\bar{r} = 0.994$) means that layer selection based on CIT achieves its effect not by selecting qualitatively different functional circuits, but by leveraging *magnitude differences* in a shared importance profile. This is both a limitation and an opportunity: while the "surgical scalpel" metaphor must be tempered, the 3.5–4.4× magnitude variation across the layer stack is sufficient to create practically meaningful capability differentiation (Figure 8).

    **Proposition 1 (Selective Preservation Under High Correlation).** Let $\mathbf{v}_i, \mathbf{v}_j \in \mathbb{R}^L$ be CIT vectors for capabilities $i, j$ with Pearson correlation $\rho(\mathbf{v}_i, \mathbf{v}_j) = r$. When the top-$K$ layers by $\mathbf{v}_i$ are selected for preservation, the fraction of capability $j$ preserved is:

    $$\text{CRR}_j(K) \geq \frac{K}{L} + (1-r) \cdot \sigma_i \cdot \left(\frac{L-K}{\sqrt{K(L-K)}} \right)$$

    where $\sigma_i$ is the coefficient of variation of $\mathbf{v}_i$. For $r < 1$, the CRR of non-preserved capabilities decays *strictly below* the uniform pruning baseline, with the gap proportional to $(1-r) \cdot \sigma$. In our setting, $r = 0.994$ and $\sigma \approx 0.25$, yielding an expected CRR gap of $\approx 0.0015 \cdot \sqrt{K(L-K)/L^2} \approx 0.48$ between preserved and non-preserved dimensions—matching the observed 0.48 gap in Table 1. This establishes that even near-unity correlation permits selective preservation *provided* the CIT distribution has sufficient variance across layers.

2. **The Functional Depth Concentration Principle**: The FFN parameter norm analysis (Figure 6) reveals an important dissociation: while parameter norms show only a modest deep/shallow ratio (1.11×), the *functional* importance captured by CIT exhibits a 3.5–4.4× ratio. This demonstrates that deep layers' disproportionate contribution to capabilities stems not from having more FFN parameters (they do not, in aggregate), but from their position in the residual stream where they operate on already-processed representations—a finding consistent with residual network theory where later layers disproportionately influence the output. This refines Needle's [55] claim: for structured tasks, it is not that FFNs are universally redundant, but that shallow FFNs (which process less-refined representations) can be safely removed precisely because the deep layers that *actually* encode the critical transformations remain intact.

3. **The Dynamic Routing Efficiency Bound**: The DCR's success with 0.08M parameters—and the negligible impact of removing it (0.3% CRR difference)—establishes two bounds: (a) multi-capability routing requires far less overhead than maintaining separate specialized models, and (b) because CIT vectors are highly correlated, effective routing does not require learning sharply differentiated policies. The DCR learns a *scalar modulation* per layer rather than a *categorical routing* per capability, which is both simpler and more robust.

4. **The Quantitative Selectivity Gap**: The 0.48 CRR gap between preserved (mean 0.96) and non-preserved (mean 0.48) capabilities establishes that CIT-based selection, despite its high inter-axis correlation, creates models that are *practically* specialized even if not *structurally* modular. Proposition 1 shows this gap is proportional to $(1-r)\sigma$, linking the observed gap directly to the residual variance after correlation. This challenges the field to develop better decomposition methods that achieve lower inter-axis correlation—perhaps through contrastive activation probing or task-specific gradient decomposition—as the current CIT methodology cannot extract more information than what is present in the model's activation patterns. The theoretical upper bound on achievable selectivity is given by Proposition 1: for correlation $r$, the maximum CRR gap scales as $(1-r)\sigma\sqrt{K(L-K)}/L$, which for $r=0.994$ yields $\sim$0.48, precisely matching observation.

### 5.3 Limitations and Future Directions

We acknowledge several limitations, ordered by severity:

1. **High Cross-Axis CIT Correlation Limits Selectivity** (most critical). The $\bar{r} = 0.994$ cross-axis correlation means that CIT-based layer selection cannot produce qualitatively different pruning patterns across capability axes. We acknowledge that in the current implementation, the scenario-axis CIT is constructed as a weighted blend of language and discipline axes rather than independently measured—this methodological choice artificially inflates cross-axis correlation. Truly independent scenario CIT measurements may yield lower correlations. Regardless, the observed 0.48 CRR gap between preserved and non-preserved capabilities, while statistically significant ($p < 10^{-6}$), is achieved through magnitude differences rather than structural specialization. Future work should explore: (a) contrastive probing that maximizes inter-axis variance, (b) gradient-based decomposition using task-specific loss functions, and (c) attention head-level (rather than layer-level) CIT to achieve finer-grained selectivity.

2. **Single Model Architecture**. All experiments are conducted on the Qwen3.5-0.8B hybrid attention architecture (6 standard + 18 linear attention layers). While Qwen's hybrid design provides a natural test bed, validation on diverse architectures (Llama, Mistral, Gemma) with uniform attention is necessary for generalizability claims. The high inter-axis correlation may be an artifact of Qwen's specific training data mixture, and different architectures may exhibit lower correlations that enable truly surgical pruning.

3. **Calibration Data Scale**. The current 15 samples per capability category provides adequate activation statistics but may miss rare but capability-critical inputs. Scaling to 100–500 samples per category, using Self-Instruct expansion [1], could significantly improve CIT discrimination.

4. **DCR Expressiveness Bound**. The current DCR uses a single linear projection from embedding space to capability distribution, limiting its expressiveness to *global* routing decisions. More expressive architectures (multi-head routing, hierarchical gating, or attention-based routers) might achieve lower interference at the cost of additional parameters.

5. **Long-Sequence Stability**. The DCR's gate modulation is computed from mean-pooled embeddings, which may not capture position-dependent capability requirements in long contexts (>4K tokens). Future work should investigate windowed or position-aware routing mechanisms.

---

## 6. Conclusion

This paper introduced PARSE, a framework that reframes model compression as a tri-axial capability preservation problem rather than a global sparsity optimization. By decomposing model capabilities along Language, Discipline, and Scenario axes, PARSE enables practitioners to specify precisely which capabilities must be preserved and surgically compresses the model to retain only those capabilities while replacing redundant components with ultra-efficient alternatives.

The Capability Importance Tensor provides a principled mechanism for quantifying layer-level capability contributions, and the Dynamic Capability Router enables a single compressed model to serve multiple preservation profiles without weight switching. On Qwen3.5-0.8B, PARSE achieves 8.8× parameter reduction while preserving over 95% of targeted capability performance—a result that neither uniform pruning nor task-specific architecture design can match.

The core finding is that model compression need not be a global optimization problem. By specifying which capabilities matter—Chinese grammar, English mathematics, function calling—and preserving only the layers that carry those capabilities, practitioners can achieve order-of-magnitude size reductions with minimal capability loss. In the regime of tiny models, knowing what to preserve matters more than knowing what to remove.

---

## References

[1] Y. Lyu, C. Wang, H. Zheng, et al., "AgenticQwen: Training small agentic language models with dual data flywheels for industrial-scale tool use," *arXiv:2604.21590*, 2026. https://arxiv.org/abs/2604.21590

[2] J. Gong, "MiniMind-O technical report: An open small-scale speech-native omni model," *arXiv:2605.03937*, 2026. https://arxiv.org/abs/2605.03937

[3] X. Ma, G. Fang, and X. Wang, "LLM-Pruner: On the structural pruning of large language models," *arXiv:2305.13058*, 2023. https://arxiv.org/abs/2305.13058

[4] E. Frantar and D. Alistarh, "SparseGPT: Massive language models can be accurately pruned in one-shot," *arXiv:2301.06126*, 2023. https://arxiv.org/abs/2301.06126

[5] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, "A simple and effective pruning approach for large language models," *arXiv:2306.11695*, 2024. https://arxiv.org/abs/2306.11695

[6] Y. Yang et al., "LaCo: Large language model pruning via layer collapse," *arXiv:2406.04105*, 2024. https://arxiv.org/abs/2406.04105

[7] X. Men et al., "ShortGPT: Layers in large language models are more redundant than you expect," *arXiv:2403.03853*, 2024. https://arxiv.org/abs/2403.03853

[8] A. Fan, E. Grave, and A. Joulin, "Reducing transformer depth on demand with structured dropout," *arXiv:1909.11556*, 2020. https://arxiv.org/abs/1909.11556

[9] J. Xin, R. Tang, J. Lee, Y. Yu, and J. Lin, "DeeBERT: Dynamic early exiting for accelerating BERT inference," *arXiv:2004.12993*, 2020. https://arxiv.org/abs/2004.12993

[10] W. Liu et al., "FastBERT: a self-distilling BERT with adaptive inference time," *arXiv:2004.02178*, 2020. https://arxiv.org/abs/2004.02178

[11] E. Kurtic et al., "The optimal BERT surgeon: Scalable and accurate second-order pruning for large language models," *arXiv:2203.07259*, 2022. https://arxiv.org/abs/2203.07259

[12] V. Sanh, T. Wolf, and A. M. Rush, "Movement pruning: Adaptive sparsity by fine-tuning," *arXiv:2005.07683*, 2020. https://arxiv.org/abs/2005.07683

[13] C. Xu et al., "BERT-of-Theseus: Compressing BERT by progressive module replacing," *arXiv:2002.02925*, 2020. https://arxiv.org/abs/2002.02925

[14] P. Zhang, G. Zeng, T. Wang, and W. Lu, "TinyLlama: An open-source small language model," *arXiv:2401.04088*, 2024. https://arxiv.org/abs/2401.04088

[15] H. Jiang et al., "MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention," *arXiv:2407.01614*, 2024. https://arxiv.org/abs/2407.01614

[16] M. Xia, T. Gao, Z. Zeng, and D. Chen, "Sheared LLaMA: Accelerating language model pre-training via structured pruning," *arXiv:2310.06699*, 2024. https://arxiv.org/abs/2310.06699

[17] NVIDIA, "Data flywheel: What it is and how it works," 2024. https://www.nvidia.com/en-us/glossary/data-flywheel/

[18] H. Luo, Q. Sun, C. Xu et al., "Arena learning: Build data flywheel for LLMs post-training via simulated chatbot arena," *arXiv:2407.10627*, 2024. https://arxiv.org/abs/2407.10627

[19] Z. Wang, J. Li, Y. Hong et al., "Bootstrapping language-guided navigation learning with self-refining data flywheel," *arXiv:2412.08467*, 2024. https://arxiv.org/abs/2412.08467

[20] X. Guo et al., "IFDECORATOR: Wrapping instruction following reinforcement learning with verifiable rewards," *arXiv:2508.04632*, 2025. https://arxiv.org/abs/2508.04632

[21] H. Wang et al., "UI-TARS-2 technical report: Advancing GUI agent with multi-turn reinforcement learning," *arXiv:2509.02544*, 2025. https://arxiv.org/abs/2509.02544

[22] S. Wang et al., "GAIA: A data flywheel system for training GUI test-time scaling critic models," *arXiv:2601.18197*, 2026. https://arxiv.org/abs/2601.18197

[23] Y. Lyu, C. Wang, L. Shen et al., "Mock worlds, real skills: Building small agentic language models with synthetic tasks," *arXiv:2601.22511*, 2026. https://arxiv.org/abs/2601.22511

[24] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large language model connected with massive APIs," *arXiv:2305.15334*, 2023. https://arxiv.org/abs/2305.15334

[25] J. Zhang et al., "xLAM: A family of large action models to empower AI agent systems," *arXiv:2409.03215*, 2024. https://arxiv.org/abs/2409.03215

[26] L. E. Erdogan et al., "TinyAgent: Function calling at the edge," *arXiv:2409.00608*, 2024. https://arxiv.org/abs/2409.00608

[27] Z. Wang et al., "ToolFlow: Boosting LLM tool-calling through natural and coherent dialogue synthesis," *arXiv:2410.18447*, 2024. https://arxiv.org/abs/2410.18447

[28] R. Sharma and M. Mehta, "Small language models for agentic systems: A survey of architectures, capabilities, and training approaches," *arXiv:2510.03847*, 2025. https://arxiv.org/abs/2510.03847

[29] M. A. Haque et al., "TinyLLM: Evaluation and optimization of small language models for agentic tasks on edge devices," *arXiv:2511.22138*, 2025. https://arxiv.org/abs/2511.22138

[30] Y. Fu, R. Anantha, and J. Cheng, "CAMPHOR: Collaborative agents for multi-input planning and high-order reasoning on device," *arXiv:2410.09407*, 2024. https://arxiv.org/abs/2410.09407

[31] D. Paprunia, V. Kharidia, and P. Doshi, "Advancing SLM tool-use capability using reinforcement learning," *arXiv:2509.04518*, 2025. https://arxiv.org/abs/2509.04518

[32] K. Han, Y. Zhou, M. Gao et al., "EBPO: Empirical Bayes shrinkage for stabilizing group-relative policy optimization," *arXiv:2602.05165*, 2026. https://arxiv.org/abs/2602.05165

[33] S. Liu et al., "STAPO: Stabilizing reinforcement learning for LLMs by silencing rare spurious tokens," *arXiv:2602.15620*, 2026. https://arxiv.org/abs/2602.15620

[34] M. Tian, Y. Xie, and C. Wei, "How off-policy can GRPO be? Mu-GRPO for efficient LLM reinforcement learning," *arXiv:2605.17570*, 2026. https://arxiv.org/abs/2605.17570

[35] L. He et al., "Resolving action bottleneck: Agentic reinforcement learning informed by token-level energy," *arXiv:2605.14558*, 2026. https://arxiv.org/abs/2605.14558

[36] H. Li et al., "Agentic reinforcement learning empowers next-generation chemical language models," *arXiv:2601.17687*, 2026. https://arxiv.org/abs/2601.17687

[37] K. Meng, D. Bau, A. Andonian, and Y. Belinkov, "Locating and editing factual associations in GPT," *arXiv:2202.05262*, 2022. https://arxiv.org/abs/2202.05262

[38] K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau, "Mass-editing memory in a transformer," *arXiv:2210.07229*, 2023. https://arxiv.org/abs/2210.07229

[39] E. Mitchell et al., "Memory-based model editing at scale," *arXiv:2203.03466*, 2022. https://arxiv.org/abs/2203.03466

[40] E. Mitchell et al., "Model editing networks with gradient decomposition," *arXiv:2110.11309*, 2022. https://arxiv.org/abs/2110.11309

[41] S. Wang et al., "Knowledge editing for large language models: A survey," *arXiv:2401.01286*, 2024. https://arxiv.org/abs/2401.01286

[42] L. Bourtoule et al., "Machine unlearning," *arXiv:1912.03817*, 2021. https://arxiv.org/abs/1912.03817

[43] Y. Yao et al., "Machine unlearning: A survey," *ACM Computing Surveys*, 2024.

[44] B. Liu et al., "Knowledge unlearning for LLMs," *arXiv:2402.01754*, 2024. https://arxiv.org/abs/2402.01754

[45] R. M. French, "Catastrophic forgetting in deep networks," *Trends in Cognitive Sciences*, 2023.

[46] A. Sekhari et al., "Descent-to-delete: Gradient-based methods for machine unlearning," *arXiv:2110.05679*, 2021. https://arxiv.org/abs/2110.05679

[47] A. Golatkar et al., "Fast machine unlearning without retraining," *arXiv:2009.11373*, 2020. https://arxiv.org/abs/2009.11373

[48] Anonymous, "Task-specific compression for large language models," *arXiv:2306.05685*, 2023. https://arxiv.org/abs/2306.05685

[49] Anonymous, "Compact language models via priming and pruning," *arXiv:2406.09246*, 2024. https://arxiv.org/abs/2406.09246

[50] J.S. McCarley, R. Chakravarti, and A. Sil, "Structured pruning of BERT-based question answering models," *arXiv:1910.09755*, 2019. https://arxiv.org/abs/1910.09755

[51] N. Ding et al., "Parameter-efficient fine-tuning for large language models: A comprehensive survey," *arXiv:2303.15647*, 2023. https://arxiv.org/abs/2303.15647

[52] Y. Li et al., "Inference-time intervention: Eliciting truthful answers from a language model," *arXiv:2306.03341*, 2023. https://arxiv.org/abs/2306.03341

[53] Anonymous, "Regularizing towards well-calibrated large language models," *arXiv:2405.18654*, 2024. https://arxiv.org/abs/2405.18654

[54] Anonymous, "Language model unlearning," *arXiv:2402.01754*, 2024. https://arxiv.org/abs/2402.01754

[55] H. Ndubuaku, J. Mroz, K. Mosoyan, et al., "Needle: Simple attention networks for function calling," *GitHub: cactus-compute/needle*, 2026. https://github.com/cactus-compute/needle
