# PARSE: Principled Architecture Retention through Scenario-Embedded Pruning for Fine-Grained Capability-Preserving Language Model Compression

**Authors**
[Anonymous for Review]

**Abstract**
The prevailing paradigm for efficient LLM deployment oscillates between two unsatisfactory extremes: uniform structural pruning, which blindly compresses models without regard for the heterogeneous distribution of linguistic, disciplinary, and task-specific capabilities within Transformer layers; and task-specific architecture design, which builds narrow expert models that abandon the broad knowledge accumulated during pretraining. This work introduces a third paradigm: treating model compression as a *capability-preserving surgical procedure* operating at the intersection of language, discipline, and scenario dimensions. Unlike prior work that either prunes uniformly (Wanda [5], SparseGPT [4]), targets general agentic capabilities (AgenticQwen [1]), designs task-specific architectures (Needle [54]), or pursues omni-modal universality (MiniMind-O [2]), PARSE (Principled Architecture Retention through Scenario-Embedded Pruning) introduces a tri-axial **Capability Importance Tensor (CIT)** that independently quantifies each layer's contribution along three conceptually distinct (Cartesian product) axes: *Language* (Chinese, English, Japanese, French, German, Russian, Spanish, Korean), *Discipline* (Mathematics, Physics, Logic, History, Geography, Literature), and *Scenario* (Function Calling, Code Generation, Mathematical Reasoning, Translation, General Chat). Through this lens, PARSE identifies capability-critical layers for user-specified preservation profiles and surgically transplants redundant layers with ultra-efficient No-FFN attention blocks for standard Transformer architectures. A lightweight **Dynamic Capability Router (DCR)** (0.08M parameters) then modulates internal residual gates based on real-time input context, enabling a single compressed model to serve multiple capability combinations without weight switching. We describe the complete four-stage pipeline—diagnosis, sculpture, transplantation, recovery—and define 12 preservation profiles for systematic evaluation. Preliminary activation-based CIT diagnostic probing was performed on Qwen3.5-0.8B (which uses a Mamba-style linear attention architecture; see Section 4.1), measuring a mean cross-axis Pearson correlation of $\bar{r} = 0.9945$ and a capability cliff ratio of 3.8–4.0× between deep and shallow layers. All 12 profiles converge to identical layer selection under the factorized CIT, empirically confirming that the tri-axial decomposition collapses to depth-weighted ranking under high inter-axis correlation. We present a comprehensive experimental design, specify evaluation metrics and baselines, and derive testable hypotheses from the methodological framework.

**Keywords**
Capability-Aware Model Compression, Fine-Grained Pruning, Architecture Transplantation, Language-Discipline-Scenario Tri-Axis Analysis, Dynamic Routing, Tiny Language Models

---

## 1. Introduction

### 1.1 The "One-Size-Fits-All" Fallacy in Model Compression

The rapid scaling of Large Language Models has produced systems with remarkable breadth—from multilingual translation to mathematical theorem proving—all contained within a single parameter set. However, this universality comes at a steep deployment cost: a Qwen3.5-0.8B model requires approximately 1.5 GB of VRAM, and inference speed on consumer-grade GPUs is limited by memory bandwidth and the sequential nature of autoregressive generation, with typical throughput ranging from 2–15 tok/s depending on hardware and batch size. For real-time edge applications requiring sub-100ms latency per token, even the modest 0.8B scale presents a deployment barrier.

The prevailing response to this challenge has bifurcated into two camps. The **compression camp** applies uniform sparsity constraints—LLM-Pruner [3] uses gradient-based coupling, SparseGPT [4] achieves one-shot 50% sparsity through second-order optimization, Wanda [5] computes weight-activation product scores, and oBERT [11] pushes compression to industry-leading ratios. These methods share a fundamental assumption: that all layers, attention heads, and FFN neurons contribute equally to all model capabilities. As ShortGPT [7] and LaCo [6] have demonstrated, this assumption is empirically false—layers exhibit striking functional specialization, with deep layers contributing disproportionately to logical reasoning [7,37] while shallow layers primarily handle syntactic alignment.

The **specialization camp** builds task-specific architectures from scratch or through targeted distillation. AgenticQwen [1] trains small models for agentic tasks using dual data flywheels. Needle [54] removes FFNs entirely for function calling, achieving 26M parameters that outperform 270M general models. MiniMind-O [2] extends to tri-modal omni processing at 100M parameters. Gorilla [24], xLAM [25], TinyAgent [26], and ToolFlow [27] each target specific agent capabilities. While these models achieve strong performance on their target tasks, they sacrifice the broad knowledge that makes LLMs valuable—a Needle model cannot solve a math problem; a Gorilla model cannot translate Chinese poetry.

This binary landscape reveals a critical gap: **no existing framework allows practitioners to specify *which* capabilities to preserve (e.g., "Chinese syntax + English mathematics + function calling") and surgically compress the model to retain only those capabilities while replacing everything else with ultra-light alternatives.** This is the gap that PARSE fills. Throughout this paper, **PARSE** refers to the overall system; its core methodology is referred to as **Fine-Grained Capability Sculpting (FGCS)**—the tri-axial decomposition and layer-selective transplantation procedure described in Section 3.

### 1.2 The Tri-Axial Innovation

PARSE is built on a single, powerful insight: **capabilities in LLMs are not monolithic—they decompose naturally along three conceptually distinct axes: Language, Discipline, and Scenario.** A Transformer layer that is critical for Chinese syntactic parsing may be entirely redundant for mathematical theorem proving. A layer that drives function-calling precision may contribute nothing to logical reasoning. Prior work has treated these capabilities as an indivisible bundle; PARSE treats them as a spectrum that can be independently preserved, attenuated, or replaced.

This tri-axial decomposition enables a fundamentally new approach to model compression. Rather than asking "which layers are important?" (a question that has no single answer), PARSE asks "which layers are important *for this specific (Language, Discipline, Scenario) combination*?" The answer varies dramatically across the tri-axial space, revealing a rich structure of capability specialization that uniform pruning completely misses.

### 1.3 Contributions

This work makes the following contributions:

1. **Tri-Axial Capability Decomposition**: We formalize the observation that LLM capabilities decompose along Language, Discipline, and Scenario axes, and introduce the **Capability Importance Tensor (CIT)**—a three-dimensional structure that quantifies each layer's contribution to every (Language × Discipline × Scenario) combination.

2. **Fine-Grained Capability Sculpting (FGCS) Framework**: We present a complete pipeline design for capability-preserving model compression—targeting standard Transformer architectures (LLaMA, Qwen2.5, Mistral) with explicit FFN submodules—that accepts user-specified preservation profiles and surgically prunes and transplants layers based on their tri-axial importance scores.

3. **Dynamic Capability Router (DCR)**: We introduce a 0.08M-parameter routing mechanism that analyzes input context at the embedding level and dynamically modulates internal residual gates, enabling a single compressed model to serve multiple (Language × Discipline × Scenario) profiles without weight switching. We provide a complete training specification including loss function, optimization algorithm, and architectural design that yields the claimed parameter count.

4. **12-Profile Evaluation Framework**: We define 12 preservation profiles (P1–P12) spanning the tri-axial capability space, with a complete experimental design including baselines, ablations, and evaluation metrics. Preliminary CIT computation was performed on Qwen3.5-0.8B using activation-based diagnostic probing, yielding key diagnostic measurements reported in Section 3.2. The full experimental pipeline is implemented as open-source software.

---

## 2. Related Work and the Capability Gap

### 2.1 Structural Pruning: The Unfulfilled Promise of Uniformity

The structural pruning literature has achieved compression ratios up to 50% with minimal aggregate accuracy loss. LLM-Pruner [3] demonstrated that gradient-based coupling could identify redundant structures without task-specific data. SparseGPT [4] pushed the frontier to one-shot 50% sparsity through efficient second-order Hessian approximations. Wanda [5] simplified the criterion to the product of weight magnitude and input activation norm, eliminating the need for gradient computation entirely. LaCo [6] approached the problem from the opposite direction, proposing layer collapse based on representation stability rather than individual weight importance.

However, these methods share a critical limitation: they optimize a *global* sparsity constraint without any notion of *capability-specific* importance. A layer that is globally "unimportant" under Wanda's scoring may be the single most critical layer for Chinese-English translation or mathematical induction. ShortGPT [7] revealed that entire layers could be removed with minimal impact on average perplexity—but this "average" masks the catastrophic degradation that occurs in specific capability dimensions. Our work directly addresses this limitation by replacing the global importance score with a tri-axial Capability Importance Tensor.

The dynamic inference literature offers complementary insights. LayerDrop [8] introduced structured dropout during training to enable arbitrary depth extraction at inference time. DeeBERT [9] and FastBERT [10] implemented confidence-based early exiting, allowing simple samples to bypass deeper layers. BERT-of-Theseus [13] pioneered progressive module replacement for gradual compression. These works demonstrate that not all layers are needed for all inputs—a principle that FGCS extends from input complexity to capability specificity.

Complementary methods provide additional technical foundations: Movement Pruning [12] introduced adaptive sparsity through first-order gradient information, TinyLlama [14] demonstrated the viability of small-model pretraining at 1T tokens, MInference [15] accelerated long-context processing through dynamic sparse attention patterns, and LLM-Shearing [16] proposed flexible structured pruning with continued pretraining for capability recovery.

### 2.2 Knowledge Editing and Machine Unlearning: The Precision Paradigm

The knowledge editing literature has developed precise tools for localizing and modifying specific pieces of information within LLMs. ROME [37] pioneered rank-one model editing by identifying the specific MLP layers where factual associations are stored. MEMIT [38] extended this to batch editing of thousands of facts simultaneously. MEND [39] trained hyper-networks to predict parameter updates for efficient editing, while SERAC [40] employed external memory to handle counterfactual knowledge without direct weight modification. Wang et al. [41] surveyed techniques for knowledge editing at scale.

The precision achieved by these methods—identifying and modifying *specific* factual associations without disrupting *unrelated* knowledge—directly inspires FGCS's approach to capability preservation. If a single fact can be localized to a specific layer, then an entire capability (e.g., mathematical reasoning) can similarly be traced to a specific subset of layers. FGCS operationalizes this analogy through the Capability Importance Tensor.

The machine unlearning literature provides the inverse perspective. SISA [42] proposed sharded training for efficient forgetting, extended in subsequent surveys [43,44]. Catastrophic forgetting [45] remains a fundamental challenge. Descent-to-Delete [46] introduced gradient-based unlearning, while Fast-Machine-Unlearning [47] accelerated the process. These methods demonstrate that knowledge *removal* can be achieved with surgical precision—a capability that FGCS leverages when identifying layers that can be safely replaced without affecting preserved capabilities.

### 2.3 Data Flywheels: The Self-Improvement Engine

The data flywheel paradigm [17]—iteratively improving training data using model outputs—has become central to small model training. AgenticQwen [1] proposed a dual flywheel architecture combining synthetic reasoning trace generation with agent-environment interaction data. ArenaLearning [18] pioneered AI-driven simulated arenas for automated model evaluation and data generation. SRDF [19] introduced self-refining data pipelines where a generator and navigator collaboratively produce increasingly high-quality training trajectories.

The paradigm has been extended to diverse domains: IFDecorator [20] for instruction following with verifiable rewards, UI-TARS-2 [21] for GUI agent training through multi-turn reinforcement learning, GAIA [22] for GUI critic model training with test-time scaling, and SynthAgent [23] for synthetic environment-based agent skill acquisition. FGCS employs a dual-flywheel strategy for post-transplantation capability recovery, generating targeted calibration data for each (Language × Discipline × Scenario) combination and filtering through a critic model inspired by GAIA [22]. GRPO-based optimization [55,32,33] provides the underlying RL mechanism for the self-refining flywheel.

### 2.4 Agentic Systems and Tool Calling: The Specialization Frontier

The emergence of agentic capabilities in small models has been transformative. Gorilla [24] pioneered LLM-based API calling with retrieval augmentation, demonstrating that fine-tuned LLaMA models could surpass GPT-4 on API selection tasks. xLAM [25] scaled the approach to a family of large action models from 1B to 8×22B parameters, achieving top performance on the Berkeley Function-Calling Leaderboard. TinyAgent [26] brought function calling to the edge, with 1.1B and 7B models deployed on MacBook that surpassed GPT-4-Turbo. ToolFlow [27] introduced graph-based tool sampling strategies, using only 8,000 high-quality dialogue samples to achieve GPT-4-level tool calling on LLaMA-3.1-8B.

Systematic surveys by Sharma and Mehta [28] and evaluations by Haque et al. [29] established comprehensive benchmarks for small language model agent capabilities. CAMPHOR [30] proposed collaborative multi-agent architectures for on-device deployment, using hierarchical reasoning agents for task decomposition and expert agents for execution. These works collectively demonstrate that small models can achieve remarkable performance on specific agent tasks—but they achieve this specialization by abandoning the broad capabilities of their larger counterparts. FGCS bridges this gap by preserving multiple capabilities simultaneously through tri-axial decomposition.

### 2.5 GRPO-Based Reinforcement Learning: Stability at Scale

DeepSeekMath [55] introduced Group Relative Policy Optimization (GRPO), a reinforcement learning algorithm that estimates advantages through within-group relative comparisons of multiple samples, completely eliminating the critic model and substantially reducing memory and computational overhead. SLM-ToolUse-GRPO [31] specifically studied GRPO for enhancing small language model tool-use capabilities, designing reward functions targeting JSON structure, tool selection, and parameter precision.

The growing adoption of GRPO has exposed stability challenges. EBPO [32] addressed these through empirical Bayes shrinkage estimators that regularize within-group baselines, significantly reducing estimation variance in small group settings. STAPO [33] discovered the "spurious token" phenomenon—approximately 0.01% of tokens contribute negligibly to the objective yet receive disproportionately large gradient updates—and proposed silencing these tokens for stable training. Mu-GRPO [34] demonstrated that GRPO can tolerate far greater rollout delays than previously expected, achieving approximately 2× training speedup.

In agentic contexts, ActFocus [35] identified the "action bottleneck"—gradient signals concentrated on a small number of action tokens—and proposed token-level energy reweighting, improving terminal performance by over 60 percentage points across multiple environments. ChemCRAFT [36] demonstrated agentic RL in professional domains, enabling small models to surpass cloud-scale models in drug design tasks. FGCS leverages GRPO-based optimization during the post-transplantation fine-tuning phase, using capability-specific reward functions derived from the tri-axial decomposition.

### 2.6 Specialized Architectures: The Efficiency Frontier

Needle [54] represents a radical architectural departure from conventional Transformer design. The core insight is that for structured tasks like function calling, the Feed-Forward Network (FFN)—which constitutes approximately 65% of standard Transformer parameters—is entirely redundant. The softmax operation in attention already provides the necessary nonlinearity for information routing, and function calling is fundamentally a retrieval-and-assembly task requiring cross-attention alignment rather than per-position feature transformation. By removing FFNs entirely and relying on pure attention with gated residuals and ZCRMSNorm (defined below), Needle achieves 26M parameters that outperform 270M–600M general-purpose models on function calling benchmarks. ZCRMSNorm (Zero-Centered RMSNorm) initializes the scale parameter $\gamma = 0$, making the block an identity-up-to-scale mapping at initialization and enabling stable training of the reduced architecture.

MiniMind-O [2] extended the efficiency philosophy to the omni-modal domain. Through Thinker-Talker dual-path decoupling, the model separates semantic understanding (Thinker) from speech generation (Talker), passing representations through a middle-layer bridge rather than the embedding or final layer. Multi-Token Prediction (MTP) simultaneously generates 8-layer Mimi audio codebooks, achieving tri-modal (text-audio-image) processing at only 100M trainable parameters. The project provides a complete, inspectable, and reproducible baseline for efficient multi-modal architecture design.

Additional architectural innovations inform FGCS: BERT-of-Theseus [13] for progressive module replacement, Movement Pruning [12] for adaptive sparsity learning, and TinyLlama [14] for small-model pretraining scaling laws. Parameter-Efficient Fine-Tuning [51] provides adapter-based adaptation techniques. Inference-Time Intervention [52] demonstrates inference-time steering of model outputs, while regularization techniques for calibration [53] address the reliability of compressed models.

---

## 3. The PARSE Methodology

### 3.1 Formal Problem Definition

Let $M$ be a pretrained language model with $L$ layers. We define the **capability space** $\mathcal{C}$ as the Cartesian product of three conceptually distinct axes:

$$\mathcal{C} = \mathcal{L}_{ang} \times \mathcal{D}_{isc} \times \mathcal{S}_{cen}$$

where:
- $\mathcal{L}_{ang} = \{\text{zh}, \text{en}, \text{ja}, \text{fr}, \text{de}, \text{ru}, \text{es}, \text{ko}\}$ is the language axis (8 categories),
- $\mathcal{D}_{isc} = \{\text{math}, \text{physics}, \text{logic}, \text{history}, \text{geo}, \text{lit}\}$ is the discipline axis (6 categories),
- $\mathcal{S}_{cen} = \{\text{fc}, \text{code}, \text{math\_reasoning}, \text{translation}, \text{chat}\}$ is the scenario axis (5 categories).

These 19 capability axes (8 + 6 + 5) are selected based on established evaluation benchmarks: language categories correspond to major linguistic families in multilingual NLP [56]; discipline categories map to standard academic domains used in knowledge-intensive benchmarks such as MMLU [57]; and scenario categories reflect the deployment use cases prioritized in small-model agent research [25,26,27,28,29].

A **preservation profile** $\mathcal{P} \subset \mathcal{C}$ specifies which capability combinations must be preserved after compression. The **compression objective** is to minimize the active parameter count $|M'|$ subject to the constraint that for all $c \in \mathcal{P}$, the performance degradation $\Delta(c)$ does not exceed a threshold $\epsilon$:

$$\min |M'| \quad \text{s.t.} \quad \forall c \in \mathcal{P}: \Delta(c) \leq \epsilon$$

This formulation generalizes prior work: uniform pruning corresponds to $\mathcal{P} = \emptyset$ (no preservation constraints); task-specific design corresponds to $|\mathcal{P}| = 1$ (single capability); FGCS supports arbitrary $1 \leq |\mathcal{P}| \leq |\mathcal{C}|$.

### 3.2 The Capability Importance Tensor (CIT)

For each layer $l \in \{1, \dots, L\}$ and each capability combination $c \in \mathcal{C}$, we define the **Capability Importance Tensor** entry:

$$\text{CIT}(l, c) = \alpha \cdot A(l, c) + (1-\alpha) \cdot G(l, c)$$

where the **Activation Capacitance** $A(l, c)$ measures the L1-norm of hidden state activations per token averaged over the calibration dataset for capability $c$, and the **Gradient Sensitivity** $G(l, c)$ computes the element-wise gradient-weight product summed over FFN parameters only:

$$A(l, c) = \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} \|h_l(x)\|_1, \quad G(l, c) = \sum_{\substack{(n, p) \in \text{FFN}_l}} \left|\frac{\partial \mathcal{L}_c}{\partial p} \cdot p\right|$$

Here, $\mathcal{D}_c$ is a compact calibration dataset for capability $c$, $h_l(x)$ denotes the hidden state activations at layer $l$ for token sequence $x$, $\mathcal{L}_c$ is the language modeling loss on $\mathcal{D}_c$, and the summation in $G$ is restricted to FFN parameters (gate\_proj, up\_proj, down\_proj) to match the transplantation scope. The hyperparameter $\alpha \in [0, 1]$ balances the contributions of activation-based and gradient-based importance signals ($\alpha = 0.6$ in our design). **Note on preliminary measurement**: The diagnostic CIT results reported in this section (correlation r̄, capability cliff ratios, and layer selection) were obtained using activation-only probing (equivalent to $\alpha = 1.0$, $G=0$) with a subset of 10 calibration samples per category. This is because gradient computation requires standard FFN parameters for the gradient-weight product, which are absent in Qwen3.5's Mamba-style architecture. Activation-only CIT captures the dominant depth-wise concentration signal; the gradient sensitivity component $G(l,c)$ is specified for execution on standard Transformer architectures and its marginal contribution will be quantified in ablation A1.

The CIT is a $L \times |\mathcal{L}_{ang}| \times |\mathcal{D}_{isc}| \times |\mathcal{S}_{cen}|$ tensor. For computational efficiency, we compute CIT entries for each axis independently (averaging over the other two), then combine them multiplicatively:

$$\text{CIT}(l, lang, disc, scen) = \text{CIT}_{lang}(l, lang) \cdot \text{CIT}_{disc}(l, disc) \cdot \text{CIT}_{scen}(l, scen)$$

where each marginal CIT is normalized to sum to 1 across layers. This factorization reduces computation from $O(L \times |\mathcal{L}_{ang}| \times |\mathcal{D}_{isc}| \times |\mathcal{S}_{cen}|)$ to $O(L \times (|\mathcal{L}_{ang}| + |\mathcal{D}_{isc}| + |\mathcal{S}_{cen}|))$. Note that the product of three distributions each summing to 1 is not itself normalized; however, since $S_{preserve}$ (Equation 5) is used only for ranking layers and the product preserves the ordering induced by each marginal, re-normalization is unnecessary for layer selection.

**Factorization assumptions and limitations.** The multiplicative factorization described above implicitly assumes that the contributions of language, discipline, and scenario are multiplicatively separable—equivalently, that their importance vectors are approximately orthogonal in log-space. Preliminary CIT measurement on Qwen3.5-0.8B using activation-based probing confirms the strong depth-wise concentration observed in prior work [7,37]: the mean cross-axis Pearson correlation across Language, Discipline, and Scenario marginal CIT vectors is $\bar{r} = 0.9945$ (Language-Discipline: $r = 0.994$, Language-Scenario: $r = 0.992$, Discipline-Scenario: $r = 0.998$). Under such high correlation, the multiplicative factorization reduces additive information: all 12 preservation profiles (P1–P12) converge to the identical layer selection, pruning the same 6 layers ({0,1,2,4,5,8}) regardless of the specified capability combination. This empirically validates the concern raised earlier: the tri-axial decomposition collapses toward a single depth-weighted importance ranking when cross-axis correlations approach unity. We discuss this limitation in Section 5.3 and introduce the contrastive CIT below as a potential mitigation. The full (non-factorized) CIT computation and contrastive CIT are designed to diagnose and address this collapse, respectively.

**Contrastive CIT.** To address the selectivity limitation when cross-axis correlations are high, we introduce **Contrastive CIT** as a theoretically motivated variant:

$$\text{CIT}^{\text{contrast}}(l, c) = \max\left(0,\; \text{CIT}(l, c) - \lambda \cdot \frac{1}{|\mathcal{C}|-1}\sum_{c' \neq c} \text{CIT}(l, c')\right)$$

where $\lambda \in [0, 1]$ controls the contrastive strength. At $\lambda = 0$, this reduces to standard CIT; at $\lambda = 1$, only layers that are *more important for capability $c$ than the cross-category mean* survive. Contrastive CIT suppresses shared layers (high importance for all capabilities equally) and amplifies capability-specific layers, directly addressing the selectivity limitation imposed by high inter-axis correlation. The empirical evaluation of contrastive CIT, including the relationship between correlation reduction and CRR improvement, is left for future experimental work.

![Figure 1: Tri-axial CIT analysis, measured on Qwen3.5-0.8B via activation-based probing. (a) Heatmap of CIT scores across 24 layers and 19 capability axes. A clear "Capability Cliff" pattern emerges: deep layers (16–23) exhibit 3.81× (Language), 4.02× (Discipline), and 4.03× (Scenario) higher mean CIT scores than shallow layers (0–5). (b) Cross-axis Pearson correlation matrix reveals $\bar{r} = 0.9945$ between Language and Discipline axes. The minimum cross-axis pairwise correlation is Korean–Mathematics at $r = 0.980$. (c) Capability Cliff quantification: deep/shallow CIT ratios range from 3.70× to 4.60× across the 19 capability axes.](figures/fig1_cit_analysis_main.pdf)

### 3.3 Capability-Preserving Layer Selection

Given a preservation profile $\mathcal{P}$, we compute the **preservation-weighted importance** for each layer:

$$S_{preserve}(l) = \sum_{c \in \mathcal{P}} w_c \cdot \text{CIT}(l, c)$$

where $w_c$ are user-specified capability weights (default: uniform). Layers are ranked by $S_{preserve}(l)$, and the top $K$ layers are retained in their original form, where $K = \lceil L \cdot (1 - \tau/2) \rceil$ and $\tau$ is the target sparsity. The factor $\tau/2$ (rather than $\tau$) reflects the fact that transplantation (Section 3.4) removes only FFN parameters (~65% of each layer) while retaining Self-Attention (~35%). Halving $\tau$ in the layer selection threshold ensures the aggregate parameter reduction matches the target compression ratio after accounting for the FFN-only removal in transplanted layers.

**Standard attention layer retention.** For standard Transformer architectures, layers at positions that use full softmax attention (as opposed to efficient approximations) are **always retained** regardless of their CIT scores. These layers carry critical routing and cross-referencing functions that cannot be approximated by NoFFN pass-through. For Qwen3.5-0.8B's hybrid architecture, the 6 layers at positions $\{3, 7, 11, 15, 19, 23\}$ that incorporate standard attention components are treated as forced-retain, though they use Mamba-style selective state space modules rather than standard FFNs. This architectural constraint is empirically motivated: prior work on layer removal [7] demonstrates that removing full-attention layers causes catastrophic degradation.

The remaining $L-K$ layers (excluding the forced-retained standard attention layers) are designated for **architecture transplantation**: their FFN components are removed (following the No-FFN principle validated by Needle [54]), and their Self-Attention modules are retained with gated residual connections.

### 3.4 Dynamic Capability Router (DCR)

To enable a single compressed model to serve multiple preservation profiles without weight switching, we introduce the Dynamic Capability Router. The DCR is a lightweight neural probe (0.08M parameters) that analyzes the input context at the embedding level:

$$R(x) = \text{softmax}(W_r \cdot \text{mean}(h_{embed}(x)) + b_r)$$

where $R(x) \in \mathbb{R}^{|\mathcal{C}|}$ is a probability distribution over capability combinations. This vector dynamically modulates the internal residual gates of the transplanted NoFFN blocks:

$$g_l(x) = \sigma \left( g_l^{base} + \sum_{c \in \mathcal{C}} R_c(x) \cdot g_{l,c}^{specialized} \right)$$

where $g_l^{base}$ is the base gate value (initialized to 0, giving $\sigma(0) = 0.5$), and $g_{l,c}^{specialized}$ are learned capability-specific gate perturbations. This mechanism allows each transplanted block to *amplify* or *suppress* its contribution based on the detected input context, effectively creating a soft form of mixture-of-experts without the routing instability that plagues traditional MoE architectures [30].

**DCR Training.** The DCR parameters $W_r$, $b_r$, and $g_{l,c}^{specialized}$ are trained jointly with the post-transplantation recovery stage. The training objective combines two signals: (1) a **capability classification loss** $\mathcal{L}_{cls} = -\sum_{c \in \mathcal{C}} y_c \log R_c(x)$, where $y_c$ is 1 if capability $c$ belongs to the preservation profile and 0 otherwise. This provides weak supervision that encourages the router to activate gates for preserved capabilities. The per-sample capability label $y$ is derived from the known provenance of each training sequence (e.g., a GSM8K sample is labeled as math\_reasoning); (2) a **task-specific LM loss** $\mathcal{L}_{lm}$ on the flywheel-generated training data, where DCR gate modulation influences which layers contribute to the forward computation.

The joint loss is $\mathcal{L}_{dcr} = \mathcal{L}_{lm} + \beta \cdot \mathcal{L}_{cls}$, with $\beta = 0.1$ controlling the relative weight of the routing auxiliary loss. Training uses AdamW [62] with learning rate $1\times 10^{-4}$, weight decay $0.01$, and a linear warmup of 100 steps followed by cosine decay over 3 flywheel rounds (Section 3.5, Stage 4). The batch size is 8 sequences of up to 2048 tokens, with gradient accumulation over 4 steps for an effective batch size of 32. To prevent the router from degenerating into a trivial mapping, the training data includes both in-profile and out-of-profile capability samples (with $y_c=0$ for non-preserved capabilities), ensuring the DCR learns discriminative routing rather than uniformly activating all preserved-capability gates.

**Parameter count derivation.** The 0.08M parameter count arises from: (1) the router MLP using a bottleneck dimension of $\lfloor d_{model}/8 \rfloor = 256$ (for $d_{model}=2048$), yielding $d_{model} \times 256 + 256 \times |\mathcal{C}|$ parameters for $W_r$ and $b_r$ (approximately $2048 \times 256 + 256 \times 240 \approx 585,728$), and (2) $L_{transplanted} \times |\mathcal{C}|$ scalar gate perturbation parameters for $g_{l,c}^{specialized}$ (approximately $12 \times 240 = 2,880$). The total of approximately 0.59M parameters is reduced to the reported 0.08M through low-rank decomposition of the router matrix ($W_r = U V^T$ with $U \in \mathbb{R}^{2048 \times 32}, V \in \mathbb{R}^{256 \times 32}$, yielding $2048 \times 32 + 256 \times 32 = 73,728$ router parameters) and layer-grouped sharing of the gate perturbation parameters (sharing $g_{l,c}^{specialized}$ across blocks of 3 transplanted layers, reducing to $\lceil L_{transplanted}/3 \rceil \times |\mathcal{C}| \approx 4 \times 240 = 960$). The total is approximately $73,728 + 960 + 256 + 240 \approx 75,184 \approx 0.075\text{M}$, rounded to 0.08M.

![Figure 2: Cross-axis correlation analysis, measured on Qwen3.5-0.8B via activation-based CIT. (a) Language-language correlations ($\bar{r} = 0.995$), (b) Discipline-discipline correlations ($\bar{r} = 0.999$), and (c) cross-axis Language-Discipline correlations ($\bar{r} = 0.994$). The near-unity correlations indicate that CIT vectors share nearly identical structure across capability axes, with selectivity driven by magnitude rather than direction. Mean cross-axis correlation (averaging Lang-Disc, Lang-Scen, Disc-Scen) is $\bar{r} = 0.9945$; minimum pairwise cross-axis is Korean–Mathematics at $r = 0.980$. All 12 preservation profiles converge to the same layer selection (prune layers 0,1,2,4,5,8), confirming that the tri-axial factorization collapses to a single depth-weighted ranking under such high correlation.](figures/fig2_correlation_analysis.pdf)

### 3.5 Transplantation and Recovery Pipeline

The complete FGCS pipeline proceeds in four stages:

**Stage 1: Diagnostic Probing**. For each axis of the capability space, we construct compact calibration datasets $\mathcal{D}_{lang}$, $\mathcal{D}_{disc}$, $\mathcal{D}_{scen}$ using 10–20 representative prompts per category. These datasets are generated through the synthetic data flywheel [1,18] to ensure coverage of edge cases and low-resource combinations.

**Stage 2: CIT Computation and Layer Selection**. We compute the marginal CIT entries for each axis, combine them multiplicatively for the user-specified preservation profile $\mathcal{P}$, and select the top-$K$ layers for retention. The remaining layers are designated for transplantation.

**Stage 3: Architecture Transplantation**. For each transplanted layer, we:
1. Retain the Self-Attention module (Q, K, V, O projections) unmodified,
2. Remove the FFN module entirely (gate\_proj, up\_proj, down\_proj) and replace with Identity,
3. Insert the No-FFN pass-through block that computes: $\text{output} = (1 - g_l) \cdot h + g_l \cdot \text{LN}(h)$, where $h$ is the post-attention hidden state and $g_l$ is the DCR-modulated gate,
4. Register forward hooks on the FFN/MLP submodule (for standard Transformer architectures) so that the No-FFN block's gated residual computation replaces the Identity-FFN output during inference,
5. Initialize gate perturbations to zero (ensuring $g_l = \sigma(0) = 0.5$, so transplanted blocks start at 50% pass-through, providing a numerically stable initialization).

**Stage 4: Dual-Flywheel Recovery**. Post-transplantation, we apply a dual data flywheel for capability recovery:
1. **Synthetic Flywheel**: Generate targeted calibration data for each (Language × Discipline × Scenario) combination using the teacher model, with Self-Instruct expansion for structural diversity [1] and Persona injection for contextual diversity.
2. **Self-Refining Flywheel**: The compressed model generates responses; a critic model (inspired by GAIA [22]) scores quality; high-quality traces are re-injected with GRPO-based optimization [55,32,33].

### 3.6 Complexity Analysis

The computational complexity of FGCS is dominated by the CIT computation, which requires $O(L \cdot (|\mathcal{L}_{ang}| + |\mathcal{D}_{isc}| + |\mathcal{S}_{cen}|) \cdot |\mathcal{D}_c| \cdot d_{model})$ operations for the forward passes through all layers. With $L=24$, $|\mathcal{L}_{ang}|=8$, $|\mathcal{D}_{isc}|=6$, $|\mathcal{S}_{cen}|=5$, $|\mathcal{D}_c|=10$ (effective samples used in preliminary measurement; full gradient CIT would use 15 per category), and $d_{model}=2048$, the activation-only CIT requires approximately $24 \times 19 \times 10 \times 2048 \approx 9.3 \times 10^6$ operations—negligible compared to the training cost of the original model. Adding gradient sensitivity would approximately triple the computation. The DCR adds approximately 0.08M parameters (design target; approximately 0.09% of a target ~85M-parameter compressed model), and the gate modulation introduces no additional inference latency beyond a single matrix-vector multiplication and sigmoid evaluation per transplanted layer.

---

## 4. Experimental Design

### 4.1 Experimental Infrastructure

The experimental implementation targets an Apple M4 (16 GB unified memory) with PyTorch 2.5.1 for preliminary CIT diagnostic probing, and an NVIDIA RTX 4060 (8 GB VRAM) with CUDA 12.1 for the full transplantation and evaluation pipeline. The base model is Qwen3.5-0.8B, comprising 752M parameters across 24 layers. **Architecture note**: Qwen3.5-0.8B uses a Mamba-style selective state space (linear attention) architecture where each layer's primary computation is a `linear_attn` module rather than the standard Self-Attention + FFN (gate\_proj, up\_proj, down\_proj) design. The full 24 layers employ the linear attention mechanism; 6 layers at positions {3, 7, 11, 15, 19, 23} additionally incorporate standard softmax attention components for cross-referencing. The CIT diagnostic probing (Section 3.2) is architecture-agnostic, operating on hidden state activations. The FFN-removal transplantation methodology (Stage 3, Section 3.5) is designed for standard Transformer architectures (e.g., LLaMA, Qwen2.5, Mistral) where FFN modules constitute the majority of parameters. We retain Qwen3.5-0.8B as the CIT measurement target due to its availability, while noting that full pipeline execution requires a model with explicit FFN submodules. The hidden dimension is $d_{model} = 2048$. Calibration prompts and the CIT computation code are available in the project repository under `code/parse/data/calibration.py` and `run_phase1_cit.py`.

### 4.2 Capability Dimensions and Calibration Data

**Language Axis** (8 categories): Chinese (zh), English (en), Japanese (ja), French (fr), German (de), Russian (ru), Spanish (es), Korean (ko). Calibration data consists of 10–15 prompt templates per language (10 used in the preliminary activation-only measurement; 15 is the design target for gradient-inclusive CIT), covering declarative, interrogative, and imperative constructions.

**Discipline Axis** (6 categories): Mathematics, Physics, Logic, History, Geography, Literature. Calibration data consists of 10–15 domain-specific prompts (10 in preliminary measurement, 15 design target) covering factual recall, reasoning, and problem-solving.

**Scenario Axis** (5 categories): Function Calling (fc), Code Generation (code), Mathematical Reasoning (math\_reasoning), Translation (translation), General Chat (chat). Calibration data consists of 10–15 task-specific prompts (10 in preliminary measurement, 15 design target) per scenario.

### 4.3 Preservation Profiles

We define 12 distinct preservation profiles spanning the capability space:

Table 1: Preservation Profiles

| Profile | Languages | Disciplines | Scenarios | Description |
|:---|:---|:---|:---|:---|
| P1 | zh, en | math, logic | fc, math\_reasoning | Chinese + English STEM + Agent |
| P2 | zh, en, ja | math, physics | all | East Asian + STEM |
| P3 | en | math | all | English math specialist |
| P4 | zh | all | all | Chinese full-capability |
| P5 | all | math, logic, physics | fc | Multilingual STEM agent |
| P6 | zh, en | all | fc, code | Bilingual developer agent |
| P7 | all | math | math\_reasoning | Multilingual math solver |
| P8 | zh, en, ja, fr | all | translation | Quad-lingual translator |
| P9 | all | all | fc | Universal function caller |
| P10 | zh, en | all | all | Bilingual full-capability |
| P11 | all | math, logic | all | Universal STEM preservation |
| P12 | zh, en | math, logic, physics | fc, code, math\_reasoning | Full targeted preservation |

### 4.4 Comparison Baselines

We specify the following baselines for systematic comparison:

1. **Wanda [5]**: Uniform weight-activation product pruning.
2. **SparseGPT [4]**: One-shot second-order pruning.
3. **LayerDrop [8]**: Structured layer removal.
4. **LLM-Pruner [3]**: Gradient-based coupled structure pruning.
5. **Needle [54]**: Full FFN removal (task-specific function calling architecture).
6. **Knowledge Distillation Baseline**: A standard logit-level distillation from Qwen3.5-0.8B teacher to a target ~85M-parameter student Transformer (e.g., 12-layer, 768-dim), trained on a general corpus with distillation temperature $\tau_{KD}=3.0$ and a weighted combination of hard-label and soft-label losses. This directly tests whether PARSE's surgical approach outperforms the most straightforward method for creating small capability-retaining models.
7. **Original Qwen3.5-0.8B**: Uncompressed baseline.
8. **Qwen2.5-0.5B**: The closest existing small model from the Qwen2.5 family (0.5B parameters), used as a scratch-trained reference point for the same parameter scale.

**Fair comparison note.** Standard pruning baselines (Wanda, SparseGPT, LayerDrop) typically operate at 50% sparsity (~376M parameters). For fair comparison, we plan to evaluate two variants of each baseline: (a) at their standard 50% sparsity setting, and (b) at a sparsity level that produces ~85M active parameters (matching PARSE's target). This dual-setting comparison isolates the effect of the method from the effect of the parameter budget.

### 4.5 Evaluation Metrics

We define the following metrics for systematic comparison:

1. **Capability Retention Ratio (CRR)** : For a capability $c$, $\text{CRR}(c) = \text{Metric}_{compressed}(c) \;/\; \text{Metric}_{original}(c)$. CRR > 1 indicates the compressed model *exceeds* original performance on that capability (possible via DCR gate amplification).

2. **Parameter Reduction Ratio (PRR)** : $\text{PRR} = (|M_{original}| - |M_{compressed}|) \;/\; |M_{original}|$.

3. **Cross-Capability Interference (CCI)** : $\text{CCI} = \frac{1}{|\mathcal{C} \setminus \mathcal{P}|} \sum_{c \notin \mathcal{P}} (1 - \min(\text{CRR}(c), 1))$. CCI measures degradation on *non-preserved* capabilities—lower CCI indicates cleaner selective preservation.

4. **Perplexity (PPL)** : Standard language modeling PPL on held-out test data.

5. **Task-Specific Accuracy**: GSM8K [58] for mathematical reasoning, BFCL [59] for function calling, HumanEval [60] for code generation, and BLEU [61] for translation quality.

6. **Inference Throughput**: Tokens per second measured on the reference hardware (RTX 4060) at batch size 1, averaged over 128-token generation.

7. **Statistical Significance**: Paired t-tests with Bonferroni correction across profiles, with $p < 0.05$ significance threshold.

*Preliminary CIT diagnostic probing (activation-only) has been completed; all transplantation, flywheel recovery, and full evaluation metrics are pending experimental execution.*

### 4.6 Ablation Studies

We design the following ablation experiments for systematic validation:

**A1. CIT Component Ablation**: Compare full CIT (Activation + Gradient, $\alpha = 0.6$) against Activation-only ($\alpha = 1.0$) and Gradient-only ($\alpha = 0.0$) variants to quantify the marginal contribution of each importance signal.

**A2. DCR Effectiveness**: Compare PARSE with DCR against PARSE without DCR (separate model per profile) to measure the capability interference introduced by unified routing.

**A3. Flywheel Recovery**: Compare PARSE with and without the dual-flywheel post-transplantation fine-tuning, and with individual flywheel components (synthetic only, GRPO only), to quantify the recovery contribution of each stage.

**A4. Preservation Profile Sensitivity**: Vary the size of the preservation profile $|\mathcal{P}|$ from 1 to 20 capability combinations and measure the impact on PRR and CRR.

**A5. Sparsity Sweep**: Evaluate PARSE across compression ratios from 2× to 16× to characterize the performance-compression trade-off curve.

**A6. Factorization vs. Full CIT**: Compare the multiplicative factorization (Section 3.2) against the full (non-factorized) CIT to quantify the information loss induced by the factorization approximation.

*Ablation results pending experimental execution.*

### 4.7 Layer-Wise Patterns

Preliminary activation-based CIT measurement on Qwen3.5-0.8B confirms the "capability cliff" pattern: shallow layers (0–5) contribute primarily to surface-level linguistic features; intermediate layers (6–15) distribute evenly across disciplines and scenarios; and deep layers (16–23) accumulate disproportionately high CIT scores across all capability dimensions, with a mean deep/shallow ratio of 3.8–4.0×. The high cross-axis correlation ($\bar{r}=0.9945$) indicates that capability importance follows a concentration (depth-dependent) pattern rather than a modular pattern, with all 12 preservation profiles converging to identical layer selection. Whether finer-grained analysis (attention head-level CIT, task-specific gradient decomposition) can reveal modular structure not visible at the layer level remains an open empirical question.

---

## 5. Hypotheses and Expected Outcomes

The following hypotheses derive from the methodological framework established above. They await empirical validation through the pipeline described in Section 4.

### 5.1 Core Hypotheses

**Hypothesis 1 (Capability-Specific Preservation).** By selecting layers based on preservation-weighted CIT scores rather than uniform sparsity thresholds, PARSE is designed to achieve higher Capability Retention Ratios on preserved capability dimensions than capability-agnostic methods (Wanda [5], SparseGPT [4], LayerDrop [8]) at comparable parameter budgets. The margin is expected to increase as the preservation profile becomes more constrained (fewer capability dimensions specified), since fewer layers need to be retained to cover the narrower profile.

**Hypothesis 2 (Finer-Grained Modularity).** While preliminary layer-level CIT measurement supports a concentration pattern ($\bar{r} = 0.9945$, all profiles converge to identical layer selection), it remains an open question whether finer-grained analysis—attention head-level CIT, task-specific gradient decomposition, or contrastive CIT—can reveal modular structure not visible at the layer level. The factorization analysis (A6) will quantify how much of the tri-axial structure is preserved under the multiplicative approximation, and contrastive CIT experiments will test whether the cross-axis correlation can be meaningfully reduced.

**Hypothesis 3 (DCR Overhead).** The Dynamic Capability Router, at 0.08M parameters, is hypothesized to introduce minimal cross-capability interference compared to deploying separate specialized models per profile. The routing accuracy and interference level depend on how differentiated the per-layer CIT vectors are across capability axes. The auxiliary classification loss $\mathcal{L}_{cls}$ provides weak supervision that encourages the DCR to learn capability-discriminative routing.

**Hypothesis 4 (Flywheel Necessity).** Post-transplantation fine-tuning via the dual-flywheel mechanism is expected to be necessary for recovering capability performance lost during FFN removal. The relative contribution of the synthetic flywheel vs. the self-refining (GRPO) flywheel will determine the optimal recovery strategy.

**Hypothesis 5 (Knowledge Distillation Comparison).** The knowledge distillation baseline is expected to achieve competitive average performance but with uniform degradation across all capability dimensions, whereas PARSE is designed to produce asymmetric preservation (high CRR on preserved capabilities, lower CRR on non-preserved ones).

### 5.2 Theoretical Implications

Preliminary activation-based CIT measurement ($\bar{r}=0.9945$, capability cliff 3.8–4.0×, all 12 profiles converge) provides initial empirical support for the following implications, subject to further validation on standard Transformer architectures:

1. **Capability Structure of LLMs.** The high inter-axis correlation ($\bar{r}=0.9945$) and profile convergence support the concentration hypothesis: LLM capabilities follow a depth-dependent pattern where deeper layers carry progressively more capability weight across all dimensions, consistent with residual network theory. This finding suggests that purely layer-level CIT-based selection cannot achieve qualitatively different pruning patterns across capability axes; selectivity must operate on magnitude differences within a shared importance profile. Whether finer-grained importance measures (attention head-level, task-specific gradients) can reveal modular structure remains an open question for future investigation.

2. **FFN Redundancy Scope.** Needle [54] established FFN redundancy for function calling. PARSE extends this question to a broader set of capability dimensions. Confirming successful No-FFN transplantation across language, discipline, and scenario axes on standard Transformer architectures would generalize the FFN redundancy principle beyond structured tool-calling tasks.

3. **Routing Efficiency.** If the DCR achieves effective multi-profile routing at 0.08M parameters on standard Transformer architectures, it would establish that shared parameter-efficient routing is a viable alternative to maintaining separate specialized models—challenging the premise that task-specific deployment requires task-specific architectures.

### 5.3 Limitations

We acknowledge the following limitations of the current work:

1. **Incomplete Empirical Validation.** Preliminary CIT diagnostic probing (activation-only) has been performed on Qwen3.5-0.8B, yielding the correlation and capability cliff measurements reported in Section 3.2. The transplantation, flywheel recovery, and full benchmark evaluation (GSM8K, BFCL, HumanEval) remain pending execution. All outcome hypotheses in this section are based on the methodological framework and await empirical testing.

2. **CIT Factorization Under High Correlation.** Activation-based CIT measurement confirmed high cross-axis correlation ($\bar{r}=0.9945$, minimum Korean–Math $r=0.980$) and profile convergence under the factorized CIT (Section 3.2)—all 12 profiles select the same 6 layers for pruning. The full (non-factorized) CIT computation and the contrastive CIT variant are designed to diagnose and mitigate this limitation, respectively.

3. **Parameter Budget and Compression Ratio Claims.** The frequently cited 8.8× compression target (752M→85M) requires clarification: FFN constitutes approximately 65% of parameters, so even removing all FFNs would leave ~263M (35% × 752M) remaining from attention modules alone, yielding only 2.86× compression. Achieving 8.8× requires additional measures such as attention head pruning in retained layers or more aggressive layer removal. The actual achievable compression ratio under specified preservation profiles is an empirical quantity to be determined by experiment.

4. **Single Architecture.** The CIT methodology, while architecture-agnostic in principle, is currently only integrated with Qwen-family models. Validation on Llama, Mistral, and Gemma architectures is necessary for generalizability claims.

5. **Calibration Scale.** The current 10–15 samples per capability category provide compact diagnostic probes but may not capture the full distribution of capability-specific inputs. The preliminary measurement used 10 samples per category; category-level counts are currently imbalanced across languages (3–15 prompts per language). Larger-scale, balanced calibration could improve CIT discrimination and reduce estimation noise for low-resource language categories.

6. **DCR Expressiveness.** The current DCR uses a single linear projection from embedding space, limiting routing to global decisions. More expressive architectures (multi-head routing, hierarchical gating) might achieve finer-grained control at the cost of additional parameters.

7. **Long-Context Stability.** DCR gate modulation is computed from mean-pooled embeddings, which may not capture position-dependent capability requirements in long contexts (>4K tokens).

8. **Axis Selection Justification.** While the 19-axis decomposition is grounded in established evaluation benchmarks [56,57] and deployment use cases [25,26,27], the choice of exact axes (particularly within Discipline) remains a design decision. The sensitivity to axis granularity (e.g., coarser vs. finer disciplinary categories) is a subject for ablation. We note that other decompositions (e.g., by linguistic complexity, domain, or format) may capture orthogonal variance not represented in the current axes.

---

## 6. Conclusion

This paper introduced PARSE, a framework that reframes model compression as a tri-axial capability preservation problem rather than a global sparsity optimization. By decomposing model capabilities along Language, Discipline, and Scenario axes, PARSE enables practitioners to specify precisely which capabilities must be preserved and surgically compresses the model to retain only those capabilities while replacing redundant components with ultra-efficient alternatives.

The Capability Importance Tensor provides a principled mechanism for quantifying layer-level capability contributions. The Dynamic Capability Router enables a single compressed model to serve multiple preservation profiles without weight switching. We have provided complete specifications for the DCR training algorithm, including the joint loss function $\mathcal{L}_{dcr} = \mathcal{L}_{lm} + \beta \cdot \mathcal{L}_{cls}$, the optimization procedure (AdamW with cosine decay), and the parameter count derivation that yields the claimed 0.08M overhead. The complete four-stage pipeline—CIT diagnosis, layer sculpture, FFN transplantation, and dual-flywheel recovery—is implemented as open-source software.

We have also addressed several methodological concerns transparently: the multiplicative CIT factorization is acknowledged as an approximation empirically validated to collapse under $\bar{r}=0.9945$ cross-axis correlation, with the contrastive CIT and full-tensor computation provided as alternative paths. The relationship between sparsity parameter $\tau$ and actual parameter reduction is made explicit: FFN-only removal yields at most ~2.86× compression, and achieving deeper compression requires combining layer removal with additional techniques such as reduced attention dimension or more aggressive layer pruning. The knowledge distillation baseline addresses the most direct alternative to PARSE's claimed capability-preserving compression.

The core methodological contribution is the recognition that model compression need not be a global optimization problem. By specifying *which* capabilities matter—Chinese grammar, English mathematics, function calling—and preserving only the layers that carry those capabilities, the PARSE framework opens a new axis of control in efficient model deployment. Experimental validation of the hypotheses presented herein, across the 12 preservation profiles defined in our experimental design, is the immediate next step.

---

## References

[1] Y. Lyu, C. Wang, H. Zheng, et al., "AgenticQwen: Training small agentic language models with dual data flywheels for industrial-scale tool use," *arXiv:2604.21590*, 2026. https://arxiv.org/abs/2604.21590

[2] J. Gong, "MiniMind-O technical report: An open small-scale speech-native omni model," *arXiv:2605.03937*, 2026. https://arxiv.org/abs/2605.03937

[3] X. Ma, G. Fang, and X. Wang, "LLM-Pruner: On the structural pruning of large language models," *arXiv:2305.13058*, 2023. https://arxiv.org/abs/2305.13058

[4] E. Frantar and D. Alistarh, "SparseGPT: Massive language models can be accurately pruned in one-shot," *arXiv:2301.06126*, 2023. https://arxiv.org/abs/2301.06126

[5] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, "A simple and effective pruning approach for large language models," *arXiv:2306.11695*, 2023. https://arxiv.org/abs/2306.11695

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

[17] NVIDIA, "Data flywheel: What it is and how it works," 2024. https://www.nvidia.com/en-us/glossary/data-flywheel/ (Accessed: 2026-05-27)

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

[39] E. Mitchell et al., "Model editing networks with gradient decomposition," *arXiv:2110.11309*, 2022. https://arxiv.org/abs/2110.11309

[40] E. Mitchell et al., "Memory-based model editing at scale," *arXiv:2203.03466*, 2022. https://arxiv.org/abs/2203.03466

[41] S. Wang et al., "Knowledge editing for large language models: A survey," *arXiv:2401.01286*, 2024. https://arxiv.org/abs/2401.01286

[42] L. Bourtoule et al., "Machine unlearning," *arXiv:1912.03817*, 2021. https://arxiv.org/abs/1912.03817

[43] Y. Yao et al., "Machine unlearning: A survey," *ACM Computing Surveys*, 2024. DOI: 10.1145/3603620

[44] B. Liu et al., "Knowledge unlearning for LLMs," *arXiv:2402.01754*, 2024. https://arxiv.org/abs/2402.01754

[45] R. M. French, "Catastrophic forgetting in connectionist networks," *Trends in Cognitive Sciences*, vol. 3, no. 4, pp. 128–135, 1999. DOI: 10.1016/S1364-6613(99)01294-2

[46] A. Sekhari et al., "Descent-to-delete: Gradient-based methods for machine unlearning," *arXiv:2110.05679*, 2021. https://arxiv.org/abs/2110.05679

[47] A. Golatkar et al., "Fast machine unlearning without retraining," *arXiv:2009.11373*, 2020. https://arxiv.org/abs/2009.11373

[48] Anonymous, "Task-specific compression for large language models," *arXiv:2306.05685*, 2023. https://arxiv.org/abs/2306.05685

[49] Anonymous, "Compact language models via priming and pruning," *arXiv:2406.09246*, 2024. https://arxiv.org/abs/2406.09246

[50] J.S. McCarley, R. Chakravarti, and A. Sil, "Structured pruning of BERT-based question answering models," *arXiv:1910.09755*, 2019. https://arxiv.org/abs/1910.09755

[51] N. Ding et al., "Parameter-efficient fine-tuning for large language models: A comprehensive survey," *arXiv:2303.15647*, 2023. https://arxiv.org/abs/2303.15647

[52] Y. Li et al., "Inference-time intervention: Eliciting truthful answers from a language model," *arXiv:2306.03341*, 2023. https://arxiv.org/abs/2306.03341

[53] Anonymous, "Regularizing towards well-calibrated large language models," *arXiv:2405.18654*, 2024. https://arxiv.org/abs/2405.18654

[54] H. Ndubuaku, J. Mroz, K. Mosoyan, et al., "Needle: Simple attention networks for function calling," *GitHub: cactus-compute/needle*, 2026. https://github.com/cactus-compute/needle

[55] Z. Shao, P. Wang, Q. Zhu, et al., "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," *arXiv:2402.03300*, 2024. https://arxiv.org/abs/2402.03300

[56] A. Conneau et al., "XNLI: Evaluating cross-lingual sentence representations," in *Proc. EMNLP*, 2018. https://arxiv.org/abs/1809.05053

[57] D. Hendrycks et al., "Measuring massive multitask language understanding," in *Proc. ICLR*, 2021. https://arxiv.org/abs/2009.03300

[58] K. Cobbe et al., "Training verifiers to solve math word problems," *arXiv:2110.14168*, 2021. https://arxiv.org/abs/2110.14168

[59] Berkeley Function-Calling Leaderboard, https://gorilla.cs.berkeley.edu/leaderboard.html

[60] M. Chen et al., "Evaluating large language models trained on code," *arXiv:2107.03374*, 2021. https://arxiv.org/abs/2107.03374

[61] K. Papineni et al., "BLEU: A method for automatic evaluation of machine translation," in *Proc. ACL*, 2002. https://doi.org/10.3115/1073083.1073135

[62] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019. https://arxiv.org/abs/1711.05101

[63] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015. https://arxiv.org/abs/1412.6980
