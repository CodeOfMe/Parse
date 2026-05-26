# Literature Review: Foundational Studies for PARSE

> Sequential analysis of 55 foundational works in structural pruning, knowledge editing, machine unlearning, data flywheels, agentic systems, GRPO-based RL, and specialized architectures. Citations follow paper order.

---

## 1. Data Flywheels and Self-Improving Training [1, 17–23]

### [1] AgenticQwen — Dual Data Flywheels for Small Model Training

Lyu et al. propose a dual data flywheel architecture for training small agent-capable models at industrial scale. The **Synthetic Flywheel** generates reasoning traces via Self-Instruct expansion and Persona injection, while the **Environment Interaction Flywheel** produces agent-environment experience data. Together, they bootstrap small models to competitive agentic performance without large-model distillation.

**Relevance to PARSE**: The dual-flywheel mechanism is directly adopted in PARSE's Stage 4 (Recovery), where synthetic + self-refining flywheels generate calibration data for each (Language × Discipline × Scenario) combination after FFN transplantation.

### [17] NVIDIA — Data Flywheel Fundamentals

NVIDIA's conceptual framework defines the data flywheel as a virtuous cycle: model outputs improve training data, improved data trains better models, better models produce better outputs. This recursive self-improvement loop underpins modern small-model training pipelines.

### [18] ArenaLearning — Simulated Chatbot Arena

Luo et al. pioneer AI-driven simulated arenas for automated model evaluation and training data generation. Models compete in pairwise matchups judged by LLM evaluators, generating preference data at scale without human annotation.

**Relevance to PARSE**: ArenaLearning's automated evaluation strategy inspires the critic model scoring in PARSE's self-refining flywheel.

### [19] SRDF — Self-Refining Data Flywheel

Wang et al. introduce a generator-navigator pair that collaboratively produces increasingly curated training trajectories. The generator proposes sequences; the navigator critiques and refines them through iterative feedback loops.

### [20] IFDecorator — Instruction Following with Verifiable Rewards

Guo et al. wrap instruction-following training with verifiable reward signals, enabling RL-based optimization where reward functions are grounded in measurable task completion rather than learned preference models.

### [21] UI-TARS-2 — GUI Agent with Multi-Turn RL

Wang et al. advance GUI agent training through multi-turn reinforcement learning, demonstrating that agentic capabilities benefit from iterative environment interaction rather than static supervised datasets.

### [22] GAIA — GUI Test-Time Scaling Critic

Wang et al. build a data flywheel system for training GUI test-time scaling critic models. The critic evaluates generated GUI interaction trajectories and filters for quality, creating a self-improving evaluation-training loop.

**Relevance to PARSE**: The GAIA critic model design directly informs PARSE's self-refining flywheel quality scoring mechanism.

### [23] SynthAgent (DeepSeekMath) — Synthetic Environments for Agent Skills

Lyu et al. propose training agentic language models through synthetic tasks in mock environments. This work also introduces DeepSeekMath's GRPO algorithm—see Group 5 below. The key insight is that procedurally generated environments can produce unlimited, diverse training data for agent skill acquisition.

---

## 2. Specialized Efficient Architectures [2, 54, 55]

### [2] MiniMind-O — Omni-Modal Model at 0.1B Parameters

Gong proposes the Thinker-Talker dual-path decoupling architecture. The **Thinker** path handles semantic understanding (text, vision); the **Talker** path handles speech generation. Representations pass through a middle-layer bridge rather than the embedding or final layer. Multi-Token Prediction (MTP) simultaneously generates 8-layer Mimi audio codebooks, achieving tri-modal processing at only 0.1B trainable parameters.

**Relevance to PARSE**: The Thinker-Talker bridge layer concept inspires the Dynamic Capability Router's design—representations from language, discipline, and scenario axes converge through modulated gates rather than rigid architectural paths. The extreme parameter efficiency demonstrates that architectural innovation can compensate for model shrinkage.

### [54] Language Model Unlearning

This work formalizes LLM unlearning as a distribution shift problem, providing the theoretical foundation for targeted knowledge removal. PARSE leverages this principle when identifying layers that can be safely replaced: if facts in those layers can be removed without affecting other capabilities, then the layers themselves can be replaced.

### [55] Needle — Attention-Only Networks for Function Calling

Ndubuaku et al. make a radical architectural claim: for structured tasks like function calling, the FFN—which constitutes approximately 65% of standard Transformer parameters—is entirely redundant. The softmax operation in attention already provides sufficient nonlinearity for information routing. The resulting architecture (pure attention + gated residuals + ZCRMSNorm) achieves 26M parameters that outperform 270M-600M general-purpose models.

**Relevance to PARSE**: This is the foundational reference for PARSE's FFN transplantation strategy. Needle proves that FFNs are not universally necessary; PARSE extends this insight by making FFN removal *capability-dependent* rather than universal. Layers critical for reasoning retain their FFNs; layers redundant for the preservation profile receive Needle-style pure-attention replacements.

---

## 3. Structural Pruning [3–16]

### [3] LLM-Pruner — Gradient-Based Coupled Structure Pruning

Ma et al. pioneer task-agnostic structural pruning for LLMs. The core technique is gradient-based coupling: parameter groups with correlated gradients are identified as functional units and pruned or retained together. This preserves the internal coherence of attention mechanisms (K, V coupling) during compression.

**Relevance to PARSE**: LLM-Pruner introduces the idea that parameters form functional groups. PARSE extends this from parameter coupling to *capability coupling*—layers form functional groups based on the capabilities they support, not just gradient correlation.

**Limitation**: Global sparsity optimization without capability-specific awareness. The same pruning mask is applied uniformly across all tasks.

### [4] SparseGPT — One-Shot Second-Order Pruning

Frantar and Alistarh achieve 50% unstructured sparsity in a single pass through efficient second-order Hessian approximations. The key insight: pruning-induced error can be compensated by updating remaining weights via the inverse Hessian, computed recursively row-by-row without materializing the full matrix.

**Relevance to PARSE**: SparseGPT demonstrates that importance-aware weight compensation is feasible at scale. PARSE adapts this principle to a tri-axial importance space: instead of compensating *globally*, compensate only for the capabilities the user wants to preserve.

**Limitation**: One-shot nature makes it brittle—there is no mechanism to recover if the initial pruning decision is wrong for a specific capability dimension.

### [5] Wanda — Weight-Activation Product Pruning

Sun et al. simplify magnitude pruning by multiplying weight magnitude with input activation norm. The resulting score requires only a single forward pass, yet matches SparseGPT's performance on LLaMA-family models. The elegance lies in capturing both static weight importance and dynamic input sensitivity.

**Relevance to PARSE**: Wanda's activation-dependent importance scoring is conceptually extended in PARSE's Capability Importance Tensor. While Wanda uses *input* activations (which vary per sample), PARSE uses *capability-specific* activations and gradients to compute importance scores that vary per (Language × Discipline × Scenario) combination.

**Limitation**: While activation-aware, Wanda remains capability-agnostic. It cannot distinguish between activation patterns driven by Chinese syntax versus those driven by mathematical reasoning—both are simply "activation."

### [6] LaCo — Layer Collapse Pruning

Yang et al. propose pruning from the representation stability perspective. When consecutive layers produce highly similar outputs (measured via cosine similarity or BERTScore), intermediate layers can be collapsed without information loss. This shifts the pruning decision from "which parameters are small?" to "which layers are redundant given their neighbors?"

**Relevance to PARSE**: LaCo's representation-centric view aligns with PARSE's CIT computation. Both analyze layer *outputs* rather than layer *weights*. PARSE refines this by asking: representation similarity for *which capability*? Two layers may produce identical outputs for translation but divergent outputs for logic.

### [7] ShortGPT — The Layer Redundancy Discovery

Men et al. provide the definitive empirical evidence that LLM layers are massively redundant. Beyond showing that 25%+ of layers can be removed with minimal average perplexity impact, ShortGPT reveals a crucial finding: layer removal has wildly *uneven* effects across capabilities. Deep layers contribute disproportionately to reasoning; shallow layers handle syntax.

**Relevance to PARSE**: This is the empirical foundation for PARSE's entire approach. ShortGPT proves that "important" has no single answer—a layer's importance depends entirely on which capability you are measuring. PARSE systematizes this observation into the tri-axial Capability Importance Tensor.

### [8] LayerDrop — Structured Dropout for Depth Reduction

Fan et al. introduce training-time structured dropout where entire layers are randomly dropped. This produces models robust to arbitrary depth at inference and enables dynamic depth selection per input.

**Relevance to PARSE**: LayerDrop's core idea—not all layers are needed for all inputs—is extended in PARSE from *input complexity* to *capability specificity*. A Chinese input may need different layers than a math problem, even if both have the same token count.

### [9] DeeBERT — Confidence-Based Early Exiting

Xin et al. attach lightweight classifiers to intermediate BERT layers. When a classifier's confidence exceeds a threshold, the model exits early, bypassing deeper layers. Simple samples require fewer layers; hard samples use the full depth.

**Relevance to PARSE**: DeeBERT demonstrates that inference-time layer selection is viable. PARSE generalizes this from sample difficulty to capability type, using the DCR to dynamically select effective depth based on detected task context.

### [10] FastBERT — Self-Distilling Adaptive Inference

Liu et al. combine early exiting with self-distillation: the full-depth model (teacher) distills its knowledge into shallower exits (students). This ensures that early-exit predictions maintain quality without requiring separate training.

### [11] oBERT — Optimal BERT Pruning via Second-Order Methods

Kurtic et al. apply iterative magnitude pruning with weight rewinding to achieve industry-leading compression ratios on BERT-scale models. The "optimal" in the title refers to the theoretically grounded second-order pruning criterion that minimizes the local quadratic approximation of the loss.

### [12] Movement Pruning — Adaptive Sparsity via Fine-Tuning

Sanh et al. propose learning which weights to prune during fine-tuning, using first-order gradient information to determine importance. Unlike magnitude pruning (static), movement pruning adapts to the fine-tuning objective, producing sparsity patterns that are task-aware.

**Relevance to PARSE**: Movement pruning is the conceptual precursor to capability-aware pruning. By making sparsity task-dependent, it demonstrates that "less" can mean "better for a specific purpose." PARSE generalizes this from single-task awareness to multi-dimensional capability awareness.

### [13] BERT-of-Theseus — Progressive Module Replacement

Xu et al. model compression as gradual module replacement. Smaller replacement modules are progressively swapped in for original modules. The key insight is that *training stability* matters: aggressive one-shot compression causes catastrophic forgetting, while gradual replacement allows the model to adapt.

**Relevance to PARSE**: PARSE's transplantation strategy follows the same principle. FFN modules are replaced rather than simply removed, and the DCR gate perturbations are initialized to zero to ensure the model starts from a stable state before adapting.

### [14] TinyLlama — Small Model Pretraining at Scale

Zhang et al. demonstrate that a 1.1B model pretrained on 1 trillion tokens can achieve competitive performance, establishing that small architectures are not inherently limited—they simply need sufficient training data.

**Relevance to PARSE**: TinyLlama provides the upper-bound reference: this is what a well-trained small model can achieve. PARSE's goal is to approach this upper bound through intelligent compression of a larger pretrained model, rather than training from scratch.

### [15] MInference — Dynamic Sparse Attention for Long Contexts

Jiang et al. accelerate long-context LLM pre-filling through dynamic sparse attention patterns. By identifying and exploiting sparsity in attention matrices, they reduce the quadratic complexity bottleneck for long sequences.

### [16] LLM-Shearing — Structured Pruning with Continued Pretraining

Xia et al. propose flexible structured pruning followed by continued pretraining for capability recovery. The two-stage approach decouples the pruning decision (what to remove) from the recovery process (how to compensate), allowing more aggressive initial pruning.

**Relevance to PARSE**: The prune-then-recover paradigm directly corresponds to PARSE's Stage 3 (Transplantation) + Stage 4 (Recovery) pipeline. PARSE replaces the generic continued pretraining with targeted capability-specific flywheel recovery.

---

## 4. Agentic Systems and Tool Calling [24–30]

### [24] Gorilla — LLM Connected with Massive APIs

Patil et al. pioneer LLM-based API calling by fine-tuning LLaMA models on a large corpus of API documentation. Retrieval augmentation during inference provides the model with up-to-date API signatures at test time.

**Relevance to PARSE**: Gorilla demonstrates that tool-use is a *trainable skill* rather than an emergent property of scale—a key premise for PARSE's claim that function calling capability can be preserved in small compressed models.

### [25] xLAM — Family of Large Action Models

Zhang et al. scale the API-calling paradigm to a family of models from 1B to 8×22B parameters, achieving top performance on the Berkeley Function-Calling Leaderboard across all size categories.

### [26] TinyAgent — Function Calling at the Edge

Erdogan et al. deploy 1.1B and 7B function-calling models on consumer laptops (MacBook), surpassing GPT-4-Turbo on function calling benchmarks. This directly validates PARSE's target deployment scenario.

### [27] ToolFlow — Graph-Based Tool Sampling

Wang et al. achieve GPT-4-level tool calling on LLaMA-3.1-8B using only 8,000 high-quality dialogue samples, demonstrating that sample efficiency—not data volume—is the bottleneck for tool-use capability.

**Relevance to PARSE**: ToolFlow's sample efficiency mirrors PARSE's CIT approach: understanding *which* examples matter for *which* capability is more important than training on massive unfiltered datasets.

### [28] SLM Agentic Systems Survey

Sharma and Mehta provide the first systematic survey of small language model agent architectures, capabilities, and training approaches, establishing evaluation baselines and identifying capability gaps.

### [29] TinyLLM — Edge Device Agent Evaluation

Haque et al. benchmark small language models on agentic tasks deployed on edge devices, providing hardware-aware evaluation metrics beyond abstract accuracy scores.

### [30] CAMPHOR — Collaborative Multi-Agent Architecture

Fu et al. propose a hierarchical multi-agent architecture for on-device deployment. Reasoning agents decompose complex tasks; expert agents execute subtasks; a coordinator manages inter-agent communication.

**Relevance to PARSE**: CAMPHOR's hierarchical decomposition mirrors PARSE's tri-axial decomposition: complex capabilities break down into orthogonal dimensions that can be handled independently.

---

## 5. GRPO-Based Reinforcement Learning [23, 31–36]

### [23] DeepSeekMath / SynthAgent — GRPO Algorithm

The DeepSeekMath team introduces Group Relative Policy Optimization (GRPO). Instead of training a separate value network (critic), GRPO samples multiple outputs per prompt and estimates advantages through within-group relative comparison. This eliminates the critic model entirely, reducing the RL pipeline from four models to three.

**Relevance to PARSE**: GRPO's sample efficiency and critic-free design make it ideal for PARSE's post-transplantation recovery stage, where compute budget is limited and capability-specific reward functions are needed.

### [31] SLM-ToolUse-GRPO — GRPO for Tool Calling

Paprunia et al. specifically adapt GRPO for small language model tool-use enhancement. Reward functions target JSON structure validity, tool selection accuracy, and parameter precision—three orthogonal dimensions of function calling quality.

### [32] EBPO — Empirical Bayes Stabilization of GRPO

Han et al. address GRPO's estimation variance in small group settings through empirical Bayes shrinkage estimators. The shrinkage regularizes noisy within-group baselines toward a global prior, maintaining training stability when group sizes are limited.

### [33] STAPO — Silencing Spurious Tokens in GRPO

Liu et al. discover the "spurious token" phenomenon: approximately 0.01% of tokens contribute negligibly to the objective yet receive disproportionately large gradient updates, causing training instability. Their solution silences these tokens during gradient computation.

**Relevance to PARSE**: PARSE's transplantation involves many layers with near-zero gate values—exactly the condition where spurious tokens could disrupt training. STAPO's silencing mechanism provides a direct technical solution.

### [34] Mu-GRPO — Off-Policy Efficiency

Tian et al. demonstrate that GRPO tolerates far more rollout delay than previously expected. By decoupling rollout generation from model updates, they achieve approximately 2× training speedup without quality degradation.

### [35] ActFocus — Resolving the Action Bottleneck

He et al. identify that in agent trajectories, gradient signals concentrate on a small number of action tokens (e.g., tool call arguments). Token-level energy reweighting redistributes the learning signal across the full trajectory, improving terminal performance by over 60 percentage points.

### [36] ChemCRAFT — Agentic RL in Professional Domains

Li et al. apply agentic reinforcement learning to chemical language models for drug design. Domain-specific reward engineering enables small models to surpass cloud-scale models in specialized scientific tasks.

---

## 6. Knowledge Editing [37–41]

### [37] ROME — Rank-One Model Editing

Meng et al. pioneer causal tracing for factual knowledge in GPT-style models. By analyzing activations at each layer during fact retrieval, they identify the specific MLP layers where factual associations are stored. Editing is performed via closed-form rank-one weight updates that modify behavior on target facts while preserving behavior on unrelated facts.

**Relevance to PARSE**: ROME is the foundational reference for PARSE's core premise: LLM knowledge is localizable to specific layers. If a single fact can be traced to Layer 18, MLP activations 245–267, then an entire capability (e.g., mathematical reasoning) can similarly be traced to a subset of layers. This is the theoretical link between knowledge editing and capability-preserving compression.

### [38] MEMIT — Mass Editing Memory in a Transformer

Meng et al. extend ROME to batch editing. Thousands of facts can be modified simultaneously through a constrained optimization that prevents edit interference. The key technical contribution is solving the "many-at-once" problem while maintaining specificity.

**Relevance to PARSE**: MEMIT demonstrates that multiple knowledge modifications can coexist without cross-interference—directly analogous to PARSE's multi-capability preservation, where preserving Chinese grammar must not interfere with preserving English mathematics.

### [39] MEND — Model Editing Networks with Gradient Decomposition

Mitchell et al. train hyper-networks to predict parameter updates for efficient editing. By decomposing the gradient into low-rank components, MEND learns to generate optimal weight perturbations from the edit request itself, amortizing computation across edits.

### [40] SERAC — Memory-Based Model Editing at Scale

Mitchell et al. propose an alternative to direct weight modification: a scope classifier routes queries to either the base model or a counterfactual memory cache. This completely decouples editing from model parameters.

**Relevance to PARSE**: SERAC's separation of "what the model knows" from "what we want it to output" parallels PARSE's separation of "what capabilities the model has" from "what capabilities we want to preserve." The routing mechanism is a conceptual precursor to the DCR.

### [41] Knowledge Editing Survey

Wang et al. provide a comprehensive systematization of knowledge editing methods, categorizing approaches by their modification target (weights vs. external memory), edit scope (single vs. batch), and preservation guarantees.

---

## 7. Machine Unlearning [42–47]

### [42] SISA — Sharded, Isolated, Sliced, Aggregated

Bourtoule et al. propose the foundational unlearning framework. Training data is partitioned into shards; independent sub-models train on each shard; unlearning requests only require retraining the affected shard's sub-model. This provides both efficiency and auditability.

**Relevance to PARSE**: SISA's sharding principle demonstrates that localized changes (one shard) need not affect global performance (other shards)—directly analogous to PARSE's claim that removing layers for one capability need not affect other capabilities.

### [43] Machine Unlearning Survey

Yao et al. provide a comprehensive survey of unlearning methods across domains, establishing evaluation metrics and categorizing approaches by their compliance guarantee level (exact, approximate, probabilistic).

### [44] Knowledge Unlearning for LLMs

Liu et al. focus specifically on unlearning in the language model context, addressing the unique challenges of unbounded output spaces and the difficulty of defining "forgetting" for generative models.

### [45] Catastrophic Forgetting

French's foundational work on catastrophic forgetting in neural networks. While primarily a challenge for continual learning, it also provides insight into *why* aggressive pruning degrades capabilities: removing parameters shifts the loss landscape in ways that disproportionately affect well-learned but brittle representations.

**Relevance to PARSE**: Understanding catastrophic forgetting mechanisms is essential for designing PARSE's recovery stage. The transplantation must compensate for the forgetting induced by FFN removal through targeted capability-specific fine-tuning.

### [46] Descent-to-Delete — Gradient-Based Unlearning

Sekhari et al. formalize unlearning as gradient *ascent* on target data with KL divergence constraints. The theoretical framework provides deletion guarantees while the practical implementation enables efficient approximate unlearning.

### [47] Fast Machine Unlearning Without Retraining

Golatkar et al. approximate the unlearning update through Fisher information matrix analysis of parameter sensitivity to specific data points. By computing the "optimal forgetting direction" analytically, they eliminate the need for retraining.

**Relevance to PARSE**: The Fisher-based approach to identifying "which parameters matter for what data" directly corresponds to PARSE's CIT computation—which identifies "which layers matter for what capability" using both activation and gradient signals.

---

## 8. Task-Specific Compression and Efficiency [48–51]

### [48] Task-Specific Compression for LLMs

This work proposes that compression should be guided by the target task, not by global optimization criteria. Task-relevant structures are preserved; task-irrelevant structures are aggressively compressed.

**Relevance to PARSE**: This is a direct predecessor to PARSE's capability-specific compression. PARSE extends the concept from single-task specificity to multi-dimensional (Language × Discipline × Scenario) capability awareness.

### [49] Compact Language Models via Priming and Pruning

This work addresses the complementary problem of how much architectural capacity is needed for specific language understanding tasks, proposing priming-based initialization before pruning.

### [50] Structured Pruning of BERT-Based QA Models

McCarley et al. provide early evidence that structured pruning for specific downstream tasks (question answering) can achieve substantial compression without catastrophic accuracy loss, establishing the task-specific pruning paradigm.

### [51] Parameter-Efficient Fine-Tuning Survey

Ding et al. comprehensively survey PEFT methods: adapters, prefix tuning, LoRA, and prompt tuning. These techniques modify only a small fraction of parameters during fine-tuning, enabling efficient adaptation.

**Relevance to PARSE**: PARSE's DCR (0.08M parameters, 0.09% of model size) is effectively a PEFT module—a tiny trainable component that modulates a frozen base architecture. The DCR is to capability routing what LoRA is to task adaptation.

---

## 9. Training Stability and Intervention [52, 53]

### [52] Inference-Time Intervention — Truthful Answer Elicitation

Li et al. demonstrate that activating specific attention head patterns at inference time can shift model behavior toward truthfulness without any weight modification. This establishes that model behavior can be controlled through activation-space intervention.

**Relevance to PARSE**: PARSE's DCR gate modulation—dynamically scaling transplanted block contributions based on input context—is a form of inference-time intervention. The principle is the same: control behavior through activation modulation rather than weight modification.

### [53] Regularizing Toward Well-Calibrated LLMs

This work investigates calibration-preserving regularization during fine-tuning, ensuring that compressed or adapted models maintain reliable uncertainty estimates rather than becoming overconfident on preserved capabilities and underconfident on others.

---

## Summary: Citation Structure

| Group | References | Papers | Core Theme |
|:---|:---|:---:|:---|
| Data Flywheels | [1, 17–23] | 8 | Self-improving training data generation |
| Specialized Architectures | [2, 54, 55] | 3 | Extreme parameter efficiency |
| Structural Pruning | [3–16] | 14 | Layer/parameter importance and removal |
| Agentic Systems | [24–30] | 7 | Small-model tool-use capabilities |
| GRPO-Based RL | [23, 31–36] | 7 | Stable small-model RL training |
| Knowledge Editing | [37–41] | 5 | Parameter-space knowledge localization |
| Machine Unlearning | [42–47] | 6 | Targeted knowledge removal |
| Task-Specific Compression | [48–51] | 4 | Task-aware efficiency |
| Training Stability | [52, 53] | 2 | Calibration and intervention |
| **Total** | | **55** | |
