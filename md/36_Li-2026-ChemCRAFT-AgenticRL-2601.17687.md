# Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis

> **arXiv:** [2601.17687](https://arxiv.org/abs/2601.17687)
> **TeX source:** [arXiv-2601.17687v1/](arXiv-2601.17687v1/)
> **PDF:** [ChemCRAFT-arXiv-2601.17687v1.pdf](ChemCRAFT-arXiv-2601.17687v1.pdf)

---

% — BEGIN Content/2introduction —

## Introduction

Chemical Language Models (CLMs) have emerged as transformative tools in accelerating drug discovery and materials science [han2025generalist, zhang2024scientific, xia2023systematic,lv2025navigate], demonstrating remarkable potential in tasks ranging from molecular property prediction [molformer, chemberta, chemberta2] and de novo design [grisoni2023chemical, bhattacharya2024large, bagal2021molgpt,lv2025prollama] to retrosynthesis planning [wan2022retroformer, liu2024multimodal, ma2025automated] and reaction condition recommendation [qian2023predictive, zhang2025large, cao-etal-2024-presto]. Currently, the development of CLMs primarily follows two distinct paradigms. The first approach involves supervised fine-tuning (SFT), or the continued pre-training of large language models on domain-specific corpora [taylor2022galactica, edwards2022translation, zhang2024chemllm, zhao2024chemdfm, xia2025nature]. While these models achieve competitive performance on in-distribution metrics [edwards2022translation, pei2023biot5, zhuang2024instructbiomol], they fundamentally suffer from rigid optimization objectives. By forcing the model to minimize loss on "Task-Query-Direct Answer" triplets, current SFT paradigms compel the LLM to learn a shallow mapping from chemical inputs directly to numerical values or labels [Ding2024BreakTCA, li2025detect, Rueda2025UnderstandingLSA]. Critically, this approach bypasses the "expert-like" reasoning process—such as structural analysis, intermediate hypothesis generation, and logical verification—that is essential for scientific discovery [zhang2025exploring, truhn2023large, shojaee2024llm]. Consequently, this leads to a degradation of the model's intrinsic general capabilities, often referred to as catastrophic forgetting [luo2025empirical, liu2024more, zheng2025towards], and limits its adaptability across disparate chemical tasks [toniato2023fast, zhao2023scientific, ganeeva2024lost].

To bridge this cognitive gap, recent state-of-the-art works have introduced the concept of "Chemical Reasoning" [ouyang2023structured, jang2025structural, zhao2025chemdfm, bran2025mist, zhao2025molreasoner, li2025mol, zhuang2025reasoning, wang2025chem, narayanan2025training]. By utilizing distillation from superior teacher models [zhao2025molreasoner, zhuang2025reasoning, wang2025chem] and Reinforcement Learning (RL) based post-training, these approaches aim to guide models in mimicking the step-by-step, coherent workflows of human experts. 
While enhancing interpretability, this paradigm introduces a critical bottleneck. We find that the reasoning process is frequently dominated by token-intensive, low-level tasks—such as valency checks and structural parsing [alampara2025general, mswahili2024transformer]—rather than high-level derivation. This strategy is intrinsically inefficient. Given the probabilistic nature of LLMs, utilizing them for deterministic rote calculations is error-prone [bran2023chemcrow, m2024augmenting, mirza2025framework]. More importantly, it diverts the model's limited context window away from complex chemical reasoning toward mere syntactic validation [yao2022react, liu-etal-2024-lost].

Alternatively, Multi-Agent Systems (MAS) have attempted to mitigate this by offloading tasks to external tools via commercial LLM APIs [bran2023chemcrow, m2024augmenting, boiko2023autonomous, song2025multiagent] (e.g., GPT-4). However, this introduces severe deployment bottlenecks: the prohibitive token costs for large-scale screening and the unacceptable privacy risks associated with transmitting proprietary molecular structures to cloud servers [feretzakis2024privacy, guo2025survey, muegge2024perspectives]. To resolve these challenges, we present , a next-generation chemical language model framework designed to democratize high-performance, privacy-preserving intelligence in chemical research. Moving beyond the "internalize-everything" dogma, we establish a "cognitive decoupling" architecture inspired by the human scientific workflow [kahneman2011thinking, evans2003two]. In this paradigm,  functions as the central scientific reasoner, formulating hypotheses and orchestrating a comprehensive chemical sandbox of external tools. This synergy enables the model to solve complex problems within a joint language-tool space [schick2023toolformer, mialon2023augmented], ensuring that verified tools maintain scientific rigor while the model's parameters are dedicated to high-level planning and logic.

To enable the training of such a system, we constructed and open-sourced a large-scale dataset of expert-level tool-use trajectories. Leveraging this data, our work challenges the prevailing assumption that complex tool orchestration is an emergent ability exclusive to massive, proprietary LLMs (>100B parameters) [wei2022emergent]. We demonstrate that a compact 7B-14B parameter model, when optimized via supervised fine-tuning and domain-specific reinforcement learning, can achieve tool-use capabilities comparable to commercial APIs. By aligning the model's policy with the "Hypothesis-Action-Observation" [yao2022react] scientific loop, we successfully elicit robust reasoning and self-correction in smaller architectures. This breakthrough effectively resolves the deployment trilemma of cost, performance, and privacy, enabling every laboratory to host a secure, expert-level AI chemistry copilot locally.

To rigorously validate the application potential of our framework across diverse chemical domains, we employ ChemCoTBench [li2026chemcotbench], a comprehensive evaluation suite designed to assess the multi-step reasoning and tool-use capabilities of chemical agents. Covering a spectrum of nine critical tasks—ranging from fundamental molecular structure understanding and editing to molecular property optimization and reaction-related tasks—this benchmark provides a holistic view of an agent's proficiency. Extensive evaluations reveal that  achieves a new state-of-the-art among open-source models. Remarkably, despite its compact parameter size, it demonstrates problem-solving capabilities that not only significantly surpass the baselines of similar scales but also rival, and in specific reasoning-intensive tasks, exceed those of leading commercial LLM APIs.

% — END Content/2introduction —

% — BEGIN Content/3results —

## Results

**Figure:** *
 Overview of the data-curation pipeline and the training pipeline of our  models to a rigorous assessment across fundamental molecular understanding, property-guided optimization, and reaction prediction, benchmarking them against premier reasoning-enhanced LLMs and domain-specific baselines. Our empirical analysis reveals that by offloading precise computation to a specialized sandbox, smaller agentic models can surmount the inherent trade-off between semantic reasoning and structural precision, achieving expert-level proficiency that rivals or eclipses proprietary frontier models in rigorous scientific tasks.

**Table:** ***Performance Comparison for Molecule Understanding and Molecule Editing.** We propose the experimental results on our  achieves near-perfect precision. For example, in the Function-Group-Detection case (Table §tab:molund_edit, bottom-left), where the model must isolate and count amino groups in a complex polycyclic architecture,  achieves an MAE of 0.03, significantly outperforming standard instruction models like Qwen2.5-32B (MAE 0.36). This perceptual acuity extends to hierarchical structural reasoning; in the Ring-System-Detection task (Table 1, bottom-middle), which requires determining the existence of specific fused rings, our model attains 100% accuracy compared to 87.5% for Gemini-2.5-Pro [comanici2025gemini25].

Crucially, this understanding translates into actionable Molecule Editing capabilities—operations (Add, Delete, Substitute) that serve as the fundamental "arithmetic" of chemical design. In the Molecule-Edit-Substitute visualization (Table §tab:molund_edit, bottom-right), the challenge involves precisely targeting a hydroxyl group for removal and attaching a carboxyl group at the exact locus without disrupting the surrounding scaffold.  leverages its sandbox tools to validate valency and connectivity, achieving a 95.0% success rate in deletion tasks and a 97% SMILES equivalence score. This confirms that  has transcended probabilistic token prediction, effectively grounding its reasoning in verifiable chemical rules to perform rigorous structural modifications.

**Figure:** *
 **Experimental Analysis for Molecular Optimization and Reaction Predictions**. **a** Distribution analysis for property improvements on LogP, QED, Solubility. **b** Optimized molecule visualizations for protein-activated properties, including DRD2, JNK3, GSK-3$$. **c** Comparison between  (red distribution) consistently shifts the optimization trajectory toward positive gains. For instance, in Solubility optimization,  achieves a mean improvement of $=1.58$, nearly quadrupling the performance of the baseline Qwen model ($=0.42$) and surpassing commercial reasoning models like Gemini-2.5-Pro ($=1.38$).

This statistical success translates into chemically meaningful design strategies, as visualized in the drug-target case studies in Figure §fig:model_validationb. Here, the model effectively deploys the modular "Add/Delete/Substitute" operations validated in our Editing tasks to solve complex biological constraints. In the GSK-3$$ case (middle), the model strategically substitutes an ester group with an amine ($=0.26$). This is a non-trivial bioisosteric replacement that significantly alters the hydrogen bond donor/acceptor profile to fit the binding pocket, demonstrating that the model is not merely randomizing atoms but applying medicinal chemistry logic. In the DRD-2 case (left), it executes a precise scaffold-hopping maneuver, replacing a flexible amide linker with a methyl group ($=0.84$) to potentially lock the conformation.
Crucially, these optimizations are achieved with high sample efficiency. Unlike traditional genetic algorithms that may require thousands of oracle calls (property evaluations),  reaches these high-value candidates within minimal interaction steps. This efficiency suggests that our agent does not rely on brute-force search, but rather enables a "Hypothesis-Action-Verification" loop that is practical for real-world campaigns where wet-lab feedback is costly.

**Predicting Complex Reaction Outcomes..** 

To evaluate chemical intuition at the systemic level, we assessed models on three pillars of reaction prediction: (1) Forward Prediction, encompassing both Major Products and, crucially, By-Products (vital for assessing impurity profiles and purification risks); (2) Retrosynthesis (w/o reaction type), testing the ability to deconstruct targets into available precursors; and (3) Condition Recommendation, requiring precise suggestions for reagents, solvents, and catalysts. We benchmarked against domain-specialized models (e.g., Chemformer [irwin2022chemformer], GraphRetro [somnath2021learning], RetroPrime [wang2021retroprime]) and general reasoning LLMs (e.g., GPT-o3, DeepSeek-R1 [guo2025deepseek]).

As shown in Figure §fig:model_validationc, while specialized models like Chemformer achieve competitive accuracy on Major Product prediction due to extensive pre-training on USPTO data, they falter significantly in By-Product Prediction and Retrosynthesis on out-of-distribution samples. We attribute this to architectural rigidity: sequence-based and graph-based expert models are typically optimized for single-outcome generation and often rely on precise atom-mapping, a dependency that limits their generalization to noisy or unmapped real-world inputs. In contrast,  achieves a top-1 accuracy of over 50% in by-product prediction and dominates the retrosynthesis task (40% + vs < 20% for baselines). Furthermore, in Condition Recommendation (Figure §fig:model_validationd), the violin plots reveal that while general LLMs suffer from recommending convincing reaction conditions (distributions centered near 0.0), our  produces highly concentrated predictions aligned with ground truth, particularly for complex catalysts.

The model’s superior grasp of reactivity is further evidenced in the case studies (Figure §fig:model_validatione). We selected challenging modern reactions involving non-polar bond activation (C-C, C-H, C-F), transformations that are underrepresented in standard patent data (USPTO) and typically confound expert models. In the Ruthenium-catalyzed C-C coupling case [CCBond2021] (top), Gemini-2.5-Pro misses the competitive advantage of C-O activation over remote C-H activation. In the MnO$_2$ oxidation case [knochel2013amide] (middle), it underestimates the oxidation depth, predicting an amide rather than the correct carbonyl product. In the Cobalt-catalyzed C-F activation [wei2017cobalt] (bottom), the baseline fails to recognize that the specific additives are designed to suppress the benzyne pathway, leading to a regio-isomer error.  correctly predicts these outcomes not by mere intuition, but by evidence-based reasoning: its sandbox retrieves analogous reaction templates from the library, allowing the agent to ground its mechanism in empirical precedent rather than hallucinating plausible but incorrect pathways.

**Figure:** *
 **a** Performance on general reasoning benchmarks. Our  transcends mere domain data exposure, stemming instead from a cognitive decoupling architecture refined via reinforcement learning. To rigorously verify this hypothesis, we conducted a comprehensive ablation analysis evaluating training dynamics, generalization capabilities, and inference efficiency, as summarized in Figure §fig:model_sft_rl.

**The Necessity of Agentic Reinforcement Learning..** The limitations of standard SFT become evident when analyzing the training trajectories in Figure §fig:model_sft_rlb. Models trained via "Raw SFT" without tool integration plateau early, as parameters alone are insufficient for precise scientific computation. While introducing tool-use trajectories ("Tool-Augmented SFT") resolves basic syntax errors and boosts performance, these models effectively learn the *format* of tool usage rather than the *strategy*. This is where the necessity of our agentic RL paradigm becomes undeniable. As illustrated in the right panel of Figure §fig:model_sft_rlb, while the SFT baseline stagnates, the RL-driven training curve continues to climb, propelling the model toward a higher performance ceiling. This gain is mechanistically explained by Figure §fig:model_sft_rlc: the RL-trained model exhibits a statistically significant increase in tool-calling frequency for complex subtasks like "Structure Analysis" and "Reaction Prediction." Unlike the SFT model, which remains a passive tool-user, the RL agent actively queries the sandbox to verify hypotheses, demonstrating a "Self-Correction" behavior where suboptimal initial actions are autonomously refined based on feedback.

**Defying Catastrophic Forgetting..** 
A pervasive risk in training specialized scientific models is the degradation of general reasoning capabilities, known as catastrophic forgetting. Figure §fig:model_sft_rla demonstrates that  effectively circumvents this trade-off. While domain-adapted models like ChemLLM often compromise general capabilities due to rigid knowledge injection,  preserves robust performance across general benchmarks. It achieves strong results on MMLU (STEM, Humanities, and Social Sciences) and SuperGPQA, significantly outperforming baselines. We attribute this robustness to the nature of our agentic training paradigm: rather than forcing the model to memorize static chemical knowledge or specific response templates via standard SFT, our approach instills a universal problem-solving methodology—analyzing the query, formulating a step-by-step plan, and orchestrating appropriate tools. This ensures that the model preserves its intrinsic cognitive structures for logic and analysis, allowing specialized chemical proficiency to coexist with, rather than overwrite, general intelligence.

**The Efficiency of a Unified Agent Architecture..** 
Finally, we address the computational feasibility of AI chemists. Existing Multi-Agent Systems (e.g., ChemCrow [bran2023chemcrow], SciToolAgent [ding2025scitoolagent]) rely on massive commercial models and extensive inter-agent dialogue (e.g., hand-offs between a "Planner," "Critic," and "Executor"), resulting in prohibitive token costs.  internalizes these roles into a single, streamlined reasoning stream. The impact of this architectural shift is quantified in Figure §fig:model_sft_rld:  reduces the average token length by approximately 65% compared to SciToolAgent while maintaining a higher density of "Useful Tokens". Figure §fig:model_sft_rle presents a normalized correlation analysis, where task metrics are mapped to a $[0, 1]$ scale to visualize the return on inference compute. The trend line illustrates an ideal reasoning efficiency, where increased token expenditure (longer CoT) linearly translates to performance gains.  closely adheres to this trajectory, demonstrating effective scaling of test-time compute. In contrast, multi-agent frameworks like SciToolAgent and ChemCrow diverge significantly, suffering from excessive token overhead with diminishing performance returns. This places  in the optimal quadrant of high performance and low inference cost. The efficiency breakthrough resolves the data privacy and latency bottlenecks inherent to cloud-based solutions, making it feasible to deploy expert-level AI chemists on local hardware.

% — END Content/3results —

% — BEGIN Content/5methods —

## Methods

We present a unified framework designed to empower small language models [liu2025teaser,zhu2025learning,li2023textvqa]with autonomous reasoning and operational capabilities in the chemical domain. As illustrated in Fig.§fig:framework, our methodology is orchestrated across three interconnected pillars. First, to overcome the limitations of static knowledge storage, we construct a Chemical Agent Sandbox, an interactive environment that externalizes domain expertise through computational [landrum2013rdkit], deep learning, and retrieval-based [li2023FreestyleRet,wu2026towards] tools. Second, to bridge the gap between abstract chemical knowledge and actionable problem-solving, we introduce a pipeline for constructing High-Quality Reasoning Trajectories, incorporating tool-integrated narratives and reflective refinement. Finally, we implement a Two-Stage Training Paradigm, which progressively transitions the model from supervised behavioral cloning to robust reinforcement learning, optimizing for scientific validity and rigorous chemical constraints via the SMILES-GRPO mechanism.

### Construction of the Chemical Agent Sandbox

To endow the language model with comprehensive domain expertise, we established a Chemical Agent Sandbox—an interactive environment designed to decouple chemical reasoning from static knowledge storage. This framework covers the full spectrum of the chemical discovery pipeline, ranging from molecular structure analysis and de novo design to property optimization and retrosynthetic planning. To achieve robust performance across these diverse tasks, we encapsulate domain-specific tools into three distinct categories of agents:

**Computational Software Agents:** Leveraging robust cheminformatics libraries, primarily RDKit and RDChiral, these agents provide deterministic capabilities for molecular graph manipulation. They are responsible for verifying topological validity, analyzing stereochemistry, and executing precise structural editing, ensuring that the model’s generated molecules adhere to strict chemical rules.

**Deep Learning-based Agents:** Serving as the predictive intuition of the system, these agents interface with pre-trained models and frameworks like PyTDC. By providing rapid quantitative feedback on physicochemical properties (e.g., QED, LogP) and ADMET profiles, they guide the language model in navigating the high-dimensional chemical space during multi-objective optimization tasks.

**Retrieval-based Agents:** To anchor synthesis planning in empirical reality, these agents maintain a curated repository of reaction equations and templates. Through substructure matching and similarity search, they enable the model to retrieve analogous reaction pathways and validate reaction conditions, promoting mechanistic feasibility in synthesis prediction.

### Construction of High-Quality Reasoning Trajectory

High-quality trajectory construction is pivotal for enabling the model to interact effectively with the chemical sandbox and solve complex domain tasks. We approach this construction in two phases: first by structuring the abstract chemical space into traceable reasoning steps, and subsequently by integrating dynamic tool interactions into these narratives.

**Generating Chemical Reasoning Trajectories**. To transcend simple factual recall, we align our data construction with ChemCoTBench, a framework that decomposes the chemical discovery process into 9 major tasks and 22 subtasks. This structure bridges fundamental capabilities, such as Molecule Understanding and Editing—with high, which stakes downstream applications like Molecule Optimization (e.g., binding affinity improvement) and Reaction Prediction (e.g., retrosynthesis planning). By breaking down these complex challenges into explicit sequences of modular chemical operations (e.g., functional group addition or substitution), we convert abstract chemical problems into actionable, step-by-step reasoning scaffolds. This rigorous decomposition ensures the model learns the operational logic required for real-world discovery rather than mere property memorization.

**Generating Tool-Integrated Reasoning Trajectories**. To capture the interactive nature of modern scientific workflows, we propose a Tool-Integrated Reasoning construction method. First, we leverage the Chemical Agent Sandbox to decouple reasoning from calculation, allowing the LLM to offload error-prone computations (e.g., RDKit parsing, QED evaluation) to external microservices. Second, to address the issue of disjointed API logs, we implement a "Reflective Refinement" mechanism. Instead of retaining mechanical "Action-Observation" pairs, this process injects verified tool outputs back into the context and prompts a teacher model to rewrite the reasoning trace. This transforms rigid tool logs into fluid, expert-level scientific narratives, where the agent interprets evidence, validates hypotheses, and dynamically adjusts its strategy, mirroring the cognitive process of a professional chemist.

### Model Training: A Two-Stage Paradigm

To effectively instill both chemical domain knowledge and agentic reasoning capabilities into the model, we employ a progressive two-stage training paradigm. This approach first establishes a stable behavioral policy through supervised learning and subsequently refines the model's problem-solving strategies using reinforcement learning with chemistry-specific feedback.

#### Stage 1: Cold-Start Supervised Fine-Tuning

The primary objective of the "Cold-Start" phase is to initialize the model's understanding of chemical syntax and establish the fundamental "Think $$ Call Tool $$ Observe" behavioral pattern. We utilize the synthesized tool-integrated trajectories to perform Supervised Fine-Tuning (SFT) on the base model (e.g., Qwen-2.5/3).

We treat the agentic interaction as a sequential generation task. Given an input prompt $x$ and a target trajectory $y = (y_1, y_2, ..., y_T)$ consisting of interleaved reasoning thoughts, tool calls, and observations, we optimize the model parameters $$ by minimizing the negative log-likelihood of the next token. The loss function $_{}$ is defined as:

$$
_{}() = - _{(x, y) _{}} [ _{t=1}^{T} P_(y_t x, y_{<t}) ]

$$

where $_{}$ represents the curated dataset of tool-integrated trajectories, and $y_{<t}$ denotes the history of tokens preceding step $t$. This phase ensures that the model learns to strictly adhere to the formatting requirements of the *Chemical Agent Sandbox* and mimics the expert-level reasoning logic embedded in the training data, providing a robust policy initialization $_{}$ for the subsequent reinforcement learning stage.

#### Stage 2: Reinforcement Learning with SMILES-GRPO

While SFT provides a strong foundation, it is limited by the static nature of imitation learning. To transcend mere mimicry and enable the model to explore novel chemical spaces, we advance to a reinforcement learning framework. We adopt Group Relative Policy Optimization (GRPO), which eliminates the need for a value function approximation by normalizing advantages within a group of sampled outputs.

**The GRPO Objective..** 
For each chemical query $q$, we sample a group of $G$ outputs ${o_1, o_2, ..., o_G}$ from the old policy $_{_{}}$. The optimization objective maximizes the surrogate objective while constraining the policy update via KL-divergence. The GRPO loss function is formulated as:

$$

_{}() = _{q , {o_i}_{i=1}^G _{_{}}} [ {G} _{i=1}^G ( &( r_i {_{_{}}(o_i|q)}, r_i  ( {_{_{}}(o_i|q)}, 1-, 1+) )- _{}(_|| _{}) ) ]


$$

where $$ is the clipping parameter, $$ controls the KL-divergence penalty to prevent policy collapse, and $r_i$ is the advantage term, calculated by normalizing the rewards within the group: $r_i = ({R_j}_{j=1}^G)}{({R_j}_{j=1}^G)}$.

**Multidimensional Chemical-Aware Reward..** 
To rigorously evaluate the scientific validity of the reasoning chain, we engineered a dense chemical reward function $R_{}$. Unlike generic scalar feedback, our reward signal integrates format compliance with deep chemical verification metrics. The total reward for a generated output $o$ is defined as:

$$
R_{}(o) = _{1} R_{} + _{2} R_{}

$$

where $R_{}$ is an indicator function for syntactic correctness (e.g., correct tool calling tokens). The chemical reward $R_{}$ is a weighted sum of four key components:

$$
R_{} = w_{str} _{} + w_{func} _{} + w_{opt}  + w_{rxn} _{}

$$

Specifically, this composite reward mechanism first enforces structural integrity by measuring the Tanimoto similarity of Bemis-Murcko scaffolds, ensuring that generated molecules retain the desired core architecture. It simultaneously evaluates functional fidelity by explicitly penalizing the loss of critical functional groups mandated by the input prompt. Beyond structural constraints, the objective function quantifies the magnitude of property optimization, rewarding positive shifts in target metrics such as QED or binding affinity. Finally, to guarantee synthetic realizability, it incorporates a validity check that assesses whether the predicted synthesis pathways align with established chemical reaction templates.

By optimizing against these granular metrics, SMILES-GRPO drives the model to strategize its tool usage, ensuring that the generated molecules are not only textually valid but also scientifically viable and optimized for their target properties.
% — END Content/5methods —