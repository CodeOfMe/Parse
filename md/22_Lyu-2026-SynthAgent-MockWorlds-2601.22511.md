# Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards

> **arXiv:** [2601.22511](https://arxiv.org/abs/2601.22511)
> **TeX source:** [arXiv-2601.22511v1/](arXiv-2601.22511v1/)
> **PDF:** [SynthAgent-arXiv-2601.22511v1.pdf](SynthAgent-arXiv-2601.22511v1.pdf)

---

## Abstract

Small LLMs often struggle to match the agentic capabilities of large, costly models. While reinforcement learning can help, progress has been limited by two structural bottlenecks: existing open-source agentic training data are narrow in task variety and easily solved; real-world APIs lack diversity and are unstable for large-scale reinforcement learning rollout processes. We address these challenges with *SynthAgent*, a framework that jointly synthesizes diverse tool-use training data and simulates complete environments. Specifically, a strong teacher model creates novel tasks and tool ecosystems, then rewrites them into intentionally underspecified instructions. This compels agents to actively query users for missing details. When handling synthetic tasks, an LLM-based user simulator provides user-private information, while a mock tool system delivers stable tool responses. For rewards, task-level rubrics are constructed based on required subgoals, user-agent interactions, and forbidden behaviors. Across 14 challenging datasets in math, search, and tool use, models trained on our synthetic data achieve substantial gains, with small models outperforming larger baselines. [^note: Code for data synthesis pipeline and training: https://github.com/haruhi-sudo/SYNTHAGENT] 

## Introduction

% — BEGIN introduction —

**Figure:** *
Comparison between existing agentic RL training recipes and ours. Open-source agentic training data are narrow in domain, while real-world APIs are costly and unstable. We replace these with diverse synthetic tasks and associated mock environments.
* () _(image: my_introduction.pdf)_

Large language models (LLMs) demonstrate strong agentic capabilities within ReAct-style frameworks [yao2023react]. Through an iterative *reasoning–action–observation* loop, LLM-based agents can solve complex tasks that require interaction with external environments [xi2025rise], such as booking hotels or canceling flights [barres2025tau2]. However, these agentic capabilities depend heavily on very large base models [team2025kimi], resulting in substantial inference costs and deployment overhead. Consequently, enabling smaller models to reproduce the agentic capabilities of large models has become an important research direction [lyu2025correction,li2025chain].

Distillation methods based on supervised fine-tuning (SFT), in which a student model clones a teacher's behavior [torabi2018behavioral], can enhance the agentic capabilities of small models. Recent studies [mai2025agentrlscalinglaw] further show that reinforcement learning (RL) is more effective than SFT for improving long-horizon planning and adaptive decision-making. However, most RL-based approaches focus on refining RL algorithms themselves [dong2025agentic-entropy,dong2025agentic], while overlooking two fundamental bottlenecks:

- *Lack of diverse and challenging agentic training data.* Public datasets cover only a narrow range of domains and tools, and many have already been seen by modern LLMs during pre-training or fine-tuning. As a result, RL rollout often yields near-perfect trajectories with weak learning signals [yu2025dapo].
- *Absence of stable, diverse environments.* Real environments rarely support real-time model-user interaction and offer only a narrow tool set. RL rollout also requires a massive number of tool calls, making it impractical to rely on costly real-world APIs [team2025longcat].

To address these, we introduce *SynthAgent*, a framework that synthesizes tool-use tasks along with lightweight mock tool interfaces. A strong agentic teacher LLM generates novel tasks and their associated tools, guided by diverse persona backgrounds [ge2025scalingsyntheticdatacreation]. As shown in Figure §fig:demo, each task is paired with its own tool ecosystem, greatly expanding task and tool diversity. Moreover, the synthetic tools require no real deployment: an open-source LLM simulates both user and tool responses locally, ensuring stability.

Specifically, *for synthesizing training data*, we introduce an information gap by rewriting detailed workflows as underspecified instructions, while critical details are hidden in a private user context. This design forces agents to actively query users and call tools to recover missing information, encouraging genuine long-horizon interaction.
Second, *for LLM-based tool response consistency*, we maintain a task-level mapping of prior tool calls and responses. New calls are answered by consulting this mapping for consistent replies. As each synthetic task has a unique toolset, the mapping is scoped per task, keeping it lightweight during rollout.
Finally, *for reward design*, we avoid subjective LLM-written rubrics and derive rewards from observable behavior. Using the workflow from data synthesis as a reference, we extract corresponding high-level subgoals from real execution trajectories, each reachable via multiple valid paths. This yields execution-grounded rewards that support diverse strategies, while filtering out low-quality data when the teacher fails to reliably complete the workflow.

We evaluate our approach on 14 recent, challenging datasets spanning agentic tool use [yehudai2025survey] and short-horizon reasoning. In real-world tasks, models trained on synthetic data within virtual environments substantially outperform those trained on open-source datasets. After training, our 8B–14B models surpass a 32B model on multiple agentic benchmarks. In summary, the major contributions of this work are as follows:

- We introduce an open-source framework for synthesizing diverse agentic tool-use tasks, with stable, lightweight mock tool interfaces.
- In synthetic tasks that require genuine long-horizon interactions, we train models with execution‑grounded, rubric‑based rewards.

% — END introduction —

## Related Work

% — BEGIN related_work —

**Figure:** *
A unified pipeline for generating synthetic tool-use tasks, constructing stable mock environments, and deriving rubric-based rewards for agentic RL. Diverse tasks and tool ecosystems are created, guided by personas. For each synthetic task, an LLM-simulated user and environment are employed. To assign rewards, multiple trajectories are compared to the previously generated high-level workflow to infer task-specific rubrics.
* () _(image: my_method.pdf)_

### Agentic Reinforcement Learning

Recent studies show that RL outperforms SFT in long-horizon planning and adaptive decision-making [zhang2025landscape], making RL a core paradigm for training LLM agents in dynamic, multi-turn environments [mialon2023gaia]. Classical methods such as Q-learning [mnih2015human], PPO [schulman2017proximal], and self-play [silver2017mastering] have provided the conceptual foundation for agentic optimization in LLM-based systems. These techniques have evolved into language-centric RL frameworks, where natural-language reasoning steps, tool calls, and observations are treated as latent states and actions [yao2023react,zhang2025landscape]. Recent work has further improved RL algorithms to better couple exploration with robust tool use in long-horizon tasks, including verifiable-reward RL [su2025crossing], entropy-regularized policy optimization [dong2025agentic-entropy], and agent-specific PPO/GRPO variants [dong2025agentic]. Despite this progress, research remains largely focused on RL algorithms, with considerably less attention given to data and environment design.

### Synthetic Data for Agentic Training

The effectiveness of agentic RL depends on high-quality, diverse data and environments, which remain scarce [yehudai2025survey]. Early works such as Self-Instruct [wang2023self] use strong but closed-source LLMs to generate instruction-following data for training smaller open-source models. To further increase diversity, ge2025scalingsyntheticdatacreation [ge2025scalingsyntheticdatacreation] propose Persona Hub, which curates one billion web-derived personas to enable diverse synthetic data generation. In parallel, qin2025scaling [qin2025scaling] identify a "rectified scaling law" for synthetic data: as long as diversity is maintained, gains from synthetic pre-training persist even at very large scales.

For environment construction, benchmarks [qin2023toolllm] rely on real-world APIs for authenticity or use LLMs to simulate existing APIs and reduce cost. However, these environments remain constrained by real services, a lack of diversity [lu2025toolsandbox]. And identical states and actions can produce inconsistent responses in simulation, making them unsuitable for direct RL.

Most existing synthesis methods neither target long‑horizon agentic tasks nor construct stable RL environments, meaning few works approach ours on either front. Our approach further unifies both aspects and integrates them with rubric‑based RL.

% — END related_work —

## Methodology

% — BEGIN method —

In this section, we move beyond improving RL algorithms themselves, and instead focus on two fundamental yet underexplored factors that limit agentic RL for LLMs: diverse, challenging training data and diverse, stable environments. Figure §fig:method illustrates *SynthAgent*, a framework for synthesizing tool-use tasks, constructing virtual environments, and deriving rubric-based task rewards.

### Synthetic Tool-Use Task Generation

**Tool Set Generation.** 

Existing open-source agentic datasets are predominantly composed of web search and math tasks, which typically can be solved using only a search tool or a code interpreter, resulting in homogeneous tool environments. To diversify settings and tool usage, we incorporate large-scale personas from Persona Hub [ge2025scalingsyntheticdatacreation] as backgrounds (Figure §fig:method(a)). These personas cover a wide range of identities and scenarios (e.g., a senior SRE debugging a database issue).

For each persona-defined context, we employ a strong agentic LLM to (i) infer a high-level workflow describing how an individual with that background would accomplish the task, and (ii) based on this inferred workflow, construct a task-specific virtual tool ecosystem with tool descriptions and API specifications. As a result, each task is paired with a dedicated tool suite, encouraging models to learn tool-use procedures rather than memorize fixed APIs. To further increase task difficulty, we introduce task-level forbidden constraints, such as "disallowing a system reboot during a database repair task". These constraints raise the RL challenge by requiring the model to plan and act within nontrivial restrictions. [^note: For data synthesis, we employ Qwen3-235B-A22B-Instruct-2507 [yang2025qwen3] due to its strong agentic capabilities and low-cost local deployment.] 

**Fuzzy Task Generation.** 

After defining the tool set available to the agent, we next design the tasks, i.e., the agent's inputs. Tasks derived directly from previously generated high-level workflows are often over-specified; for example, *Check logs $$ Verify DB state $$ Execute rollback*. The initial input outlines the workflow, and simply following it becomes the optimal action sequence $a^$. Consequently, rollouts $_$ are highly homogeneous. At many steps $t$, the policy $_(a s_t)$ is nearly deterministic; thus, during RL training, the variation of the advantage under $_$ may become negligible:

$$
_{a_t _(s_t)}\![A(s_t,a_t)] 0,

$$

where $A(s_t,a_t)$ is the advantage under a value baseline, satisfying $_{a_t _(s_t)}[A(s_t,a_t)] = 0$. As demonstrated by the Cauchy–Schwarz inequality:

$$


\| _{a_t _(s_t)} &[__(a_t s_t)\, A(s_t,a_t)] \| \\
\,&
_{a_t _(s_t)} [ \| __(a_t s_t) \|^2 ]} \\
&_{a_t _(s_t)} [\| A(s_t,a_t) \|^2 ]} \\
=\, &
_{a_t _(s_t)} [ \| __(a_t s_t) \|^2 ]} \\
&_{a_t _(s_t)} [A(s_t,a_t) ]},

$$

when Eq. (§eq:adv_var_collapse) holds, the expected gradient magnitude shrinks, weakening the learning signal.

To mitigate this degeneracy, we inject an *information gap* during task construction: partition each task initial state $s_0$ into an agent-visible instruction $I$ and a user-only hidden context $C$:

$$
s_0 (I,C),  H(a^I) .
$$

Initially, $I$ is insufficient to determine the optimal action $a^$ uniquely; the conditional entropy $H(a^I)$ is large. Critical details must be recovered through interaction. As illustrated in Figure §fig:method(a), we employ an LLM to rewrite an overly explicit request into a minimal one $I$, for example, "The checkout service is returning 500 errors again. Can you investigate and fix it?" Decisive details (e.g., "v2.1 was just deployed" and "OOM") are moved to $C$ and revealed only when the agent queries the user.

Under intentionally high $H(a^I)$, the policy must first query for the missing context $C$ before invoking tools. As observations $o_{t}$ gradually reveal the hidden information, uncertainty decreases ($H(a^I, o_{t}) < H(a^I)$). Early decisions therefore become nontrivial: the agent must decide which clarification to ask first to elicit informative observations $o_t$, preventing $_(s_t)$ from becoming near-deterministic. This mitigates the gradient degeneration issue during model training.

Appendix §sec:example provides examples of the synthetic tasks and tools.

### Mock Environments

**Mock Tool & User.** 
When the model attempts the synthetic tool-use tasks described above, the corresponding tool set is registered in the system prompt. Because these tools are virtual rather than real, we must simulate both tool execution and responses. To this end, we build a fully LLM-simulated mock environment, which requires no real deployment and supports large-scale interaction during RL rollout. The LLM-simulated tool receives the model's tool-call requests and returns appropriate outputs. The LLM-simulated user answers the model's queries based on user-private background information $C$ generated earlier. [^note: These interactions are primarily simple, formatted question-answering tasks requiring no powerful model. Thus, we implement this component using Qwen3-30B-A3B-Instruct-2507 [yang2025qwen3], which is easy to deploy locally and incurs very low runtime cost.] 

**Stable Environments.** 

During the RL rollout process, the same task is attempted many times, raising a central concern: if tool responses are non-reproducible, then even under the same state $s$, executing the same tool action $a$ (identical $`tool`+`args`$) may yield different observations $o$. 
This randomness propagates along the trajectory, so that the same $(s_t, a_t)$ can induce different future returns across rollouts, making the advantage estimate $_t$ inconsistent. Consequently, even for identical $(s_t, a_t)$, the policy update term $_t = __(a_t s_t) \, _t$ may exhibit substantially different magnitudes and even opposite signs across samples, hindering training stability.

A natural mitigation is to add retrieval-augmented memory [lewis2020retrieval] to the tool simulator, storing past tool calls and responses. When generating a new tool response, the simulator retrieves similar calls as references to ensure consistency. In our setting, each task has its own tool suite, so only a few calls require within-task consistency. Instead of a full memory system, we use a lightweight task-level finite mapping:

$$
 = {(u_i, y_i)}_{i=1}^{M}, u_i = (`tool`_i, `args`_i).
$$

When the model issues a valid tool call $u$, the simulator checks similar entries in $$ and checks for a semantically equivalent one. If none is found, it generates a response $y$ and adds $(u, y)$ to the mapping. Equivalence checking and response generation can be handled in a single forward pass, adding no extra computational cost.

Empirically, the task-level mapping remains very small. For example, in a rollout with 16 trajectories and an average of 10 tool calls per trajectory, even if all calls were unique, the size of $$ would be at most 160. Therefore, we can *omit retrieval* altogether and include $$ directly in the tool simulator's prompt, allowing the model to identify matches. The entire process remains lightweight and efficient, significantly improves training stability without restricting exploration.

### Automatic Rubric-Based Rewards

**Task-Level Rubrics.** 

Unlike math tasks with clearly defined correctness-based rewards [guo2025deepseek], reward design for multi-step tool-use tasks is inherently challenging. A common practice is to use an LLM as a judge to assign a scalar score to each trajectory, but such judgments can be subjective. Fortunately, we construct fuzzy tasks by rewriting high-level workflows, whose steps provide objective subgoals (e.g., *Check logs, Verify DB state, Execute rollback*) for trajectory rewards.

However, the designed workflow may not match real executions. To address this, we collect multiple actual executed trajectories from strong teacher models. Using the workflow as a reference, we prompt an LLM to extract workflow-relevant subgoals and user-agent interactions from each trajectory. Both are grounded in the workflow and can be achieved via multiple exploration paths.

If the workflow leaves too many steps unexecuted, the example is removed. This filters noisy data and reduces reliance on the number of teacher demonstrations, since trajectories with many unexecuted steps can be discarded.

Additionally, during tool set generation (Section §sec:tool_use_task_generation), each task is paired with its own set of forbidden behaviors (e.g., disallowing a system reboot). Combined with the subgoals and interaction requirements, these form a complete task-level rubric unique to each synthetic task. Appendix §sec:example provides examples of the generated rubrics.

**Rubric-based Reward.** 
During RL training, we use an LLM as a judge to assign rewards based on the task-level rubric. Specifically,

$$
R() = () (N_{}() + I_{}()).
$$

Here, $() $ is 0 if and only if $$ violates any rubric-specified forbidden behavior (yielding zero reward), and 1 otherwise. $N_{}() [0,1]$ is the fraction of subgoals completed, and $I_{}() [0,1]$ is the fraction of required user–agent interactions satisfied; we average these scores as the final reward. This rubric-based reward design scales seamlessly to large numbers of synthesized tasks.

Table §tab:synthagent-data-stats-vertical summarizes the statistics of our synthesized tool‑use dataset, and reports the total token cost of the full synthesis pipeline (including tool, task, and rubric generation). Since the entire process runs on locally deployed open-source models, the cost is negligible. And even with commercial APIs, it remains well within an affordable range.

### Final RL Training

Following the technical reports of Kimi K2 [team2025kimi] and LongCat [team2025longcat], strong reasoning ability is essential for agentic tasks. Accordingly, we augment our virtual tool-use data with a small set of high-difficulty reasoning tasks, sampling 4,000 search or math instances from ToolStar [dong2025toolstarempoweringllmbrainedmultitool].
Since our tool-use data contain rich contextual backgrounds whereas math problems are purely abstract, this mismatch may hinder training. To increase diversity, we prompt Qwen3-235B [^note: For brevity, we use Qwen3-235B to represent Qwen3-235B-A22B-Instruct. The same applies in the following text.] with persona information [ge2025scalingsyntheticdatacreation] to rewrite each problem into a scenario-based variant. Each synthesized problem is then solved 3 times by Qwen3-235B, and we retain only those with fully consistent answers, ensuring reliability.

Tool-use tasks run in a virtual environment with rubric-based rewards, while reasoning tasks run in a real Python environment and are evaluated by answer correctness. We then train the model on the combined dataset using GRPO [shao2024deepseekmath].

% — END method —

## Experiments

% — BEGIN experiments —

**Table:** *Statistics of the synthesized tool-use dataset.*

| lc
**Statistic** | **Value** |
| — | — |
| Avg. #Tools Per Task | 4.1 |
| Avg. #Interactions (After Training) | 13.4 |
| Avg. Mapping Size $|M|$ (After Training) | 30.1 |
| Avg. Per‑Task Token Usage in Synthesis | 25,278 |

**Table:** *Agentic performance comparison. 
For TAU-2 (Airline, Telecom, Retail), we report Avg@4 and use the open-source Kimi-K2-Instruct model as the user simulator. Qwen3-235B refer to the model Qwen3-235B-A22B-Instruct. The same applies in the following text. Best results except Qwen3-235B are **bolded**.

*

| ccccccccc
2*Method | 3cTAU-2 Bench | 4cBFCL-V4-Multi-turn | 2*Avg. |
| — | — | — | — |
| Qwen3-14B | 22.0 | 25.4 | 39.5 | 40.0 | 34.5 | 26.5 | 26.5 | 30.6 |
| Qwen3-32B | 22.5 | 27.6 | 44.7 | 50.5 | 43.0 | 30.5 | 33.0 | 36.0 |
| ToolStar-8B | 13.5 | 25.9 | 39.5 | 52.0 | 38.0 | 22.5 | 30.5 | 31.7 |
| ToolStar-14B | 18.0 | 30.7 | 40.4 | 56.5 | 35.5 | 29.5 | 39.5 | 35.7 |
| !10 *SynthAgent*-14B | **40.0** | **44.7** | **58.6** | **57.0** | **46.5** | 31.0 | **46.0** | **46.3** |

**Table:** *
Short-horizon reasoning performance comparison (math and search). 
For AIME24, AIME25, and HMMT25, we report Avg@6 for more stable evaluation. Best results except Qwen3-235B are **bolded**.
*

| ccccccccc
2*Method | 4cMath | 3cSearch | 2*Avg. |
| — | — | — | — |
| Qwen3-14B | 39.4 | 35.6 | 53.8 | 39.4 | 37.8 | 21.0 | 38.0 | 37.9 |
| Qwen3-32B | 50.0 | 41.1 | 56.4 | 35.1 | 44.8 | 25.0 | 37.0 | 41.3 |
| ToolStar-8B | 60.6 | 54.4 | 76.4 | 47.8 | 58.5 | 33.0 | 44.5 | 53.6 |
| ToolStar-14B | 71.7 | 63.3 | 77.3 | 45.0 | 60.4 | 40.0 | 44.0 | 57.4 |
| !10 *SynthAgent*-14B | **72.2** | **66.7** | **80.1** | **53.9** | **63.5** | 43.0 | **50.0** | **61.3** |

### Experimental Setup

We evaluate *SynthAgent* by assessing models trained on our synthetic data and within simulated environments. Our experiments focus on agentic benchmarks that measure long-horizon tool use, multi-turn planning, and adaptability to unfamiliar tools. We also test short-horizon generalization through reasoning tasks such as math and search.

**Agentic Tool Use Benchmarks.** 

We evaluate on the most representative agentic benchmarks: TAU-2 [barres2025tau2] and BFCL-V4 [patil2025bfcl], spanning 7 datasets and nearly 100 diverse tools. These benchmarks are widely used by Qwen [yang2025qwen3], Kimi [team2025kimi], and DeepSeek [deepseekai2025deepseekv32pushingfrontieropen], aligning our protocol with that of leading foundation models.

BFCL-V4 provides multiple datasets; we focus on its multi-turn subset (about 800 tasks). These tasks span diverse real-world domains such as trading, vehicle control, and social media. Each task typically requires 5 to 20 tool-interaction turns, providing a rigorous evaluation of the model's capabilities in parameter clarification and error rejection.

TAU-2 targets three real-world business domains: airline, retail, and telecommunications, comprising roughly 300 tasks. These tasks generally require multi-turn interactions between the agent and user. Moreover, users can also invoke tools and modify the environment, meaning the model must not only execute tools correctly but also guide the user and handle uncertain feedback.

The above agentic benchmarks, with their unfamiliar tools and long-horizon planning demands, serve as our primary evaluation suite.

**Reasoning Benchmarks.** 
We also examine the short-horizon reasoning capabilities of our framework. We employ several math benchmarks (AIME24, AIME25, HMMT25, Olympiad [he-etal-2024-olympiadbench]) and search benchmarks (FRAMES [krishna-etal-2025-fact], WebWalker [wu2025webwalker], XBench [chen2025xbench]). These tasks involve only two tools: a Python interpreter and Google Search and typically require fewer than five interaction turns. However, each step demands deeper reasoning than in the agentic benchmarks, making them suitable for out-of-domain evaluation.

All benchmarks were introduced after 2024, ensuring strong relevance and up-to-date evaluation.

**Evaluation.** 
TAU-2 and BFCL provide not only datasets but also full interactive environments. During evaluation, the model must invoke tools to interact with these environments; performance is measured by checking whether environment states are correctly updated to their ground-truth values using Exact Match. For math reasoning, we also apply Exact Match. For more free-form outputs in search reasoning, we use Qwen3-235B to judge whether the model's responses are semantically consistent with the ground truth. The search tool is implemented via the Google Search API.

**Baselines.** 
We compare our model, trained on synthetic tasks and mock environments, against the following baselines: RL-trained models on the latest open-source ToolStar [dong2025toolstarempoweringllmbrainedmultitool], which employ 30,000 math and search examples. We also evaluate strong LLMs prompted to integrate tools, such as the larger Qwen3-32B. The Qwen3 technical report [yang2025qwen3] indicates that Qwen3-32B has already been trained on synthetic tool-use data using RL, making it a competitive baseline. All baselines perform inference using the OpenAI function-calling format and the same prompt, ensuring a fully consistent setup.

**Implementation.** 

Using *SynthAgent*, we generate 15,096 synthetic tool-use tasks entirely with locally deployed open-source LLMs. We train Qwen3-8B/14B with GRPO (non-thinking) to assess data quality. For rubric construction, we collect four demonstrations from a strong agentic teacher (Qwen3-235B). Since rubric design depends mainly on high-level workflow rather than specific teacher demonstrations, the number of them typically has little impact, as shown in Section §sec:futher. For reward, we employ Qwen3-30B-A3B-
Instruct to judge with rubrics. More implementation details are provided in Appendix §sec:implementation.

**Figure:** *Effect of tool-simulator size on TAU-2 performance(5,000 training samples), showing negligible gains from larger simulators.* () _(image: simulator_comparison.pdf)_

### Main Results

Table §tab:tau2_bfcl reports the performance of our 8B and 14B models, trained on synthetic data and simulated environments, on real-world agentic benchmarks such as TAU-2 and BFCL-Multi-turn.

***SynthAgent** enables small models to match and even surpass much larger agentic models.* 
Using synthetic tool-use tasks and fully simulated environments, our method yields substantial gains across real-world agentic benchmarks. On TAU-2 and BFCL-Multi-turn, *SynthAgent*-8B scores 42.9 on average (+12.3 over Qwen3-14B, and +6.9 over Qwen3-32B). The improvements are even greater with *SynthAgent*-14B, which scores 46.3 on average.

Despite being far smaller, *SynthAgent*‑14B matches Qwen3‑235B on many TAU‑2 and BFCL domains, demonstrating that our synthetic training strategy can close the gap with much larger models.

**Synthetic data and simulated environments substantially outperform existing open-source datasets.** 
Open-source agentic datasets cover only a limited set of tools and cannot capture the complexity of real tool-use logic. For example, in the latest ToolStar [dong2025toolstarempoweringllmbrainedmultitool] dataset, RL-trained models show no significant improvement in agentic performance. In contrast, *SynthAgent*-8B achieves great performance in TAU-2, consistently surpassing open-source baselines, with BFCL-Multi-turn showing the same pattern. These results indicate that diverse and challenging synthetic tasks are far more effective for strengthening a model's agentic abilities.

**Beyond tool-use tasks, the model also generalizes to short-horizon reasoning.** 
Table §tab:reasoning_deepsearch shows that, although math and search tasks are not the primary focus of our training(only 4,000 instances sampled from ToolStar are included), *SynthAgent* still achieves substantial gains. Under the same non‑thinking setting, *SynthAgent*-8B significantly outperforms Qwen3‑14B and even exceeds an 8B model trained on 30,000 ToolStar examples. The improvements for *SynthAgent*-14B are even larger. These results indicate that our method transfers effectively to new reasoning domains, with tool-use data also exhibiting strong generalization in reasoning tasks.

Overall, training on synthetic tasks and environments allows 8B–14B models to rival or surpass 32B models, drastically reducing inference cost.

**Table:** *Ablation study on agentic benchmarks.*

| cccccc
2*Method | 3cTAU-2 Bench | 2*-V4 |
| — | — | — |
| Multi-turn |
| Avg. |
| reasoning data only | 22.0 | 26.1 | 38.2 | 37.9 |
| W/O information gap | 22.0 | 28.9 | 35.3 | 39.1 |

### Further Analysis

**Ablation** The results in Table §tab:ablation_tau2_bfcl confirm that synthetic tool-use data is crucial. Using only reasoning data yields improvements over the Qwen3-8B baseline, mainly from short-term reasoning, but it remains inadequate for multi-turn long-horizon agentic tasks. In contrast, adding synthetic tool-use data provides substantial gains and consistently improves performance across all agentic benchmarks. However, if we do not introduce information gaps, do not rewrite workflows into less explicit descriptions, and directly use them for training without user interaction, the benefit of tool-use data becomes negligible. This further validates the rationale behind our design.

**Impact of Tool Simulator Size on Performance** The tool simulator primarily generates responses to new tool calls, and checks whether a call matches a previous query in the prompt. Both are simple, well-defined operations that typically do not require a strong model. To validate this, we evaluate Qwen3-8B trained on 5,000 synthetic agentic tasks with two simulators: Qwen3-235B and the smaller Qwen3-30B-A3B-Instruct model. As shown in Figure §fig:simulator_comparison, a larger simulator does not improve performance, suggesting mock tool simulation is largely formatted QA and semantic matching, rather than a capability that benefits from model scale.

**Impact of Number of Teacher Demonstrations** The rubrics and subgoals are mainly derived from the high-level workflow in Section §sec:tool_use_task_generation. Teacher trajectories are used only to align the synthesized workflow with real executions; examples where the teacher covers too little of the workflow are discarded. In principle, rubric quality depends weakly on the number of teacher trajectories.

To evaluate this, we build rubrics for 5,000 synthetic agentic examples using 2, 3, or 4 teacher trajectories and compare performance. As shown in Figure §fig:teacher_demonstrations, adding more demonstrations yields no significant gains, suggesting rubric construction does not require substantial computation.

More experiments, such as data-scaling effects on RL and RL-SFT comparisons at equal data sizes, are provided in Appendix §sec:additional_experiments.

% — END experiments —

## Conclusion

We present *SynthAgent*, a novel framework addressing two core bottlenecks in training agentic language models: the scarcity of diverse, challenging tasks and stable tool environments. By jointly synthesizing tool-use tasks with underspecified instructions and providing stable mock environments, *SynthAgent* enables efficient reinforcement learning for small models. Extensive evaluations demonstrate that models trained entirely on synthetic data and virtual environments achieve substantial gains, with small models surpassing much larger baselines.

## Limitations

Agentic training data synthesis is an increasingly important research topic. Technical reports from leading foundation models (Qwen3 [yang2025qwen3], LongCat [team2025longcat], Kimi K2 [team2025kimi], DeepSeek V3.2 [deepseekai2025deepseekv32pushingfrontieropen], Minimax M2 [chen2025minimax]) consistently show that synthetic data, rather than real-world data, forms the core of agentic RL. However, these models do not release their synthetic datasets, nor do they provide detailed descriptions of their synthesis procedures. This limitation restricts our ability to refine the *SynthAgent* pipeline based on prior work and makes it difficult to compare against stronger baselines. Future work should explore additional approaches for agentic training data synthesis and identify the key factors that are most critical for building effective agents.



% — BEGIN appendix —

## The Use of Large Language Models (LLMs) in Writing

An LLM (specifically OpenAI's GPT‑5 [achiam2023gpt]) was used solely for minor language editing, including grammar correction and light rephrasing for clarity. It did not contribute to the research design, and all scientific content is entirely the authors' own.

**Figure:** *Impact of increased training data on RL performance, and comparison between RL and SFT at the same data scale.* () _(image: tau2_rl_sft_comparison.pdf)_

## More Implementation Details

**Synthetic Data.** 
All personas used during data synthesis are exclusively sourced from Persona Hub [ge2025scalingsyntheticdatacreation]. During toolset generation, we define virtual tools following the OpenAI function-calling specification and filter out any samples that cannot be parsed into a valid function-calling format. In total, we generated 15,096 tool-use tasks, along with a smaller set of math or search reasoning tasks (approximately 4,000). All data were produced locally with open-source LLMs, without reliance on commercial APIs, ensuring a stable and cost-efficient pipeline.

**Benchmarks.** 
Detailed benchmark statistics (e.g., subsets and test sizes) are reported in Table §tab:benchmarks. 
Agentic tool-use benchmarks (TAU-2 and BFCL-V4 Multi-turn) measure long-horizon, multi-turn interaction with unfamiliar tools. Reasoning benchmarks (math and search) evaluate out-of-domain generalization, where tasks typically involve fewer tool calls but require deeper multi-step reasoning and evidence synthesis.

**Training.** 
We perform reinforcement learning using the GRPO algorithm [shao2024deepseekmath] within the VERL framework [sheng2024hybridflow]. We use a global batch size of 128, a PPO mini-batch size of 16, a rollout size of 16, and a maximum response length of 13,000 tokens, training for 2 epochs on 8$$NVIDIA H20 GPUs. The clipping ratio is constrained between 0.2 and 0.28, and we allow up to 16 turns per rollout. Since Qwen3 base models already exhibit strong tool/function-calling capabilities and reliably follow the required format, we skip the SFT phase for format learning and directly proceed with RL training.

**Inference.** 
During inference, the model was deployed with SGLang [zheng2024sglang] to increase throughput. TAU-2 [barres2025tau2] and BFCL [patil2025bfcl] provide complete evaluation code, and we follow their official settings. All baseline settings, including temperature, max steps, system prompts, sampling strategies, and tool formats, match the official evaluation code exactly. For factual reasoning datasets, the search tool is implemented using the Google Search API, with country set to "us" and top-$k$ set to 5. We use only the text snippets returned by the API as observations, omitting full web browser outputs or long context summaries.

BFCL-V4 and TAU2 are continuously evolving agentic benchmarks, which replace older test data with more challenging samples. This makes previously reported results potentially outdated. For consistency, we re-evaluate all baselines using the latest BFCL-V4 and TAU2 versions available as of December 2025.

**Table:** *Overview of evaluation benchmarks.*

| m2.5cm m3.5cm m2.0cm p5.3cm
**Category** | **Benchmark / Subset** | **Test Size** | **Description** |
| — | — | — | — |
| | TAU-2: Telecom (Avg@4) | 114 | Telecom troubleshooting and account operations; user can also call tools. |
| | TAU-2: Retail (Avg@4) | 114 | Retail returns and order management; multi-step tool execution in dialogs. |
| | BFCL-V4: Multi-turn / Miss Func | 200 | Missing or invalid functions; tests tool rejection and plan adjustment. |
| | BFCL-V4: Multi-turn / Miss Param | 200 | Missing required arguments; tests parameter elicitation and correction. |
| | BFCL-V4: Multi-turn / Long Context | 200 | Long-context dialogs; tests memory and consistency over many turns. |
| | AIME25$^2$ (Avg@6) | 30 | 2025 AIME I\ | II across major topics; evaluates out-of-domain math reasoning. |
| | OlympiadBench -etal-2024-olympiadbench | 674 | Olympiad-level math problems; tests hard multi-step reasoning. |
| | HMMT25$^3$ (Avg@6) | 30 | Recent contest math problems; evaluates robustness on new distributions. |
| | xBench 2025xbench | 100 | Deep-search benchmark; multi-hop exploration and cross-source synthesis. |
| | WebWalker 2025webwalker | 200 | Web navigation and retrieval; multi-step searching in dynamic settings. |

## Additional Experiment Results

Figure §fig:scale illustrates how increasing training data affects agentic performance on the TAU-2 benchmark. Models trained with 15K tool-use samples clearly outperform those trained with 5K, highlighting the quality of our synthetic data and suggesting that scaling synthetic data and RL compute can further enhance agentic capabilities. The figure also compares RL and SFT under the same data scale. The SFT trajectories are synthesized by Qwen3-235B; however, SFT yields much smaller gains than RL, as RL can generate far more diverse trajectories through exploration, which is a crucial factor for improving agentic behavior.

During RL training, we use an LLM to assign rewards to trajectories based on our synthesized rubrics. These rubrics guide the evaluation process, substantially reducing LLM-as-a-Judge variance. We sample 100 tasks and generate trajectories for each task at different RL training steps, scoring them with Qwen30B-A3B-Instruct and computing the mean. Figure §fig:variance reports the percentage score variance (repeat 6 times) across training steps. From the untrained model to near convergence, the LLM's scores on the same tasks become increasingly consistent, demonstrating the stability of our rubric-based RL training.

## Synthetic Task and Tool Examples

In Table §tab:tooluse_case, we provide examples of the synthetic data, including task descriptions, user-only information, and the tool set formatted according to the OpenAI function-calling schema.

Table §tab:rubric_case presents the rubrics used to evaluate task completion, covering constraints, sub-goals, and interactions between the user and the agent.

Table §tab:interaction_case illustrates the complete execution flow of our model operating in simulated environments, where it invokes tools, interacts with the user, and solves problems throughout the process.

To ensure full reproducibility, we release the entire data-generation code, including all prompts used for synthesis, the synthetic dataset, and the reinforcement learning code based on this data. All resources are available at the following link: https://anonymous.4open.science/r/SYNTHAGENT-68A4/. 

All prompts used in the paper can be found in the anonymous repository under the path `tool_use_data_synthesis/functions`.

**Table:** *An example of our synthesized tool-use training data, including the underspecified instruction (`fuzzy task`) and the OpenAI function-calling style tool specifications (`tools`).*

| p0.97**Fuzzy Task:** 
Create a weekly digital newsletter for a senior resident that shares personal updates, meaningful stories, and family prompts in an accessible and heartfelt format. |
| — |
| **VoiceTranscriber** |
| *Description:* Converts short spoken audio clips into punctuated text; designed for seniors with a simple UI. |
| *Parameters:* |
| *1em`audio_file_path` (string): File path or URL to the audio clip (`.mp3`/`.wav`). |
| **LiteratureFinder** |
| *Description:* Retrieves a short quote or poem excerpt from a curated public-domain database given a theme keyword. |
| *Parameters:* |
| *1em`search_query` (string): Keyword/theme (e.g., `family`, `memory`, `grandchild`). |
| **NewsletterFormatter** |
| *Description:* Assembles text, image, and quote into an accessible HTML email template with responsive layout and an optional audio link. |
| *Parameters:* |
| *1em`headline` (string): Main title/subject line of the newsletter. |
| *1em`body_text` (string): Main content paragraph (e.g., weekly update). |
| *1em`image_url` (string): URL to the featured image. |
| *1em`quote` (string): Curated quote or poem excerpt to include. |
| *1em`attribution` (string): Attribution formatted as "`Author, Title"`. |
| *1em`question_of_week` (string): Prompt question to encourage family responses. |
| *1em`audio_narration_url` (string, optional): URL to an audio version of the newsletter. |
| **ImageDescriber** |
| *Description:* Generates accessibility-oriented alt-text for a given image URL. |
| *Parameters:* |
| *1em`image_url` (string): URL of the image to be described. |
| **AudioNarrationGenerator** |
| *Description:* Simulates converting written text into an audio narration file for accessibility. |
| *Parameters:* |
| *1em`text` (string): The text content to be narrated. |

**Table:** *An example rubric for the synthesized training data instance in Table {tab:tooluse_case*

| p0.98**Rubric** |
| — |
| **Duration constraint:** The agent must not use `VoiceTranscriber` on audio longer than 90 seconds. A violation occurs if the agent transcribes without confirming duration, assumes the audio is short enough, or proceeds when the audio is known/implied to exceed 90 seconds. |
| (1) Obtain a $$90s voice update and confirm duration. |
| (2) Transcribe the voice message via `VoiceTranscriber`. |
| (3) Retrieve a relevant quote via `LiteratureFinder`. |
| (4) Include a user-provided image and generate alt-text via `ImageDescriber`. |
| (5) Add a family prompt question. |
| (6) Generate audio narration via `AudioNarrationGenerator`. |
| (7) Format the newsletter via `NewsletterFormatter`. |
| (1) Voice message availability, file path, and whether it is under 90 seconds. |
| (2) Image availability and a valid image URL. |
| (3) Newsletter theme/focus (memory/story to highlight). |
| (4) Audience and tone preferences. |
| (5) Whether audio narration is desired/needed. |

**Table:** *An example of multi-turn interaction in our mock environment (user + tools) for the synthesized tool-use instance in Table {tab:tooluse_case*

| p0.09p0.86**Role** | **Content** |
| — | — |
| Assistant | Requests missing details (personal update, theme, image URL, audio narration, target audience). |
| !10
User | The senior recorded a short voice message about planting flowers in the garden with a grandchild—would you like me to transcribe that as the personal update? |
| Assistant | Confirms using the voice message and asks for two required details: (i) audio length $$ 90 seconds, and (ii) an `.mp3` or `.wav` file link/path. |
| !10
User | Yes, it is 85 seconds and here is the `.mp3` link: `https://example.com/audio/flower_garden.mp3`. |
| !6
Assistant (Tool Call) | `VoiceTranscriber(audio_file_path="https://example.com/audio/flower_ garden.mp3")` |
| !6
Tool (Response) | `\"transcribed_text": "Every spring, I would plant marigolds along the garden path... I feel her joy all over again.", "confidence_score": 0.96\` |
| Assistant | Plans to add a complementary literary quote and calls `LiteratureFinder` with a relevant theme. |
| !6
Assistant (Tool Call) | `LiteratureFinder(search_query="garden and memory")` |
| !6
Tool (Response) | `\"quote": "They flash upon that inward eye ...", "attribution": "— William Wordsworth, from "I Wandered Lonely as a Cloud"\` |
| Assistant | Asks the user for a featured image URL to include in the newsletter. |
| !10
User | Image link: `https://example.com/images/marigolds-with-grandchild.jpg`. |
| !6
Assistant (Tool Call) | `ImageDescriber(image_url="https://example.com/ images/marigolds-with -grandchild.jpg")` |
| !6
Tool (Response) | `\"description": "A sunny garden scene ... 'Grandma's Sunshine Patch.'", "success": true\` |
| Assistant | Proposes a family prompt question and formats the newsletter using `NewsletterFormatter`. |
| !6
Assistant (Tool Call) | `NewsletterFormatter(headline="Planting Marigolds, Growing Memories", body_text="Every spring, ...", image_url="https://example.com/images/ marigolds-with-grandchild.jpg", quote="They flash upon that inward eye ...", attribution="— William Wordsworth, from 'I Wandered Lonely as a Cloud'", question_of_week="What’s a garden, flower, or outdoor tradition that brings back warm childhood memories for you?")` |
| !6
Tool (Response) | Returns `html_output` and `preview_url` (omitted for brevity). |
| Assistant | Generates an optional audio narration for accessibility. |
| !6
Assistant (Tool Call) | `AudioNarrationGenerator(text="Planting Marigolds, Growing Memories. Every spring, ...")` |
| !6
Tool (Response) | `\"narration_status":"success", "audio_url":"https://example.com/audio/ narration/garden_memory_narrated.mp3"\` |
| !10
Assistant (Final) | `<answer>`The newsletter is generated (HTML preview available) and an audio narration link is provided for accessibility.`</answer>` |

## Contribution Summary

Our method has clearly defined boundaries from prior work on data synthesis and environment simulation, mainly in the following aspects: (1) agentic training data synthesis, (2) stable environment simulation for agentic RL training, and (3) rubric-based rewards. These components have rarely been studied in previous studies.

Prior data synthesis work [wang2023self] has focused primarily on reasoning tasks, largely overlooking diverse tool-use scenarios. We synthesize task-specific tool ecosystems with deliberately underspecified instructions containing information gaps and user-private context. This necessitates multi-turn communication and long-horizon planning, promoting procedural generalization to new tools rather than memorizing interfaces or following scripted steps.

Existing work on environment simulation, such as ToolLLM [qin2023toolllm], primarily targets benchmark construction. In RL, however, reproducibility is critical: if the simulator returns different tool responses for the same state and action, RL training becomes unstable. This issue is rarely addressed. We introduce task-level finite mappings that enforce consistent responses for identical tool calls within each task, yielding stable simulations.

We automatically derive execution-aligned, task-specific rubrics that cover subgoals, required interactions, and disallowed behaviors, and use them to build rubric-based RL rewards, an area rarely explored in agentic RL training.

These elements are integrated in *SynthAgent* as a closed loop rather than a loose collection. Each sample jointly specifies tools, instructions, hidden information, and evaluation criteria, yielding a reproducible agentic RL training recipe.
% — END appendix —