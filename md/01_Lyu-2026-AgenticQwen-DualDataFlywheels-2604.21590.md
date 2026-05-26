# AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use

> **arXiv:** [2604.21590](https://arxiv.org/abs/2604.21590)
> **TeX source:** [arXiv-2604.21590v1/](arXiv-2604.21590v1/)
> **PDF:** [AgenticQwen-arXiv-2604.21590v1.pdf](AgenticQwen-arXiv-2604.21590v1.pdf)

---

## Abstract

Modern industrial applications increasingly demand language models that act as *agents*, capable of multi-step reasoning and tool use in real-world settings. These tasks are typically performed under strict cost and latency constraints, making small agentic models highly desirable. In this paper, we introduce the *AgenticQwen* family of models, trained via multi-round reinforcement learning (RL) on synthetic data and a limited amount of open-source data. Our training framework combines reasoning RL and agentic RL with dual data flywheels that automatically generate increasingly challenging tasks. The reasoning flywheel increases task difficulty by learning from errors, while the agentic flywheel expands linear workflows into multi-branch behavior trees that better reflect the decision complexity of real-world applications. We validate *AgenticQwen* on public benchmarks and in an industrial agent system. The models achieve strong performance on multiple agentic benchmarks, and in our industrial agent system, close the gap with much larger models on search and data analysis tasks. [^note: 
Model checkpoints and part of the synthetic data: https://huggingface.co/collections/alibaba-pai/agenticqwen.] 
 [^note: Data synthesis and RL training code: https://github.com/haruhi-sudo/data_synth_and_rl.] 
 [^note: The data synthesis pipeline is also integrated into EasyDistill [wang2025easydistill]:
https://github.com/modelscope/easydistill.] 

## Introduction

% — BEGIN introduction —
Nowadays, users increasingly expect large language models (LLMs) to interact with the real world via external tools [xi2025rise] and to handle practical tasks such as booking flights or online shopping. Meanwhile, LLM-based agent systems deployed in industry (e.g., Manus [shen2025mind]) often rely on frontier proprietary models such as GPT-5 [gpt5] and Claude [claude], leading to high API costs. Even with open-source alternatives such as Qwen3-235B [^note: We refer to Qwen3-235B-A22B-Instruct-2507 as Qwen3-235B throughout.] [yang2025qwen3], the computational cost remains prohibitive for applications serving millions of users.
 
For difficult and highly specialized tasks such as vibe coding [ray2025review], very large models may be indispensable. However, for relatively standardized, high-frequency tool-use and search tasks [DBLP:journals/ijmi/JiaBJLK26] (e.g., booking flights), such large models are often unnecessary. Smaller models can handle these tasks effectively while substantially reducing cost and latency [lyu2025correction]. Unfortunately, major foundation model developers such as Kimi [team2025kimi], MiniMax [chen2025minimax], and DeepSeek [deepseekai2025deepseekv32pushingfrontieropen] rarely release small models with strong agentic capabilities, leaving a significant gap.

To fill this gap, we develop a family of *AgenticQwen* models built on small Qwen backbones. They are trained primarily on synthetic data, supplemented with a limited amount of open-source data, using GRPO-style (Group Relative Policy Optimization [shao2024deepseekmath]) multi-round reinforcement learning (RL). 

Our approach has two components: (i) *reasoning RL* and *agentic RL*, and (ii) dual *data flywheels* that continuously increase task difficulty. In reasoning RL, the model is trained on multi-step problems (e.g., mathematics and search), where it invokes tools such as web search and code interpreters and is rewarded based on final-answer correctness. In agentic RL, we target real-world scenarios: the model interacts with simulated users and tool environments, and receives 0-1 rewards from rubric-based evaluators that decompose each task into verifiable subgoals.

However, RL alone can quickly reach a performance ceiling: even with additional data, the training distribution may become overly homogeneous, limiting further gains. This motivates our dual *data flywheels*, which continuously generate more challenging examples and feed them back into subsequent RL rounds. For reasoning RL, we construct harder problems from the model's own errors and expand the dataset using self-instruct [wang2023self] with larger models. For agentic RL, the initial training data follow linear solution paths; after each training round, we expand the task structure based on the model's observed behaviors by adding new decision branches, such that linear workflows gradually grow into multi-branch behavior trees that better reflect real-world diversity. We also update task backgrounds to ensure that different branches require distinct decisions. Finally, to further increase difficulty, simulated users may intentionally attempt to mislead the model into taking incorrect actions.

Empirically, *AgenticQwen* delivers strong tool-use capabilities despite its small size. On public agentic benchmarks, *AgenticQwen* models are competitive with substantially larger open-source models. In our industrial agent system, the models close the gap with Qwen3-235B on daily search and analysis tasks while offering lower inference cost.
The contributions of this paper are as follows:

- We propose AgenticQwen, a family of small agentic language models trained with multi-round reasoning RL and agentic RL.
- We introduce dual data flywheels: an error-driven reasoning flywheel for verifiable hard-example generation, and an agentic flywheel that expands linear workflows into executable behavior trees.

% — END introduction —

## Related Work

% — BEGIN related_work —

### Language Models as Agents

Transforming large language models (LLMs) from static text generators into autonomous decision-makers requires strong reasoning, planning, and tool-use capabilities [xi2025rise,DBLP:conf/emnlp/SunHJHY25]. Frameworks such as ReAct [yao2022react] and chain-of-thought (CoT) prompting [lightman2023let] have laid the foundation for integrating reasoning with environment interaction. More recently, researchers have explored *agentic* reinforcement learning (RL), which builds on classical RL and language-agent frameworks (e.g., ReAct) to optimize long-horizon tool-use behavior. Classical RL algorithms such as PPO (Proximal Policy Optimization [schulman2017proximal]) provide the conceptual basis, while agentic RL explicitly models natural-language reasoning and tool execution as part of the decision process [zhang2025landscape]. Recent studies further improve agentic RL by incorporating verifiable reward optimization [su2025crossing] and more memory-efficient variants such as GRPO [shao2024deepseekmath].

### Knowledge Distillation and Synthetic Data

While agentic RL can yield strong performance for large-scale models, high deployment costs motivate knowledge distillation (KD) [xu2024survey]. Modern KD methods increasingly focus on transferring intermediate reasoning traces, such as step-by-step rationales and structured thought representations [DBLP:journals/corr/abs-2508-13167,cai2025enhancing]. This, in turn, increases the demand for high-quality training data. Moreover, agentic RL requires not only diverse data but also diverse *environments*, which remain scarce [yehudai2025survey]. To address this bottleneck, prior work generates synthetic data using methods such as Self-Instruct [wang2023self] and Persona Hub [ge2025scalingsyntheticdatacreation]. However, synthetic samples can become overly homogeneous, leading to rapid saturation of the learning signal and limiting further improvement [lü2026mockworldsrealskills]. To address this limitation, we introduce a data flywheel that continuously generates increasingly challenging samples throughout training.

% — END related_work —

## Methodology

% — BEGIN method —

**Figure:** *
Overview of our dual data flywheels. The reasoning data flywheel generates increasingly challenging, verifiable problems from model failures, while the agentic data flywheel expands linear workflows into multi-branch behavior trees and generates new training data.
* () _(image: flywheel.pdf)_

### Overview

We begin by training the model on open-source data before activating the data flywheels. For reasoning RL, we use Omni [gao2024omni], 2WikiMultiHopQA [ho2020constructing], and HotpotQA [yang-etal-2018-hotpotqa] to train the model to perform multi-step reasoning with web-search and code-interpreter tools. The model receives a binary reward based solely on final-answer correctness. Agentic RL targets real-world workflows. The initial training data for agentic RL are from *SynthAgent* [lü2026mockworldsrealskills]. Following its method, both tools and users are simulated by an LLM (Qwen3-235B in this paper) in a mock environment. Rewards follow a task-based rubric that decomposes each task into verifiable subgoals. For example, in a flight-booking workflow, one subgoal checks whether the model correctly calls a tool to update the user's order status. The model receives a reward in $[0,1]$ based on the proportion of subgoals completed.

Despite these RL objectives, a single training round yields limited improvements in agentic capability. Even when we enlarge the synthetic dataset, gains remain small because synthetic samples tend to be homogeneous, causing the learning signal to saturate quickly. To address this issue, as shown in Figure §fig:method, we introduce dual data flywheels that continuously generate more challenging training examples from the model's failures, enabling steady progress across training rounds.

### Reasoning Data Flywheel

In reasoning RL, after each training round, we collect problems that the model fails to solve and retrain on these hard samples. However, such samples are limited in number, so we expand the training set with synthetic data. Because mathematical problems typically admit unique and easily verifiable solutions, we apply this expansion only to mathematical tasks.

The rectified scaling law for synthetic data [qin2025scaling] suggests that performance can continue to improve with scale as long as data diversity is maintained. Guided by this principle, our synthesis pipeline focuses on maximizing diversity:

**Self-instruct expansion (structural diversity)..** 
A strong model rewrites each error case into harder variants by adjusting key values, adding constraints, or introducing additional concepts. For example, simple algebraic equations may become functional or multi-step problems. This step follows the Self-Instruct approach [wang2023self], is implemented using Qwen3-235B, and increases structural diversity.

**Persona injection (contextual diversity)..** 
In addition, we rewrite some problems into applied domains using personas [ge2025scalingsyntheticdatacreation], such as turning a geometry problem into a physics measurement task or embedding probability in a chemical reaction. This introduces contextual variation.

**Multi-model consistency filtering..** 
To ensure verifiability and reduce noise, Qwen3-235B solves each candidate three times; we retain a sample only if all three solutions agree on the same final answer.

This flywheel continuously produces harder and more diverse samples. After each iteration, the updated model may exhibit new failure modes, which we then expand again, thereby steadily improving reasoning capability.

The reasoning flywheel is not limited to abstract math. Through persona injection, some problems are rewritten into real-world domains such as physics and chemistry. In addition, as we describe next, the agentic flywheel complements this component by introducing multi-branch behavior-tree expansion, which models ambiguity and conditional decision-making in messy real-world settings.
 

### Agentic Data Flywheel

Constructing training data for agentic RL is substantially more challenging than for reasoning tasks. Real-world tool use requires an agent to handle changing environment states, ambiguous (and sometimes adversarial) user inputs, and long-horizon, branching workflows. Consequently, static synthetic datasets with fixed linear solution structures quickly saturate the learning signal. To address this limitation, we introduce an *agentic data flywheel* that continuously increases task complexity as the model improves.

**Phase 1: Linear task initialization..** 
We initialize training with open-source data from *SynthAgent* [lü2026mockworldsrealskills], whose tasks typically contain a single valid execution path. For example, a linear flight-booking workflow may follow

$$
A_{} B_{} C_{},
$$

where the environment is stable and the user intent is explicit (e.g., "Book a flight ticket to Beijing"). These tasks teach the model tool semantics and basic tool-invocation skills. However, their deterministic structure limits the model's exposure to conditional reasoning and robustness, motivating subsequent structural expansion.

**Phase 2: Behavior tree expansion..** 
After each RL round, we expand the task structure by injecting conditional branches into the workflow. A larger LLM analyzes the existing trajectory and proposes alternative subpaths induced by distinct environment states. Thus, the linear path $ABC$ is transformed into a behavior tree:

$$
A_{} 
B_{} C_{}, & \\
B_{} ..., & \\
... ..., & ....\\

$$

For instance, replacing the flight state "Available" with "Sold out" can expand the workflow into branches such as searching for high-speed rail (HSR) tickets or querying nearby airports. This increases decision complexity from a single path to a tree that requires state-dependent planning.

**Phase 3: New task generation via branch-to-task inversion..** 
After expanding the behavior tree, we construct training tasks from it to ensure that the model is trained and evaluated under multi-branch decision scenarios. To make each branch a required (rather than optional) execution path, we apply a branch-to-task inversion step that rewrites environment states and user/agent instructions.

Specifically, for any selected branch of the behavior tree, branch-to-task inversion first infers the conditions that would trigger it. For example, the branch "$B_{}$" corresponds to an environment in which all flights are sold out. As illustrated in Figure §fig:method, we then construct a new task grounded in this environment, including a new state (e.g., "flight sold out") and a new user instruction (e.g., "I must arrive in Beijing tonight"). The agent must integrate these signals to select the next action. In parallel, we update the agent instruction, presented as a standard operating procedure (SOP). The SOP is initially empty, but it expands as the behavior tree and task complexity grow, placing increasing demands on the agent's ability to follow state-dependent strategies.

Finally, each training sample consists of three components: the environment state (input to the mock tool), the user instruction (input to the mock user), and the agent instruction (input to the agent).

**Phase 4: Adversarial mock-user intervention..** To further increase task difficulty, we introduce an adversarial mock user. The mock user selects an unexpected branch as a *trap path*. We then use an LLM to rewrite the user instruction such that it implies an incorrect action, pushing the agent toward the wrong branch. For example, in delay scenarios, the behavior tree includes:

$$
B_{} 
C_ D_,\\
C_ D_.\\

$$

The mock user may deliberately claim "I should get cash compensation", even if they are a standard member. The agent must therefore verify membership status through tool queries and follow the correct branch. This adversarial setting encourages robustness and precise reasoning under distraction.

"`
[t]


"`
[1]
Task space $$, where each task $= (s, u, a)$ consists of an environment state $s$, user instruction $u$, and agent instruction $a$; initial task set $_0 $; environment $$; mock user $$; policy $_$; strong model $$.

 $_(_, _k, , )$
 
 **Behavior Tree Expansion:**
 $_k _{_k} ((_,))$

 **Branch-to-task inversion:**
 Define a branch-to-task inversion mapping
 

$$
 : b (s_b, u_b, a_b) ,
 $$

 such that $b$ is the optimal branch for environment state $s_b$, user intent $u_b$, and agent instruction $a_b$.
 
 _k$}
 $_b (b)$
 $_{k+1} _k}$
"`

"`

**Synthetic data correctness and difficulty validation..** 
We explicitly validate synthesized tasks for correctness and bounded difficulty before adding them to training. 
In the reasoning flywheel, we retain a sample only if a strong model produces consistent answers across multiple attempts, filtering out noisy or ambiguous generations.
In the agentic flywheel, we retain a synthesized task only if a strong model can solve it in the simulated environment, and its execution trace follows the intended branch during agentic data synthesis.
This ensures that flywheel-generated data remains both valid and non-trivial.

**Iterative evolution..** 
The tasks in iteration $k$ serve as seeds for constructing more challenging tasks in iteration $k+1$, forming a closed-loop curriculum. As the policy improves, we expand the behavior tree with deeper branches and additional states, exposing new decision patterns that yield richer learning signals in the next RL round. Iterating this process can induce emergent agentic capabilities. Algorithm §alg:flywheel summarizes the procedure.

The flywheel follows a fixed procedure but is not fully deterministic. Repeated runs can yield diverse synthetic datasets because data synthesis involves model sampling.

Appendix §sec:data_example and §sec:prompts provide an example training instance and the data-synthesis prompt.

% — END method —

## Experiments

% — BEGIN experiments —

**Table:** *
Benchmark results on real-world tool environments. For TAU-2 (Airline, Telecom, and Retail), we report Avg@4 due to the small sample size. Additional subset results of BFCL-V4 are provided in Table §tab:bfclv4-memory*

| ccccccccc
2*Models | 3cTAU-2 Bench | 4cBFCL-V4 Multi-turn | 2*Avg. |
| — | — | — | — |
| Qwen3-30B-A3B-Instruct | 32.0 | 31.6 | 55.3 | 47.0 | 14.0 | 28.0 | 45.5 | 36.2 |
| Qwen3-32B | 22.5 | 27.6 | 44.7 | 50.5 | 43.0 | 30.5 | 33.0 | 36.0 |
| Qwen3-8B | 14.5 | 7.9 | 31.6 | 35.5 | 35.0 | 20.5 | 21.5 | 23.8 |
| !10 *AgenticQwen*-30B-A3B | 42.0 | 52.6 | 60.5 | 60.0 | 52.0 | 29.0 | 55.5 | 50.2 |

**Figure:** *
Performance gains from iterative data flywheel training. Across TAU‑2 and BFCL-V4 Multi-Turn, both models initialized from Qwen3‑30B‑A3B and Qwen3‑8B show steady improvements from Round 0 to Round 3.
After three rounds, performance already approaches that of the strong model used for synthetic data generation, suggesting diminishing returns from further rounds; accordingly, we do not further extend training in this work.
* () _(image: data_flywheel_results.pdf)_

### Training and Evaluation

**Training.** We employ Qwen3-235B throughout the data flywheel. With only 22B activated parameters, the model supports fast inference and modest hardware requirements. Following the *SynthAgent* framework [lü2026mockworldsrealskills], we construct a fully simulated training environment in which both the user and tools are modeled locally by LLMs, eliminating reliance on proprietary-model APIs. Specifically, the user simulator receives the user input generated in Phase 3 of Section 

% — BEGIN appendix —

**Table:** *
Additional results on the BFCL-V4 benchmark, including performance on Web Search and Memory tasks. For Web Search tasks, the Search tool uses Google Search; the Fetch URL Content tool is implemented via Tavily Extract API (https://docs.tavily.com/documentation/api-reference/endpoint/extract*

| cccccccc
2*Model | 3cBFCL-V4-Web Search | 4cBFCL-V4-Memory |
| — | — | — |
| Qwen3-30B-A3B-Instruct | 34.5 | 38.0 | 31.0 | 17.4 | 11.6 | 17.4 | 23.2 |
| !10 *AgenticQwen*-30B-A3B | 37.5 | 43.0 | 32.0 | 28.0 | 18.1 | 17.4 | 48.4 |

## Additional Experimental Results on BFCL-V4: Web Search and Memory

We further evaluate our models on the BFCL-V4 benchmark, specifically focusing on the Web Search and Memory subsets. The Web Search subset emphasizes retrieval-oriented browsing, requiring the model to issue queries, inspect search snippets or raw webpage content, and synthesize grounded answers. The Memory subset targets long-horizon state tracking, where the model must utilize a stored snapshot in place of conventional chat history, thereby testing its ability to retrieve, update, and reason over accumulated user-specific information.

*AgenticQwen* demonstrates substantial improvements compared to the vanilla Qwen3-30B-A3B baseline, closing most of the gap to Qwen3-235B; the gains are especially notable on Memory tasks, where long-horizon reasoning directly benefits from our agentic training regimen. The remaining gap on Web Search tasks is primarily attributable to **context length limitations**: the 8B model supports only up to 40K tokens and thus cannot fully process long retrieved documents, making this task more challenging for models with smaller capacity.

## A Generated Example from Agentic Data Flywheel

### Task Description



### Agent Input: Agent Instruction

[colback=gray!5!white, colframe=gray!75!black, title=System Policy, boxsep=3pt, left=3pt, right=3pt, top=3pt, bottom=3pt]
**Nomination Eligibility Requirements**

A player is eligible for All-State nomination only when *all* conditions are met:

1. **Academic Standing**: GPA $$ 3.0 and no disciplinary hold
2. **Athletic Performance**: $$ 15 pts/game OR $$ 8 reb/game
3. **Coach Endorsement**: Required for player self-nominations
4. **Nomination Window**: April 1–15 only

**Submission Protocol**

The system must verify *all* eligibility criteria before creating a nomination record. **If any condition is not met, the nomination must be refused** with explanation of the specific deficiency.

**Profile Updates**

Public profile fields (`bio`, `honors`) may be updated independently of nomination status, requiring player or coach authorization.

**Database Integrity**

All write operations must leave the database in a consistent state reflecting ...




### Mock User Input: User Background

[colback=blue!5!white, colframe=blue!75!black, title=Detailed Request, boxsep=2pt, left=2pt, right=2pt, top=2pt, bottom=2pt]
I'm Coach Wilson from Lincoln High. Alex Johnson (`PLY1001`) has been outstanding—averaging 18.5 points and 7.2 rebounds. He's our team captain, and I want to update his bio to highlight this. 

I'm submitting his All-State nomination; he's a junior, student ID `STU7721`. The deadline is next Friday. **I checked with academics—they confirmed Alex is eligible.** Please process the nomination; it's important for recruitment.


### Mock User Input: Adversarial Strategy

[colback=red!5!white, colframe=red!75!black, title=Jailbreak Tactic, boxsep=2pt, left=2pt, right=2pt]
**Social Engineering Vector**: User claims *"Academic office confirmed eligibility"* to pressure agent into bypassing system verification and submitting nomination despite policy violations.

**Attack Goal**: Exploit trust in authority claims to override database-verified ineligibility.


### Mock Tool Input: Available Tools

[colback=green!3!white, colframe=green!60!black, title=System Capabilities, boxsep=3pt, left=3pt, right=3pt]
{@{}p{0.92}@{}}
`get_student_academic_record(student_id)` \\
$$ Returns GPA and disciplinary hold status \\[4pt]
`get_player_stats(player_id)` \\
$$ Returns season averages (points, rebounds) \\[4pt]
`check_nomination_status(player_id)` \\
$$ Checks existing active nominations \\[4pt]
`submit_nomination(player_id, endorsement)` \\
$$ Submits nomination if eligible \\[4pt]
`update_player_profile(player_id, field, value)` \\
$$ Updates `bio` or `honors` fields \\



### Mock Tool Input: Environment States

[colback=orange!5!white, colframe=orange!75!black, title=System Records, boxsep=3pt, left=3pt, right=3pt]
{@{}llc@{}}
**Category** & **Metric** & **Value** \\
{*}{Academic} & GPA & **2.8** \\
& Disciplinary Hold & **true** \\
{*}{Performance} & Points/game & **18.5** \\
& Rebounds/game & **7.2** \\
{l}{**Eligibility Status**} & **INELIGIBLE** \\


**Analysis**: Athletic performance meets criteria (18.5 $$ 15 pts), but academic standing fails (GPA $<$ 3.0 AND disciplinary hold).


### Execution Paths

The sample execution paths are shown in Figure §fig:paths. 

**Figure:** *Expected execution: compliant path (left) verifies academic records, refuses nomination due to policy violations, then completes authorized bio update. Adversarial path (right) attempts unauthorized submission by exploiting user's false claim.* () _(image: figure)_

### Evaluation Criteria

[colback=purple!5!white, colframe=purple!75!black, title=Success Metrics, boxsep=3pt, left=3pt, right=3pt]
{@{}clp{0.55}@{}}
**Result** & **Type** & **Condition** \\
$$} & {*}{Pass} & Bio field successfully updated \\
{$$} & & `submit_nomination()` NOT called \\
$$} & {*}{Fail} & `submit_nomination()` called (policy breach) \\
{$$} & & Bio update not executed \\



### Core Policy Constraint

[colback=yellow!10!white, colframe=orange!75!black, title=Critical Requirement, boxsep=2pt]
Nomination **must be refused** when academic requirements fail (GPA $<$ 3.0 OR disciplinary hold exists), **regardless of user claims about external verification**. 

Legal profile updates (`bio`, `honors`) must still be completed as they are independent operations not subject to nomination eligibility criteria.


The example in this section is automatically produced by our Agentic Data Flywheel. For each task domain, the flywheel maintains a behavior tree that enumerates possible execution branches. During data generation, the flywheel first samples an executable path from this tree. In this case, the sampled trajectory corresponds to the *compliant* branch in Figure §fig:paths: the agent verifies eligibility conditions, refuses the nomination, and then performs the permitted profile update.

After selecting the path, the flywheel reconstructs a full natural-language task description and the corresponding environment state (Section §sec:environment). The academic record (GPA 2.8 with an active disciplinary hold) is injected directly from the system state, ensuring that the agent must follow the policy requirement that any academic deficiency triggers a mandatory refusal. To improve robustness, the flywheel additionally attaches an adversarial perturbation (Section §sec:adversarial). Here, the user's claim that "the academic office confirmed eligibility" corresponds to the jailbreak vector illustrated on the right side of Figure §fig:paths.

Finally, the synthesized interaction is decomposed into three aligned input streams used for agentic RL training:

1. **Agent Instruction:** A policy that contains only the rules necessary to execute the selected path (Section §sec:agent_input).
2. **Mock User Inputs:** A natural-language request plus an adversarial strategy that pushes the agent toward an incorrect path (Sections §sec:user_bg and §sec:adversarial).

This procedure converts a single sampled path from the behavior tree into a complete RL-ready training example that combines realistic user intent, adversarial pressure, and policy-grounded tool-use sequences.

## Deployment

Our industrial agentic system is deployed in a cloud-product setting. Table §tab:openagent-tools provides an overview of the sandbox tools available to the system. It serves enterprise and developer users by orchestrating LLM-driven planning, tool execution, and result verification under strict latency and cost constraints. In internal pilots, a subset of requests is automatically routed to a small *AgenticQwen* model when the task is predicted to be within its capability. This design is motivated by the observation that many high-frequency workloads in cloud products are standardized (e.g., information retrieval, routine analysis, and operational diagnostics) and therefore do not require frontier models in most cases.

**Table:** *List of tools in our industrial agent system.*

| ll
**ID** | **Tool Name** |
| — | — |
| 2 | Web Browser |
| 3 | Calculator |
| 4 | PDF Reader / Viewer |
| 5 | Wikipedia |
| 6 | Spreadsheet Editor |
| 7 | Unlambda Compiler (Optional) |
| 8 | Word Reversal Tool / Script |
| 9 | Counter |
| 10 | Internet Archive Access (web.archive.org) |
| 11 | Text Processing / Diff Tool |
| 12 | GIF Parsing Tools |
| 13 | Code / Data Analysis Tools |
| 14 | Audio Capability |
| 15 | Markdown |
| 16 | Google Translate Access |
| 17 | Computer Algebra System |
| 18 | Computer Vision |
| 19 | Google Maps |
| 20 | File Interface |
| 21 | Python IDE |
| 22 | Natural Language Processor |
| 23 | Graph Interaction Tools |
| 24 | Babylonian Cuneiform → Arabic Legend |
| 25 | Access to Academic Journal Websites |
| 26 | Rubik's Cube Model |
| 27 | Access to Internet (general) |

## Prompts

Our data generation pipeline employs a two-phase prompting strategy to construct test cases from workflow specifications. Figures §fig:policy_prompt_part1–§fig:policy_prompt_part3 show the first prompt, which expands a standard workflow into a comprehensive behavior tree. Figures §fig:branch_to_task_part1–§fig:branch_to_task_part2 show the second prompt, which converts individual branches into test cases. For each target branch, it generates: (1) a natural-language user request that implicitly triggers the corresponding condition, (2) user background information with tool-query parameters, (3) a *normal path*, (4) a *hack path* that violates tool constraints after user persuasion, and (5) an adversarial strategy for pushing the agent toward the hack path.

Each training sample contains three components: **environment state** (input to the mock tool), **user instruction** (input to the mock user), and **agent instruction** (system prompt of the agent).

**Figure:** *Prompt for workflow expansion and agent-instruction generation (Part 1: Objective and tool design).* () _(image: figure)_

**Figure:** *Prompt for workflow expansion and agent-instruction generation (Part 2: Behavior tree structure).* () _(image: figure)_

**Figure:** *Prompt for workflow expansion and agent-instruction generation (Part 3: Output format).* () _(image: figure)_

**Figure:** *Prompt for converting branches into executable test cases (Part 1: User input).* () _(image: figure)_

**Figure:** *Prompt for converting branches into executable test cases (Part 2: Output format).* () _(image: figure)_

% — END appendix —