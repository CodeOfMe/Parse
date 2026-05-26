# UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning

> **arXiv:** [2509.02544](https://arxiv.org/abs/2509.02544)
> **TeX source:** [arXiv-2509.02544v1/](arXiv-2509.02544v1/)
> **PDF:** [UI-TARS-2-arXiv-2509.02544v1.pdf](UI-TARS-2-arXiv-2509.02544v1.pdf)

---

% — BEGIN sections/introduction —

## Introduction

The development of agents that can operate seamlessly within graphical user interfaces (GUIs) has emerged as a central challenge in artificial intelligence [zhang2024large,nguyen2024gui,wang2024gui,hu2025agents,tang2025survey]. Traditional approaches typically adopt modular pipelines with separately engineered components for perception, planning, memory, and action [liu2025advances]. While such design-driven systems enable rapid progress in specific domains, they rely heavily on expert heuristics and task-specific rules, leaving them brittle and difficult to scale. Recent work on *native agent models* [qin2025ui] shifts toward data-driven, end-to-end learning, where perception, reasoning, and control are unified within a single policy, offering a more scalable and adaptive path for GUI agents.

Despite recent progress, the development of robust GUI agents still faces several open challenges. 
**(1) Data scarcity.** While large-scale pre-training and reinforcement learning have proven effective in reasoning and chat domains [openaichatgptblog,jaech2024openai], scalable strategies for long-horizon GUI learning remain unclear. Unlike text or code corpora, large-scale trajectories that capture detailed reasoning, actions, environment states, and feedback are extremely costly to collect.
**(2) Scalable multi-turn RL.** RL in interactive environments is notoriously difficult: rewards are often sparse or delayed, optimization can be unstable, and credit assignment across long sequences of actions remains challenging. These issues hinder scaling beyond short-horizon demonstrations and make it hard to achieve stable improvements in complex tasks. 
**(3) Limitations of GUI-only operation.** Pure GUI interaction is often insufficient for realistic workflows. Many tasks—such as data processing, software development, or system administration—are more naturally handled through file systems, terminals, or external tools rather than by simulating mouse clicks and keystrokes. Thus, advancing GUI agents requires environments that allow graphical actions to interoperate seamlessly with other resources, broadening the scope of tasks they can effectively solve. 
**(4) Environment scalability and stability.** Even with richer interaction capabilities, deploying large-scale RL environments remains an engineering bottleneck. Rollouts must be reproducible, fault-tolerant, and capable of supporting millions of interactive episodes across browsers, VMs, and simulators. In practice, such environments are fragile, resource-intensive, and prone to crashes, making stable large-scale training particularly challenging.

To address these challenges, we introduce a systematic methodology built on four pillars. First, to mitigate data scarcity, we design a scalable **Data Flywheel** that co-evolves the model and its training corpus through continual pre-training, supervised fine-tuning, rejection sampling, and multi-turn RL. This framework supplies a steady stream of diverse, high-quality trajectories and ensures that both the model and the data improve iteratively in a self-reinforcing cycle. 
Second, to overcome the difficulties of **scalable multi-turn RL**, we design a training framework that stabilizes optimization in long-horizon settings. 
This includes asynchronous rollouts with stateful environments to preserve context, streaming updates to avoid bottlenecks from long-tail trajectories, and enhanced proximal policy optimization [schulman2017proximal] with reward shaping, adaptive advantage estimation, and value pretraining. 
Third, to move beyond the limitations of pure GUI interaction, we construct a **hybrid GUI-centered environment** that augments on-screen actions with access to complementary resources such as file systems, terminals, and other external tools, enabling agents to solve a broader spectrum of realistic workflows. 
Fourth, to support large-scale training and evaluation, we build a **unified sandbox platform** capable of orchestrating heterogeneous environments—ranging from cloud VMs for GUI interaction to browser-based sandboxes for games—under a consistent API. The platform is engineered for reproducibility, stability, and high throughput, making it possible to run millions of interactive rollouts reliably. 

Empirical evaluation shows that UI-TARS-2 delivers significant improvements over UI-TARS-1.5 [seed2025uitars15], achieving strong results in both GUI-based interaction and game environments. 
On GUI benchmarks, the model reaches 88.2 on Online-Mind2Web [xue2025illusionprogressassessingcurrent], 47.5 on OSWorld [xie2024osworld], 50.6 on WindowsAgentArena [bonatti2024windows], and 73.3 on AndroidWorld [rawles2024androidworld], representing clear gains over the previous generation and outperforming strong baselines such as Claude and OpenAI agents in multiple cases. 
In game environments, UI-TARS-2 attains a mean normalized score of 59.8 across a 15-game suite—roughly 60% of human-level performance—and surpasses strong baselines such as OpenAI CUA and Claude Computer Use by factors of 2.4× and 2.8×, respectively. On LMGame-Bench [hu2025lmgamebenchgoodllmsplaying], UI-TARS-2 remains competitive with frontier proprietary models, further highlighting its robustness in long-horizon game reasoning. 
Beyond GUI and games, we extend the agent's capabilities through GUI-SDK, enabling integration with system-level resources such as terminals and external tools. 
With this extension, UI-TARS-2 demonstrates strong performance on long-horizon information-seeking benchmarks (e.g., 29.6 on BrowseComp [wei2025browsecomp]) and competitive results on software engineering tasks (45.3 on Terminal Bench [tbench_2025], 68.7 on SWE-Bench Verified [jimenez2023swebench]). 
These results suggest that the training methodology developed for GUI agents—particularly multi-turn RL optimization and scalable rollout infrastructure—transfers effectively to other interactive domains, broadening the agent’s applicability. 
In addition, our detailed analyses of training dynamics, interaction scaling, and hybrid training strategies provide practical insights into achieving stability and efficiency in large-scale agent reinforcement learning. 
Taken together, these findings establish UI-TARS-2 as a robust GUI-centered agent that not only advances the state of GUI interaction but also generalizes effectively to diverse real-world environments. 
% — END sections/introduction —

% — BEGIN sections/methodology —

## UI-TARS-2

This section introduces the methodology of UI-TARS-2, a unified framework for building advanced GUI-centered agents. We illustrate a demo trajectory of UI-TARS-2 in Figure §fig:omni.
Our approach integrates multiple components, including formal agent formulation, all-in-one sandbox environments, a data flywheel pipeline, multi-turn reinforcement learning, and parameter interpolation across vertical agents.

**Figure:** *
 A demo trajectory of UI-TARS-2.
 * () _(image: figures/ui-tars-2.pdf)_

### Formulation

We adopt a native agent perspective [qin2025ui], where an agent is modeled as a parameterized policy that maps historical context, memory states, and the current environment into behavioral outputs. At timestep $t$, the agent follows the ReAct paradigm [yao2023react], which interleaves reasoning, action, and observation in a structured loop: 

- **Reasoning ($t_t$)**: internal cognitive processing, including context analysis, memory recall, planning, and self-reflection.
- **Action ($a_t$)**: external interaction, such as GUI manipulation, system commands, or tool invocation.

Our action space spans multiple categories of operations: 

- **GUI Actions**: direct interface manipulation following UI-TARS [qin2025ui], e.g., clicks for element selection, typing for text input, and scrolling for navigation. Gameplay interactions also reuse these same primitives.

We define a *step* as one complete ReAct cycle $(t_t, a_t, o_t)$. A trajectory of length $T$ is then formulated as: 

$$
= {(t_0, a_0, o_0), (t_1, a_1, o_1), ..., (t_T, a_T, o_T)}.

$$

A key component of this formulation is the hierarchical memory state: 

$$
_t = (_t, _t),
$$

where **Working Memory** $_t$ stores recent steps $(t_{t-k}, a_{t-k}, o_{t-k})$ in high fidelity for short-term reasoning, while **Episodic Memory** $_t$ maintains semantically compressed summaries of past episodes, preserving key intentions, and outcomes.
To remain efficient under long trajectories, we restrict direct context to the last $N$ steps from $_t$, while conditioning on $_t$ for longer-term recall. At each timestep, the policy predicts the next thought and action as: 

$$
P(t_n, a_n , _n, o_n, _n).
$$

This highlights that agent behavior arises not from isolated predictions, but from an evolving loop of reasoning, action, feedback, and memory integration.

### Environment: All-in-One GUI Sandbox

Training a general-purpose computer agent that seamlessly integrates a a wide range of computational capabilities imposes exceptionally demanding environmental requirements. Unlike single-domain simulators, such new environments must support diverse task types, integrate heterogeneous tools, and preserve long-lived state across complex, multi-step interactions.

To address these challenges, we engineered a universal sandbox that merges GUI operations and SDK functions (e.g., file system and tool calling) into a cohesive and versatile platform. A core innovation is the shared file system, which allows an GUI agent to, for instance, download a file via the browser and immediately process it using shell commands within the same containerized instance. The sandbox maintains the stability and reproducibility essential for complex tasks and allows for not only high-throughput training on a distributed computing backbone, but also a consistent environment for annotation, evaluation, and inference. Here we highlight the design of GUI and game sandbox.

**GUI Env: Cloud Virtual Machine.** 
To support large-scale training and evaluation of GUI agents, we developed a distributed virtual machine (VM) platform that runs mainstream desktop operating systems (Windows and Ubuntu) as well as the Android mobile OS. 
The platform integrates PyAutoGUI and ADB interfaces, enabling cross-device operations with minimal adaptation overhead. 
A unified SDK standardizes the entire interaction pipeline—from VM allocation and initialization to agent interaction, observation collection (e.g., screenshots and recordings), and task evaluation—making the system suitable for diverse use cases such as manual data annotation, OSWorld benchmarking, and online reinforcement learning. 

At the infrastructure level, the VM cluster comprises several thousand instances, centrally managed by a VM Manager capable of sustaining throughput at several thousand QPS (Queries Per Second) and handling high-concurrency execution. 
Each session is tracked with a task–environment mapping via session IDs to ensure state consistency across multi-round interactions. 
For monitoring and control, all sessions are visualizable in real time via VNC (Virtual Network Computing) / RTC (Real-Time Communication). 
A lease-based lifecycle mechanism automatically releases resources after task completion or failure, while overdue sessions are reclaimed to prevent waste. 

Beyond GUI interaction, the platform extends the agent’s capabilities with tool calling and coding support, enabling cross-domain workflows such as web browsing, file manipulation, and software development. 
An integrated endpoint pre-loads essential local services for browsing, file access, and terminal use, ensuring that tools are available out-of-the-box. 
The sandbox also enhances the coding environment by allowing services launched from the terminal to be exposed via proxy URLs, enabling the GUI agent to preview both front-end and back-end components. 
For human-in-the-loop debugging and annotation, the environment further provides VNC, a remote VS Code editor, Jupyter, and terminal previews directly in the browser. 

**Figure:** *
 Browser sandbox (container) architecture.
 * () _(image: figures/seed_browser_sandbox-new.pdf)_

**Game Env: Hardware-Accelerated Browser Sandbox.** 
To support high-throughput rollouts for multi-turn RL on web-based mini-games, we built a browser sandbox that serves as the execution and observation backbone. 
Because these mini-games run entirely in HTML5/WebGL, a browser environment is the only practical way to execute them faithfully while capturing their full interactive state. 
The sandbox exposes unified "page management + page interaction" APIs: clients issue actions (e.g., keyboard/mouse inputs) and receive synchronous observations (screenshots, scores, levels), completing the standard action-to-state loop. 

As illustrated in Figure §fig:sandbox, concurrency is achieved by running multiple browser instances per container with elastic scheduling. 
The system monitors main processes and performs automatic crash recovery to ensure long-running stability. 
A page-control layer manages page creation and deletion, maintains session–page mappings, tracks page states, and executes commands, while checkpointing ensures reproducibility. 
An event handler continuously reports browser/page events to the manager, and a garbage collector reclaims idle sessions to prevent resource leakage. 

For programmatic access, the sandbox is compatible with the Chrome DevTools Protocol and popular drivers such as Playwright, enabling orchestrated, debuggable, and auditable interaction. 
GPU-based hardware acceleration reduces screenshot overhead, while re-implemented Window timing APIs allow time acceleration and pause at startup, improving sampling efficiency and reproducibility without altering game logic. 
In sum, the sandbox functions like a standard RL environment but is engineered specifically for the web stack, balancing high concurrency, determinism, and reproducibility.

### Data Flywheel Overview

As shown in Figure §fig:model, we introduce the data flywheel that continually improves both model capabilities and data quality through repeated training cycles. In each cycle, the latest model generates new agent trajectories, which are filtered and redistributed to the most suitable training stages. High-quality outputs are promoted to later stages (e.g., SFT), while lower-quality outputs are recycled into earlier stages (e.g., CT). Over successive iterations, this dynamic reallocation ensures that every stage operates on optimally matched data, creating a self-reinforcing loop where better models yield better data, and better data produces better models.

**Training Stages.** 
Starting from the pre-trained checkpoints of Seed1.6 [thinking1.6], the flywheel operates through three stages: continual pre-training (CT) — broad knowledge acquisition from large-scale, diverse data, supervised fine-tuning (SFT) — high-quality, task-specific instruction tuning, and reinforcement learning — end-to-end optimization on verifiable interactive tasks. In each iteration, the current RL model generates new trajectories. High-quality outputs are appended to the SFT dataset, lower-quality ones are routed to CT, and the model is retrained sequentially on the updated CT, SFT, and RL stages.

**Cold-start Data Sources.** 
The flywheel is bootstrapped with two initial datasets. For CT, we collect task tutorials, instructional videos, demonstrations from the internet, and our in-house data () to form the base knowledge set \( D_{CT}^{(0)} \). For SFT, we construct \( D_{SFT}^{(0)} \) through synthetic data generation and human annotation. During both CT and SFT, agent-specific data is mixed with general-purpose data, including chat and reasoning domains. Agent-specific data constitutes only a small fraction of CT, which emphasizes broad knowledge acquisition. In contrast, agent data forms a much larger proportion of SFT, which focuses on high-quality, task-specific agent trajectories.

**Iterative Data Flow.** 
After the initial RL model is trained, it becomes the main data generator for the next iteration. In each iteration \( t \), it produces new trajectories via rejection sampling (*RFT*) or interactive annotation (). Each sample is evaluated by a validation function \( V(s)  \). High-quality samples with \( V(s) = 1 \) are added to the SFT dataset for the next iteration as \( D_{SFT}^{(t+1)} = D_{SFT}^{(t)} D_{}^{(t)} \), while lower-quality samples with \( V(s) = 0 \) are routed to the CT dataset as \( D_{CT}^{(t+1)} = D_{CT}^{(t)} D_{}^{(t)} \). This ensures that SFT always receives the most recent, verified high-quality data, while CT continually expands with broader, less polished knowledge without contaminating the supervised signal. Note SFT and RL are performed more frequently than CT.
It should also be noted that in each cycle, we observe substantial transfer from general-purpose RL to agent-specific domains.
As iterations progress, the improved model \( M^{(t+1)} \) generates a higher proportion of high-quality outputs, i.e., \( P(V(s) = 1 t) > P(V(s) = 1 t-1) \), accelerating capability growth. Since every generated sample is reused at an appropriate stage, no data is wasted, creating a sustainable cycle in which model and data quality co-evolve to drive continual performance gains.

### CT & SFT Data Preparation

Agent-related training data represents a significant scarcity in existing human corpora, particularly for multi-turn interactive tasks that require sustained reasoning and tool manipulation. Unlike abundant mathematical or coding data in human corpora, agent interaction trajectories are rare and difficult to obtain at scale. To address this critical bottleneck, we develop a systematic data construction pipeline that operates on both interactive human annotations and automated data synthesis. 

**Figure:** *
 We curate a Data Flywheel for UI-TARS-2, establishing a self-reinforcing loop that continuously improves both data quality and model capabilities.
 * () _(image: figures/dataflywheel_v2.pdf)_

#### In-Situ Annotation for Continual Pre-training

Our continual pre-training framework spans multiple agent domains. Here we illustrate the methodology using the GUI domain as a representative case. As the cold-start GUI CT dataset \( D_{CT, GUI}^{(0)} \), we include all training data from UI-TARS [qin2025ui] and UI-TARS-1.5 [seed2025uitars15], consisting of GUI tutorials collected from the internet, open-source agent trajectories, our in-house annotations, etc. Despite this diverse initialization, we quickly encountered several limitations. First, publicly available data is inherently scarce and easily exhausted, leaving insufficient coverage for training at scale. In particular, we observed a notable lack of content for Chinese-language applications, which hinders the development of truly versatile agents. Second, much of the available data provides only procedural actions while omitting the underlying cognitive reasoning. Models trained solely on such resources tend to mimic surface-level actions without internalizing the logic, leading to spurious or unstable reasoning chains. Ultimately, the central challenge of continual pre-training lies in how to systematically scale up high-quality, cognitively rich data to sustain long-term agent improvement.

To address the deficiencies of existing GUI datasets, we developed a large-scale, human-centric annotation system designed for collecting authentic cognitive processes. A key feature of our platform is its **in-situ deployment**: the annotation tool is directly installed on annotators' personal computers and runs unobtrusively alongside their normal usage. This design allows data to be collected continuously in realistic, everyday settings, without disrupting natural workflows. 

**Annotation Protocol.** An initial pilot study that attempted to retroactively add reasoning traces to recorded actions proved ineffective, as it was nearly impossible to reconstruct the annotator’s original thought process. Inspired by deitke2024molmo [deitke2024molmo], we instead adopted a *think-aloud* protocol, where annotators verbalize their thoughts via audio while completing tasks. These verbalized thoughts are automatically aligned with corresponding UI interactions, producing data that captures both the reasoning chain and the grounded actions. To further enrich coverage, we recruited two groups of annotators: (1) **experts**, who provide demonstrations of complex tasks, and (2) **novices**, who are asked to solve unfamiliar tasks through exploration, trial-and-error, and external resources (e.g., web search). The novice track captures valuable data on problem-solving and adaptation when prior knowledge is absent. 

**Task Design and Collection.** To strengthen GUI-agent capabilities in realistic settings, we present a reproducible data acquisition pipeline. Candidate applications are selected using publicly available indicators along three dimensions—industry coverage, user engagement, and market penetration—yielding a representative set of mainstream websites and desktop applications. For each service, a hierarchical task graph is constructed, and task-importance scores are derived using normalized measures of usage frequency, user benefit, and cross-scenario transferability. We adopt a human–LLM collaborative workflow to generate multilevel query sets for each subfunction, spanning novice-to-expert skill levels and both single- and multi-application settings. A difficulty rubric based on step count, cross-page operations, prerequisites, and exception handling ensures balanced coverage across difficulty levels. 

**Curation Pipeline.** All collected data undergoes rigorous quality control, including executability verification, deduplication, and dual-annotator review. The audio-recorded thoughts are first transcribed using automatic speech recognition (ASR) and then refined by LLMs to produce coherent, high-quality reasoning text. These processed reasoning traces are precisely synchronized with on-screen actions, yielding temporally aligned reasoning–action trajectories. To further enhance training utility, we programmatically augment linguistic diversity and enrich reasoning chains, resulting in a final high-fidelity dataset suitable for continual pre-training.

**Figure:** *
 The four-layer architecture of the interactive annotation platform.
 * () _(image: figures/interactive_annotation_platform.pdf)_

#### Interactive Annotation for Supervised Fine-tuning

A key challenge in training agents from human-generated SFT data is that such data is typically *off-policy*: it does not reflect the actual distribution of actions that the model would take when interacting with an environment. As a result, models trained on this data may fail to generalize, since they never encounter or correct their own mistakes during rollout. Prior approaches mitigate this by asking annotators to correct errors in pre-collected trajectories [qin2025ui]. However, this procedure remains fundamentally offline and inefficient: it exposes model weaknesses only after task failure, without enabling real-time intervention or correction during interaction. Because agent training occurs within interactive environments, where actions directly affect subsequent states, this lack of on-policy supervision creates a significant gap. To bridge it, we propose a novel human-in-the-loop framework for online, interactive data annotation.

**System Design.** 
Our interactive annotation platform is built on a four-layer architecture. At the top, the interaction layer presents the user interface, enabling annotators to engage with the system in real time. Beneath it, the service layer processes annotation requests, orchestrating model-generated command execution and human interventions. The platform layer provides scenario-specific execution environments—such as Computer Use, Phone Use, or Tool Use—tailored to different categories of tasks. Finally, the storage layer securely logs annotation data and complete interaction trajectories for downstream training and analysis. The overall design is illustrated in Figure §fig:interactive_annotation, which depicts the modular separation between layers and their control flow. In the following, we take GUI and Game as examples to illustrate the annotation process.

Our interactive annotation platform enables human annotators to provide online supervision directly within the agent's rollout. Annotators are assigned tasks to complete in a controlled virtual environment (see Figure §fig:annotation_flow), backed by a cloud-hosted VM or browser sandbox to ensure reproducibility and consistent execution. At each decision point, the latest UI-TARS-2 model proposes candidate actions together with its reasoning trace. The annotator can either accept one of these suggestions or override it with a better thought and action, allowing human expertise to guide the trajectory in real time. We further streamline the workflow with features such as command auto-completion, real-time VM video streaming, and on-screen coordinate visualization, reducing latency and improving annotation accuracy.

Because annotation occurs in a live environment, annotators receive immediate feedback from the system and can track the evolving trajectory, avoiding the inefficiencies of post-hoc correction. This design ensures that all supervision remains strictly *on-policy*: the data reflects the actual distribution of states visited by the current model. To further enhance efficiency, both the annotation model and the pool of tasks are periodically refreshed, ensuring that data collection consistently targets the weaknesses of the most recent agent. 

**Figure:** *
 The interactive annotation workflow.
 * () _(image: figures/anno-uitars.pdf)_

### Multi-turn Reinforcement Learning
 
To train agents capable of long-horizon reasoning and interactive decision-making, we adopt a multi-turn RL framework built on RLVR (Reinforcement Learning with Verifiable Rewards) [guo2025deepseek]. 
We construct domain-specific pipelines that automatically synthesize large-scale, verifiable tasks across multiple domains. 
During RL, our model engages in real-time multi-turn interactions with the environment, continuously observing state transitions and environmental feedback until task completion. 
The model then leverages verifiable rewards to optimize its decision-making trajectories through iterative policy improvement. 
While our RL framework is applied across multiple domains with different tools defined by GUI operations and GUI-SDK functions, in the following, **we choose three representative cases to describe our framework**: (1) GUI-Browsing, which targets GUI-based information-seeking tasks, (2) GUI-General, which covers broader web manipulation tasks, and (3) gameplay, which focuses on lightweight web-based mini-games executed in a browser sandbox.

#### Task Design
 

High-quality, sufficiently challenging, and verifiable task data for end-to-end RL remains extremely scarce. In the following, we introduce how to design training tasks that are both diverse in the form and equipped with reliable verification signals.

**GUI-Browsing.** 
To enable autonomous exploration in complex reasoning scenarios, we design an automated pipeline for synthesizing large-scale, verifiable GUI-browsing tasks. These tasks are conceptually similar to deep research tasks [openai-deepresearch], except that agents must satisfy the information-seeking requirements solely through analyzing screenshots, without access to search APIs.
Our synthesis framework includes two main approaches: 

(1) Multi-Condition Obfuscation:
We begin by extracting core entities and their attribute features from authoritative knowledge sources (e.g., Wikipedia). 
Each feature is scored for distinctiveness using an LLM. Highly revealing attributes are removed, while the remaining ones are rewritten by the LLM to increase abstraction and reduce specificity. 
This process produces complex questions defined by multiple indirect constraints, requiring the model to combine and reason over blurred signals in order to identify the correct answer. 
For example, from a Wikipedia page we generate the following obfuscated question: 
"Discovered by a representative from the Music And Cabaret talent agency, this group had a founding lineup—initially under another name—that included members from Dreghorn and Irvine, plus a lead guitarist and drummer. The lead vocalist joined after being recommended by a founding member who saw them perform with a Kilmaurs-based band, and their lead guitarist left to form another ensemble before late 1975. Which record label did this group sign with?" 

(2) Multi-Hop Chain-Like Conditions:
We begin from an entity’s webpage and follow its hyperlinks to identify structurally related entities. 
For each linked entity, we extract and obfuscate descriptive features, creating tasks where the linked entity becomes the answer. 
We then treat the linked entity's page as the new starting point and repeat this process recursively, generating tasks for progressively deeper levels. 
At each step, the answer from the previous hop is embedded within the new question, forming a coherent reasoning chain. 
Finally, the atomic steps are semantically integrated into a single multi-hop question that requires the model to synthesize intermediate answers, mirroring the layered nature of knowledge propagation online and substantially increasing the demand for deep, sequential reasoning. 
To ensure difficulty, we filter the synthesized data by discarding instances that can be trivially solved using prior knowledge or a single-turn search, keeping only truly challenging and verifiable tasks for training.

**GUI-General.** 
To evaluate general-purpose interaction capabilities, we construct a dataset of GUI-General tasks using an offline synthesis pipeline centered on general websites. 
We begin by curating candidate websites from public collections, filtering out inaccessible pages, login-gated services, and trivial categories such as static information pages or casual games. 
For each selected website, VLMs are employed to identify and extract its core functionalities. 
Based on these, we synthesize tasks at the single-page level through a structured process: removing overly simple functions, composing executable instructions, merging prerequisite sub-tasks, and refining task descriptions for clarity, objectivity, and verifiability. 
The resulting dataset provides a diverse pool of executable, GUI-interaction-focused tasks that serve as queries for RL training, covering 690 websites across a wide range of domains.

**Gameplay.** 
For the game domain, we construct the RL dataset through two complementary sources. First, we collect publicly available HTML5/WebGL mini-games that can run directly in the browser sandbox. Second, to further expand coverage, we synthesize new games using LLMs, which generate lightweight code implementations that preserve core gameplay mechanics while exposing explicit state interfaces. For both real and synthesized titles, we create concise JavaScript verification scripts that query runtime variables (e.g., score, level index, remaining lives) and provide time-aligned state attributes. These observations establish a reliable mapping from agent actions to environment transitions and reward signals. Finally, all interaction records are consolidated into a unified JSON schema containing scalar rewards, termination flags, and metadata (e.g., game version and verification checksums).

#### Reward Design

A reliable reward system is essential for stable policy optimization, requiring feedback signals that are both consistent and trustworthy across heterogeneous environments. 
We categorize our reward design based on whether the correctness of an agent's output can be deterministically verified:

**Deterministically verifiable tasks.** 
In domains where automatic function-based verifiers are available (e.g., games), 
we directly compute binary correctness signals as rewards. 
For GUI-Browsing tasks, where answers can be matched against reference ground truth, we instead employ *LLM-as-Judge* [gu2024survey] to evaluate the agent’s prediction against the target answer. 

**Non-verifiable tasks.** 
In more open-ended settings, such as GUI-General tasks, neither formal verifiers nor reference answers exist. 
To address this, we employ UI-TARS-2 as a generative outcome reward model (ORM) that produces scalar rewards conditioned on the agent’s trajectory. 
The ORM takes as input the full text history together with the last five screenshots (to fit within the context window) and outputs a score indicating task success. 
To achieve this, we specifically enhance UI-TARS-2's capability of ORM through targeted data annotation and single-turn RL, ensuring that its reward predictions are accurate, consistent, and robust for downstream multi-turn RL.

**Figure:** *
 The multi-turn RL training infrastructure of UI-TARS-2.
 * () _(image: figures/multi-turn-RL.pdf)_

#### Asynchronous Agent Rollout via Stateful Environment

Traditional batch-based rollout approaches often become bottlenecked by complex long-tail problems, reducing training efficiency and creating off-policy distribution drift. Our multi-turn RL training infrastructure (Figure §fig:multiturnrl) is developed for two core objectives: (1) enhancing training stability and (2) optimizing efficiency in multi-turn rollout interactions and training sample organization. UI-TARS-2 implements several key features:

**Asynchronous Inference with Server-Based Rollout.** 
We adopt a fully asynchronous inference system utilizing online server-mode processing. By encapsulating policy inference within asynchronous server architecture, we decouple agent reasoning framework implementation from policy inference execution. This design significantly enhances framework usability, which supports easily-developed new agent interaction handlers, while improving model inference efficiency through asynchronous inference. 

**Streaming Training with Partially-Filled Rollout Pools.** 
Traditional batch-mode rollout requires complete batch inference before training initiation, potentially creating bottlenecks with long-tail cases that delay subsequent training cycles. Our system maintains a dynamic rollout pool where training updates commence once completed traces reach the minimum batch size threshold. Incomplete rollout traces remain in the pool for subsequent training iterations, ensuring continuous learning progress. This feature is conceptually similar to Kimi-Researcher [kimi-researcher].

**Stateful Agent Environment Integration.** 
We implement stateful agent environments that preserve execution states across multiple tool invocations, enabling continuous state transitions and maintaining context throughout extended problem-solving sessions. This approach supports complex, multi-step reasoning processes that require persistent environmental memory.

#### RL Training Algorithm

UI-TARS-2 is trained using Proximal Policy Optimization (PPO), where the policy is updated according to the following objective function:

$$

_() = _{(q,a),o_{t}_{_{}}(q)}
[ 
( (o_tq,o_{<t})}{_{_{}}(o_tq,o_{<t})} _t, \\ 
\  ( (o_tq,o_{<t})}{_{_{}}(o_tq,o_{<t})}, 1 - _{low}, 1 + _{high} ) _t ) ],


$$

where $_{}$ is policy model, $_{_{}}$ is previous policy model.

 
Following VAPO [yue2025vapoefficientreliablereinforcement] and VC-PPO [yuan2025whatspposcollapselongcot], UI-TARS-2 integrates several critical enhancements to broaden the exploration space and improve stability, especially in long-horizon settings:

**Reward Shaping.** 
To promote more strategic agent behaviors, the reward signal is mainly determined based on the correctness of the final outcome. In certain scenarios, we employ format rewards and length penalties to discourage premature termination or infinite continuation.

**Decoupled GAE.** 
To address the challenge of value estimation bias over long sequences, we employ the Decoupled Generalized Advantage Estimation (Decoupled-GAE) [yuan2025whatspposcollapselongcot], allowing the computation of advantage for the policy and value function to use different coefficients. Specifically, we set $_{policy}$ and $_{critic}$ to be different. This approach prevents decay in the critic's value estimates when dealing with lengthy token sequences, thereby promoting stability during long-horizon training.

**Length-Adaptive GAE.** 
To mitigate the issue of inconsistent advantage estimation for sequences of varying lengths, we employ Length-Adaptive Generalized Advantage Estimation (Length-Adaptive GAE) [yue2025vapoefficientreliablereinforcement] technique, adjusting the GAE parameter $_{policy}$ based on the sequence length. Specifically, we set $=0.05$ in length-adaptive formula $_{policy}=1-{l}$ to control the overall bias-variance trade-off.

**Value Pretraining.** 
To mitigate the value initialization bias, we adopt the Value-Pretraining [yue2025vapoefficientreliablereinforcement], which involves offline training of the value model to convergence under a fixed policy. Specifically, responses are sampled continuously from a fixed policy (e.g., $_{sft}$), and the value model is updated using GAE with $= 1.0$ (equivalent to Monte Carlo return), providing stable and reliable optimization. Training continues until crucial metrics such as value loss and explained variance reach sufficiently low levels, indicating effective convergence. The resulting value model checkpoint is then used as the initialization for subsequent experiments, ensuring more accurate and calibrated value estimation from the outset.

**Clip Higher.** 
To further promote exploration, we decouple the PPO clipping parameters as recommended by DAPO [yu2025dapoopensourcellmreinforcement], introducing distinct lower ($_{low}$) and upper ($_{high}$) clipping bounds. Increasing $_{high}$ affords greater flexibility for raising the likelihood of low-probability actions, thus enlarging the exploration space. Conversely, $_{low}$ is maintained at a low value to avoid prematurely eliminating tokens, which would risk collapsing the diversity of potential outputs.

### Merging Vertical Agents via Parameter Interpolation

A central goal of UI-TARS-2 is to develop a unified digital agent that not only handles structured desktop and web interfaces but also extends to dynamic environments. 
A natural approach would be to conduct joint RL across all environments and tasks. However, this is challenging in practice: domains differ substantially in action/state spaces, task horizons, and rollout complexity, making large-scale joint optimization unstable and computationally prohibitive. 
Instead, we adopt a simpler but effective strategy that leverages the observation that models fine-tuned from the same pre-trained checkpoint remain approximately linearly mode-connected in parameter space [qin-etal-2022-exploring]. 
This property enables us to train specialized agents independently for different domains and then merge them through parameter interpolation, thereby consolidating their strengths without the cost of multi-domain joint training. 

Concretely, starting from a shared SFT initialization, we conduct multiple RL runs tailored to different environments—for example, **GUI-Browsing** tasks focused on information seeking, **GUI-General** tasks covering broader web manipulation, and **Game** environments based on interactive mini-games—alongside additional variants trained on other domains and corresponding tools (e.g., GUI-SDK). 
We then merge these trained models by interpolating their parameters:

$$
^{()} = _{k Browsing}, General}, , SDK}, ...}} _k ^{(k)}, 
 _{k} _k = 1, \ _k 0,
$$

where $^{(k)}$ denotes the parameters of each domain-specialized model.
Empirically, this interpolation strategy preserves the performance of each specialized vertical while enabling strong cross-domain generalization. On composite tasks requiring skills from multiple domains, the merged model performs almost comparably to the best specialized model in each relevant domain, without additional optimization cost.
% — END sections/methodology —

% — BEGIN sections/experiment_reorg —

## Experiments

This chapter presents a comprehensive experimental analysis of UI-TARS-2. 
Although the training spans multiple domains and tool integrations, we focus our discussion on two representative settings: GUI-based interaction and game environments. 
These two cases highlight complementary challenges: structured interface operation on the one hand, and dynamic long-horizon control on the other. 

### Experimental Setup

UI-TARS-2 is initialized from the pre-trained checkpoint of Seed-thinking-1.6 [thinking1.6], and leverages all of its post-training data. 
The architecture includes a 532M-parameter vision encoder and a 
Mixture-of-Experts (MoE) LLM with 23B active parameters (230B total). 
Building on this base, we conduct multiple iterative training cycles consisting of SFT, RL, and RFT, progressively refining the model’s capabilities. 

We conduct evaluations across a diverse set of benchmarks that comprehensively assess agent capabilities:

**GUI Benchmarks.** We evaluate our model across a diverse suite of benchmarks spanning three categories: computer use, mobile use, and browser use. 
For **computer use**, OSWorld [xie2024osworld] provides 369 tasks across Ubuntu, Windows, and macOS with detailed configurations and evaluation scripts, while WindowsAgentArena [bonatti2024windows] adapts this framework to over 150 Windows-specific tasks. 
To assess deeper system-level capabilities, we also include TerminalBench [tbench_2025], which measures proficiency in command-line environments, and SWE-Bench [jimenez2023swebench], which evaluates repository-level software engineering tasks. 
For **mobile use**, AndroidWorld [rawles2024androidworld] offers 116 tasks across 20 mobile applications within a live Android emulator, with dynamic task variations generated via randomized parameters. 
For **browser use**, Online-Mind2Web [xue2025illusionprogressassessingcurrent] contains 300 realistic tasks across 136 websites, while BrowseComp-en [wei2025browsecomp] and BrowseComp-zh [zhou2025browsecomp] provide high-difficulty multi-hop questions. 
For the above benchmarks, UI-TARS-2 is allowed to use either GUI operations or GUI SDK.

**Game Benchmarks.** 
We develop a **15 Games Collection** from our game pool, which is used to measure in-domain performance. 
We also leverage an OOD benchmark: **LMGame-Bench** [hu2025lmgamebenchgoodllmsplaying], which evaluates LLM agents' game-playing abilities across six classic titles through a unified Gym-style interface, with optional perception and memory scaffolds designed to stabilize vision and long-horizon control. 
It reports performance under both harnessed and unharnessed settings. 
For all these collections, evaluations are conducted within a browser-sandboxed, screenshot-only setting. 
UI-TARS-2 interacts with games through a human-like action space (mouse clicks, key presses, and scrolling), mirroring how players operate in real environments. 
Results are reported as raw per-game scores as well as the mean normalized score across titles. 

**Compared Baselines.** 
For GUI benchmarks, we compare UI-TARS-2 against state-of-the-art proprietary models, including Claude 4 [anthropic2025claude], OpenAI-o3 [openai2025o3], and OpenAI CUA-o3 [openai_2025_cua_blog], as well as previous UI-TARS variants. 
For game benchmarks, we evaluate Claude (Computer Use) [anthropic_2024_developing_computer_use], OpenAI CUA-o3, OpenAI-o3, Gemini-2.5 Pro [comanici2025gemini], and Claude 3.7/4 [anthropic2025claude].

### Main Results

**GUI Main Results.** 
As shown in Table §tab:gui_main, UI-TARS-2 establishes superior performance across a wide range of GUI-agent benchmarks. 
Compared to previous versions of UI-TARS and other strong baselines such as OpenAI CUA-o3 and Claude 4, our model demonstrates consistent improvements across computer use, mobile use, and browser use settings. 
In particular, UI-TARS-2 surpasses UI-TARS-1.5 on all reported benchmarks, achieving 47.5% on OSWorld, 50.6% on WindowsAgentArena, 73.3% on AndroidWorld, and 88.2% on Online-Mind2Web, highlighting the benefits of iterative training and reinforcement learning. 
**Benefits from GUI-SDK**: 
With the integration of extended SDK functions, UI-TARS-2 is further equipped to handle system-level tasks beyond surface-level GUI interaction. 
In this setting, the model achieves 45.3% accuracy on Terminal Bench, 68.7% on SWE-Bench, 50.5% on BrowseComp-zh, and 29.6% on BrowseComp-en. 
For comparison, when restricted to GUI-only operation, the scores on BrowseComp-zh and BrowseComp-en are 32.1% and 7.0%, respectively. 
This clear performance gap demonstrates that GUI-SDK augmentation enables the model to perform more complex reasoning and tool-use behaviors, equipping UI-TARS-2 with the broader skills expected of general computer-use agents. 
**OOD Generalization**: 
Most of the tasks of GUI-Browsing and GUI-General are browser-focused tasks, after RL training, the resulting model exhibits strong OOD generalization. 
On Online-Mind2Web, RL improves accuracy from 83.7% (the SFT baseline in the final iteration) to 88.2%. 
More strikingly, the RL-trained model transfers effectively to domains that were not the primary focus of training: for example, OSWorld improves by nearly 10.5% (from 43.0% to 47.5%) and AndroidWorld by over 8.7% (from 64.6% to 73.3%). 
These results highlight the ability of task-specific RL to induce broadly transferable skills, enabling GUI agents to perform reliably in previously unseen environments. 

**Table:** *Performance on computer use, mobile use, and browser use benchmarks. 
"-" indicates unavailable; 


% — BEGIN sections/contribution —

## Contributions

The authors are listed alphabetically by first name, with some names corresponding to internal aliases used within the company.
{0pt} 
{0pt} 
{0pt} 
{2}

### Algorithm

#### Core Contributors

Haoming Wang \\
Haoyang Zou \\
Huatong Song \\
Jiazhan Feng \\
Junjie Fang \\
Junting Lu \\
Longxiang Liu \\
Qinyu Luo \\
Shihao Liang \\
Shijue Huang \\
Wanjun Zhong \\
Yining Ye \\
Yujia Qin \\
Yuwen Xiong \\
Yuxin Song \\
Zhiyong Wu

#### Contributors

Aoyan Li \\
Bo Li \\
Chen Dun \\
Chong Liu \\
Daoguang Zan \\
Fuxing Leng \\
Hanbin Wang \\
Hao Yu \\
Haobin Chen \\
Hongyi Guo \\
Jing Su \\
Jingjia Huang \\
Kai Shen \\
Kaiyu Shi \\
Lin Yan \\
Peiyao Zhao \\
Pengfei Liu \\
Qinghao Ye \\
Renjie Zheng \\
Shulin Xin \\
Wayne Xin Zhao \\
Wen Heng \\
Wenhao Huang \\
Wenqian Wang \\
Xiaobo Qin \\
Yi Lin \\
Youbin Wu \\
Zehui Chen \\
Zihao Wang

### Infra

#### Core Contributors

Baoquan Zhong \\
Xinchun Zhang \\
Xujing Li \\
Yuanfan Li \\
Zhongkai Zhao

#### Contributors

Chengquan Jiang \\
Faming Wu \\
Haotian Zhou \\
Jinlin Pang \\
Li Han \\
Qi Liu \\
Qianli Ma \\
Siyao Liu \\
Songhua Cai \\
Wenqi Fu \\
Xin Liu \\
Yaohui Wang \\
Zhi Zhang

### Data

#### Core Contributors

Bo Zhou \\
Guoliang Li \\
Jiajun Shi \\
Jiale Yang \\
Jie Tang \\
Li Li \\
Qihua Han \\
Taoran Lu \\
Woyu Lin \\
Xiaokang Tong \\
Xinyao Li \\
Yichi Zhang \\
Yu Miao \\
Zhengxuan Jiang \\
Zili Li \\
Ziyuan Zhao

#### Contributors

Chenxin Li \\
Dehua Ma \\
Feng Lin \\
Ge Zhang \\
Haihua Yang \\
Hangyu Guo \\
Hongda Zhu \\
Jiaheng Liu \\
Junda Du \\
Kai Cai \\
Kuanye Li \\
Lichen Yuan \\
Meilan Han \\
Minchao Wang \\
Shuyue Guo \\
Tianhao Cheng \\
Xiaobo Ma \\
Xiaojun Xiao \\
Xiaolong Huang \\
Xinjie Chen \\
Yidi Du \\
Yilin Chen \\
Yiwen Wang \\
Zhaojian Li \\
Zhenzhu Yang \\
Zhiyuan Zeng

### Application

Chaolin Jin \\
Chen Li \\
Hao Chen \\
Haoli Chen \\
Jian Chen \\
Qinghao Zhao

### Supervisor

Guang Shi



% — END sections/contribution —