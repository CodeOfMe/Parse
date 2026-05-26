# GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models

> **arXiv:** [2601.18197](https://arxiv.org/abs/2601.18197)
> **TeX source:** [arXiv-2601.18197v1/](arXiv-2601.18197v1/)
> **PDF:** [GAIA-arXiv-2601.18197v1.pdf](GAIA-arXiv-2601.18197v1.pdf)

---

## Abstract

While Large Vision-Language Models (LVLMs) have significantly advanced GUI agents' capabilities in parsing textual instructions, interpreting screen content, and executing tasks, a critical challenge persists: the irreversibility of agent operations—where a single erroneous action can trigger catastrophic deviations. 

To address this, we propose the **G**UI **A**ction Cr**i**tic's Dat**a** Flywheel System (GAIA), a training framework that enables the models to have iterative critic capabilities, which are used to improve the Test-Time Scaling (TTS) of basic GUI agents' performance.

Specifically, we train an **Intuitive Critic Model** (ICM) using positive and negative action examples from a base agent first. This critic evaluates the immediate correctness of the agent's intended actions, thereby selecting operations with higher success probability.

Then, the initial critic guides agent actions to collect refined positive/negative samples, initiating the self-improving cycle. The augmented data then trains a second-round critic with enhanced discernment capability.

We conduct experiments on various datasets and demonstrate that the proposed ICM can improve the test-time performance of various closed-source and open-source models, and the performance can be gradually improved as the data is recycled. The code and dataset will be publicly released.

## Introduction

The automation of Graphical User Interface (GUI) interactions represents a critical frontier in developing intelligent digital assistants [wang2024gui,hu2024agents,nguyen2024gui]. Recent breakthroughs in Large Vision-Language Models (LVLMs) [wang2024qwen2,bai2025qwen2], leveraging advanced post-training techniques, have substantially enhanced agents' capabilities in interpreting natural language commands, perceiving visual elements, and executing multi-step tasks [hong2024cogagent,cheng2024seeclick].

Within this rapidly evolving landscape, the development of robust GUI agents has largely converged on two primary methodological paradigms. The first approaches [wu2024atlas,xu2024aguvis,qin2025ui,liu2025infiguiagent] train models through Supervised Fine-Tuning (SFT) to directly align their behavior with predefined task objectives. The second approaches employ Reinforcement Fine-Tuning (RFT) [lu2025ui,xia2025gui,liu2025infigui], which significantly enhances generalization in complex tasks by adopting a reasoning format.

Despite these advances, the dynamic and continuous nature of real-world GUI tasks means that agents can still produce ambiguous or incorrect action proposals at any step. **A single mis-click or mis-typed output can be irreversible**, derailing the entire workflow and leaving the system in an unrecoverable state. This high-stakes environment imperatively demands **a mechanism for pre-execution validation.**

To avoid irreversible errors in execution and improve the performance of basic GUI agents during testing, previous studies have designed action verifiers for GUI agents [wu2025gui,xiao2025ui,yang2025gta1], which are used to judge and filter GUI agents' actions. However, these existing implementations suffer from two primary limitations.

First, training a correctness verifier requires defining positive and negative action samples. Existing work on defining negative samples relies on heuristic algorithms, such as randomly selecting click locations on the current screenshot [xiao2025ui], which fails to capture the realistic action distribution and leads to suboptimal judgment performance.

Second, the reasoning-based verifiers [wanyan2025look] implemented in existing work violate the intuitive properties of binary judgments. For an intuitive correctness judgment problem, biological research suggests that higher-level judgment pathways are often more adept than performing extensive multi-step reasoning [liu2011neural,poldrack2005neural,doyon2005reorganization], which indicates that excessive reasoning can be less effective [bilalic2008good,wan2011neural]. Furthermore, reasoning-based judgment outputs more tokens, thereby reducing the efficiency of test-time scaling.

**Figure:** *The promotion process of the critic model to the GUI Agent during testing.* () _(image: Figs/fig1_intuitive.pdf)_

To fully leverage pre-execution evaluation to enhance GUI agent capabilities and execution correctness, we developed a **G**UI **A**ction Cr**i**tic's Dat**a** Flywheel System (GAIA). This system comprises two core phases: the initialization phase (Phase 1) and the iteration phase (Phase 2), yielding the **Intuitive Critic Model** (ICM).

In **Phase 1**, we use real GUI agents to act on an existing dataset to collect positive and negative action data that are random but consistent with the behavior distribution. Using this binary-labeled action dataset, we train ICM to assess action correctness given environmental context.

In **Phase 2**, as illustrated in Figure §fig_intuitive(a), ICM employs a Best-of-N approach to select the highest-probability correct actions from agent rollouts. While ICM guidance significantly improves action accuracy, challenging samples persist and produce errors. These difficult cases are annotated and fed back into the data flywheel. Through iterative data augmentation, the flywheel continuously incorporates new action samples, progressively covering challenging scenarios within the action space. Driven by this enriched dataset, we train an enhanced critic—Intuitive Critic Model on Round Two (ICM-r2)—which achieves higher discriminative accuracy for more precise behavioral guidance. This establishes a self-evolutionary virtuous cycle between the data flywheel and critic models, continuously improving GUI agent action accuracy.

Leveraging the proposed system GAIA, ICM achieves SOTA performance in action critique. Naturally, we integrate it into Test-Time Scaling (TTS) [snell2025scaling,chen2024expanding,snell2024scaling,prabhudesai2023diffusion,wang2025mcts,tian2025think] during inference, where ICM evaluates stochastically generated actions from the TTS process, releasing only high-confidence operations surpassing predetermined thresholds for execution. To validate the framework's general applicability, we conduct joint experiments using mainstream GUI Agents (including GPT-4o [hurst2024gpt] and UI-TARS [qin2025ui]) on several GUI Agent benchmarks.

As shown by the comparative results in Figure §fig_intuitive (b), the guidance from our iteratively evolved critic models (ICM and ICM-r2) leads to significant performance improvements in basic GUI agents, including GUI operation task planning and grounding capabilities.

Overall, the main contributions are summarized as follows:

1. We introduce GAIA—a novel Data Flywheel System designed for training GUI action-critic models. By iteratively curating positive and negative samples from real-world action data, GAIA continuously boosts model performance and robustness.
2. We propose the ICM for GUI interaction tasks, a critic model trained on data curated by our data flywheel. The ICM enhances the performance of existing GUI agents by employing a best-of-N approach to select the most probable correct action with TTS. This initial boost is then continuously refined as the ICM's discriminatory accuracy is iteratively improved by the data flywheel.

## Related Work

### GUI Agent

The development of autonomous agents powered by LLMs and LVLMs has significantly advanced interactive functionalities within digital environments. 

Early GUI systems primarily leveraged LLMs to interpret structured representations [hong2024cogagent,nong2024mobileflow,song2024visiontasker].

The development of LVLM simplifies the paradigm, allowing GUI agents to receive raw visual signals from the screenshots [hu2024agents,liu2024autoglm,shen2024falcon,tang2025think,christianos2024lightweight,zheng2025vem,gou2024navigating,wu2024atlas]. 

Recent efforts, such as Aguvis [xu2024aguvis] and UI-TARS [qin2025ui], have advanced autonomous GUI navigation by integrating explicit planning, sophisticated reasoning, and GUI-specific pretraining to handle complex digital environments.

Concurrently, the advent of rule-based Reinforcement Learning (RL) approaches [jaech2024openai,guo2025deepseek] has further enhanced GUI agent capabilities. These RFT methods improve reasoning and generalization by enabling models to learn universal action strategies from high-quality samples [liu2025visual,shen2025vlm,lu2025ui,xia2025gui,liu2025infigui]. 

While fine-tuning and model scaling can enhance GUI agent capabilities, these methods are often prohibitively resource-intensive. This highlights a clear need for test-time enhancements that can offer universal performance improvements across various agent models without costly retraining.

**Figure:** ***Data flywheel curation pipeline for GAIA.** A sample dataset is constructed using GUI agent interactions. The positive and negative labels are marked by comparing the ground truth actions to train an action correctness discrimination model. After the critic model guides the GUI agent, it further expands the dataset, pushing the data flywheel to cover more action distributions, thereby promoting the iterative improvement of model performance.* () _(image: Figs/fig2.pdf)_

### Critic Model

To solve the problem of suboptimal single-shot model output [zhang2025llm,martino2023knowledge,wen2024reinforcing,chen2024optima], research has gradually focused on improving the performance of the basic model during testing with the help of the critic model [mcaleese2024llm,ji2023towards,kalyanpur2024llm,zhang2025critic,xiong2025llava]. This concept has been expanded to the GUI domain with notable works like GUI-Genie [xiao2025ui], GUI-Actor [wu2025gui], GTA1 [yang2025gta1], and GUI-Critic-R1 [wanyan2025look].

However, existing GUI critics often rely on synthetic data generated by heuristic algorithms, such as randomly selecting click locations [wu2025gui], cross-task substitution, or early truncation [xiao2025ui]. This approach fails to accurately simulate the complex behavior of real GUI agents across the full action space, thereby preventing the critic from learning faithful discrimination criteria. 

Furthermore, while some approaches use RL to inject reasoning capabilities into the critic [wanyan2025look], this often contradicts the very motivation for intuitive judgment [liu2011neural,wan2011neural] and introduces delays due to extended output token generation.

## Method

In this section, we detail the design of our data flywheel-driven GAIA system for the GUI agent shown in Figure §fig2. We begin in Section 3.1 by introducing the general definition of the GUI agent task and the crucial role of the critic model. Section 3.2 delves into the design and application of our data flywheel system within the initial round of the evaluation process. In Section 3.3, we present the model training in the second round, which builds upon the outcomes from the first iteration and forms a virtuous cycle.

### Preliminaries

The interaction between a GUI agent and its environment can be formulated as a Markov Decision Process (MDP), denoted by the tuple $, , , ,  $. Here, $$ defines the state space of possible screen states, while $$ encompasses the action space, including interaction types like clicking, typing, and scrolling. The observation space $$ captures inputs such as screenshots or structured UI representations. The state transition probability is given by $:    [0,1]$, mapping a state and action to a new state. Similarly, $:   $ describes the likelihood of observing a particular output given a state and an action. During GUI task execution, at each discrete time step $t$, the agent receives an input tuple $(z_t, u, h)$, comprising the current screen observation $z_t $, the global task instruction $u$, and the accumulated interaction history $h$. The agent's decision-making process for GUI actions is then formalized by a structured policy function $$:

$$
 (z_t,u,h) o_t = {a_t, c_t},
$$

where $o_t$ represents the agent output at time $t$, consisting of the action type $a_t$ (e.g., click, scroll, and type) and its corresponding parameters $c_t$ (e.g., click coordinates, text content for typing). After $a_t$ is executed, the environment transitions to a new state $z_{t+1}$, and this iterative process continues until the task is successfully completed or a predefined termination condition is met.

**Figure:** ***Test-Time scaling pipeline.** Through best-of-N rollout, multi-candidate actions of GUI agents are given, and the correct action with the highest probability is selected after ICM evaluation.* () _(image: Figs/fig3.pdf)_

The proposed ICM, building upon the same observations and the GUI agent's current proposed action $o_t$, outputs a judgment $j_t$ regarding the correctness of that action:

$$
 (o_t | (z_t,u,h)) l_t = {j_t, p_t},
$$

where $j_t$ is a binary indicator, "*correct*" for correct actions and "*wrong*" for incorrect, $p_t$ represents the probability of the judgment, which supports finding the correct action with the highest confidence. By enabling the sampling of multiple candidate actions and prioritizing them based on their respective correctness probabilities, ICM ensures that a more optimal action for the current state is selected and executed, significantly enhancing the agent's actual success rate.

### Action Decision with Intuitive Critic

#### Data Curation

To enable the judgment model to distinguish the correctness of real actions, we meticulously define both positive and negative samples of GUI agent actions. We begin by having existing GUI agents $= {_1, _2, ..., _i}$ [qin2025ui] interact with and traverse publicly accessible datasets [li2024effects,lu2024gui], allowing us to collect authentic, step-level operations across various GUI scenarios. For each action executed in a specific state $(z,u,h)$, we then leverage ground truth labels to determine its correctness. An action is designated positive (with a correctness judge $j="*correct*"$) if it aligns with the GT.

Conversely, we identify negative samples (with a correctness score $j="*wrong*"$) based on states where the agent's action deviates from the GT. This approach ensures that our collected negative operations are closely aligned with the actual error distribution observed in real GUI environments, significantly enhancing the quality and realism of our training dataset. To prevent bias during ICM training, we balance the collected positive and negative samples, ensuring an equal 50% split for each. This carefully curated dataset, denoted as $ = {j_k | z_k,u_k,h_k,o_k^{_i}}_{k=1}^K$, forms the foundation of our data flywheel GAIA.

#### ICM Training and Guidance

Based on the dataset $$, ICM is trained to intuitively judge the correctness of actions. Specifically, the input of ICM includes the screen observation $ z_k$, the instruction description $u_k$, the action history $h_k$, and the given agent action $o_k^{_i}$. We implement ICM using LVLM and use standard cross-entropy loss to supervise the ICM's output tokens. For each sample in our dataset D, the model's output is a token representing either "*correct*" or "*wrong*". The training process aims to minimize the discrepancy between the model's predicted probability and the ground truth label:

$$

& _{} = -{K} _{k=1}^{K} [ j_k (P_{_c}(*"correct"* z_k, u_k, h_k, o_k^{_i})) \\
 + & (1 - j_k) (1 - P_{_c}(*"correct"* z_k, u_k, h_k, o_k^{_i})) ],


$$

where $P_{_c}(*"correct"* z_k, u_k, h_k, o_k^{_i})$ represents the probability assigned by the critic model $_c$ to the "*correct*" token.

During test-time, a GUI agent $_i$ generates $N$ candidate actions $ = {o_1, ..., o_N}$ through N-rollout sampling. ICM evaluates these candidates by assigning each action a correctness judge $j_n$ and a confidence score $s_n$. Leveraging the best-of-N filtering strategy, we select the optimal action $o^*$ from the subset of correct candidates $_{}$ that has the highest confidence score, which is formalized as:

$$
 o^* = 
 
 _{o_n _{}} s_n, &  _{} \\
 o_1. & 
 
 
$$

This approach effectively guides the agent to bypass single-shot output failures and select the most promising action, thereby significantly boosting its overall execution accuracy. 

**Table:** ***GUI planning accuracy on AndroidControl and GUI-Odyssey.** $^$ represents the closed-source UI-TARS 1.5 called through the Doubao API. $^*$ represents an agent that reproduces the open-source model. $^+ = {j_k | z_k,u_k,h_k,o_k^{_i},_c}_{k=1}^{K'}$. $^+$ further covers the distribution of actions, providing a foundation for performance scaling.

Based on the challenging samples in this enriched dataset, we train the ICM on Round Two (ICM-r2), using the same cross-entropy loss as defined in Equation §eq_loss. This new dataset, which is specifically curated to expose the critic's most significant blind spots, allows ICM-r2 to acquire a more nuanced and accurate discriminative ability. Consequently, as illustrated in Figure §fig3, ICM-r2 provides more precise guidance for the agent's action selection, thereby fundamentally strengthening the critic's overall judgment and significantly improving the agent's performance on the most difficult tasks. Together with ICM, ICM-r2 demonstrates the power of a data flywheel-driven approach to stimulate the performance of GUI agents during testing.

## Experiment

### Implementation Details

**Experimental Setup.**
We use UI-TARS 1.0 [qin2025ui] and UI-TARS 1.5 [qin2025ui] 
{r}{0.60}
 
 
 $ and $^+$ respectively represent the data of the first and the second round of GAIA.}
 
 {clll}
 
 Category & Source & Postive & Negtive \\
 {*}{$$} & AndroidControl & 68.2k & 69.9k \\
 & GUI-Odyssey & 65.4k & 66.8k \\
 {*}{$^+$} & AndroidControl & (68.2+15.1)k & (69.9+14.0)k \\
 & GUI-Odyssey & (65.4+26.1)k & (66.8+26.3)k \\
 
 

for inference on the AndroidControl [li2024effects] and GUI-Odyssey [lu2024gui] training sets, and compare the real actions with GT to build $$ and $^+$. 
On the corresponding data, we develop the ICM and ICM-r2 based on Qwen2.5 VL 7B [bai2025qwen2] and adopt the ms-swift [zhao2024swiftascalablelightweightinfrastructure] framework for training.

All action judgments followed the high-level approach, providing only global instructions to the ICM and ICM-r2, not single-step instructions. The distribution of the data flywheel is shown in Table §tab_data. The critic model guides the agents in the N-rollout process with $N=8$. To allow the base agent to sample a reasonable range of potential actions, its temperature coefficient, top_k, and top_p are set to 1.0, 30, and 0.8, respectively. All experiments are conducted on 8 NVIDIA H100-80G GPUs.

**Table:** ***GUI grounding accuracy on ScreenSpotV2.** $^*$ indicates reproduced open-source agent performance. $$ and $^+$, but ICM and ICM-r2 still achieved significant performance improvements, demonstrating the inherent consistency of real action space data. The real action sampling in the proposed data flywheel effectively covers this space, providing effective support for critic training.

Both our ICM and ICM-r2 use a high-level approach to judge correctness and guide action, meaning the critic model is not aware of the current action plan. This setting is more consistent with practical applications, where only global instructions are given, and the agent must independently reason about each step's plan and action. For AndroidControl-Low, the agent is aware of the current action plan, resulting in higher baseline performance. Despite this, our ICM still achieves a certain degree of performance improvement, demonstrating the effectiveness of our proposed approach.

Table §tab_grounding shows the improvement of ICM and ICM-r2 on the grounding ability of agents on ScreenSpotv2. The ScreenSpotv2 data is not included in the proposed GAIA, and as a single-step operation, its environmental information is not completely consistent with the aforementioned datasets. Even so, our evaluation model still improves the performance of agents, which is sufficient to prove the validity of the data flywheel definition.

Please refer to the appendix for more visualization results.

### Ablation Study

While utilizing ICM and ICM-r2, we used N-rollout to improve the test-time performance of existing GUI agents, where $N$ is set to 8 by default. To measure the impact of $N$ on final performance, we selected GPT4o and UI-TARS 1.5$^*$ as representative closed-source and open-source models, respectively, and compared their SR at different $N$ values on GUI-Odyssey. We also measured the models' Pass@N during the rollout process to reflect the model's performance ceiling. As shown in Figure §fig_N, the increase in Pass@N accuracy reveals the potential of the agents themselves, while the evaluation model approaches this upper limit through N-rollout. The improvement in ICM-r2 and the gap between the upper limit provide potential performance gains for further cycles of GAIA.

**Figure:** *GPT4o on GUI-Odyssey* () _(image: Figs/fig4_1.pdf)_

**Table:** ***Critic comparison.** The performance is calculated as (Qwen2.5 VL 7B w/ critic) - (Qwen2.5 VL 7B w/o critic).*

| !60 ICM | 8 | 1.0 | 5.0 | 2.8 |
| — | — | — | — | — |
| !60 ICM-r2 | 8 | 1.4 | 5.0 | 3.2 |
| height 1.0pt |

### Qualitative Experiment

**Critic Model Comparison.**
To evaluate the effectiveness of the proposed discriminant model, we compared the accuracy of the Qwen 2.5 VL using a best-of-N approach to guide inference on AndroidControl-High with UI-Genie-RM [xiao2025ui]. As shown in Table §tab_reward, due to the use of real action data, the proposed ICM significantly improves the accuracy of the base model, and ICM-r2 can further expand the advantage.

**Intuitive and Reasoning Critic.**
To verify that the intuitive judgment proposed in this article is superior, this section implements a critic model based on reinforcement learning design. Specifically, the input of the Reasoning Critic Model (RCM) is consistent with ICM, and the output includes ... and ..., which are supervised by format reward and critic reward. The training of RCM is achieved through Group Relative Policy Optimization (GRPO). Considering the property of reinforcement learning, which is that it can stimulate model capabilities with less data, we randomly sampled 30k data from $^+$ to train RCM. This data includes samples from two rounds of GAIA and has the same distribution as the training data for ICM-r2. To intuitively compare the discriminative performance of different critic models, we collected the GAIA test set in a high-level manner on the AndroidControl and GUI-Odyssey test sets in the same way as we collected the training data.

Table §tab_critic shows that the proposed ICM achieves an accuracy of 83.19% for correctness assessment, providing a foundation for action guidance. ICM-r2, benefiting from improved data quality, further achieves an accuracy of 83.56%. In contrast, RCM's classification accuracy is 70.82%, indicating that the thinking component fails to significantly contribute to the final assessment. In terms of action guidance accuracy, while UI-TARS 1.5$^*$ under RCM guidance outperforms the original model, it still falls short of ICM. This experimental result demonstrates that intuitive judgment outperforms reasoning for improving agents using a critic model.

## Conclusion

In this work, we addressed the critical challenge of high-stakes, irreversible errors in GUI agents by proposing a novel framework designed to unleash their latent potential at test time. Our GUI Action Critic's Data Flywheel System (GAIA) comprises a data flywheel that iteratively curates a dataset of realistic action samples and the Intuitive Critic Model (ICM) that evaluates action correctness. This framework establishes a self-evolutionary cycle: the flywheel continuously enriches its data, which in turn trains an increasingly powerful critic (ICM-r2). By leveraging a Best-of-N strategy, our ICM enables agents to select more reliable actions without the need for resource-intensive retraining. Experimental results on both closed-source and open-source agents demonstrate that GAIA provides significant performance gains in task planning and grounding capabilities, presenting a promising, scalable solution for building more robust and intelligent GUI agents.

In future work, we will consider unifying high-level and low-level guidance methods and collecting richer data in online testing, thereby continuously iterating the data flywheel and promoting the exploration of agent capabilities.

## Ethics statement

The research content of this paper is based on the LVLM GUI Agent. The research process of this paper does not violate ICLR ethics. There are no discrimination, bias, or fairness issues that need to be addressed. Our models are not expected to generate potentially harmful content.

## Reproducibility statement

This article studies a GUI Agent based on LVLM, focusing on proposing a critic model for existing agent actions. The base model and dataset used in this article are all from open-source and well-referenced, so this aspect does not affect the reproducibility. To further ensure reproducibility, we describe the parameters in detail in the main text Section §exp_implementation and the appendix, and provide prompts for all models involved in the appendix. We will release the source code and model checkpoints to support reproducibility.




## Clarification of the Usage of LLMs

This paper only used LLMs to assist and polish the writing. The retrieval, core innovation, method design, and experiments related to the paper were not conducted with the help of LLMs.

## Inference Prompts

In this section, we introduce the inference parameters and prompt template of the LVLM used. The proposed GAIA data flywheel and trained ICM are used to guide existing GUI agents at test time. GAIA data is also constructed using the action reasoning of existing agents.

### GUI Agent Planning Task Prompts

**GPT4o** We use the API to test the closed-source model GPT4o [hurst2024gpt] as a GUI agent. Prompts refer to UI-TARS 1.0 [qin2025ui] to ensure consistency in the action space. System prompts and user prompts refer to Prompt §p1 and Prompt §p3. The "You need to: {step_plan}" in User Prompt is only used when testing AndroidControl-Low [li2024effects]. In other conditions, the agent will not receive specific step instructions.

**Doubao$^$** Doubao, ByteDance's closed-source model interface, provides a closed-source version of UI-TARS 1.5, which represents the most advanced GUI agent. We test the performance of closed-source UI-TARS 1.5 by calling Doubao's API [doubao]. The Prompt used for the test is consistent with the open source version UI-TARS 1.5 [qin2025ui], see Prompt §p2 and Prompt §p3. The "You need to: {step_plan}" in User Prompt is only used when testing AndroidControl-Low. In other conditions, the Agent will not receive specific step instructions.

**UI-TARS 1.0$^*$** For UI-TARS 1.0 [qin2025ui], we use the open source weight UI-TARS-7B-DPO for testing. System prompts and user prompts refer to Prompt §p1 and Prompt §p3. The "You need to: {step_plan}" in User Prompt is only used when testing AndroidControl-Low. In other conditions, the agent will not receive specific step instructions. It should be noted that UI-TARS can receive up to five historical images as input. To simplify the process, we only use the text of "action_history" to describe the historical steps. For the image, we only input the current screenshot.

**UI-TARS 1.5$^*$** For UI-TARS 1.5 [qin2025ui], we use the open source weight UI-TARS-1.5-7B for testing. System prompts and user prompts refer to Prompt §p2 and Prompt §p3. The "You need to: {step_plan}" in User Prompt is only used when testing AndroidControl-Low. In other conditions, the agent will not receive specific step instructions. It should be noted that UI-TARS 1.5 can receive up to five historical images as input. To simplify the process, we only use the text of "action_history" to describe the historical steps. For the image, we only input the current screenshot.

**Qwen 2.5 VL** For Qwen 2.5 VL [bai2025qwen2], we use the open source weight Qwen-2.5-VL-7B-Instruct for testing. We refer to the official use case and use the function calls to test Qwen's GUI Agent capabilities. The prompt is shown in Prompt §p41 and §p42.

### GUI Agent Grounding Task Prompts

Grounding capabilities are tested on the ScreenSpotV2 [cheng2024seeclick] dataset. Because Grounding only provides single-step instructions and screenshots, and operations only involve click locations, the Prompts in Grounding differ from those in planning. The prompts for UI-TARS 1.0$^*$ and UI-TARS 1.5$^*$ refer to Prompt §p5 and Prompt §p6 respectively. The Qwen 2.5 VL test also applies function calls and constrains its output format through JSON, where Prompt is shown in Prompt §p41 and §p42. Prompt is the same as planning task, but without "action_history".

### ICM Prompts

The ICM and ICM-r2 trained on GAIA follow the same prompt and make judgments across the full action space. Therefore, the same prompt is used for both Planning and Grounding, as shown in Prompt §p7. The information of "global_instruction" and "action_history" is consistent with that obtained by the basic GUI agent. "actor_set" describes the current action. If the action is click or long press, it is constructed as "Tap at [x, y]", where x and y are the absolute coordinates of the click position in the original image. If the action is swipe, "actor_set" is constructed as "Swipe to up/down/left/right". If the action is type or open, "actor_set" is constructed as "Type/Open [text]", where[text] is the input text or the name of the App to be opened. The other actions have no parameters, so "actor_set" is directly the action name, such as "Wait", "Home", "Back", etc.

For the input image, we refer to the Set-of-Mark (SoM) approach [yang2023setofmark]. If the action is click or long press, a red circle is drawn at the click location. Otherwise, the original image is used directly as the ICM reference. The model's attention is implemented using FlashAttention [dao2022flashattention]. The data type is bfloat16. The epoch is 1, and the batch size is 16. 

## Training Parameters

Using GAIA data, we train ICM and ICM-r2 with the following parameters. We fine-tune Qwen 2.5 VL 7B by inserting LoRA [hu2022lora] into all linear layers, with lora_rank set to 8 and lora_alpha set to 32. The epoch is 1, and the batch size is 16. The optimizer is AdamW with a learning rate of 1e-4 and a warmup ratio of 0.05.

## Best-of-N Method

ICM and ICM-r2 use the Best-of-N approach to select the correct action with the highest probability from the N actions in the GUI agent rollout as the actual output, where the probability is expressed as the probability of the "correct" token. The core code of this process is shown in Code §c1.

## Visualization Results

We show the actions of the basic GUI agents on the sample, as well as the actions after being guided by ICM and ICM-r2. The comparison is shown in Figure §vis_f1 to §vis_f4.

[t]
 
 [blue]{GPT4o and UI-TARS 1.0$^*$ System GUI Prompt}
 
"`

You are a GUI agent. You are given a task and your 
action history, with screenshots. You need to perform 
the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space

click(point='(x1 y1)')
long_press(point='(x1 y1)')
type(content=")
scroll(point='(x1 y1)', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='(x1 y1)', end_point='(x2 y2)')
press_home()
press_back()
finished(content='xxx')

## Note
- Use English in Thought part.
- Summarize your next action (with its target element) in one 
sentence in Thought part.

## User Instruction
 
"`

 





[t]

 [blue]{Doubao$^$ and UI-TARS 1.5$^*$ System GUI Prompt}
 
"`

You are a GUI agent. You are given a task and your 
action history, with screenshots. You need to perform 
the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space

click(point='<|box_start|>(x1 y1)<|box_end|>')
long_press(point='<|box_start|>(x1 y1)<|box_end|>')
type(content=")
scroll(point='<|box_start|>(x1 y1)<|box_end|>', 
 direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<|box_start|>(x1 y1)<|box_end|>', 
 end_point='<|box_start|>(x2 y2)<|box_end|>')
press_home()
press_back()
finished(content='xxx')

## Note
- Use English in Thought part.
- Summarize your next action (with its target element) in one 
sentence in Thought part.

## User Instruction
 
"`

 





[t]

[blue]{GPT4o, Doubao$^$, UI-TARS 1.0$^*$ and UI-TARS 1.5$^*$ User GUI Prompt}

"`

- User Instruction
{global_instruction} (You need to: {step_plan})

- Action History
{action_history}

- Current Screenshot
{image_path}

"`






[t]

[blue]{Qwen 2.5 VL GUI and Grounding Prompt}

"`

You are a GUI Agent.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> 
XML tags:
<tools>
{"type": "function", 
 "function": 
 {
 "name": "mobile_use", "description": "Use a touchscreen to 
 interact with a mobile device, and take screenshots.
 * This is an interface to a 
 mobile device with touchscreen. You can perform actions like 
 clicking, typing, swiping, etc.
 * Some applications may take time to start or 
 process actions, so you may need to wait and take successive 
 screenshots to see the results of your actions.
 * The screen\'s resolution is 1092x2408.
 * Make sure to click any buttons, links, icons, etc with the 
 cursor tip in the center of the element. Don't click boxes 
 on their edges unless asked.", 
 "parameters": {
 "properties": {
 "action": {
 "description": "The action to perform. The available 
 actions are:
 * `key`: Perform a key event on the mobile device.
 - This supports adb\'s `keyevent` syntax.
 - Examples: \\"volume_up\\", \\"volume_down\\", \\"power\\", 
 \\"camera\\", \\"clear\\".
 * `click`: Click the point on the screen with 
 coordinate (x, y).
 * `long_press`: Press the point on the screen with coordinate 
 (x, y) for specified seconds.
 * `swipe`: Swipe from the starting point with coordinate 
 (x, y) to the end point with coordinates2 (x2, y2).
 * `type`: Input the specified text into the activated 
 input box.
 * `system_button`: Press the system button.
 * `open`: Open an app on the device.
 * `wait`: Wait specified seconds for the change to happen.
 * `terminate`: Terminate the current task and report its 
 completion status.", 
 "enum": ["key", "click", "long_press", "swipe", "type", 
 "system_button", "open", "wait", "terminate"], 
 "type": "string
 }, 

"`







[t]

[blue]{Qwen 2.5 VL GUI and Grounding Prompt (cont.)}

"`

 "coordinate": {"description": "(x, y): The x (pixels from 
 the left edge) and y (pixels from the top edge) 
 coordinates to move the mouse to. Required only by 
 `action=click`, `action=long_press`, and `action=swipe`.", 
 "type": "array"}, 
 "coordinate2": {"description": "(x, y): The x (pixels from 
 the left edge) and y (pixels from the top edge) 
 coordinates to move the mouse to. Required only by 
 `action=swipe`.", "type": "array"}, 
 "text": {"description": "Required only by `action=key`, 
 `action=type`, and `action=open`.", "type": "string"}, 
 "time": {"description": "The seconds to wait. Required only 
 by `action=long_press` and `action=wait`.", 
 "type": "number"}, 
 "button": {"description": "Back means returning to the 
 previous interface, Home means returning to the desktop, 
 Menu means opening the application background menu, 
 and Enter means pressing the enter. Required only 
 by `action=system_button`", 
 "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, 
 "status": {
 "description": "The status of the task. Required only 
 by `action=terminate`.", 
 "type": "string", "enum": ["success", "failure"]
 }
 }, 
 "required": ["action"], "type": "object"
 }
 }
}
</tools>

For each function call, return a json object with function name 
and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

The user query: 
{global_instruction}
Task progress (You have done the following operation on 
the current device): 
{action_history}

{image_path}

"`







[t]

[green]{UI-TARS 1.0$^*$ GUI Grounding Prompt}

"`

You are a GUI agent. You are given a task and your action 
history, with screenshots. You need to perform the next 
action to complete the task. 

## Output Format

Action: ...

## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{instruction}

{image_path}

"`







[t]

[green]{UI-TARS 1.5$^*$ GUI Grounding Prompt}

"`

You are a GUI agent. You are given a task and your action 
history, with screenshots. You need to perform the next 
action to complete the task. 

## Output Format

Action: ...

## Action Space
click(point='<|box_start|>(x1,y1)<|box_end|>')

## User Instruction
{instruction}

{image_path}

"`







[t]

[orange]{ICM and ICM-r2 Critic Prompt}

"`

You are an expert in evaluating the performance of a phone 
operating agent. The agent is designed to help a user to 
complete a task or retrieve information from the phone.
Given the user's task instruction, current action and current 
screenshot, your goal is to decide whether the agent's current 
action is correct or not.
Each action in the sequence is preceded by a corresponding 
screenshot that captures the context in which the action occurs.

## Evaluation Criteria
Whether the agent's current action is correct and corresponding 
to the user's task instruction.

## IMPORTANT
1. An action always follows a corresponding screenshot (even if 
only the last few are provided).
2. If the current action is a tap on the screen, the point where 
the action is clicked is marked with a red circle on 
the screenshot.
3. You should whether answer [correct] or [wrong].

## Input
The input is given next, including global_task_instruction, 
action_history, current_action, and screenshot.
The goal of the task (instruction): {global_instruction}
Action (plan) history: {action_history}
Current action of the agent: {actor_set}
Screenshot: {som_image_path}

"`







[t]

[yellow]{Best-of-N Example Code}

"`

# 1. construct critic input
texts = [
 critic_processor.apply_chat_template(msg, tokenize=False, 
 add_generation_prompt=True) for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = critic_processor(
 text=texts,
 images=image_inputs,
 videos=video_inputs,
 padding=True,
 return_tensors="pt",
)
inputs = inputs.to(critic_device)

# 2. generate output
output = critic_model.generate(**inputs, 
 do_sample=False, 
 max_new_tokens=2048, 
 return_dict_in_generate=True,
 output_scores=True)
generated_ids = output.sequences

# 3. get token score
scores = output.scores[0]
critic_scores = scores[:,-2]

# 4. get output text: correct|wrong
for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
 generated_ids_trimmed = [out_ids[len(in_ids) :]]
responses = critic_processor.batch_decode(
 generated_ids_trimmed, skip_special_tokens=True, 
 clean_up_tokenization_spaces=False
)

# 5. find the index of the best action
max_score = -float('inf')
best_idx = -1
for idx in range(len(critic_outputs)):
 if responses[idx] == 'correct':
 if critic_scores[idx] > max_score:
 max_score = critic_scores[idx]
 best_idx = idx 

"`







**Figure:** ***Visualization result.** The basic agent selects the wrong action. Based on the action rollout, both ICM and ICM-r2 select the correct action from the candidates.* () _(image: Figs/vis1.pdf)_

**Figure:** ***Visualization result.** The basic agent selects the wrong action. Based on the action rollout, both ICM and ICM-r2 select the correct action from the candidates.* () _(image: Figs/vis2.pdf)_

**Figure:** ***Visualization result.** The basic agent selects the wrong action. ICM fails to select the correct one from the rollout candidates, while the enhanced ICM-r2 guides the correct selection.* () _(image: Figs/vis3.pdf)_

**Figure:** ***Visualization result.** The basic agent selects the wrong action. ICM fails to select the correct one from the rollout candidates, while the enhanced ICM-r2 guides the correct selection.* () _(image: Figs/vis4.pdf)_