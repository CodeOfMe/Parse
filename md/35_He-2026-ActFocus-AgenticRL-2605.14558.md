# Resolving Action Bottleneck: Agentic Reinforcement Learning Informed by Token-Level Energy

> **arXiv:** [2605.14558](https://arxiv.org/abs/2605.14558)
> **TeX source:** [arXiv-2605.14558v1/](arXiv-2605.14558v1/)
> **PDF:** [ActFocus-arXiv-2605.14558v1.pdf](ActFocus-arXiv-2605.14558v1.pdf)

---

## Abstract

Agentic reinforcement learning trains large language models using multi-turn trajectories that interleave long reasoning traces with short environment-facing actions. Common policy-gradient methods, such as PPO and GRPO, treat each token in a trajectory equally, leading to uniform credit assignment. In this paper, we critically demonstrate that such uniform credit assignment largely misallocates token-level training signals. From an energy-based modeling perspective, we show that token-level training signals, quantified by their correlations with reward variance of different rollouts sampled from a given prompt, concentrate sharply on action tokens rather than reasoning tokens, even though action tokens account for only a small fraction of the trajectory. We refer to this phenomenon as the Action Bottleneck. Motivated by this observation, we propose an embarrassingly simple token reweighting approach, *ActFocus*, that downweights gradients on reasoning tokens, along with an additional energy-based redistribution mechanism that further increases the weights on action tokens with higher uncertainty. Across four environments and different model sizes, *ActFocus* consistently outperforms PPO and GRPO, yielding final-step gains of up to 65.2 and 63.7 percentage points, respectively, without any additional runtime or memory cost.

## Introduction

As large language models (LLMs) are increasingly deployed as autonomous agents that combine reasoning [guo2025deepseek, jaech2024openai], planning [zhou2023webarena], and tool use [wolflein2025llmagentsmakingagent] to solve multi-turn, long-horizon tasks, reinforcement learning (RL) has emerged as the natural post-training framework for sequential decision-making settings [wang2025ragenunderstandingselfevolutionllm, wang2024survey, xi2025agentgymrltrainingllmagents, abdulhai2023lmrlgymbenchmarksmultiturn]. Despite rapid progress with different policy optimization methods [schulman2017proximalpolicyoptimizationalgorithms, shao2024deepseekmathpushinglimitsmathematical], training LLM agents with RL remains difficult. One of the core challenges is credit assignment [pignatelli2024surveytemporalcreditassignment, kazemnejad2025vinepporefiningcreditassignment], where, in multi-turn trajectories, a sparse episodic reward must be propagated back through thousands of generated tokens, making it difficult to determine which parts of a trajectory should drive learning. The problem is compounded by optimization instability [cui2025entropymechanismreinforcementlearning], as delayed feedback, long generated traces, and noisy policy updates often lead to degraded late-stage performance or collapse. 

Recent evidence shows that multi-turn, underspecified conversations substantially increase outcome variability for LLMs, sharpening the need for reliable credit assignment in agentic RL [laban2025llmslostmultiturnconversation]. In a typical agentic RL rollout, such outcome variability often hinges on a few critical environment-facing action tokens that drive state transitions, yet these tokens are buried within massive amounts of reasoning tokens in the trajectory [yao2023reactsynergizingreasoningacting, schick2023toolformerlanguagemodelsteach, wen2024reinforcinglanguageagentspolicy]. Even though these action tokens are the only parts of the output that directly interact with the environment, standard policy optimization methods treat all generated tokens uniformly [schulman2017proximalpolicyoptimizationalgorithms, liu2025uniformheterogeneoustailoringpolicy]. Consequently, gradient updates are spread equally across the whole trajectory, diluting training signals for the critical action tokens that trigger the decision-making [chan2024denserewardfreereinforcement, vassoyan-etal-2025-ignore]. In multi-turn settings, this mismatch compounds with each additional turn: reasoning tokens keep accumulating while the environment is driven by only a few informative actions. 

Figure §fig:action-bottleneck shows that this structural mismatch manifests as token-level signal sparsity in agentic RL. In a representative Sokoban trajectory, action tokens constitute only 4% of the generated output, while reasoning tokens account for the remaining 96%. Under uniform credit assignment, gradient mass therefore follows token count and is dominated by the reasoning majority. However, in this paper, we demonstrate that the training signals follow the opposite pattern. We quantify each token's training signal as its predictive uncertainty under a frozen reference model, measured by the free energy of its next-token predictive distribution. [^note: An ideal indicator for quantifying training signals should reveal where outcome uncertainty concentrates, stay stable across training, and be computationally inexpensive. Energy, used in this work, satisfies all of these criteria; alternatives such as entropy and log-likelihood are compared empirically in Sec. §sec:signal-sweep.] To further evaluate how informative the signal is for training, we hinge it on outcome variability: for each prompt, we calculate the variance of episodic rewards $_g$ across multiple rollouts (sampled from the same prompt), which captures how uncertain the current policy is about the prompt's outcome. We find that the mean energy of action tokens is strongly correlated with reward variance (with $=+0.537$), while the mean energy of reasoning tokens or full trajectories, which average over both action and reasoning tokens, is nearly uninformative (with $0$). **This suggests that informative training signals are concentrated in a small number of action tokens rather than being distributed uniformly across the trajectory.** We refer to this phenomenon as the *Action Bottleneck*.

**Figure:** ***Action Bottleneck in agentic reinforcement learning.**
*Left:* A real response from a trajectory, where reasoning tokens far outnumber action tokens. 
*Middle:* Action tokens constitute only 4% of model-generated tokens; our token-level reweighting redirects gradient mass towards them. 
*Right:* Training signal concentrates in action spans. Results are from Sokoban 3B.* () _(image: figures/fig1_action_bottleneck1.png)_

Motivated by this observation, we propose an energy-based approach to token-level credit assignment in agentic RL. The method is conceptually intuitive and can be decomposed into two aspects: 1) We downweight gradients on reasoning tokens, which is equivalent to increasing the weights on action tokens and helps to redirect gradient mass toward the sparse action span that determines environmental outcomes; 2) We utilize token-level energy to further redistribute the weights across action tokens, which prioritizes action tokens with high energy (i.e., more predictive uncertainty). In this way, we transform our empirical observations into an actionable approach that systematically guides the model to better leverage the informative training signals underlying a few critical tokens in the long trajectory.

We evaluate our approach, *ActFocus*, on four representative multi-turn agentic environments using Qwen2.5-Instruct [qwen2025qwen25technicalreport] models at multiple scales, and train each configuration with both PPO [schulman2017proximalpolicyoptimizationalgorithms] and GRPO [shao2024deepseekmathpushinglimitsmathematical]. Through these experiments, we find that *ActFocus* produces consistent performance gains over PPO and GRPO, both of which use uniform credit assignment, with final success-rate improvements of up to 65.2 and 63.7 percentage points, respectively. Beyond absolute performance metrics, we observe that *ActFocus* effectively improves GRPO's training stability by substantially reducing peak-to-final degradation. Our main contributions are summarized as follows:

**(i)** We analyze token-level attributions in agentic RL training through an energy-based modeling perspective and reveal a counter-intuitive phenomenon: while reasoning tokens dominate in long-horizon multi-turn trajectories, infrequent action tokens contribute the majority of informative training signals. In particular, the uncertainty in action tokens, as measured by the free energy of next-token predictive distributions, is highly correlated with outcome variability. 

**(ii)** We propose a token reweighting mechanism that downweights gradients on reasoning tokens and additionally redistributes the weights on action tokens according to their uncertainty levels as characterized by the free energy. This simple scheme shifts gradient mass towards the most informative action tokens while remaining fully compatible with common policy-gradient algorithms.

**(iii)** Through extensive experiments across four environments and multiple model scales, *ActFocus* consistently outperforms PPO and GRPO, with average gains of 17.0 and 34.6 percentage points and final-step gains of up to 65.2 and 63.7 percentage points, respectively. In particular, *ActFocus* can significantly enhance the training stability of GRPO by mitigating the peak-to-final degradation observed under uniform credit assignment.

## Related Work

**Agentic Reinforcement Learning for LLMs..** 
Reinforcement learning has become a central paradigm for aligning large language models with sequential decision-making objectives beyond single-turn text generation [ouyang2022traininglanguagemodelsfollow]. 
Recent work extends RL to multi-turn agent settings from different angles. For instance, LOOP [chen2025reinforcementlearninglonghorizoninteractive] formulates interactive digital agents as RL in stateful environments, while RAGEN [wang2025ragenunderstandingselfevolutionllm] presents a general framework for multi-turn agent training under stochastic environments. ARLArena [wang2026arlarenaunifiedframeworkstable] systematizes the study of training stability by decomposing policy-gradient design into core dimensions. However, despite these advances, credit assignment in existing agentic RL methods remains relatively under-explored and does not explicitly distinguish between different parts of an agent trajectory, particularly reasoning and action tokens.
 

**Credit Assignment in Agentic RL..** 
A growing line of work argues that uniform credit assignment largely under-represents the token-level training signals of long-horizon agent trajectories. Most existing efforts operate at the step or trajectory level. CARL [shen2026carlfocusingagenticreinforcement] uses policy entropy as a proxy for action importance and concentrates optimization on critical actions across turns. GiGPO [feng2025groupingrouppolicyoptimizationllm] introduces a two-level advantage estimator based on anchor states that reappear across trajectories, enabling step-level relative credit assignment. iStar [liu2025agenticreinforcementlearningimplicit] learns implicit step-wise rewards through trajectory-level preference optimization and combines step-level and trajectory-level signals in policy updates. These methods recognize that credit should vary across an agent trajectory, but they remain coarse in granularity and do not distinguish reasoning tokens from action tokens within each step. Several works in reasoning RL have explored finer-grained reward redistribution and intermediate supervision [tan2026gtpogrpostokensequencelevel,cui2025processreinforcementimplicitrewards, ma2026fipoelicitingdeepreasoning], but these approaches are designed for single-turn reasoning without diving into the reasoning-action disentanglement in multi-turn agent trajectories.
 

## Preliminaries

### Multi-Turn Agentic RL

We formulate multi-turn agentic RL as a Markov Decision Process $ = , , P, R $ [sutton1998reinforcement], where states $s $ are interaction histories, actions $a $ are environment-facing operations available to the agent, the transition $P(s_{k+1} s_k, a_k)$ specifies how the environment responds to an action, and the reward $R()$ assigns a scalar episodic return based on the terminal state of the trajectory. At each turn $k $, the policy $_$ observes the current state $s_k$ and generates a structured response:

$$

o_k = `<think>`...`</think>`\;`<answer>`\;a_k\;`</answer>`,
$$

where the `<think>` block contains the agent's reasoning and the `<answer>` block contains the action $a_k$. The environment extracts $a_k$ from $o_k$, transitions to the next state $s_{k+1}$, and returns a reward $R()$ at episode termination. The full trajectory is $= (s_1, o_1, s_2, o_2, ..., s_K, o_K)$, and the RL objective is to maximize expected episodic reward: $J() = _{_}[R()]$.

**Token-level policy optimization..** 
Each response $o_k$ consists of tokens $(y_{k,1}, ..., y_{k,T_k})$ from both the reasoning and action segments. For notational simplicity, we drop the turn index and use $y_t$ for a generic response token. Both PPO and GRPO optimize this objective via a clipped token-level surrogate loss that shares the same general form:

$$

() =
-{||}
_{(y_t,A_t)}
w_t \!(
_t()A_t,\;
(_t(),1-,1+)A_t
),
$$

where $_t() = _(y_tc_t) / _{_{}}(y_tc_t)$ is the importance ratio between the current and old policy, $$ is the clipping threshold, and $_t$ is the per-token advantage. Here, $c_t$ denotes all tokens preceding $y_t$, to distinguish it from the environment state $s_k$, and $w_t$ is a per-token weight.

PPO estimates $_t$ via Generalized Advantage Estimation (GAE) [schulman2018highdimensionalcontinuouscontrolusing] with a learned value function, providing a distinct advantage for each token position. GRPO eliminates the critic and instead computes a single trajectory-level advantage:

$$

_= (R())}{(R())},
$$

where the mean and standard deviation are computed over all trajectories sampled from the same prompt. Under GRPO, every token in a trajectory shares the same advantage: $_t = _$ for all $t$ in $$. In both algorithms, the common practice is to set $w_t = 1$ for all response tokens, which corresponds to the uniform credit assignment that we critically challenge in this paper. Complete descriptions for algorithm-specific objectives and implementation details are deferred to Appendix §app:rl-objectives.
 

**Table:** *Action token fraction in inference-only rollouts before RL training. The imbalance is consistent across environments and model scales.*

_(table content)_

### Frozen-Reference Token Energy

Recent work [grathwohl2020classifiersecretlyenergybased, xie2016theory] establishes an equivalence between neural classifiers and energy-based models (EBMs). Given a classifier that maps inputs to $C$-dimensional logits $h(x) ^C$, the classifier-EBM equivalence is established by setting the energy of an input-label pair as $E(x, y) = -h(x)[y]$. Marginalizing over labels yields a free energy $E(x) = -_{c} (h(x)[c])$, which captures the uncertainty of the model for a given input [liu2020energy]: lower energy indicates a confident, peaked prediction, while higher energy indicates uncertainty across multiple plausible outputs.

We extend this idea to autoregressive language models, which predict a distribution over the vocabulary at each position. At each token position $t$, a frozen reference model produces next-token logits $f_t^{} ^{|V|}$ over vocabulary $V$, conditioned on the preceding context. We define the **token-level energy** as:

$$

E_t = -_{v V} \!(f_{t,v}^{}).
$$

A lower $E_t$ corresponds to higher confidence under the reference model, while a higher $E_t$ indicates greater uncertainty about the next token. This signal is well-suited for diagnosing agentic RL for two reasons. First, because the reference model is frozen, $E_t$ remains stable throughout training, unlike policy-derived quantities such as entropy or confidence that drift as the policy evolves. Second, computing $E_t$ requires only a single forward pass, making it cheaper than Monte Carlo estimates that require repeated rollouts [shen2026carlfocusingagenticreinforcement]. We use token-level energy to analyze where the reward-predictive signal concentrates within agent trajectories, and this analysis motivates the intervention in Sec. §sec:method.

## Token Credit Assignment

We now turn the Action Bottleneck diagnosis into a token-level weighting mechanism that aims to address two issues: 1) reasoning tokens dominate the trajectory, and 2) training signals are not uniformly distributed across action tokens. Table §tab:token-composition reveals the imbalance: across environments and model scales, action tokens account for less than 16% of generated tokens in a trajectory. Given this, uniform credit assignment would concentrate most gradient mass on reasoning spans.

To further validate the results in Figure §fig:action-bottleneck across different cases, we examine where the training signals live within a trajectory. For each prompt in the training set, we run $G$ independent rollouts under the current policy, each producing a complete multi-turn trajectory and its episodic reward $R()$. We refer to the $G$ trajectories sampled from the same prompt as a *trajectory group*, with group reward variance $_g = (R(^{(1)}), ..., R(^{(G)}))$. A high $_g$ means the policy sometimes succeeds and sometimes fails on this prompt, so the prompt provides useful learning signals. For each trajectory group, we then compute the mean frozen-reference token energy over three token subsets: the full response, reasoning tokens only, and action tokens only. We measure the Spearman correlation between the mean energy of each subset and $_g$ across all trajectory groups. Figure §fig:energy-sigma-corr shows the result on Sokoban 3B. We find that action-only energy correlates with $_g$, with $=+0.537$ and $p<10^{-30}$, while full-response and reasoning-only energy are indistinguishable from random noise. This shows that the training signals captured by frozen-reference energy largely stem from action tokens instead of reasoning tokens that dominate in the trajectory.

**Figure:** *
 **Detailed energy–reward diagnostic behind Figure §fig:action-bottleneck**.* () _(image: figures/fig2_energy_scope_scatter.png)_

The observation motivates a token-level reweighting scheme that proceeds in two steps. Specifically, we first use the explicit `<think>`/`<answer>` boundary to reallocate gradient mass from reasoning tokens towards environment-facing action tokens. We then use frozen-reference energy within the action span to prioritize positions where the reference model is more uncertain.

### Reasoning Downweighting and Energy-Based Action Prioritization

For each response token belonging to either a reasoning span $}$ or an action span $_{}$, as identified by the `<think>` and `<answer>` tags, we assign a weight:

$$

w_t =

, & t _{}, \\[4pt]
1 + \,\!({_E}), & t _{},

$$

where $[0,1]$ controls how much gradient is retained on reasoning tokens, $0$ controls how strongly energy modulates the weights on action tokens, and $E_t$ is the frozen-reference token energy from Eq. §eq:token-energy. $_E$ and $_E$ denote the mean and standard deviation of $E_t$ over all action tokens in the current training batch, respectively. The sigmoid function is applied to a batch-normalized action-token energy score, making energy modulation depend on relative uncertainty within the action-token distribution rather than on the raw energy scale.

When setting $=1$ and $=0$, which gives $w_t=1$ for all tokens, the reweighting mechanism described by Eq. §eq:token-weight reduces to standard uniform credit assignment. Our proposed approach uses $<1$ and $>0$, which simultaneously downweights reasoning tokens and applies energy-based redistribution within the action span. The former simply shifts the gradient mass towards informative action tokens, while the latter additionally prioritizes the action tokens for which the model is more uncertain.
In our experiments, we compare standard *Uniform* credit assignment against our proposed token reweighting approach, which we refer to as *ActFocus*.

### Modified Objective

We incorporate the token-level credits by replacing the averaging aggregation in Eq. §eq:ppo-loss with a weighted sum:

$$

()
=
-{_t w_t}
_t w_t \,
\!(
_t()\,_t,\;
\!(_t(),\,1{-},\,1{+})_t
),
$$

where $w_t$ is given by Eq. §eq:token-weight. 
Tokens with larger $w_t$ contribute more to the policy-gradient update, while tokens with smaller $w_t$ contribute less. Importantly, this modification does not change how the advantage is computed. For PPO, $_t$ still comes from the original critic-based estimator; for GRPO, it still comes from the group-normalized trajectory return.
Thus, the method only changes how the existing advantage signal is distributed across tokens.

We normalize by $_t w_t$ so that changing $$ or $$ does not simply make the whole loss larger or smaller. Instead, $$ and $$ control where the gradient mass goes: decreasing $$ reduces the contribution of reasoning tokens, while increasing $$ gives more weight to high-energy action tokens.

Ignoring PPO clipping for clarity, the weighted objective separates the update into a reasoning component and an action component:

$$

_
\;=\;
-{_t w_t}\![
_{}}
_t __(y_t c_t)}_{}
+
_{}}
w_t _t __(y_t c_t)}_{}
].
$$

The two terms correspond directly to the two steps introduced at the start of this section. Reducing $$ reallocates gradient mass from reasoning spans to action spans. Positive $$ then redistributes mass within the action span, giving larger weight to high-energy tokens where the frozen reference model is more uncertain.

**Table:** *PPO final-step results across four environments at two capability levels. Success rate (%) is reported at step 200 for Sokoban, FrozenLake, and Sudoku, and at step 100 for WebShop. WebShop reports both purchase rate and strict success. Peak values are reported in Table §tab:main-ppo-grpo-peak*

_(table content)_

## Experiments

We evaluate our proposed method on four multi-turn environments spanning distinct reasoning tasks: **Sokoban**, a spatial planning task where the agent pushes boxes onto targets on a grid; **FrozenLake**, a navigation task on a stochastic slippery surface; **Sudoku 4$$4**, a constraint-satisfaction task requiring logical deduction; and **WebShop** [yao2023webshopscalablerealworldweb], an e-commerce task involving product search, evaluation, and purchase. For WebShop, we report both purchase rate and strict success: purchase rate measures whether the agent completes a purchase, while strict success additionally requires the purchased item to match the user instruction. We use the **Qwen2.5-Instruct family** [qwen2025qwen25technicalreport] throughout all experiments. For each environment, we use two model scales to test whether the Action Bottleneck and reweighting benefits persist across different base-model capabilities. We train each model with both PPO [schulman2017proximalpolicyoptimizationalgorithms] and GRPO [shao2024deepseekmathpushinglimitsmathematical], built on veRL following StarPO-S configuration [wang2025ragenunderstandingselfevolutionllm]. We compare two token-weighting schemes: *Uniform* applies standard equal credit 
and *ActFocus* applies the full structure-aware weighting rule in Eq. $ on WebShop 3B and train each setting with PPO for 100 steps. As shown in Figure §fig:beta-ablation, performance is non-monotonic. Strict success improves from 29.7% at $=0$ to 39.8% at $=1.0$, but drops to 25.4% at $=2.0$, falling below the $=0$ baseline. This indicates that energy modulation must be strong enough to differentiate uncertain action tokens from confident ones, since treating all action tokens equally fails to exploit the signal energy carries about which decisions are most informative. At the same time, $$ cannot be too large: when the sigmoid-based weighting becomes too peaked, gradient mass collapses onto a small number of high-energy outliers and leaves the remaining action tokens under-trained. 

**Figure:** ***Effect of $$.** Sweeping $$ on WebShop 3B: strict success peaks at $ 
 

## Notation

Table §tab:notation summarizes the notation used throughout the paper. Symbols are grouped by category for ease of reference.

**Table:** *Summary of notation used in the paper.*

| ll
**Symbol** | **Meaning** |
| — | — |
| $s_k $ | State at turn $k$ (interaction history) |
| $a_k $ | Environment-facing action at turn $k$ |
| $o_k$ | Policy response at turn $k$ (reasoning + action) |
| $= (s_1, o_1, ..., s_K, o_K)$ | Multi-turn trajectory |
| $K$ | Total number of turns in trajectory $$ |
| $T$ | Total number of response tokens in trajectory $$ |
| $R()$ | Terminal episodic reward |
| $r_t$ | Per-token reward; $r_t = R()$ if $t = T$, else $0$ |
| $__$ | Rollout policy (snapshot used for sampling) |
| $_$ | Frozen reference policy (KL anchor and energy source) |
| $y_t$ | Generic response token at position $t$ |
| $c_t$ | Context preceding $y_t$ (state, prior turns, prior tokens) |
| $_$ | Set of token positions inside `<think>` spans |
| $_$ | Set of token positions inside `<answer>` spans |
| $_t()$ | Token-level importance ratio $_(y_t c_t) / __(y_t c_t)$ |
| $_t$ | Per-token advantage estimate |
| $_$ | Trajectory-level (GRPO) advantage |
| $V_$ | Token-level value function used by PPO+GAE |
| $$ | PPO clipping threshold |
| $(, )$ | Discount factor and GAE coefficient |
| $G$ | Number of trajectories per trajectory group (GRPO) |
| $_, _$ | KL and entropy regularization coefficients |
| $E_t$ | Frozen-reference token energy at position $t$ |
| $_E, _E$ | Mean and std of $E_t$ over action tokens in a batch |
| $w_t$ | Per-token weight in the structure-aware loss |
| $$ | Reasoning-retention coefficient ($w_t$ on `<think>` tokens) |
| $$ | Action component energy modulation strength |
| $_g$ | Reward variance within a trajectory group |

## Reinforcement Learning Background

We follow the notation of Sec. §sec:prelim: $y_t$ denotes a generic response token, $c_t$ denotes the context consisting of all tokens preceding $y_t$, including state observations and prior turns within the trajectory, and the token-level importance ratio between the current policy and the rollout policy is:

$$
_t() = {_{_{}}(y_t c_t)}.
$$

Throughout this appendix, $T$ denotes the total number of response tokens in a trajectory $$.

**General objective..** 
The agentic RL objective maximizes expected episodic return over multi-turn trajectories:

$$
J() = _{_}[R()],
$$

where $= (s_1, o_1, ..., s_K, o_K)$ is a $K$-turn trajectory and $R()$ is the terminal episodic reward defined in Sec. §sec:prelim. The corresponding policy gradient is:

$$
_J() 
= _{_}\![_{t=1}^{T} _t\, __(y_t c_t)],
$$

which distributes credit across all response tokens $y_t$ in $$ via a per-token advantage estimate $_t$. PPO and GRPO differ primarily in how $_t$ is constructed.

**Proximal Policy Optimization (PPO)..** 
PPO [schulman2017proximalpolicyoptimizationalgorithms] stabilizes the policy update through a clipped surrogate objective combined with a learned token-level value function $V_$, which estimates the expected return-to-go from context $c_t$. The token-level surrogate loss is:

$$
_{}() 
= -_t\![\!(_t()_t,\; \!(_t(), 1-, 1+)_t)],
$$

where $$ is the clipping threshold. The advantage $_t$ is computed via Generalized Advantage Estimation (GAE) [schulman2018highdimensionalcontinuouscontrolusing]:

$$
_t^{(, )} 
= _{l=0}^{T-t} ()^l\, _{t+l}, 
_t = r_t + V_(c_{t+1}) - V_(c_t),
$$

where $$ and $$ control the bias and variance of the advantage estimate. Following standard practice in LLM RL, the episodic reward $R()$ is delivered at the final response token of the trajectory: $r_t = R()$ if $t = T$, and $r_t = 0$ otherwise. The value function $V_$ is trained jointly with the policy by regressing toward the empirical return-to-go.

**Group Relative Policy Optimization (GRPO)..** 
GRPO [shao2024deepseekmathpushinglimitsmathematical] removes the value function by sampling a group of $G$ trajectories ${^{(i)}}_{i=1}^{G}$ from the same prompt and computing a group-normalized trajectory advantage:

$$
_{^{(i)}} = ) - ({R(^{(j)})}_{j=1}^{G})}{({R(^{(j)})}_{j=1}^{G})},
$$

which is the explicit group-level form of $_$ in Eq. §eq:grpo-advantage. Because this advantage is defined at the trajectory level, it is broadcast to every token within the trajectory: $_t = _{^{(i)}}$ for all tokens $y_t$ in $^{(i)}$. The clipped surrogate then takes the form:

$$
_{}() 
= -{G}_{i=1}^{G} _t\![\!(_t()_{^{(i)}},\; \!(_t(), 1-, 1+)_{^{(i)}})].
$$

GRPO reduces memory and computation by removing the critic, but its uniform broadcasting of $_{}$ across all tokens makes it particularly sensitive to the Action Bottleneck studied in this work.

**Auxiliary regularizers..** 
Both PPO and GRPO are typically augmented with two regularizers. A KL term against a reference policy $_{}$ controls how far the trained policy drifts from its initialization:

$$
_{}() 
= _{}\,_t\![D_{}(_(c_t)\,\|\,_{}(c_t))],
$$

and an entropy bonus $_{}() = -_{}\,_t[H(_(c_t))]$ encourages exploration during early training. The total objective is $_{} = () + _{}() + _{}()$, where $()$ is the surrogate loss of either PPO or GRPO. In practice, the same frozen base model serves both as the reference policy $_{}$ for KL regularization and as the source of the frozen-reference logits used to compute token energy in Eq. §eq:token-energy. The specific values of $_{}$ and $_{}$ used in our experiments are listed in Appendix §app:experiments.

**Relation to our weighting scheme..** 
The structure-aware reweighting in Eq. §eq:ewaa-loss is orthogonal to both algorithms' choice of $_t$. Under PPO, $_t$ remains the GAE estimate; under GRPO, it remains the group-normalized trajectory advantage. *ActFocus* only changes the per-token weight $w_t$ that multiplies the surrogate, so it composes with PPO and GRPO without modifying their advantage estimators or auxiliary regularizers.

## Experimental Details

### Task Environment Details

We evaluate our method on four multi-turn environments that span a broad range of reasoning demands and interaction structures. Sokoban, FrozenLake, and Sudoku are compact symbolic environments with fully specified dynamics, making them well-suited for controlled analysis of reinforcement learning behavior and token-level credit assignment. Although all three are synthetic, they stress different forms of difficulty: irreversible planning in Sokoban, stochastic transitions in FrozenLake, and structured constraint reasoning in Sudoku. To complement these controlled settings, we also include WebShop [yao2023webshopscalablerealworldweb], a more realistic environment in which the agent must interpret natural-language instructions, navigate a semi-structured interface, and complete a purchase through a sequence of grounded actions. Taken together, these environments allow us to study the Action Bottleneck under both tightly controlled symbolic tasks and a more open-ended web interaction setting.

**Sokoban..** 
Sokoban is a grid-based planning task in which the agent must push boxes onto target cells within a limited action budget. Unlike standard navigation problems, Sokoban is irreversible: boxes can be pushed but not pulled back, so an early mistake can permanently destroy a valid solution path. This makes the environment a useful testbed for long-horizon planning and action precision in multi-turn agent training.

**FrozenLake..** 
FrozenLake combines goal-directed planning with stochastic transitions. The agent must navigate a grid world to reach the goal while avoiding holes, but actions are executed on slippery tiles and may lead to unintended outcomes. As a result, good performance depends not only on moving toward the goal but also on choosing actions that remain robust under transition uncertainty.

**Sudoku..** 
Sudoku evaluates structured reasoning in a sequential decision-making setting. The agent fills a $4 4$ grid one move at a time, subject to row, column, and subgrid constraints. Because each valid action depends on a set of coupled logical conditions, the environment places more emphasis on inference and constraint satisfaction than on exploration or long-horizon control.

**WebShop..** 
WebShop [yao2023webshopscalablerealworldweb] is a realistic multi-turn web interaction environment in which the agent must satisfy a natural-language shopping request through search, navigation, option selection, and purchase. Compared with the symbolic environments, WebShop introduces richer language grounding, a semi-structured interface, and a less cleanly defined action space. It therefore serves as a complementary setting for studying whether token-level credit assignment mechanisms continue to help when reasoning and action unfold in a more realistic decision-making process.
 

### Training and Evaluation Settings

We conduct our experiments using the **Qwen2.5-Instruct** family [qwen2025qwen25technicalreport] at three scales: 0.5B, 1.5B, and 3B. For each configuration, we train with both PPO [schulman2017proximalpolicyoptimizationalgorithms] and GRPO [shao2024deepseekmathpushinglimitsmathematical] under two token-weighting variants: *Uniform* and *ActFocus*. All experiments are built on the RAGEN framework [wang2025ragenunderstandingselfevolutionllm], which is implemented on top of veRL [^note: https://github.com/volcengine/verl] , and are run on a single node with $4$ NVIDIA H100 GPUs. We train for 200 rollout–update iterations on Sokoban, FrozenLake, and Sudoku 4$$4, and for 100 iterations on WebShop [yao2023webshopscalablerealworldweb] due to its long-context nature.

Each on-policy update uses grouped rollouts. On Sokoban, FrozenLake, and Sudoku, we use prompt-group size $P=8$ and sample $G=16$ trajectories per prompt; on WebShop, we use $P=16$ and $G=8$. This yields 128 trajectories per training step before filtering. Agents are allowed at most 2 actions per turn and 10 actions per episode on Sokoban and FrozenLake (5-turn horizon), an 8-turn horizon with a 16-step cap on Sudoku, and a 9-turn horizon with a single action per turn on WebShop. Following StarPO-S [wang2025ragenunderstandingselfevolutionllm], we apply reward-variance trajectory selection with filter ratio 0.25 on Sokoban and Sudoku. On WebShop and GRPO-based FrozenLake, we use a filter ratio of 1.0 (no filtering), since environmental stochasticity contaminates within-prompt reward variance as a selection signal. PPO uses a mini-batch size of $E=32$, with per-GPU micro-batch sizes scaled to memory (1–8, depending on model scale and environment), under FSDP and Ray for distributed training.

Policy optimization uses GAE with $(,)=(1.0,1.0)$, Adam with $(_1,_2)=(0.9,0.999)$, actor learning rate $110^{-6}$, and critic learning rate $110^{-5}$. We use asymmetric clipping $(_{}, _{})=(0.2,0.28)$ following yu2025dapoopensourcellmreinforcement [yu2025dapoopensourcellmreinforcement], together with entropy regularization of coefficient 0.001. For Sokoban, FrozenLake, and Sudoku, we drop the KL loss term during training and track KL post hoc, following yu2025dapoopensourcellmreinforcement [yu2025dapoopensourcellmreinforcement]. For WebShop under GRPO, we add a KL loss with coefficient 0.001 and the $k_1$ estimator to reduce late-stage collapse, following the vanilla StarPO setting in wang2025ragenunderstandingselfevolutionllm [wang2025ragenunderstandingselfevolutionllm]. We impose a format penalty of $-0.1$ when the agent fails to produce a valid `<think>`/`<answer>` structure. Our two token-weighting variants instantiate $w_t$ in Eq. §eq:token-weight: *Uniform* $(1.0,0)$ and *ActFocus* $(0.1,0.5)$. Rollouts are generated with vLLM using `tensor_parallel_size`=1 and `enforce_eager`, with a maximum model length of 3600 tokens and 400-token responses for Sokoban, FrozenLake, and Sudoku, and an 8192-token context window for WebShop.

**Evaluation..** 
We evaluate every 10 training steps and also before training begins (`val_before_train`). For Sokoban, FrozenLake, and Sudoku, we use a fixed evaluation set of 32 distinct prompts rolled out 16 times each under stochastic decoding, for a total of 512 trajectories per evaluation. For WebShop, we use 256 distinct prompts with one rollout each (256 trajectories). Decoding uses a temperature of 0.5 to measure robustness under sampling. Evaluation episodes are truncated using the same per-environment turn and action budgets as in training. We report evaluation success rate on this fixed evaluation set at step 200 for Sokoban, FrozenLake, and Sudoku, and at step 100 for WebShop.

### Signal Variants for the Energy Ablation

This appendix defines the four action-token signals compared in Sec. §sec:signal-sweep. 
The goal is to separate the weighting architecture from the choice of signal. 
All variants use the same weighting pipeline. They differ only in the raw score $s_t$ assigned to each action token.

**Common pipeline..** 
For a batch of action-token positions $_{}$, we first normalize the raw signal $s_t$ within the action tokens:

$$
_s = {|_{}|}
_{t _{}} s_t,
_s =
{|_{}|}
_{t _{}}
(s_t-_s)^2
+}.
$$

We then map the normalized score to $(0,1)$:

$$
_t =
\!({_s}).
$$

The final token weight is

$$
w_t =

, & t _{}, \\[2pt]
1 + \,_t, & t _{}.

$$

Thus, all four variants use the same $$, $$, normalization, and sigmoid mapping. 
Only the raw action-token signal $s_t$ changes.

**(1) Frozen-reference energy..** 
Energy is the signal used by *ActFocus*. 
It measures uncertainty from the frozen reference model before applying any policy update:

$$
s_t^{} = E_t
= -_{v V} \!(f^{}_{t,v}),
$$

where $f^{}_{t,v}$ is the frozen-reference logit for vocabulary token $v$. 
This matches the token-level energy definition in Eq. §eq:token-energy. 
Because the reference model is frozen, this score does not move as training progresses.

**(2) Frozen-reference entropy..** 
Entropy is another fixed-reference uncertainty signal. 
It measures how spread out the frozen reference distribution is:

$$
s_t^{}
=
H\!(_{}(c_t))
=
-_{v V} p^{}_{t,v}
p^{}_{t,v},
p^{}_{t,v}
=
}_{t,v})}
{_{u V}(f^{}_{t,u})}.
$$

Entropy uses the same frozen-reference forward pass as energy. The difference is that entropy is computed after softmax normalization, and therefore captures only the dispersion of the distribution, discarding the absolute logit scale.

**(3) Policy NLL..** 
Policy NLL measures how surprising the sampled token is under the rollout policy:

$$
s_t^{}
=
-_{_{}}(y_t c_t).
$$

This signal is cheap because PPO already stores rollout log-probabilities. 
However, it is policy-dependent: as the policy changes during training, the meaning and scale of this signal also change.

**(4) Log-probability shift..** 
Log-probability shift measures how much the rollout policy has moved away from the frozen reference at the sampled token:

$$
s_t^{}
=
_{_{}}(y_t c_t)
-
_{}(y_t c_t).
$$

A large positive value indicates that the rollout policy assigns a higher probability to $y_t$ than the frozen reference does. 
The log-probabilities are computed from logits without additional temperature scaling. 
Unlike energy and entropy, this signal partly moves with the policy because it contains $_{_{}}$.

**Signal stability..** 
Table §tab:signal-stability summarizes whether each raw signal changes as the rollout policy updates. The empirical ranking in Sec. §sec:signal-sweep matches this stability distinction. Energy and entropy are computed from a fixed model, so they provide stable weighting targets throughout training. Policy NLL and log-probability shift depend on the rollout policy, so their values change as the policy is updated. Among the two frozen-reference signals, energy performs better in our experiments. A useful interpretation is that energy keeps scale information from the reference logits, while entropy keeps only the normalized distribution shape. This makes frozen-reference energy a natural fit for *ActFocus*: it is stable during training and still preserves confidence information useful for prioritizing uncertain action tokens.

**Table:** *Stability of the four token-level signals during training. Frozen-reference signals provide a fixed weighting target, while policy-derived signals move as the policy changes.*

| lll
**Signal** | **Source** | **Drifts during training?** |
| — | — | — |
| Entropy | Frozen reference $_$ | No |
| Policy NLL | Rollout policy $__$ | Yes |
| Log-probability shift | Rollout policy and frozen reference | Partly |

## Additional Experimental Results

**Table:** *PPO and GRPO peak performance across four environments. Values are success/purchase rate (%); deltas are relative to Uniform. WebShop reports both purchase rate and strict success. Best results are **bolded**.*

_(table content)_

**Table:** *GRPO final-step results across four environments at the larger of the two model scales tested per environment. Success rate (%) is reported at step 200 for Sokoban, FrozenLake, and Sudoku, and at step 100 for WebShop. WebShop reports both purchase rate and strict success. Small-scale GRPO results are omitted because uniform credit largely fails to learn at this scale (e.g., Sudoku 0.5B reaches 0.0 in Table §tab:main-ppo-grpo-peak*

_(table content)_

## Environment Prompts and Example Rollouts

### Prompt Templates

Across environments, we use the same three-role chat template: a fixed
`system` message that describes the environment, a per-turn
`user` message containing the current state, and an
`assistant` message produced by the policy. The system message
is assembled from the per-environment instruction together with an
optional grid-symbol vocabulary and, when applicable, an enumeration
of admissible actions. The user message follows the same general
structure across environments:
`(Turn $n$):$$nState:$$n{state\`$$nYou have
{actions_left\ actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text. Strictly
follow this format. Max response length: {max_tokens} words (tokens).}

Below, we show, for each of the four environments used in our
experiments, the exact `system` prompt emitted by the training
pipeline.

#### Sokoban (`SimpleSokoban
)`

[
 colback=lightgray!10,
 colframe=black,
 title={**Sokoban System Prompt**},
 breakable
]

"`

You're a helpful assistant. You are solving the Sokoban puzzle.
You are the player and you need to push all boxes to targets.
When you are right next to a box, you can push it by moving in
the same direction.
You cannot push a box through a wall, and you cannot pull a box.
The answer should be a sequence of actions, like
<answer>Right || Right || Up</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player,
S: player on target
Your available actions are:
Up, Down, Left, Right
You can make up to 10 actions, separated by the action separator " || "

"`



#### Sudoku (`SimpleSudoku
, 4$$4)`

[
 colback=lightgray!10,
 colframe=black,
 title={**Sudoku System Prompt**},
 breakable
]

"`

You're a helpful assistant. You are solving a 4x4 Sudoku puzzle.
Fill in the grid so that every row, column, and 2x2 box contains
the numbers 1-4 without repetition.
Initial cells are shown in [brackets] and cannot be modified.
Empty cells are shown as dots (.).
Place numbers one at a time using the format:
<answer>place 3 at row 1 col 2</answer> or <answer>1,2,3</answer>
The environment will provide feedback on valid/invalid moves and
show conflicts if any occur.

"`



#### FrozenLake (`CoordFrozenLake
, 4$$4, slippery)`

[
 colback=lightgray!10,
 colframe=black,
 title={**FrozenLake System Prompt**},
 breakable
]

"`

You're a helpful assistant. You are solving the FrozenLake puzzle.
The observation includes both a symbol grid and zero-indexed
coordinates for the start, goal, player, and any holes.
Beware that the ice is slippery, so the agent might slide and end
up in an unintended tile.
Respond with a sequence of actions such as
<answer>Left || Up || Up</answer>.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole,
√: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to 10 actions, separated by the action separator " || "

"`



#### WebShop

The WebShop system prompt contains the task description only; the
admissible-action list is inserted into each turn's state string rather
than the system prompt itself.

[
 colback=lightgray!10,
 colframe=black,
 title={**WebShop System Prompt**},
 breakable
]

"`

You're a helpful assistant. You are browsing an online shop. Based
on the instruction, buy a product that close to the production
description. You need to search, read the search results, pick a
product, choose the size and color and buy. You should only choose
action from the available actions list provided later. Example
process: I need a gingko light and 20x20 pillow cover that is hand
painted. First search[gingko light 20x20 pillow cover hand painted],
answer format: <answer>search[blanket with fleece throw]</answer>.
Valid answer is search[<keywords>] or click[<clickable>].

"`



### Multi-turn State–Action Templates

Each example below is a verbatim excerpt from an evaluation-time
rollout. We preserve the exact `system / user / assistant`
role markers, state formatting, and model outputs.

#### Sokoban

[
 colback=lightgray!10,
 colframe=black,
 title={**Sokoban Rollout Example (Turns 1–2)**},
 breakable
]
**Input:**

"`

system
You're a helpful assistant. You are solving the Sokoban puzzle.
You are the player and you need to push all boxes to targets.
When you are right next to a box, you can push it by moving in
the same direction.
You cannot push a box through a wall, and you cannot pull a box.
The answer should be a sequence of actions, like
<answer>Right || Right || Up</answer>

The meaning of each symbol in the state is:
#: wall, _: empty, O: target, √: box on target, X: box, P: player,
S: player on target
Your available actions are:
Up, Down, Left, Right
You can make up to 10 actions, separated by the action separator " || "

user

Turn 1:
State:
######
###P_#
###X_#
###__#
###O_#
######
You have 10 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 100 words (tokens).

"`

**Output (Turn 1):**

"`

<think>The player is right next to a box. The box is not on a target
yet. The target is directly above the player.</think>
<answer>Right || Up</answer>

"`

**Next User message (Turn 2):**

"`

user
Reward:
-0.2

Turn 2:
State:
######
###_P#
###X_#
###__#
###O_#
######
You have 8 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 100 words (tokens).

"`

**Output (Turn 2):**

"`

<think>After pushing the box right, the player is now right next to
the target. The box is now on the target.</think>
<answer>Up</answer>

"`



#### Sudoku

[
 colback=lightgray!10,
 colframe=black,
 title={**Sudoku Rollout Example (Turn 1)**},
 breakable
]
**Input:**

"`

system
You're a helpful assistant. You are solving a 4x4 Sudoku puzzle.
Fill in the grid so that every row, column, and 2x2 box contains
the numbers 1-4 without repetition.
Initial cells are shown in [brackets] and cannot be modified.
Empty cells are shown as dots (.).
Place numbers one at a time using the format:
<answer>place 3 at row 1 col 2</answer> or <answer>1,2,3</answer>
The environment will provide feedback on valid/invalid moves and
show conflicts if any occur.

user

Turn 1:
State:
==================================================
SUDOKU PUZZLE
==================================================
[3][1]| [2][4]
[4] . | . .
————-
 . [3]| [4] .
 . [4]| [3][2]

Legend: [N]=initial cell, N=user-placed, *N*=conflict, .=empty

VALID NUMBERS FOR EMPTY CELLS:
 - (2,2): [2]
 - (2,3): [1]
 - (2,4): [1, 3]
 - (3,1): [1, 2]
 - (3,4): [1]
 - (4,1): [1]

Progress: 10/16 cells filled (10 initial, 0 placed)
Steps: 0/16
You have 10 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 150 words (tokens).

"`

**Output:**

"`

<think>Considering the initial placement, the only number that can go
in (2,2) is 2.</think>
<answer>2,2,2</answer>

"`



#### FrozenLake

[
 colback=lightgray!10,
 colframe=black,
 title={**FrozenLake Rollout Example (Turns 1–2)**},
 breakable
]
**Input:**

"`

system
You're a helpful assistant. You are solving the FrozenLake puzzle.
The observation includes both a symbol grid and zero-indexed
coordinates for the start, goal, player, and any holes.
Beware that the ice is slippery, so the agent might slide and end
up in an unintended tile.
Respond with a sequence of actions such as
<answer>Left || Up || Up</answer>.

The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole,
√: player on goal
Your available actions are:
Left, Down, Right, Up
You can make up to 10 actions, separated by the action separator " || "

user

Turn 1:
State:
Coordinates:
Board size: 4 rows x 4 cols (zero-indexed).
Start: (2, 3)
Goal: (3, 0)
Player: (2, 3)
Holes: (0, 3), (1, 1)
Grid Map:
___O
_O__
___P
G___
You have 10 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 120 words (tokens).

"`

**Output (Turn 1):**

"`

<think>The player is at position (2, 3) and needs to move towards
the goal at (3, 0). Moving left is the first logical step.</think>
<answer>Left || Left</answer>

"`

**Next User message (Turn 2):**

"`

user
Reward:
0

Turn 2:
State:
Coordinates:
Board size: 4 rows x 4 cols (zero-indexed).
Start: (2, 3)
Goal: (3, 0)
Player: (2, 1)
Holes: (0, 3), (1, 1)
Grid Map:
___O
_O__
_P__
G___
You have 8 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 120 words (tokens).

"`

**Output (Turn 2):**

"`

<think>After moving left, the player is now at (2, 1). Moving left
again would lead the player into a hole. The next best move is to
move up.</think>
<answer>Up</answer>

"`



#### WebShop

[
 colback=lightgray!10,
 colframe=black,
 title={**WebShop Rollout Example (Turn 1)**},
 breakable
]
**Input:**

"`

system
You're a helpful assistant. You are browsing an online shop. Based
on the instruction, buy a product that close to the production
description. You need to search, read the search results, pick a
product, choose the size and color and buy. You should only choose
action from the available actions list provided later. Example
process: I need a gingko light and 20x20 pillow cover that is hand
painted. First search[gingko light 20x20 pillow cover hand painted],
answer format: <answer>search[blanket with fleece throw]</answer>.
Valid answer is search[<keywords>] or click[<clickable>].

user

Turn 1:
State:
Instruction: Find me machine wash, wash cold women's fashion hoodies
& sweatshirts for dry clean, tumble dry with color: vintage camo,
and size: large, and price lower than 70.00 dollars.We must buy a
product within 10 actions. It doesn't have to match perfectly with
description.
Search term should not include details like size, color.
Never search for more than 2 times.
Do not be too strict about the description, it's more important to
buy one that is close enough within action limit.
Prioritize click a product in the current page over going to next
page.
Almost never click[next >] for more than 2 times.Almost never
click[< prev] unless you are sure the product is on one of the
previous pages.
If you have less than 3 actions left, just buy the first product you
see in the current page.
If an matching option exists, make sure to click[size] then
click[color], one at a time, before click[buy now], but don't have
to if only 1 action left, in that case you just click[buy now].
Never click description.
You must choose from these actions:search[<content>].
You have 9 actions left. Always output: <think> [Your thoughts]
</think> <answer> [your answer] </answer> with no extra text.
Strictly follow this format. Max response length: 200 words (tokens).

"`

**Output:**

"`

<think>Based on the instruction, I need to find a machine washable,
wash cold women's hooded sweatshirt that is vintage camo pattern and
in large size, with a price lower than 70.00 dollars. The search
term should not include size or color.</think>
<answer>search[women hooded sweatshirt vintage camo large machine
wash cold]</answer>

"`