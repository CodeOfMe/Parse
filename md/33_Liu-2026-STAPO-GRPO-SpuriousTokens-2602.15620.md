# STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens

> **arXiv:** [2602.15620](https://arxiv.org/abs/2602.15620)
> **TeX source:** [arXiv-2602.15620v1/](arXiv-2602.15620v1/)
> **PDF:** [STAPO-arXiv-2602.15620v1.pdf](STAPO-arXiv-2602.15620v1.pdf)

---

% — BEGIN sections/010intro —

**Figure:** ***Core Idea.**
(a) In RLVR training, we argue that spurious tokens, which are rare and uninformative tokens within otherwise correct responses, can harm training stability, analogous to a dissonant vocalist disrupting the harmony of a performance.
(b) By masking this negligible fraction (near $ 0.01%$) of spurious tokens during the RL process of Qwen3-8B-Base, STAPO approaches the Pareto frontier of AIME24 Acc and entropy stability, compared to GRPO, 20-Entropy, and JustRL.* () _(image: figures/intro2.pdf)_

## Introduction

Recent large language models (LLMs), including OpenAI-o1 [gpt4], DeepSeek-R1 [guo2025deepseek], and Qwen3 [yang2025qwen3], have demonstrated remarkable capabilities in complex reasoning domains such as mathematics and programming. Central to this success is Reinforcement Learning (RL) [li2023reinforcement], which optimizes directly for outcome-level correctness and has been empirically linked to the emergence of advanced, long-horizon reasoning behaviors [cao2024survey]. 

Existing approaches have largely focused on enhancing exploration to improve the learning capacity of LLMs under RL training. For example, DAPO [yu2025dapo] increases the clipping threshold, while SAPO [gao2025soft] adopts a soft-clipping strategy. Other works instead emphasize the exploration of forking tokens, as introduced in 20-Entropy [wang2025beyond]. However, in practice, these approaches often drive entropy toward explosive growth, ultimately leading to catastrophic degradation, where models collapse from coherent reasoning into shallow, repetitive, or even nonsensical outputs [zhang2025survey].

To mitigate such entropy-induced instability, prior efforts have introduced various stabilization techniques, including entropy regularization [cui2025entropy,yang2025entropic], sample augmentation [simoni2025gtpo,qiu2025noisygrpo], and advantage reweighting [yang2025not]. These methods regulate entropy at a global level but remain coarse-grained and fail to capture the heterogeneous roles of individual tokens during optimization. As a result, they may either over-suppress useful exploration or induce oscillatory entropy dynamics, leading to suboptimal performance.

In this work, we aim to strike a balance between effective exploration and strong reasoning capabilities. We first introduce the concept of *spurious tokens*, defined as a sparse subset of tokens within otherwise accurate responses that, rather than contributing to the underlying reasoning process, introduce logical misdirection or harmful noise, as illustrated in Figure §fig:intro. From a token-level perspective, we systematically analyze the detrimental impact of these spurious tokens on optimization dynamics, demonstrating that they introduce misleading and destabilizing update signals. Building on this insight, we propose an efficient method for identifying spurious tokens and formulate Spurious-Token-Aware Policy Optimization (STAPO). By masking a negligible fraction (approximately $0.01%$) of such tokens during training, STAPO significantly stabilizes policy entropy and consistently yields performance improvements. Overall, our main contributions are summarized as follows:

- We identify spurious tokens as a key source of training instability: a minor fraction of tokens (around $0.01%$) that contribute little to the actual reasoning yet receive disproportionately large gradient updates by inheriting the full sequence-level reward.
To analyze this issue, we present a unified framework that systematically evaluates token-level optimization dynamics in terms of spurious risk, gradient norms, and entropy changes.
- By analyzing token characteristics that severely disrupt optimization, we propose the Silencing Spurious Tokens (S2T) mechanism to efficiently identify such disruptive tokens and suppress their gradient perturbations. We further integrate S2T into a group-based objective and develop STAPO for stable and effective large-scale model refinement.
- Across six mathematical reasoning benchmarks (AIME24, AIME25, AMC23, MATH500, Minerva, and OlympiadBench) and three model scales (Qwen 1.7B, 8B, and 14B base models [yang2025qwen3]), STAPO substantially stabilizes policy entropy and consistently improves reasoning accuracy under diverse evaluation settings.

% — END sections/010intro —

% — BEGIN sections/020preliminary —

## Preliminaries

### Problem Formulation

We consider the problem of fine-tuning large language models (LLMs) via reinforcement learning (RL) for reasoning tasks. Let $$ denote a distribution over input prompts. Given a prompt $ $, an LLM parameterized by $$ acts as a stochastic policy $_$ that autoregressively generates an output sequence $ = (y_1, ..., y_T)$. Specifically, at each step $t$, a token $y_t $ is sampled according to $_(y_t , _{<t})$.

Supervision is provided via a sparse, sequence-level verifiable reward $R(, ) $, which evaluates the correctness of the generated sequence $$ using an external verifier (e.g., a code compiler or a mathematical rule checker). The optimization objective is to maximize the expected reward:

$$
() = _{ ,\,  _()} [ R(, ) ].
$$

### Group Relative Policy Optimization (GRPO)

We briefly review Group Relative Policy Optimization (GRPO) [shao2024deepseekmath], which estimates advantages without relying on an explicit value function. For each prompt $$, GRPO samples a group of $G$ output sequences ${_1, ..., _G}$ from a reference behavior policy $_{_{}}$. The optimization objective is defined as the average clipped surrogate loss over the sampled group:

$$

_{}() &= _{, {_i}_{i=1}^G_{_{}}()} [ {G} _{i=1}^{G} {|_i|} _{t=1}^{|_i|} ( _{i,t}() _i, \\
&(_{i,t}(), 1-, 1+)_i ) ] -_(__),

$$

$$

 _{i,t}() &=  , _{i,<t})}{_{_{}}(y_{i,t} , _{i,<t})}, _i = , _i) - ({R(, _j)}_{j=1}^G)}{({R(, _j)}_{j=1}^G)},

$$

where $_{i,t}()$ is the importance sampling ratio and $_i$ is the sequence-level advantage signal, derived by standardizing the reward across the $G$ samples within the group.
The $_$ serves as the reference policy, and $$ is the scaling coefficient.

### Clip-Higher and Token Normalization

Building upon the GRPO framework, several large-scale RL algorithms have recently emerged. Notably, DAPO [yu2025dapo] removes the KL penalty and introduces a set of training enhancements, with token-level normalization and an asymmetric clip-higher mechanism proving particularly effective for stabilizing optimization. Subsequent work, such as JustRL [he2025justrl], has adopted these two components and demonstrated strong empirical performance.
The corresponding objective is written as:

$$

_{}() &= _{, {_i}_{i=1}^G_{_{}}()} [ {_{i=1}^G |_i|} _{i=1}^{G} _{t=1}^{|_i|} ( _{i,t}() _{i}, \\
&(_{i,t}(), 1-_, 1+_) _{i} ) ],

$$

where $_$ and $_$ denote the asymmetric clipping parameters integral to the clip-higher mechanism. Given its simplicity and robust empirical success, we adopt this augmented configuration as our baseline objective.
% — END sections/020preliminary —

% — BEGIN sections/030method —

## Methodology

### The Hidden Threat: Spurious Tokens

As discussed in Section §sec:problem_for, , rewards in Reinforcement Learning with Value-Regularized (RLVR) are typically derived solely from the final outcome. Consequently, all tokens $y_{i,t}$ within a given trajectory share an identical sequence-level advantage, $_i$. This coarse-grained credit assignment can inadvertently reinforce extraneous tokens. We formalize this phenomenon as follows:

**Definition:** [Spurious Tokens]
Spurious tokens are intermediate tokens $y_{i,t}$ that contribute negligibly to the correct reasoning outcome, yet receive disproportionately large positive updates.

To empirically examine this effect, we train a Qwen3-1.7B base model under the JustRL setting on DAPO-MATH-17K [yu2025dapo], and record all generated tokens along with their associated statistics during training. Figure §fig:spurious_examples presents concrete instances of this phenomenon, showing that spurious tokens can induce misleading update signals that steer the policy toward detrimental directions and destabilize training.

Motivated by this observation, we propose Spurious-Token-Aware Policy Optimization (STAPO), which introduces a binary mask, $^{}_{i,t}$, to discard gradient contributions from spurious tokens:

$$
^{}_{i,t} = 
 
0, &  y_{i,t} _i, \\
1, & ,

$$

where $_i$ denotes the set of identified spurious tokens within the $i$-trajectory. Incorporating this, the STAPO objective is formulated as:

$$

_{}() &= _{, {_i}_{i=1}^G_{_{}}()} [ {_{i=1}^{G} _{t=1}^{|_i|} ^{}_{i,t}} _{i=1}^{G} _{t=1}^{|_i|} ^{}_{i,t} ( _{i,t}() _{i}, \\
&(_{i,t}(), 1-_, 1+_) _{i} ) ].

$$

The terms $_{i,t}$ and $_i$ follow standard definitions provided in (§eq:ratio_adv_def). Comparing the STAPO objective in Eq. (§eq:stapo_loss) with the standard DAPO objective in Eq. (§eq:dapo_loss), two primary distinctions emerge. First, STAPO leverages this binary mask to selectively zero out the loss calculations for spurious tokens. Second, the normalization term in Eq. (§eq:stapo_loss) is dynamically adjusted to average the loss exclusively over the remaining valid tokens.

**Figure:** *{Illustrative examples of spurious tokens.* () _(image: figures/example_tokens.pdf)_

### Token-Level Optimization Analysis

Fundamentally, the generation dynamics of LLMs are shaped by pre-training, which assigns lower prior probabilities to ill-formed or semantically anomalous token sequences. 
As a result, spurious tokens, often appearing as incoherent or illogical reasoning steps, tend to exhibit *low sampling probabilities*. Motivated by this, we extend JustRL with full masking of low-probability tokens within correct answers (JustRL-FullMask) and compare its entropy dynamics with the default one. As shown in Figure §fig:entropy_aime24, JustRL-FullMask leads to severe entropy collapse, resulting in insufficient exploration and degraded performance, while the default JustRL update maintains persistently high policy entropy in later stages, hindering convergence. This trade-off exposes a paradox in entropy control, explained by the following update dynamics:

**Lemma:** [Entropy Update Mechanism {[xi2025bapo]}]

Consider the language policy $_()$ updated via a natural policy gradient step with learning rate $$. Let $y'$ denote a generic token variable distributed according to the current policy $_(, _{i,<t})$ at step $t$. The change in entropy $(y_{i,t})$ associated with a specific, already-sampled token $y_{i,t}$ between two consecutive policy iterates satisfies:

$$

 (y_{i,t}) &-_(y_{i,t} , _{i,<t}) [ _(y_{i,t} , _{i,<t}) \\
 &- _{y' _} [ _(y' , _{i,<t}) ] ] [ _i(y_{i,t}) - _{y' _} [ _i(y') ] ].

$$

{r}{0.46}
 
 
 [width=]{figures/dual_axis_entropy_aime24.pdf}
 

 
 


As formalized in Lemma §lemma:entropy_pg_general, low-probability tokens within correct responses induce positive entropy updates, thereby sustaining policy exploration. 
This trade-off implies that the key challenge is to selectively remove truly detrimental spurious tokens among low-probability tokens while preserving stable exploration.

To operationalize this distinction, we partition the representation space into four quadrants based on token statistics (Figure §fig:quadrants). Spurious tokens predominantly occupy the low-probability regime, demonstrating a clear concentration within low-entropy states. We attribute this to the underlying generation dynamics: low-probability tokens in high-entropy states generally signify legitimate exploration and are therefore structurally reasonable. Conversely, low-entropy (high-confidence) states inherently possess highly probable, valid candidates; the realization of a low-probability token in such contexts is merely an artifact of random sampling, rendering it anomalous and significantly elevating the risk of spurious generation.

Furthermore, we analyze the gradient norm at the token level during training. Specifically, at decoding step $t$ of sequence $_i$, the LLM produces a logit vector $_{i,t} ^{||}$ over the vocabulary $$, inducing the policy distribution $_(, _{i,<t})$ via the softmax function. We analyze the per-token gradient associated with $y_{i,t}$ as it propagates to intermediate layers.

**Theorem:** [Policy Gradient Norm Bounds]
Consider the optimization objective at step $t$ for sample $i$ with target token $y_{i,t}$. 
The squared $_2$-norm of the gradient $_{_{i,t}} $ w.r.t. the logits $_{i,t} ^{||}$ is bounded by the entropy $(_{})$ and the target probability $_(y_{i,t} , _{i,<t})$ as follows:

$$

 |w_{i,t}|^2 ( 1 - 2_(y_{i,t} , _{i,<t}) + e^{-(_{})} ) 
 &\|_{_{i,t}} \|^2 \\
 &|w_{i,t}|^2 ( 2 - 2_(y_{i,t} , _{i,<t}) - C_V (_{})^2 ),

$$

where $C_V = | - 1}{|| (||)^2}$ and the scaling weight $w_{i,t}$ is defined as:

$$
w_{i,t} = 
 
0, &  (_i > 0 _{i,t} > 1 + _) (_i < 0 _{i,t} < 1 - _), \\
 , _{i,<t})}
{_{_{}}(y_{i,t} , _{i,<t})} _i, & .

$$

> **Proof:** 
See Appendix §app:grad_bounds_proof for the detailed derivation.

Theorem §prop:grad_bounds demonstrates that the tokens characterized by both low probability and low entropy induce a larger gradient norm, a phenomenon empirically corroborated in Figure §fig:mean_grad. Such tokens are likely spurious, thereby introducing erroneous updates that can destabilize training.

To systematically analyze how different token types influence optimization, we focus on tokens within correct answers, which inherently possess a positive advantage ($ > 0$), and categorize them along two binary axes: token probability (high/low) and entropy (high/low). As summarized in Table §tab:token_analysis, tokens characterized by *low probability and low entropy* consistently exhibit detrimental optimization effects across all evaluation criteria. First, such tokens are highly likely to be spurious, leading to incorrect update directions. Second, they are associated with disproportionately large gradient norms, thereby amplifying their adverse impact during optimization. Finally, they contribute to entropy explosion in standard training, further exacerbating training instability.

**Table:** ***Taxonomy of Token-Level Optimization Mechanisms.** 
Left: Token properties. 
Right: Influence on optimization. 
^{}_{i,t} = 
 
0, & \ _i > 0 (y_{i,t}) < _p _t < _h^{(q)}, \\
1, & ,

$$

where $_p$ is a fixed probability threshold that defines an absolute notion of rarity, and $_h^{(q)}$ denotes the $q$-th quantile of entropy, computed dynamically over the low-probability tokens within correct responses in each mini-batch. 

{r}{0.40}
 
 
 [width=]{figures/exploration.pdf}
 

 
 


The core idea is to first identify low-probability tokens within correct responses, which are characterized as exploratory tokens according to Lemma §lemma:entropy_pg_general. Among these tokens, we further partition them based on entropy using the quantile threshold $_h^{(q)}$. Tokens in the low-entropy region (i.e., the bottom $q$ fraction), which correspond to the most destructive cases identified in Table §tab:token_analysis, are suppressed. In contrast, the remaining high-entropy tokens (the top $1-q$ fraction) are preserved to maintain sufficient exploration, as illustrated in Figure §fig:exploration. This adaptive Silencing mechanism achieves a principled balance between stabilizing optimization and preserving exploration capacity.To provide further insight, we include word cloud comparisons for both types of tokens in Appendix §app:word_cloud.

In practice, we approximate the ideal mask $^{}$ using the S2T mechanism, i.e., $^{}_{i,t} ^{}_{i,t}$ in Equation (§eq:stapo_loss), resulting in a computationally efficient instantiation of STAPO. The complete training procedure is outlined in Appendix §app:algorithm.

 

 

% — END sections/030method —

% — BEGIN sections/060related —

## Related Work

**Reinforcement Learning for LLMs.** 
In the era of small-model training, various effective RL algorithms have been proposed, including TD3 [fujimoto2018addressing], TRPO [schulman2015trust], DSAC [10858686], and BOOM [zhanbootstrap].
Recently,
Reinforcement Learning from human feedback (RLHF) has emerged as a predominant approach for aligning LLMs with human preferences and diverse downstream objectives [jaech2024openai, guo2025deepseek]. While early approaches relied on on-policy optimization such as PPO [schulman2017proximal], recent work has shifted toward more efficient preference-based Direct Preference Optimization (DPO) [rafailov2023direct], which avoids explicit reward modeling and online rollouts. The focus has further expanded from general alignment to improving reasoning capabilities, motivating RL and training schemes such as GRPO [shao2024deepseekmath]. In this context, a range of policy optimization variants, including DAPO [yu2025dapo], GSPO [zheng2025groupsequencepolicyoptimization], SAPO [gao2025soft], and other related methods [yue2025vapo, wang2025aspo, qiu2025noisygrpo], have been developed to enhance optimization stability, sample efficiency, and scalability for reasoning-oriented language models.

**Entropy Instability in RL.**
Entropy is a central issue in RL, as it directly affects exploration and training stability [duan2021distributional, wang2024diffusion]. In reasoning-oriented language models trained with RL, a persistent challenge is the rapid collapse of policy entropy during the early stages of optimization, which often leads to premature convergence and degraded performance. Prior work mitigates this issue through interventions such as selectively regularizing high-entropy tokens [wang2025beyond], increasing the proportion of entropy-enhancing samples [xi2025bapo], and modifying clipping strategies in policy optimization [yu2025dapo, chen2025minimax, gao2025soft]. However, these methods often introduce the opposite failure mode, where entropy grows excessively or becomes unstable, degrading reasoning coherence and leading to repetitive or unstructured outputs. Although some studies analyze entropy dynamics during RL training [cui2025entropy, wang2026entropy] and others explicitly enforce entropy stability as an optimization objective [yang2025entropic], existing approaches largely treat entropy as a surface-level training signal rather than addressing the underlying sources of instability.

**Gradient Domination by Low-Probability Tokens.**
Gradients are a key factor in the stability of RL, and various forms of policy gradients have been proposed,
such as the bicriteria policy gradient [zhan2025bicriteria] and continuous-time policy gradient [zhan2023continuous].
A key microscopic source of instability stems from the disproportionate gradient influence of low-probability tokens. As proved by Yang et al. [yang2025not], rare tokens can generate excessively large updates, allowing a small subset of unstable predictions to dominate optimization, which harms fine-tuning stability. Recent work addresses this through probability-aware modulation. 
To avoid suppressing informative exploratory signals, Low-Probability Regularization (Lp-Reg) [huang2025low] filters noise while preserving meaningful rare tokens. However, these approaches largely rely on scalar probability thresholds, lacking a joint, fine-grained treatment of token-level confidence and probability, and thus failing to distinguish useful exploration from aleatoric noise under local model calibration.

% — END sections/060related —

% — BEGIN sections/040experiments —

## Experiments

### Settings

 **Baselines.** We compare our approach against several RL algorithms for LLMs, including GRPO [shao2024deepseekmath], 20-Entropy [wang2025beyond], and JustRL [he2025justrl]. For all baselines, we follow the parameter settings reported in their original papers. Unless otherwise specified, STAPO uses $_p = 0.002$ and $q = 75%$ across all experiments. For a fair comparison, we do not apply the dynamic sampling technique introduced in [yu2025dapo] to any method. Detailed training settings are provided in Appendix §app:traing_details.

 **Benchmarks.** We conduct experiments across three model scales: Qwen 1.7B, 8B, and 14B base models and evaluate on six widely adopted and challenging mathematical reasoning benchmarks: AIME24 [li2024numinamath], AIME25 [opencompass2025aime], AMC23 [li2024numinamath], MATH500 [hendrycks2021math], Minerva [lewkowycz2022solving], and OlympiadBench [he2024olympiadbench]. We generate $N$ independent responses per problem ($N=4$ for MATH500, Minerva, and OlympiadBench; $N=32$ for others) across two decoding configurations: temperature $_{}$=1.0, top-p=1.0 and temperature $_{}$=0.7, top-p=0.9 , with a maximum length of 20,480 tokens. In the absence of additional instructions, the default evaluation configuration is set to $_{}$=0.7, top-p=0.9. Results are reported as the average accuracy. To ensure evaluation rigor, we employ CompassVerifier-3B [liu2025compassverifier], a lightweight LLM verifier, to rectify misclassifications from the rule-based verification.

### Main Results

#### Training Behavior Analysis

**Entropy Analysis.**
Entropy curves are a key indicator of learning progress in LLM training. As shown in Figure [fig:main_results_1_7b]{*{fig:main_results_1_7b}(b)}, JustRL and 20-Entropy suffer from entropy explosion, while GRPO exhibits entropy collapse. Conversely, STAPO maintains a stable, well-regulated entropy profile after warmup.
Token-level quantile analysis (Figure [fig:main_results_1_7b]{*{fig:main_results_1_7b}(d)}) explains this stability: STAPO concentrates entropy exclusively at high quantiles (e.g., above the 80th percentile), keeping most tokens deterministic. This strategically preserves exploration for critical tokens while enhancing reasoning precision. In contrast, the uniformly high entropy in JustRL and 20-Entropy triggers repetitive generation, whereas GRPO's persistently low entropy stifles exploration and limits learning capacity

**Performance Comparison.** Beyond maintaining a stable entropy profile and preserving critical exploratory tokens, STAPO demonstrates superior overall performance. During RL training, as shown in Figure [fig:main_results_1_7b]{*{fig:main_results_1_7b}(c)}, STAPO achieves the highest reward, maintaining a clear advantage over all baselines. This suggests that STAPO successfully explores and acquires reasoning capabilities that other methods fail to capture. Furthermore, Figure [fig:main_results_1_7b]{*{fig:main_results_1_7b}(a)} illustrates that STAPO not only attains the highest accuracy on AIME24 but also sustains strong performance without noticeable degradation even after 3000 steps, evidencing remarkable training stability. 
Ultimately, the stark contrast with JustRL underscores the detrimental impact of spurious tokens in baseline methods, corroborating the validity of our prior analysis in Section §subsec:eg_coupling_analysis.

**Table:** ***Main Results on Six Benchmarks across Three Models.** Each cell reports performance under the training-aligned configuration (temperature $_{}$=1.0, top-p=1.0), STAPO demonstrates excellent scalability and enhanced intrinsic reasoning capability, surpassing the strongest baselines with average relative accuracy improvements of 18.49%, 5.78%, and 10.20% at the 1.7B, 8B, and 14B scales, respectively.

{r}{0.42}
 
 
 [width=]{figures/silencing_tokens_10e-4_compact_log.pdf}
 

 
 


When evaluated under the JustRL configuration ($_{}$=0.7, top-p=0.9), STAPO continues to achieve state-of-the-art performance. The performance gap narrows slightly because high-entropy baselines (JustRL and 20-Entropy) benefit more from decoding heuristics that suppress tail generations. Crucially, STAPO consistently attains optimal results across varied evaluation settings, demonstrating strong robustness and overall performance. This confirms that its inherently stable distribution is less dependent on such decoding heuristics.

Notably, as shown in Figure §fig:spurious_ratio_independent, these significant gains are achieved by masking a negligible fraction of tokens (mostly $<0.01%$) across the 1.7B, 8B, and 14B models. This indicates that RL instability is caused by sparse spurious tokens, which disproportionately affect gradient updates and are precisely identified and effectively suppressed by STAPO.
To better illustrate the impact of spurious tokens, we also provide additional examples, along with a simple case classification, in Appendix §app:spurious_examples.

### Ablation and Sensitivity Analysis

**Masking Strategy Ablation.** We first ablate different masking strategies applied to positive-advantage, low-probability tokens on the Qwen3-1.7B-Base model. As shown in Figure §fig:ablation_independent, low-entropy masking (STAPO) consistently achieves the best performance, whereas high-entropy masking significantly degrades results, suggesting that masking high-entropy tokens can suppress useful exploratory signals. A qualitative comparison of word cloud (see Appendix §app:word_cloud) further highlights the difference between exploratory and destructive tokens. 
Random masking yields intermediate performance, gradually approaching that of low-entropy masking as the masking ratio increases. This indicates that while low-entropy masking remains the optimal strategy, random masking can function as a heuristic approximation when the masking budget is sufficiently large.

**Hyperparameter Sensitivity.** We further analyze the model's sensitivity to two key hyperparameters: the entropy quantile $q$ and the probability threshold $_p$. As shown in Figure §fig:ablation_independent, decreasing $q$ leads to a rapid degradation in performance for both high-entropy and random masking strategies. In contrast, STAPO exhibits only a marginal decline, demonstrating superior robustness to variations in the entropy quantile.
Figure §fig:tau_p illustrates the effect of varying $_p$, which establishes the threshold for identifying low-probability tokens. An insufficiently low $_p$ may fail to isolate spurious tokens, leading to increased training entropy and subsequent performance degradation. Conversely, an excessively high $_p$ may erroneously penalize legitimate tokens, rendering the policy overly conservative and impairing its reasoning capabilities. Despite these effects, overall performance remains relatively stable, indicating that the model is largely insensitive to $_p$.

**Figure:** ***Masking Strategy Ablation.** Various entropy masking strategies are applied to positive-advantage, low-probability tokens on the 1.7B base model with $_{



% — BEGIN sections/appendix —

## Gradient Norm Decomposition

We first establish an exact decomposition of the gradient norm with respect to the logits, which serves as the foundation for our bounds.

**Lemma:** [Gradient Norm Decomposition [yang2025not]]

Let $_{i,t} R^{||}$ denote the logits and $_(v^n , _{i,<t}) = }{_m e^{a^m}}$ be the induced softmax distribution. Let $y_{i,t}$ be the target token with index $k$ (i.e., $v^k = y_{i,t}$).
The squared $_2$-norm of the gradient of the group-style objective $$ with respect to $_{i,t}$ satisfies:

$$

\|_{_{i,t}} (y_{i,t})\|^2
=
|w_{i,t}|^2
(
1 - 2_(y_{i,t} , _{i,<t}) + _{n=1}^{||}_(v^n)^2
),
$$

where the weight $w_{i,t}$ is defined as:

$$
w_{i,t} = 
 
0, &  (_i > 0 _{i,t} > 1 + ) (_i < 0 _{i,t} < 1 - ), \\
 , _{i,<t})}
{_{_{}}(y_{i,t} , _{i,<t})} _i, & .

$$

> **Proof:** 
Recall that the gradient of the log-likelihood $_(y_{i,t})$ with respect to the logit $a^n$ is given by $_{kn} - _(v^n)$, where $_{kn}$ is the Kronecker delta. The gradient of the weighted objective $$ is therefore:

$$
(y_{i,t})}{a^n} 
= w_{i,t} (_{kn} - _(v^n))
= 
 
w_{i,t} (1 - _(v^k)), &  n = k, \\
-w_{i,t} _(v^n), &  n k.

$$

Computing the squared $_2$-norm by summing over all $n |}$:

$$

\|_{_{i,t}} (y_{i,t})\|^2 
&= ( w_{i,t} (1 - _(v^k)) )^2 + _{n k} ( -w_{i,t} _(v^n) )^2 \\
&= |w_{i,t}|^2 ( 1 - 2_(v^k) + {_(v^k)^2 + _{n k} _(v^n)^2}),\\
&= |w_{i,t}|^2 ( 1 - 2_(v^k) + _{n=1}^{||} _(v^n)^2).

$$

Identifying $v^k$ as $y_{i,t}$ completes the proof.

## Proof of Theorem §prop:grad_bounds

> **Proof:** 
We now complete the proof of Theorem §prop:grad_bounds.
By Lemma §lem:grad_norm_decomp, the squared gradient norm can be written as

$$

\|_{_{i,t}} (y_{i,t})\|^2
=
|w_{i,t}|^2
(
1 - 2_(y_{i,t})
+
_{n=1}^{||}_(v^n)^2
).
$$

To simplify the notation throughout this proof, we denote the components of the distribution $__{||-1}$ as $_n _(v^n)$ for $n |}$. 
The term $_{n=1}^{||}_n^2$ represents the *collision probability* of the distribution, which measures its concentration. We universally denote the Shannon entropy as $(_) = - _{n=1}^{||} _n _n$.

**Lower bound..** 

To establish a rigorous lower bound, we relate the collision probability to the Shannon entropy. Recall that the Rényi entropy of order 2 is defined as $_2(_) = - ( _{n=1}^{||}_n^2 )$.
Expressing the argument of the logarithm as an expectation, we have $_{n=1}^{||}_n^2 = _{v _}[_(v)]$. Because the function $f(x) = -(x)$ is strictly convex, applying Jensen's inequality yields:

$$
_2(_)
=
- ( _{v _}[_(v)] )
_{v _}[-_(v)]
=
H(_).
$$

Consequently, the collision probability satisfies $_{n=1}^{||}_n^2 = e^{-_2(_)} e^{-H(_)}$. Substituting this inequality into Eq. (§eq:grad_norm_recall) yields the entropy-based lower bound on the gradient norm:

$$
\|_{_{i,t}} (y_{i,t})\|^2
|w_{i,t}|^2
(
1 - 2_(y_{i,t})
+
e^{-H(_)}
).
$$

**Upper bound..** 

We begin by formulating the objective via equivalent transformations. We introduce the Gini impurity $L = 1 - _{n=1}^{||} _n^2$. The target inequality we aim to prove is:

$$
_{n=1}^{||} _n^2 1 - C_V (_)^2,
$$

where $C_V = |-1}{||(||)^2}$. Rearranging the terms and applying the definition of $L$ yields the equivalent condition $C_V (_)^2 L$. At the vertices of the probability simplex (where a single component $_n=1$ and all others vanish), $(_)=0$ and $L=0$, rendering the inequality trivially satisfied ($0 0$). For any non-vertex distribution, $L > 0$, allowing us to reformulate the problem as bounding the functional $F(_) = (_)^2}{L} {C_V}$. Thus, it suffices to prove that the global maximum of $F(_)$ on the simplex $_{||-1}$ satisfies $F(_) |(||)^2}{||-1}$.

To determine the interior constraints, let $__{||-1}$ denote a global maximizer of $F(_)$. Excluding the trivial vertices ensures $(_) > 0$ and $F(_) > 0$. By the Karush-Kuhn-Tucker (KKT) conditions [boyd2004convex_ch5], this extremum must reside in the relative interior of a sub-simplex characterized by $k$ non-zero components ($1 < k ||$). We apply the method of Lagrange multipliers to these non-zero components subject to the probability sum equality constraint. Because the logarithmic transformation is strictly monotonically increasing, it preserves the locations of extrema, allowing us to equivalently maximize $= 2(_) - L - (_n - 1)$.
For any $_i > 0$, the first-order stationarity condition is:

$$
{_i}[2(_) - L] = {(_)}(-1-_i) - {L}(-2_i) = .
$$

Rearranging this yields ${L} - {(_)} = {2} + {(_)}$. Defining the auxiliary function $g(t) = {L} - {(_)}$ for $t (0,1)$, its second derivative is $g"(t) = {(_) t^2}$. Since $(_) > 0$, $g"(t) > 0$ universally on this domain, confirming that $g(t)$ is strictly convex. Consequently, the equation $g(_i) = C$ can possess at most two distinct roots, implying the non-zero components of the extremum can assume at most two distinct values.

We now prove that no two-value stationary points can exist. Assume, toward a contradiction, that the non-zero components take exactly two distinct values, $x$ and $y$ (with $x, y (0,1)$ and $x y$), occurring with multiplicities $k'$ and $m'$, respectively. The constraints dictate $k'x + m'y = 1$ and $k'x^2 + m'y^2 = 1 - L$. Because both $x$ and $y$ satisfy the stationarity condition, $g(x) = g(y)$, which implies:

$$
{L} - {(_)} = {L} - {(_)} (_) = L {y - x}.
$$

Introducing the logarithmic difference quotient $= {y - x}$, we obtain $(_) = L$, which implies $y = x + (y-x)$. Substituting this into the entropy expression yields $(_) = -k'xx - m'y(x + (y-x))$. Grouping terms and applying $k'x + m'y = 1$ gives $(_) = -x - m'y(y-x)$. Equating this with $L$ results in $-x = [L + m'y(y-x)]$. Substituting $L = 1 - k'x^2 - m'y^2$ and simplifying the bracketed term reduces it to $1 - x(k'x + m'y) = 1 - x$. Consequently, $-x = (1-x)$, yielding $= {1-x}$. By symmetry, we analogously obtain $= {1-y}$. Therefore, any two-value stationary point must satisfy:

$$
{1-x} = {1-y}.
$$

Let $u(t) = {1-t}$ for $t (0,1)$. Its derivative is $u'(t) = {(1-t)^2}$. Let $v(t) = 1 - {t} - t$ denote the numerator. Since $v'(t) = {t^2} > 0$ for all $t (0,1)$, $v(t)$ is strictly monotonically increasing. Because $v(1) = 0$, it follows that $v(t) < 0$ on $(0,1)$. This establishes that $u'(t) < 0$ globally on this interval, implying $u(t)$ is strictly monotonically decreasing. Thus, $u(x) = u(y)$ holds if and only if $x = y$, directly contradicting $x y$.

The preceding contradiction establishes that the stationary point must be a uniform distribution over a support of size $k$ ($1 < k ||$), where each of the $k$ non-zero components equals ${k}$. At this point, the functional evaluates to $h(k) = {k-1}$. Treating $k$ as a continuous variable on $(1, ||]$ and differentiating yields:

$$
h'(k) = {(k-1)^2}.
$$

For $k > 1$, the standard inequality $k < k - 1$ guarantees that $2(k-1) - k > k - 1 > 0$. Because $k > 0$, $h'(k) > 0$. This demonstrates that the objective function strictly monotonically increases with the support size $k$. Thus, the global maximum is uniquely attained at the full support $k=||$:

$$
_{__{||-1}} F(_) = h(||) = |(||)^2}{||-1}.
$$

Given this global maximum, the inequality $(_)^2}{1 - _{n=1}^{||} _n^2} |(||)^2}{||-1}$ holds universally for any distribution $__{||-1}$. Multiplying both sides by the non-negative denominator gives $(_)^2 |(||)^2}{||-1} (1 - _{n=1}^{||} _n^2)$. Multiplying by the constant factor $|-1}{||(||)^2}$ and substituting the definition of $C_V$ results in $C_V (_)^2 1 - _{n=1}^{||} _n^2$. Isolating the summation term yields the desired upper bound:

$$
_{n=1}^{||} _n^2 1 - C_V (_)^2.
$$

Combining these lower and upper bounds completes the proof of Theorem §prop:grad_bounds.




"`
[!t]


"`
[1]
Dataset $$, Initial Policy $_$, Group size $G$, Thresholds $_p, _h$, Batch size $B$
Initialize policy parameters $$

 Synchronize: $_{} $ 
 Sample prompts $^B $ and generate responses ${_1, ..., _G}^B _{_{}}(^B)$
 Compute advantages $^B_i$ using Eq. (§eq:ratio_adv_def)
 $ of size $B$}
 _i$ and token $t$}
 Obtain $p_{i,t} = _(y_{i,t} , _{i,<t})$ and $h_{i,t} = (_(, _{i,<t}))$
 $^{}_{i,t} 1$ 
 **if** $_i > 0 p_{i,t} < _p h_{i,t} < ^{(q)}_h$ **then** $^{}_{i,t} 0$ 
 Update $$ through Eq. (§eq:stapo_loss) with $^{}_{i,t}$
 **return** Final policy $_$

"`

"`

## Algorithm

The complete STAPO procedure is summarized in Algorithm §alg:stapo.

## Training Details

We implement our proposed STAPO algorithm based on the open-source alignment framework `veRL` [sheng2025hybridflow]. We utilize the DAPO-Math-17K [yu2025dapo] as the training dataset, where each prompt is formatted with the instruction: "Please reason step by step, and put your final answer within \ boxed{}". All models are trained using the AdamW optimizer with a constant learning rate of $1 10^{-6}$ and a warm-up phase of 10 steps. To ensure training stability, we apply a global gradient clipping norm of 1.0.

For efficient data generation, we utilize vLLM [kwon2023efficient] as the inference backend. The training process employs a global batch size of 256, with each prompt generating a group of $G=8$ rollouts. Following the DAPO formulation, we do not employ a separate value network or an additional KL divergence penalty term in the loss function; instead, we rely on group-relative advantages and the clipping mechanism to constrain policy updates. We conduct all experiments on 64 NVIDIA H20 GPUs, with each training session taking an average of 5 to 7 days. The complete set of hyperparameters used across all model scales is detailed in Table §tab:full_hyperparameters.

**Table:** *Full Training Hyperparameters*

| lc
**Hyperparameter** | **Value** |
| — | — |
| Reward Function | DAPO |
| Probability Threshold $_p$ | 0.002 |
| Entropy Percentile $q$ | 75 |
| Use KL Loss | No |
| Use Entropy Regularization | No |
| Train Batch Size | 256 |
| Max Prompt Length | 1k |
| Max Response Length | 15k |
| PPO Mini Batch Size | 64 |
| PPO Micro Batch Size/GPU | 1 |
| Clip Ratio Range | [0.8, 1.28] |
| Grad Clip | 1.0 |
| Learning Rate | 1e-6 |
| Warm-up Step | 10 |
| Training Temperature | 1.0 |
| Training Top-p | 1.0 |
| Validation Temperature | 1.0 or 0.7 |
| Validation Top-p | 1.0 or 0.9 |
| Rollout N | 8 |

## Supplementary Experimental Results

### Supplementary Training Dynamics

**Figure:** *AIME24 Acc, entropy, and training reward for Qwen3-8B base.* () _(image: figures/8b_V5.pdf)_

We further present the training dynamics for the larger model scales (Qwen3-8B and Qwen3-14B) in this section. As shown in Figure §fig:app_results, the trends observed on the 1.7B model scale consistently to larger models. Across both 8B and 14B base models, STAPO successfully prevents premature entropy collapse, maintaining stable exploration throughout the training process and achieving state-of-the-art performance. To clearly visualize the extended training process, we downsampled the plotted curves, which were originally evaluated every 20 gradient steps, by retaining only every third data point. Crucially, the true peak performance of each accuracy curve remains strictly preserved.

### Word Cloud Visualization

During the training of Qwen-1.7B, we visualized word clouds for low-probability correct answers, categorizing tokens into low-entropy "spurious tokens" and high-entropy "exploratory tokens" as illustrated in Figures §fig:spurious_tokens_cloud and §fig:non_spurious_tokens_cloud. The most frequent spurious tokens primarily include specific digits (e.g., "1", "2", "3") and mathematical symbols (e.g., "$", "x"). While these tokens may appear in correct responses, they are often associated with formatting or calculation errors. When combined with large gradient updates, they can induce disproportionately unstable optimization dynamics.
In contrast, the retained exploratory tokens correspond to the structural vocabulary of mathematical reasoning, including words like "Let", "This", "So", and "Wait". These tokens represent key procedural and logical components that sustain coherent reasoning chains. Retaining these tokens ensures that the model's core logical reasoning capabilities remain intact.

**Figure:** *Spurious tokens.* () _(image: figures/low_entropy_tokens_teal.pdf)_

### Supplementary Spurious Cases

In this section, we provide a detailed taxonomic description of the three categories of spurious tokens identified in our mechanistic analysis. Below we detail each category with a representative example referenced from the accompanying tables.

- **Category I: Uncommon Syntax (Table §tab:cat_1**). 
 This category comprises tokens that are linguistically valid but highly improbable given the model's pre-trained distribution. The model often substitutes standard technical terminology with colloquialisms or awkward phrasing. 
 \\
 *Example:* As shown in **Case 2**, when describing operations on a graph, the model selects the token "`broken`" (Prob: 0.05%) to describe the removal of edges. The canonical term "`removed`" dominates the Top-5 distribution (Prob: 85.53%). Reinforcing such low-probability synonyms forces the policy to drift away from standard mathematical language, destabilizing the linguistic entropy.
- **Category II: Hallucinations and Math Errors (Table §tab:cat_2**). 
 This category involves factual inaccuracies, calculation errors, or hallucinated values embedded within a trajectory that coincidentally results in a correct final answer (trajectories with locally incorrect steps but correct final answers).
 \\
 *Example:* In **Case 3**, the model attempts to verify a factorization with the equation $6901 = 67 103 }} 1$. Since $67 103$ equals $6901$ exactly, the inclusion of the subtraction operator "`-`" renders the statement mathematically false ($6900 6901$). Because the final answer matches the ground truth, the RL objective erroneously increases the likelihood of this arithmetic hallucination.

Throughout this section, model outputs are presented as rendered  to facilitate content verification. However, for errors categorized under formatting or syntax, we retain the raw token sequence to explicitly demonstrate the distributional anomalies.

{p{0.96}}
 \\
{c}{{\  – continued from previous page}} \\
{r}{{Continued on next page...}} \\
 
**Case 1** \\
**Context:** The length of $A$ is $1981$, and the length of $B$ is $1014$. Since the length of $U$ is $2012$, for the minimum length of $A B$, $B$ should start immediately from the right of $A$. The length of $A B$ is the sum of the lengths of $A$ and $B$ minus the length of $U$ , so its minimum value is $1981 + 1014 - 2012 = 983$. Here is a Python  to verify the above reasoning... \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0115% & 1.62 & code(51.90%) | program(31.48%) | solution(11.58%) | script(3.32%) | function(0.58%) \\
 \\ 
 
**Case 2** \\
**Context:** We are given $n$ islands with ferry lines between each pair of distinct islands, which is essentially a complete graph $K_n$. Each edge of the graph is colored by one of $k$ colors, corresponding to the companies operating the ferry lines.

For any closure of one company, all its lines must be  (the corresponding edges ignored) such that the remaining graph can no longer support a Hamiltonian Cycle, meaning it's no longer possible to visit all the islands exactly once without returning to the starting point.

According to the condition, every color, therefore, must serve to connect the graph in such a way that the removal of edges of any single color makes the graph Hamiltonian non-existent.

In a complete graph $K_n$, we can color the edges in such a way that removing any single color results in a graph that lacks a Hamiltonian Cycle. For this, we can use a result from graph theory, informally known as "edge coloring minimum spanning trees": for $n$ nodes, we can divide the set of nodes into $2{2}}$ color classes (as far as coloring spans, and where $$ means "the greatest integer less than $x$").

So when we have $n=2m$ (even) nodes, the maximal number of colors $k$ we can use is $m-1$, since we can divide our $n$ nodes into $m$ colors and deleting any one will still result in a complete graph $K_{2m-1}$ which remains Hamiltonian (since $K_{2m-1}$ is always Hamiltonian for $m1$).

For $n=2m+1$ (odd) nodes, we can divide our $n$ nodes into $m$ classes, so the  of any one of these $m$ classes will disintegrate the graph such that a Hamiltonian Cycle will become impossible because it contains a vertex of degree 1 or a subdivision thereof. \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0471% & 2.47 & removed(85.53%) | closed(3.32%) | deleted(2.58%) | disconnected(1.57%) | taken(0.95%) \\
  & 0.0001% & 2.47 & maximal(85.89%) | maximum(9.05%) | number(1.57%) | largest(0.45%) | answer(0.45%) \\
 \\

 
**Case 3** \\
**Context:** Therefore, the final  answer is: $$

<|endoftext|>} \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.1486% & 0.54 & answer(98.88%) | result(0.52%) | value(0.19%) | boxed(0.15%) | solution(0.15%) \\
 \\ 
 
**Case 4** \\
**Context:** To solve the given problem, we need to find the smallest number of candies $N$ that satisfies the given conditions. Specifically:

1. When $N$ candies are divided  21 people (Albert and his 20 friends), the remainder is 5. \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0667% & 0.72 & among(64.55%) | by(23.75%) | evenly(8.74%) | amongst(1.95%) | equally(0.92%) \\
 \\ 
 
**Case 5** \\
**Context:** To isegment, we must first count the number of points in the polygon.

 Step 1:sume the number of points in the polygon

Let $n$ ea the number of sides (or vertices) in a polygon $P$ as given in the problem statement. We are given that $P_{1}, P_{2}, ... P_{n} /$ are the $n$ vertices of the polygon. \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0004% & 0.72 & olve(63.83%) | determine(30.15%) | find(5.24%) | triang(0.20%) | count(0.10%) \\
  & 0.0003% & 0.72 & | be(88.02%) | represent(5.63%) | denote(3.41%) | =(0.76%) | and(0.59%) \\
 \\


{p{0.96}}

 \\
{c}{{\  – continued from previous page}} \\
{r}{{Continued on next page...}} \\
 
**Case 1** \\
**Context:** 

$$A = {2}|_1(_2-_3) + _2(_3-_{}) + _3(_1-_2) |$$

Here, $ x and $y$s are the coordinates of the points. \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0142% & 0.54 & 1(99.95%) | 2(0.01%) | 3(0.01%) | 0(0.00%) | }_(0.00%) \\
  & 0.0117% & 0.54 & _(94.78%) | 1(1.05%) | =(0.82%) | _{(0.82%) | $(0.50%) \\
 \\ 
 
**Case 2** \\
**Context:** \( (a - 10, b - 10) = (193, 11) \) → \( a = 203 \), \( b = 21 \) \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0017% & 0.94 & `\` (99.98%) | `\ n`(0.01%) | 1(0.01%) | `\\ n`(0.00%) | 3(0.00%) \\
 \\

 
**Case 3** \\
**Context:** $6901 = 67 103  1$ \\

[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0202% & 1.21 & $,(99.48%) | $(0.36%) | +(0.13%) | -(0.02%) | =(0.00%) \\
 \\ 
 
**Case 4** \\
**Context:** 
$-16a^3 = 350,$

$a^3 = -21.625.$ \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.2433% & 1.62 & 8(98.16%) | 2(1.40%) | 5(0.24%) | 9(0.07%) | 7(0.04%) \\
 \\ 
 
**Case 5** \\
**Context:** Compute \( a \) for Each Pair:\\
$
 (-5, 30) & a = -25 \\
 (-4, 12) & a = -8 \\
 (-3, 6) & a = -3 \\
 (-2, 3) & a = -1 \\
 (0, 0) & a = 0 \\
 (-7, -42) & a = 49 \\
 (-8, -24) & a = 32 \\
 (-9, -18) & a =  \\ 
 (-10, -15) & a = 5 \\
 (-12, -12) & a = 24

$\\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0123% & 0.35 & 2(99.98%) | 9(0.01%) | 1(0.01%) | 3(0.00%) | 8(0.00%) \\
 \\
 
**Case 6** \\
**Context:** 
Thus,

$$
x = {35} = {533}
$$

 \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0867% & 1.62 & 5(95.28%) | 3(3.69%) | `\`(0.64%) | 6(0.11%) | 4(0.09%) \\
 \\ 



{p{0.96}}

 \\
{c}{{\  – continued from previous page}} \\
{r}{{Continued on next page...}} \\
 
**Case 1** \\
**Context:** To solve this problem, let's start by using the given information about the function \( f(x) = ax^2 + bx + c \):

1. Since \( f(1) = 0 \), we know that:
 

$$
 a + b + c = 0
 $$

 This means \( c = -a - b \).

, substitute \( c = -a - b \) into the quadratic function:
 

$$
 f(x) = ax^2 + bx - (a + b)
 $$

 \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0552% & 0.35 & 2(99.74%) | Next(0.12%) | Now(0.06%) | So(0.03%) | Sub(0.02%) \\
 \\ 
 
**Case 2** \\
**Context:** 
`\`boxed{5text{ agony}}

\\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0015% & 0.35 & `\`\ n}(99.57%) | }(0.25%) | }`\ n\ n`(0.12%) | }`\`(0.03%) | }.`\ n`(0.01%)
 \\

 
**Case 3** \\
**Context:** `\ therefore 10^ y`` \(\ sqrt{25\`=5} \\

[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0063% & 0.72 & =(84.69%) | =`\`(6.95%) | =(4.22%) | `\`n(1.55%) | `\`n(0.35%) \\
 \\ 
 
**Case 4** \\
**Context:** 

$$
_3 = 11^3 + 10^3 + 9^3 + 8^3 + 7^3 + 6^3 + 5^3 + 4^3 +\!^3 +2^3 + 1^3 + 0^3
$$

 \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0710% & 1.62 & ``(99.92%) | 3(0.07%) | `\`n(0.00%) | `\`(0.00%) | +(0.00%) \\
 \\ 
 
**Case 5** \\
**Context:** Repeating these steps three more times for each roll back to $P_4$:
- If $A$ was rolled, $P_5 = (56, 368)$
- If $B$ was rolled, $P_5 = (56, 184)$
- If $C$ was rolled, $P_5 = (768, 184)$\\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0035% & 2.47 & `\`n`\`n(88.06%) | ((4.38%) | `\`n(4.38%) | `\`n`\`n(1.26%) | ,(0.36%) \\
 \\
 
**Case 6** \\
**Context:** 
Hence, the maximum value of `$\ frac{n_i\`{k}$} for `1 \ leq i \ leq 70` is: `\[` `\ boxed{553\`} `\]` ` \\
[0.5pt]
 {@{}l l l p{8cm}@{}}
 **Spurious Token** & **Prob ($P$)** & **Adv** & **Top-5 Distribution** \\
  & 0.0000% & 1.21 & `\`(98.58%) | `\`)(0.13%) | `\`(0.10%) | $(0.08%) | $$(0.07%) \\
 \\ 



% — END sections/appendix —