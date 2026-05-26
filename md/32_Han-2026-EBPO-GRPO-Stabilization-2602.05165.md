# EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization

> **arXiv:** [2602.05165](https://arxiv.org/abs/2602.05165)
> **TeX source:** [arXiv-2602.05165v1/](arXiv-2602.05165v1/)
> **PDF:** [EBPO-arXiv-2602.05165v1.pdf](EBPO-arXiv-2602.05165v1.pdf)

---

% — BEGIN arxiv/1_introduction —

## Introduction

Post-training for Large Language Models (LLMs) has increasingly shifted toward reinforcing reasoning using verifiable signals. Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a stable and reproducible paradigm for tasks such as mathematical reasoning and code generation, where objective correctness can serve as a ground-truth reward signal [shao2024deepseekmath, guo2025deepseek]. To avoid the computational overhead of training separate value networks, recent advancements have coalesced around group-based methods, most notably Group Relative Policy Optimization (GRPO) [shao2024deepseekmath]. By normalizing rewards within a sampled group of outputs for a given prompt, GRPO provides a computationally efficient baseline that has demonstrated significant improvements in model reliability [shao2024deepseekmath, yu2025dapo].

Despite its success, GRPO faces inherent limitations regarding the stability of its advantage estimator. Because the baseline is derived solely from the local group mean, it is susceptible to high variance when the group size ($G$) is small [yu2025dapo]. Furthermore, GRPO lacks robustness in "saturated" regimes: if a model fails all attempts for a difficult prompt (all rewards are zero), the relative advantage vanishes, resulting in a null gradient and a wasted training step [liu2025understanding]. While recent approaches like DAPO [yu2025dapo] attempt to mitigate this volatility by filtering or reweighting unstable updates, they do so by effectively discarding the problematic examples, leading to unavoidable data waste. Consequently, standard GRPO often necessitates larger group sizes to suppress noise without losing data, thereby drastically increasing the computational cost of training [shao2024deepseekmath].

In this work, we introduce **Empirical Bayes Policy Optimization (EBPO)**, which reframes advantage estimation through the lens of Empirical Bayes (EB) inference [robbins1992empirical]. Classical EB theory suggests that when estimating parameters for parallel tasks, one can reduce estimation error by assuming those parameters share a common underlying distribution [robbins1992empirical]. Applying this to RLVR, we postulate that the latent success probability of a prompt is drawn from a global distribution characterized by the policy's historical performance.

EBPO replaces the purely local GRPO baseline with a shrinkage estimator that pulls the noisy group mean toward a global mean, $_{glob}$. The degree of this shrinkage is determined dynamically by the ratio of within-group variance to between-group variance. This formulation allows EBPO to distinguish between failing a globally "hard" task (consistent with the prior) versus failing an "easy" task (deviating from the prior), assigning adaptive penalty signals accordingly. To ensure scalability, we estimate these global priors dynamically using Welford's online algorithm [welford1962note]. 

We validate EBPO through both theoretical analysis and extensive empirical evaluation. Theoretically, we prove that the EBPO baseline achieves a strictly lower Mean Squared Error (MSE) than the standard sample mean used in GRPO and provides informative, non-zero gradients even when group rewards are saturated. Empirically, we evaluate EBPO on various datasets with different scales of base models. Our results show that EBPO consistently outperforms baselines on challenging benchmarks such as AIME-2024 [li2024numinamath] and OlympiadBench [he2024olympiadbench]. Furthermore, we demonstrate that EBPO is highly sample-efficient, outperforming GRPO by over 11% in extremely resource-constrained settings ($G = 8$), while effectively leveraging curriculum learning via difficulty-stratified sampling.

Our contributions are summarized as follows:

- **Algorithmic Framework:** We propose EBPO, which integrates online Empirical Bayes estimation into the GRPO framework to stabilize advantage computation.
- **Theoretical Guarantees:** We provide proofs that EBPO resolves the vanishing gradient problem in saturated groups and strictly reduces the variance of the baseline estimator compared to GRPO.

% — END arxiv/1_introduction —

% — BEGIN arxiv/2_related —

## Related Work

**Reinforcement Learning with Verifiable Rewards..** 

Previous work has shown that RLVR is effective in enhancing reasoning and factual correctness in LLMs [shao2024deepseekmath, guo2025deepseek]. By leveraging automatically checkable signals, RLVR enables stable and reproducible optimization for reasoning tasks. A variety of algorithmic variants were proposed, including GRPO-based approaches that integrate verifiable rewards into group-based policy optimization frameworks [shao2024deepseekmath, liu2025understanding, yu2025dapo, cui2025entropy, zheng2025group, zhou2025mixture]. These methods demonstrate that verifiable reward signals can significantly improve model reliability. 

Among these GRPO variants, a broad range of studies focus on improving the effectiveness and efficiency of RLVR training, such as by proposing novel learning objectives [liu2025understanding, zhou2025disco, yu2025dapo, yue2025vapo, liu2025uniform], introducing entropy-based mechanisms to encourage broader exploration [cui2025entropy, wang2025beyond, zhang2025edge, le2025no, yang2025let], or designing more robust reward functions [jia2025autorubric, li2025semantically, xiong2026token]. However, few have considered the estimation of the mean or standard deviation of rewards during advantage computation, which can substantially influence group-level differences. Our work addresses this gap by employing an Empirical Bayes method to more robustly estimate the mean and standard deviation of group rewards, thereby stabilizing the training dynamics of GRPO.

zeng2025shrinking [zeng2025shrinking] also address the high-variance bottleneck in GRPO, using Stein’s Paradox to theoretically guarantee that their shrinkage-based baseline reduces variance. Unlike their approach, our framework uses Empirical Bayes inference to dynamically control shrinkage based on variance ratios, which is crucial for avoiding vanishing gradients in challenging prompts. Both works highlight the importance of global-local baseline interpolation for stable RL with verifiable rewards.

**Empirical Bayes Estimation.** 

Empirical Bayes provides a framework to handle estimation in the face of parallel, noisy data. It is a powerful methodology that blends the data-driven nature of frequentist statistics with the hierarchical modeling structure of Bayesian inference, offering a principled approach to regularization and robust estimation. Instead of specifying the prior hyperparameters in advance, Empirical Bayes uses the observed data to estimate them [robbins1992empirical]. Later, statisticians found that shrinkage estimators like the James-Stein estimator [james1961estimation] can be viewed as Empirical Bayes estimators with a Gaussian prior. Empirical Bayes methods have been widely adopted for hyperparameter tuning in Bayesian reinforcement learning [reisinger2008online] and prior estimation in multi-task meta-learning [hu2020empirical, grant2018recasting]. Additionally, these methods facilitate uncertainty estimation in deep neural networks [krishnan2020specifying] and the development of Bayesian dialogue agents through data-driven parameter initialization [lee-etal-2023-empirical].

While Bayesian principles have been increasingly applied to LLMs, their utility has been largely confined to evaluation frameworks and black-box optimization rather than online policy training. For instance, liu2024large [liu2024large] integrates LLMs into Bayesian Optimization to enhance surrogate modeling and candidate sampling for hyperparameter tuning. xiao2025confidence [xiao2025confidence] and hariri2025don [hariri2025don] propose a Bayesian approach to estimate model capabilities in the evaluation process. However, these approaches operate primarily on static evaluation or auxiliary optimization tasks. To the best of our knowledge, EBPO is the first framework to integrate Empirical Bayes directly into the online optimization loop of GRPO, utilizing shrinkage estimators to stabilize gradient variance during active training.

% — END arxiv/2_related —

% — BEGIN arxiv/3_method —

We further analyze the impact of EBPO on policy entropy and the role of data ordering in minimizing estimator bias.

**Theorem:** [Variance Reduction and Entropy Conservation]

Let $H()$ denote the expected reduction in policy entropy per optimization step. Under the standard assumption that the magnitude of the policy update is proportional to the advantage magnitude, i.e., $\|\| ||$, the expected entropy decay of EBPO is strictly lower than that of GRPO for small group sizes $G$. Formally:

$$
 [H(_{t}) - H(_{t+1}) ] < [H(_{t}) - H(_{t+1}) ]
$$

> **Proof:** 
The rate of entropy decay in policy gradient methods is driven by the variance of the gradient estimator. High-variance advantages lead to large, erratic updates that push probability mass towards extreme values based on noise, rapidly reducing entropy (mode collapse).

Recall from Theorem §thm:mse_stability that the variance of the EBPO advantage is scaled by the shrinkage factor $(1 - )^2$:

$$
 (_{}) = (1 - )^2 (_{})
$$

Since $0 <  < 1$, we have $(_{}) < (_{})$. By Jensen's inequality, the expected norm of the update step is bounded by the variance of the advantage. A lower variance estimator produces smaller expected shifts in the policy parameters $$:

$$
 [\|_{}\|^2] < [\|_{}\|^2]
$$

Since the change in entropy $H -H $, smaller, more stable updates imply a slower rate of divergence from the current maximum-entropy state. Thus, EBPO preserves exploration longer than GRPO, preventing premature convergence on noisy, small-sample groups.

**Theorem:** [Entropy Conservation via Covariance Suppression]

Let $H() H(_{t+1}) - H(_t)$ be the single-step change in policy entropy. Following the derivation in cui2025entropy [cui2025entropy], $H()$ is dominated by the negative covariance between the token probability $p_(o|q)$ and the estimated advantage $(o,q)$:

$$
 H() -_{o} (p_(o|q), (o,q))
$$

where $$ is the learning rate. Under the condition that the GRPO advantage $_{}$ is positively correlated with token probability (i.e., the model is reinforcing its current mode), the entropy decay of EBPO is strictly slower than that of GRPO by a factor proportional to the shrinkage coefficient $(1-)$.

> **Proof:** 
cui2025entropy [cui2025entropy] demonstrated that in Policy Gradient methods, the change in entropy is driven by the update to the logits. Specifically, $H - _{q} [ _o p(o) (o) (1 - p(o)) ]$. This term effectively measures the covariance between the likelihood of an action and the reward signal it receives. When the policy assigns high probability to high-advantage actions (the typical "collapse" phase), this covariance is large and positive, driving $H$ to be large and negative.

In EBPO, the advantage is given by the shrinkage estimator:

$$
 _{} = )_{} + _{})}{+ }
$$

Rearranging terms relative to the GRPO advantage $_{} }}{}$, we can express the EBPO advantage roughly as:

$$
 _{} _{} - (_{} - _{})}{}
$$

However, a more direct analysis of the gradient magnitude reveals that the shrinkage factor $$ dampens the sensitivity of the baseline to the local batch. In the critical regime where $_{}$ is high (the model is confident and correct), GRPO produces large advantage differences that reinforce the mode. EBPO, by pulling the baseline towards $_{}$ (which is typically $< _{}$ for high-performing prompts), actually *increases* the baseline $B_q$ relative to the group mean, thereby reducing the magnitude of the positive advantage for the mode.

More formally, considering the variance reduction property (Theorem §thm:mse_stability), the magnitude of the update vector is scaled down. Since $H -\|_J\|$, and the EBPO gradient norm is bounded by the shrinkage:

$$
 |H_{}| (1 - ) |H_{}|
$$

Since $ (0, 1)$, it follows that $|H_{}| < |H_{}|$. Thus, EBPO suppresses the covariance term that drives rapid entropy collapse, preserving exploration for longer periods.

## Methodology

### Preliminaries

**Reinforcement Learning from Verifiable Rewards (RLVR).** 
RLVR provides a memory-efficient framework for optimizing reasoning policies $_$ in domains where the correctness of a response $o$ can be objectively determined by a verifier [shao2024deepseekmath]. Unlike traditional actor-critic methods that require a learned value network, RLVR utilizes group-relative advantage estimation to provide a baseline for policy updates.

For a given prompt $q$, the policy $_$ samples a group of $G$ independent trajectories ${o_1, o_2, ..., o_G}$. Each trajectory is assigned a verified reward $r_i $ (or a shaped scalar reward) based on its correctness. The advantage $_i$ for each response is then computed by standardizing the reward within its group: $_i = }}{_{} + }$, 
where $_{}$ and $_{}$ are the empirical mean and standard deviation of the rewards within the group:

$$
 _{} = {G} _{j=1}^G r_j, _{} = {G-1} _{j=1}^G (r_j - _{})^2}
$$

The policy parameters $$ are updated by maximizing the clipped surrogate objective:

$$
 _{}() = _{q } [ {G} _{i=1}^G ( _i() _i, (_i(), 1-, 1+) _i ) ]
$$

where $_i() = {_{_{}}(o_i|q)}$ is the importance sampling ratio. While RLVR eliminates the computational cost of a critic model, the estimator $_{}$ suffers from high variance when $G$ is small. This instability often leads to noisy gradients, particularly in mathematical reasoning where the reward distribution is sparse. 

**Empirical Bayes and Simultaneous Estimation.** 

In classical Bayesian inference, a prior distribution is specified before any data is observed. In Empirical Bayes (EB), the prior is instead estimated from the data itself [robbins1992empirical]. Consider the general problem of simultaneously estimating a set of $M$ related parameters $_1, ..., _M$ based on independent observations $y_1, ..., y_M$, where each $y_m$ serves as a noisy estimate of $_m$. Rather than estimating each parameter in isolation—which often yields high variance when data is sparse—EB assumes that the parameters are exchangeable and drawn from a common global prior distribution:
$_m (, ^2)$,
where $$ is the global mean and $^2$ is the between-parameter variance (or prior variance). By estimating these global hyperparameters $(, ^2)$ from the marginal distribution of the entire dataset, we can construct a posterior estimate for each individual $_m$. This results in a shrinkage estimator, which "pulls" individual, noisy observations toward the global mean $$. 

### Empirical Bayes Policy Optimization (EBPO)

A primary challenge in post-training LLMs on reasoning tasks, such as mathematics or formal logic, is the *sparsity and high variance of rewards* [uesato2022solving, zelikman2022star, lightman2023lets, shao2024deepseekmath, yu2025dapo]. In GRPO methods, the advantage is computed by normalizing rewards within a group of completions for a single prompt. However, when a prompt is excessively difficult, the model may fail all attempts (i.e., all rewards are zero), resulting in a null gradient. Conversely, for trivial prompts, the model might succeed in all attempts, again providing no relative signal for improvement. 

We propose **Empirical Bayes Policy Optimization (EBPO)**, which regularizes the local group-based baseline by "borrowing strength" from the global performance distribution of the policy. We assume that for any prompt $q$, the true latent success probability $_q$ is drawn from a global distribution: $_q (_{}, ^2)$, where $_{}$ represents the global average success rate of the policy and $^2$ captures the variance in difficulty across different prompts. While Beta-Binomial models are natural for binary rewards, we employ a Gaussian approximation to facilitate tractable online inference, which yields a closed-form linear shrinkage estimator that avoids the numerical hyperparameter optimization required by Beta priors, ensuring computational efficiency for online training. Following the Empirical Bayes paradigm, we do not fix these hyperparameters *a priori* but estimate $_{}$ and $^2$ dynamically from the data across all prompts in the training history. This allows us to compute a shrinkage estimator that pulls the noisy local mean $_{}$ toward the global policy average (Figure §fig:ebpo_overview).

**Figure:** ***Overview of the EBPO Framework.** Unlike standard GRPO which relies solely on the local group mean, EBPO computes a "Smart Baseline" (Shrinkage Estimator) by blending the noisy local group mean with a stable global prior (updated via global history). This allows for informative advantage estimates even in small groups or saturated failure regimes.* () _(image: figs/ebpo_illus.pdf)_

Let ${q_1, ..., q_M}$ be a batch of prompts. For each prompt $q_m$, we sample $G$ responses ${o_{m,1}, ..., o_{m,G}}$ and obtain rewards $r_{m,i}$. We define the local group mean as $_m = {G} _{i=1}^G r_{m,i}$. EBPO models the relationship between local performance and global policy capability using two levels of variance:

1. **Within-group variance ($^2$)**: The variance of individual rewards for a given prompt across all generations. We estimate this with $^2 = (r_{m,i})$.

**Remark:** [Online Stability vs. Bias for Between Group Variances] 
 We note that strictly speaking we would define the prior variance as $_{latent}^2 = (_m) - ^2/G$ to remove the sampling noise. However, in online training settings, this subtraction can lead to negative variance estimates. By using the raw $(_m)$ as a proxy for $^2$, we effectively inflate the denominator, resulting in a *conservative* shrinkage estimator. This ensures the shrinkage factor $_q$ remains strictly in $[0, 1]$ and favors the local group mean when uncertainty is high, providing a safe "soft" regularization that prevents policy collapse without over-constraining the model.

The EBPO Baseline $V_q^{EB}$ for a prompt $q$ is defined as the posterior mean of the reward:

$$
 V_q^{EB} = (1 - _q) _{} + _q _{}
$$

where the shrinkage factor $_q [0, 1]$ is determined by the ratio of variances:

$$
 _q = {^2 / G + ^2}
$$

To ensure training stability and scalability, we maintain running estimates of $_{glob}$, $^2$, and $^2$ using **Welford's Online Algorithm** [welford1962note]. This avoids the noise associated with small-batch statistics. For each step, the advantage for a completion $(q_m, o_{m,i}, r_{m,i})$ is computed as the deviation from this regularized baseline:

$$
 _{m,i} =  - V_{q_m}^{EB}) - _{}}{_{} + }
$$

where $_{}$ and $_{}$ are batch-level mean and standard deviation of raw advantages. The full algorithm is presented in Algorithm §alg:EBPO in Appendix §sec:supp_materials.

### Theoretical Analysis of EBPO

In this section, we analyze the theoretical properties of the EBPO estimator and formally demonstrate its advantages over the standard GRPO baseline. Proofs of results in this section can be found in Appendix §sec:proof_details.

#### Definitions and Setup

Let $q$ be a prompt and ${o_1, ..., o_G}$ be a group of $G$ responses generated by policy $_$. Let $r_i $ be the binary reward for response $o_i$.

**GRPO Baseline:** The GRPO baseline is the local sample mean:

$$
 V^{} = _{} = {G}_{i=1}^G r_i
$$

The GRPO raw advantage (without normalization) is $^{raw}_{}(o_i) = r_i - V^{}$.

**EBPO Baseline:** The EBPO baseline is the shrinkage estimator:

$$
 V^{} = (1 - )_{} + _{}
$$

where $ (0, 1]$ is the shrinkage factor and $_{} > 0$ is the global policy success rate. The EBPO raw advantage is $^{raw}_{}(o_i) = r_i - V^{}$.

#### Theoretical Properties of EBPO
 
The first two theorems are straightforward from our construction. Unlike GRPO, which is entirely local, EBPO allows the model to differentiate between failing a difficult task (where the reward remains close to the baseline) and failing an easy task (where the baseline is high, leading to a large negative advantage).

**Theorem:** [Non-Vanishing Gradients in Saturation Regimes]

Consider a **saturated failure group** where the policy fails on all sampled responses for a specific prompt $q$, i.e., $r_i = 0$ for all $i $. Under these conditions:

1. The GRPO gradient contribution is zero (vanishes).

**Theorem:** [Stability via MSE Reduction]

Assume the true latent success rate for prompt $q$ is $_q$, and the observed group mean $_{}$ is an unbiased estimator of $_q$ with variance $^2/G$. Under the Gaussian approximation where $_q (_{}, ^2)$, the EBPO baseline $V^{}$ achieves strictly lower Mean Squared Error (MSE) than the GRPO baseline $_{}$ for estimating the true difficulty $_q$.

**Remark:** Unlike the unbiased sample mean in GRPO, EBPO introduces a bias towards the global prior $_{glob}$ to minimize the MSE, a standard trade-off in shrinkage estimation. The global statistics $_{glob}$ and $^2$ are updated online using the current batch (Algorithm §alg:EBPO), with the bias introduced by this correlation decays at a rate of $O(1/T)$, where $T$ is the total number of accumulated training steps. So for large $T$, the global statistics act as fixed priors relative to the local group updates, keeping the policy gradient asymptotically consistent.

**Corollary:** [Global Context Sensitivity]

The penalty assigned by EBPO for a failure ($r_i=0$) in a saturated group scales dynamically with the global task difficulty $_{}$. Specifically, failures on globally "easy" tasks (high $_{}$) incur larger penalties than failures on globally "hard" tasks (low $_{}$) (as visualized in Figure §fig:penalty_scaling).

**Figure:** ***Advantage Signal in Failure Scenarios ($r_i=0$ for all $i$).** 
 Comparison of the advantage signal of GRPO (dashed line) versus EBPO (solid blue line) when the model fails all attempts for a given prompt. 
 While GRPO yields a vanishing gradient ($)$ denote the reduction in policy entropy after a single gradient descent step with learning rate $$. 
Under the assumption that the magnitude of the likelihood-reward covariance $_q$ is independent of the local group scale $_q$, the entropy reduction of EBPO is strictly bounded by that of GRPO:

$$
 [H()_{}] < [H()_{}]
$$

provided the task distribution exhibits non-zero between-group variance (i.e., $_q(_q) > 0$), ensuring that the global normalization constant effectively dampens updates for low-variance groups.

**Remark:** 
 While the independence assumption may not strictly hold in all regimes (e.g., highly saturated groups might exhibit lower covariance), it captures the dominant dynamics of entropy decay. We empirically validate its practical hold in Section §sec:results (Figure §fig:entropy_curve).

#### Optimizing Prior Estimation via Clustered Sampling

While EBPO stabilizes the baseline by leveraging global history, a potential limitation arises from the heterogeneity of the training data. In a standard random shuffle, the online estimator for $_{}$ aggregates performance across vastly different domains (e.g., mixing simple arithmetic with complex calculus) or different difficulties (e.g., mixing IMO problems with AMC 8 problems). This high variance in the data stream can cause the global prior to converge to a coarse average that creates a "mismatch" for specific, distinct tasks.

To address this, we propose organizing the training stream into coherent clusters. We explore two strategies:

1. **Topic Clustering:** Grouping prompts by mathematical domain (e.g., Algebra, Geometry).

By presenting data in clustered sequences, the online estimator adapts to consistent local distributions rather than oscillating between extremes. We formally justify this approach with the following proposition, which shows that clustered sampling reduces the estimation error of the prior.

**Proposition:** [Advantage of Clustered Sampling]

Let the dataset $$ be partitioned into $K$ distinct clusters (topics) ${_1, ..., _K}$. Let $_q$ be the latent success rate of a prompt $q$. Assume the true difficulty varies by topic, such that $_k = [_q q _k]$ varies across $k$, while the global mean is $_{} = _k[_k]$.

Let $_{}$ be the global statistic used by EBPO. We define two streaming regimes:

- **Random Shuffle:** The stream is sampled uniformly from $$, such that $_{}$ converges to $_{}$.

The Mean Squared Error (MSE) of the prior estimate w.r.t the true difficulty $_q$ is strictly minimized in the Topic-Coherent regime.

% — END arxiv/3_method —

% — BEGIN arxiv/4_experiments —

## Experiments

We first outline the experimental setup. Then, we compare the proposed method with GRPO and its variants under different experimental settings.

### Experimental Setup

**Dataset..** 
We use the DAPO-Math-17K dataset as the training corpus [yu2025dapo]. For evaluation, we select a suite of competitive math reasoning benchmarks, including AIME2024 [li2024numinamath], AIME2025 [li2024numinamath], AMC23 [li2024numinamath], Math-500 [hendrycks2021measuring], and OlympiadBench [he2024olympiadbench]. Each model is evaluated using the Pass@1 metric. To ensure statistical robustness, we repeat each evaluation set 32 times with different random seeds and report the average Pass@1 across runs.

**Models..** 
We adopt a diverse set of LLMs with different architectures and parameter scales. Specifically, we use **LLaMA3.1-8B** [dubey2024llama], **Qwen3-8B**, and **Qwen3-14B** [yang2025qwen3]. This selection spans multiple LLM families (LLaMA and Qwen) and model sizes, providing a robust and comprehensive testbed for evaluating our agent workflow and training recipe.

**Baseline..** 
We compare our proposed **** method with several representative baselines, including Naive GRPO [shao2024deepseekmath], DAPO [yu2025dapo], Dr. GRPO [liu2025understanding], and EntropyMech [cui2025entropy]. 
For a fair comparison, all methods are trained under identical conditions: the same training data ordering, batch size, and optimization configuration are applied across all baselines and our method. Specific configuration details for each baseline are provided in Appendix §sec:hyper_details.

**Clustered Sampling Details..** 
To implement the clustered sampling optimization in Section §sec:clustering, we evaluate two sampling strategies: *clustering by topic* and *clustering by difficulty*, which correspond to **-topic** and **-diff**, respectively. 
For difficulty-based clustering, we use the base model's pass rate as the difficulty metric, grouping problems of similar success rates together. 
For topic-based clustering, we employ GPT-4.1 [achiam2023gpt] to annotate each problem with one topic label. 
Details are provided in Appendix §sec:hyper_details.

### Results
 

**Table:** *
Performance comparison of and baseline methods across different evaluation datasets (group size = 4). 
The dataset is clustered by topics during training. Each value represents Pass@1 (%). 
The best result within a base model group is highlighted in bold.
*

| lccccc|c
**Method** | **MATH-500** | **AIME-2024** | **AIME-2025** | **AMC23** | **OlympiadBench** | **Average** |
| — | — | — | — | — | — | — | — |
| GRPO | 65.60 | 50.21 | 42.29 | **89.53** | 45.99 | 58.72 |
| Dr GRPO | 67.68 | 51.04 | 32.71 | 85.00 | 44.91 | 56.67 |
| DAPO | 58.39 | 45.63 | 32.71 | 82.81 | 43.62 | 52.63 |
| EntropyMech | 53.88 | 37.92 | 30.42 | 79.99 | 43.69 | 49.18 |
| GRPO | 22.66 | 2.71 | 0.00 | 12.66 | 6.97 | 9.00 |
| Dr GRPO | 22.12 | **2.91** | **0.42** | 12.36 | 6.65 | 8.89 |
| DAPO | 18.05 | 2.29 | 0.00 | 11.09 | 6.05 | 7.90 |
| EntropyMech | 15.01 | 1.25 | 0.21 | 10.16 | 5.30 | 6.79 |
| GRPO | **75.73** | 49.99 | 37.92 | 85.00 | 50.77 | 59.88 |
| Dr GRPO | 73.63 | 53.33 | 39.79 | 82.97 | **51.08** | 60.16 |
| DAPO | 67.68 | 57.29 | **51.46** | 84.22 | 46.03 | 61.34 |
| EntropyMech | 59.94 | 49.37 | 40.83 | 78.75 | 41.21 | 54.42 |

**Figure:** *Evolution of Gradient Norm ($||_J||_2$).* () _(image: figs/icml_grad_norm.pdf)_

**consistently outperforms GRPO and its variants across multiple reasoning benchmarks and model sizes..** 

We report results obtained under the setting where the training data are clustered by topic and the group size of GRPO is 4, as summarized in Table §tab:ebpo_results. 
Across all evaluated models and benchmarks, **-topic** achieves clear and consistent improvements over prior value-free reinforcement learning baselines. 
For example, on the Qwen3-8B model, reaches an average Pass@1 of **64.39%**, outperforming GRPO by more than **5%**. 
Similar trends are observed on other models, indicating that the benefits of generalize across architectures and scales. Overall, among the 15 possible (model, dataset) combinations, achieves the best result in 9 cases, demonstrating its robustness across both architectures and reasoning tasks.

These results highlight the advantage of , which stabilizes gradient estimation across prompt groups and improves reward signal utilization. 
As a result, produces stronger and more reliable reasoning performance compared to existing GRPO-based approaches.

**achieves stable and non-vanishing policy optimization.** To analyze the training dynamics, we examine the magnitude of policy updates and gradient flow for the Qwen3-8B training. First, in Figure §fig:grad_norm, we plot the gradient norm ($||_J||_2$). Standard GRPO exhibits consistently lower gradient magnitudes, indicative of **vanishing gradients** in saturated regimes where all group responses fail and yield zero relative advantage. In contrast, EBPO maintains a robust gradient flow throughout training, confirming that the shrinkage-regularized baseline provides informative penalty signals even when local variance is zero. Second, we track the per-step KL divergence ($(_t || _{t+1})$) in Figure §fig:per_step_kl. While GRPO shows signs of instability in later stages, with update sizes spiking unpredictably, EBPO maintains a strictly bounded update magnitude. These metrics demonstrate that EBPO strikes a critical balance: it prevents the policy from stalling due to vanishing signals while regularizing it against the catastrophic large-step deviations that lead to model collapse.

**demonstrates stronger exploration behavior compared to GRPO.**

We explicitly verify the theoretical claims of Proposition §thm:entropy_conservation by tracking the policy entropy in Figure §fig:entropy_curve. consistently maintains **higher entropy** during training compared to GRPO, suggesting that it encourages greater exploration and prevents premature policy collapse. This aligns with the algorithmic design of : by incorporating a batch-level shrinkage baseline, the policy updates are better balanced between exploitation of high-reward samples and exploration of underrepresented regions in the solution space. This preservation of exploration capability has a direct impact on generalization performance. As shown in Figure §fig:val_performance, we track the evolution of validation reward (Majority Vote@16 and Pass@16) throughout training. While GRPO plateaus due to mode collapse, EBPO maintains a continuous upward trajectory in both Majority Vote@16 and Pass@16, yielding a policy with higher robustness and peak capability. Furthermore, We also conduct an ablation study in Appendix §appendix:ablation_clustering to investigate the impact of topic clustering on the efficacy of the framework.

**Figure:** *
 **Evolution of Policy Entropy ($G=4$).** maintains a consistently higher policy entropy than GRPO, demonstrating that the global prior effectively sustains exploration throughout training.* () _(image: figs/icml_policy_entropy.pdf)_

**Figure:** *
 **Validation Performance.** While GRPO plateaus or degrades in late-stage training—indicative of overfitting or policy collapse—EBPO exhibits sustained performance gains.
 * () _(image: figs/icml_acc_maj_16.pdf)_

In summary, EBPO stabilizes optimization by mitigating vanishing gradients and strictly bounding update magnitudes, effectively preventing both stagnation and model collapse. This algorithmic resilience, combined with sustained exploration via entropy conservation, drives the consistent performance gains observed across reasoning benchmarks.

**Figure:** *Reward curve (group size = 4).* () _(image: figs/reward_curve_g4.png)_

**Table:** *Performance comparison across different group sizes. consistently demonstrates superior sample efficiency.*

| lccccccc
**Method** | $G$ | **MATH** | **AIME24** | **AIME25** | **AMC23** | **Olympiad** | **Avg.** |
| — | — | — | — | — | — | — | — |
| GRPO | 8 | 68.79 | 47.29 | 31.46 | 79.69 | 44.06 | 54.26 |
| DAPO | 8 | 56.63 | 35.42 | 28.33 | 77.34 | 35.75 | 46.69 |
| GRPO | 16 | **77.98** | 55.42 | 44.58 | 85.94 | 51.89 | 63.16 |
| DAPO | 16 | 73.26 | 46.67 | 37.92 | 76.87 | 47.07 | 56.36 |
| GRPO | 32 | **76.94** | 59.37 | 40.00 | **86.88** | **51.34** | **62.91** |
| DAPO | 32 | 72.71 | 46.67 | 35.21 | 83.13 | 46.77 | 56.90 |

### Sensitivity to Group Size

**consistently achieves superior sample efficiency, reaching high-tier performance at small group sizes.** The performance of group-based reinforcement learning methods is fundamentally tied to the group size $G$, which governs the fidelity of the advantage signal. Theoretically, standard GRPO computes advantage by normalizing rewards against the empirical mean and standard deviation of the group. In regimes where $G$ is small, this estimation is prone to high variance based on noise. addresses this by grounding the advantage in topic-level evidence, which provides a more stable semantic anchor for policy updates even when few responses are sampled per prompt. To evaluate this stability, we compare , GRPO, and DAPO across $G $ on the Qwen3-8B model. The results are detailed in Table §tab:group_size_results.

The scaling behavior across $G=4$ to $G=32$ reveals a critical trend: is remarkably sample-efficient. While standard GRPO and DAPO show significant performance gaps at smaller group sizes (e.g., $G=8$), achieves high-tier performance even with limited samples. Notably, at $G=8$, outperforms GRPO by 11.28% on average, suggesting that topic-based evidence gathering effectively "densifies" the sparse rewards found in smaller groups. 

As $G$ increases, the performance of the baselines improves as their empirical statistics become more reliable. For , the most substantial gains are realized at lower $G$ values, effectively reaching a performance plateau earlier than the baselines. This indicates that is well-suited for training environments with computational constraints, providing a robust training signal where traditional group-based normalization struggles with high variance.

### Curriculum Learning with Difficulty Clustering

**Curriculum learning provides a more stable training signal for , consistently outperforming GRPO.**

To evaluate the impact of data ordering, we train various models for one epoch on a dataset re-ranked from easy to hard difficulty with a group size $G=4$. This structured exposure, referred to as -diff, aligns the complexity of the topic-level evidence clusters with the model's evolving capability. By starting with simpler samples, the model establishes a reliable logical foundation before being exposed to highly intricate reasoning chains. The comparative results are presented in Figure §fig:curriculum_results.

**Figure:** *Performance comparison under difficulty-based curriculum learning.* () _(image: figs/curriculum_results_filtered_bold.pdf)_

As illustrated in Figure §fig:curriculum_results, -diff shows a clear performance advantage over the GRPO baseline on high-difficulty benchmarks. Specifically, for Qwen3-8B, -diff outperforms GRPO-diff by 3.95% on AIME24 and 6.04% on AIME25; a consistent lead is also observed for Qwen3-14B on AIME and OlympiadBench. This performance gap suggests that while standard group-based normalization suffices for conventional tasks like MATH and AMC23, grounding advantage estimation in stabilized evidence clusters is significantly more effective for the sparse reward landscapes of elite-level competition. By using a difficulty-based curriculum to isolate strong logical rationales early in training, -diff reduces evidence noise and enables more effective transfer to complex, multi-step reasoning needed for competition-level problems.
% — END arxiv/4_experiments —

% — BEGIN arxiv/6_conclusion —

## Conclusion

In this paper, we presented , a framework that enhances policy optimization by utilizing an Empirical Bayes approach to estimate reward mean and variance across clustered logical evidence. By replacing standard group-wide normalization with evidence-grounded statistics, significantly improves sample efficiency. Furthermore, our difficulty-based curriculum stabilizes the estimation of these empirical priors, leading to substantial performance gains on reasoning benchmarks. Ultimately, demonstrates that grounding reinforcement learning in shared semantic rationales provides a theoretically robust and computationally efficient path to elite-level mathematical reasoning.

% — END arxiv/6_conclusion —




% — BEGIN arxiv/7_appendix —

## Hyperparameter Details

We implement the framework and all baselines using the `verl` package. For optimization, we utilize a mini-batch size of 128 per backpropagation step, yielding a total global batch size of 512. The learning rate is fixed at $1 10^{-6}$, with a KL-divergence coefficient ($$) set to $0.001$. For the specific hyperparameters associated with the DAPO baseline, we adhere strictly to the configurations established in the original study [yu2025dapo].

To evaluate the impact of difficulty-based curriculum learning, we train for a single epoch without data shuffling to maintain the easy-to-hard ordering, utilizing the final checkpoint for performance assessment. For topic-level clustering experiments, training is conducted for up to 1,000 steps. In this setting, we select the optimal checkpoint based on the highest performance achieved on a held-out validation set, evaluated using a majority vote over 16 rollouts. All baseline comparisons are conducted under identical hyperparameter configurations to ensure a fair evaluation.

To quantify the difficulty of each task relative to the base model, we perform sixteen independent rollouts per prompt using a sampling temperature of $T=1.0$. The difficulty metric for each question is defined as the empirical pass rate, calculated as the percentage of correct trajectories generated by the base model. 

For semantic topic categorization, we establish nine fixed domains within the mathematical dataset: *Algebra & Number Theory*, *Calculus & Analysis*, *Geometry & Trigonometry*, *Discrete Mathematics*, *Probability & Statistics*, *Linear Algebra*, *Applied Mathematics & Word Problems*, *Physics & Chemistry*, and *Advanced Mathematics*. We leverage GPT-4.1 [achiam2023gpt] to classify each question into exactly one of these categories. This high-level semantic labeling facilitates the topic-level evidence aggregation required for the Empirical Bayes advantage estimation in the framework.

## Supplementary Materials

### Ablation Study: The Impact of Topic Clustering

To isolate the contribution of semantic topic clustering to the overall performance of , we compare the proposed -topic against a baseline variant, -naive. In the -naive setting, the Empirical Bayes shrinkage mechanism is applied to standard, randomly shuffled training batches where prompts are likely to be semantically heterogeneous. This comparison allows us to verify if "borrowing strength" is a general statistical benefit or if its efficacy is tied to the semantic consistency of the evidence clusters.

**Table:** *Ablation results comparing topic-clustered Empirical Bayes (-topic) against naive, shuffled-batch Empirical Bayes (-naive). Results are reported as Pass@1 (%).*

| lccccc
 **Method** | **MATH-500** | **AIME24** | **AIME25** | **AMC23** | **OlympiadBench** |
| — | — | — | — | — | — |
| -topic | **76.80** | **56.04** | **47.92** | 86.25 | **54.93** |
| -naive | 68.58 | 53.75 | 41.88 | **87.34** | 48.44 |
| -topic | 71.01 | **58.13** | **45.63** | 85.47 | **48.52** |
| -naive | **72.08** | 52.29 | 40.21 | **86.56** | 48.41 |

The results in Table §tab:ablation_clustering demonstrate that semantic clustering is a vital prerequisite for effective Empirical Bayes regularization. Across both model scales, -topic consistently outperforms the naive version on high-difficulty reasoning benchmarks like AIME and OlympiadBench. For instance, in the Qwen3-8B model, topic clustering improves the AIME25 score by 6.04%. 

The failure of -naive to reach similar heights can be attributed to the nature of the global prior. When a batch contains unrelated topics, the estimated prior mean $_{}$ and variance $^2$ represent a broad, high-entropy distribution of rewards that may not accurately describe the latent reward potential of any specific prompt. Consequently, the shrinkage estimator $V_q^{EB}$ pulls the local mean $_{}$ toward a noisy center, potentially suppressing valid gradient signals or introducing excessive bias. By contrast, clustering by topic ensures that the prior $_{}$ is derived from semantically related tasks, providing a much sharper and more relevant baseline for advantage regularization.

"`
[tb]
 
 

"`
[1]
  Policy $_$, Reference policy $_{}$, Group size $G$
 Initialize Welford statistics $_{}$ and $_{}$
 
 Sample batch of prompts ${q_1, ..., q_M} $
  $M$}
 Sample $G$ responses ${o_{m,1}, ..., o_{m,G}} _(| q_m)$
 Compute rewards $r_{m,i} $ and $_{, m} = {G}_{i=1}^G r_{m,i}$
 Update $_{}$ with all ${r_{m,i}}_{i=1}^G$
 Update $_{}$ with $_{, m}$
 $_{} (_{})$
 $^2 (_{})$
 $^2 (_{})$

  $M$}
 $_{q_m} {^2 / G + ^2}$ 
 $V_{q_m}^{EB} (1 - _{q_m}) _{, m} + _{q_m} _{}$ 
  $G$}
 $_{m,i}^{} r_{m,i} - V_{q_m}^{EB}$ 
 $_{} ({_{m,i}^{}})$
$_{} ({_{m,i}^{}})$

 $M$}
  $G$}
 $_{m,i} _{m,i}^{} - _{}}{_{} + }$ 
 $ _{m,i}$
 "`

"`

## Proof Details

> **Proof:** [Proof of Theorem §thm:non_vanishing]

**1. GRPO Case:**
If $r_i = 0$ for all $i$, then the group mean is $_{} = {G}0 = 0$.
The raw advantage for any response $o_i$ is:

$$ ^{raw}_{}(o_i) = r_i - _{} = 0 - 0 = 0 $$

Consequently, the gradient update $J = [ ] = 0$. The model receives no learning signal.

**2. EBPO Case:**
With $_{} = 0$, the EBPO baseline becomes:

$$ V^{} = (1 - )(0) + _{} = _{} $$

The raw advantage for any response $o_i$ is:

$$ ^{raw}_{}(o_i) = r_i - V^{} = 0 - _{} = -_{} $$

Since $ > 0$ and $_{} > 0$ (assuming the policy has succeeded at least once in history), we have $^{raw}_{} < 0$. This yields a non-zero negative gradient, penalizing the generation of incorrect responses even when no correct response is present in the group.

> **Proof:** [Proof of Theorem §thm:mse_stability]
The Mean Squared Error of an estimator $$ is defined as $[( - _q)^2]$.
For the GRPO baseline (Sample Mean):

$$ (V^{}) = (_{}) = {G} $$

For the EBPO baseline (Shrinkage Estimator), we seek to minimize the Bayes Risk (Expected MSE). The optimal linear estimator of the form $V = (1-w)_{} + w_{}$ minimizes:

$$ (w) = _{, r} [((1-w)_{} + w_{} - _q)^2] $$

Standard Bayesian derivation shows the optimal weight $w^*$ (which corresponds to our shrinkage factor $$) is:

$$  = (_{})}{(_{}) + (_q)} = {^2/G + ^2} $$

Substituting $$ back into the MSE equation yields:

$$ (V^{}) = (1 - ) {G} $$

Since $0 <  < 1$ (provided $^2 > 0$), it follows that:

$$ (V^{}) < {G} = (V^{}) $$

Thus, EBPO provides a more stable, lower-variance estimate of the prompt's difficulty, stabilizing the advantage computation when $G$ is small.

> **Proof:** [Proof of Corollary §thm:global_context]
From the result of Theorem §thm:non_vanishing, the raw advantage for a failure in a saturated group ($r_i=0, i$) is:

$$ ^{raw}_{}(fail) = -_{} $$

Let us compare two hypothetical regimes represented by the global history:

1. **Easy Regime:** The policy generally succeeds, so $_{} 1$.

Taking the magnitude of the penalty (gradient signal):

$$ |^{raw}_{}| (1)  |^{raw}_{}| () $$

Clearly, $|^{raw}_{}| |^{raw}_{}|$. 
EBPO uses the global context to differentiate these scenarios: it applies a strong correction signal when the model fails a task that is statistically "easy" (deviating significantly from the prior), but applies a gentle correction when the model fails a task that is "hard" (consistent with the prior), thereby preventing catastrophic forgetting or instability on difficult tasks.

> **Proof:** [Proof of Proposition §thm:entropy_conservation]
We invoke the result from cui2025entropy [cui2025entropy], which establishes that the entropy decay is dominated by the covariance between the policy likelihood and the estimated advantage:

$$
 H() _{q } [ _{o} _{}( _(o|q), (o, q) ) ]
$$

We analyze the sensitivity of the entropy update to the advantage normalization schemes. Let $_q = _{}((o|q), r(o))$ denote the alignment between the policy's log-likelihood and the raw rewards for a specific group $q$.

In GRPO, the advantage is normalized by the local group statistics. The advantage is $_{}(o) = {_q}$. Substituting this into the entropy update:

$$
 H_{} &_{q} [ ( , {_q} ) ] \\
 &= _{q} [ {_q} ( , r - _q ) ] \\
 &= _{q} [ {_q} ] \\
 & 
$$

This update is sensitive to $1/_q$. For "saturated" prompts where responses are highly similar, $_q 0$, causing the update magnitude to diverge.

In EBPO, the raw advantage is $A^{} = r(o) - V^{EBPO}_q$. The final advantage is standardized over the entire batch:

$$
 _{}(o) = } - _{}}{_{}} = _q - _{}}{_{}}
$$

where $_{}$ and $_{}$ are the mean and standard deviation of the raw advantages across the entire batch.
Substituting this into the entropy update:

$$
 H_{} &_{q} [ ( , _q - _{}}{_{}} ) ] \\
 &= {_{}} _{q} [ ( , r - _q + _{})}_{ o} ) ]
$$

By the translation invariance of covariance ($(X, Y - C) = (X, Y)$), the baseline $V^{EBPO}_q$ and the batch mean $_{}$ vanish from the covariance term:

$$
 H_{} = {_{}} _{q} [ _q ]
$$

Let $_q$ be the correlation between the policy's log-likelihood and the rewards, and let $_{}$ be the standard deviation of the log-likelihoods. We can decompose the covariance term as:

$$
 _q = (, r) = _q _{} _q
$$

Substituting this into the entropy update equations:

- **For GRPO:** The local standard deviation $_q$ cancels out:
- **For EBPO:** The update is scaled by the global batch deviation:

We now prove that the entropy decay under EBPO is bounded by that of GRPO:

$$
 _q [ _q _{} ( {_{}} ) ] _q [_q _{}]
$$

Let $X_q = _q _{}$. It suffices to show:

$$
 _q [X_q _q]}{_q [X_q]} _{}
$$

Applying the Cauchy-Schwarz inequality to the numerator:

$$
 _q [X_q _q] _q [X_q^2]} _q [_q^2]}
$$

By the Law of Total Variance, the global batch standard deviation is:

$$
 _{} = _q [_q^2] + _q(_q)} _q [_q^2]}
$$

**Step 3: Sufficient Condition (Independence).**
While the correlation between signal $X_q$ and noise $_q$ is complex, we invoke the standard independence assumption ($X_q _q$) to establish a baseline behavior. Under independence:

$$
 _q [X_q _q] = _q [X_q] _q [_q]
$$

The condition simplifies to $_q [_q] _{}$. By Lyapunov's inequality ($[Z^2]} [Z]$), we strictly have:

$$
 _{} _q [_q^2]} _q [_q]
$$

Thus, provided the batch exhibits heterogeneity ($_q(_q) > 0$), the EBPO update is strictly bounded:

$$
 H_{} _q [_q]}{_{}} H_{} < H_{}
$$

This confirms that the global normalization in EBPO acts as a regularizer, scaling down updates relative to the total diversity of the training batch.

Since the batch variance $_{}^2$ includes the between-group variance $(_q)$, we have $_{} _q$ for the average group. Crucially, the term ${_{}}$ acts as a *dynamic damping factor*:

- When a group has low variance (low information), ${_{}} 0$, suppressing the entropy reduction.

Thus, $H_{} H_{}$, proving that EBPO conservatively regulates policy updates based on the information content (variance) of the prompt group.
Assuming the covariance term $_q$ is independent of the group scale $_q$, the expected entropy reduction is governed by the effective scaling factors:

$$ _q [ {_q} ]  {_{}} $$

By the Law of Total Variance, the global batch variance $_{}^2$ decomposes exactly into the expected within-group variance and the between-group variance:

$$
 _{}^2 = _q[_q^2] + _q(_q)
$$

Given that the task distribution is heterogeneous (i.e., prompts have varying difficulty), we have $_q(_q) > 0$. This implies a strict inequality for the second moments:

$$
 _{} = _q[_q^2] + _q(_q)} > _q[_q^2]} 
$$

Note that for any random variable $X$, $[X^2]} [X]$. Letting $X = _q$:

$$
 _q[_q^2]} _q[_q] 
$$

Combining (§step1) and (§step2):

$$
 _{} > _q[_q]
$$

Consider the function $f(x) = 1/x$, which is strictly convex for $x > 0$. By Jensen's Inequality:

$$
 _q [ {_q} ] {_q[_q]}
$$

Chaining the inequalities from above yields a strict ordering of the decay coefficients:

$$
 _q [ {_q} ] {_q[_q]} > {_{}}
$$

Consequently, the entropy reduction for GRPO strictly upper-bounds that of EBPO:

$$
 H_{} = _q [ {_q} ] > {_{}} _q [_q] = H_{}
$$

This proves that the global normalization in EBPO strictly enforces a more conservative policy update bound than GRPO, preventing the excessive entropy collapse associated with small-variance groups.

> **Proof:** [Proof of Proposition §prop:clustering]
We seek to minimize the expected squared error of the prior, defined as $J = _{q } [(_{} - _q)^2]$.

**Case 1: Random Shuffle (Topic-Agnostic)**
By Law of Large Numbers, the online estimator converges to the global expectation over the entire dataset. Thus, $_{} = _{}$. The loss function becomes the total variance of the difficulty distribution:

$$
 J_{} = _{q} [(_{} - _q)^2] = (_q)
$$

**Case 2: Topic-Coherent (Topic-Conditional)**
In the idealized topic-coherent regime (assuming the Welford update is sufficiently fast to adapt to the topic shift), the estimator converges to the conditional expectation of the current cluster. For a prompt $q _k$, the prior is $_{} = _k$. The loss function is the expected variance within each cluster:

$$
 J_{} = _{k} [ _{q _k} [(_k - _q)^2] ] = _k [(_q k)]
$$

The total variance can be decomposed into the expected conditional variance (within-group) and the variance of the conditional expectations (between-group):

$$
 (_q) = _k [(_q k)] + _k ([_q k])
$$

Substituting our loss terms:

$$
 J_{} = J_{} + _k(_k)
$$

Since the topics have distinct mean difficulties by definition, the between-topic variance is strictly positive: $_k(_k) > 0$.
Therefore:

$$
 J_{} < J_{}
$$

The topic-coherent ordering strictly reduces the estimation error of the prior, thereby improving the accuracy of the EBPO baseline.

% — END arxiv/7_appendix —