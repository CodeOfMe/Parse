# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

> **arXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)
> **TeX source:** [arXiv-2402.03300v1/](arXiv-2402.03300v1/)
> **PDF:** [DeepSeekMath-arXiv-2402.03300v1.pdf](DeepSeekMath-arXiv-2402.03300v1.pdf)

---

{UTF8}{gbsn}

**Figure:** * Top\@1 accuracy of open-source models on the competition-level MATH benchmark [MATH* _(image: figures/Math.png)_

 for training, configuring the vector dimension to 256, learning rate to 0.1, the maximum length of word n-gram to 3, the minimum number of word occurrences to 3, and the number of training epochs to 3.
To reduce the size of the original Common Crawl, we employ URL-based deduplication and near-deduplication techniques, resulting in 40B HTML web pages.
We then recall mathematical web pages from deduplicated Common Crawl with the fastText model.
To filter out low-quality mathematical content, we rank the collected pages according to their scores predicted by the fastText model, and only preserve the top-ranking ones.
The volume of data preserved is assessed through pre-training experiments on the top 40B, 80B, 120B, and 160B tokens.
In the first iteration, we choose to keep the top 40B tokens.

After the first iteration of data collection, numerous mathematical web pages remain uncollected, mainly because the fastText model is trained on a set of positive examples that lacks sufficient diversity.
We therefore identify additional mathematical web sources to enrich the seed corpus, so that we can optimize the fastText model.
Specifically, we first organize the entire Common Crawl into disjoint domains;
a domain is defined as web pages sharing the same base URL.
For each domain, we calculate the percentage of web pages that are collected in the first iteration.
Domains where over 10% of the web pages have been collected are classified as math-related (e.g., mathoverflow.net).
Subsequently, we manually annotate the URLs associated with mathematical content within these identified domains (e.g., mathoverflow.net/questions).
Web pages linked to these URLs, yet uncollected, will be added to the seed corpus.
This approach enables us to gather more positive examples, thereby training an improved fastText model capable of recalling more mathematical data in the subsequent iteration.
After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B tokens.
In the fourth iteration, we notice that nearly 98% of the data has already been collected in the third iteration, so we decide to cease data collection.

To avoid benchmark contamination, we follow [deepseek-coder] to filter out web pages containing questions or answers from English mathematical benchmarks such as GSM8K [gsm8k] and MATH [MATH] and Chinese benchmarks such as CMATH [wei2023cmath] and AGIEval [agieval].
The filtering criteria are as follows: any text segment containing a 10-gram string that matches exactly with any sub-string from the evaluation benchmarks is removed from our math training corpus.
For benchmark texts that are shorter than 10 grams but have at least 3 grams, we employ exact matching to filter out contaminated web pages.

### Validating the Quality of the Corpus

We run pre-training experiments to investigate how the Corpus is compared with the recently released math-training corpora:

- **MathPile** [mathpile]: a multi-source corpus (8.9B tokens) aggregated from textbooks, Wikipedia, ProofWiki, CommonCrawl, StackExchange, and arXiv, with the majority (over 85%) sourced from arXiv;
- **OpenWebMath** [openwebmath]: CommonCrawl data filtered for mathematical content, totaling 13.6B tokens;

#### Training Setting

We apply math training to a general pre-trained language model with 1.3B parameters, which shares the same framework as the DeepSeek LLMs [deepseek-llm], denoted as DeepSeek-LLM 1.3B.
We separately train a model on each mathematical corpus for 150B tokens. All experiments are conducted using the efficient and light-weight HAI-LLM [haillm] training framework.
Following the training practice of DeepSeek LLMs, we use the AdamW optimizer [adamW] with $_1=0.9$, $_2=0.95$, and $=0.1$, along with a multi-step learning rate schedule where the learning rate reaches the peak after 2,000 warmup steps, decreases to its 31.6% after 80% of the training process, and further decreases to 10.0% of the peak after 90% of the training process.
We set the maximum value of learning rate to 5.3e-4, and use a batch size of 4M tokens with a 4K context length.

{3pt} 
 

**Table:** *
 Performance of DeepSeek-LLM 1.3B trained on different mathematical corpora, evaluated using few-shot chain-of-thought prompting.
 Corpus sizes are calculated using our tokenizer with a vocabulary size of 100K.
 *

| llcccccccc 
3*Math Corpus | 3*Size | 5cEnglish Benchmarks | 3cChinese Benchmarks |
| — | — | — | — |
| STEM |

**Figure:** * Benchmark curves of DeepSeek-LLM 1.3B trained on different mathematical corpora.* () _(image: figures/corpus_comparisons.png)_

#### Evaluation Results

**The Corpus is of high quality, covers multilingual mathematical content, and is the largest in size.**

- **High-quality**:
 We evaluate downstream performance on 8 mathematical benchmarks using few-shot chain-of-thought prompting [cot].
 As shown in Table §tab:corpora_comparison, there is a clear performance lead of the model trained on the Corpus. Figure §fig:corpora_comparisons shows that the model trained on the DeepSeekMath Corpus demonstrates better performance than Proof-Pile-2 at 50B tokens (1 full epoch of Proof-Pile-2), indicating the average quality of DeepSeekMath Corpus is higher.
- **Multilingual**:
 The Corpus encompasses data in multiple languages, predominantly featuring English and Chinese as the two most represented languages.
 As shown in Table §tab:corpora_comparison, training on the Corpus enhances mathematical reasoning performance in both English and Chinese.
 In contrast, existing mathematical corpora, which are primarily English-centric, show limited improvement and may even hinder performance in Chinese mathematical reasoning.

### Training and Evaluating -Base 7B

In this section, we introduce -Base 7B, a base model with strong reasoning abilities, especially in mathematics.
Our model is initialized with DeepSeek-Coder-Base-v1.5 7B [deepseek-coder] and trained for 500B tokens. The distribution of the data is as follows: 56% is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github code, and the remaining 10% is natural language data from Common Crawl in both English and Chinese.
We mainly adopt the training setting specified in Section §sec:quality-policy, except that we set the maximum value of the learning rate to 4.2e-4 and use a batch size of 10M tokens.

We conduct a comprehensive assessment of the mathematical capabilities of -Base 7B, focusing on its ability to produce self-contained mathematical solutions without relying on external tools, solve mathematical problems using tools, and conduct formal theorem proving.
Beyond mathematics, we also provide a more general profile of the base model, including its performance of natural language understanding, reasoning, and programming skills.

**Mathematical Problem Solving with Step-by-Step Reasoning.** 

We evaluate -Base's performance of solving mathematical problems using few-shot chain-of-thought prompting [cot], across eight benchmarks in English and Chinese.
These benchmarks encompass quantitative reasoning (e.g., GSM8K [gsm8k], MATH [MATH], and CMATH [wei2023cmath]) and multiple-choice problems (e.g., MMLU-STEM [mmlu] and Gaokao-MathQA [agieval]), covering diverse fields of mathematics from elementary to college-level complexity.

As shown in Table §tab:base_math_cot, -Base 7B leads in performance across all eight benchmarks among the open-source base models (including the widely-used general model Mistral 7B [mistral] and the recently released Llemma 34B [llemma] which underwent math training on Proof-Pile-2 [llemma]).
Notably, on the competition-level MATH dataset, -Base surpasses existing open-source base models by over 10% absolute, and outperforms Minerva 540B [minerva], a closed-source base model 77 times larger which builds on PaLM [palm] and is further trained on mathematical texts.

{3pt} 
 

**Table:** *
 Comparisons between -Base 7B and strong base models on English and Chinese mathematical benchmarks.
 Models are evaluated with chain-of-thought prompting.
 Minerva results are quoted from [minerva*

| llcccccccc 
1l3*Model | 1l3*Size | 5cEnglish Benchmarks | 3cChinese Benchmarks |
| — | — | — | — |
| STEM | CMATH | |
| MathCloze | |
| MathQA |
| Minerva | 62B | 52.4% | 27.6% | 12.0% | - | 53.9% | - | - | - |
| Minerva | 540B | 58.8% | 33.6% | 17.6% | - | 63.9% | - | - | - |
| Llemma | 34B | 54.0% | 25.3% | 10.3% | 71.9% | 52.9% | 56.1% | 11.9% | 26.2% |

**Mathematical Problem Solving with Tool Use.** 
We evaluate program-aided mathematical reasoning on GSM8K and MATH using few-shot program-of-thought prompting [pot,pal].
Models are prompted to solve each problem by writing a Python program where libraries such as *math* and *sympy* can be utilized for intricate computations.
The execution result of the program is evaluated as the answer.
As shown in Table §tab:base_math_pot_proof], -Base 7B outperforms the prior state-of-the-art Llemma 34B.

 
 

**Table:** *
 Few-shot evaluation of base models' ability to solve mathematical problems using tools and the ability to conduct informal-to-formal theorem proving in Isabelle.
 *

| llcccc 
1l2*Model | 1l2*Size | 2cProblem Solving w/ Tools | 2cInformal-to-Formal Proving |
| — | — | — | — |
| CodeLlama | 34B | 52.7% | 23.5% | 18.5% | 18.0% |
| Llemma | 34B | 64.6% | 26.3% | 21.0% | 21.3% |

**Formal Mathematics.** 

Formal proof automation is beneficial to ensure the accuracy and reliability of mathematical proofs and enhance efficiency, with increasing attention in recent years.
We evaluate -Base 7B on the task of informal-to-formal proving from [dsp_proof] which is to generate a formal proof based on an informal statement, a formal counterpart of the statement, and an informal proof.
We evaluate on miniF2F [minif2f], a benchmark for formal Olympiad-level mathematics, and generate a formal proof in Isabelle for each problem with few-shot prompting.
Following [dsp_proof], we leverage models to generate proof sketches, and execute the off-the-shelf automated prover Sledgehammer [sledgehammer] to fill in the missing details.
As shown in Table §tab:base_math_pot_proof, -Base 7B demonstrates strong performance in proof autoformalization.

{3pt} 
 

**Table:** *
 Evaluation on natural language understanding, reasoning, and code benchmarks.
 DeepSeek-Coder-Base-v1.5{3pt} 
 

**Table:** *
 Performance of Open- and Closed-Source models with both Chain-of-Thought and Tool-Integrated Reasoning on English and Chinese Benchmarks.
 Scores in gray denote majority votes with 32 candidates; The others are Top1 scores.
 -RL 7B beats all open-source models from 7B to 70B, as well as the majority of closed-source models. Although -RL 7B is only further trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, it improves over -Instruct 7B on all benchmarks.
 *

| llcccc 
1l2*Model | 1l2*Size | 2cEnglish Benchmarks | 2cChinese Benchmarks |
| — | — | — | — |
| GPT-4 | - | 92.0% | 52.9% | - | 86.0% |
| Inflection-2 | - | 81.4% | 34.8% | - | - |
| GPT-3.5 | - | 80.8% | 34.1% | - | 73.8% |
| Gemini Pro | - | 86.5% | 32.6% | - | - |
| Grok-1 | - | 62.9% | 23.9% | - | - |
| GLM-4 | - | 87.6% | 47.9% | - | - |
| Qwen | 72B | 78.9% | 35.2% | - | - |
| Math-Shepherd-Mistral | 7B | 84.1% | 33.0% | - | - |
| WizardMath-v1.1 | 7B | 83.2% | 33.0% | - | - |
| DeepSeek-LLM-Chat | 67B | 84.1% | 32.6% | 74.0% | 80.3% |
| MetaMath | 70B | 82.3% | 26.6% | 66.4% | 70.9% |
| SeaLLM-v2 | 7B | 78.2% | 27.5% | 64.8% | - |
| ChatGLM3 | 6B | 72.3% | 25.7% | - | - |
| WizardMath-v1.0 | 70B | 81.6% | 22.7% | 64.8% | 65.4% |
| **-RL** | 7B | **88.2%** | **51.7%** | **79.6%** | **88.8%** |
| DeepSeek-LLM-Chat | 67B | 86.7% | 51.1% | 76.4% | 85.4% |
| ToRA | 34B | 80.7% | 50.8% | 41.2% | 53.4% |
| MAmmoTH | 70B | 76.9% | 41.8% | - | - |
| **-RL** | 7B | **86.7%** | **58.8%** | **78.4%** | **87.6%** |

As shown in Table §tab:sft_rl_math, under the evaluation setting where tool use is disallowed, -Instruct 7B demonstrates strong performance of step-by-step reasoning.
Notably, on the competition-level MATH dataset, our model surpasses all open-source models and the majority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute.
This is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifically enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B).
While -Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH, it still underperforms GPT-4 and Gemini Ultra.

Under the evaluation setting where models are allowed to integrate natural language reasoning and program-based tool use for problem solving, -Instruct 7B approaches an accuracy of 60% on MATH, surpassing all existing open-source models.
On the other benchmarks, our model is competitive with DeepSeek-LLM-Chat 67B, the prior state-of-the-art that is 10 times larger.

## Reinforcement Learning

### Group Relative Policy Optimization

Reinforcement learning (RL) has been proven to be effective in further improving the mathematical reasoning ability of LLMs after the Supervised Fine-Tuning (SFT) stage [wang2023math,wizardmath].
In this section, we introduce our efficient and effective RL algorithm, Group Relative Policy Optimization (GRPO).

#### From PPO to GRPO
 Proximal Policy Optimization (PPO) [schulman2017proximal] is an actor-critic RL algorithm that is widely used in the RL fine-tuning stage of LLMs [ouyang2022training]. In particular, it optimizes LLMs by maximizing the following surrogate objective:

$$
_{PPO}() = {[q P(Q), o _{_{old}}(O|q)]} {|o|} _{t=1}^{|o|} [  | q, o_{<t})}{_{_{old}}(o_{t} | q, o_{<t})} A_{t},  (  | q, o_{<t})}{_{_{old}}(o_{t} | q, o_{<t})}, 1 - , 1 + ) A_{t} ] ,
$$

where $_{}$ and $_{_{old}}$ are the current and old policy models, and $q, o$ are questions and outputs sampled from the question dataset and the old policy $_{_{old}}$, respectively. $$ is a clipping-related hyper-parameter introduced in PPO for stabilizing training. $A_t$ is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) [gae], based on the rewards ${r_{t}}$ and a learned value function $V_{}$. Thus, in PPO, a value function needs to be trained alongside the policy model and to mitigate over-optimization of the reward model, the standard approach is to add a per-token KL penalty from a reference model in the reward at each token [ouyang2022training], i.e., 

$$
 r_{t} = r_(q, o_{t}) - (o_{t}|q, o_{<t})}{_{ref}(o_{t}|q, o_{<t})},

$$

where $r_$ is the reward model, $_{ref}$ is the reference model, which is usually the initial SFT model, and $$ is the coefficient of the KL penalty.

**Figure:** *Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources.* () _(image: figures/GRPO.pdf)_

As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction. While in the LLM context, usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, as shown in Figure §fig:grpo, we propose Group Relative Policy Optimization (GRPO), which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. More specifically, for each question $q$, GRPO samples a group of outputs ${o_1, o_2, ..., o_G}$ from the old policy $_{_{old}}$ and then optimizes the policy model by maximizing the following objective:

$$

 _{GRPO}() &= {[q P(Q), {o_i}_{i=1}^G _{_{old}}(O|q)]} \\
 & {G}_{i=1}^G{|o_i|} _{t=1}^{|o_i|}  | q, o_{i,<t})}{_{_{old}}(o_{i,t} | q, o_{i,<t})} _{i,t},  (  | q, o_{i,<t})}{_{_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - , 1 + ) _{i,t} ] - _{KL}[_{} || _{ref}]} ,


$$

where $$ and $$ are hyper-parameters, and $_{i,t}$ is the advantage calculated based on relative rewards of the outputs inside each group only, which will be detailed in the following subsections. The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models, as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss, avoiding complicating the calculation of $_{i,t}$. And different from the KL penalty term used in (§eq:PPO-reward), we estimate the KL divergence with the following unbiased estimator [kl_approx]: 

$$
_{KL}[_{} || _{ref}] = (o_{i,t}|q,o_{i,<t})}{_{}(o_{i,t}|q,o_{i,<t})}- (o_{i,t}|q,o_{i,<t})}{_{}(o_{i,t}|q,o_{i,<t})} - 1,
$$

which is guaranteed to be positive. 

"`
[t]
 
 **Input** initial policy model $_{_{}}$; reward models $r_$; task prompts $$; 
 hyperparameters $$, $$, $$
 
"`
[1]
 policy model $__{_{}}$
 
 reference model $_{ref} _{}$
 
 Sample a batch $_b$ from $$
 Update the old policy model $_{_{old}} _{}$ 
 Sample $G$ outputs ${o_i}_{i=1}^G _{_{old}} (q) $ for each question $q _b$
 Compute rewards ${r_i}_{i=1}^{G}$ for each sampled output $o_i$ by running $r_{}$ 
 Compute $_{i,t}$ for the $t$-th token of $o_i$ through group relative advantage estimation.
 
 Update the policy model $_{}$ by maximizing the GRPO objective (Equation §eq:GC-GRPO)
 Update $r_$ through continuous training using a replay mechanism. 
 "`

 **Output** $_$
 

"`

#### Outcome Supervision RL with GRPO
 
Formally, for each question $q$, a group of outputs ${o_1, o_2, ..., o_G}$ are sampled from the old policy model $_{_{old}}$. A reward model is then used to score the outputs, yielding $G$ rewards $={r_1, r_2, ..., r_G}$ correspondingly. Subsequently, these rewards are normalized by subtracting the group average and dividing by the group standard deviation. Outcome supervision provides the normalized reward at the end of each output $o_i$ and sets the advantages $_{i, t}$ of all tokens in the output as the normalized reward, i.e., $_{i, t} = _i = ()}{{std}()}$, and then optimizes the policy by maximizing the objective defined in equation (§eq:GRPO-obj).

#### Process Supervision RL with GRPO
 
Outcome supervision only provides a reward at the end of each output, which may not be sufficient and efficient to supervise the policy in complex mathematical tasks. Following [wang2023math], we also explore process supervision, which provides a reward at the end of each reasoning step. Formally, given the question $q$ and $G$ sampled outputs ${o_1, o_2, ..., o_G}$, a process reward model is used to score each step of the outputs, yielding corresponding rewards: $ = { {r_1^{index(1)},...,r_1^{index(K_1)}}, ..., {r_G^{index(1)},...,r_G^{index(K_G)}} }$, where $index(j)$ is the end token index of the $j$-th step, and $K_i$ is the total number of steps in the $i$-th output. We also normalize these rewards with the average and the standard deviation, i.e., $_i^{index(j)} =  - {mean()}}{{std()}}$.
Subsequently, the process supervision calculates the advantage of each token as the sum of the normalized rewards from the following steps, i.e., $_{i, t} = _{index(j) t} _i^{index(j)}$,
and then optimizes the policy by maximizing the objective defined in equation (§eq:GRPO-obj).

#### Iterative RL with GRPO

As the reinforcement learning training process progresses, the old reward model may not be sufficient to supervise the current policy model.
Therefore, we also explore the iterative RL with GRPO.
As shown in Algorithm §alg:iter-grpo, in iterative GRPO, we generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a replay mechanism that incorporates 10% of historical data.
Then, we set the reference model as the policy model, and continually train the policy model with the new reward model.

### Training and Evaluating -RL

We conduct RL based on -Instruct 7B.
The training data of RL are chain-of-thought-format questions related to GSM8K and MATH from the SFT data, which consists of around 144K questions. 
We exclude other SFT questions to investigate the impact of RL on benchmarks that lack data throughout the RL phase.
We construct the training set of reward models following [wang2023math].
We train our initial reward model based on the -Base 7B with a learning rate of 2e-5.
For GRPO, we set the learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample $64$ outputs. The max length is set to 1024, and the training batch size is 1024.
The policy model only has a single update following each
exploration stage.
We evaluate -RL 7B on benchmarks following -Instruct 7B.
For -RL 7B, GSM8K and MATH with chain-of-thought reasoning can be regarded as in-domain tasks and all the other benchmarks can be regarded as out-of-domain tasks.

Table §tab:sft_rl_math demonstrates the performance of open- and closed-source models with both chain-of-thought and tool-integrated reasoning on English and Chinese benchmarks. We find that:
1) -RL 7B attains accuracies of 88.2% and 51.7% on GSM8K and MATH, respectively, utilizing chain-of-thought reasoning. This performance surpasses that of all open-source models in the 7B to 70B range, as well as the majority of closed-source models. 
2) Crucially, -RL 7B is only trained on chain-of-thought-format instruction tuning data of GSM8K and MATH, starting from -Instruct 7B. Despite the constrained scope of its training data, it outperforms -Instruct 7B across all evaluation metrics, showcasing the effectiveness of reinforcement learning.

## Discussion

In this section, we will share our findings in pre-training and RL experiments. 

### Lessons Learnt in Pre-Training

We first share our experience in pre-training. Unless otherwise specified, we will adhere to the training settings outlined in Section §sec:quality-policy. It is worth noting that, when referring to the DeepSeekMath Corpus in this section, we use an 89B-token dataset from the second iteration of the data collection process.

#### Code Training Benefits Mathematical Reasoning

A popular yet unverified hypothesis suggests that code training improves reasoning.
We attempt to offer a partial response to this, particularly within the mathematical domain:
code training improves models' ability to do mathematical reasoning both with and without tool use.

To study how code training affects mathematical reasoning, we experimented with the following two-stage training and one-stage training settings:

**Two-Stage Training** 

- **Code Training for 400B Tokens $$ Math Training for 150B Tokens**:
 We train DeepSeek-LLM 1.3B for 400B code tokens followed by 150B math tokens;

**One-Stage Training**

- **Math Training for 150B Tokens**:
 We train DeepSeek-LLM 1.3B for 150B math tokens;

{3pt} 
 

**Table:** *
 Investigation of how code affects mathematical reasoning under different training settings.
 We experiment with DeepSeek-LLM 1.3B, and evaluate its mathematical reasoning performance without and with tool use via few-shot chain-of-thought prompting and few-shot program-of-thought prompting, respectively.
 *

| llllccccc 
2*Training Setting | 3lTraining Tokens | 3cw/o Tool Use | 2cw/ Tool Use |
| — | — | — | — |
| Stage 2: Math Training | – | – | 150B | 19.1% | 14.4% | 37.2% | 14.3% | 6.7% |
| Stage 2: Math Training | – | – | 150B | **21.9%** | **15.3%** | **39.7%** | 17.4% | 9.4% |

{3pt} 
 

**Table:** *
 Investigation of how different settings of code and math training affect model performance of language understanding, reasoning, and coding.
 We experiment with DeepSeek-LLM 1.3B.
 We evaluate the models on MMLU and BBH using few-shot chain-of-thought prompting.
 On HumanEval and MBPP, we conduct zero-shot and few-shot evaluations, respectively.
 *

| llllcccc 
2*Training Setting | 3lTraining Tokens | 2*MMLU | 2*BBH | 2*HumanEval (Pass@1) | 2*MBPP (Pass@1) |
| — | — | — | — | — | — |
| Stage 2: Math Training | – | – | 150B | 33.1% | 32.7% | 12.8% | 13.2% |
| Stage 2: Math Training | – | – | 150B | **36.2%** | 35.3% | 12.2% | 17.0% |

**Results.** 
Table §tab:code-math and Table §tab:code-math-general-eval demonstrate the downstream performance under different training settings.

Code training benefits program-aided mathematical reasoning, both under the two-stage training and one-stage training settings.
As shown in Table §tab:code-math, under the two-stage training setting, code training alone already significantly enhances the ability to solve GSM8K and MATH problems using Python.
Math training in the second stage yields further improvements.
Interestingly, under the one-stage training setting, mixing code tokens and math tokens effectively mitigates the issue of catastrophic forgetting that arises from two-stage training, and also synergizes coding (Table §tab:code-math-general-eval) and program-aided mathematical reasoning (Table §tab:code-math).

Code training also improves mathematical reasoning without tool use.
Under the two-stage training setting, the initial stage of code training already results in moderate enhancements.
It also boosts the efficiency of the subsequent math training, eventually leading to the best performance.
However, combining code tokens and math tokens for one-stage training compromises mathematical reasoning without tool use.
One conjecture is that DeepSeek-LLM 1.3B, due to its limited scale, lacks the capacity to fully assimilate both code and mathematical data simultaneously.

{3pt} 
 

**Table:** *
 Effect of math training on different arXiv datasets.
 Model performance is evaluated with few-shot chain-of-thought prompting.
 *

| lllcccccccc 
1l3*Model | 1l3*Size | 3*ArXiv Corpus | 5cEnglish Benchmarks | 3cChinese Benchmarks |
| — | — | — | — | — |
| STEM | CMATH | |
| MathCloze | |
| MathQA |
| | | ArXiv-RedPajama | 3.3% | 3.4% | 4.0% | 9.4% | 9.0% | 7.4% | 0.8% | 2.3% |
| | | ArXiv-RedPajama | 28.1% | 11.1% | 7.7% | 50.0% | 35.2% | 42.6% | 7.6% | 24.8% |

{3pt} 
 

**Table:** *
 Effect of math training on different arXiv corpora, the base model being DeepSeek-Coder-Base-v1.5 7B.
 We evaluate informal-to-formal proving in Isabelle.
 *

| lcc 
ArXiv Corpus | miniF2F-valid | miniF2F-test |
| — | — | — |
| ArXiv-RedPajama | 14.8% | 11.9% |

#### ArXiv Papers Seem Ineffective in Improving Mathematical Reasoning

ArXiv papers are commonly included as a component of math pre-training data [minerva,gpt-f,llemma,mathpile].
However, detailed analysis regarding their impact on mathematical reasoning has not been extensively conducted.
Perhaps counter-intuitively, according to our experiments, arXiv papers seem ineffective in improving mathematical reasoning.
We experiment with models of different sizes, including DeepSeek-LLM 1.3B and DeepSeek-Coder-Base-v1.5 7B [deepseek-coder], using arXiv corpora that underwent varied processing pipelines:

- **MathPile** [mathpile]:
 an 8.9B-token corpus developed with cleaning and filtering heuristic rules, over 85% of which are scientific arXiv papers;

In our experiments, we separately train DeepSeek-LLM 1.3B for 150B tokens and DeepSeek-Coder-Base-v1.5 7B for 40B tokens on each arXiv corpus. It seems that arXiv papers are ineffective in improving mathematical reasoning.
When trained on a arXiv-only corpus, both models display no notable improvements or even deterioration across various mathematical benchmarks of different complexities employed in this study.
These benchmarks include quantitative reasoning datasets like GSM8K and MATH (Table §tab:arxiv-cot), multiple-choice challenges like MMLU-STEM (Table §tab:arxiv-cot), and formal mathematics like miniF2F (Table §tab:arxiv_atp).

However, this conclusion has its limitations and should be taken with a grain of salt.
We have not yet studied:

- The impact of arXiv tokens on specific math-related tasks not included in this research, such as informalization of theorems which is to convert formal statements or proofs to their informal versions;
- The effect of arXiv tokens when combined with other types of data;

Thus, further exploration is required, which we leave for future studies.

### Insights of Reinforcement Learning

#### Towards to a Unified Paradigm

In this section, we provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO, and further conduct experiments to explore the factors of the unified paradigm. 
Generally, the gradient with respect to the parameter $$ of a training method can be written as:

$$
 _{}_{}() = [}_{Data \ Source}]( {|o|} _{t=1}^{|o|} }}(q, o, t, _{{rf}})}_{Gradient \ Coefficient} _{}_{}(o_t | q, o_{<t})).

$$

There exist three key components: 
1) *Data Source $, which determines the training data;
2) *Reward Function $_{{rf*}$}, which is the source of the training reward signal;
3) *Algorithm $: which processes the training data and the reward signal to the gradient coefficient $GC$ that determines the magnitude of the penalty or reinforcement for the data. We analyze several representative methods based on such a unified paradigm:

**Table:** *The data source and gradient coefficient of different methods. $P_{sft*

| lccc
 **Methods** | **Data Source** | **Reward Function** | **Gradient Coefficient** |
| — | — | — | — |
| DPO | $q P_sft(Q)$, $o^+, o^- _sft(O|q)$ | Rule | Equation :GC-DPO |
| PPO | $q P_sft(Q)$, $o _(O|q)$ | Model | Equation :GC-PPO |
| GRPO | $q P_sft(Q)$, $_i_i=1^G _(O|q)$ | Model | Equation :GC-GRPO |

- **Supervised Fine-tuning (SFT)**: SFT fine-tunes pretrained model on human selected SFT data.
- **Rejection Sampling Fine-tuning (RFT)**: RFT further fine-tunes the SFT model on the filtered outputs sampled from the SFT model based on SFT questions. RFT filters the outputs based on the correctness of their answers.
- **Direct Preference Optimization (DPO)**: DPO further refines the SFT model by fine-tuning it on augmented outputs sampled from the SFT model, using pair-wise DPO loss.
- **Online Rejection Sampling Fine-tuning (Online RFT)**: Different from RFT, Online RFT initiates the policy model using the SFT model and refines it by fine-tuning with the augmented outputs sampled from the real-time policy model.

We summarize the components of these methods in Table §tab:data_policy.
Please refer to Appendix §app:analysis-rl for a more detailed derivation process.

**Figure:** *Performance of the -Instruct 1.3B model, which was further trained using various methods, on two benchmarks.* () _(image: figures/combined_figure_rl.pdf)_

**Figure:** *Performance of iterative reinforcement learning with -Instruct 7B on two benchmarks.* () _(image: figures/iter_rl.pdf)_

**Observation about Data Source.** 
We divide the data source into two categories, online sampling, and offline sampling.
Online sampling denotes that the training data is from the exploration results of the real-time training policy model, while offline sampling denotes that the training data is from the sampling results of the initial SFT model.
RFT and DPO follow the offline style, while Online RFT and GRPO follow the online style. 

As shown in Figure §fig:rl-analysis,
we find that the Online RFT significantly outperforms RFT on two benchmarks.
Specifically, Online RFT is comparable to RFT in the early stage of training but gains an absolute advantage in the later stage, demonstrating the superiority of online training. 
This is intuitive, as in the initial stage, the actor and the SFT model exhibit close resemblance, with the sampled data revealing only minor differences. In the later stage, however, the data sampled from the actor will exhibit more significant differences, and real-time data sampling will offer greater advantages.

**Observation about Gradient Coefficient.** 
The algorithm processes the reward signal to the gradient coefficient to update the model parameter.
We divide the reward function as `Rule' and `Model' in our experiments.
Rule refers to judging the quality of a response based on the correctness of the answer, and Model denotes that we train a reward model to score each response. The training data of the reward model is based on the rule judgment.
Equations §eq:GC-RFT and §eq:GC-GRPO highlight a key difference between GRPO and Online RFT: GRPO uniquely adjusts its gradient coefficient based on the reward value provided by the reward model. This allows for differential reinforcement and penalization of responses according to their varying magnitudes. In contrast, Online RFT lacks this feature; it does not penalize incorrect responses and uniformly reinforces all responses with correct answers at the same level of intensity. 

As demonstrated in Figure §fig:rl-analysis, GRPO surpasses online RFT, thereby highlighting the efficiency of altering positive and negative gradient coefficients. In addition, GRPO+PS shows superior performance compared to GRPO+OS, indicating the benefits of using fine-grained, step-aware gradient coefficients.
Furthermore, we explore the iterative RL, in our experiments, we conduct two rounds of iteration. As shown in Figure §fig:iter-rl, we notice that the iterative RL significantly improves the performance, especially at the first iteration.

**Figure:** *The Maj@K and Pass@K of SFT and RL 7B on GSM8K and MATH (temperature $0.7$). It was noted that RL enhances Maj@K but not Pass@K.* () _(image: figures/combined_MAJ_PASS.pdf)_

#### Why RL Works?

In this paper, we conduct reinforcement learning based on a subset of instruction tuning data, and it achieves significant performance enhancement upon the instruction tuning model. To further explain why reinforcement learning works.
We evaluate the Pass@K and Maj@K accuracy of the Instruct and RL models on two benchmarks.
As shown in Figure §fig:maj-pass, RL enhances Maj@K’s performance but not Pass@K. These findings indicate that RL enhances the model's overall performance by rendering the output distribution more robust, in other words, **it seems that the improvement is attributed to boosting the correct response from TopK rather than the enhancement of fundamental capabilities.** Similarly, [wang2023making] identified a **misalignment problem** in reasoning tasks within the SFT model, showing that the reasoning performance of SFT models can be improved through a series of preference alignment strategies [yuan2023rrhf,song2023preference,wang2023making].

#### How to Achieve More Effective RL?

We demonstrate RL works pretty well in mathematical reasoning tasks. We also provide a unified paradigm to understand different representative training methods.
Within this paradigm, all methods are conceptualized as either direct or simplified RL techniques. 
As summarized in Equation §eq:objective, there exist three key components: Data Source, Algorithm, and Reward Function. We provide some potential future directions about the three components.

**Data Source.** Data source is the raw material of all training methods.
In the context of RL, we specifically refer to the data source as the unlabeled questions with the outputs sampled from the policy model. In this paper, we only use the questions from the instruction tuning stage and a naive nucleus sampling to sample outputs. We think this is a potential reason that 
our RL pipeline only improves the Maj@K performance. In the future, we will explore our RL pipeline on out-of-distribution question prompts, in conjunction with **advanced sampling (decoding) strategies**, like those based on tree-search methods [yao2023tree].
Also, the **efficient inference techniques** [xia-etal-2023-speculative,leviathan2023fast,kwon2023efficient,xia2024unlocking], which determines the exploration efficiency of policy models, also play an exceedingly important role.

**Algorithms.** Algorithms process the data and reward signal to the gradient coefficient to update the model parameter. 
Based on Equation §eq:objective, to some extent, all methods now fully **TRUST** the signal of the reward function to increase or decrease the conditional probability of a certain token.
However, it is impossible to ensure the reward signal is always reliable, especially in extremely complex tasks. For example, even the PRM800K datasets [lightman2023let], which have been carefully annotated by well-trained annotators, still contain approximately 20% of incorrectly annotations [^note: https://github.com/openai/prm800k/issues/12#issuecomment-1728491852] . To this end, we will explore the reinforcement learning algorithm that is robust against noisy reward signals. We believe such **WEAK-TO-STRONG** [burns2023weak] alignment methods will bring a fundamental change to the learning algorithms.

**Reward Function.** Reward function is the source of the training signal.
In RL, the reward function is usually the neural reward model.
We think there exist three important directions for reward models:
1) **How to enhance the generalization ability of the reward model.** The reward model must be effectively generalized to handle out-of-distribution questions and advanced decoding outputs; otherwise, reinforcement learning may merely stabilize the distribution of LLMs rather than improve their fundamental capabilities;
2) **How to reflect the uncertainty of reward model.** The uncertainty could potentially act as a linking bridge between the weak reward model and the weak-to-strong learning algorithms;
3) **How to efficiently build high-quality process reward models** that can provide fine-grained training signals for the reasoning process [lightman2023let,wang2023math].

## Conclusion, Limitation, and Future Work

We present , which outperforms all open-source models on the competition-level MATH benchmark and approaches the performance of closed models.
is initialized with DeepSeek-Coder-v1.5 7B and undergoes continual training for 500B tokens, with a significant component of the training data being 120B math tokens sourced from Common Crawl.
Our extensive ablation study shows web pages offer significant potential for high-quality mathematical data, while arXiv may not as beneficial as we expected.
We introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), which can notably improve mathematical reasoning capabilities with less memory consumption.
The experiment results show that GRPO is effective even if -Instruct 7B has reached a high score on benchmarks. 
We also provide a unified paradigm to understand a series of methods and summarize several potential directions for more effective reinforcement learning.

Although achieves impressive scores on quantitative reasoning benchmarks, its capability on geometry and theorem-proof are relatively weaker than closed models.
For instance, in our dry run, the model cannot handle problems related to triangles and ellipses, which may indicate data selection bias in pre-training and fine-tuning. In addition, restricted by the model scale, is worse than GPT-4 on few-shot capability.
GPT-4 could improve its performance with few-shot inputs, while shows similar performance in zero-shot and few-shot evaluation.
In the future, we will further improve our engineered data selection pipeline to construct more high-quality pre-trained corpus.
In addition, we will explore the potential directions (Section §sec:effec_rl) for more effective reinforcement learning of LLMs.



## Appendix

### Analysis of Reinforcement Learning

We provide the detailed derivation of the data source and gradient coefficient (algorithm and reward function) across various methods, including SFT, RFT, Online RFT, DPO, PPO, and GRPO.

#### Supervised Fine-tuning

The objective of Supervised Fine-tuning is maximizing the following objective:

$$
_{SFT}()=[q, o P_{sft}(Q, O)]({|o|}_{t=1}^{|o|} _(o_t | q, o_{<t})).

$$

The gradient of $_{SFT}()$ is:

$$
_{}_{SFT} = [q, o P_{sft}(Q, O)]({|o|}_{t=1}^{|o|} _{} _(o_{t} | q, o_{<t})).
$$

Data Source: The dataset employed for SFT. Reward Function: This can be regarded as human selection. Gradient Coefficient: always set to 1.

#### Rejection Sampling Fine-tuning
 
Rejection Sampling Fine-tuning first samples multiple outputs from the supervised fine-tuned LLMs for each question, and then trains LLMs on the sampled outputs with the correct answer.
Formally, the objective of RFT is to maximize the following objectives:

$$
_{RFT}()= [q P_{sft}(Q), o _{sft}(O|q)]( {|o|}_{t=1}^{|o|} (o) _(o_{t} | q, o_{<t})).
$$

The gradient of $_{RFT}()$ is:

$$
_{}_{RFT}()= [{q P_{sft}(Q), o _{sft}(O|q)}]( {|o|}_{t=1}^{|o|} {(o)} _{}_(o_{t} | q, o_{<t})).
$$

Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: Rule (whether the answer is correct or not). Gradient Coefficient:

$$
GC_{RFT}(q, o, t) = (o)=
1 & & {the \ answer \ of \ o \ is \ correct} \\
0 & & {the \ answer \ of \ o \ is \ incorrect} \\

.

$$

#### Online Rejection Sampling Fine-tuning

The only difference between RFT and Online RFT is that the outputs of Online RFT are sampled from the real-time policy model $_{}$, rather than from the SFT model $_{_{sft}}$. Therefore, the gradient of online RFT is:

$$
_{}_{OnRFT}()= [{q P_{sft}(Q), o _{}(O|q)}]( {|o|}_{t=1}^{|o|} {(o)} _{}_(o_{t} | q, o_{<t})).
$$

#### Direct Preference Optimization (DPO)

The objective of DPO is:

$$

 _{DPO}() = {[q P_{sft}(Q), o^+, o^- _{sft}(O|q)]} ( {|o^+|}_{t=1}^{|o^+|} (o^+_t | q, o^+_{<t})}{_{}(o^+_t | q, o^+_{<t})} - {|o^-|}_{t=1}^{|o^-|} (o^-_{<t} | q, o^-_{<t})}{_{}(o^-_{<t} | q,o^-_{<t})} ) 

$$

The gradient of $_{DPO}()$ is:

$$

 _{}_{DPO}() = {[q P_{sft}(Q), o^+, o^- _{sft}(O|q)]}
 & ( {|o^+|}_{t=1}^{|o^+|} GC_{DPO} (q,o,t) _{}_{}(o^+_t | q, o^+_{<t}) . \\
 - & . {|o^-|}_{t=1}^{|o^-|} GC_{DPO} (q,o,t) _{}_{}(o^-_t | q, o^-_{<t}) )

$$

Data Source: question in SFT dataset with outputs sampled from SFT model.
Reward Function: human preference in the general domain (can be `Rule' in mathematical tasks).
Gradient Coefficient: 

$$
GC_{DPO}(q,o,t) = ((o^-_t | q, o^-_{<t})}{_{}(o^-_t | q, o^-_{<t})} - (o^+_t | q, o^+_{<t})}{_{}(o^+_t | q, o^+_{<t})}) 

$$

#### Proximal Policy Optimization (PPO)

The objective of PPO is:

$$
_{PPO}() = {[q P_{sft}(Q), o _{_{old}}(O|q)]} {|o|} _{t=1}^{|o|} [  | q, o_{<t})}{_{_{old}}(o_{t} | q, o_{<t})} A_{t},  (  | q, o_{<t})}{_{_{old}}(o_{t} | q, o_{<t})}, 1 - , 1 + ) A_{t} ].
$$

To simplify the analysis, it is assumed that the model only has a single update following each exploration stage, thereby ensuring that $_{_{old}} = _{}$.
In this case, we can remove the $$ and ${clip}$ operation:

$$
_{PPO}() = {[q P_{sft}(Q), o _{_{old}}(O|q)]} {|o|} _{t=1}^{|o|}  | q, o_{<t})}{_{_{old}}(o_{t} | q, o_{<t})} A_{t}.
$$

The gradient of $_{PPO}()$ is:

$$

 _{}_{PPO}() = {[q P_{sft}(Q), o _{_{old}}(O|q)]} {|o|} _{t=1}^{|o|} A_t _{}_(o_{t} | q, o_{<t})

$$

Data Source: question in SFT dataset with outputs sampled from policy model.
Reward Function: reward model.
Gradient Coefficient:

$$
 GC_{PPO}(q, o, t, _{_{rm}}) = A_t,

$$

where $A_t$ is the advantage, which is computed by applying Generalized Advantage Estimation (GAE) [gae], based on the rewards ${r_{t}}$ and a learned value function $V_{}$. 

#### Group Relative Policy Optimization (GRPO)

The objective of GRPO is (assume $_{_{old}} = _{}$ for simplified analysis):

$$

 _{GRPO}() &= {[q P_{sft}(Q), {o_i}_{i=1}^G _{_{old}}(O|q)]} \\
 & {G}_{i=1}^G{|o_i|} _{t=1}^{|o_i|} [ | q, o_{i,<t})}{_{_{old}}(o_{i,t} | q, o_{i,<t})} _{i,t} - ((o_{i,t}|q,o_{i,<t})}{_{}(o_{i,t}|q,o_{i,<t})}- (o_{i,t}|q,o_{i,<t})}{_{}(o_{i,t}|q,o_{i,<t})} - 1)].

$$

The gradient of $_{GRPO}()$ is:

$$

 _{}_{GRPO}() & = {[q P_{sft}(Q), {o_i}_{i=1}^G _{_{old}}(O|q)]} \\
 & {G}_{i=1}^G{|o_i|} _{t=1}^{|o_i|} 
 [_{i,t} + ((o_{i,t}|o_{i,<t})}{_{}(o_{i,t}|o_{i,<t})} - 1)] _{}_(o_{i,t} | q, o_{i,<t}). 

$$

Data Source: question in SFT dataset with outputs sampled from policy model.
Reward Function: reward model.
Gradient Coefficient:

$$
GC_{GRPO}(q, o, t, _{_{rm}}) = _{i,t} + ((o_{i,t}|o_{i,<t})}{_{}(o_{i,t}|o_{i,<t})} - 1),

$$

where $_{i,t}$ is computed based on the group reward scores.