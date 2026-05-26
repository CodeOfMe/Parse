# Bootstrapping Language-Guided Navigation Learning with Self-Refining Data Flywheel

> **arXiv:** [2412.08467](https://arxiv.org/abs/2412.08467)
> **TeX source:** [arXiv-2412.08467v1/](arXiv-2412.08467v1/)
> **PDF:** [SRDF-arXiv-2412.08467v1.pdf](SRDF-arXiv-2412.08467v1.pdf)

---

## Abstract

Creating high-quality data for training robust language-instructed agents is a long-lasting challenge in embodied AI. In this paper, we introduce a Self-Refining Data Flywheel (SRDF) that generates *high-quality* and *large-scale* navigational instruction-trajectory pairs by iteratively refining the data pool through the collaboration between two models, the instruction generator and the navigator, without any human-in-the-loop annotation. 
Specifically, SRDF starts with using a base generator to create an initial data pool for training a base navigator, followed by applying the trained navigator to filter the data pool. This leads to higher-fidelity data to train a better generator, which can, in turn, produce higher-quality data for training the next-round navigator. Such a flywheel establishes a data self-refining process, yielding a continuously improved and highly effective dataset for large-scale language-guided navigation learning. Our experiments demonstrate that after several flywheel rounds, the navigator elevates the performance boundary from 70% to 78% SPL on the classic R2R test set, surpassing human performance (76%) for the first time. 
Meanwhile, this process results in a superior generator, evidenced by a SPICE increase from 23.5 to 26.2, better than all previous VLN instruction generation methods. Finally, we demonstrate the scalability of our method through increasing environment and instruction diversity, and
the generalization ability of our pre-trained navigator across various downstream navigation tasks, surpassing state-of-the-art methods by a large margin in all cases.{ [^note: Code and data are available at https://github.com/wz0919/VLN-SRDF.\\ 
*{1.4em] $^{*}$Interned at Shanghai AI Laboratory.}}

**Figure:** *
 **(a) Our Pipeline:** After using the (instruction) generator to label paths for data augmentation in navigator training, we leverage the trained navigator to filter high-quality data to train a better generator, and the improved generator refines the data pool to train a stronger navigator, iteratively running on the flywheel.
 **(b) Comparison with state-of-the-art (SoTA) methods**: Our approach significantly outperforms SoTA on all tasks. It also **surpasses human performance** on R2R and approaches human-level results on RxR-English and CVDN (for other tasks, human performance is not reported in their paper). The R2R result is from the test set, while others are from val unseen. 
 * () _(image: figures/teaser.pdf)_

% — BEGIN sections/1_intro —

## Introduction

The lack of high-quality data is one of the main bottlenecks in training embodied agents to complete real-world human activities. 
Unlike many other discriminative or generative learning problems, where the data itself naturally formulates a self-supervised learning objective [devlin2018bert,he2022masked] or the data labeling can be facilitated by existing models [ros2016synthia,tian2024stablerep]
, training embodied agents usually requires expensive human annotation on complex vision-linguistic contents and physical interactions. 
In the essential embodied AI problem of Vision-and-Language Navigation (VLN) [zhang2024vision], one widely considered solution is to first train a trajectory-to-instruction generator [fried2018speaker, tan2019envdrop, chen2022hm3dlearning] using a small amount of human-annotated data and then use the generator to caption large-scale trajectories sampled from interactive environments [hao2020prevalent, chen2022hm3dlearning, wang2023scalevln].
Among them, [hao2020prevalent] demonstrates the effectiveness of scaling the synthetic instruction-trajectory pairs in the existing training environments, while [chen2022hm3dlearning, wang2023scalevln, li2024panogen] emphasizes the importance of scaling training environments for better generalization ability of agents. 
However, the quality of the synthetic data, especially the language fidelity, is highly under-investigated. 
We find that training solely with the large-scale synthetic data (352${}$larger in instruction-trajectory pairs, and 13$$more diverse in environments) yields worse results compared to training with a small human-annotated dataset (see Table §tab:solely), which indicates the need for high-quality instructions beside simply scaling the data and environments. 

Though many methods have been proposed for improving data quality in multi-modal understanding and generation tasks [fang2023data,li2024datacomp,BetkerImprovingIG,fang2024vila,li2024if,nguyen2024improving], improving synthetic instruction-trajectory pairs for language-guided navigation has several distinct challenges. 
First, the most straightforward approach is to build a strong instruction generator, but the limited amount of high-quality training data (e.g., R2R [anderson2018r2r] train split has only 14K human-labeled data) makes it challenging to train a robust generator capable of producing high-fidelity instructions for diverse trajectories. Additionally, manually correcting instructions by humans is resource-intensive and costly. 
{r}{0.42}



{!}{
{l|cccc}

Training Data & #data & #Env. & SR$$ & SPL$$ \\
R2R & 14K & 61 & 65.9 & **55.9** \\
Prevalent & 1.0M & 60 & **67.1** & 54.8 \\
ScaleVLN & 4.9M & 800 & 63.9 & 50.1 \\


}


Moreover, the alignment of instruction-trajectory pairs in VLN is hard to evaluate, as
the instructions not only contain semantic information (e.g., `Walk past the table') but also have rich directional information (e.g., `Turn left') to match with the corresponding trajectory. 
Besides, the visual elements mentioned in the instructions are also temporally aligned to panoramas in multiple scenes. As a result, traditional metrics like CLIP score [radford2021clip] struggle to evaluate such multi-scene directional and semantic alignment, as they only capture single-scene semantic relationships.

In this work, we propose a Self-Refining Data Flywheel, , that automatically evaluates and improves the quality of the generated instructions at scale, through an iterative collaboration between the navigator and the instruction generator. 
As shown in Figure §fig:teaser (a), our first step is similar to ScaleVLN's process [wang2023scalevln], which trains an instruction generator using the original human-labeled data, then generates instructions for unlabeled paths sampled from 800 HM3D training environments and trains a strong navigator using the generated data. 
Then, for evaluating the generated instructions, we propose to use the trained navigator's path-fidelity score (nDTW [ilharco2019ndtw] and SPL [anderson2018spl], measuring how closely the navigator's followed path matches the original trajectory) as the similarity score.
Since the trained navigator is strong enough (achieves human-level performance in [wang2023scalevln]) and has already learned to connect visual landmarks and directional cues to actions effectively, its fidelity in following the instructions can naturally reflect how well the instruction-trajectory pairs are aligned. It also avoids manually setting vague thresholds when using CLIP-score-like metrics for filtering. 
In our case, SPL=1 yields a perfect trajectory match.
After using the navigator as a filter to obtain a high-quality subset of the generated data, this subset of data will be used to train a better instruction generator in the next iteration. The improved instruction generator re-generates instructions for bad samples to produce better datasets, which is used to train a stronger navigator. 
The process iterates, with the navigator improving the instruction generator via data filtering, and the instruction generator improving the navigator via data refining, ultimately producing both a highly capable navigator and instruction generator, along with a substantially high-quality synthetic VLN dataset.

We build our flywheel upon the R2R dataset [anderson2018r2r] and provide detailed analysis.
Empirically, we show that our navigator and instruction generator can iteratively improve each other with our . We also demonstrate the scalability of our method: the instruction generator consistently improves with additional environments and data, and the navigator benefits more from increased instruction diversity when trained with our high-quality instructions compared to with low-quality datasets.
On the R2R dataset, our approach surpasses previous state-of-the-art results by a wide margin in both instruction following and generation, and notably, for the first time, we significantly surpass human performance (76% SPL)
in instruction following, demonstrating the effectiveness of our SRDF to improve data quality.
Furthermore, we evaluate the transferability of our pre-trained navigator across various downstream navigation tasks, including VLN with dialogue-based instructions (CVDN [thomason2020cvdn]), long-term VLN (RxR-English [anderson2020rxr], R4R [zhu2020babywalk]), high-level VLN (SOON [zhu2021soon], REVERIE [qi2020reverie]), and even VLN in continuous environments (R2R-CE [krantz2020beyond]). As shown in Figure §fig:teaser (b), we achieve state-of-the-art performance on all the tasks, while approaching human performance for RxR-English and CVDN, underscoring the superior quality of our generated data and the robust transferability of our pre-trained navigator. 
% — END sections/1_intro —

% — BEGIN sections/2_related_work —

## Related Work

**Vision-and-Language Navigation..** 

VLN requires an agent to navigate in unseen environments based on natural language instructions. 
Numerous datasets have been proposed for this task [anderson2018r2r, anderson2020rxr, qi2020reverie, shridhar2020alfred, thomason2020cvdn, padmakumar2022teach, nguyen2019help, chen2019touchdown, kim-etal-2021-ndh], spanning both indoor and outdoor environments with varied levels of instruction detail.
The limited availability of human-annotated data for training generalizable VLN agents to achieve near-human performance is a key challenge due to the high cost of collecting instruction-trajectory pairs. To address this, various data augmentation approaches have been explored. Some focus on scaling environments by editing existing ones [li2022envedit, liu2021vision] or generating new ones with text-to-image models [li2024panogen]. Others scale instruction data by training instruction generators to generate instructions for unannotated paths [hao2020prevalent,zhang2023vlntrans,zhang2024navhint], or by leveraging large sets of rendered environments from simulators (e.g., HM3D [ramakrishnan2021hm3d], Gibson [xia2018gibson]) [wang2023scalevln, kamath2022marval, chen2022hm3dlearning].
While data scaling has been effective for VLN, the quality of the data, particularly the alignment between instructions and trajectories, remains under-explored. In this paper, we investigate the impact of data quality on VLN and propose a data flywheel that iteratively refines itself through collaboration between the navigator and the generator.

**High-Quality Multimodal Dataset Curation..** 

Many multimodal studies show that improving data quality can significantly enhance model performance, either through advanced dataset filtering [fang2023data,li2024datacomp,gadre2024datacomp,sun2023dreamsync] or refining captions with strong models [BetkerImprovingIG,fang2024vila,li2024if,nguyen2024improving,wang2024internvideo2,tan2024large]. Recently, Segment Anything (SAM) [kirillov2023segment] demonstrated how data and models can improve each other through a data flywheel with a human-in-the-loop process, evolving from model-assisted to fully automated annotation. Our data flywheel similarly integrates filtering and re-captioning data via navigator verification and generator refinement, operating in a data-model loop like SAM, but without any human intervention.

**Self-Improving Language Models..** 

Studies show that Large Language Models (LLMs) can improve themselves by training on their own generated outputs across tasks like programming [haluptzok2022language], summarization [patil2024refinesumm], question-answering [lee2024llm2llm, yu2024self], reasoning [prasad2024self], and others [li2023self,madaan2024self,zhou2024calibrated], where the quality of the self-generated data is ensured via human [Ouyang2022TrainingLM,Bai2022TrainingAH], off-the-shelf verifiers/reward models [ni2023lever,wang2019reinforced,dou2024reflection,Bai2022ConstitutionalAH,Lee2023RLAIFVR,nguyen2024laser] or model's self feedback [yuan2024self,wu2024meta,wang2024cream]. 
In our pipeline, the instruction generator iteratively self-improves using its own generated data. Unlike prior work, our approach establishes a fully automated, multi-round, two-model mutual improvement process, enabling the navigator, the generator, and the dataset to evolve concurrently through continuous model-driven feedback.

% — END sections/2_related_work —

% — BEGIN sections/3_method —

## Methodology

### Background

VLN data typically consists of instruction-trajectory pairs, where a trajectory represents a path in a 3D environment, and the corresponding instruction guides the agent to follow it. This data can be used for training either instruction-to-action navigators or trajectory-to-instruction generators. Since manually annotating trajectories is costly, a common approach is to first train a generator on limited human-annotated data and then use it to generate instructions for paths in unlabeled environments, which are subsequently used to augment navigator training.

While this method helps increase the data amount, it poses challenges regarding the quality and fidelity of the generated instructions. These challenges imply two essential problems – how to generate better data and evaluate the data quality, which we aim to address in the following sections.

**Figure:** *
 (a) Our (instruction) generator refines low-quality data filtered in the previous round via sampling to train the next-round navigator, and greedily generates high-confidence instructions as candidates to train next-round generator. Then the navigator filters the data for further use. 
(b) In filtering, the navigator re-runs instructions to compute the path-path similarity score (nDTW & SPL). High-fidelity data is kept for training while low-quality data will be refined.
 * () _(image: figures/fig_bolder.pdf)_

### SRDF: Self-Refining Data Flywheel

In this section, we introduce the Self-Refining Data Flywheel (SRDF) to tackle the challenge of evaluating and improving VLN data to bootstrap its learning. Broadly, our system comprises a navigator, $N$, and an instruction generator, $G$, shown in Figure §fig:teaser (a). We use $N$ to assist $G$ by optimizing G using the data filtered by $N$, while $G$ enhances $N$ by refining the low-quality data. This iterative process is repeated multiple times, consistently enhancing both $G$ and $N$'s performance.

**Main Components..** Our SRDF requires the following resources at the beginning:
(1) seed data
 $D_{Seed}$, typically human-annotated, for training a base $G$ and $N$
(2) unlabeled trajectory pool, $D_{Traj}$, usually collected from large-scale environment datasets for generating new training data.

**Training Base Instruction Generator..** 
Most previous works [fried2018speaker,tan2019envdrop,dou2022foam] train instruction generators from scratch, which limits their ability to generalize text effectively. Some recent approaches leverage pretrained vision-and-language models but neglect critical directional information during trajectory encoding [li2024panogen,kong2024controllable], leading to instructions with inconsistent or incorrect directional cues.

We hypothesize that an effective instruction generator should be capable of understanding both multi-image inputs and interleaved image-text inputs. Multi-image understanding is crucial for accurately encoding multi-panorama trajectories and interleaved image-text comprehension helps encode directional images within the raw text space, enabling simple yet effective visual prompting of trajectories. To achieve this, we utilize Mantis [Jiang2024MANTISIM], an interleaved multi-image multimodal large language model (MLLM), which includes a SigLIP vision encoder [zhai2023sigmoid], a multimodal projector, and a LLaMA-3 language model backbone [dubey2024llama], pretrained on a curated corpus of interleaved image-text pairs. We use our designed template (see Appendix §fig:prompt, §tab:template exp for details) to convert the instruction-trajectory pairs from $D_{Seed}$ into supervised fine-tuning (SFT) data, and then LoRA-fine-tune the language backbone using this converted data. This results in a robust instruction generator $G_1$, which serves as the foundation for our data generation process.

**Generating Base Training Data..** 
Once the base instruction generator, $G_1$, is trained, it is used to generate two types of data from the unlabeled trajectories, $D_{Traj}$: one for training the navigator and another for improving the round-2 instruction generator. For data training the instruction generator, we use greedy decoding to generate the most reliable instruction for each trajectory to form $D^G_2$. For data used to train the navigator, we prioritize diversity by generating multiple instructions via random sampling to form $D^N_1$. The data generation step aims to provide sufficient training data for both improving the navigator and the instruction generator in subsequent iterations.

**Training Base Navigator..** 
We employ the DUET model [chen2022duet] as our navigator. The model is pre-trained using $D^N_1$ and subsequently fine-tuned on $D_{Seed}$. This results in a highly capable navigator $N_1$, which will be used to evaluate the quality of the generated data.

**Evaluating and Filtering the Generated Data..** 

We propose using the trained navigator $N_1$ to self-evaluate the generated data $D^N_1$. Intuitively, if the navigator successfully follows the generated instruction and navigates along the original path, it suggests that the instruction is well-aligned with the trajectory and the strong performance of this navigator (Could achieve near-human performance in previous work [wang2023scalevln]) further ensures its reliability. Such self-consistency transforms the challenging task of computing instruction-trajectory similarity into the simpler problem of computing trajectory-trajectory similarity. 

We run the navigator on the generated instructions and compute trajectory-trajectory similarity scores. We filter high-quality data $FD^G_{2}$ for the round-2 instruction generator training by selecting instances from $D^G_2$ with SPL=1 (indicating the navigator perfectly follows the path). For $D^N_{1}$ (navigator training data), we filter with nDTW$$0.9 (indicating very close alignment between trajectories) to get $FD^N_{<2}$, to use in the round-2 training, as shown in Figure §fig:pipeline. This filtered-out high-quality data will be kept in the follow-up round, and we only re-generate low-quality data $LD^N_{2}$ filtered by nDTW$<$0.9. This filtering step ensures that only reliable data is used for subsequent iterations of training, reducing data noise and increasing alignment quality.

**Iterative Self-Refining..** 
The core of SRDF is its iterative loop between the instruction generator and the navigator, where each model contributes to improving the other by providing data feedback. Specifically, at each iteration $t$, we first train the generator $G_t$ using the filtered high-quality data $FD^G_t$. Then we use $G_t$ to generate new navigation-training data, $ND^N_t$, for previously bad samples, $LD^N_t$, and new generator training data, $D^G_{t+1}$, for $D_{Traj}$, following the same details in generating base training data. Then we Combine $ND^N_t$ with previously filtered navigation data, $FD^N_{<t}$, to form the complete dataset, $D^N_t$, for training the navigator. Note that the whole data size won't change as we are only refining base samples. After training the navigator $N_t$ using $D^N_t$, we use $N_t$ to filter high-quality data subsets, resulting in $FD^G_{t+1}$ filtered from $D^G_{t+1}$ and $FND^N_t$ from $ND^N_t$. Finally, we combine the newly filtered navigation data, $FND^N_t$, to form $FD^N_{<t+1}$ for the next-round training.
This looping mechanism ensures that the data quality continually improves through a self-refining process. Each iteration benefits from the enhanced quality of both the instruction generator and the navigator, ultimately yielding a high-quality dataset and highly capable models. To summarize, the pseudocodes of the SRDF are detailed in Appendix Alg. §alg:srdf.

### Final Dataset

We build our flywheel upon the R2R dataset [anderson2018r2r] as $D_{Seed}$, containing 14,039 human-annotated training data, along with the 178,270 and 2,890,267 unlabelled trajectories from MP3D [chang2017matterport3d] and HM3D [wang2023scalevln] environments, respectively, as $D_{Traj}$. 
We run the flywheel for three rounds to create the final dataset, named -20M. This dataset consists of 19.5 million pre-training examples $D^N_3$, with 6 instructions generated for each HM3D path (via top-3 sampling) and 12 instructions (6 top-3 and 6 top-5 sampling) for each MP3D path, and 0.9M greedy-decoded filtered data $FD^G_4$ (which will be used to fine-tune the pretrained navigator and the third-round generator here). Overall, we generated 20M instructions across 860 environments for navigator training.

**Comparison to Previous VLN Datasets..** Table §tab:stats presents detailed statistics of our dataset and previous VLN datasets. A common issue in existing augmentation datasets is the lack of vocabulary diversity. For instance, prior datasets like Prevalent [hao2020prevalent], despite being significantly larger than R2R, possess only 1/3 of R2R's vocabulary size, even with 76 times more instructions. This issue is even more pronounced in ScaleVLN [wang2023scalevln], which suffers from a highly limited vocabulary. In contrast, our dataset contains over 10,000 unique words and 20 million instructions— three times the vocabulary diversity and 1,454 times the instruction count compared to the original R2R dataset — surpassing all previous human-labeled and synthetic VLN datasets in both vocabulary richness and instruction quantity. Importantly, this substantial increase in size and diversity is achieved without compromising quality, due to the robustness of our instruction generator and the high-precision filtering process guided by our strong navigator. We also provide some visualization examples of our generated instructions in Appendix Figure §fig:vis1 and §fig:vis2.

% — END sections/3_method —

% — BEGIN sections/4_exp —


## Experiments

### Experimental Setup

**Datasets and Evaluation Metrics..** We establish our data flywheel and perform ablation studies primarily on the R2R dataset, while also assessing the transferability of our pre-trained navigator on a range of downstream tasks. These include fine-grained VLN (R2R, RxR-English), VLN with dialog history (CVDN), high-level VLN (REVERIE, SOON), long-term VLN (R4R), and VLN in continuous environments (R2R-CE). Each dataset is split into training, val_seen, and val_unseen sets, with R2R, CVDN, REVERIE, SOON, and R2R-CE also containing test splits. The statistics for the training splits are summarized in Table §tab:stats (manually-labeled datasets), and further details can be found in the Appendix.

We evaluate our navigator using the standard path-fidelity metrics including Success Rate (SR), Success Rate Weighted by Path Length (SPL) [anderson2018spl], Goal Progress (GP) [thomason2020cvdn], Navigation Error (NE), normalized Dynamic Time Warping (nDTW) [ilharco2019ndtw] and Success Rate Weighted by Dynamic Time Warping (sDTW) [ilharco2019ndtw]. We leave the details of these metrics to Appendix.

Although REVERIE and SOON include object grounding tasks, we focus on evaluating navigation performance, as our generated dataset is specifically designed to enhance navigation tasks. We use SPL as the primary metric for R2R, REVERIE, SOON, and R2R-CE, nDTW for RxR-English, sDTW for R4R, and GP for CVDN.

For evaluating our instruction generator, we assess the linguistic quality of the generated instructions using standard text similarity metrics such as BLEU [papineni2002bleu], Meteor [banerjee2005meteor], and Rouge [lin2004rouge], along with commonly used image captioning metrics, including SPICE [anderson2016spice] and CIDEr [vedantam2015cider]. Additionally, we consider SPICE-D [zeng2023kefa], a directional-aware variant of SPICE tailored for VLN instruction evaluation. We adopt SPICE as our primary metric.

**Implementation..** We utilize InternViT [chen2024internvl], a powerful ViT with 6B parameters, as the visual encoder in our instruction-following experiments unless otherwise specified. For continuous environments, we employ CLIP-B/16 [radford2021clip] as the visual backbone.

In our data flywheel, we pre-train the DUET navigator from scratch for 45,000 iterations using a batch size of 1024 and a learning rate of $5 10^{-5}$ on 8 NVIDIA Tesla A100 GPUs. Multiple checkpoints are fine-tuned to select the best pre-training model. The selected model is then fine-tuned for 6K iterations with a batch size of 16 on a single GPU using only the R2R dataset. For the instruction generator, we fine-tune Mantis-8B-SigLIP [Jiang2024MANTISIM] using LoRA, applying it to the query and value layers in each transformer block. Initially, both the navigator and instruction generator are trained from random weights, while subsequent rounds use previous-round weights.

After final-round navigator pre-training, we fine-tune the navigator for various downstream tasks, using PanoGen [li2024panogen] for MP3D-level environment augmentation. For augmentation, we use round-3 filtered data $FD^G_4$ for R2R, ScaleVLN(REVERIE) [wang2023scalevln] for REVERIE, and marky-English [wang2022less] for RxR-English, and no augmentation for other tasks. For R2R-CE, we fine-tune our pre-trained navigator on ETPNav [an2023etpnav] with a waypoint predictor [hong2022bridging,an20221st].
Modules not pre-trained, such as the object grounding module in REVERIE and the depth-image embedding module in ETPNav, are randomly initialized and trained from scratch. We also use $FD^G_4$ to fine-tune the 3rd-round generator to build the final instruction generator. This process can also be viewed as the fourth round of generator training.



### Flywheel Running Results
 

Table §tab: multi-round results presents the results for both the instruction generator and navigator across all rounds. We follow ScaleVLN [wang2023scalevln] to train the navigator with ScaleVLN-HM3D and Prevalent data for augmentation, while we use InternViT [chen2024internvl] features for fair comparison. The instruction generator for ScaleVLN is EnvDrop [tan2019envdrop]. In round 1, our new instruction generator, fine-tuned on R2R with LoRA using the pre-trained Mantis, significantly surpasses EnvDrop. This results in a substantial SR boost for the navigator from 78.6% to 82.4%.

**Navigator and Instruction Generator Improve Each Other..** 
At each round, we use the navigator to filter high-quality data $FD^G_t$ to re-train the instruction generator, and use the improved instruction generator to refine low-quality instructions $LD^N_t$ to re-train the navigator. Despite the strong performance of the round 1 baseline, the generator is further improved by incorporating navigator-filtered data in round 2. The high-quality data filtered by the navigator leads to +1.5 SPICE and +4.2 CIDEr. This trend continues in round 3, where SPICE reaches 25.7, while other metrics remain stable, demonstrating the crucial role of the navigator in enhancing the instruction generator via data filtering.
For the navigator, the data-refining process
leads to continuous improvements in navigation performance with +1.2% SR in round 2, and +0.8% SR in round 3, underscoring the importance of the generator data refinement in enhancing the navigator, as well as the effectiveness of iterative navigator-generator collaboration to build an effective self-refining flywheel.

### Analysis

**Comparison of Different Scoring Functions..** 

We analyzed classical filtering methods will likely fail to capture complex path-instruction similarity in the previous discussion. In Table §filtering_method, we further verify the importance of our navigator-filtering compared to other filtering baselines.
Using our round 1 instruction generator, we produce instructions for 783 trajectories from the validation unseen split, rank them based on various scoring functions, and filter the top 400 to assess their similarity to GT instructions. `No filter' refers to the average score of all 783 instructions, and intuitively, more similar instructions should yield higher NLP metric scores.

We compare our navigator’s nDTW-score with CLIP-Sim [radford2021clip] and Mantis score [Jiang2024MANTISIM]. CLIP-Sim is computed by averaging image-instruction similarities across all observations in the trajectory, while Mantis-score is produced by inputting the path as interleaved image-text pairs and asking Mantis to provide a similarity score. Results show that the Mantis score fails to improve over the baseline, likely due to the trajectories is too complex to understand for the MLLM. CLIP-Sim provides a slight SPICE improvement, possibly because it can capture some landmark-level similarities between the trajectory images and the instructions, but does not improve SPICE-D as it lacks directional understanding. In contrast, our Navigator-nDTW similarity filtering method successfully identifies high-quality instructions, leading to a substantial improvement, demonstrating our navigator-nDTW captures path-instruction similarities much better than others.



**Effect of Instruction Diversity in Navigator Training.** 
We assess the impact of instruction diversity by training the navigator with different numbers of instructions per path ($#instr=$ 1, 3, 6, 12) on the MP3D environments, as shown in Table §tab: prevalent scaling. We use CLIP-B/16 as the visual feature to establish a well-known baseline [li2024panogen,wang2023scalevln] with the Prevalent dataset, while "Our" uses instructions generated by our round 2 instruction generator. Compared to Prevalent, our instructions consistently achieve stronger downstream results at each $#instr$ level, emphasizing the importance of instruction quality. Our navigator also benefits significantly when increasing $#instr$ from 1 to 3 and 3 to 6, while Prevalent's performance saturates after 3, demonstrating that scaling instruction diversity will be more effective when instruction
quality is higher. Increasing $#instr$ to 12 yields similar results to 6, suggesting that $#instr=6$ is an optimal balance for training.

**Effect of Additional Data in Instruction Generator Training..** 

In Table §additional, we examine the importance of high-quality data in instruction generator training, and the potential scalability of our pipelines by evaluating the influence of environment numbers. The round-1 generator without supplementary data serves as the baseline. Adding the ScaleVLN dataset does not improve performance, likely due to its low diversity, which limits generalization in text generation tasks.

When training with the Prevalent dataset, which has greater diversity, performance gains remain minimal, possibly because of the data noise, as it also shows low quality in Table §tab:solely. In contrast, adding data from ours with increased environments (we split $FD^G_2$ by environments) consistently enhances performance. This improvement is likely due to both the diversity and quality of our data, which are carefully maintained throughout the process,
boosting the SPICE from 23.7 to 25.2.

We also experimented with training using sampled versus greedy-decoded instructions. Sampled instructions resulted in slightly lower performance, suggesting they may introduce noise, whereas greedy-decoded instructions, produced with higher confidence, are more reliable. These results show the strong extensibility of our pipeline, and the critical role of high-quality data, both in instruction generator training.





### Comparision with State of the Arts

**R2R and R2R-CE..** Table §tab:r2r 
compares agent performance on the R2R and R2R-CE datasets. 
The DUET navigator, trained on our high-quality datasets, improves SoTA SPL (ScaleVLN [wang2023scalevln]) by 8% on the R2R test set, demonstrating our data's strong instruction-trajectory alignment that allows effective decision-making learning. Additionally, The gap between oracle success and success is reduced to 4%, compared to the previous best of 6%, highlighting that our data provides stronger clues for the agent to learn when to stop. Notably, our data-centric approach yields greater improvements compared to most model design modifications, highlighting the importance of building high-quality data to boost model performance.
For R2R-CE, despite ETPNav using in-domain pre-training with Habitat-rendered RGBD images [savva2019habitat], our model, using no-rendered images from MP3D and ScaleVLN's HM3D without depth-image pre-training, achieves an absolute gain of +8% in SR and SPL, demonstrating the strong transferability of our pre-trained navigator. 

**REVERIE and SOON..** For high-level navigation tasks, as shown in Table §tab:results_others, our method achieves notable improvements on the val unseen split for REVERIE and SOON, with +3.5% SPL over ScaleVLN and +10.0% SPL over AutoVLN [chen2022hm3dlearning], respectively. These gains are especially impressive given that both AutoVLN and ScaleVLN [wang2023scalevln] used in-domain pre-training data, while ours comes from the out-of-domain R2R-style dataset. This demonstrates that our high-quality data not only improves fine-grained instruction-following but also enhances goal-finding, likely due to diverse stopping guidance. Surprisingly, our model achieves results comparable to the pre-exploration agent RREX-Bot [sigurdsson2023rrex] on REVERIE (-0.6% SR), and even surpasses it on SOON (+1.4% SR), showing our model’s robust goal-finding ability.

**CVDN..** Our method also improves the previous best on the val unseen set of CVDN by a large margin (+1.51 meters) in Table §tab:results_others, showing that our model can be generalized to different instruction styles, likely due to learning strong landmark alignment ability, which can be shared across tasks.

**RxR-English and R4R..** For long-horizon (and fine-grained) VLN tasks including RxR-en and R4R, our navigator also shows strong results on the val unseen split, surpassing previous SoTAs by a large margin shown in Table §tab:results_others. It's worth noting that our results on R4R provide a very strong improvement of +16.6% SR. This shows our highly aligned data facilitates learning step-by-step instruction following even with very long trajectories.

**R2R Instruction Generation..** We also compare our instruction generator with previous SoTAs for path-to-instruction generation task on R2R val unseen split in Table §tab:speaker. Thanks to our navigator-filtered high-quality data, our round 2 generator has already beat the previous SoTA significantly, with + 1.8 SPICE and + 4.6 CIDEr. The stronger data generated in the third round results in an additional +0.5 SPICE improvement while keeping other scores comparable or better. This substantial strong performance can even be further enhanced by incorporating $FD^G_4$ for fine-tuning, leading to a +0.5 improvement in SPICE and better results across five additional metrics, indicating the importance of high-quality data in instruction generator training.

% — END sections/4_exp —

% — BEGIN sections/5_conclusion —

## Conclusion

In this work, we introduce a fully automatic self-refining data flywheel to construct a substantially high-quality VLN dataset for augmentation. We propose to iteratively refine the data with the navigator and instruction generator working in tandem—the navigator filtering high-quality data to train a better instruction generator and the instruction generator regenerating better instructions to train a better navigator, ultimately producing both a strong navigator, instruction generator and a high-quality VLN dataset. We thoroughly analyzed the impact of each component in the flywheel, demonstrating that our approach significantly surpasses state-of-the-art methods across multiple VLN benchmarks, covering various instruction styles (R2R, CVDN, REVERIE, SOON), trajectory lengths (R4R, RxR-English), and control spaces (R2R-CE), as well as instruction generation task on R2R. Our self-refining flywheel provides a novel, scalable solution to the data bottleneck in VLN, highlighting the crucial role of instruction quality and alignment in training embodied agents. This method has the potential to drive future advancements in embodied navigation models and paves the way for exploring more sophisticated tasks that rely on high-quality, dynamic, and scalable data.
% — END sections/5_conclusion —

% — BEGIN sections/6_ack —

## Acknowlegement

This work is partially supported by the National Key R&D Program of China (NO. 2022ZD0160100).
We warmly thank Hao Tan, Archiki Prasad, Zhaoyang Wang, and Yiyang Zhou for their helpful suggestions and feedback on the paper.
% — END sections/6_ack —




## Appendix

% — BEGIN sections/appendix —
We first present additional implementation details of our experiments in Section §sec.a, including specifics of the generator, navigator model, and training procedures. Section §sec.b illustrates the SRDF pipeline with pseudo code, while Section §sec.c provides further details and experiments regarding our trajectory-encoding template design. Details of datasets and evaluation metrics are provided in Section §sec.d, with more comprehensive downstream results shown in Section §sec.e. Finally, Section §sec.f presents detailed examples of our generated instructions in comparison with other baselines.

### Additional Implementation details

**Navigator..** 
We use DUET model [chen2022duet] as our navigator, which integrates global and local information through a dual-scale graph transformer architecture. This architecture processes both high-level scene representations as well as detailed local features simultaneously, enhancing the model's capability to interpret language instructions in complex visual contexts. By constructing a topological map in the meanwhile, DUET extends the navigational action space from the current viewpoint to all navigable directions encountered, thus improving planning and error correction. The model employs attention mechanisms to balance global scene contexts with local observations, significantly improving navigation accuracy towards targets based on natural language commands. 

**Generator..** We use Mantis [Jiang2024MANTISIM] as our base model for instruction generation. Mantis comprises a SigLIP vision encoder [zhai2023sigmoid], a multimodal projector, and a LLaMA-3-8B-Instruct language model backbone [dubey2024llama]. It is first pre-trained on multimodal projector data using LLaVa pre-training, followed by fine-tuning on the multi-image interleaved Mantis-Instruct dataset.For our task, we initialize the instruction-tuned model, Mantis-8B-siglip-llama3, and apply updates only to the added LoRA layers in the language backbone. 

**Training Details..** The navigator is trained using similar objectives as in [wang2023scalevln], including Masked Language Modeling and Single Action Prediction. We initialize our model with LXMERT [tan2019lxmert] for the first and second rounds, and with our round-2 pre-trained model for the third round. During generator training, all parameters except the injected LoRA layers are frozen, and only the LoRA layers are fine-tuned. For navigator training, we initialize the instruction generator with Mantis in the first and second rounds and use the round-2 trained generator directly for round 3. Additionally, for downstream task supervisions, we encourage the model to go back to the viewpoint on the GT path yielding best nDTW to the current progress for R4R and RxR-English as their trajectories are not shortest-path, while we use shortest-path supervision as teacher action for other tasks.

### Pseudo Code of SRDF
 

Alg. §alg:srdf provides detailed pseudo code for our Self-Refining Data Flywheel (SRDF), illustrating the process described in Section §3.2. Specifically, the pipeline starts with training an initial instruction generator using seed data. The generator creates training data for the navigator, which is then used to train a base navigator. The navigator, in turn, filters high-quality data to further improve both itself and the generator in subsequent rounds. This iterative process continues for $T$ iterations, refining data quality and improving model robustness with each cycle.


**Figure:** *Interleaved trajectory prompt and expected response for training our instruction generator. The LLM's response is the corresponding instruction of this path.* () _(image: figure)_

### SFT Data Template
 
We use the template shown in Figure §fig:prompt to construct our SFT data for fine-tuning the instruction generator, which encodes a trajectory into an interleave image-action sequence. At each viewpoint, we include key views, such as the view when arriving at the viewpoint and the view when leaving it. For each view, we also append the corresponding action in raw text after the image tokens, creating a multi-image interleaved format to effectively encode the trajectories. The output is the corresponding instruction of the input trajectory.

### Addtional Experiments

**Fourth-Round Generator-Training..** 
Our method demonstrates the potential for continued performance improvement with additional refinement rounds. To illustrate this, we conducted a fourth-round generator training experiment (which is equivalent to the "round 3 w/ $FD^G_4$" results in Table §tab:speaker) Following a similar procedure, we generated \( FD^G_4 \), trained the generator using this data, and observed consistent performance gains over the third round. Results in Table §tab:4round show a new state-of-the-art (SoTA) SPICE score of 26.2, marking an improvement of +2.8 over the previous SoTA (as shown in Table 9). These results highlight the scalability and effectiveness of our approach, indicating that with sufficient computational resources and time, further rounds of improvement can be sustained.

**Effect of Different Scoring Metrics in Generator Training..** 
We conducted additional experiments to demonstrate that classical methods for language model self-improvement face limitations in the VLN context without reliable feedback from the navigator. Specifically, we evaluated two approaches: (1) self-score, a self-rewarding method where the language model scores its own outputs, and (2) CLIP-score, which uses an external tool (CLIP) to provide similarity scores. In these experiments, conducted during round 1, instructions were scored and the top 300K instructions filtered using either (1) or (2) were used to train the instruction generator in round 2. The results in Table §tab:scorer showed that neither self-scores nor CLIP scores yielded significant improvement over the round 1 baseline. In contrast, our navigator-filtering method using nDTW demonstrated substantial gains, highlighting the challenges of providing effective feedback and emphasizing the effectiveness of our approach.

**Table:** *Generator results in the additional fourth round.*

| Ours (round 1) | 23.7 | 71.4 | 29.5 | 46.5 | 23.1 | 50.2 |
| — | — | — | — | — | — | — |
| Ours (round 2) | 25.2 | 73.7 | 31.0 | **50.7** | 24.2 | 51.3 |
| Ours (round 3) | 25.7 | 74.5 | 30.8 | 49.7 | 24.5 | 51.3 |
| Ours (round 4) | **26.2** | **75.3** | **31.1** | 49.2 | **25.0** | **51.4** |

**Table:** *Second-round generator performance with different scorers.*

| round 2 | Self-score | 23.6 | 71.3 | 29.4 | 46.4 | 23.5 | 50.3 |
| — | — | — | — | — | — | — | — |
| round 2 | CLIP-score | 23.9 | 70.6 | 30.0 | 48.6 | 23.1 | 50.4 |
| round 2 | navigator-nDTW | **25.2** | **73.7** | **31.0** | **50.7** | **24.2** | **51.3** |

**Table:** *Two-round generator performance with mPLUG-Owl.*

| mPLUG-owl (round 1) | 22.7 | 70.3 | 28.0 | 44.4 | 23.0 | 49.1 |
| — | — | — | — | — | — | — |
| mPLUG-owl (round 2) | **24.3** | **72.2** | **29.1** | **45.2** | **23.7** | **50.0** |

**Effect of Different MLLM.** 
To demonstrate that our model-boosting process is not reliant on Mantis [Jiang2024MANTISIM], we conducted additional experiments with a weaker multimodal large language model (MLLM), mPLUG-Owl-7B [ye2023mplug]. Using the same methodology, we applied the flywheel process and completed the first two rounds of generator training. In Table §tab:mplug, we observed a significant improvement in the round 2 generator’s performance when trained on its data filtered by the navigator, highlighting the navigator’s critical role in enhancing the generator. Given that the reciprocal improvement of the navigator by the generator has been validated in prior work using the Speaker-Follower framework, these results strongly support our assertion that the model-boosting process is robust and not tied to a specific MLLM.

**Effect of Different Encoding Formats..** In our previous discussion, we hypothesized that the interleaved image-text understanding ability is crucial for training instructor generator based on pre-trained MLLMs. Some prior works [li2024panogen,kong2024controllable] use only image information to build an image sequence for fine-tuning the VLM, but we argue that this approach loses important directional clues. Additionally, if action information is added without an interleaved format (i.e., an action sequence followed by an image sequence), the model may struggle to reason effectively between the two sequences.

We verify this hypothesis in Table §tab:template exp, using two baselines: (1) an image-only sequence (Figure §fig:prompt without action descriptions) and (2) an action sequence followed by an image sequence (Figure §fig:prompt with actions listed after the image sequence). Our results show that using only an image sequence leads to much lower performance, primarily because the model finds it difficult to infer actions between key frames. For the image-sequence + action-sequence format, it still underperforms compared to our interleaved image-text sequence template, likely due to challenges in reasoning across separate sequences. In contrast, the Interleaved Image-Action Sequence performs best, demonstrating its effectiveness in trajectory encoding, which is used in our experiments.

**Table:** * Effect of different trajectory-encoding templates. 
*

| 2-8 | SPICE$$ | SPICE-D$$ | Bleu-1$$ | Bleu-4$$ | CIDEr$$ | Meteor$$ | Rouge$$ |
| — | — | — | — | — | — | — | — |
| Image Seq. + Action Seq. | 22.9 | 27.1 | 70.4 | 29.1 | 46.5 | 22.9 | 50.2 |
| Interleave Image-Action Seq. | **23.7** | **28.4** | **71.4** | **29.5** | **46.5** | **23.1** | **50.2** |

### Details of Datasets and Evaluation Metrics

**Datasets..** We conduct our downstream experiments on 7 datasets listed below.

- **R2R**: Consists of 22k human-annotated navigational instructions, each describing a trajectory that traverses multiple rooms in MP3D. On average, an instruction contains 32 words, and each ground-truth path is formed by seven nodes with a total length of 10 meters.
- **REVERIE**: Inherits the trajectories in R2R but provides high-level instructions that describe a target object. The task for an agent is first to find the object and then localize it in the observation.
- **SOON**: Provides instructions describing target rooms and objects. The average length of instructions is 47 words. SOON does not provide object bounding boxes and requires the agent to predict object center locations in the panorama. We use an automatic object detector to obtain candidate object boxes. The length of expert paths ranges from 2 to 21 steps, with an average of 9.5 steps.
- **CVDN**: Provides dialogues between a navigator who tries to find a target by asking for guidance and an oracle with a privileged view of the best next step. The agent must find the way by interpreting the dialogue history.
- **R2R-CE**: Transfers the discrete trajectories in R2R to continuous 3D scans rendered by the Habitat simulator, where an agent can freely travel in the open space and interact with obstacles. The dataset contains 16k instruction-trajectory pairs after removing non-transferable paths.
- **RxR**: An extension of R2R that addresses shortest path biases and includes more object references. We use the English segment of RxR, which consists of 42,002 instructions, averaging 108 words per instruction.

**Detailed Evaluation Metrics..** (1) Success Rate (SR), which measures whether the agent stops within 3 meters of the target; (2) Success Rate Weighted by Path Length (SPL), which penalizes inefficient, longer paths; (3) Goal Progress (GP), which calculates the agent's progress toward the target; (4) Navigation Error (NE), which is the average distance between the agent's final position and the target in meters; (5) normalized Dynamic Time Warping (nDTW), which measures step-wise alignment between the ground truth and the agent-predicted path; (6) Success Rate Weighted by Dynamic Time Warping (sDTW); (7) Coverage weighted by Length Score (CLS); (8) Remote Grounding Success (RGS), the proportion of successfully executed instructions; and (9) RGS Penalized by Path Length (RGSPL). Metrics (7) to (9) are used only in the detailed results provided in the appendix, with (8) and (9) specifically evaluating object grounding in REVERIE and SOON.

### Detailed Results

In this section, we present detailed results for our downstream navigators, across multiple datasets: R2R (Table §tab:r2r_full), R4R(Table §tab:r4r), RxR-English (Table §tab:rxr), REVERIE (Table §tab:rvr), and SOON datasets (Table §tab:soon). Specifically, on the R2R dataset, our model not only demonstrates improved generalizability to unseen environments but also achieves higher success rates and SPL in seen environments, surpassing previous state-of-the-art (SoTA) approaches by over 3% in SR and 4% in SPL. Our method achieved more than a 10% improvement in both SR and sDTW on the R4R dataset.
Besides, on the RxR English dataset, our approach significantly enhances SPL and sDTW in addition to SR and nDTW, elevating the state-of-the-art SPL to 69.2% and sDTW to 66.3%. Lastly, on REVERIE and SOON datasets, our navigator not only enhances the agents' navigation performance significantly but also substantially improves their grounding capabilities, improving RGSPL by 0.4% on REVERIE test set, and 1.7% on SOON test set compared with previous SoTA approaches. Notably, our navigator relies solely on pretraining with Masked Language Modeling (MLM), and Single Action Prediction (SAP) objectives on R2R datasets and the augmentation dataset collected with our data flywheel. This is in contrast to other approaches that additionally employ an Object Grounding (OG) objective.
The superior performance indicates the strong instruction-trajectory alignment in our high-quality data, which is crucial for effectively learning object grounding from scratch during fine-tuning.

**Table:** *Comparison of single-run performance on R4R dataset.*

| 2-6 | NE$$ | SR$$ | CLS$$ | nDTW$$ | sDTW$$ |
| — | — | — | — | — | — |
| LOVIS 2022lovis | 6.07 | 45.0 | 45.0 | 43.0 | 23.0 |
| VLN-Trans 2023vlntrans | **5.87** | 46.0 | 45.0 | - | 25.0 |
| PanoGen 2024panogen | 6.02 | **47.8** | - | - | - |
| BSG 2023bird | 6.12 | 47.0 | 59.0 | 53.0 | **34.0** |
| NavHint 2024navhint | 6.04 | 46.0 | 45.0 | - | 25.0 |
| VER 2024volumetric | 6.10 | 47.0 | **61.0** | **54.0** | 33.0 |

**Table:** *
Comparison of single-run performance on RxR English dataset. $$ indicates the results are reproduced using their officially released checkpoints.
*

| 2-11
1c| | 1cNE$$ | 1cSR$$ | 1cSPL$$ | 1csDTW$$ | 1c|nDTW$$ | 1cNE$$ | 1cSR$$ | 1cSPL$$ | 1csDTW$$ | 1cnDTW$$ |
| — | — | — | — | — | — | — | — | — | — | — | — | — |
| HAMT$$ 2021hamt | - | 59.4 | - | 50.9 | 65.3 | - | 56.5 | - | 48.3 | 63.1 |
| MARVAL 2022marval | 3.31 | 74.0 | - | 66.7 | **77.5** | 4.47 | 64.7 | - | 57.1 | 70.5 |
| PRET$$ 2024pret | **2.68** | 77.1 | 71.8 | 67.1 | **77.5** | **3.36** | 71.0 | 63.6 | **63.5** | **70.9** |
| MAGIC-L 2024magic | - | **81.3** | **77.5** | **69.2** | 76.6 | - | **72.9** | **65.4** | 58.7 | 68.1 |
| BEVBert$$ 2023bevbert | - | - | - | - | - | 4.2 | 66.7 | 61.1 | 57.0 | 68.6 |
| GOAT 2024GOAT | - | 74.1 | 68.1 | 61.4 | 71.0 | - | 68.2 | 61.7 | 56.6 | 67.1 |

**Table:** *
Comparison of single-run performance on SOON dataset.
*

| 2-9
1c| | 3c|Navigation | 1c|Grounding | 3c|Navigation | 1cGrounding |
| — | — | — | — | — | — | — | — | — |
| 2-9
1c| | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1c|RGSPL$$ | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1cRGSPL$$ |
| AutoVLN 2022hm3dlearning | 53.2 | **41.0** | **30.7** | 4.1 | 48.7 | 40.4 | **27.8** | 5.1 |
| KERM 2023kerm | 51.6 | 38.1 | 23.2 | 4.0 | - | - | - | - |
| NaviLLM 2024navillm | - | 38.3 | 29.2 | - | - | 35.0 | 26.3 | - |
| GOAT 2024GOAT | **54.7** | 40.4 | 28.1 | 5.9 | **50.6** | **40.5** | 25.2 | **6.1** |

**Table:** *
Comparison of single-run performance on REVERIE datasets.
*

| 2-11
1c| | 3c|Navigation | 2c|Grounding | 3c|Navigation | 2cGrounding |
| — | — | — | — | — | — | — | — | — |
| 2-11
1c| | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1cRGS$$ | 1c|RGSPL$$ | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1cRGS$$ | 1cRGSPL$$ |
| DUET 2022duet | 51.1 | 47.0 | 33.7 | 32.2 | 23.0 | 56.9 | 52.5 | 36.1 | 31.9 | 22.1 |
| BEVBert 2023bevbert | 56.4 | 51.8 | 36.4 | 34.7 | 24.4 | 57.3 | 52.8 | 36.4 | 32.1 | 22.1 |
| AutoVLN 2022hm3dlearning | 62.1 | 55.9 | 40.9 | 36.6 | 26.8 | - | - | - | - | - |
| KERM 2023kerm | 55.2 | 50.4 | 35.4 | 34.5 | 24.5 | 57.6 | 52.4 | 39.2 | 32.4 | 23.6 |
| BSG 2023bird | 58.1 | 52.1 | 35.6 | 35.4 | 24.2 | 62.8 | 56.5 | 38.7 | 33.2 | 22.3 |
| ScaleVLN 2023scalevln | **63.9** | 57.0 | 41.8 | - | - | 62.7 | 56.1 | 39.5 | - | - |
| MiC 2023march | 62.4 | 57.0 | 43.6 | 37.5 | **28.7** | 62.4 | 55.7 | 42.0 | 35.3 | 26.2 |
| NaviLLM 2024navillm | 53.7 | 44.6 | 36.6 | - | - | 56.2 | 43.5 | 34.4 | - | - |
| VER 2024volumetric | 61.1 | 56.0 | 39.7 | 33.7 | 23.7 | 62.2 | 56.8 | 38.8 | 33.9 | 23.2 |
| GOAT 2024GOAT | - | 53.4 | 36.7 | **38.4** | 26.1 | - | 57.7 | 40.5 | 38.3 | **26.7** |
| VLN-Colipot 2024llm | 62.6 | **57.4** | **43.6** | 38.9 | 29.8 | **63.3** | **57.8** | **42.3** | **36.6** | 26.6 |

**Table:** *
Comparison of single-run performance on R2R dataset.
*

| 2-13
1c| | 1cNE$$ | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1cNE$$ | 1cOSR$$ | 1cSR$$ | 1c|SPL$$ | 1cNE$$ | 1cOSR$$ | 1cSR$$ | 1cSPL$$ |
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Speaker Follower 2018speaker | 3.36 | 74 | 66 | - | 6.62 | 45 | 36 | - | 6.62 | - | 35 | 28 |
| RCM 2019reinforced | 3.53 | 75 | 67 | - | 6.09 | 50 | 43 | - | 6.12 | 50 | 43 | 38 |
| EnvDrop 2019envdrop | 3.99 | - | 62 | 59 | 5.22 | - | 52 | 48 | 5.23 | 59 | 51 | 47 |
| PREVALENT 2020prevalent | 3.67 | - | 69 | 65 | 4.71 | - | 58 | 53 | 5.30 | 61 | 54 | 51 |
| EntityGraph 2020graph | 3.47 | - | 67 | 65 | 4.73 | - | 57 | 53 | 4.75 | 61 | 55 | 52 |
| NvEM 2021neighbor | 3.44 | - | 69 | 65 | 4.27 | - | 60 | 55 | 4.37 | 66 | 58 | 54 |
| SSM 2021structured | 3.10 | 80 | 71 | 62 | 4.32 | 73 | 62 | 45 | 4.57 | 70 | 61 | 46 |
| AirBert 2021airbert | 2.68 | - | 75 | 70 | 4.10 | - | 62 | 56 | 4.13 | - | 62 | 57 |
| 2020recurrent | 2.90 | - | 72 | 68 | 3.93 | - | 63 | 57 | 4.09 | 70 | 63 | 57 |
| HAMT 2021hamt | 2.51 | - | 76 | 72 | 2.29 | - | 66 | 61 | 3.93 | 72 | 65 | 60 |
| EnvMix 2021envmixup | 2.48 | - | 75 | 72 | 3.89 | - | 64 | 58 | 3.87 | 72 | 65 | 59 |
| SnapEnsemble 2021ensemble | - | - | - | - | 3.63 | - | 67 | 60 | 3.82 | - | 65 | 60 |
| EXOR 2022explicit | - | - | 60 | 58 | - | - | 52 | 49 | - | - | 49 | 46 |
| SEvol 2022sevol | 3.56 | - | 67 | 63 | 3.99 | - | 62 | 57 | 4.13 | - | 62 | 57 |
| MARVAL 2022marval | 2.99 | - | 73 | 69 | 4.06 | - | 65 | 61 | 4.18 | 67 | 62 | 58 |
| LOVIS 2022lovis | 2.40 | - | 77 | 72 | 3.71 | - | 65 | 59 | 4.07 | - | 63 | 58 |
| HOP+ 2023hop+ | 2.33 | - | 78 | 73 | 3.49 | - | 67 | 61 | 3.71 | - | 66 | 60 |
| TD-STP 2022target | 2.34 | 83 | 77 | 73 | 3.22 | 76 | 70 | 63 | 3.73 | 72 | 67 | 61 |
| DUET 2022duet | 2.28 | 86 | 79 | 73 | 3.31 | 81 | 72 | 60 | 3.65 | 76 | 69 | 59 |
| ScaleVLN 2023scalevln | 2.12 | 87 | 81 | 75 | **2.09** | **88** | **81** | **70** | **2.27** | **86** | **80** | **70** |
| BEVBert 2023bevbert | 2.17 | 88 | 81 | 74 | 2.81 | 84 | 75 | 64 | 3.13 | 81 | 73 | 62 |
| VLN-Trans 2023vlntrans | 2.45 | - | 77 | 72 | 3.34 | - | 69 | 63 | 3.94 | - | 66 | 60 |
| Lily 2023learning | - | - | - | - | 2.90 | - | 74 | 62 | 3.44 | - | 72 | 60 |
| NaviLLM 2024navillm | - | - | - | - | 3.51 | - | 67 | 59 | 3.71 | - | 68 | 60 |
| NavHint 2024navhint | - | - | - | - | 3.23 | - | 69 | 65 | 4.00 | - | 65 | 60 |
| NavGPT-2 2024navgpt | 2.84 | 83 | 74 | 63 | 2.84 | 84 | 74 | 61 | 3.33 | 80 | 72 | 60 |
| SAME 2024same | - | - | - | - | 2.73 | - | 76 | 66 | 3.03 | - | 74 | 64 |
| VER 2024volumetric | - | - | - | - | 2.80 | - | 76 | 65 | 2.74 | - | 76 | 66 |
| MAGIC-L 2024magic | **1.73** | **89** | **84** | **79** | 2.22 | 86 | 79 | **70** | 2.75 | 82 | 77 | 69 |
| GOAT 2024GOAT | 1.79 | **89** | **84** | **79** | 2.40 | 85 | 78 | 68 | 3.04 | 80 | 75 | 65 |

### Qualitative Case Study of Generated Instructions

In Figure §fig:vis1 and §fig:vis2, we visualized some examples of our generated instructions, and compare them with Prevalent [hao2020prevalent] and ScaleVLN [wang2023scalevln] baselines. All the example trajectories in Figure §fig:vis1, and Figure §fig:vis2 (a), (b) are collected using recovered environment images from ScaleVLN [wang2023scalevln], while Figure §fig:vis2 (c), (d) are from MP3D environments.

**Rare-Room/Landmark Recognition Ability..** Figure §fig:vis1 demonstrates the strong image-text understanding capability of our instruction generator. Specifically, our generator can recognize rare objects, such as *dentist chair/room* or *a grandfather clock*, thanks to our interleaved image-action trajectory-encoding design. This design preserves the original abilities of the pretrained MLLM while effectively encoding trajectories to generate instructions with rich and accurate landmarks. In contrast, the baseline instruction generator fails to capture these rare concepts due to its from-scratch training paradigm. Instead, it only generates some general landmarks with weak clues.

**Detailed Object-Describing Ability..** Figures §fig:vis1 (c) and (d) illustrate that our instruction generator can describe key objects along the path with greater detail. For instance, while the baseline mentions the *bar* and *the painting* successfully, our generator provides more specifics, such as *brown leather bar stool* and *large painting on the wall*. Such detailed descriptions are crucial for helping the navigator learn richer visual cues. Additionally, in Figure §fig:vis1 (d), our generator performs slight spatial reasoning between objects, resulting in more precise stopping guidance – *the blue and white throw pillows on the right side of the couch*.

**Generalization to Outdoor Environments..** 
In Figure §fig:vis2 (a), (b), we demonstrate the ability of our generator to produce some useful instructions for outdoor environments, even though the model is training using instructions from indoor environments. For instance, in (a), our generated instruction identifies *the glass doors leading outside*, which is more distinct the ScaleVLN's *table* – still a general landmark without a strong viewpoint-specific clue. In (b), the generator successfully identifies outdoor landmarks including *the car* and *the buches*, while the baseline only knows *walkway*.

**OCR Ability..** 
Surprisingly, our instruction generator demonstrates interesting OCR capabilities, as shown in Figures §fig:vis2 (c) and (d). In example (c), the generator successfully identifies the words *cape* and *plug* on the wall, while in example (d), it even identifies a full sentence—*Let's start to redefine how work gets done*. This OCR ability is likely inherited from the pre-trained MLLM, and our fine-tuning approach effectively retains this capability, resulting in highly detailed and accurate guidance in the generated instructions.

**Idiomatic Expressions..** 
Our generator sometimes uses idiomatic expressions in its instructions. An example is shown in Figure §fig:vis2 (c), Sample 2, where the generator says, *go past the desk then stop at the end of the rope*. The phrase *at the end of the rope* usually means that someone has reached the limit of their patience or endurance. In this context, however, it refers to reaching the farthest point that the navigator can proceed—likely the wall. This ability adds diversity to the instructing style, making the generated instructions more varied and engaging.

**Figure:** *
 Visualization of generated instructions.
 * () _(image: figures/vis1.pdf)_

**Figure:** *
 Visualization of generated instructions.
 * () _(image: figures/vis2.pdf)_

% — END sections/appendix —