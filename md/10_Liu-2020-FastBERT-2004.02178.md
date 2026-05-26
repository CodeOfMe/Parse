## FastBERT: a Self-distilling BERT with Adaptive Inference Time

### Weijie Liu[1,2], Peng Zhou[2], Zhe Zhao[2], Zhiruo Wang[3], Haotang Deng[2] and Qi Ju[2,][∗]

1Peking University, Beijing, China
2Tencent Research, Beijing, China
3Beijing Normal University, Beijing, China


dataliu@pku.edu.cn, {rickzhou, nlpzhezhao, haotangdeng, damonju}@tencent.com, SherronWang@gmail.com


### Abstract


Pre-trained language models like BERT have
proven to be highly performant. However, they
are often computationally expensive in many
practical scenarios, for such heavy models can
hardly be readily implemented with limited resources. To improve their efficiency with an assured model performance, we propose a novel
speed-tunable FastBERT with adaptive inference time. The speed at inference can be flexibly adjusted under varying demands, while
redundant calculation of samples is avoided.
Moreover, this model adopts a unique selfdistillation mechanism at fine-tuning, further
enabling a greater computational efficacy with
minimal loss in performance. Our model
achieves promising results in twelve English
and Chinese datasets. It is able to speed up by
a wide range from 1 to 12 times than BERT if
given different speedup thresholds to make a
speed-performance tradeoff.

### 1 Introduction


Last two years have witnessed significant improvements brought by language pre-training, such as
BERT (Devlin et al., 2019), GPT (Radford et al.,
2018), and XLNet (Yang et al., 2019). By pretraining on unlabeled corpus and fine-tuning on labeled ones, BERT-like models achieved huge gains
on many Natural Language Processing tasks.
Despite this gain in accuracy, these models have
greater costs in computation and slower speed at inference, which severely impairs their practicalities.
Actual settings, especially with limited time and
resources in the industry, can hardly enable such
models into operation. For example, in tasks like
sentence matching and text classification, one often
requires to process billions of requests per second.
What’s more, the number of requests varies with
time. In the case of an online shopping site, the

_∗Corresponding author:_ Qi Ju (damonju@tencent.com)


number of requests during the holidays is five to
ten times more than that of the workdays. A large
number of servers need to be deployed to enable
BERT in industrial settings, and many spare servers
need to be reserved to cope with the peak period of
requests, demanding huge costs.
To improve their usability, many attempts in
model acceleration have been made, such as quantinization (Gong et al., 2014), weights pruning
(Han et al., 2015), and knowledge distillation (KD)
(Romero et al., 2014). As one of the most popular
methods, KD requires additional smaller student
models that depend entirely on the bigger teacher
model and trade task accuracy for ease in computation (Hinton et al., 2015). Reducing model sizes to
achieve acceptable speed-accuracy balances, however, can only solve the problem halfway, for the
model is still set as fixated, rendering them unable
to cope with drastic changes in request amount.
By inspecting many NLP datasets (Wang et al.,
2018), we discerned that the samples have different levels of difficulty. Heavy models may overcalculate the simple inputs, while lighter ones are
prone to fail in complex samples. As recent studies
(Kovaleva et al., 2019) have shown redundancy in
pre-training models, it is useful to design a onesize-fits-all model that caters to samples with varying complexity and gains computational efficacy
with the least loss of accuracy.
Based on this appeal, we propose FastBERT,
a pre-trained model with a sample-wise adaptive
mechanism. It can adjust the number of executed
layers dynamically to reduce computational steps.
This model also has a unique self-distillation process that requires minimal changes to the structure,
achieving faster yet as accurate outcomes within
a single framework. Our model not only reaches
a comparable speedup (by 2 to 11 times) to the
BERT model, but also attains competitive accuracy
in comparison to heavier pre-training models.


-----

Experimental results on six Chinese and six English NLP tasks have demonstrated that FastBERT
achieves a huge retrench in computation with very
little loss in accuracy. The main contributions of
this paper can be summarized as follows:

_•_ This paper proposes a practical speed-tunable
BERT model, namely FastBERT, that balances the speed and accuracy in the response
of varying request amounts;

_•_ The sample-wise adaptive mechanism and the
self-distillation mechanism are combined to
improve the inference time of NLP model for
the first time. Their efficacy is verified on
twelve NLP datasets;

_•_ The code is publicly available at [https://](https://github.com/autoliuweijie/FastBERT)

[github.com/autoliuweijie/FastBERT.](https://github.com/autoliuweijie/FastBERT)

### 2 Related work

BERT (Devlin et al., 2019) can learn universal
knowledge from mass unlabeled data and produce
more performant outcomes. Many works have followed: RoBERTa (Liu et al., 2019) that uses larger
corpus and longer training steps. T5 (Raffel et al.,
2019) that scales up the model size even more.
UER (Zhao et al., 2019) pre-trains BERT in different Chinese corpora. K-BERT (Liu et al., 2020)
injects knowledge graph into BERT model. These
models achieve increased accuracy with heavier
settings and even more data.
However, such unwieldy sizes are often hampered under stringent conditions. To be more specific, BERT-base contains 110 million parameters
by stacking twelve Transformer blocks (Vaswani
et al., 2017), while BERT-large expands its size to
even 24 layers. ALBERT (Lan et al., 2019) shares
the parameters of each layer to reduce the model
size. Obviously, the inference speed for these models would be much slower than classic architectures (e.g., CNN (Kim, 2014), RNN (Wang, 2018),
etc). We think a large proportion of computation is
caused by redundant calculation.
**Knowledge** **distillation:** Many attempts have
been made to distill heavy models (teachers) into
their lighter counterparts (students). PKD-BERT
(Sun et al., 2019a) adopts an incremental extraction process that learns generalizations from intermediate layers of the teacher model. TinyBERT
(Jiao et al., 2019) performs a two-stage learning involving both general-domain pre-training and taskspecific fine-tuning. DistilBERT (Sanh et al., 2019)

|ction 𝑃<br>$|Col2|
|---|---|
|_Softmax_|_Softmax_|
|Small Model<br>(Student)|Small Model<br>(Student)|


Figure 1: Classic knowledge distillation approach: Distill a small model using a separate big model.

further leveraged the inductive bias within large
models by introducing a triple loss. As shown in
Figure 1, student model often require a separated
structure, whose effect however, depends mainly
on the gains of the teacher. They are as indiscriminate to individual cases as their teachers, and only
get faster in the cost of degraded performance.
**Adaptive inference:** Conventional approaches
in adaptive computations are performed token-wise
or patch-wise, who either adds recurrent steps to
individual tokens (Graves, 2016) or dynamically adjusts the number of executed layers inside discrete
regions of images (Teerapittayanon et al., 2016;
Figurnov et al., 2017). To the best of our knowledge, there has been no work in applying adaptive
mechanisms to NLP pre-training language models
for efficiency improvements so far.

### 3 Methodology

Distinct to the above efforts, our approach fusions
the adaptation and distillation into a novel speed-up
approach, shown in Figure 2, achieving competitive
results in both accuracy and efficiency.

**3.1** **Model architecture**

As shown in Figure 2, FastBERT consists of
backbone and branches. The backbone is built
upon 12-layers Transformer encoder with an additional teacher-classifier, while the branches include student-classifiers which are appended to
each Transformer output to enable early outputs.

**3.1.1** **Backbone**
The backbone consists of three parts: the embedding layer, the encoder containing stacks of
Transformer blocks (Vaswani et al., 2017), and the
teacher classifier. The structure of the embedding
layer and the encoder conform with those of BERT


-----

Figure 2: The inference process of FastBERT, where the number of executed layers with each sample varies based
on its complexity. This illustrates a sample-wise adaptive mechanism. Taking a batch of inputs (batch _size = 4) as_
an example, the Transformer0 and Student-classifier0 inferred their labels as probability distributions and calculate
the individual uncertainty. Cases with low uncertainty are immediately removed from the batch, while those with
higher uncertainty are sent to the next layer for further inference.


(Devlin et al., 2019). Given the sentence length
_n,_ an input sentence _s_ = [w0, w1, ...wn] will be
transformed by the embedding layers to a sequence
of vector representations e like (1),

_e = Embedding(s),_ (1)

where e is the summation of word, position, and
segment embeddings. Next, the transformer blocks
in the encoder performs a layer-by-layer feature
extraction as (2),

_hi_ = Transformer _i(hi−1),_ (2)

where _hi_ (i = _−1, 0, 1, ..., L_ _−_ 1) is the output
features at the _ith_ layer, and _h−1_ = _e._ _L_ is the
number of Transformer layers.
Following the final encoding output is a teacher
classifier that extracts in-domain features for downstream inferences. It has a fully-connected layer
narrowing the dimension from 768 to 128, a selfattention joining a fully-connected layer without
changes in vector size, and a fully-connected layer
with a softmax function projecting vectors to an
_N_ -class indicator pt as in (3), where N is the taskspecific number of classes.

_pt_ = Teacher _Classifier(hL−1)._ (3)

**3.1.2** **Branches**
To provide FastBERT with more adaptability, multiple branches, i.e. the student classifiers, in the


same architecture with the teacher are added to the
output of each Transformer block to enable early
outputs, especially in those simple cases. The student classifiers can be described as (4),

_psi_ = Student _Classifier_ _i(hi)._ (4)

The student classifier is designed carefully to balance model accuracy and inference speed, for simple networks may impair the performance, while
a heavy attention module severely slows down the
inference speed. Our classifier has proven to be
lighter with ensured competitive accuracy, detailed
verifications are showcased in Section 4.1.

**3.2** **Model training**

FastBERT requires respective training steps for the
backbone and the student classifiers. The parameters in one module is always frozen while the other
module is being trained. The model is trained in
preparation for downstream inference with three
steps: the major backbone pre-training, entire backbone fine-tuning, and self-distillation for student
classifiers.

**3.2.1** **Pre-training**

The pre-training of backbone resembles that of
BERT in the same way that our backbone resembles BERT. Any pre-training method used for
BERT-like models (e.g., BERT-WWM (Cui et al.,
2019), RoBERTa (Liu et al., 2019), and ERNIE


-----

(Sun et al., 2019b)) can be directly applied. Note
that the teacher classifier, as it is only used for
inference, stays unaffected at this time. Also conveniently, FastBERT does not even need to perform
pre-training by itself, for it can load high-quality
pre-trained models freely.

**3.2.2** **Fine-tuning for backbone**
For each downstream task, we plug in the taskspecific data into the model, fine-tuning both the
major backbone and the teacher classifier. The
structure of the teacher classifier is as previously
described. At this stage, all student classifiers are
not enabled.

**3.2.3** **Self-distillation for branch**
With the backbone well-trained for knowledge extraction, its output, as a high-quality soft-label containing both the original embedding and the generalized knowledge, is distilled for training student
classifiers. As student are mutually independent,
their predictions ps are compared with the teacher
soft-label pt respectively, with the differences measured by KL-Divergence in (5),


means we can adjust the number of executed encoding layers within the model according to the
sample complexity.
At each Transformer layer, we measure for each
sample on whether the current inference is credible
enough to be terminated.
Given an input sequence, the uncertainty of a
student classifier’s output _ps_ is computed with a
normalized entropy in (7),


_Uncertainty_ =


�Ni=1 _[p][s][(][i][) log][ p][s][(][i][)]_

_,_ (7)

log [1]

_N_


where ps is the distribution of output probability,
and N is the number of labeled classes.
With the definition of the uncertainty, we make
an important hypothesis.

**Hypothesis 1.** _LUHA: the Lower the Uncertainty,_
_the Higher the Accuracy._

**Definition 1.** _Speed:_ _The threshold to distinguish_
_high and low uncertainty._

LUHA is verified in Section 4.4. Both _Uncer-_
_tainty and Speed range between 0 and 1._ The adaptive inference mechanism can be described as: At
each layer of FastBERT, the corresponding student
_classifier will predict the label of each sample with_
measured Uncertainty. Samples with Uncertainty
below the _Speed_ will be sifted to early outputs,
while samples with Uncertainty above the Speed
will move on to the next layer.
Intuitively, with a higher Speed, fewer samples
will be sent to higher layers, and overall inference
speed will be faster, and vice versa. Therefore,
_Speed can be used as a halt value for weighing the_
inference accuracy and efficiency.

Table 1: FLOPs of each operation within the FastBERT
(M = Million, N = the number of labels).

**Operation** **Sub-operation** **FLOPs** **Total FLOPs**

Self-attention
603.0M
Transformer (768 → 768) 1809.9M
Feedforward
(768 → 3072 1207.9M
_→_ 768)

Fully-connect
25.1M
(768 → 128)

Classifier Self-attention 46.1M

16.8M
(128 → 128)

Fully-connect
4.2M
(128 → 128)

Fully-connect
                        (128 → _N_ )


_DKL(ps, pt) =_


_N_
� _ps(i) · log_ _[p][s][(][i][)]_ (5)

_pt(j)_ _[.]_

_i=1_


As there are L − 1 student classifiers in the FastBERT, the sum of their KL-Divergences is used as
the total loss for self-distillation, which is formulated in (6),

_L−2_

_Loss(ps0, ..., psL−2, pt) =_ � _DKL(psi, pt),_

_i=0_

(6)
where psi refers to the probability distribution of
the output from student-classifier i.
Since this process only requires the teachers output, we are free to use an unlimited number of unlabeled data, instead of being restricted to the labeled
ones. This provides us with sufficient resources
for self-distillation, which means we can always
improve the student performance as long as the
teacher allows. Moreover, our method differs from
the previous distillation method, for the teacher and
student outputs lie within the same model. This
learning process does not require additional pretraining structures, making the distillation entirely
a learning process by self.

**3.3** **Adaptive inference**

With the above steps, FastBERT is well-prepared
to perform inference in an adaptive manner, which


-----

Table 2: Comparison of accuracy (Acc.) and FLOPs (speedup) between FastBERT and Baselines in six Chinese
datasets and six English datasets.

|Dataset/<br>Model|ChnSentiCorp<br>FLOPs<br>Acc.<br>(speedup)|Book review<br>FLOPs<br>Acc.<br>(speedup)|Shopping review<br>FLOPs<br>Acc.<br>(speedup)|LCQMC<br>FLOPs<br>Acc.<br>(speedup)|Weibo<br>FLOPs<br>Acc.<br>(speedup)|THUCNews<br>FLOPs<br>Acc.<br>(speedup)|
|---|---|---|---|---|---|---|

|BERT|21785M<br>95.25<br>(1.00x)|21785M<br>86.88<br>(1.00x)|21785M<br>96.84<br>(1.00x)|21785M<br>86.68<br>(1.00x)|21785M<br>97.69<br>(1.00x)|21785M<br>96.71<br>(1.00x)|
|---|---|---|---|---|---|---|

|DistilBERT<br>(6 layers)<br>DistilBERT<br>(3 layers)<br>DistilBERT<br>(1 layers)|10918M<br>88.58<br>(2.00x)<br>5428M<br>87.33<br>(4.01x)<br>1858M<br>81.33<br>(11.72x)|10918M<br>83.31<br>(2.00x)<br>5428M<br>81.17<br>(4.01x)<br>1858M<br>77.40<br>(11.72x)|10918M<br>95.40<br>(2.00x)<br>5428M<br>94.84<br>(4.01x)<br>1858M<br>91.35<br>(11.72x)|10918M<br>84.12<br>(2.00x)<br>5428M<br>84.07<br>(4.01x)<br>1858M<br>71.34<br>(11.72x)|10918M<br>97.69<br>(2.00x)<br>5428M<br>97.58<br>(4.01x)<br>1858M<br>96.90<br>(11.72x)|10918M<br>95.54<br>(2.00x)<br>5428M<br>95.14<br>(4.01x)<br>1858M<br>91.13<br>(11.72x)|
|---|---|---|---|---|---|---|

|Dataset/<br>Model|Ag.news<br>FLOPs<br>Acc.<br>(speedup)|Amz.F<br>FLOPs<br>Acc.<br>(speedup)|Dbpedia<br>FLOPs<br>Acc.<br>(speedup)|Yahoo<br>FLOPs<br>Acc.<br>(speedup)|Yelp.F<br>FLOPs<br>Acc.<br>(speedup)|Yelp.P<br>FLOPs<br>Acc.<br>(speedup)|
|---|---|---|---|---|---|---|

|BERT|21785M<br>94.47<br>(1.00x)|21785M<br>65.50<br>(1.00x)|21785M<br>99.31<br>(1.00x)|21785M<br>77.36<br>(1.00x)|21785M<br>65.93<br>(1.00x)|21785M<br>96.04<br>(1.00x)|
|---|---|---|---|---|---|---|

|DistilBERT<br>(6 layers)<br>DistilBERT<br>(3 layers)<br>DistilBERT<br>(1 layers)|10872M<br>94.64<br>(2.00x)<br>5436M<br>93.98<br>(4.00x)<br>1816M<br>92.88<br>(12.00x)|10872M<br>64.05<br>(2.00x)<br>5436M<br>63.84<br>(4.00x)<br>1816M<br>59.48<br>(12.00x)|10872M<br>99.10<br>(2.00x)<br>5436M<br>99.05<br>(4.00x)<br>1816M<br>98.95<br>(12.00x)|10872M<br>76.73<br>(2.00x)<br>5436M<br>76.56<br>(4.00x)<br>1816M<br>74.93<br>(12.00x)|10872M<br>64.25<br>(2.00x)<br>5436M<br>63.50<br>(4.00x)<br>1816M<br>58.59<br>(12.00x)|10872M<br>95.31<br>(2.00x)<br>5436M<br>93.23<br>(4.00x)<br>1816M<br>91.59<br>(12.00x)|
|---|---|---|---|---|---|---|

|FastBERT<br>(speed=0.1)<br>FastBERT<br>(speed=0.5)<br>FastBERT<br>(speed=0.8)|6013M<br>94.38<br>(3.62x)<br>2108M<br>93.14<br>(10.33x)<br>1858M<br>92.53<br>(11.72x)|21005M<br>65.50<br>(1.03x)<br>10047M<br>64.64<br>(2.16x)<br>2356M<br>61.70<br>(9.24x)|2060M<br>99.28<br>(10.57x)<br>1854M<br>99.05<br>(11.74x)<br>1853M<br>99.04<br>(11.75x)|16172M<br>77.37<br>(1.30x)<br>4852M<br>76.57<br>(4.48x)<br>1965M<br>75.05<br>(11.08x)|20659M<br>65.93<br>(1.05x)<br>9827M<br>64.73<br>(2.21x)<br>2602M<br>60.66<br>(8.37x)|6668M<br>95.99<br>(3.26x)<br>3456M<br>95.32<br>(6.30x)<br>2460M<br>94.31<br>(8.85x)|
|---|---|---|---|---|---|---|


### 4 Experimental results

In this section, we will verify the effectiveness of
FastBERT on twelve NLP datasets (six in English
and six in Chinese) with detailed explanations.

**4.1** **FLOPs analysis**

Floating-point operations (FLOPs) is a measure of
the computational complexity of models, which
indicates the number of floating-point operations
that the model performs for a single process. The
FLOPs has nothing to do with the model’s operating environment (CPU, GPU or TPU) and only
reveals the computational complexity. Generally
speaking, the bigger the model’s FLOPs is, the
longer the inference time will be. With the same accuracy, models with low FLOPs are more efficient
and more suitable for industrial uses.
We list the measured FLOPs of both structures
in Table 1, from which we can infer that, the cal**culation load (FLOPs) of the Classifier is much**
**lighter than that of the Transformer.** This is the
basis of the speed-up of FastBERT, for although it
adds additional classifiers, it achieves acceleration
by reducing more computation in Transformers.


**4.2** **Baseline and dataset**

**4.2.1** **Baseline**

In this section, we compare FastBERT against two
baselines:

_•_ **BERT[1]** The 12-layer BERT-base model was
pre-trained on Wiki corpus and released by
Google (Devlin et al., 2019).

_•_ **DistilBERT[2]** The most famous distillation
method of BERT with 6 layers was released by
Huggingface (Sanh et al., 2019). In addition,
we use the same method to distill the DistilBERT with 3 and 1 layer(s), respectively.

**4.2.2** **Dataset**

To verify the effectiveness of FastBERT, especially
in industrial scenarios, six Chinese and six English datasets pressing closer to actual applications are used. The six Chinese datasets include

[1https://github.com/google-research/](https://github.com/google-research/bert)
[bert](https://github.com/google-research/bert)

[2https://github.com/huggingface/](https://github.com/huggingface/transformers/tree/master/examples/distillation)
[transformers/tree/master/examples/](https://github.com/huggingface/transformers/tree/master/examples/distillation)
[distillation](https://github.com/huggingface/transformers/tree/master/examples/distillation)


-----

**(d)** **(e)** **(f)**

Figure 3: The trade-offs of FastBERT on twelve datasets (six in Chinese and six in English): **(a) and (d) are Speed-**
_Accuracy_ relations, showing changes of _Speed_ (the threshold of _Uncertainty)_ in dependence of the accuracy; **(b)**
and **(e)** are _Speed-Speedup_ relations, indicating that the _Speed_ manages the adaptibility of FastBERT; **(c)** and **(f)**
are the Speedup-Accuracy relations, i.e. the trade-off between efficiency and accuracy.


the sentence classification tasks (ChnSentiCorp,
Book review(Qiu et al., 2018), Shopping review,
Weibo and THUCNews) and a sentences-matching
task (LCQMC(Liu et al., 2018)). All the Chinese
datasets are available at the FastBERT project. The
six English datasets (Ag.News, Amz.F, DBpedia,
Yahoo, Yelp.F, and Yelp.P) are sentence classification tasks and were released in (Zhang et al., 2015).

**4.3** **Performance comparison**


To perform a fair comparison, BERT / DistilBERT
/ FastBERT all adopt the same configuration as
follows. In this paper, _L_ = 12. The number of
self-attention heads, the hidden dimension of embedding vectors, and the max length of the input
sentence are set to 12, 768 and 128 respectively.
Both FastBERT and BERT use pre-trained parameters provided by Google, while DistilBERT is pretrained with (Sanh et al., 2019). We fine-tune these
models using the AdamW (Loshchilov and Hutter) algorithm, a 2 × 10[−][5] learning rate, and a 0.1
warmup. Then, we select the model with the best
accuracy in 3 epochs. For the self-distillation of
FastBERT, we increase the learning rate to 2×10[−][4]

and distill it for 5 epochs.
We evaluate the text inference capabilities of
these models on the twelve datasets and report their
accuracy (Acc.) and sample-averaged FLOPs under
different Speed values. The result of comparisons
are shown in Table 2, where the _Speedup_ is ob

tained by using BERT as the benchmark. It can
be observed that with the setting of Speed = 0.1,
FastBERT can speed up 2 to 5 times without losing accuracy for most datasets. If a little loss of
accuracy is tolerated, FastBERT can be 7 to 11
times faster than BERT. Comparing to DistilBERT,
FastBERT trades less accuracy to catch higher efficiency. Figure 3 illustrates FastBERT’s tradeoff
in accuracy and efficiency. The speedup ratio of
FastBERT are free to be adjusted between 1 and
12, while the loss of accuracy remains small, which
is a very attractive feature in the industry.


100%
80%
60%

100%
80%
60%

100%

80%

60%


50.08


62.45 [63.78] 62.63 57.18

77.60 76.05

66.17

57.82

|Col1|95.75|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||83.35|77.1|4 74.55|69.20 69.8|8|66.94|
||96.90|<br>|<br>|||64.45|60.91<br>5|
|||90.09|83.4|4 79.05|73.51 74.1|||
|||||||||
|||||||62.45|63.78 62.63 5|
||98.92|96.66|93.9|9 90.16|85.76|<br>||
||||||83.2|77.60|76.05<br>66.17<br>5|
|||||||||
|0.|0~0.1<br>0|.1~0.2<br>0|.2~0.3<br>|0.3~0.4<br>0|.4~0.5<br>0.5~0.6<br>|0.6~0.7<br>0.|7~0.8<br>0.8~0.9<br>0.9~|


Uncertainty

Figure 4: The relation of classifier accuracy and average case uncertainty: Three classifiers at the bottom, in
the middle, and on top of the FastBERT were analyzed,
and their accuracy within various uncertainty intervals
were calculated with the Book Review dataset.


-----

60%

Set speed = 0.8

40% Average = 0.92, median = 0

20%

0%

60% Set speed = 0.5

Average = 2.3, median = 1

40%

20%

0%

60%

Set speed = 0.3

40% Average = 3.2, median = 2

20%

0%

0 1 2 3 4 5 6 7 8 9 10 11

Distribution of layers

Figure 5: The distribution of executed layers on average in the Book review dataset, with experiments at
three different speeds (0.3, 0.5 and 0.8).

**4.4** **LUHA hypothesis verification**

As is described in the Section 3.3, the adaptive inference of FastBERT is based on the LUHA hypothesis, i.e., “the Lower the Uncertainty, the Higher
_the Accuracy”._ Here, we prove this hypothesis using the book review dataset. We intercept the classification results of _Student-Classifier0,_ _Student-_
_Classifier5,_ and _Teacher-Classifier_ in FastBERT,
then count their accuracy in each uncertainty interval statistically. As shown in Figure 4, the statistical indexes confirm that the classifier follows the
LUHA hypothesis, no matter it sits at the bottom,
in the middle or on top of the model.

From Figure 4, it is easy to mistakenly conclude
that Students has better performance than Teacher
due to the fact that the accuracy of Student in each
uncertainty range is higher than that of _Teacher._
This conclusion can be denied by analysis with
Figure 6(a) together. For the Teacher, more samples are located in areas with lower uncertainty,
while the Students’ samples are nearly uniformly
distributed. Therefore the overall accuracy of the
_Teacher is still higher than that of Students._

**4.5** **In-depth study**

In this section, we conduct a set of in-depth analysis
of FastBERT from three aspects: the distribution
of exit layer, the distribution of sample uncertainty,
and the convergence during self-distillation.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||A<br>|vera<br>            S|ge =<br>            et sp|0.9<br>             eed|2, m<br>              = 0.|edia<br>               8|n =|0||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
||||||||A<br>|vera<br>            S|ge =<br>            t sp|2.3<br>             eed|me<br>              = 0.|dian<br>               5|= 1|||
|||||||||||||||||
|||||||||||||||||
||||||||<br>|<br>            Se|<br>            t sp|<br>             eed|<br>              = 0.|<br>               3||||
||||||||A|vera|ge =|3.2|me|dian|= 2|||
|||||||||||||||||


Set speed = 0.8
Average = 0.92, median = 0


Set speed = 0.3
Average = 3.2, median = 2


Figure 6: The distribution of _Uncertainty_ at different
layers of FastBERT in the Book review dataset: **(a)**
The _speed_ is set to 0.0, which means that all samples
will pass through all the twelve layers; **(b) ∼** **(d):** The
_Speed_ is set to 0.3, 0.5, and 0.8 respectively, iand only
the samples with Uncertainty higher than Speed will be
sent to the next layer.

**4.5.1** **Layer distribution**

In FastBERT, each sample walks through a different number of Transformer layers due to varied
complexity. For a certain condition, fewer executed
layers often requires less computing resources. As
illustrated in Figure 5, we investigate the distribution of exit layers under different constraint of
_Speeds (0.3, 0.5 and 0.8) in the book review dataset._
Take Speed = 0.8 as an example, at the first layer
_Transformer0, 61% of the samples is able to com-_
plete the inference. This significantly eliminates
unnecessary calculations in the next eleven layers.

**4.5.2** **Uncertainty distribution**

The distribution of sample uncertainty predicted by
different student classifiers varies, as is illustrated
in Figure 6. Observing these distributions help us to


Set speed = 0.5
Average = 2.3, median = 1


-----

88%

86%


Speed = 0.5


20,000M

15,000M


84%

82%


10,000M

5,000M


Table 3: Results of ablation studies on the Book review
and Yelp.P datasets.

**Book review** **Yelp.P**
**Config.**

FLOPs FLOPs
Acc. Acc.
(speedup) (speedup)

FastBERT

|Config.|Book review<br>FLOPs<br>Acc.<br>(speedup)|Yelp.P<br>FLOPs<br>Acc.<br>(speedup)|
|---|---|---|


80%

0 1 2 3 4 5 6 7 8

Fine-tuning Self-distillation
(0~3 epochs) (3~8 epochs)

Epoch

Figure 7: The change in accuracy and FLOPs of FastBERT during fine-tuning and self-distillation with the
Book review dataset. The accuracy firstly increases at
the fine-tuning stage, while the self-distillation reduces
the FLOPs by six times with almost no loss in accuracy.

|speed=0.2<br>speed=0.7|9725M<br>86.98<br>(2.23x)<br>3621M<br>85.69<br>(6.01x)|52783M<br>95.90<br>(4.12x)<br>2757M<br>94.67<br>(7.90x)|
|---|---|---|


FastBERT without self-distillation

|speed=0.2<br>speed=0.7|9921M<br>86.22<br>(2.19x)<br>4282M<br>85.02<br>(5.08x)|4173M<br>95.55<br>(5.22x)<br>2371M<br>94.54<br>(9.18x)|
|---|---|---|


further understand FastBERT. From Figure 6(a), it
can be concluded that the higher the layer is posited,
the lower the uncertainty with given Speed will be,
indicating that the high-layer classifiers more decisive than the lower ones. It is worth noting that
at higher layers, there are samples with uncertainty
below the threshold of Uncertainty (i.e., the Speed),
for these high-layer classifiers may reverse the previous judgments made by the low-layer classifiers.

**4.5.3** **Convergence of self-distillation**
Self-distillation is a crucial step to enable FastBERT. This process grants student classifiers with
the abilities to infer, thereby offloading work from
the teacher classifier. Taking the Book Review
dataset as an example, we fine-tune the FastBERT
with three epochs then self-distill it for five more
epochs. Figure 7 illustrates its convergence in
accuracy and FLOPs during fine-tune and selfdistillation. It could be observed that the accuracy
increases with fine-tuning, while the FLOPs decrease during the self-distillation stage.

**4.6** **Ablation study**


FastBERT without adaptive inference

11123M 11123M
layer=6 86.42 95.18
(1.95x) (1.95x)

3707M 3707M
layer=2 82.88 93.11
(5.87x) (5.87x)

From Table 3, we have observed that: **(1)** At
almost the same level of speedup, FastBERT without self-distillation or adaption performs poorer;
**(2) When the model is accelerated more than five**
times, downstream accuracy degrades dramatically without adaption. It is safe to conclude that
both the adaptation and self-distillation play a key
role in FastBERT, which achieves both significant
speedups and favorable low losses of accuracy.

|layer=6<br>layer=2|11123M<br>86.42<br>(1.95x)<br>3707M<br>82.88<br>(5.87x)|11123M<br>95.18<br>(1.95x)<br>3707M<br>93.11<br>(5.87x)|
|---|---|---|

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
|||||||||Acc.<br>FLO|Ps||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|Fine-tuning<br>(0~3 epochs)<br>Self-distillation<br>  (3~8 epochs)<br>Epoch<br>0<br>1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>7: The change in accuracy and FLO<br>during ﬁne-tuning and self-distillatio<br> eview dataset. The accuracy ﬁrstly i<br> -tuning stage, while the self-distillati<br> Ps by six times with almost no loss i<br> understand FastBERT. From Figu<br>  concluded that the higher the layer<br> er the uncertainty with given_ Spe_<br>ing that the high-layer classiﬁers<br> than the lower ones. It is worth n<br> er layers, there are samples with u<br> he threshold of_ Uncertainty_ (i.e., t<br> se high-layer classiﬁers may rever<br> udgments made by the low-layer<br>**Convergence of self-distillation**<br>stillation is a crucial step to en<br> This process grants student classi<br> lities to infer, thereby ofﬂoading<br>cher classiﬁer. Taking the Boo<br> as an example, we ﬁne-tune the|


Adaptation and self-distillation are two crucial
mechanisms in FastBERT. We have preformed ablation studies to investigate the effects brought
by these two mechanisms using the Book Review dataset and the Yelp.P dataset. The results
are presented in Table 3, in which ‘without selfdistillation’ implies that all classifiers, including
both the teacher and the students, are trained in
the fine-tuning; while ‘without adaptive inference’
means that the number of executed layers of each
sample is fixated to two or six.


### 5 Conclusion

In this paper, we propose a fast version of BERT,
namely FastBERT. Specifically, FastBERT adopts
a self-distillation mechanism during the training
phase and an adaptive mechanism in the inference
phase, achieving the goal of gaining more efficiency with less accuracy loss. Self-distillation
and adaptive inference are first introduced to NLP
model in this paper. In addition, FastBERT has a
very practical feature in industrial scenarios, i.e.,
its inference speed is tunable.
Our experiments demonstrate promising results
on twelve NLP datasets. Empirical results have
shown that FastBERT can be 2 to 3 times faster
than BERT without performance degradation. If
we slack the tolerated loss in accuracy, the model is
free to tune its speedup between 1 and 12 times. Besides, FastBERT remains compatible to the parameter settings of other BERT-like models (e.g., BERTWWM, ERNIE, and RoBERTa), which means
these public available models can be readily loaded


-----

for FastBERT initialization.

### 6 Future work

These promising results point to future works in (1)
linearizing the _Speed-Speedup_ curve; **(2)** extending this approach to other pre-training architectures
such as XLNet (Yang et al., 2019) and ELMo (Peters et al., 2018); (3) applying FastBERT on a wider
range of NLP tasks, such as named entity recognition and machine translation.

### Acknowledgments

This work is funded by 2019 Tencent Rhino-Bird
Elite Training Program. Work done while this author was an intern at Tencent.

### References

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin,
Ziqing Yang, Shijin Wang, and Guoping Hu. 2019.
Pre-training with whole [word](https://arxiv.org/pdf/1906.08101.pdf) masking for chinese
[BERT.](https://arxiv.org/pdf/1906.08101.pdf) _arXiv preprint arXiv:1906.08101._

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: [Pre-training](https://doi.org/10.18653/v1/N19-1423) of
deep bidirectional [transformers](https://doi.org/10.18653/v1/N19-1423) for language under[standing.](https://doi.org/10.18653/v1/N19-1423) In Proceedings of ACL, pages 4171–4186.

Michael Figurnov, Maxwell D Collins, Yukun Zhu,
Li Zhang, Jonathan Huang, Dmitry Vetrov, and Ruslan Salakhutdinov. 2017. [Spatially adaptive compu-](http://openaccess.thecvf.com/content_cvpr_2017/papers/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.pdf)
[tation time for residual networks.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.pdf) In Proceedings of
_CVPR, pages 1790–1799._

Yunchao Gong, Liu Liu, Ming Yang, and Lubomir
Bourdev. 2014. Compressing [deep](https://arxiv.org/pdf/1412.6115.pdf) convolutional
networks using [vector](https://arxiv.org/pdf/1412.6115.pdf) quantization. _arXiv_ _preprint_
_arXiv:1412.6115._

Alex Graves. 2016. Adaptive [computation](https://arxiv.org/pdf/1603.08983v4.pdf) time
for recurrent [neural](https://arxiv.org/pdf/1603.08983v4.pdf) networks. _arXiv_ _preprint_
_arXiv:1603.08983._

Song Han, Jeff Pool, John Tran, and William Dally.
2015. Learning both [weights](https://arxiv.org/pdf/1506.02626v3.pdf) and connections for
efficient [neural](https://arxiv.org/pdf/1506.02626v3.pdf) network. In _Advances_ _in_ _NeurIPS,_
pages 1135–1143.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.

[Distilling the knowledge in a neural network.](https://arxiv.org/pdf/1503.02531.pdf) _Com-_
_puter Science, 14(7):38–39._

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang,
Xiao Chen, Linlin Li, Fang Wang, and Qun
Liu. 2019. TinyBERT: [Distilling](https://arxiv.org/pdf/1909.10351.pdf) BERT for
natural [language](https://arxiv.org/pdf/1909.10351.pdf) understanding. _arXiv_ _preprint_
_arXiv:1909.10351._

Yoon Kim. 2014. Convolutional [neural](https://doi.org/10.3115/v1/D14-1181) networks for
sentence [classification.](https://doi.org/10.3115/v1/D14-1181) In _Proceedings_ _of_ _EMNLP,_
pages 1746–1751.


Olga Kovaleva, Alexey Romanov, Anna Rogers, and
Anna Rumshisky. 2019. Revealing [the](https://arxiv.org/pdf/1908.08593.pdf) dark se[crets of BERT.](https://arxiv.org/pdf/1908.08593.pdf) In Proceedings of EMNLP-IJCNLP,
pages 4356–4365.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut.
2019. ALBERT: A lite [BERT](https://arxiv.org/pdf/1909.11942.pdf) for self-supervised
[learning of language representations.](https://arxiv.org/pdf/1909.11942.pdf) _arXiv preprint_
_arXiv:1909.11942._

Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju,
Haotang Deng, and Ping Wang. 2020. [K-BERT:](https://arxiv.org/abs/1909.07606)
Enabling language [representation](https://arxiv.org/abs/1909.07606) with knowledge
[graph.](https://arxiv.org/abs/1909.07606) In Proceedings of AAAI.

Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng,
Jing Chen, Dongfang Li, and Buzhou Tang. 2018.
Lcqmc: A large-scale [chinese](https://www.aclweb.org/anthology/C18-1166.pdf) question matching
[corpus.](https://www.aclweb.org/anthology/C18-1166.pdf) In _Proceedings_ _of_ _the_ _ICCL,_ pages 1952–
1962.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
RoBERTa: A robustly [optimized](https://arxiv.org/pdf/1907.11692.pdf) BERT pretraining
[approach.](https://arxiv.org/pdf/1907.11692.pdf) _arXiv preprint arXiv:1907.11692._

Ilya Loshchilov and Frank Hutter. Fixing weight
decay [regularization](https://arxiv.org/pdf/1711.05101.pdf) in adam. _arXiv_ _preprint_
_arXiv:1711.05101._

Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt
Gardner, Christopher Clark, Kenton Lee, and Luke
Zettlemoyer. 2018. Deep [contextualized](https://www.aclweb.org/anthology/N18-1202/) word rep[resentations.](https://www.aclweb.org/anthology/N18-1202/) In Proceedings of NAACL-HLT, pages
2227–2237.

Yuanyuan Qiu, Hongzheng Li, Shen Li, Yingdi Jiang,
Renfen Hu, and Lijiao Yang. 2018. [Revisiting](https://link.springer.com/chapter/10.1007/978-3-030-01716-3_18) cor[relations between intrinsic and extrinsic evaluations](https://link.springer.com/chapter/10.1007/978-3-030-01716-3_18)
[of word embeddings.](https://link.springer.com/chapter/10.1007/978-3-030-01716-3_18) In Proceedings of CCL, pages
209–221. Springer.

Alec Radford, Karthik Narasimhan, Tim Salimans, and
Ilya Sutskever. 2018. Improving [language](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) understanding by [generative](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) pre-training. _Technical_ _re-_
_port, OpenAI._

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine
Lee, Sharan Narang, Michael Matena, Yanqi Zhou,
Wei Li, and Peter J Liu. 2019. [Exploring the limits](https://arxiv.org/pdf/1910.10683.pdf)
of transfer learning with [a](https://arxiv.org/pdf/1910.10683.pdf) unified text-to-text trans[former.](https://arxiv.org/pdf/1910.10683.pdf) _arXiv preprint arXiv:1910.10683._

Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua
Bengio. 2014. Fitnets: Hints [for](https://arxiv.org/pdf/1412.6550v2.pdf) thin deep nets.
_arXiv preprint arXiv:1412.6550._

Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. 2019. DistilBERT, [a](https://arxiv.org/pdf/1910.01108.pdf) distilled version of bert: smaller, [faster,](https://arxiv.org/pdf/1910.01108.pdf) cheaper and lighter. In
_NeurIPS EMC2 Workshop._


-----

Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. 2019a.

Patient knowledge [distillation](https://www.aclweb.org/anthology/D19-1441.pdf) for bert model com[pression.](https://www.aclweb.org/anthology/D19-1441.pdf) In Proceedings of EMNLP-IJCNLP, pages
4314–4323.

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi
Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao
Tian, and Hua Wu. 2019b. ERNIE: [Enhanced](https://arxiv.org/pdf/1904.09223.pdf) representation through [knowledge](https://arxiv.org/pdf/1904.09223.pdf) integration. _arXiv_
_preprint arXiv:1904.09223._

Surat Teerapittayanon, Bradley McDanel, and HsiangTsung Kung. 2016. Branchynet: [Fast](https://ieeexplore.ieee.org/abstract/document/7900006) inference via
[early exiting from deep neural networks.](https://ieeexplore.ieee.org/abstract/document/7900006) In Proceed_ings of ICPR, pages 2464–2469._

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. [Attention](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) is all
you need. In _Advances_ _in_ _NeurIPS,_ pages 5998–
6008.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018.
GLUE: A multi-task [benchmark](https://www.aclweb.org/anthology/W18-5446/) and analysis platform for natural [language](https://www.aclweb.org/anthology/W18-5446/) understanding. In _Pro-_
_ceedings of EMNLP, pages 353–355._

Baoxin Wang. 2018. Disconnected [recurrent](https://doi.org/10.18653/v1/P18-1215) neural
networks for [text](https://doi.org/10.18653/v1/P18-1215) categorization. In _Proceedings_ _of_
_ACL, pages 2311–2320._

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le.
2019. XLNet: Generalized [autoregressive](https://arxiv.org/abs/1906.08237v1) pretraining for [language](https://arxiv.org/abs/1906.08237v1) understanding. _arXiv_ _preprint_
_arXiv:1906.08237._

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015.

[Character-level convolutional networks for text clas-](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
[sification.](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) In Advances in NeurIPS, pages 649–657.

Zhe Zhao, Hui Chen, Jinbin Zhang, Xin Zhao, Tao
Liu, Wei Lu, Xi Chen, Haotang Deng, Qi Ju, and
Xiaoyong Du. 2019. UER: An [open-source](https://arxiv.org/abs/1909.05658) toolkit
[for pre-training models.](https://arxiv.org/abs/1909.05658) In Proceedings of EMNLP_IJCNLP 2019, page 241._


-----

