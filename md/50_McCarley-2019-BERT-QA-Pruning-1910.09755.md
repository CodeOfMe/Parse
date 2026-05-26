## Structured Pruning of BERT-based Question Answering Models

### J.S. McCarley and Rishav Chakravarti and Avirup Sil IBM Research AI Yorktown Heights, NY {jsmc,rchakravarti,avi}@us.ibm.com


### Abstract


The recent trend in industry-setting Natural
Language Processing (NLP) research has been
to operate large pretrained language models
like BERT under strict computational limits.
While most model compression work has focused on “distilling" a general-purpose language representation using expensive pretrain_ing_ _distillation,_ less attention has been paid
to creating smaller task-specific language representations which, arguably, are more useful
in an industry setting. In this paper, we investigate compressing BERT- and RoBERTabased question answering systems by structured pruning of parameters from the underlying transformer model. We find that an inexpensive combination of _task-specific_ _struc-_
_tured_ _pruning_ and _task-specific_ _distillation,_
without the expense of _pretraining_ _distilla-_
_tion,_ yields highly-performing models across
a range of speed/accuracy tradeoff operating
points. We start from existing full-size models trained for SQuAD 2.0 or Natural Questions and introduce gates that allow selected
parts of transformers to be individually eliminated. Specifically, we investigate (1) structured pruning to reduce the number of parameters in each transformer layer, (2) applicability
to both BERT- and RoBERTa-based models,
(3) applicability to both SQuAD 2.0 and Natural Questions, and (4) combining structured
pruning with distillation. We achieve a neardoubling of inference speed with less than a
0.5 F1-point loss in short answer accuracy on
Natural Questions.

### 1 Introduction


While knowledge distillation from large pretrained
language models (e.g. BERT-large as a teacher)
has mitigated some of the computational burdens
of these models, computationally expensive _pre-_
_training distillation unnecessarily limits the ability_
of efficient student models to adopt the latest innovations in pretrained language models and transformer architecture. In this paper, we show that


a combination of _task-specific_ _structured_ pruning and _task-specific_ _distillation,_ yields highlyperforming compressed versions of existing models
across a range of speed/accuracy tradeoff operating points, without the expense of revisiting the
pretraining data.
Among Natural Language Processing (NLP)
tasks, question answering (QA), in particular, has
immediate applications in real-time systems. A
relatively new field in the open domain question
answering (QA) community is machine reading
comprehension (MRC) which aims to read and
comprehend a given text, and then answer questions based on it. MRC is one of the key steps
for natural language understanding. MRC also has
wide applications in the domain of conversational
agents and customer service support. Transformerbased models have led to striking gains in accuracy on MRC tasks recently, as measured on the
SQuAD v1.1 (Pajpurkar et al., 2016) and SQuAD
v2.0 (Rajpurkar et al., 2018) leaderboards. We
briefly mention three MRC tasks: SQuAD v1.1
consists of reference passages from Wikipedia with
answers and questions constructed by annotators
after viewing the passage. SQuAD v2.0 augmented
the SQuAD v1.1 collection with additional questions that did not have answers in the reference passage. Natural Questions (NQ) (Kwiatkowski et al.,
2019) removed the observational bias by starting
from questions submitted to Google and providing
annotated answers from appropriate passages.
MRC seems to be a particularly difficult task to
speed up. While distillation papers have advertised
impressive speedups with near-negligible loss in
accuracy on GLUE benchmarks, published applications of distillation to MRC have been less impressive (often relegated to the appendix.) In Table
1, we compare the accuracies (F1 score) of Distilbert and TinyBert, two well-known compressions
of BERT, with baseline ("out-of-the-box" pretraining) BERT-large and BERT-base models on three


-----

|model|params|SQuAD 1.1|SQuAD 2.0|NQ|
|---|---|---|---|---|
|BERT-large<br>BERT-base<br>DistilBert<br>TinyBert|356M<br>125M<br>63M<br>63M|90.9 (c)<br>88.4 (a) 88.5 (b)<br>86.2 (a) 86.9 (b)<br>87.5(a)|83.52<br>76.4 (a)<br>69.5 (a)<br>73.4(a)|56.14<br>52.75<br>50.46<br>44.64|


Table 1: Comparison of published F1 scores of well-known distillation’s of BERT on several question-answering
tasks. Though not strictly comparable, we observe that on SQuAD 1.1 smaller models approach BERT-large in
accuracy, whereas the smaller models underperform notably on both SQuAD 2.0 and Natural Questions (NQ). We
show the short answer F1 for NQ. Sources: (a)=(Jiao et al., 2019), (b)=(Sanh et al., 2019), (c)=(Devlin et al., 2019).


MRC tasks, using the number of parameters as a
crude proxy for speed. [1] Compared to BERT-large,
models with fewer parameters achieved modest
losses on SQuAD 1.1. The shortfalls on the more
challenging SQuAD 2.0 were much larger. We
also note that the shortfalls of smaller models were
large on NQ. SQuAD is also seen as a worst-case
performance loss for speed up techniques based
on quantization, (Shen et al., 2020) while the difficulty of distilling a SQuAD model (compared to
sentence-level GLUE tasks) is acknowledged in
(Jiao et al., 2019). We speculate that these difficulties are because answer selection via pointer
networks requires token level predictions rather
than passage level classification, and requires long
range attention between query and passage.
The contributions of this paper are

1. Application of structured pruning techniques
to the hidden dimension of the feed-forward
layer, not just the attention heads (Michel
et al., 2019),

2. the combination of distillation and pruning,

3. thereby significantly pruning the MRC system
with minimal loss of accuracy and considerable speedup, all without the expense of revisiting pretraining (Sanh et al., 2019; Jiao et al.,
2019)

Furthermore we survey multiple pruning techniques (both heuristic and trainable) and provide
recommendations specific to transformer-based
question answering models. We focus exclusively
on structured pruning (Anwar et al., 2017) to avoid
sparsity issues. During the course of the investigation, we also learn that an optimal pruning learns

1MobileBert (Sun et al., 2020) required extensive pretraining architecture search experiments in order to customize the
teacher model, and does not represent a fair comparison when
the goal is to compress existing models.


a structure consisting of non-identical transformers, namely lightweight transformers near the top
and bottom while retaining more complexity in the
intermediate layers, instead of the typically 12-24
layers of identically sized transformers, common
in widely distributed pre-trained models

### 2 Related work

The field of neural networks compression has been
extensively reviewed in (O’Neill, 2020). Here we
focus on results relevant to MRC. While distillation
(student-teacher) of BERT has produced notably
smaller and faster models (Tang et al., 2019; Turc
et al., 2019; Tsai et al., 2019; Yang et al., 2019), the
focus has been on passage level annotation tasks
(e.g. GLUE) that do not require long-range attention links between query and passage.
Distillation of typical MRC models has been
much more limited: DistilBERT (Sanh et al.,
2019) used pretraining distillation to obtain 60%
speedups on GLUE tasks while retaining 97% of
the accuracy. However, MRC results, after additional task-specific distillation, included a modest
speedup and small performance loss on SQuAD
1.1. TinyBERT (Jiao et al., 2019) used both
pretraining and task-specific distillation to obtain
9.4× speedups on GLUE. However, they restricted
SQuAD evaluation to using BERT-base as a teacher,
and deferred deeper investigation to future work.
MobileBERT (Sun et al., 2020) obtains strong results after an extensive architecture search in order
to construct a teacher model with custom architecture which is both pre-trained and used for pretraining distillation of the student model. This approach
represents a notable increase in pretraining expense,
and is further removed from this paper’s goal of
shrinking existing models. (Turc et al., 2019) investigated pretraining and distilling smaller models
from scratch, but tested only on passage-level annotation tasks. The authors are not aware of any


-----

results from distilled models on NQ.

Investigations into pruning BERT have also omitted MRC. Michel et al. (2019) applied simple
gating heuristics to prune BERT attention heads
and achieve speedups on MT and MNLI. Voita et
al. (2019) introduced _L0_ regularization to BERT
while focusing on linguistic interpretability of attention heads but did not report speedups. _L0 regular-_
ization was combined with matrix factorization to
prune transformers for classification in (Wang et al.,
2019). Gale et al. (2019) induced unstructured sparsity on a transformer-based MT model, but did not
report speedups. Kovaleva et al. (2019) also focused on interpreting attention, and achieved small
accuracy gains on GLUE tasks by disabling (but
not pruning) certain attention heads. Structured
pruning as a form of dropout is explored in (Fan
et al., 2020). They prune entire layers of BERT,
but suggest that smaller structures could also be
pruned. They evaluate on MT, language modeling,
and generation-like tasks, but not SQuAD.

Another set of approaches omit cross-attention
between documents and queries in the lower layers so that precomputed document representations
can be used at inference time. These approaches
report results only on SQuAD 1.1 (Cao et al., 2020)
and various IR tasks (Khattab and Zaharia, 2020;
MacAvaney et al., 2020), but not SQuAD 2.0 or
NQ.

Other approaches to speeding up transformers
include ALBERT (Lan et al., 2020), which shared
parameters across layers in order to accelerate training, but did not report timings of inference.

QBERT (Shen et al., 2020) and Q8BERT (Zafrir
et al., 2019) aggressively quantized floating point
calculations to ultra-low precision in order to compress BERT. They noted that SQuAD was harder
to quantize (greater performance drop) than other
tasks.

Finally, (Li et al., 2020) investigated both unstructured pruning and quantization of RoBERTa
as a function of model size, and found that both
pruning and quantization were complementary, an
important reminder that multiple types of compression are not mutually exclusive. Very recently, (Kim and Hassan, 2020) combined distillation, structured pruning, and quantization and
achieved impressive speedup on both CPU and
GPU on GLUE tasks, but did not report results
on SQuAD/NQ-style question answering.


### 3 Pruning transformers

**3.1** **Gate placement**

Our approach to pruning consists of inserting additional trainable parameters, masks, into a transformer. The value of each mask variable controls
whether an entire block of transformer parameters (e.g. an attention head) is used by the model.
Specifically, each mask is a vector of gate variables γi _∈_ [0, 1], pointwise multiplied into a slice
of transformer parameters, where γi = 1 allows a
slice to remain active, and γi = 0 deactivates the
slice. We insert two types of masks into each transformer. We describe the placement of each mask
with the terminology of (Vaswani et al., 2017), indicating relevant sections of that paper.
In each self-attention sublayer, we place a mask,
Γ[attn] of size nH which selects attention heads to
remain active. (section 3.2.2)
In each feed-forward sublayer, we place a mask,
Γ[ff] of size _dI_ which selects ReLU/GeLU activations to remain active. (section 3.3)
Here nH is the number of heads per transformer
layer (12 or 16), _dE_ is the the size of the embeddings (768 or 1024) as well as the inner hidden
dimension, and dI is the size of the intermediate
activations in the feed-forward part of the transformer (3072 or 4096.) Sizes are for (BERT-base,
BERT-large).

**3.2** **Determining Gate Values**

We investigate four approches to determining the
gate values.
(1) Random: each γi is sampled from a Bernoulli
distribution of parameter α, where α is manually
adjusted to control the sparsity. This method is the
naive baseline, and is expected to be worse than
other methods.
(2) Gain: We follow the method of (Michel et al.,
2019) and estimate the influence of each gate γi on
the training set likelihood L by treating each γi as
a continuous parameter and computing the mean


(“head importance score”) during one pass over the
training data. We threshold gi to determine which
transformer slices to retain.
(4) L0 regularization: Following the method described in (Louizos et al., 2018), the gate variables
_γi are sampled_

_γi_ _∼_ hc(αi) (2)


_∂L_
_gi_ =
_∂γi_
����


(1)
����γi=1


-----

**Algorithm 1 Pruning an L0 regularized model:** ff(Sq) + attn(Sq) + retrain(Sq)

**Require:** _⟨BERTQA, D, λ[attn], λ[ff]_

{ BERTQA is an already-trained BERT question answering model that will be pruned, D is question-answering (SQuAD)
training data, λ[attn] and λ[ff] are penalty weights that determine how much to prune }
1: _αi[attn]_ _←⟨BERTQA, D⟩_ _�_ train attention gate parameters by optimizing L + attn(λ[attn])
2: _αi[ff]_ _[←⟨][BERT][QA][, D][⟩]_ _�_ train feed-forward gate parameters by optimizing L + ff (λ[ff] )
3: Γ[attn] _←_ _threshold(α[attn])_ _�_ select final gate values for attention heads
4: Γ[ff] _←_ _threshold(α[ff]_ ) _�_ select final gate values for feed forward heads
5: _BERTQA[′]_ _←⟨BERTQA, Γ[attn]⟩_ _�_ prune the attention heads
6: _BERTQA[′′]_ _←⟨BERTQA[′], Γ[ff]⟩_ _�_ prune the feedforward layers
7: _BERTQA[′′′]_ _←⟨BERTQA[′′], D⟩_ _�_ continued training of remaining BERT parameters subject to L


from a hard-concrete distribution hc(αi) (Maddison et al., 2017) parameterized by a corresponding variable _αi_ _∈_ R. The _αi_ are trained by optimizing the task-specific objective function L (typically cross-enropy) penalized in proportion to the
number of expected instances of γ = 1, with proportionality constants _λ[attn]_ in the penalty terms
attn(λ[attn]), e.g.

_Lcross-entropy + attn(λ[attn]) = Lcross-entropy_


that are round numbers rather than strictly respecting the threshold.

**3.4** **Extended training**

As noted by (Anwar et al., 2017), the task-specific
training of all parameters of a pruned model may
be continued further with the (unpenalized) taskspecific objective function L. In some experiments
we continue training by incorporating distillation:
the unpruned model is the teacher, and the pruned
model is the student.
In summary, the entire pruning procedure, starting from a trained model for an MRC task, consists
of

1. inserting masks into each transformer layer

2. determining values of the masks, either heuristically (methods (1)-(3)) or training them with
penalized objective functions (method (4))

3. replacing transformer parameter matrices with
smaller matrices, pruned according the masks
determined in the previous step

4. Either

     - continued training of the pruned transformer parameters with the original objective function (retrain)

     - _or continued training with a distillation_
objective function (distill), using the
original unpruned model as the teacher

This algorithm is presented in pseudocode in
Algorithm 1.

### 4 Experiments

**4.1** **Overall Setup and outline**

We evaluate our proposed method on two benchmark QA datasets: SQuAD 2.0 (Rajpurkar et al.,
2018) and Natural Questions (NQ) (Kwiatkowski


� (3)


_−_ _λ[attn]Ehc_


�� _δγi,1_

_i_


(and similarly for the ff(λ[ff] ).) The λ are hyperparameters controlling the sparsity. The expectation
is over the same hard-concrete distribution from
which we sample. We resample the γi with each
minibatch. This objective function is differentiable
with respect to the αi because of the reparameterization trick. (Kingma and Welling, 2014; Rezende
et al., 2014) The αi are updated by backpropagation
for up to one training epoch on the task training
data, with all other transformer parameters held
fixed. The final values for the gates γi are obtained
by thresholding the αi. We note that either the loglikelihood or a distillation-based objective can be
penalized as in Eq. (3). The cost of training the
gate parameters is comparable to extending fine
tuning for an additional epoch.

**3.3** **Structured Pruning**

After the values of the _γi_ have been determined
by one of the above methods, we prune the model.
Attention heads corresponding to γi[attn] = 0 are removed. Slices of both linear transformations in the
feed-forward sublayer which correspond to γi[ff] [= 0]
are removed. The pruned model no longer needs
masks, and now consists of smaller transformers of
varying, _non-identical sizes._ For experiments on
some hardware, matrices are forced to have sizes


-----

et al., 2019). SQuAD 2.0 is a dataset of questions
from Wikipedia passages, proposed by human annotators while viewing these Wikipedia passages.
NQ is a dataset of Google search queries with answers from Wikipedia pages provided by human
annotators. Of the two, NQ is more natural, as the
questions were asked by humans on Google without having seen the passage. On the other hand,
SQuAD annotators read the Wikipedia passage first
and then formulated the questions.
We address several empirical questions here: 1.
Do techniques developed on BERT-base transfer
to BERT-large? 2. Do the proposed techniques
transfer across datasets? 3. Does incorporating a
distillation objective further improve our model’s
performance?
To answer these we tune our hyper-parameters
on a subset of SQuAD 2.0 using a BERT-base
model, and then test them on the full SQuAD
2.0 with a BERT-large model. Further, we show
that the same techniques are applicable on the NQ
dataset not just with BERT but also with RoBERTa.
Finally we show that incorporating distillation
achieves even stronger and more flexible results.
When practical we report numbers as an average of
5 seeds.

**4.2** **SQuAD 2.0**

**4.2.1** **Experimental Setup and**
**hyper-parameters**

For selection of hyper-parameters (learning rate
and penalty weight exploration) and in order to minimize overuse of the official dev-set, we use 90%
of the official SQuAD 2.0 training data for training gates, and report results on the remaining 10%.
This resulting model (base-qa) is initialized from a
_bert-base-uncased SQuAD 2.0 system trained on_
the 90% with a baseline performance of F1 = 75.0
on the 10% dataset. Experiments described were
implemented using code from the HuggingFace
repository (Wolf et al., 2019) and incorporated either bert-base-uncased or bert-large-uncased with
a standard task-specific head.
Our final SQuAD 2.0 model (large-qa) use the
standard train/dev split of SQuAD 2.0 and is initialized from a bert-large-uncased system trained
with the method described in (Glass et al., 2019). It
achieves an F1 = 84.6 on the official dev set, somewhat exceeding "out-of-the-box" BERT question
answering models.
The gate parameters of the L0 regularization ex

Figure 1: Comparison of pruning methods on SQuAD
2.0: F1 vs percentage of attention heads and feed forward activations pruned from base-qa

periments are trained for one epoch starting from
the models above, with all transformer and embedding parameters fixed. We investigated learning
rates of 10[−][3], 10[−][2], and 10[−][1] on _base-qa,_ and
chose 10[−][1] for presentation and results on large_qa. This is notably larger than typical learning rates_
to tune BERT parameters. We used a minibatch
size of 24 and otherwise default hyperparameters
of the BERT-Adam optimizer. We used identical
parameters for our _large-qa_ experiments, except
with gradient accumulation of 3 steps.

**4.2.2** **Accuracy as function of pruning**

In Figure 1 we plot the base-qa F1 as a function
of the percentage of heads removed. The performance of ‘random‘ decays abruptly. ’Gain’ is
better.L0regularization is best, allowing 48% pruning at a cost under 5 F1-points.
Also in Figure 1 we plot the (accuracy) F 1 of
removing feed-forward activations. We see broadly
similar trends as above, except that the performance
is robust to even larger pruning. As before _L0_
regularization is best, allowing 70% pruning at cost
under 5 F1-points.

**4.2.3** **Validating these results**

On the basis of the development experiments, we
select an operating point, namely the largest values of _λ[attn]_ and _λ[ff]_ with _<_ 5 F1-point loss. After rescaling to the larger model size, we denote
the weights as _λ[a][∗]_ = 1.875 × 10[−][3] and _λ[f]_ _[∗]_ =
7.5 × 10[−][6]. We train the feed-forward and attention gates of large-qa with these penalties, as well
as multiples 2×, 3×, and 4×. The decoding times,
accuracies, and model sizes are summarized in Table 2. Accuracies are medians of 5 seeds, and
timings are medians of 5 decoding runs with the
median seed, on a single Nvidia K80 with batch
size 1. Models in which both attention and feed

-----

|pruning of λattn λff<br>BERT-large λa∗ λf∗|time F1 F1 % attn %ff size<br>sec. +retrain no retrain removed removed (MiB)|
|---|---|
|_a_: no pruning<br>0<br>0<br>_b_ : attn(Sq)<br>1<br>0<br>_c_ : ﬀ(Sq)<br>0<br>1|2712<br>84.6<br>0<br>0<br>1279<br>2288<br>84.2<br>44.3<br>0<br>1112<br>2103<br>83.2<br>0<br>48.1<br>908|
|_d_ : ﬀ(Sq) + attn(Sq)<br>1<br>1<br>_e_ : ﬀ(Sq) + attn(Sq)<br>2<br>2<br>_f_ : ﬀ(Sq) + attn(Sq)<br>3<br>3<br>_g_ : ﬀ(Sq) + attn(Sq)<br>4<br>4|1667<br>83.7<br>82.6<br>44.0<br>48.1<br>740<br>1391<br>83.2<br>80.9<br>53.1<br>64.9<br>576<br>1213<br>82.4<br>76.8<br>57.6<br>73.7<br>492<br>1128<br>81.5<br>67.8<br>60.1<br>78.4<br>441|


Table 2: Decoding times, accuracies on SQuAD 2.0, and space savings achieved at sample operating points of
pruned BERT large-qa, with and without continued training.

(2) Do pruning techniques developed for BERT
also apply to RoBERTa?
(3) Can we combine distillation and pruning to
achieve even smaller, faster models?


Figure 2: Percentage of attention heads and feed forward activations remaining after pruning, by layer

forward components are pruned were built from the
_independently trained gate configurations of atten-_
tion and feed forward. For corresponding penalty
weights, the _large-qa_ was pruned somewhat less
than base-qa, and the F 1 loss due to pruning was
smaller.
Much of the loss in accuracy is recovered by continuing the training for an additional epoch (column
5) after the pruning, even though the accuracy without retraining (column 6) decreases substantially as
more is pruned. We highlight the operating point
of Table 2, row e, which after continued training,
loses less than 1.5 F 1 points, while nearly doubling
the decoding speed.

**4.2.4** **Impact of pruning each layer**

In Fig. 2 we show the percentage of attention heads
and feed forward activations remaining after pruning, by layer. We see that intermediate layers retained more, while layers close to the embedding
and close to the answer were pruned more heavily.

**4.3** **Natural Questions**

We address three questions in this section:
(1) Are the pruning techniques developed for the
SQuAD 2.0 task also applicable to the NQ task?


**4.3.1** **Transfer of gates**
We take the pruned BERT-large models described
above and use the identical model parameters as
the initialization for continued training (using the
cross-entropy objective function) of an NQ model.
In other words, the gate variables are trained on
SQuAD 2.0, and the only use of the NQ training
data is in the continued training of the remaining
transformer parameters, denoted _retrain(NQ)._
The results shown in Table 3, while far from optimal, are encouraging. They suggest that the redundancies in BERT that are removed by pruning
are not task-specific or domain-specific and that a
pruned model is relatively robust.

**4.3.2** **RoBERTa**
RoBERTa-based models have achieved notably
higher accuracy than BERT-based models across a
variety of tasks (Liu et al., 2019), including MRC.
For example, on NQ short answers, our RoBERTalarge model achieves 58.8 - over 4 F1-points better than the comparable BERT-large model, which
achieved 54.7. RoBERTa has the same topology
as BERT. It differs slightly in such aspects as tokenization, training data (during pretraining) and
training procedure. The nature of these differences
suggests that the pruning techniques developed for
BERT should continue to work largely unchanged
with RoBERTa. However, as noted by (Liu et al.,
2019), BERT is significantly undertrained, which
raises the concern that RoBERTa might achieve
its better performance by more effectively utilizing
the transformer parameters that were under-utilized
and prunable in BERT.
We pruned this RoBERTa-large NQ model, using the same techniques as described above, select

-----

|pruning of λattn λff<br>BERT-large λa∗ λf∗|% attn<br>removed|%ff<br>removed|LA<br>(F1)|SA<br>(F1)|
|---|---|---|---|---|
|no pruning<br>0<br>0<br>ﬀ(Sq) + attn(Sq) + retrain(NQ)<br>2<br>2<br>ﬀ(Sq) + attn(Sq) + retrain(NQ)<br>4<br>4|0<br>44<br>53|0<br>48<br>65|66.1<br>65.9<br>64.2|54.7<br>51.7<br>49.6|


Table 3: NQ accuracy of BERT models pruned on SQuAD, continued cross-entropy training on NQ


ing the gate values by _L0_ regularization for one
epoch on approximately 20% of the NQ training
data, and continued training for an epoch on the
full NQ training set. In Table 4 we show the accuracy and the amount pruned. We found that to
have a similar percentage of parameters pruned,
we needed smaller values of _λ[attn]_ and _λ[ff]_ when
training the pruning on NQ, compared to training
the pruning on SQuAD. The loss in accuracy for
comparable amounts of pruning is similar to that
observed in BERT/SQuAD experiments, indicating
that RoBERTa models can be pruned successfully
with these techniques.

**4.3.3** **Combining distillation and pruning**

The simplest way to combine distillation with pruning is, after the model has been pruned, to replace the continued training (retrain(NQ)) by
continued training (distill(NQ)) with a distillation objective. Here the unpruned model acts as
the teacher and the pruned model is the student. In
Table 5, we show results using distillation only in
the continued training phase. Line c is especially
notable - a 2.9 F1-point gain compared to line c in
Table 4, with less than 0.5 F1-point loss relative to
unpruned, while approaching a doubling of speed.
Timings are median of 5 decoding runs over the
entire NQ developement set on an NVidia V100
using 16-bit floating point with batch size 64. In
this experiment, matrices were forced to have sizes
that are round numbers, resulting in small changes
(< 1%) in reported pruning fractions. We also include for comparison RoBERTa-base model (line
_e) that has been similarly distilled using RoBERTa-_
large as a teacher.
Alternately, the pruning phase itself may be
driven by a distillation objective. Here we replace
the cross-entropy term in Eq.(3) with a distillation
objective function, and prune the model based on
the modified objective function. We will denote
distillation-driven pruning _prune(distillation),_
in contrast to _prune(cross-entropy)._ All experiments with _prune(distillation)_ involved distillation continued training distill(NQ). When the


pruning phase itself is driven by the distillation objective, the results are not directly comparable because the same values of λ[attn] and λ[ff] yield significantly less pruning for prune(distillation) than
for prune(cross-entropy).
In Fig. 3 we plot the performance of various pruned models as a function of the number of remaining parameters. (We have found
that the number of parameters is well-correlated
with the decoding time for this range of parameters.) The points labeled _prune(distillation)-_
_large represent various degrees of distillation prun-_
ing followed by _distill(NQ)_ continued training of a RoBERTa-large model. The points labeled _prune(cross-entropy)-large_ represent various degrees of cross-entropy pruning followed by
_distill(NQ) (corresponding to Table 5, rows b-d)_
of the same initial model. The unpruned RoBERTalarge model of Table 5, row a is the point unpruned_large at the far right of the graph._ The distillation
pruning does not provide a notable improvement
over the cross-entropy driven pruning, unlike the
case of distillation-driven continued training vs
cross-entropy driven continued training.
The point labeled base in Fig.3 is the RoBERTabase model (line _e_ of Table 5) trained with the
same distillation technique as our pruned models. It lies above and to the left of the envelope
of large prune(distillation) points, which suggests
that the pruning+distillation processes are not quite
achieving full potential. On the other hand, the
pruning+distillation processes offer more flexibility of operating points, without requiring expensive
masked language model pretraining at each size.
The pruning-distillation process may also be
applied to the RoBERTa-base model _base+dist,_
and this is illustrated by the points _prune(cross-_
_entropy)-base in Fig 3._ These points lie even further above and to the left of the envelope of base
points, and point the way to even smaller and faster
NQ models achievable by a combination of distillation and structured pruning. For comparison, the
results we have obtained for DistilBERT (50.46)
and TinyBERT (44.64) are at or below the bottom


-----

|pruning of λattn λff<br>BERT-large 4λa∗ 15λf∗|% attn<br>removed|%ff<br>removed|LA<br>(F1)|SA<br>(F1)|
|---|---|---|---|---|
|_a_ : no pruning<br>0<br>0<br>_b_ : ﬀ(NQ) + attn(NQ) + retrain(NQ)<br>1<br>2<br>_c_ : ﬀ(NQ) + attn(NQ) + retrain(NQ)<br>2<br>4<br>_d_ : ﬀ(NQ) + attn(NQ) + retrain(NQ)<br>4<br>10|0<br>42<br>53<br>68|0<br>40<br>56<br>75|70.3<br>68.3<br>67.8<br>65.2|58.8<br>57.7<br>55.5<br>52.2|


Table 4: NQ accuracy of RoBERTa models pruned on NQ, continued cross-entropy training on NQ

pruning of _λ[attn]_ _λ[ff]_ % attn %ff LA SA time
BERT-large 4λ[a][∗] 15λ[f] _[∗]_ removed removed (F1) (F1) sec.

_a : no pruning_ 0 0 0 0 0 70.3 58.8 2789
_b : ff(NQ) + attn(NQ) + distill(NQ)_ 1 2 42 40 69.8 58.4 1867
_c : ff(NQ) + attn(NQ) + distill(NQ)_ 2 4 53 55 69.3 58.4 1523
_d : ff(NQ) + attn(NQ) + distill(NQ)_ 4 10 68 75 67.6 55.4 1135
_e : RoBERTa-base_ NA NA NA NA 67.3 55.9 1151

Table 5: RoBERTa models pruned on NQ, continued training on NQ by distillation from unpruned model

with minimal lost of accuracy.

|pruning of λattn λff<br>BERT-large 4λa∗ 15λf∗|% attn<br>removed|%ff<br>removed|LA<br>(F1)|SA<br>(F1)|time<br>sec.|
|---|---|---|---|---|---|
|_a_ : no pruning<br>0<br>0<br>_b_ : ﬀ(NQ) + attn(NQ) + distill(NQ)<br>1<br>2<br>_c_ : ﬀ(NQ) + attn(NQ) + distill(NQ)<br>2<br>4<br>_d_ : ﬀ(NQ) + attn(NQ) + distill(NQ)<br>4<br>10|0 0<br>42<br>53<br>68|0<br>40<br>55<br>75|70.3<br>69.8<br>69.3<br>67.6|58.8<br>58.4<br>58.4<br>55.4|2789<br>1867<br>1523<br>1135|
|_e_ : RoBERTa-base<br>NA<br>NA|NA|NA|67.3|55.9|1151|


Figure 3: Short answer accuracy vs number of parameters (millions), contrasting distillation-driven pruning
with cross-entropy-driven pruning. (See text.)

edge of this graph.
Averaging across five different initializations
(random seeds) of gate parameters, a sample operating point for prune(distillation)-large has attention
heads pruned by 60.0 ± 1.0%, feed-forward activations pruned by 71.9 ± 0.1% yielding long-answer
(LA) F1 of 68.2 ± 0.2% and short-answer (SA) F1
of 56.2±0.2%. Similarly, a sample operating point
for prune(cross-entropy)-base has attention heads
pruned by 20.3 ± 1.7%, feed-forward activations
pruned by 17.7 _±_ 0.4%, yielding long-answer (LA)
F1 of 68.0 ± 0.2% and short-answer (SA) F1 of
57.0 ± 0.2%.

### 5 Conclusions

We investigate various methods to prune existing
transformer-based MRC models, and evaluate the
accuracy-speed tradeoff for these prunings. We find
that both the attention head layers and especially
the feed forward layers can be pruned considerably


We find that L0 regularization pruning is particularly effective for pruning these two transformer
components, compared to the more heuristic ’Gain’
method. The pruned feed-forward layer and the
pruned attention heads are easily combined. Especially after retraining, this combination yields
a considerably faster question answering model
with minimal loss in accuracy. One operating point
nearly doubles the decoding speed on SQuAD 2.0,
with a loss of less than 1.5 F1-points.

The same methods that worked with a BERTbased SQuAD 2.0 model also yield strong results
when applied to a RoBERTa-based NQ model. The
best performance is achieved by combining distillation with structured pruning. One operating point
almost doubles the inference speed of RoBERTalarge based model for Natural Questions, while
losing less than 0.5 F1-point on short answers,
less than 20% of the difference between baseline
RoBERTa-large and RoBERTa-base systems.

We emphasize that our method probes a wide
range of speed/accuracy operating points. It only
requires revisiting task-specific training data, an
expense comparable to fine-tuning, and does not
require revisiting transformer pretraining, a much
larger expense comparable to the original pretraining of a transformer model. Our method is robust
across both BERT- and RoBERTa-based models.
It is also robust across both SQuAD and NQ, despite the different biases incorporated into the construction of these datasets. Furthermore our observation that the resulting transformer layers are
non-identical may inform future efforts at pruning.


-----

### 6 Ethical Consideration

The methods described in this paper are able to
reduce the energy-intensiveness of transformer language models, both at runtime, and by reducing
the need for pretraining of such models. All experiments were done with publicly available data
sets that are not known to contain personally identifiable information. Although deployed question
answering system have the potential for misuse,
this work is not likely to affect this potential.

### References

Sajid Anwar, Kyuyeon Hwang, and Wonyong Sung.
2017. Structured pruning of deep convolutional neural networks. _J._ _Emerg._ _Technol._ _Comput._ _Syst.,_
13(3).

Qingqing Cao, Harsh Trivedi, Aruna Balasubramanian, and Niranjan Balasubramanian. 2020. DeFormer: Decomposing pre-trained transformers for
faster question answering. In _Proceedings_ _of_ _the_
_58th Annual Meeting of the Association for Compu-_
_tational_ _Linguistics,_ pages 4487–4497, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language understanding. In _Proceedings_ _of_ _the_ _2019_ _Conference_
_of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_
_for_ _Computational_ _Linguistics:_ _Human_ _Language_
_Technologies,_ _Volume_ _1_ _(Long_ _and_ _Short_ _Papers),_
pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Angela Fan, Edouard Grave, and Armand Joulin. 2020.
Reducing transformer depth on demand with structured dropout. In _International_ _Conference_ _on_
_Learning Representations._

Trevor Gale, Erich Elsen, and Sara Hooker. 2019. The
state of sparsity in deep neural networks. _CoRR,_
abs/1902.09574.

Michael Glass, Alfio Gliozzo, Rishav Chakravarti, Anthony Ferritto, Lin Pan, G P Shrivatsa Bhargav, Dinesh Garg, and Avirup Sil. 2019. Span selection pretraining for question answering.

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang,
Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.
2019. Tinybert: Distilling bert for natural language
understanding.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized
late interaction over bert. In Proceedings of the 43rd
_International_ _ACM_ _SIGIR_ _Conference_ _on_ _Research_
_and_ _Development_ _in_ _Information_ _Retrieval,_ SIGIR
’20, page 39–48, New York, NY, USA. Association
for Computing Machinery.


Young Jin Kim and Hany Hassan. 2020. FastFormers:
Highly efficient transformer models for natural language understanding. In Proceedings of SustaiNLP:
_Workshop on Simple and Efficient Natural Language_
_Processing, pages 149–158, Online. Association for_
Computational Linguistics.

Diederik P. Kingma and Max Welling. 2014. Autoencoding variational Bayes. _International_ _Confer-_
_ence on Learning Representations._

Olga Kovaleva, Alexey Romanov, Anna Rogers, and
Anna Rumshisky. 2019. Revealing the dark secrets
of BERT. In Proceedings of the 2019 Conference on
_Empirical Methods in Natural Language Processing_
_and the 9th International Joint Conference on Natu-_
_ral Language Processing_ _(EMNLP-IJCNLP), pages_
4365–4374, Hong Kong, China. Association for
Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Matthew Kelcey,
Jacob Devlin, Kenton Lee, Kristina N. Toutanova,
Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: a benchmark for question answering
research. _Transactions of the Association of Compu-_
_tational Linguistics._

Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut.
2020. Albert: A lite bert for self-supervised learning
of language representations. In _International_ _Con-_
_ference on Learning Representations._

Zhuohan Li, Eric Wallace, Sheng Shen, Kevin Lin,
Kurt Keutzer, Dan Klein, and Joseph E. Gonzalez.
2020. Train large, then compress: Rethinking model
size for efficient training and inference of transformers.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized BERT pretraining approach. _CoRR, abs/1907.11692._

Christos Louizos, Max Welling, and Diederik P.
Kingma. 2018. Learning sparse neural networks
through _L0_ regularization. In _International_ _Confer-_
_ence on Learning Representations._

Sean MacAvaney, Franco Maria Nardini, Raffaele
Perego, Nicola Tonellotto, Nazli Goharian, and
Ophir Frieder. 2020. Efficient document re-ranking
for transformers by precomputing term representations. _Proceedings_ _of_ _the_ _43rd_ _International_ _ACM_
_SIGIR Conference on Research and Development in_
_Information Retrieval._

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh.
2017. The concrete distribution: A continuous relaxation of discrete random variables. In _5th_ _Inter-_
_national_ _Conference_ _on_ _Learning_ _Representations,_


-----

_ICLR 2017, Toulon, France, April 24-26, 2017, Con-_
_ference Track Proceedings. OpenReview.net._

Paul Michel, Omer Levy, and Graham Neubig. 2019.
Are sixteen heads really better than one? In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc,
E. Fox, and R. Garnett, editors, Advances in Neural
_Information_ _Processing_ _Systems_ _32,_ pages 14014–
14024. Curran Associates, Inc.

James O’Neill. 2020. An overview of neural network
compression.

Pranav Pajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. In _Proceedings_ _of_
_the 2016 Conference on Empirical Methods in Natu-_
_ral Language Processing, pages 2383–2392, Austin,_
Texas. Association for Computational Linguistics.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.
Know what you don’t know: Unanswerable questions for SQuAD. In _Proceedings_ _of_ _the_ _56th_ _An-_
_nual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics_ _(Volume_ _2:_ _Short_ _Papers),_ pages 784–
789, Melbourne, Australia. Association for Computational Linguistics.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan
Wierstra. 2014. Stochastic backpropagation and approximate inference in deep generative models. In
_Proceedings of the 31st International Conference on_
_Machine Learning, volume 32 of Proceedings of Ma-_
_chine Learning Research, pages 1278–1286, Bejing,_
China. PMLR.

Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. 2019. Distilbert, a distilled version of
bert: smaller, faster, cheaper and lighter.

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei
Yao, Amir Gholami, Michael W Mahoney, and Kurt
Keutzer. 2020. Q-bert: Hessian based ultra low precision quantization of bert. In _AAAI,_ pages 8815–
8821.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu,
Yiming Yang, and Denny Zhou. 2020. MobileBERT:
a compact task-agnostic BERT for resource-limited
devices. In _Proceedings_ _of_ _the_ _58th_ _Annual_ _Meet-_
_ing of the Association for Computational Linguistics,_
pages 2158–2170, Online. Association for Computational Linguistics.

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga
Vechtomova, and Jimmy Lin. 2019. Distilling taskspecific knowledge from bert into simple neural networks.

Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, and Amelia Archer. 2019. Small
and practical BERT models for sequence labeling.
In Proceedings of the 2019 Conference on Empirical
_Methods_ _in_ _Natural_ _Language_ _Processing_ _and_ _the_


_9th_ _International_ _Joint_ _Conference_ _on_ _Natural_ _Lan-_
_guage_ _Processing_ _(EMNLP-IJCNLP),_ pages 3632–
3636, Hong Kong, China. Association for Computational Linguistics.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. 2019. Well-read students learn better:
On the importance of pre-training compact models.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In I. Guyon, U. V. Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing Systems 30, pages 5998–6008. Curran Asso-_
ciates, Inc.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. 2019. Analyzing multi-head
self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In _Proceedings_ _of_ _the_
_57th_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Com-_
_putational_ _Linguistics,_ pages 5797–5808, Florence,
Italy. Association for Computational Linguistics.

Ziheng Wang, Jeremy Wohlwend, and Tao Lei. 2019.
Structured pruning of large language models.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R’emi Louf, Morgan Funtowicz, and Jamie Brew. 2019. Huggingface’s transformers: State-of-the-art natural language processing. _ArXiv, abs/1910.03771._

Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, and
Daxin Jiang. 2019. Model compression with multitask knowledge distillation for web-scale question
answering system.

Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe
Wasserblat. 2019. Q8bert: Quantized 8bit bert.


-----

