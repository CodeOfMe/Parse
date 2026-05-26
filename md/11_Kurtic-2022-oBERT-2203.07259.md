## The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models

### Eldar Kurtic[∗] [1], Daniel Campos[2,3], Tuan Nguyen[2], Elias Frantar[1], Mark Kurtz[2], Benjamin Fineran[2], Michael Goin[2], and Dan Alistarh[1,2]


1Institute of Science and Technology Austria


2Neural Magic Inc.

3Department of Computer Science, University of Illinois Urbana-Champaign


### Abstract

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||
||BER|TBASE||||
||Lott|ery Ticket||||
||Spar<br>|se BERT<br>||||
||Mov<br>|ement Pruni<br>|ng|||
||~~oBE~~|~~RT (ours)~~||||


60 70 80 90 97
Sparsity (%)


In this paper, we consider the problem of
sparsifying BERT models, which are a key
building block for natural language processing, in order to reduce their storage and
computational cost. We introduce the Optimal
_BERT_ _Surgeon_ (oBERT), an efficient and accurate pruning method based on approximate
second-order information, which we show to
yield state-of-the-art results for compression
in both stages of language tasks: pre-training
and fine-tuning. Specifically, oBERT extends
existing work on second-order pruning by
allowing for pruning blocks of weights, and
is the first such method that is applicable at
BERT scale. Second, we investigate _com-_
_pounding_ compression approaches to obtain
highly compressed but accurate models for
deployment on edge devices. These models
significantly push boundaries of the current
state-of-the-art sparse BERT models with
respect to all metrics: model size, inference
speed and task accuracy. For example, relative to the dense BERTBASE, we obtain 10x
model size compression with < 1% accuracy
drop, 10x CPU-inference speedup with
< 2% accuracy drop, and 29x CPU-inference
speedup with < 7.5% accuracy drop. Our
code, fully integrated with Transformers and
SparseML, is available at [https://github.](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)
[com/neuralmagic/sparseml/tree/main/](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)
[research/optimal_BERT_surgeon_oBERT.](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)

### 1 Introduction


90.0
87.5
85.0
82.5
80.0
77.5
75.0
72.5
70.0


Pre-trained Transformer models (Vaswani et al.,
2017; Devlin et al., 2019) provide robust language
representations which can be specialized on various tasks. Given their massive growth (Radford
et al., 2019; Smith et al., 2022), techniques for
reducing their computational overheads have become popular. One classic technique is Knowledge

_∗_ Corresponding author: eldar.kurtic@ist.ac.at.


Figure 1: Performance overview relative to stateof-the-art unstructured downstream pruning methods
Chen et al. (2020), Xu et al. (2021), Sanh et al.
(2020), in this order, of the BERTBASE model on the
SQuADv1.1 task.

Distillation (KD) (Hinton et al., 2015), which transfers knowledge from a larger teacher to a smaller
student model. Other work has leveraged lowerprecision representations to produce quantized
models. An orthogonal approach, which is our
primary focus, has been to apply unstructured and
block pruning, i.e. removing individual weights, to
produce compressed but accurate language models. Figure 1 provides a comparative overview of
state-of-the-art results for unstructured pruning.
In this paper, we introduce a method for improved unstructured and semi-structured (block)
pruning, by leveraging the second-order approach
pioneered by the Optimal Brain Surgeon framework (LeCun et al., 1989; Hassibi and Stork, 1992),
which we scale for the first time to LLMs. Further,
we put our results in the context of a compound
compression approach, which combines several
compression techniques to obtain sparse models
which we execute on a sparsity-aware CPU-based
runtime (NeuralMagic, 2021), showing order-ofmagnitude speedups at low accuracy loss.
In summary, our contributions are as follows:

   - We perform a thorough exploration of


-----

weight pruning approaches applied to LLMs,
including lottery-ticket, movement pruning,
magnitude and second-order pruning.

   - We introduce a general second-order pruning method called _Optimal_ _BERT_ _Surgeon_
(oBERT), which supports unstructured and
block pruning, and is the first second-order
method to be both highly-accurate and scalable to the dimensionality of BERT models.

   - We illustrate the benefits of oBERT by
significantly improving upon existing stateof-the-art pruning methods, in both stages
of language tasks: pre-training and finetuning. For illustration, when pruning
BERTBASE, oBERT outperforms Movement
Pruning (MvP), the most accurate prior approach, by more than 2% absolute F1 score at
the same sparsity, and can match the accuracy
of MvP models with 3x fewer parameters.

   - We investigate the applicability of this pruning method in a framework which compounds
popular compression approaches for LLMs,
i.e. applying pruning in combination with
layer dropping and/or quantization. In this
context, we show that our resulting sparse
models provide order-of-magnitude improvements compared to other compound compressed models, and that they can be easily
deployed for CPU inference.

### 2 Background and Related Work

**Transformer LLMs are usually built using multi-**
ple transformer layers with self-attention (Vaswani
et al., 2017). Each transformer has a variation of
two sub-components: multi head attention (MHA)
and fully connected feed forward network (FFN).
Given the massive size of well-performing models,
there has been growing interest in LLM compression. They have been shown to be fragile as minor
perturbations can lead to model collapse (Kovaleva
et al., 2021). Pruning schemes are motivated by
weight saliency metrics which represent the loss
in accuracy due to pruning. It is common to prune
in iterative steps, each of which removes weights
until a desired sparsity level is reached. Now, we
briefly overview existing approaches.
**Structured pruning for LLMs focuses on reduc-**
ing the number of layers and/or attention heads,
and requires structural understanding of the model.
Michel et al. (2019) and Voita et al. (2019) demon

strated that for some tasks nearly 40% of attention
heads can be removed without major impact on
accuracy. Other work has focused on removing
layers (Sridhar and Sarah, 2020), and on the order
in which they are removed (Sajjad et al., 2020). In
some of our experiments, we apply standard “direct” layer dropping in conjunction with pruning.
**Semi-structured** **pruning** is an intermediate approach, by which smaller groups, e.g. rectangular
sets of weights (Lagunas et al., 2021), are set to
zero. This approach has recently gained in popularity thanks to efficient computational support.
We extend the second-order pruning formulation
to such groupings, and show results for a specific
grouping supported by a CPU-inference engine.
**Unstructured** **pruning** removes individual
weights by setting them to zero. Gradual Magnitude Pruning (GMP) is a classic approach, which
makes use of weight magnitudes as a saliency metric for pruning (Han et al., 2015; Gale et al., 2019).
**First-order** **pruning** methods use a gradient
based formulation of the saliency metric. A popular method is Movement Pruning (MvP) (Sanh
et al., 2020), specifically designed for pruning
in the fine-tuning stage. Intuitively, it removes
weights that are moving towards zero. The
resulting models were the first to achieve high
sparsity with tolerable accuracy loss. Methods
such as PLATON (Zhang et al., 2022) attempt
to capture the uncertainty of weights importance
scores by upper confidence bound estimation.
Prior to our work, MvP and PLATON approaches
set state-of-the-art results for unstructured pruning.
**Second-order** **pruning** methods (LeCun et al.,
1989; Hassibi and Stork, 1992; Singh and Alistarh, 2020; Frantar et al., 2021) were developed
in the context of image classification, and leverage complex approximations of the loss curvature.
However, second-order pruning methods require
an approximation of the inverse Hessian, which
is expensive to store and compute with for LLM
parameter counts. The approach we propose is similar to WoodFisher/M-FAC methods (Singh and
Alistarh, 2020; Frantar et al., 2021), but is the
first to work accurately at LLM scale. Specifically, the WoodFisher approach is infeasible at
BERT scale, as it requires storing gradients for
inverse Fisher calculation in memory at the point
of pruning. The M-FAC approach scales, but we
show that its parametrization yields worse prun

2


-----

ing results (Appendix Figure 3). This is because
M-FAC performs full-matrix (non-blocked) inversion by default, which is inherently noisy. In addition, we extend the theoretical OBS approach
to semi-structured (block) compression. We also
show that our method can be applied during LLM
pre-training and fine-tuning, yielding state-of-theart results in both regimes.
**Knowledge** **Distillation** (Hinton et al., 2015)
trains a smaller student model against outputs of a
larger teacher model by adding a loss component
which minimizes the KL-divergence between the
two output distributions, which is the approach we
adopt in our setup too. A hardness parameter is
used to control the mixture of regular and distillation loss, and a temperature parameter to control softness of the distribution. Contrary to this,
approaches like DistilBERT (Sanh et al., 2019),
TinyBERT (Jiao et al., 2020), MobileBERT (Sun
et al., 2020a), and MiniLM (Wang et al., 2020) utilize more complex distillation schemes, based on
transferring knowledge from intermediate model’s
representations. Our sparse models provide orderof-magnitude improvements upon some of these
methods.
**Quantization represents weights and activations**
in lower precision (Courbariaux et al., 2016), and
was used to obtain models such as Q8BERT (Zafrir
et al., 2019) and TernaryBERT (Zhang et al., 2020).

Shen et al. (2020) uses information about
the Hessian spectrum to choose quantization bitwidths, whereas Yu et al. (2022) uses an approximation of the Hessian trace for structured pruning.
These Hessian-based approaches are different from
the one we propose, as we use completely different
inverse-Hessian approximations to guide pruning
decisions. The focus of our work is on weight prun_ing, and on computational speedups achievable on_
commodity CPUs. As such, the methods we investigate are orthogonal to quantization. Moreover, it
is impossible to directly compare to low-bitwidth
quantized models as most inference frameworks
do not support such custom formats. Therefore, we
will only make use of the standard QuantizationAware Training (QAT) to 8-bit weights, which is
well-supported on Intel CPUs, and showcase the
resulting speedups in conjunction with layer dropping and weight pruning.
**Downstream** **compression** methods attempt to
compress directly while fine-tuning on a specific


task. MvP method is specially designed for this
setup. **Upstream compression methods compress**
during the pre-training phase, reducing the need
for task-specific pruning. Chen et al. (2020)
examined the “Lottery Ticket” strategies (Frankle
and Carbin, 2018) which, as we illustrate later,
incur huge accuracy loss even at moderate
sparsities. Recent work “Prune Once for All”
(Prune OFA) by Zafrir et al. (2021) showed that
well-tuned magnitude pruning can be competitive
with downstream methods like MvP.
We first examine the performance of prior pruning methods, notably MvP, Prune OFA, and Lottery
Tickets, relative to the new second-order oBERT
method. The approach we propose consistently
improves upon all these prior methods, both in
pre-training (upstream) and fine-tuning (downstream) stages, and can be compounded with other
compression techniques to obtain models that are
smaller, faster and more accurate than models like
DistilBERT, TinyBERT, and block MvP.
Additional approaches for efficient inference of
LLMs exist, like token-pruning and early-exiting.
These approaches are orthogonal to ours; therefore
we discuss them in Appendix A.2.

### 3 The Optimal BERT Surgeon (oBERT)

**3.1** **Generalized Second-Order Block**
**Pruning**

The pruning problem starts from a well-optimized
dense model w[∗] _∈_ R[d], and aims to find a sparse
version of w[∗], where many of the weights are set
to zero, and the remaining weights may be updated accordingly in order to preserve the loss. It
is common for this process to occur gradually, i.e.
by progressively removing the weights. A classic
approach (LeCun et al., 1989; Hassibi and Stork,
1992) for “optimal” pruning of weights from w[∗]

at a step is to expand the loss function _L locally_
around w[∗] with respect to a sparse 0/1 weight mask
**M.** If we denote by wM = (M ⊙ **w[∗]), the model**
resulting from the Hadamard (element-wise) product between M ∈{0, 1}[d] and w[∗], we can use the
Taylor expansion at wM to obtain:

_L(wM_ ) ≃L(w[∗]) + (wM _−_ **w[∗])[⊤]∇L(w[∗])**

+ [1] _[−]_ **[w][∗][)][⊤][H][L][(][w][∗][)(][w][M]** _[−]_ **[w][∗][)][.]**

2 [(][w][M]

Given that w[∗] is well-optimized, it is reasonable
in practice to assume that _∇L(w[∗])_ _≈_ **0.** Then,


3


-----

the change in loss incurred by pruning a subset of
weights can be expressed as

_δL(δw) ≃_ [1] (1)

2 _[δ][w][⊤][H][L][(][w][∗][)][δ][w]_

where _δL(δw)_ := _L(wM_ ) −L(w[∗]) and _δw_ :=
**wM** _−_ **w[∗].** A popular way of approximating the
Hessian at w[∗] is via a dampened empirical Fisher
information matrix (Hassibi and Stork, 1992):

_m_

**HL(w)≃F[�](w)=λId** + [1] � _∇Li(w)∇L[⊤]i_ [(][w][)][,]

_m_

_i=1_

(2)
where _λ_ _≥_ 0 is a small dampening constant,
**Id** _∈_ R[d][×][d] identity matrix and _m_ is the number
of gradient outer products used to approximate the
Hessian. Given the positive-definiteness of (2), the
quadratic form (1) is always nonnegative which
is why we will refer to δL(δw) as a loss increase
incurred by pruning.
Returning to our pruning problem, assume we
wish to identify a block of weights Q of a given
shape whose removal by zero-masking would incur minimum increase in loss. This leads to the
following constrained optimization problem:

1
min
_δw_ 2 _[δ][w][⊤][F][�][(][w][∗][)][δ][w]_ (3)

s.t. **e[⊤]k** _[δ][w][ +][ w][k]_ [= 0][,] _∀k_ _∈_ Q


where **ek** _∈_ R[d] stands for the _k-th_ canonical
basis vector. Here, we will provide a generalized solution, which applies to general _Q._ First,
for convenience, we express the system of _|Q|_
equality constraints in matrix-equation form as
**EQδw** + **EQw[∗]** = **0,** where **EQ** _∈_ R[|][Q][|×][d] is
a matrix composed of the corresponding canonical basis vectors **ek (∀k** _∈_ Q) arranged in rows.
This optimization problem can be solved with the
method of Lagrange multipliers. Specifically, we
wish to find stationary points of the Lagrangian
_L(δw, λ),_ where **_λ_** _∈_ R[|][Q][|] denotes a vector of
Lagrange multipliers. Solving the system of equations _[∂L]∂δ[(][δ][w]w[,][λ][)]_ = **0 and** _[∂L][(]∂[δ]λ[w][,][λ][)]_ = **0 yields the**

following optimal weight update:

� �−1
_δw[∗]_ = −F[�] _[−][1](w[∗])E[⊤]Q_ **EQF[�]** _[−][1](w[∗])E[⊤]Q_ **EQw[∗]**

which prunes a set of weights Q and updates the
remaining weights to preserve the loss. Now, the


corresponding loss increase incurred by the optimal weight update _δw[∗]_ can be expressed as the
saliency score of weights Q, which we denote by:

�−1

_ρQ_ = 2 [1] [(][E][Q][w][∗][)][⊤] [�]EQF[�] _[−][1](w[∗])E[⊤]Q_ **EQw[∗].**

We use this saliency/importance score to rank
groups of weights for pruning. As a sanity check,
if we prune a single weight wj at a time, our derivations will yield the standard formulas of (Hassibi
and Stork, 1992). The full version of Singh and
Alistarh (2020) provided a slightly less general
derivation for the blocked case, under additional
assumptions.

**3.2** **An Efficient Implementation**

Directly implementing the previously described
approach for LLMs, where number of weights
**w ∈** R[d] is huge, is infeasible. In particular, this is
due to the dependence on the inverse of the empirical Fisher information matrix **F[−][1](w)** _∈_ R[d][×][d],

[�]
appearing in formulations of the saliency score and
of the optimal weight update. We now describe
how to circumvent these issues.

**3.2.1** **Pruning the optimal set of weights**
Assume a gradual pruning setup, in which at each
pruning step we wish to prune a model to a target sparsity _s_ _∈_ (0, 1], effectively zeroing out
_s × d_ weights, in groups of size _|Q|._ Typically
_s × d_ _≫|Q|,_ meaning that we want to remove
multiple groups at the same time. Finding the
optimal set of _[s]|[×]Q[d]|_ [groups is an intractable combi-]

natorial problem, due to all possible correlations
between them, given by the binomial coefficient
�nk�, where n = _|Qd_ _|_ [and][ k] [=] _[s]|[×]Q[d]|_ [.] [This problem]

can be alleviated by ignoring correlations between
different groups of weights Q, and solving only for
correlations between the weights within the same
group. In practice, this boils down to evaluating the
saliency score ρQ for each group Q, and pruning
the _[s]|[×]Q[d]|_ [groups with the lowest score.] [As pruning]

many weights in the same step can make the Taylor
approximation of the loss function less accurate,
one can consider pruning with multiple smaller
sub-steps with recomputations of the Hessian approximation in between (without intermediate finetuning). While this can further improve the quality
of the pruning step (Frantar et al., 2021), we do not
implement this additional optimization since the
competing methods do not utilize recomputations.


4


-----

**3.2.2** **Inverse empirical Fisher computation**
The key space and time complexity cost of the
above procedure is computing products with the
inverse empirical Fisher. A direct approach would
be to perform a block-wise diagonal approximation
of this matrix (which we detail next), and perform
direct block inversion. However, we found experimentally that this approach is too expensive
in terms of time, and quite numerically-sensitive.
As an alternative, we rely on the fact that the matrix we wish to invert is a sum of rank-1 matrices, and employ the Woodbury/Sherman-Morrison
(WSM) inversion formula. Specifically, given a
sum (A + uv[⊤]) of an invertible matrix A and an
outer product of vectors **u** and **v** with compatible dimensions, the inverse (A + uv[⊤])[−][1] can be
exactly calculated as **A[−][1]** _−_ **[A][−][1][uv][⊤][A][−][1]** [Plac-]

1+v[⊤]A[−][1]u [.]
ing the expression of the empirical Fisher in the
WSM formula, we obtain the following recursive
formulation, where m is the number of gradients
employed in the approximation:

**F�** _[−][1](w) =_ **F�** _[−]m[1][(][w][) =]_
�F� _m−1(w) +_ _m[1]_ _[∇L][m][(][w][)][∇L]m[⊤]_ [(][w][)]�−1 _._

Unrolling the recursion with **F[−]0** [1][(][w][)] [=] _λ[1]_ **[I][d][, we]**

[�]

can obtain an iterative formula to exactly calculate
the inverse of the empirical Fisher matrix as

**F�** _[−][1](w) =_ **F�** _[−]m[1][(][w][) =]_

_⊤_
_λ1_ **[I][d][ −]** [�]i[m]=1 (F[�] _[−]i−[1]1m[(][w]+∇L[)][∇L][⊤]i_ _[i][(][(][w][w][)][)][F][)(][�]_ _i[−][F]−[�][1]1i[−]−[(][1][w]1[(][)][w][∇L][)][∇L][i][(][w][i][(][)][w][)][)]_ _._


The iterative formulation enjoys a number of computational advantages over the direct implementation. The most notable ones are 1) avoiding explicit
calls to the expensive and dampening-sensitive matrix inversions, and 2) allowing successive updates
of the inverse as new gradients are computed, never
needing to store all m gradients of size d and thus
significantly reducing memory requirements.

**3.3** **Memory and run-time complexity**

Computing and storing the inverse empirical Fisher
**F[−][1](w)** _∈_ R[d][×][d] is prohibitively expensive for
�
modern LLMs, which have hundreds of millions of
parameters, due to the quadratic complexity on the
number of weights d. However, Singh and Alistarh
(2020) have shown that a diagonal block-wise approximation of the empirical Fisher matrix can be


very accurate for pruning of convolutional neural
networks. We adapt the same approach here, in
the context of LLMs. Thus, for blocks of width
_B along the main diagonal, memory requirements_
for the computation of the inverse Fisher matrix
are reduced from the quadratic O(d[2]) to a linear
_O(Bd)_ dependence on the number of weights _d._
At the same time, run-time complexity relaxes
from O(md[2]) to O(mBd). As we will show, this
computation can be efficiently and accurately performed for moderate values of m and B.
Another alternative we investigated was the
matrix-free approach of Frantar et al. (2021), which
does not require a block-wise approximation and
has complexity Θ(dm). However, our investigation showed that this approach required high values
of m to be accurate (Appendix Figure 3), which
leads to excessive memory cost in the case of
BERT models.

**3.4** **Efficient and scalable implementation**

On the practical side, we have identified general
hyper-parameters B = 50 for the block size, and
_m = 1024 for the number of gradients which pro-_
duce state-of-the-art results for all analyzed BERT
models (for more details please see Appendix A.4),
while still being able to fit on the 24GB RTX 3090
GPU. We reflect upon the computational costs
in more detail in Appendix A.3. Moreover, for
these parameter values, the block-wise approximation of **F[−][1](w) can be implemented very effi-**

[�]
ciently on modern accelerators. Specifically, we
take advantage of the fact that such hardware favors batched matrix operations, and that the blocks
of size B × B in **F[−][1](w) are independent.** With

[�]
_NB_ = _B[d]_ [we refer to the total number of blocks,]

i.e. the batch-dimension. The procedure works as
follows. First, we compute batched matrix-vector
products **F[−]i−[1]1[(][w][)][∇L][i][(][w][)]** _[∈]_ [R][N][B][×][B] [and scalar]

[�]
denominators m + ∇L[⊤]i [(][w][)][F][�] _i[−]−[1]1[(][w][)][∇L][i][(][w][)]_ _[∈]_
R[N][B] _. Then, we update the inverse Fisher for each_
block by computing the scalar-scaled outer prod
� �� �⊤
ucts **F�** _[−]i−[1]1[(][w][)][∇L][i][(][w][)]_ **F�** _[−]i−[1]1[(][w][)][∇L][i][(][w][)]_

of shape R[N][B][×][B][×][B].

### 4 Experimental Validation

To ease reproducibility, we conduct our experiments in modified versions of the popular
open-source libraries: Transformers (Wolf et al.,


5


-----

2020), and SparseML (Kurtz et al., 2020). All
of our experiments are using publicly available
datasets via Lhoest et al. (2021) and focus on the
BERTBASE model (Devlin et al., 2019), one of the
most commonly used LLMs, composed of 12 transformer layers with 110M parameters. Following
community standards, we prune encoder’s weights
(85M) and report sparsities relative to this number.
All of our models, compression recipes and the full
implementation will be made public.

**4.1** **Downstream Unstructured Pruning**

We first revisit the accuracy-compression trade-off
for pruning on downstream tasks.
**Goals** **and** **setup.** We compare existing approaches, notably Movement Pruning (MvP) (Sanh
et al., 2020) and Lottery Ticket (LT-BERT) (Chen
et al., 2020), against the gradual unstructured
oBERT method, introduced in Section 3. Our experiments evaluate performance on a variety of
downstream (English) tasks commonly used to
evaluate model compression: question answering
SQuAD v1.1 (Rajpurkar et al., 2016), sentence
classification Quora Duplicate Query Dataset QQP
(Shankar et al., 2017), and natural language inference MNLI (Williams et al., 2018).
**Comparison** **with** **MvP.** For a fair comparison
with MvP, we consider the 10-epoch gradual pruning setup used to obtain the best results by Sanh
et al. (2020). Specifically, we start from the
BERTBASE model and perform 2 epochs of finetuning, followed by 6 epochs of pruning, and 2 further epochs of fine-tuning of the compressed model.
We impose a global sparsity distribution over all
layers, prune with oBERT two times per epoch,
and use KD from the fine-tuned BERTBASE teacher.
For oBERT pruning we use m = 1024 gradients,
block size _B_ = 50, and dampening _λ_ = 10[−][7]

to approximate the inverse Hessian matrix. In
all of our runs, the first pruning step prunes 70%
of weights and then follows the cubic interpolation (Zhu and Gupta, 2018) to the target sparsity.
This large first pruning step gives more time to
recover from the later pruning steps, which impose higher sparsities. All hyper-parameters are
described in detail in Appendix A.5, and the results
are given in Table 1 (in the 10 Epochs section).
We observe that Optimal BERT Surgeon outperforms Movement Pruning by a significant margin,
more than 2 points of F1 score at the same spar

Table 1: Downstream tasks dev-set performance of
pruned BERTBASE models. ([∗] approximate results as
the exact numbers are not available.)

BERT Soft oBERT LT- oBERT
Task Spars.
BASE MvP (ours) BERT (ours)

Epochs 10 Epochs 30 Epochs

80%        -        - 86.54 **89.04**
SQuAD
88.54 90% 84.90 **87.98** 68.00[∗] **88.31**
F1
97% 82.30 **84.65**        - **85.98**

80%        -        - 82.60 **84.32**
MNLI
84.54 90% 81.20 **83.20** 75.00[∗] **83.79**
m-acc
97% 79.50 **81.00**        - **81.77**

80%        -        - 90.30 **91.57**
QQP
91.06 90% 90.20 **90.89** 90.00 **91.35**
Acc
97% 89.10 **90.23**        - **90.87**

sity. Remarkably, the model pruned with oBERT to
97% sparsity has similar accuracy to MvP-pruned
model at 90% sparsity, which has roughly 3x
more weights. This reinforces the effectiveness
of second-order information for pruning.
**Extended pruning and fine-tuning.** Next, we examine effects of extending the gradual schedule
to 30 epochs, matching the setup used for LTBERT (Chen et al., 2020). The only difference
compared to our 10 epoch setup is that we now
prune with oBERT every four epochs, and rewind
learning rate after each pruning step. The extended
setup leaves more time to recover from pruning,
which reflects in the improved results in Table 1
(30 Epochs section). We report the mean over three
runs. For additional evaluation metrics and standard deviations please see Tables 12 and 15 in the
Appendix. The results show a clear accuracy difference between oBERT and LT-BERT, especially
at high sparsities. This difference is justified since
the LT based approach attempts to mainly transfer
_network connectivity, whereas the oBERT can also_
benefit from the weight values. Finally, we examined the impact of extended setup with Soft MvP
on SQuAD, targeting 90% sparsity (not shown in
the Table), leading to an (F1, EM) combination
of (87.42, 79.83) for MvP. The F1 gap in favor of
oBERT is lower than at 10 epochs, suggesting that
extended finetuning helps all methods; yet, it is far
from negligible.

**4.2** **Upstream Unstructured Pruning**

An appealing alternative to downstream pruning
is to compress models upstream, on the semisupervised pre-training task (Zafrir et al., 2021).
Given the upstream pruned model, computational

|BERT<br>Task Spars.<br>BASE|Soft oBERT<br>MvP (ours)|LT- oBERT<br>BERT (ours)|
|---|---|---|

|80%<br>MNLI<br>84.54 90%<br>m-acc<br>97%|- -<br>81.20 83.20<br>79.50 81.00|82.60 84.32<br>75.00∗ 83.79<br>- 81.77|
|---|---|---|

|80%<br>QQP<br>91.06 90%<br>Acc<br>97%|- -<br>90.20 90.89<br>89.10 90.23|90.30 91.57<br>90.00 91.35<br>- 90.87|
|---|---|---|


6


-----

requirements for obtaining downstream fine-tuned
models are significantly reduced, as only finetuning of the remaining weights is necessary.
**Goals** **and** **setup.** To compare with existing approaches, notably Prune OFA (Zafrir et al., 2021)
and LT-BERT (Chen et al., 2020), we gradually
prune with oBERT directly at upstream datasets,
BookCorpus and English Wikipedia, and then finetune the remaining unpruned weights on the subset
of GLUE tasks.
**Teacher preparation.** Following Liu et al. (2019),
we start with the HuggingFace BERTBASE uncased
model, and fine-tune it for additional 10 epochs
only on the masked language modeling task.
**Pruning** **at** **upstream.** Once the distillation
teacher is trained, we gradually prune and fine-tune
the BERTBASE model for 3 epochs, using KD from
the dense teacher. We prune four times per epoch,
and rewind learning rate to the initial value after
each pruning step. Hyper-parameters for oBERT
are the same as for downstream pruning in 4.1; a
full description can be found in Appendix A.6.
**Sparse-transfer to downstream.** To evaluate the
resulting upstream-pruned models, we finetune the
unpruned weights on downstream tasks with KD
from the fine-tuned BERTBASE model. For a fair
comparison with Prune OFA, we fine-tune for 8
epochs. The results in Table 2 show that sparse
models produced by oBERT outperform state-ofthe-art methods by significant margins. We report
the mean over four runs. For additional evaluation metrics and standard deviations please see Appendix Tables 13 and 16. It is worth emphasizing
that in contrast to Prune OFA, which performed extensive hyper-parameter tuning for sparse-transfer,
our recipe is simple and general across downstream
tasks: 8 epochs of fine-tuning with linearly decaying learning rate. This suggests that sparse
pre-trained models found by oBERT constitute a
strong starting point for sparse transfer learning,
which can be further improved by task-specific
hyper-parameter tuning.

**4.3** **Compound Compression for CPUs**

To probe the potential practical impact of our approach, we specialize the technique for deployment
on CPUs, corresponding to “edge” deployments.
Specifically, we tailor our sparse models to the
DeepSparse (NeuralMagic, 2021) sparsity-aware
runtime, by compounding unstructured pruning


Table 2: Sparse-transfer dev-set performance of
upstream-pruned BERTBASE models. ([∗] approximate
results as the exact numbers are not available.)

BERT LT- Prune oBERT
Task Sparsity
BASE BERT OFA (ours)

SQuAD 90% 68.00[∗] 87.25 **88.49**
88.54
F1 97%  -  - 84.92

|BERT<br>Task Sparsity<br>BASE|LT- Prune oBERT<br>BERT OFA (ours)|
|---|---|

|SQuAD 90%<br>88.54<br>F1 97%|68.00∗ 87.25 88.49<br>- - 84.92|
|---|---|


MNLI 90% 75.00[∗] 81.45 **83.40**
84.54
m-acc 97%  -  - 80.91

QQP 90% 90.00 90.93 **90.99**
91.06
Acc 97%  -  - 90.33

SST-2
93.01 90% 85.00[∗] 90.88 **92.20**
Acc

QNLI
91.25 90% 80.00[∗] 89.07 **89.97**
Acc

with additional compression techniques.
**Direct** **layer** **dropping.** The competitive results
obtained at high sparsities in sections 4.1 and 4.2
suggest that BERTBASE may be overparameterized
for downstream tasks. To improve compression
ratio and inference speed, we apply “direct” layer
dropping: we initially drop all but 3 or 6 of the
BERT’s 12 layers. We drop layers from our upstream teacher, and, following (Turc et al., 2019),
fine-tune them with KD in the same setup used
to prepare the upstream teacher. These 3 and 6
layer models are used as starting points for downstream pruning. More sophisticated layer dropping
techniques (Fan et al., 2019), could bring further
accuracy gains; we leave this for future work.
**Block** **pruning** **and** **QAT.** High-performance inference usually benefits more from (semi) structured sparsity patterns than from the unstructured
ones. Hence, we employ the generalized oBERT
formulation introduced in the section 3 and prune
weights in the 4-block pattern, meaning that contiguous blocks of 4 weights are either set to zero
or kept dense. Both pruning types, unstructured
and 4-block, can be leveraged for computational
speedups with the DeepSparse runtime, but 4block pruning coupled with INT8 quantization can
provide further performance gains. For quantization, we apply standard quantization-aware training (QAT) (Jacob et al., 2018) on top of the 4-block
models (see Appendix A.7 for a full description).
**Compounding for deployment.** To determine the
impact of different compression schemes, we investigate unstructured and 4-block pruning of the 3, 6,
and 12-layer models. For all runs, we use the same
set of hyper-parameters from the extended pruning

|MNLI 90%<br>84.54<br>m-acc 97%|75.00∗ 81.45 83.40<br>- - 80.91|
|---|---|

|QQP 90%<br>91.06<br>Acc 97%|90.00 90.93 90.99<br>- - 90.33|
|---|---|

|SST-2<br>93.01 90%<br>Acc|85.00∗ 90.88 92.20|
|---|---|


7


-----

Table 3: F1 score of the 3, 6, and 12-layer models
compound-compressed on the SQuADv1.1.


and model size. As baseline for full recovery, we
follow the community-standard e.g. (Sanh et al.,
2020), and adopt the dense BERTBASE model with
88.54 F1 score. The baseline for inference speed
is dense BERTBASE inference with DeepSparse,
which matches the industry-standard ONNX Runtime inference engine. Results suggest a roughlylinear trade-off between compression and accuracy
loss, with a compression jump around 1% accuracy
drop, due to quantization being applied. Specifically, we observe 8.4x higher inference speedup at
< 1% accuracy drop, 10x speedup at < 2% drop,
15x speedup at < 3% drop, and 29x speedup at <
7.5% accuracy drop. This shows how compound
compression can optimize LLMs to various latencies. See Appendix Table 17 for full results.

**4.4** **Pruning for GPU speedups (N:M**
**sparsity)**

|12|0%<br>80%<br>90%|89.48<br>89.04<br>88.31|89.48 89.06<br>88.57 87.89<br>87.57 86.68|
|---|---|---|---|

|6|0%<br>80%<br>90%|88.32<br>88.20<br>86.78|88.32 87.94<br>87.00 86.10<br>85.34 84.59|
|---|---|---|---|

|3|0%<br>80%<br>90%|84.66<br>84.08<br>82.50|84.66 84.25<br>82.79 82.04<br>80.69 79.66|
|---|---|---|---|


30

25

20

15

10

5

0

|Col1|Col2|Infere|nce sp|eed|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||~~Mode~~|~~l com~~|~~ ressio~~|~~ n~~||||
||||||||||
||||||||||
||||||||||
||||||||||


100 99 98 97 96 95 94 93

F1 recall (%)


Figure 2: F1 recall on the SQuADv1.1 task relative to
improvements in CPU-inference speed and model size.

and fine-tuning setup in Section 4.1. The results
are given in Table 3, where we also report accuracy
of the corresponding dense models (0% sparsity)
in the same setup. For additional evaluation metrics, please see Table 14. The results indicate that
compression methods can be combined without
model collapse, although the accuracy drops do
compound. The fact that the layer-dropped models
are also highly compressible suggests that structured and fine-grained (unstructured) compression
are complementary. We find it remarkable that
our 6-layer unstructured oBERT-pruned model is
competitive with the 12-layer MvP-pruned model
when both are pruned to 90% sparsity.
**Practical** **trade-offs.** We now benchmark these
models in end-to-end fashion, both in terms of
model size and inference speed. For model size,
we report size of the checkpoint in MB after standard gzip compression. For inference speed, we
report number of items per second (throughput) on
the well-established SQuAD v1.1 CPU-inference
benchmark with a sequence length of 128 and a
batch size of 32. Figure 2 depicts relative accuracy versus magnitude of improvement in speed


Even though our previous results targeted CPUs
for deployment, we now show that our pruning
approach can also be relevant to GPUs. We apply the semi-structured variant of oBERT to impose the 2-out-of-4 sparsity pattern, which is supported on NVIDIA Ampere GPUs (Mishra et al.,
2021). More specifically, we prune in _one-shot,_
and compare against the magnitude pruning baseline in Table 4. All other methods require full
fine-tuning, and thus don’t support the one-shot
setup. oBERT significantly outperforms magnitude pruning, and with only 1-epoch of fine-tuning
it is able to fully recover dense accuracy with (F1,
EM) = (88.58, 81.16). With this sparsity pattern,
the pruned model achieves 1.85x speedup on Ampere devices.

Table 4: One-shot 2:4 pruning of the fine-tuned
BERTBASE model.

|SQuAD<br>88.54 / 81.41<br>F1 / EM|49.97 / 35.24 83.17 / 74.18|
|---|---|


### 5 Discussion

**Comparison with concurrent work.** Concurrent
work introduced PLATON (Zhang et al., 2022),
which addresses unstructured pruning of BERT
models via estimates of confidence bounds. It does
not make use of KD, so for a fair comparison we


8


-----

rerun our experiments without KD as well. Contrary to PLATON, which reports best results after
an extensive hyper-parameter search for each task
independently, we apply our sparse-transfer setup
with the upstream pruned model and only sweep
for the number of epochs _∈_ [1, 8]. We employ
early stopping to prevent overfitting on smaller
GLUE tasks. As can be seen from Table 5, oBERT
outperforms PLATON across all tasks.

Table 5: Compressed BERTBASE models to 90% sparsity on GLUE tasks without knowledge distillation.

oBERT
Task BERTBASE PLATON
(ours)

MNLI
84.6 / 83.4 82.0 / 82.2 **82.2 / 82.5**
m / mm

QQP
91.5 / 88.5 90.2 / 86.8 **90.4 / 87.1**
Acc / F1

QNLI
91.3 88.9 **89.3**
Acc

MRPC
86.4 / 90.3 84.3 / 88.8 **85.6 / 89.3**
Acc / F1

SST-2
92.7 90.5 **92.0**
Acc

CoLA
58.3 44.3 **48.47**
Mcc

STS-B
90.2 / 89.7 87.4 / 87.1 **88.0 / 87.6**
Pear / Spear

**Broader** **comparison.** We now contrast our
compound-compressed BERTBASE models relative
to alternative compression techniques. We compare against DistilBERT (Sanh et al., 2019), TinyBERT (Jiao et al., 2020), and Block Pruning For
Faster Transformers (Hybrid Filled MvP) (Lagunas
et al., 2021). DistilBERT leverages KD during pretraining and fine-tuning to obtain a 6-layer model
fine-tuned for a specific downstream task. TinyBERT makes use of a specialized Transformer-KD
scheme to distill knowledge and intermediate representations at both stages, pre-training and finetuning on a specific task. In contrast, we use a
simpler approach and employ KD from teacher’s
outputs only. Hybrid Filled MvP (Lagunas et al.,
2021) employs semi-structured pruning and weight
reintroduction. The comparison is given in Table 6, where we report the number of unpruned
encoder weights as size, compression ratio and inference speedup relative to the dense BERTBASE in
the same inference environment, and F1 score on
the dev-set of the SQuAD v1.1 dataset. The re

sults suggest that our compressed models improve
upon the current state-of-the-art techniques, setting
new very competitive baselines with respect to all
metrics: accuracy, model size, and inference speed.

Table 6: Compressed BERTBASE models on the
SQuADv1.1 task. (oBERT6,80 stands for the 6-layer
model pruned to 80% sparsity.)

Model Size Compr. Speedup F1 Dev.

BERTBASE 85.0M 1.00x 1.00x 88.54

_< 6-layers_
TinyBERT4 4.5M 18.88x 9.40x 82.10 GPU
oBERT3,90 **2.1M** **40.00x** **14.80x** **82.50** CPU

_6-layers_
DistilBERT 42.5M 2.00x 2.00x 86.90 GPU
TinyBERT6 42.5M 2.00x 2.00x 87.50 GPU
oBERT6,80 **8.5M** **10.00x** **6.38x** **88.20** CPU

_12-layers_
Hybrid F. MvP 30.7M 2.76x 1.84x 88.70 GPU
oBERT12,80 **17.0M** **5.00x** **3.38x** **89.04** CPU

**BERTLARGE results.** Most of our results presented in Section 4 targeted the widely-adopted
BERTBASE model. This gave us an opportunity for
a fair comparison against many different methods.
To verify that our approach does not pertain only to
the BERTBASE model, in Table 7 we present downstream pruning results on the three times larger
BERTLARGE model and the SQuADv1.1 task. As
can be seen from the Table, even the model pruned
with oBERT at double the sparsity (95%) outperforms Prune OFA (90%).

Table 7: Compressed BERTLARGE models on the
SQuADv1.1 task.

BERTLARGE Sparsity Prune oBERT
F1 / EM OFA (ours)

91.22 / 84.45 90% 90.20 / 83.35 **91.07 / 84.61**

91.22 / 84.45 95% NA 90.29 / 83.58

**MLPerf** **Inference** **Benchmark.** Motivated by
our state-of-the-art results across-the-board, we
apply our full compound compression pipeline
to compress BERTLARGE and MobileBERT (Sun
et al., 2020b) models in the context of the industrial MLPerf Inference Benchmark[1]. In brief,
we were able to achieve order-of-magnitude improvements in terms of model size and inference
speedups, while maintaining >99% of the dense

1https://mlcommons.org/en/

|Task BERT BASE|oBERT<br>PLATON<br>(ours)|
|---|---|

|MNLI<br>84.6 / 83.4<br>m / mm|82.0 / 82.2 82.2 / 82.5|
|---|---|

|QQP<br>91.5 / 88.5<br>Acc / F1|90.2 / 86.8 90.4 / 87.1|
|---|---|

|QNLI<br>91.3<br>Acc|88.9 89.3|
|---|---|

|MRPC<br>86.4 / 90.3<br>Acc / F1|84.3 / 88.8 85.6 / 89.3|
|---|---|

|BERT<br>LARGE Sparsity<br>F1 / EM|Prune oBERT<br>OFA (ours)|
|---|---|

|91.22 / 84.45 95%|NA 90.29 / 83.58|
|---|---|


9


-----

BERTLARGE accuracy. For details please see Appendix A.1, as well as our open-source submission.

### 6 Broader Impact

Our work is part of the general trend of producing inference efficient models which approximate
performance of their larger bases. By and large,
this work should help increase model efficiency,
thereby reducing computational and ultimately
monetary cost of executing such models. Moreover, it could allow models to be used by those
who do not have access to expensive specialized
computing clusters: for instance, our main speedup
results are aimed at widely-available CPUs.

### 7 Limitations

As any academic study, our work is not without
its limitations. We split their discussion into limitations that are inherent to our method, and limitations _of_ _our_ _present_ _study;_ the latter can be
overcome by extensions of our work. In the first
category, we begin by highlighting the fact that
our second-order method relies on approximations,
which are inherent in order to scale such methods
to BERT scale. Prior studies, e.g. (Singh and Alistarh, 2020) have performed careful examinations
of the validity of these approximations in the context of CNN models. The strength of our empirical
results can be seen as indirect evidence that these
approximations apply to BERT models as well. A
second, technical, limitation is the fact that our
method requires non-trivial additional storage cost;
while we have shown that our experiments can be
executed on a single commodity GPU (NVIDIA
RTX 3090), this limits the range of devices on
which the technique may be applied. However, we
provide an efficient and easy way to scale our approach with more GPUs, which is automatically
utilized in a multi-GPU environment.
Another limitation which we aim to remove in
future work is the focus on relatively fine-grained
sparsity types, such as unstructured and semistructured pruning.

### References

Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia
Liu, Yang Zhang, Zhangyang Wang, and Michael
Carbin. 2020. The lottery ticket hypothesis for pre

trained bert networks. _Advances in neural informa-_
_tion processing systems, 33:15834–15846._

Matthieu Courbariaux, Itay Hubara, Daniel Soudry,
Ran El-Yaniv, and Yoshua Bengio. 2016. Binarized neural networks: Training deep neural networks with weights and activations constrained to+
1 or-1. _arXiv preprint arXiv:1602.02830._

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training of
deep bidirectional transformers for language understanding. In _Proceedings_ _of_ _the_ _2019_ _Conference_
_of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_
_for_ _Computational_ _Linguistics:_ _Human_ _Language_
_Technologies,_ _Volume_ _1_ _(Long_ _and_ _Short_ _Papers),_
pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Angela Fan, Edouard Grave, and Armand Joulin. 2019.
Reducing transformer depth on demand with structured dropout. In _International_ _Conference_ _on_
_Learning Representations._

Wikimedia Foundation. [Wikimedia downloads.](https://dumps.wikimedia.org)

Jonathan Frankle and Michael Carbin. 2018. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learn_ing Representations._

Elias Frantar, Eldar Kurtic, and Dan Alistarh. 2021. Mfac: Efficient matrix-free approximations of secondorder information. _Advances in Neural Information_
_Processing Systems, 34._

Trevor Gale, Erich Elsen, and Sara Hooker. 2019. The
state of sparsity in deep neural networks. _arXiv_
_preprint arXiv:1902.09574._

Song Han, Huizi Mao, and William J Dally. 2015. A
deep neural network compression pipeline: Pruning, quantization, huffman encoding. _arXiv preprint_
_arXiv:1510.00149, 10._

Babak Hassibi and David Stork. 1992. Second order
derivatives for network pruning: Optimal brain surgeon. _Advances_ _in_ _neural_ _information_ _processing_
_systems, 5._

Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. 2015.
Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531, 2(7)._

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig
Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for efficient
integer-arithmetic-only inference. In _Proceedings_
_of the IEEE conference on computer vision and pat-_
_tern recognition, pages 2704–2713._


10


-----

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang,
Xiao Chen, Linlin Li, Fang Wang, and Qun Liu.
2020. Tinybert: Distilling bert for natural language
understanding. In _Findings_ _of_ _the_ _Association_ _for_
_Computational_ _Linguistics:_ _EMNLP_ _2020,_ pages
4163–4174.

Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Woosuk Kwon, Joseph Hassoun, and Kurt
Keutzer. 2022. Learned token pruning for transformers. In Proceedings of the 28th ACM SIGKDD
_Conference on Knowledge Discovery and Data Min-_
_ing, KDD ’22, page 784–794. Association for Com-_
puting Machinery.

Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers,
and Anna Rumshisky. 2021. BERT busters: Outlier
layernorm dimensions that disrupt BERT. _CoRR,_
abs/2105.06990.

Mark Kurtz, Justin Kopinsky, Rati Gelashvili, Alexander Matveev, John Carr, Michael Goin, William
Leiserson, Sage Moore, Bill Nell, Nir Shavit, and
Dan Alistarh. 2020. Inducing and exploiting activation sparsity for fast inference on deep neural networks. In _Proceedings_ _of_ _the_ _37th_ _International_
_Conference_ _on_ _Machine_ _Learning,_ volume 119 of
_Proceedings_ _of_ _Machine_ _Learning_ _Research,_ pages
5533–5543, Virtual. PMLR.

François Lagunas, Ella Charlaix, Victor Sanh, and
Alexander Rush. 2021. Block pruning for faster
transformers. In _Proceedings_ _of_ _the_ _2021_ _Confer-_
_ence_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_
_Processing,_ pages 10619–10629, Online and Punta
Cana, Dominican Republic. Association for Computational Linguistics.

Yann LeCun, John Denker, and Sara Solla. 1989. Optimal brain damage. _Advances in neural information_
_processing systems, 2._

Quentin Lhoest, Albert Villanova del Moral, Yacine
Jernite, Abhishek Thakur, Patrick von Platen, Suraj
Patil, Julien Chaumond, Mariama Drame, Julien
Plu, Lewis Tunstall, Joe Davison, Mario Šaško,
Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu,
Nicolas Patry, Angelina McMillan-Major, Philipp
Schmid, Sylvain Gugger, Clément Delangue, Théo
Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar,
François Lagunas, Alexander Rush, and Thomas
Wolf. 2021. Datasets: A community library for natural language processing. In _Proceedings_ _of_ _the_
_2021_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natu-_
_ral_ _Language_ _Processing:_ _System_ _Demonstrations,_
pages 175–184. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,


Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692._

Paul Michel, Omer Levy, and Graham Neubig. 2019.
Are sixteen heads really better than one? _Advances_
_in neural information processing systems, 32._

Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko
Stosic, Dusan Stosic, Ganesh Venkatesh, Chong
Yu, and Paulius Micikevicius. 2021. Accelerating sparse deep neural networks. _arXiv_ _preprint_
_arXiv:2104.08378._

NeuralMagic. 2021. [Deep sparse: A fast cpu inference](http://arxiv.org/abs/https://github.com/neuralmagic/deepsparse)
[engine.](http://arxiv.org/abs/https://github.com/neuralmagic/deepsparse)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners.
_OpenAI blog, 1(8):9._

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev,
and Percy Liang. 2016. Squad: 100,000+ questions
for machine comprehension of text. In EMNLP.

Hassan Sajjad, Fahim Dalvi, Nadir Durrani, and
Preslav Nakov. 2020. Poor man’s BERT:
smaller and faster transformer models. _CoRR,_
abs/2004.03844.

Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. 2019. Distilbert, a distilled version
of bert: smaller, faster, cheaper and lighter. _arXiv_
_preprint arXiv:1910.01108._

Victor Sanh, Thomas Wolf, and Alexander Rush.
2020. Movement pruning: Adaptive sparsity by
fine-tuning. _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing Systems, 33:20378–20389._

Iyer Shankar, Dandekar Nikhil, and Csernai Kornel.
2017. First quora dataset release: Question pairs.

S. Shankar. 2017. Identifying quora question pairs having the same intent.

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei
Yao, Amir Gholami, Michael W Mahoney, and Kurt
Keutzer. 2020. Q-bert: Hessian based ultra low
precision quantization of bert. In _Proceedings_ _of_
_the AAAI Conference on Artificial Intelligence,_ volume 34, pages 8815–8821.

Sidak Pal Singh and Dan Alistarh. 2020. Woodfisher:
Efficient second-order approximation for neural network compression. _Advances in Neural Information_
_Processing Systems, 33._

Shaden Smith, Mostofa Patwary, Brandon Norick,
Patrick LeGresley, Samyam Rajbhandari, Jared
Casper, Zhun Liu, Shrimai Prabhumoye, George
Zerveas, Vijay Korthikanti, et al. 2022. Using


11


-----

deepspeed and megatron to train megatron-turing
nlg 530b, a large-scale generative language model.
_arXiv preprint arXiv:2201.11990._

Sharath Nittur Sridhar and Anthony Sarah. 2020. Undivided attention: Are intermediate layers necessary
for bert? _arXiv preprint arXiv:2012.11881._

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu,
Yiming Yang, and Denny Zhou. 2020a. Mobilebert:
a compact task-agnostic bert for resource-limited devices. In ACL.

Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu,
Yiming Yang, and Denny Zhou. 2020b. Mobilebert:
a compact task-agnostic bert for resource-limited devices. In _Proceedings_ _of_ _the_ _58th_ _Annual_ _Meeting_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics,_
pages 2158–2170.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. 2019. Well-read students learn better:
The impact of student initialization on knowledge
distillation. _ArXiv, abs/1908.08962._

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. _Advances in neural information process-_
_ing systems, 30._

Elena Voita, David Talbot, F. Moiseev, Rico Sennrich,
and Ivan Titov. 2019. Analyzing multi-head selfattention: Specialized heads do the heavy lifting, the
rest can be pruned. In ACL.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. Minilm: Deep selfattention distillation for task-agnostic compression
of pre-trained transformers. _Advances in Neural In-_
_formation Processing Systems, 33:5776–5788._

Liu Weijie, Zhou Peng, Zhao Zhe, Wang Zhiruo, Deng
Haotang, and Ju Qi. 2020. Fastbert: a self-distilling
bert with adaptive inference time. In Proceedings of
_ACL 2020._

Adina Williams, Nikita Nangia, and Samuel Bowman.
2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceed_ings of the 2018 Conference of the North American_
_Chapter_ _of_ _the_ _Association_ _for_ _Computational_ _Lin-_
_guistics:_ _Human_ _Language_ _Technologies,_ _Volume_
_1 (Long Papers), pages 1112–1122. Association for_
Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,
Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu,
Teven Le Scao, Sylvain Gugger, Mariama Drame,
Quentin Lhoest, and Alexander M. Rush. 2020.


Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on
_Empirical_ _Methods_ _in_ _Natural_ _Language_ _Process-_
_ing:_ _System_ _Demonstrations,_ pages 38–45, Online.
Association for Computational Linguistics.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and
Jimmy J. Lin. 2020. Deebert: Dynamic early exiting for accelerating bert inference. In ACL.

Dongkuan Xu, Ian En-Hsu Yen, Jinxi Zhao, and Zhibin
Xiao. 2021. Rethinking network pruning – under
the pre-train and fine-tune paradigm. In NAACL.

Shixing Yu, Zhewei Yao, Amir Gholami, Zhen Dong,
Sehoon Kim, Michael W Mahoney, and Kurt
Keutzer. 2022. Hessian-aware pruning and optimal
neural implant. In _Proceedings_ _of_ _the_ _IEEE/CVF_
_Winter Conference on Applications of Computer Vi-_
_sion, pages 3880–3891._

Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe
Wasserblat. 2019. Q8bert: Quantized 8bit bert.
_2019_ _Fifth_ _Workshop_ _on_ _Energy_ _Efficient_ _Machine_
_Learning_ _and_ _Cognitive_ _Computing_ _-_ _NeurIPS_ _Edi-_
_tion (EMC2-NIPS), pages 36–39._

Ofir Zafrir, Ariel Larey, Guy Boudoukh, Haihao Shen,
and Moshe Wasserblat. 2021. Prune once for all:
Sparse pre-trained language models. _arXiv preprint_
_arXiv:2111.05754._

Qingru Zhang, Simiao Zuo, Chen Liang, Alexander
Bukharin, Pengcheng He, Weizhu Chen, and Tuo
Zhao. 2022. Platon: Pruning large transformer
models with upper confidence bound of weight importance. In _International_ _Conference_ _on_ _Machine_
_Learning, pages 26809–26823. PMLR._

Wei Zhang, Lu Hou, Yichun Yin, Lifeng Shang, Xiao
Chen, Xin Jiang, and Qun Liu. 2020. Ternarybert:
Distillation-aware ultra-low bit bert. _arXiv preprint_
_arXiv:2009.12812._

M. Zhu and Suyog Gupta. 2018. To prune, or not to
prune: exploring the efficacy of pruning for model
compression. _ArXiv, abs/1710.01878._

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan
Salakhutdinov, Raquel Urtasun, Antonio Torralba,
and Sanja Fidler. 2015. Aligning books and movies:
Towards story-like visual explanations by watching movies and reading books. _2015_ _IEEE_ _Inter-_
_national_ _Conference_ _on_ _Computer_ _Vision_ _(ICCV),_
pages 19–27.

### A Appendix

**A.1** **MLPerf Inference benchmark**

Following the MLPerf benchmark guidelines on
producing compressed and fast models while
maintaining >99% of the BERTLARGE F1 score on


12


-----

the SQuADv1.1 task, we explore two directions. In
the first one, dubbed oBERT-Large, we compound
compress the BERTLARGE model without any
changes to its architecture. Therefore, we apply
4-block downstream pruning to 95% sparsity
followed by the quantization aware training (QAT).
In the second direction we focus on recovering
the BERTLARGE accuracy by compressing an
already compact MobileBERT model, dubbed
oBERT-MobileBERT. More specifically, we
apply direct layer dropping, leaving only 14
transformer layers out of the original 24, followed
by the 4-block pruning to 50% sparsity and
quantization aware training. We present results in
Table 8, where models were evaluated with the
DeepSparse inference engine, using a server with
two Intel(R) Xeon(R) Platinum 8380 (IceLake)
CPUs with 40 cores each, batch-size 128 and
sequence length 384. For more details please see
[our official submission at https://github.com/](https://github.com/neuralmagic/mlperf_inference_results_v2.1/tree/master/open/NeuralMagic)
[neuralmagic/mlperf_inference_results_v2.](https://github.com/neuralmagic/mlperf_inference_results_v2.1/tree/master/open/NeuralMagic)
[1/tree/master/open/NeuralMagic.](https://github.com/neuralmagic/mlperf_inference_results_v2.1/tree/master/open/NeuralMagic)

**A.2** **Additional comparisons**

Here we reflect upon some other methods focused
on efficient inference for LLMs, which are orthogonal to weight pruning. For example, Learned Token Pruning (Kim et al., 2022) tries to adaptively
remove unimportant tokens in input sequences and
provides 2x higher throughput at < 1% accuracy
drop; at the same accuracy drop, our compressed
model is able to achieve 8.4x higher throughput.
DeeBERT (Xin et al., 2020) and FastBERT (Weijie et al., 2020) apply an early-exit technique for
inference speedup. The latter achieves 2-3x faster
inference without performance degradation. However, the method only applies to batch size one.
Nevertheless, in terms of direct comparison, our
compressed models are able to achieve 4x faster
inference on CPUs without accuracy degradation.
Overall, we emphasize the fact that these methods
are complementary to our compression techniques,
so it would be interesting to investigate computational gains by combining such methods.

**A.3** **Computational costs**

In practice, for the 12-layer BERTBASE model with
_d = 85M_ encoder weights and block size B = 50,
the O(Bd) memory requirement translates to approximately 17GB, which can be easily kept on


the 24GB RTX 3090 card. While this amount of
memory is available on high-performance GPUs,
it is also straightforward to split the NB _× B × B_
tensor along the batch-dimension NB and utilize
additional GPUs or even memory swapping with
CPU. Our implementation updates the inverse Hessian approximation in negligible time, and can run
asynchronously while the next gradient is being
fetched. Computing saliency scores and optimal
weight updates takes only a few seconds.

**A.4** **Optimal BERT Surgeon (oBERT)**
**hyper-parameters**

**Hyper-parameters.** The oBERT pruning method
has three tunable hyper-parameters: number of
gradients (m), block size (B), and dampening (λ).
These are supposed to be tuned with respect to
the model and available computational resources.
In all of our runs, across all models and datasets,
we use the same set of hyper-parameters which we
found to work best for the BERTBASE model on the
SQuAD v1.1 dataset. We conjecture that further
tuning for smaller models (3 and 6-layer models)
could improve their results, but for simplicity and
fairness to other methods, we apply the same ones
found for the BERTBASE .
**Ablation studies.** The procedure to find the optimal set of hyper-parameters for a model consists
of a grid search over the possible hyper-parameter
combinations and one-shot pruning runs to various
high sparsity targets to evaluate the quality of the
pruning approximation for each combination. We
found that m = 1024, B = 50, and λ = 10[−][7] produce state-of-the-art results for a negligible computational overhead with the BERTBASE model. Frantar et al. (2021) shows that larger block sizes
require more gradients for better approximation.
Given the massive size of the BERTBASE model,
we picked this setup as it was the best performing
one that could still fit on a single 24GB RTX 3090
GPU card. In Figures 3, 4, and 5 we visualize a
fraction of the one-shot pruning ablations with respect to all three hyper-parameters that motivated
us to pick these specific values.

**A.5** **Downstream pruning**

**Teacher** **preparation.** For all downstream pruning runs we make use of the KD from the finetuned BERTBASE teacher outputs. The teacher
is fine-tuned on the corresponding downstream


13


-----

Table 8: MLPerf inference results for oBERT compressed BERTLARGE and MobileBERT models.

F1 Score Compression Throughput
Model Precision File Size Speedup
(R=X% recovery) Ratio (samples/sec)


BERT-Large
FP32 90.87 (R=100%) 1.30 GB 1x 15.49 1x
dense baseline

oBERT-Large INT8 90.21 (R=99.27%) 38.20 MB 34x 230.74 15x
oBERT-MobileBERT INT8 90.32 (R=99.39%) 9.56 MB 136x 928.58 60x

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||~~BERT~~|||
||~~BASE~~<br>= 10<br>6<br> <br>8|||
||~~= 10~~<br><br>= 10<br>7|||
|||||


50 60 70
Sparsity (%)


90
80
70
60
50
40
30
20
10


88

86

84

82

80

78

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
||BERTBASE|||
||B = 50<br>|||
||~~B = 5k~~<br>|||
||~~B = 500k~~<br>M-FAC|||


50 60 70
Sparsity (%)


Figure 3: One-shot pruning ablation study with respect
to the block size (B), with m = 1024 and λ = 10[−][7],
on the BERTBASE model and the question-answering
SQuAD v1.1 dataset. M-FAC stands for the full inverse Hessian approximation (Frantar et al., 2021).


Figure 5: One-shot pruning ablation study with respect
to the dampening (λ), with _m_ = 1024 and _B_ = 50,
on the BERTBASE model and the question-answering
SQuAD v1.1 dataset.

task following the default hyper-parameters for
SQuAD[2] and GLUE (QQP and MNLI)[3].
**Pruning setup.** In Table 9 we describe in detail all
hyper-parameters for downstream pruning results
presented in Tables 1 and 3. For easier comprehension, we also visualize learning rate schedules in
Figures 6 and 8, and sparsity schedules in Figures
7 and 9.
**3-, 6-layer models.** We prepare our 3 and 6 layer
models for downstream runs in two stages: layer
dropping and retraining phase. We drop layers
from our upstream teacher model (more details on
it in Appendix A.6). After dropping, we retrain the
remaining layers, following insights from (Turc
et al., 2019), in the same setup used to prepare the
upstream teacher with addition of the KD from it.


88

87

86

85

84

83

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
||BERTBASE<br>|||
||~~m = 128~~<br>m = 512|||
||m = 1024|||


50 60 70
Sparsity (%)


Figure 4: One-shot pruning ablation study with respect
to the number of gradients (m), with _B_ = 50 and
_λ_ = 10[−][7], on the BERTBASE model and the questionanswering SQuAD v1.1 dataset.


**A.6** **Upstream pruning**

**Teacher** **preparation.** We prepare a teacher for
upstream pruning by following some insights from
(Liu et al., 2019). More concretely we start with the


2https://github.com/huggingface/transformers/tree/main/
examples/pytorch/question-answering
3https://github.com/huggingface/transformers/tree/main/
examples/pytorch/text-classification


14


-----

|Batch size|16 for SQuAD,<br>32 for GLUE|
|---|---|

|Learning rate (initial, final)|(8e-5, 3e-5) for SQuAD, (8e-5, 8e-6) for SQuAD,<br>(8e-5, 2e-5) for GLUE (5e-5, 5e-6) for GLUE|
|---|---|

|Learning rate rewinds|periodic every 4 epochs,<br>one at epoch=8<br>start at epoch=2|
|---|---|

|Student model|12-layer: bert-base-uncased<br>6-layer: layer drop + pre-train with KD<br>3-layer: layer drop + pre-train with KD|
|---|---|

|Prune start|epoch=2|
|---|---|

|Initial sparsity step|12-layer: 70%<br>6-layer: 30%<br>3-layer: 30%|
|---|---|


Number of gradients m = 1024

oBERT parameters Block size B = 50

Dampening λ = 10[−][7]

Table 9: Downstream pruning hyper-parameters used to obtain results presented in Tables 1 and 3.

|oBERT parameters|Number of gradients m = 1024<br>Block size B = 50<br>Dampening λ = 10−7|
|---|---|


8

7

6

5

4

3

2


1e 5

SQuAD
GLUE

0 2 4 6 8 10
Epoch

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||~~S~~|~~uAD~~|||||
||G|LUE|||||


Figure 6: Visualized learning rate schedule for 10epoch downstream runs.

_bert-base-uncased[4]_ model, adopt pre-training on
two datasets (BookCorpus[5] & English Wikipedia[6])
with focus on the masked language modeling task
(MLM) for 10-epochs with batch size 256 and
learning rate linearly decaying to zero from the
initial value of 1e-4.


Figure 7: Visualized sparsity schedule for 10-epoch
downstream runs with initial sparsity of 70% and target
sparsity of 90%, following the cubic interpolation (Zhu
and Gupta, 2018).

**Pruning** **setup.** In Table 10 we describe in detail our upstream pruning recipe. As can be
noticed, our upstream pruning recipe is just a
downscaled version of our 30-epoch downstreampruning recipe to 3-epochs.


4https://huggingface.co/bert-base-uncased
5https://huggingface.co/datasets/bookcorpus
6https://huggingface.co/datasets/wikipedia


15


-----

8
7
6
5
4
3
2
1


1e 5

SQuAD
GLUE

0 5 10 15 20 25 30
Epoch

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|S|QuAD|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||
|||||||||||G|LUE||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||
||||||||||||||

|Initial learning rate<br>Learning rate schedule<br>Learning rate rewinds|5e-4<br>linear decay with rewinds<br>periodic every 0.5 epochs|
|---|---|

|Max sequence length<br>Weight decay|512<br>0.01|
|---|---|

|Knowledge Distillation<br>(hardness, temperature)|(1.0, 5.5)|
|---|---|


Table 10: Upstream pruning hyper-parameters.

8 Epochs

Initial learning rate 1.5e-4
Learning rate schedule linear decay to 1.5e-6

16 for SQuAD,
Batch size
32 for GLUE

Knowledge Distillation
(1.0, 5.5)
(hardness, temperature)

Teacher model BERTBASE

Table 11: Sparse-transfer learning hyper-parameters
used to fine-tune upstream-pruned models at downstream tasks. These hyper-parameters are used to obtain results presented in Table 2.


Figure 8: Visualized learning rate schedule for 30epoch downstream runs.

90
80
70

0

0 5 10 15 20 25 30
Epoch

Figure 9: Visualized sparsity schedule for 30-epoch
downstream runs with initial sparsity of 70% and target
sparsity of 90%, following the cubic interpolation (Zhu
and Gupta, 2018).


**A.7** **Downstream quantization**

We perform QAT on top of dense and 4-block
pruned models on SQuAD v1.1 as shown in Table
3. We quantize to 8 bits the embedding matrices,
linear modules of all encoder units which includes
matrices in their attention and feed forward layers,
and the linear module of the output layer. Weights
that were pruned are kept constant (zero) during
quantization (sparsity mask preserved). Non-linear
operations within the Softmax, LayerNorm and
GeLU are not quantized. For each dense and 4block pruned model in Table 3, we perform a total
of ten epochs training where the quantization observers are active for the first five and the remaining
is fine-tuning. We do hyper-parameter search over
the learning rates of 1e-4, 8e-5, 5e-5, 3e-5 and the
distillation hardness of 0.9 and 1.0. We then pick
the model with the best F1 score.

|Initial learning rate<br>Learning rate schedule|1.5e-4<br>linear decay to 1.5e-6|
|---|---|

|Batch size|16 for SQuAD,<br>32 for GLUE|
|---|---|

|Knowledge Distillation<br>(hardness, temperature)|(1.0, 5.5)|
|---|---|


16


-----

**A.8** **Additional performance metrics**

Due to the space constraints, in the paper we report
F1 score for SQuAD v1.1, matched accuracy for
MNLI, and accuracy for QQP dataset. As all of our
hyper-parameters for MNLI and QQP are exactly
the same, we refer to these two datasets as GLUE.
In Table 12 we report the additional metrics too:
exact match (EM) for SQuAD v1.1, mismatched
accuracy for MNLI, and F1 score for QQP dataset.
Tables 15 and 16 present standard deviations of the
corresponding results in Tables 1, 2 and 12. Finally,
Table 14 presents the exact-match metric for the
corresponding results in Table 3.

BERT Soft oBERT oBERT
Task Sparsity
BASE MvP (ours) (ours)

Epochs 10 Epochs 30 Epochs

80%        -        - **82.08**
SQuAD
81.22 90% 76.60 **80.76** **81.12**
EM
97% 72.70 **76.14** **78.11**

80%        -        - **84.91**
MNLI
85.06 90% 81.80 **83.58** **84.35**
mm-acc
97% 80.10 **80.67** **82.01**

80%        -        - **88.63**
QQP
88.00 90% 86.80 **87.69** **88.30**
F1
97% 85.50 **87.05** **87.66**

Table 12: Additional evaluation metrics for results presented in Table 1.

BERT Prune oBERT
Task Sparsity
BASE OFA (ours)

SQuAD 90% 79.83 **81.43**
81.42
EM 97%   - 76.90

MNLI 90% 82.43 **83.78**
85.06
mm-acc 97%   - 81.13

QQP 90% 87.72 **87.81**
88.00
F1 97%    - 86.97

Table 13: Additional evaluation metrics for results presented in Table 2.

**A.9** **Inference speedups and compression**
**ratios of compressed models**

Details on the results shown in Figure 2 are drawn
from Table 17. As shown in the results, not all
compound compressed models yield improvements
in inference or compression relative to retained
model performance but those that do allow for
massive improvements.

|12|0%<br>80%<br>90%|82.71<br>82.08<br>81.12|82.71 81.99<br>81.46 80.57<br>80.14 78.84|
|---|---|---|---|

|6|0%<br>80%<br>90%|81.17<br>81.15<br>79.16|81.17 80.85<br>79.55 78.27<br>77.65 76.56|
|---|---|---|---|

|3|0%<br>80%<br>90%|76.62<br>75.62<br>73.61|76.62 76.06<br>74.07 72.70<br>71.36 70.00|
|---|---|---|---|

|BERT<br>Task Sparsity<br>BASE|Soft oBERT<br>MvP (ours)|oBERT<br>(ours)|
|---|---|---|

|80%<br>SQuAD<br>90%<br>F1, EM<br>97%|0.11, 0.03<br>0.13, 0.13<br>0.11, 0.17|
|---|---|

|80%<br>SQuAD<br>81.22 90%<br>EM<br>97%|- -<br>76.60 80.76<br>72.70 76.14|82.08<br>81.12<br>78.11|
|---|---|---|

|80%<br>MNLI<br>90%<br>m, mm<br>97%|0.14, 0.13<br>0.05, 0.04<br>0.35, 0.22|
|---|---|

|80%<br>MNLI<br>85.06 90%<br>mm-acc<br>97%|- -<br>81.80 83.58<br>80.10 80.67|84.91<br>84.35<br>82.01|
|---|---|---|

|80%<br>QQP<br>90%<br>acc, F1<br>97%|0.08, 0.08<br>0.04, 0.06<br>0.05, 0.08|
|---|---|

|80%<br>QQP<br>88.00 90%<br>F1<br>97%|- -<br>86.80 87.69<br>85.50 87.05|88.63<br>88.30<br>87.66|
|---|---|---|

|SQuAD 90%<br>F1, EM 97%|0.13, 0.13<br>0.03, 0.14|
|---|---|

|BERT<br>Task Sparsity<br>BASE|Prune<br>OFA|oBERT<br>(ours)|
|---|---|---|

|MNLI 90%<br>m, mm 97%|0.08, 0.24<br>0.17, 0.35|
|---|---|

|SQuAD 90%<br>81.42<br>EM 97%|79.83<br>-|81.43<br>76.90|
|---|---|---|


Table 16: Standard deviations for results presented in
Table 2 and 13.

**A.10** **Responsible NLP Research -**
**Reproducibility Checklist**

In addition to many items from the “Reproducibility Checklist” which are already carefully addressed throughout the paper and Appendix sections, here we provide the remaining details to
facilitate reproducibility of our results.

**A.10.1** **Scientific Artifacts**

**Datasets.** Our experiments use existing and well
established benchmarks for pre-training and fine

Table 14: Additional evaluation metric (exact-match)
for results presented in Table 3.

oBERT
Task Sparsity
(ours)

Epochs 30 Epochs

80% 0.11, 0.03

SQuAD

90% 0.13, 0.13

F1, EM

97% 0.11, 0.17

80% 0.14, 0.13

MNLI

90% 0.05, 0.04

m, mm

97% 0.35, 0.22

80% 0.08, 0.08

QQP

90% 0.04, 0.06

acc, F1

97% 0.05, 0.08

Table 15: Standard deviations for results presented in
Tables 1 and 12.

oBERT
Task Sparsity
(ours)


QQP 90%
acc, F1 97%


0.06, 0.07
0.09, 0.18

|QQP 90%<br>88.00<br>F1 97%|87.72<br>-|87.81<br>86.97|
|---|---|---|


17


-----

Layers Sparsity Compression F1 score F1 recall Throughput Speedup Model size Compression
(%) Method (%) (items per sec.) DeepSparse (gzip MB) Ratio (w.r.t. gzip)

12 0 none 88.54 100.00 65.81 1.00 384.7 1.00

12 80 unstructured 89.04 100.56 222.66 3.38 173.1 2.22
12 90 unstructured 88.31 99.74 292.40 4.44 140.1 2.75
12 80 4-block+QAT 87.89 99.26 552.22 8.39 37.8 10.18

6 80 unstructured 88.20 99.62 419.68 6.38 128.3 3.00
6 90 unstructured 86.78 98.01 663.02 10.07 111.8 3.44
6 80 4-block+QAT 86.10 97.24 989.54 15.04 26.2 14.70

3 80 unstructured 84.08 94.96 737.62 11.21 105.9 3.63
3 90 unstructured 82.50 93.18 974.00 14.80 97.7 3.94
3 80 4-block+QAT 82.04 92.66 1892.27 28.75 20.3 18.92

Table 17: Compression effects on model size and inference speed, evaluated at batch size 32 with sequence length
128 on SQuAD v1.1 dataset. Evaluated at the c5.12xlarge AWS instance.


tuning of LLMs. Each dataset was used without
any additional forms of modifications. Given that
we did not modify any of the datasets, we did not
inspect for personal, sensitive, or offensive content, nor did we perform any kind of anonymization. For pre-training, we make use of the Toronto
Book Corpus (TBC) (Zhu et al., 2015) [7] and the
wikipedia.20200501.en (Foundation) [8]. For finetuning we make use of SQuAD v1.1 (Rajpurkar
et al., 2016) [9], Quora Duplicate Question Dataset
(QQP) (Shankar, 2017) [10], and Multi-Genre Natural Language Inference (MNLI) (Williams et al.,
2018) [11] datasets. All these datasets are publicly available via HuggingFace datasets repository (Lhoest et al., 2021). The terms of usage and
further details on each dataset can be found in their
respective repositories.
**Models.** The model used as a starting point for all
of our experiments is BERTBASE, publicly available via HuggingFace Hub [12]. All other models
presented in this paper will be released in openlyavailable repositories along with their compression
recipes, training metrics and hyper-parameters.

**A.10.2** **Dataset Statistics**

Dataset statistics are detailed in Table 18.

**A.10.3** **Computational Experiments**

**Upstream.** All upstream runs are in general computationally expensive due to the large batch sizes

7https://huggingface.co/datasets/bookcorpus
8https://huggingface.co/datasets/wikipedia
9https://huggingface.co/datasets/squad
10https://huggingface.co/datasets/glue
11https://huggingface.co/datasets/glue
12https://huggingface.co/bert-base-uncased


Dataset Train Eval

SQuAD (examples) 87599 10570

MNLI (examples) 392702 19628

QQP (examples) 363,846 40,430

Wikipedia (words) 6078422     
TBC (words) 74004228    
Table 18: Statistics for training and evaluation datasets

and huge datasets. In our experiments we make
use of 4x A100 40GB NVIDIA GPUs. In this
configuration, a single training epoch takes approximately 6 hours. Since the cost of such a large compute instance is high, these experiments were only
run with a single seed and without major hyperparameter exploration.
**Downstream.** Our downstream experiments make
use of various different GPU cards that were at
out disposal: 16GB V100, 11GB RTX 2080 Ti,
and 24GB RTX 3090. Each training epoch takes
approximately 30 minutes, and as a result the 30
epoch runs take approximately 15 hours. For these
experiments, we report mean results of three runs
with different random seeds.
**DeepSparse inference.** We pair our compressed
models with DeepSparse (NeuralMagic, 2021) a
publicly-available sparsity-aware CPU inference
engine. This CPU runtime can leverage both
structured and unstructured sparsity, and quantization to deliver high performance on commodity
CPUs. We ran DeepSparse on a 24-core Intel AWS
c5.12xlarge server with 24 cores, 96 vCPUs, 192
GB of RAM and an AVX-512 compatible instruction set. All models are exported using the standard


18


-----

ONNX[13] format.

**A.10.4** **Computational Packages**
Our experiments build on publicly available libraries to ensure ease of reproduction and extensibility. All of our implementations, training and
evaluation code are built on top of HuggingFace’s
Transformers [14] and Datasets [15] libraries, NeuralMagic’s SparseML [16] library for model compression, and their DeepSparse [17] engine for efficient
inference on commodity CPUs.

13https://onnx.ai/
14https://github.com/huggingface/transformers
15https://github.com/huggingface/datasets
16https://github.com/neuralmagic/sparseml
17https://github.com/neuralmagic/deepsparse


19


-----

