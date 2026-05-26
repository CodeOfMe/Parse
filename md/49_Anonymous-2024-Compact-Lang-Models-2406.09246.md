## Transformers, parallel computation, and logarithmic depth

#### Clayton Sanford[1], Daniel Hsu[1], and Matus Telgarsky[2]


1Department of Computer Science, Columbia University, New York, NY, USA
2Courant Institute, New York University, New York, NY, USA

#### February 15, 2024


**Abstract**


We show that a constant number of self-attention layers can efficiently simulate—and be simulated
by—a constant number of communication rounds of _Massively_ _Parallel_ _Computation._ As a consequence,
we show that logarithmic depth is sufficient for transformers to solve basic computational tasks that
cannot be efficiently solved by several other neural sequence models and sub-quadratic transformer
approximations. We thus establish parallelism as a key distinguishing property of transformers.

### 1 Introduction


The transformer (Vaswani et al., 2017) has emerged as the dominant neural architecture for many sequential
modeling tasks such as machine translation (Radford et al., 2019) and protein folding (Jumper et al., 2021).
Reasons for the success of transformers include suitability to modern hardware and training stability: unlike
in recurrent models, inference and training can be efficiently parallelized, and training is less vulnerable to
vanishing and exploding gradients. However, the advantages of transformers over other neural architectures can
be understood more fundamentally via the lens of _representation,_ which regards neural nets as parameterized
functions and asks what they can efficiently compute.
Many previous theoretical studies of transformers establish (approximation-theoretic and computational)
universality properties, but only at large model sizes (Yun et al., 2020; Pérez et al., 2021). These results
are not unique to transformers and reveal little about which tasks can be solved in a _size-efficient_ manner.
Several other works (e.g., Hahn, 2020; Merrill and Sabharwal, 2022; Sanford et al., 2023) give fine-grained
representational results in the scaling regime where context length grows but model depth is constant. In this
regime, basic algorithmic tasks like matching parentheses and evaluating Boolean formulas are impossible.
In this work, we identify parallelism as a key to distinguishing transformers from other architectures. While
recurrent architectures process their inputs serially, transformers allow independent interactions between the
input tokens, mediated by the inner products between query and key embeddings in self-attention units. We
leverage this property of self-attention to establish a formal connection between transformers and _Massively_
_Parallel_ _Computation_ _(MPC)_ (Karloffet al., 2010). Concretely, we design transformers that simulate MPC
protocols (and vice versa), and in doing so, we exhibit a wide range of computational tasks that are solved by
logarithmic-depth transformers, including tasks that cannot be efficiently solved with other architectures
such as graph neural nets and recurrent models.

#### 1.1 Our results


We advance the understanding of transformers’ representational capabilities with the following results.

1. The algorithmic capabilities and limitations of logarithmic-depth transformers are captured by the MPC
model (Section 3).

2. There is a simple sequential task that (i) is solved by (and, empirically, learned from data using) logarithmicdepth transformers, but (ii) _cannot_ be efficiently solved by several alternative architectures (Sections 4
and 5).

1


-----

In more detail, our first collection of results, Theorems 3.1 and 3.4, show that any _R-round_ MPC protocol
can be implemented by a transformer of depth O(R), and that any depth-L transformer can be simulated by an
_O(L)-round_ MPC protocol. The former implies that several graph problems are solved by logarithmic-depth
transformers (Corollary 3.3); the latter implies the near-optimality of these transformers (Corollary 3.5)
conditional on a well-known conjecture about the limitations of MPC algorithms (Conjecture 2.4). A key
technical step (Lemma 3.2) shows how transformers can implement the simultaneous message-passing used in
MPC protocols to communicate between machines. While previous works (Sanford et al., 2023) have used
communication complexity to understand the representational limitations of self-attention layers, our results
show the benefits of the communication lens for understanding the strengths of transformers as well.
Our second set of results concern the _k-hop_ _induction_ _heads_ task, a synthetic sequential task that draws
inspiration from the induction heads primitive of Elhage et al. (2021). The theoretical results of Section 4 prove
that depth _L = Θ(log k)_ is necessary and sufficient for efficient transformer representation. An accompanying
empirical investigation reveals that transformers trained on the task obey the same threshold and recover
a similar model to the theoretical construction. In contrast, Section 5 illustrates that non-parallelizable
recurrent architectures—including state-space models like Mamba (Gu and Dao, 2023)—are unable to solve
the task in a size-efficient manner. Moreover, well-known transformer models with computationally-efficient
alternatives to self-attention, like Performer (Choromanski et al., 2022) and Longformer (Beltagy et al.,
2020), and shallow transformers with chain-of-thought prompting sacrifice their abilities to implement parallel
algorithms, as evidenced by their proven inability to solve this task.

#### 1.2 Related work

Some of the types of lower bounds we sought in this work were inspired by the literature on depth-separation
for feed-forward neural networks (e.g., Eldan and Shamir, 2016; Daniely, 2017; Telgarsky, 2016), which exhibit
functions that are efficiently approximated by deep networks, but not by shallower networks.
Many theoretical approaches have been used to understand the representational capabilities of transformers
and self-attention units in various scaling regimes. Some works model (variants of) transformers as machines
for recognizing formal languages, such as the Dyck languages (Hahn, 2020; Bhattamishra et al., 2020; Yao
et al., 2021; Hao et al., 2022) and star-free regular languages (Angluin et al., 2023). These approaches reveal
inability of fixed-size transformers to handle arbitrarily long inputs. Other works show how transformers
can simulate finite-state automata Liu et al. (2022) with logarithmic depth, and Turing machines with
(unrolled) depth (or chain-of-thought length) scaling polynomially with total runtime Wei et al. (2021);

Malach (2023); Merrill and Sabharwal (2023b). However, it is unclear if these results are near optimal or
even transformer-specific.
Theoretical results about the limitations of constant-depth transformers have been articulated by way of
analogy to circuit complexity (Merrill and Sabharwal, 2023a; Merrill et al., 2022; Merrill and Sabharwal, 2022;
Strobl, 2023; Strobl et al., 2023), implying the inability of constant-depth transformers to solve tasks like
graph connectivity and Boolean formula evaluation. Other works characterize the representational capabilities
of one-layer transformers (Likhosherstov et al., 2021; Sanford et al., 2023), but these approaches do not apply
to deeper models. Sanford et al. study multi-headed attention using communication complexity, a framing
that informs this work’s connection to distributed computing.
The MPC model Karloffet al. (2010); Beame et al. (2017); Goodrich et al. (2011); Andoni et al. (2014);
Im et al. (2023) was introduced to study distributed computing frameworks such as MapReduce Dean and
Ghemawat (2004). A major goal is to design protocols that use few rounds of communication for setups
in which each machine’s local memory is sublinear in the input size. Many advances have been made in
MPC algorithms for important problems (see, e.g., Im et al., 2023, for a recent survey). However, a basic
problem that has resisted progress is connectivity in sparse graphs, where all MPC protocols in this memory
regime appear to require Ω(log n) rounds for input graphs on _n_ vertices. Lower bounds in MPC and related
models were studied by Beame et al. (2017), Roughgarden et al. (2018), and Charikar et al. (2020). The
conjectured impossibility of _o(log n)-round_ protocols for connectivity is now used as basis for conditional
lower bounds (Ghaffari et al., 2019).
Simulation of transformers by recurrent models (Oren et al., 2024) and simulation of graph neural nets
(GNNs) by transformers (Kim et al., 2022) offer some coarse-grain insight into the relationship between these

2


-----

architectures, but separations are not implied by these previous works. Our connection between transformers
and MPC is most similar to that established by Loukas (2019) between GNNs and the Congest model of
distributed computation. Both works establish positive and negative results by identifying neural architectures
with communication protocols. In Section 5.1, we show that the MPC connection allows transformers solve
graph connectivity more efficiently than GNNs.
Our _k-hop_ induction heads task is designed as a _k-fold_ composition of its standard analogue (Elhage
et al., 2021). It is similar to a special case of the LEGO reasoning task Zhang et al. (2023), which reveals
the super-linear benefit of depth with respect to _k;_ in our case, we theoretically and empirically exhibit an
exponential benefit. We also draw a connection to the well-studied problem of pointer-chasing (Papadimitriou
and Sipser, 1982; Duris et al., 1984; Nisan and Wigderson, 1993), which enables the proof of our separation
between parallel and serial architectures. Our fine-grained empirical interpretability analysis for synthetic
tasks draws inspiration from similar approaches for the analysis of sequential algorithms like sorting and
reversal (Li and McClelland, 2022).

### 2 Preliminaries

#### 2.1 Transformers

We first define a self-attention head, the core primitive of a transformer. The softmax operator is softmax(v) =
(exp(v1), . . ., exp(vN ))/ [�]j[N]=1 [exp][(][v][j][)] [for] _[v]_ _[∈]_ [R][N] [.] [We] [apply] [softmax] [to] [matrices] _[A]_ _[∈]_ [R][N] _[×][N]_ [row-wise,] [i.e.]
softmax(A)i = softmax((Ai,1, . . ., Ai,N )).

**Definition** **2.1** (Self-attention head). A _self-attention_ _head_ is a mapping _fQ,K,V_ : R[N] _[×][m]_ _→_ R[N] _[×][m]_ defined
by
_fQ,K,V (X) = softmax(Q(X)K(X)[T])V (X)_

and parameterized by row-wise _query,_ _key,_ and _value_ _embeddings_ _Q, K, V_ : R[N] _[×][m]_ _→_ R[N] _[×][m]_ (e.g., _Q(X) =_
(Q1(X1), . . ., QN (XN )). Let Attn[N]m [denote] [the] [set] [of] [all] [self-attention] [heads] [with] [embedding] [dimension] _[m]_
and context length _N_ .

A transformer composes L layers of H self-attention heads per layer, plus an output multi-layer perceptron
(MLP).

**Definition** **2.2** (Transformer). A _transformer_ is a mapping T : R[N] _[×][d][in]_ _→_ R[N] _[×][d][out]_ specified by self-attention
heads (fℓ,h _∈_ Attn[L]m[)]ℓ∈[L],h∈[H] [and] [an] [element-wise] [output] [MLP] _[ψ]_ [=] [(][ψ][1][, . . ., ψ][N] [)] [:] [R][N] _[×][m]_ _[→]_ [R][N] _[×][d][out][.]_
Upon input _X_ _∈_ R[N] _[×][d][in],_ the transformer computes intermediate embeddings _X_ [0], . . ., X _[L]_ _∈_ R[N] _[×][m]_ with
_X_ [0] = X and
_H_
_X_ _[ℓ]_ = X _[ℓ][−][1]_ + �

_h=1[f][ℓ,h][(][X]_ _[ℓ][−][1][)][,]_

and returns _T_ (X) = ψ(X _[L])_ as output. Let Transformer[N]m,L,H,din,dout [denote] [the] [set] [of] [all] [such] [transformers,]
and Transformer[N]m,L,H [:=][ Transformer]m,L,H,[N] 1,1[.]

**Modeling** **assumptions.** We treat the transformer as a computational model that permits arbitrary
element-wise computation, but restricts the manner in which multiple elements are processed together.
This manifests in our decisions to model query/key/value embeddings and MLPs as arbitrary functions
on the embedding space; Loukas (2019) employs a similar modeling assumption for GNNs. Note that the
element-wise embeddings and MLPs may be index-specific, obviating the need for positional embeddings.
Our theoretical results cover the scaling regime where the context length _N_ is the main asymptotic
parameter; while the embedding dimension _m,_ the number of heads _H,_ and the depth _L_ grow sub-linearly in
_N_ . This reflects real-world trends in large-language models, where context length has sharply increased in
recent years.
Throughout, we assume all intermediate computations in transformers are represented by _p-bit_ precision
numbers for _p = Θ(log N_ ). Limiting the precision is consistent with recent practice of using low-precision
arithmetic with transformers (e.g., Wang et al., 2022; Dettmers et al., 2022). We discuss this precision
assumption in greater detail in Appendix A.1, along with other minor technical assumptions (such as the
inclusion of a “start token” for mathematical convenience).

3


-----

Figure 1: Formal execution of an MPC protocol for computing _f :_ Z[n]2[p][in] _[→]_ [Z][n]2[p][out]. (|Msg| is the number of
words in `Msg.)`

**Masked** **transformers.** We also consider masked _self-attention,_ where only certain inner products influence
the softmax output. Let Λ ∈{−∞, 0}[N] _[×][N]_ be a _masking_ _matrix_ with at least one zero entry in every row.
Then, a Λ-masked _self-attention_ unit is defined by

_fQ,K,V[Λ]_ [(][X][) = softmax(][Q][(][X][)][K][(][X][)][T][ + Λ)][V][ (][X][)][.]

Let Λ-Attn[N]m [and] [Λ-][Transformer][N]m,L,H [,] [respectively,] [denote] [the] [sets] [of] [all] [Λ-masked] [self-attention] [heads] [and]
all transformers comprised of those heads. We define causally-masked _transformers_ by MaskAttn[N]m [:= Γ-][Attn]m[N]
and MaskTransformer[N]m,L,H [:=] [Γ-][Transformer][N]m,L,H [,] [where] [Γ] [is] [the] [lower-triangular] [mask] [with] [Γ][i,j] [=] [0] [iff]
_i ≥_ _j._

#### 2.2 Massively Parallel Computation model

We use the definition of MPC from Andoni et al. (2018).

**Definition** **2.3** (MPC protocol). For any global and local memory constants _γ, δ_ _> 0,_ a (γ, δ)-MPC _protocol_
for a function _f_ : Z[n]2[p][in] _[→]_ [Z][n]2[p][out] specifies a distributed computing protocol for _q_ = Θ(n[1+]in _[γ][−][δ])_ machines,
each with _s_ = _O(n[δ]in[)]_ [words][1] [of] [local] [memory] [to] [jointly] [compute] _[f]_ [(][Input][)] [for] [any] [given] `[Input]` _[∈]_ [Z][n]2[p][in] [as]
follows. The Input ∈ Z[n]2[p][in] [is distributed across the local memories of the first][ ⌈][n][in][/s][⌉] [machines.] [Computation]
proceeds in rounds. In each round, each machine computes an arbitrary function of its local memory to
prepare at most s words to send to other machines; messages are simultaneously transmitted, and the protocol
ensures that each machine receives at most _s_ words at the end of the round. After the final round, the
`Output = f` (Input) ∈ Z[n]2[p][out] is in the local memories of the first _⌈nout/s⌉_ machines. See Figure 1 for details.

Our negative results in Section 3.2 are conditional on the well-known “one-versus-two cycle” conjecture
(Beame et al., 2017; Roughgarden et al., 2018; Ghaffari et al., 2019).

1We assume the word size is _p = Θ(log nin)_ bits. For convenience, we regard words as elements of Z2p (integers mod 2p).

4


-----

**Conjecture** **2.4** (see, e.g., Ghaffari et al., 2019). _For_ _any_ _γ_ _>_ 0, _δ_ _<_ 1, _and_ _N_ _,_ _if_ _π_ _is_ _an_ (γ, δ)-MPC
_protocol_ _that_ _distinguishes_ _a_ _single_ _cycle_ _on_ _N_ _nodes_ _and_ _a_ _union_ _of_ _two_ _cycles_ _each_ _on_ _N/2_ _nodes,_ _then_ _π_
_uses_ Ω(log N ) _rounds._

#### 2.3 Graphs as sequential inputs

When providing a graph _G = (V, E)_ as input to transformers or MPC protocols, we serialize _G_ as a sequence
in [|V |][2][|][E][|] that encodes each edge as a pair of vertex tokens. The resulting transformer has _N_ = 2|E| and
_din_ = 1, and the resulting MPC protocol has _nin_ = 2|E|.

### 3 Relating transformers and MPC

We coarsely characterize the computational power of transformers in a certain size regime by establishing a
bidirectional relationship between transformers and MPC. Theorems 3.1 and 3.4 show that any MPC protocol
can be simulated by a transformer, and vice versa. As corollaries (Corollaries 3.3 and 3.5), we obtain tight
upper and lower bounds on the depth of bounded-size transformers for computing connected components in
graphs.

#### 3.1 Simulation of MPC protocols by transformers

The following theorem shows that any MPC protocol _π_ with sublinear local memory can be simulated by
a transformer whose depth _L_ is linear in the number of rounds _R_ of _π,_ and embedding dimension _m_ is
polynomial in the local memory size _s = O(N_ _[δ])_ of machines used by _π._

**Theorem** **3.1.** _For_ _constants_ 0 _<_ _γ_ _<_ _δ_ _<_ 1 _and_ _any_ _deterministic_ _R-round_ (γ, δ)-MPC _protocol_ _π_
_on_ _nin_ _input_ _words_ _and_ _nout_ _≤_ _nin_ _output_ _words,_ _there_ _exists_ _a_ _transformer_ _T_ _∈_ Transformer[N]m,L,H _[with]_
_N_ = _nin, m_ = _O(n[4]in[δ]_ [log][ n][in][)][, L] [=] _[R][ + 1][, H]_ [=] _[O][(][log log][ n][in][)]_ _[such]_ _[that]_ _[T]_ [(][Input][)][:][n]out [=] _[π][(][Input][)]_ _[for]_ _[all]_
`Input ∈` Z[N]2[p][.]

The theorem provides a non-trivial construction in the strongly sub-linear local memory regime when
_s = O(N_ [1][/][4][−][ϵ]) for any _ϵ > 0.[2]_ Whether the simulation can be improved to _m = O(N_ [1][−][ϵ][′]) for some _ϵ[′]_ _> 0_
whenever _s = O(N_ [1][−][ϵ]) is an interesting question for future work.

**Theorem** **3.1** **proof** **overview.** At a high level, the proof in Appendix B.2 entails simulating each round
of parallel computation with a single-layer transformer and applying those constructions serially to `Input.`
The local computation on each machine (represented by `MachineOut[(]i[r][)]` = Localr,i(MachineIn[(]i[r][)][))] [is] [directly]
encoded using element-wise query/key/value embeddings.
The crux of the proof involves the simulation of a _routing_ _protocol_ to determine `MachineIn[(][r][+1)]` from
`MachineOut[(][r][)].` We construct a self-attention unit that ensures that an encoding of a sequence of addressed
messages from each machine are properly routed to their destinations.[3]

For any message size _β,_ message count bound _s,_ and number of tokens _N_, we say that (Sent, Rcvd) _∈_
R[N] _[×][m]_ _× R[N]_ _[×][m]_ is a _valid_ (β, s)-routing if, for each _i ∈_ [N ], the _i-th_ row of `Sent` (resp. `Rcvd)` is the vector
encoding of some `Senti` _⊂_ Z[β]2[p] _[×][ [][N]_ []] [(resp.] `[Rcvd][i]` _[⊂]_ [Z][β]2[p] _[×][ [][N]_ [])] [such] [that]

`Rcvdi` = {(Msg, Src) : (Msg, i) ∈ `SentSrc},`

and each of `Rcvdi` and `Senti` has cardinality at most _s.[4]_

**Lemma** **3.2.** _For_ _any_ _β, s, N_ _∈_ N, _there_ _exists_ _a_ _transformer_ routeβ,s _∈_ Transformer[N]m,1,1 _[with]_ _[m]_ [=]
_O(s[4]β log N_ ) _satisfying_ routeβ,s(Sent) = Rcvd _for_ _any_ _valid_ (β, s)-routing (Sent, Rcvd).

2Applying Theorem 3.1 when _δ_ _≥_ 41 [yields] [transformers] [with] [embedding] [dimension] _[m][ ≥]_ _[N]_ [,] [which] [trivializes] [the] [transformer]

architecture and negates any advantages of depth under our MLP universality assumption. This is due to the fact a transformer
with _N_ -dimensional embeddings could aggregate the entire input sequence _X_ _∈_ R[N] in a single embedding and use its output
MLP to compute any arbitrary function on that input.
3This routing between machines uses the all-pairs structure of self-attention and may not admit a subquadratic approximation.
4We abuse notation by writing `Dest ∈` `Senti` to mean there exists some `Msg` such that (Msg, Dest) ∈ `Senti.`

5


-----

The proof of Lemma 3.2 appears in Appendix B.1 and combines two key techniques: sparse propagation
and multiple hashing. The former is a simple variant of the “sparse averaging” task of Sanford et al. (2023),
which simultaneously computes _N_ averages over subsets of inputs; this task is solved a single self-attention
head with small embedding dimension (Proposition B.1). Using sparse propagation, we construct a selfattention head that averages the ≤ _s_ encodings of each RcvdSrc for every Src ∈ `Rcvdi.` In order to ensure that
we can decode that average of encodings, we apply error-correction by encoding each `Outputi` in a sparse and
redundant manner, where each outgoing messages appears as multiple copies of the same addressed “packet.”

**Application:** **connectivity with log-depth transformers.** As an immediate consequence of Theorem 3.1,
any graph problem solvable with a logarithmic number of rounds of MPC computation (and local memory _s)_
is also computable by a logarithmic depth transformer (and embedding dimension _O[˜](s[4]))._ The following
result—which bounds transformer depth needed to compute connected components of a graph _G—follows_
from Theorem 6.2 of Coy and Czumaj (2022), which derandomizes an MPC algorithm of Behnezhad et al.
(2019), and Theorem 3.1.

**Corollary** **3.3.** _For_ _any_ _constant_ _ϵ ∈_ (0, 1) _and_ _any_ _D_ _≤_ _N_ _,_ _there_ _exists_ _a_ _transformer_ _in_ Transformer[N]m,L,H
_with_ _m = O(N_ _[ϵ]),_ _H_ = O(log log N ), _and_ _L = O(log D)_ _that_ _identifies_ _the_ _connected_ _components_ _of_ _any_ _input_
_graph_ _G = (V, E)_ _with_ _|V |, |E| = O(N_ ) _where_ _each_ _connected_ _component_ _has_ _diameter_ _at_ _most_ _D._

Coy and Czumaj also give efficient MPC algorithms for other related problems (e.g., spanning forest), so
we obtain efficient transformers for these problems, too (Appendix B.3).

#### 3.2 Simulation of transformers by MPC protocols

The following theorem shows that MPC protocols can simulate transformers and prove depth lower bounds
on transformers, conditioned on Conjecture 2.4. We get, as a corollary, the conditional optimality of the
transformer depth bound in Corollary 3.3.

**Theorem** **3.4.** _For_ _any_ _transformer_ _T_ _∈_ Transformer[N]m,L,H _[(or]_ [Λ-][Transformer][N]m,L,H _[)]_ _[with]_ _[mH]_ [=][ O][(][N][ δ][)] _[for]_
_δ_ _∈_ (0, 1) _and_ _any_ _δ[′]_ _∈_ (δ, 1), _there_ _exists_ _a_ _O(_ _δ[′]L−δ_ [)][-round] [(1+] _[δ][′][, δ][′][)][-MPC]_ _[protocol]_ _[with]_ _[q]_ [=][ O][(][N][ 2][)] _[machines]_
_with_ _s = O(N_ _[δ][′])_ _local_ _memory_ _for_ _computing_ _T_ _._

Theorem 3.4 demonstrates that the algorithmic capabilities of transformers are no stronger than those of
MPC protocols with a quadratic scaling in the number of machines. While Theorems 3.1 and 3.4 do not
jointly provide a sharp characterization of the two computational models, the reductions are tight enough to
provide strong evidence for the optimality of the connected components construction of Corollary 3.3.

**Theorem** **3.4** **proof** **overview.** At a high-level, the proof constructs an MPC protocol that simulates
a self-attention layer by separating the computation of MLPs and attention matrices into three separate
categories of machines.

- Each input token is provided to its own _token_ _machine,_ responsible for preparing the query/key/value
embeddings.

- Each pair of tokens is associated with an _inner_ _product_ _machine_ that will compute the inner product
between their respective query and key embeddings.

- _Propagation machines_ ensure that embeddings are routed to the proper inner product machine and compute
outputs of each softmax unit.
The proof gives the communication protocol for these machines, shows how they simulate a layer of selfattention in _O(1/(δ[′]_ _−_ _δ))_ rounds, and establishes the sufficiency of _O(N_ [2]) machines with _O(N_ _[δ][′])_ local
memory.

**Application:** **conditional** **optimality** **of** **Corollary** **3.3.** Assuming the well-established Conjecture 2.4,
we prove an Ω(log D) lower bound on the depth of parameter-efficient transformers for determining connectivity
of graphs where connected components may have diameter up to _D._

**Corollary** **3.5.** _Let_ _ϵ ∈_ (0, 1) _be_ _any_ _constant,_ _and_ _let_ _D_ _≥_ _N_ _[ϵ]._ _Assume_ _Conjecture_ _2.4,_ _and_ _suppose_ _there_
_exists_ _T_ _∈_ Transformer[N]m,L,H _[with]_ _[mH]_ [=][ O][(][D][1][−][ϵ][)] _[that]_ _[decides]_ _[connectivity]_ _[of]_ _[any]_ _[input]_ _[graph]_ _[with]_ _[connected]_
_components_ _having_ _diameter_ _≤_ _D._ _Then_ _L = Ω(log D)._

6


-----

### 4 Transformers for k-hop induction heads

We complement the generality of Section 3 by studying, both empirically and theoretically, a specific toy
sequential modeling task which will also serve (in Section 5) as a problem to separate the representational
capabilities of transformers from that of other neural architectures.
This task, called the _k-hop_ _induction_ _heads_ task, draws inspiration from the original _induction_ _heads_ task
defined and analyzed on trained language models and in synthetic environments by Elhage et al. (2021) (see
also Bietti et al., 2023). The standard induction heads task completes bigrams auto-regressively by predicting
the token that follows the last previous occurrence of the final token in the sequence. For example, given the
input _X_ = baebcabebdea, the standard induction heads task is to complete the final bigram by predicting `b`
for the final token.
The _k-hop_ induction heads tasks generalizes this mechanism by repeatedly using the completion of a
bigram to determine the next bigram to complete. In the previous example, the 2-hop induction heads task
is to predict `c` for the final token:
```
                    baebcabebdea.

```
**Definition** **4.1.** For any finite alphabet Σ, define the map hopk : Σ[N] _→_ (Σ ∪{⊥})[N] by hopk(X)i = XfindkX [(][i][)]
if find[k]X [(][i][)][ ̸][= 0] [and] _[⊥]_ [otherwise,] [where]

find[1]X [(][i][) = max(][{][0][} ∪{][j] _[∈]_ [N][ :][ j] _[≤]_ _[i,]_ _[X][j][−][1]_ [=][ X][i][}][);]

find[k]X [(][i][) = find][1]X [(find]X[k][−][1](i)) for _k_ _≥_ 2.

The k-hop _induction_ _heads_ _task_ is to compute, for each i = 1, . . ., N, the value of hopk(X)i from (X1, . . ., Xi).

We note a similarity to the LEGO tasks of Zhang et al. (2023), who empirically study the ability of
transformers to learn sequential operations on Abelian groups and observe the ability to perform more
operations than the depth of the network.

#### 4.1 Log-depth transformer for k-hop induction heads

Although hopk appears to requires _k_ steps to solve, we show that it is solved by a transformer of depth
_O(log k)._

**Theorem** **4.2.** _For_ _any_ _k_ _∈_ N _and_ _alphabet_ Σ _with_ _|Σ|_ _≤_ _N_ _,_ _there_ _exists_ _T_ _∈_ MaskTransformer[N]m,L,H _[that]_
_computes_ hopk : Σ[N] _→_ (Σ ∪{⊥})[N] _with_ _m = O(1),_ _L = ⌊log2 k⌋_ + 2, _and_ _H_ = 1.

In contrast to Corollary 3.3, this construction has constant embedding dimension and is achieved by a
causally-masked transformer. As such, its proof in Appendix D.1 depends on other techniques that exploit
the simplicity of the problem and build on the induction heads construction of Bietti et al. (2023), rather
than simply applying Theorem 3.1.
We give evidence for the optimality of this construction by proving a conditional lower bound using
Theorem 3.4, as was done in Corollary 3.5.

**Corollary** **4.3.** _Assuming_ _Conjecture_ _2.4,_ _for_ _any_ _constants_ _ξ_ _∈_ (0, 1/2] _and_ _ϵ_ _∈_ (0, 1), _and_ _any_ _even_
_k_ = Θ(N _[ξ]),_ _every_ _transformer_ _T_ _∈_ MaskTransformer[N]m,L,H _[with]_ _[mH]_ [=][ O][(][k][1][−][ϵ][)] _[that]_ _[computes]_ [hop]k _[has]_ _[depth]_
_L = Ω(log k)._

#### 4.2 Log-depth transformer learned from data

We empirically assess whether the representational trade-offs elucidated by tasks efficiently solved by
parallelizable algorithms have implications for optimization and generalization properties of transformers. To
that end, we trained auto-regressive transformer architectures of varying sizes to solve hopk(X) for a variety
of values of _k_ in order to understand how changing depth impacted the performance of the learned models,
the goal being to verify the sufficiency of logarithmic depth, just as in our theory.

7


-----

Figure 2: Evaluation of transformers of depths _L_ _∈{2, 3, 4, 5, 6}_ trained on a mixture of hopk for _k_ _∈_
_{0, . . ., 16}_ evaluated on n = 100 samples of size N = 100 from each hopk. Incrementing depth approximately
doubles the largest _k_ such that hopk is learnable with small error.

In brief, we trained transformers with 500K to 5M parameters and depths _{2, 3, 4, 5, 6}_ with Adam
to solve hopk(X) for _k_ _∈{0, . . ., 16}_ with context length _|N_ _|_ = 100 and alphabet size _|Σ|_ = 4. We
trained the transformers in a multi-task setting, where a single model was trained to predict the sequence
hopk(X) auto-regressively when provided with _X_ and _k_ drawn at random. Further experimental details
can be found in Appendix G.1, and the experimental code is available at `https://github.com/chsanford/`
```
hop-induction-heads.

```
We found that transformers are indeed capable of learning hopk given sufficient training time, and that the
largest learnable _k_ grows exponentially with the depth. As can be seen in Figure 2, a six-layer neural network
performs well on all _k_ _≤_ 16, a five-layer on _k_ _≤_ 8, a four-layer on _k_ _≤_ 4, and so forth. We further explore
these experimental results in Appendix G.2 and observe a performance threshold appears to specifically lie at
_⌊log2 k⌋_ + 2 that coincides with Theorem 4.2. This logarithmic dependence of the depth on _k_ persists in a
larger-width regime, which is explored in Appendix G.3. In the finite sample regime where neural networks
are prone to overfit, our investigations in Appendix G.5 note improved generalization in deeper models, which
suggests that deeper models have a favorable inductive bias for tasks like hopk.
Moreover, the learned models are surprisingly interpretable. We examined the activation patterns of
attention matrices, and found close correspondences to useful intermediate products such as find[j]X [.] [Taken]
together, these indicate that the learned models mechanistically resemble the construction employed in the
proof of Theorem 4.2. See Appendix G.4 for our investigation of model interpretability.

### 5 Separations between transformers and alternative architectures

Sections 3 and 4 characterize the representational capability of transformers by providing algorithmic problems
they can solve with logarithmic depth and small polynomial or constant width. In contrast, other well-known
architectures are unable to solve those same problems in a parameter-efficient manner. This section provides
lower bounds on the parameter complexity of graph neural networks (GNNs), recurrent neural architectures,
transformers with computationally efficient alternatives to softmax self-attention, and single-layer transformers
with autoregressive chain-of-thought tokens needed to solve graph connectivity and the _k-hop_ task.

8


-----

#### 5.1 GNNs need polynomial depth for graph connectivity

The bidirectional relationship between transformers and MPC draws inspiration from past work drawing a
similar connection between message passing graph neural networks (GNNmp) and the Congest distributed
computing model Loukas (2019). Their computation model of GNNmp for width _m_ and depth _L_ closely
resembles our Transformer[N]m,L,H [in] [providing] [a] [general] [framework] [for] [the] [analysis] [of] [graph] [neural] [networks]
by allowing unbounded computation in each vertex with bounded communication on edges. On some input
graph _G,_ vertices send neighbors messages of size at most _m—which_ are aggregated and crafted into new
messages with MLPs—over _L_ rounds of communication.
By restating Corollary 4.2 of Loukas (2019), we demonstrate a sharp contrast in the abilities of GNNs
and transformers to solve graph algorithmic tasks.

**Theorem** **5.1** (Corollary 4.2 of Loukas (2019)). _There_ _exists_ _a_ _graph_ _G_ _with_ _N_ _edges_ _such_ _that_ _any_ GNNmp
_with_ _width_ _m_ _and_ _depth_ _L_ _that_ _determines_ _whether_ _an_ _input_ _subgraph_ _H_ _either_ _(1)_ _is_ _connected_ _or_ _(2)_ _forms_
_a_ _spanning_ _tree_ _of_ _G_ _requires_ _L[√]m =_ Ω([˜] _N_ [1][/][4]).

While Corollaries 3.3 and B.8 demonstrate the ability of transformers to determine whether any input
graph is connected[5] or to identify a spanning tree with logarithmic depth and small polynomial width (i.e.
_m = O(N_ _[ϵ])),_ GNNs require depth _L =_ Ω[˜] (N [1][/][4][−][ϵ/][2]) in the same regime. This gap is explainable by the fact
that transformers on graph inputs _G_ are not bound to pass messages exclusively along the edges of _G._ By
“rewiring” the graphical structure in each layer, transformers can perform aggregation and “pointer passing”
tasks with greater parametric ease than GNNs.

#### 5.2 Suboptimality of recurrent architectures for hopk
The logarithmic-depth and constant-width transformer implementation of hopk in Theorem 4.2 cannot be
replicated by recurrent neural architectures (Chung et al., 2014; Bengio et al., 1994; Turkoglu et al., 2021),
including not just multi-layer recurrent neural networks (RNNs) but any sequential prediction procedure
equivalent to them at inference time, which includes state space models such as Mamba (Gu and Dao, 2023).
We first consider a family of multi-layer RNNs of depth L and width _m,_ consisting of arbitrary MLP units
_gℓ_ : R[m][×][m] _→_ R[m][×][m], which on input X _∈_ R[N] _[×][d][in]_ produce output Y _∈_ R[N] _[×][d][out]_ as follows using intermediates
_X_ = Z [0], Z [1], . . ., Z _[L][−][1], Z_ _[L]_ = Y _∈_ R[N] _[×][m][6]and_ hidden states _H_ [1], . . ., H _[L]_ _∈{0, 1}[N]_ _[×][m]_ with _H0[ℓ]_ [=][ ⃗][0:]

(Zi[ℓ][, H]i[ℓ][) =][ g][ℓ][(][Z]i[ℓ][−][1], Hi[ℓ]−1[)][,] _[∀][i][ ∈]_ [[][N] []][, ℓ] _[∈]_ [[][L][]][.]

We provide a polynomial bound on the width and depth of a multi-layer RNN solving hopk.

**Corollary** **5.2.** _A_ _multi-layer_ _RNN_ _of_ _depth_ _L_ _and_ _width_ _m_ _as_ _above_ _with_ _YN_ = hopk(X)N _satisfies_ _either_
_L ≥_ _k_ _or_ _m = Ω(_ _k[N][6][ )][.]_

In contrast to Theorem 4.2, which demonstrates that depth _O(log k)_ transformers with constant width
suffice to solve hopk for any _k,_ Corollary 5.2 demonstrates that all multi-layer RNNs with width _O(N_ [1][/][7])
require depth _k_ when _k_ = O(N [1][/][7]).
Mamba (Gu and Dao, 2023) can be seen as the combination of three ideas: (1) a continuous-time
dynamics model of sequential prediction, powerful enough to model Kalman filters, hidden markov models,
and many others; (2) a family of time-discretization schemes; (3) an unrolling technique to enable efficient
linear-time training, using ideas similar to FlashAttention (Dao et al., 2022). Ultimately, at inference time,
the time-discretization step results in an RNN (see Gu and Dao, 2023, Algorithm 2 and Theorem 1), and is
therefore directly handled by Corollary 5.2.
This corollary is a near immediate application of a communication complexity fact about the hardness
of solving multi-player _pointer-chasing_ problems with limited communication among players (Guha and
McGregor, 2009; Assadi and N, 2021). We provide the communication model and this result in Appendix E.1,
and the reductions necessary to prove the above hardness results in Appendix E.2.

5While the problem of subgraph connectivity for GNNs may at first glance appear more difficult than general graph connectivity
for transformers, an implementation of this exact task can be implemented by modifying the protocol Corollary 3.3 to remove all
edges from the graph that do not belong to _H._
6We assume that _din, dout_ _≤_ _m_ and treat _X_ and _Y_ as if they are padded with zeros.

9


-----

#### 5.3 Suboptimality of sub-quadratic attention transformers for hopk

Due to the quadratic computational cost of computing the attention matrix softmax(Q(X)K(X)[T] ) ∈ R[N] _[×][N]_

and the continued desire for ever-larger context lengths, there is substantial interest in improving the
computational complexity of the transformer architecture while preserving its expressive capabilities and
inductive biases. As a result, a rich literature has emerged that proposes computationally-efficient alternatives
to standard softmax attention. In this section, we demonstrate how several representative examples of
sub-quadratic attention mechanisms lose the ability to perform efficient parallel computation under a
logarithmic-depth scaling.

**Kernel-based** **sub-quadratic** **attention.** One approach to computationally-efficient approximation of
transformers are kernel-based sub-quadratic attention mechanisms such as Performer (Choromanski et al., 2022),
and Poly-Sketchformer (Kacham et al., 2023). Both approximate the attention matrix softmax(Q(X)K(X)[T])
with a low-rank matrix _Q[′](X)K_ _[′](X)[T]_ where _Q[′], K_ _[′]_ : R[m] _→_ R[m][′] are applied element-wise. For sufficiently
small _m[′]_ _≪_ _N_, _Q[′](X)K_ _[′](X)[T]V (X)_ can be computed efficiently by first computing _K_ _[′](X)[T]V (X) ∈_ R[m][′][×][m],
bounding the total runtime as _O(Nmm[′]),_ rather than _O(N_ [2]m).
Let KernelFormer[N]m,m[′],L,H [denote] [all] _[H][-headed]_ _[L][-layer]_ [transformer] [whose] [softmax] [attention] [modules] [are]
replaced by kernel-based sub-quadratic attention. We demonstrate the limitations of KernelFormer[N]m,m[′],L,H
by showing that, unlike Transformer[N]m,L,H [,] [they] [have] [no] [depth-efficient] [implementation] [of] [hop]k[.]

**Corollary** **5.3.** _Any T_ _∈_ KernelFormer[N]m,m[′],L,H _[with][ T]_ [(][X][)][N] [=][ hop]k[(][X][)][N] _[satisfies either][ L][ ≥]_ _[k]_ _[or][ mm][′][Hp][ =]_
Ω( _k[N][6][ )][.]_

Under a parameter-efficient regime where _mpHL_ = _O(N_ _[ϵ]),_ solving hopk for _k_ = Θ(N _[ϵ])_ necessitates
kernel feature dimension _m[′]_ = Ω(N [1][−][9][ϵ]), which forces each attention unit to compute an _N_ _× N_ [1][−][9][ϵ] matrix,
yielding a nearly quadratic runtime. We prove Corollary 5.3 in Appendix E.3 using a similar pointer chasing
reduction.

**Masking-based** **sub-quadratic** **attention.** Another method that reduces the computational cost of
transformers is to used masked models of Λ-Transformer[N]m,L,H [for] [a] [sparse] [mask] [Λ.] The Longformer
architecture (Beltagy et al., 2020) introduces a particular masked architecture that combines sliding windows
with sparse unmasked global tokens. Put concretely, for window radius _w_ and global frequency _g,_ let
Λ[w,g] _∈{−∞, 0}[N]_ _[×][N]_ be masking matrix with


Λ[w,g]i,j [=]


�0 if _|i −_ _j| ≤_ _w_ or _j_ _≡_ 0 (mod _g),_
_−∞_ otherwise.


Then, the output of a single unit of Λ[w,g]-masked attention is computable in time _O((w +_ _[N]g_ [)][Nm][).]

**Corollary** **5.4.** _Any_ _T_ _∈_ Λ[w,g]-Attn[N]m,L,H _[with]_ _[T]_ [(][X][)][N] [=][ hop]k[(][X][)][N] _[satisfies]_ _[either]_ _[L][ ≥]_ _[k]_ _[or]_ [(][w] [+] _gk[N]_ [)][mHp][ =]

Ω( _k[N][6][ )][.]_

Like kernel-based attention, sparsely-masked attention models fail to efficiently compute hopk. Similarly,
in the same parameter-efficient regime, a Longformer must have either _w_ = Ω(N [1][−][9][ϵ]) or _g_ = O(N [9][ϵ]), which
jointly ensures that the masked matrix has at least Ω(N [2][−][9][ϵ]) entries and diminishes any computational
advantages. This proof also appears in Appendix E.3.

#### 5.4 Limitations of 1-layer transformers with chain-of-thought

While most of the paper considers transformers as sequence-to-sequence models, we can also frame them as
auto-regressive models performing next-token-prediction with chain-of-thought prompting. In this regime, a
single causally-masked transformer aims to compute a function of its input by repeatedly predicting the next
token, appending previously predicted tokens to the end of the input. In doing so, a function is computable if
there exists an intermediate _chain-of-thought_ produced by the model that eventually reaches the answer.

10


-----

**Definition** **5.5.** We say that _T_ _∈_ MaskTransformerm,L,H[N] [+][N][CoT] computes _f_ : Σ[N] [+][N][CoT] _→_ Σ[N], where the
additional _N_ tokens denote chain-of-thought, if for every _X_ _∈_ dom(f ), there exists _XCoT_ _∈_ Σ[N][CoT] such that
_T_ (X _◦_ _XCoT)N_ :N +NCoT = (XCoT ◦ _f_ (X)).

The theoretical capabilities of chain-of-thought augmented transformers to simulate finite-state automata
and Turing machines have been studied (Malach, 2023; Merrill and Sabharwal, 2023b), but the comparative
capabilities of shallow models with chain-of-thought prompting and deep sequential models are unknown. In
contrast to the fact that any transformer with _NCoT_ tokens can be simulated by a sequential model with
depth scaled by _NCoT,_ we show that deep transformers cannot necessarily be efficiently simulated by shallow
chain-of-thought models. We do so by demonstrating that a linear amount of chain-of-thought prompting in
_k_ is necessary to solve hopk(X)N, and also sufficient.

**Corollary** **5.6.** _Any_ _T_ _∈_ MaskTransformer[N]m,[+]1[N],H[CoT] _that_ _computes_ hopk(X)N _with_ _NCoT_ _tokens_ _of_ _chain-of-_
_thought_ _requires_ _either_ _NCoT_ _≥_ _k_ _or_ _mHp = Ω(_ _k[N][6][ )][.]_

The proof appears in Appendix E.4. For future work, it remains to consider the comparative powers of
chain-of-thought models of depths greater than one.

### 6 Conclusion and future work

This work highlights parallelism as a central feature of transformers that sets them apart from other neural
architectures. The focus on the log-depth and sublinear-width regime and specific computational tasks allows
us to accentuate the benefits of parallelism, even for tasks like _k-hop_ that appear inherently serial at first
glance.
There is some efficiency loss in the “compilation” of MPC protocols to transformers that we hope to
understand better in future work. Furthermore, although we have empirically demonstrated the learnability
of transformers that exploit parallelism in crucial ways, a theoretical understanding of learning such solutions
remains an open question.

### References

Alekh Agarwal, Olivier Chapelle, Miroslav Dudík, and John Langford. A reliable effective terascale linear
learning system. _Journal_ _of_ _Machine_ _Learning_ _Research,_ 15(1):1111–1133, 2014.

Alexandr Andoni, Aleksandar Nikolov, Krzysztof Onak, and Grigory Yaroslavtsev. Parallel algorithms
for geometric graph problems. In _Proceedings_ _of_ _the_ _forty-sixth_ _annual_ _ACM_ _symposium_ _on_ _Theory_ _of_
_computing,_ pages 574–583, 2014.

Alexandr Andoni, Zhao Song, Clifford Stein, Zhengyu Wang, and Peilin Zhong. Parallel graph connectivity in
log diameter rounds. In _2018_ _IEEE_ _59th_ _Annual_ _Symposium_ _on_ _Foundations_ _of_ _Computer_ _Science_ _(FOCS)._
IEEE, October 2018. doi: 10.1109/focs.2018.00070. [URL http://dx.doi.org/10.1109/FOCS.2018.00070.](http://dx.doi.org/10.1109/FOCS.2018.00070)

Dana Angluin, David Chiang, and Andy Yang. Masked hard-attention transformers and boolean rasp
recognize exactly the star-free languages, 2023.

Sepehr Assadi and Vishvajeet N. Graph streaming lower bounds for parameter estimation and property
testing via a streaming xor lemma. In _Proceedings_ _of_ _the_ _53rd_ _Annual_ _ACM_ _SIGACT_ _Symposium_
_on_ _Theory_ _of_ _Computing,_ STOC ’21. ACM, June 2021. doi: 10.1145/3406325.3451110. URL `http:`
```
 //dx.doi.org/10.1145/3406325.3451110.

```
Paul Beame, Paraschos Koutris, and Dan Suciu. Communication steps for parallel query processing. _Journal_
_of_ _the_ _ACM_ _(JACM),_ 64(6):1–58, 2017.

Soheil Behnezhad, Sebastian Brandt, Mahsa Derakhshan, Manuela Fischer, MohammadTaghi Hajiaghayi,
Richard M Karp, and Jara Uitto. Massively parallel computation of matching and mis in sparse graphs. In
_Proceedings_ _of_ _the_ _2019_ _ACM_ _Symposium_ _on_ _Principles_ _of_ _Distributed_ _Computing,_ pages 481–490, 2019.

11


-----

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer, 2020.

Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult.
_IEEE_ _Transactions_ _on_ _Neural_ _Networks,_ 5(2):157–166, 1994. doi: 10.1109/72.279181.

Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to
recognize formal languages. In _Proceedings_ _of_ _the_ _2020_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_
_Language_ _Processing,_ 2020.

Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer:
A memory viewpoint, 2023.

Moses Charikar, Weiyun Ma, and Li-Yang Tan. New lower bounds for massively parallel computation from
query complexity, 2020.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos,
Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian
Weller. Rethinking attention with performers, 2022.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated
recurrent neural networks on sequence modeling. _arXiv_ _preprint_ _arXiv:1412.3555,_ 2014.

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning. What does bert look at? an
analysis of bert’s attention. _arXiv_ _preprint_ _arXiv:1906.04341,_ 2019.

Sam Coy and Artur Czumaj. Deterministic massively parallel connectivity. In _Proceedings_ _of_ _the_ _54th_ _Annual_
_ACM_ _SIGACT_ _Symposium_ _on_ _Theory_ _of_ _Computing,_ STOC 2022, page 162–175, New York, NY, USA,
2022. Association for Computing Machinery. ISBN 9781450392648. doi: 10.1145/3519935.3520055. URL
```
 https://doi.org/10.1145/3519935.3520055.

```
Amit Daniely. Depth separation for neural networks. In Satyen Kale and Ohad Shamir, editors, _Proceedings_
_of_ _the_ _2017_ _Conference_ _on_ _Learning_ _Theory,_ volume 65 of _Proceedings_ _of_ _Machine_ _Learning_ _Research,_ pages
690–696. PMLR, 07–10 Jul 2017. URL `https://proceedings.mlr.press/v65/daniely17a.html.`

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. In _NeurIPS,_ 2022.

Jeffrey Dean and Sanjay Ghemawat. Mapreduce: Simplified data processing on large clusters. In _OSDI,_
pages 137–150, 2004.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication
for transformers at scale. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems,_ volume 35, 2022.

Pavol Duris, Zvi Galil, and Georg Schnitger. Lower bounds on communication complexity. In _Proceedings_ _of_
_the_ _Sixteenth_ _Annual_ _ACM_ _Symposium_ _on_ _Theory_ _of_ _Computing,_ page 81–91, 1984.

Ronen Eldan and Ohad Shamir. The power of depth for feedforward neural networks. In Vitaly Feldman,
Alexander Rakhlin, and Ohad Shamir, editors, _29th_ _Annual_ _Conference_ _on_ _Learning_ _Theory,_ volume 49 of
_Proceedings_ _of_ _Machine_ _Learning_ _Research,_ pages 907–940, Columbia University, New York, New York,
USA, 23–26 Jun 2016. PMLR. URL `https://proceedings.mlr.press/v49/eldan16.html.`

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell,
Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds,
Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown,
Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer
circuits. _Transformer_ _Circuits_ _Thread,_ 2021. https://transformer-circuits.pub/2021/framework/index.html.

12


-----

Mohsen Ghaffari, Fabian Kuhn, and Jara Uitto. Conditional hardness results for massively parallel computation
from distributed lower bounds. In _IEEE_ _60th_ _Annual_ _Symposium_ _on_ _Foundations_ _of_ _Computer_ _Science,_
pages 1650–1663, 11 2019. doi: 10.1109/FOCS.2019.00097.

Michael T Goodrich, Nodari Sitchinava, and Qin Zhang. Sorting, searching, and simulation in the mapreduce
framework. In _International_ _Symposium_ _on_ _Algorithms_ _and_ _Computation,_ pages 374–383. Springer, 2011.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023.

Sudipto Guha and Andrew McGregor. Stream order and order statistics: Quantile estimation in randomorder streams. _SIAM_ _Journal_ _on_ _Computing,_ 38(5):2044–2059, 2009. doi: 10.1137/07069328X. URL
```
 https://doi.org/10.1137/07069328X.

```
Michael Hahn. Theoretical limitations of self-attention in neural sequence models. _Trans._ _Assoc._ _Comput._
_Linguistics,_ 8:156–171, 2020. doi: 10.1162/tacl\_{a}{\_{0}{0}{3}}{0}6. URL `https://doi.org/10.`
```
 1162/tacl_a_00306.

```
Yiding Hao, Dana Angluin, and Robert Frank. Formal language recognition by hard attention transformers:
Perspectives from circuit complexity. _Trans._ _Assoc._ _Comput._ _Linguistics,_ 10:800–810, 2022. URL `https:`
```
 //transacl.org/ojs/index.php/tacl/article/view/3765.

```
Sungjin Im, Ravi Kumar, Silvio Lattanzi, Benjamin Moseley, Sergei Vassilvitskii, et al. Massively parallel
computation: Algorithms and applications. _Foundations_ _and_ _Trends®_ _in_ _Optimization,_ 5(4):340–417, 2023.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn
Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure
prediction with alphafold. _Nature,_ 596(7873):583–589, 2021.

Praneeth Kacham, Vahab Mirrokni, and Peilin Zhong. Polysketchformer: Fast transformers via sketches for
polynomial kernels, 2023.

Howard Karloff, Siddharth Suri, and Sergei Vassilvitskii. A model of computation for mapreduce. In
_Twenty-first_ _Annual_ _ACM-SIAM_ _Symposium_ _on_ _Discrete_ _Algorithms,_ pages 938–948, 12 2010. doi: 10.1137/
1.9781611973075.76.

Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, and Seunghoon
Hong. Pure transformers are powerful graph learners, 2022.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014.

Yuxuan Li and James L. McClelland. Systematic generalization and emergent structures in transformers
trained on structured tasks, 2022.

Valerii Likhosherstov, Krzysztof Choromanski, and Adrian Weller. On the expressive power of self-attention
matrices. _arXiv_ _preprint_ _arXiv:2106.03764,_ 2021.

Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn
shortcuts to automata, 2022.

Andreas Loukas. What graph neural networks cannot learn: depth vs width. _arXiv_ _preprint_ _arXiv:1907.03199,_
2019.

Eran Malach. Auto-regressive next-token predictors are universal learners, 2023.

William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers, 2022.

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers.
_Transactions_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics,_ 11:531–545, 2023a. ISSN 2307-387X. doi:
10.1162/tacl_a_00562. URL `http://dx.doi.org/10.1162/tacl_a_00562.`

13


-----

William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought, 2023b.

William Merrill, Ashish Sabharwal, and Noah A. Smith. Saturated transformers are constant-depth threshold
circuits. _Transactions_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics,_ 10:843–856, 2022. ISSN 2307-387X.
doi: 10.1162/tacl_a_00493. URL `http://dx.doi.org/10.1162/tacl_a_00493.`

MPICH. Mpi allreduce, 2023. URL `https://www.mpich.org/static/docs/latest/www3/MPI_Allreduce.`
```
 html.

```
Noam Nisan and Avi Wigderson. Rounds in communication complexity revisited. SIAM Journal on Computing,
22(1):211–219, 1993. doi: 10.1137/0222016. URL `https://doi.org/10.1137/0222016.`

Matanel Oren, Michael Hassid, Yossi Adi, and Roy Schwartz. Transformers are multi-state rnns, 2024.

Christos H. Papadimitriou and Michael Sipser. Communication complexity. In _Proceedings_ _of_ _the_ _Fourteenth_
_Annual_ _ACM_ _Symposium_ _on_ _Theory_ _of_ _Computing,_ page 196–200, 1982.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is turing complete. _Journal_ _of_ _Machine_
_Learning_ _Research,_ 22(1):3463–3497, 2021.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models
are unsupervised multitask learners. _OpenAI_ _blog,_ 1(8):9, 2019.

Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in bertology: What we know about how bert
works. _Transactions_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics,_ 8:842–866, 2021.

Tim Roughgarden, Sergei Vassilvitskii, and Joshua Wang. Shuffles and circuits (on lower bounds for modern
parallel computation). _Journal_ _of_ _the_ _ACM,_ 65:1–24, 11 2018. doi: 10.1145/3232536.

Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers,
2023.

Lena Strobl. Average-hard attention transformers are constant-depth uniform threshold circuits, 2023.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. Transformers as recognizers of
formal languages: A survey on expressivity, 2023.

Matus Telgarsky. Benefits of depth in neural networks. In Vitaly Feldman, Alexander Rakhlin, and Ohad
Shamir, editors, 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning
_Research,_ pages 1517–1539, Columbia University, New York, New York, USA, 23–26 Jun 2016. PMLR.
URL `https://proceedings.mlr.press/v49/telgarsky16.html.`

Mehmet Ozgur Turkoglu, Stefano D’Aronco, Jan Dirk Wegner, and Konrad Schindler. Gating revisited: Deep
multi-layer rnns that can be trained. _IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_ _Intelligence,_ 44
(8):4081–4092, 2021.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser,
and Illia Polosukhin. Attention is all you need. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _30,_
2017.

Ziwei Wang, Changyuan Wang, Xiuwei Xu, Jie Zhou, and Jiwen Lu. Quantformer: Learning extremely
low-precision vision transformers. _IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_ _Intelligence,_ 2022.

Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on approximating turing machines with transformers, 2021.

14


-----

Shunyu Yao, Binghui Peng, Christos H. Papadimitriou, and Karthik Narasimhan. Self-attention networks can
process bounded hierarchical languages. In _Proceedings_ _of_ _the_ _59th_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_
_Computational_ _Linguistics_ _and_ _the_ _11th_ _International_ _Joint_ _Conference_ _on_ _Natural_ _Language_ _Processing,_
2021.

Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. Are transformers
universal approximators of sequence-to-sequence functions? In _International_ _Conference_ _on_ _Learning_
_Representations,_ 2020.

Yi Zhang, Arturs Backurs, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling
transformers with lego: a synthetic reasoning task, 2023.

15


-----

### A Supplemental Preliminaries

#### A.1 Further details about transformers

We discuss a few minor technicalities and modifications of the self-attention unit (Definition 2.1) and
transformer model (Definition 2.2) defined in Section 2.1 that are necessary for readers looking for a
comprehensive understanding of the proofs of our theoretical results.

**Fixed-bit** **precision** **arithmetic.** As discussed in Section 2.1, we assume that all numbers that appear in
the intermediate products and outputs of self-attentions are representable with _p-bit_ precision arithmetic,
where _p = Θ(log N_ ). While the details of fixed-precision arithmetic will be uninteresting to most readers, it
is necessary to explain precisely what we mean in order to ensure that proofs of results like Theorem 3.4 are
sound. Throughout the paper, we allow _p_ to depend on of constants, such as _γ,_ _δ,_ and _ϵ._
Concretely, we assume that all query, key, and value embeddings _Q(X), K(X), V (X)_ evaluated on all
inputs contain scalar values _z_ _∈_ R that are polynomially bounded (i.e. _|z| ≤_ exp(O(p)) = N _[ζ]_ for sufficiently
large constant exponent _ζ_ _> 0)_ and are inverse-polynomially discretized (i.e. _z · N_ _[ζ]_ _∈_ Z). Depending on the
desired exponent _ζ,_ some _p = Θ(log N_ ) can be chosen to guarantee this property. While we do not formally
analyze the precision needed to approximate the particular embeddings employed by our proofs, we note
that our recurring sinusoidal embeddings (e.g. Lemma D.1) can be discretized without losing their central
properties and that discretizations of the restricted isometry embeddings of Proposition B.1 are analyzed by
Sanford et al. (2023).
Rather than stipulating a particular bounded-precision implementation that computes the output of a selfattention unit must be implemented, we specify a rounding constraint that any computational implementation
of a self-attention unit must satisfy. Precisely, we require that any output round to the same inverse-polynomial
discretization as the true mathematical attention.

**Definition** **A.1.** For a self-attention unit _f_ _∈_ Attn[N]m[,] [let] _[f][ˆ]_ [be] [an] [finite-precision] [implementation] [of] [that] [unit.]
We say that _f[ˆ]_ is a _valid_ _implementation_ if


sup
_X∈R[N]_ _[×][m]_


� 1
_f_ (X) − _fˆ(X)_
��� ���∞ [=][ O] 2[p]


�
_._


This definition is only to establishing the fact that self-attention units with sufficient margins can precisely
compute hardmax outputs in Lemma A.2 and to showing that MPC models can indeed compute the outputs
precisely in Theorem 3.4.

**Hardmax** **attention.** While we exclusively consider attention units with the softmax, our constructions
periodically rely on the exact computation of averages of embeddings. We define the _hardmax_ operator to
allow the consideration of discrete averaging operations. For some _v_ _∈_ R[N], let


hardmax(X)i =


� _|Imax1_ (v)| _[,]_ if _i ∈_ _Imax(v)_

0 otherwise,


where _Imax(v) = {i ∈_ [N ] : vi = maxi′ vi′}.
We show that bounded-precision softmax self-attention units that satisfy a margin property can be
modified slightly to have identical outputs to an analogous hardmax unit.

**Lemma** **A.2.** _Let_ _f_ _∈_ Attn[N]m _[be]_ _[a]_ _[self-attention]_ _[unit]_ _[with]_ _[precision]_ _[p][ = Θ(][log][ N]_ [)] _[and]_ _[embedding]_ _[functions]_
_Q, K, V_ _such_ _that_ _for_ _some_ _fixed_ 1 ≥ _ξ_ = N _[−][O][(1)]_ _and_ _every_ _X_ _∈_ R[N] _[×][m]_ _and_ _i ∈_ [N ]:

_A(X)i,i′_ _≤_ max _[−]_ _[ξ,]_ _[∀][i][′]_ _[̸∈]_ _[I][max][(][A][(][X][)][i][)][,]_
_i[′′]_ _[A][(][X][)][i,i][′′]_

_where_ _A(X) = Q(X)K(X)[T]._ _Then_ _there_ _exists_ _a_ _self-attention_ _unit_ _f_ _[′]_ _∈_ Attn[N]m _[with]_ _[a]_ _[valid]_ _[p][′][-bit]_ _[implemen-]_
_tation_ _with_ _p[′]_ = O(p) _satisfying_
_f_ _[′](X) = hardmax(A(X))V (X)._

The proof of Lemma A.2 is provided in Appendix F.

16


-----

**Start** **tokens.** Our technical proofs are occasionally simplified by including a “dummy token” whose value
is passed in self-attention layers as a default or null value. For example, in the proof of Lemma D.2, the
dummy token handles the case where the reference token does not appear previously in the sequence. While
we believe that this extra token is not necessary for our technical arguments, we include it for the sake of
simplicity.
We model this dummy token as a _start-of-sequence_ token _X0._ Concretely, if we employ _X0_ in a selfattention f _∈_ Attn[N]m [which] [takes] [as] [input][ X][,] [we] [instead] [treat][ f] [as] [an] [attention] [unit] [in][ Attn][N]m[+1] that operates
on (X0, X1, . . ., XN ). We assume that _X0_ is constant-valued, and therefore never both to pay attention to its
outputs; it’s only relevance is via its key and value embeddings _K0(X0), V0(X0) ∈_ R[m]. If _X0_ is unmentioned,
we assume that it does not exist, or is set such that its key embedding inner products are all zero.

**Supplemental** **chain-of-thought** **tokens.** We periodically (see Theorem B.3 and the proofs of Corollaries 3.5 and 4.3) consider transformers with supplemental blank “chain-of-thought” tokens appended to the
end of the sequence. Unlike the start token, these are only constant _at_ _initialization_ and may be used deeper
in the model to perform meaningful computations.
Let Transformer[N,M]m,L,H,din,dout [denote transformers with][ M][ −] _[N]_ [extra blank elements appended to the input]
sequence. Concretely, we represent _T_ _∈_ Transformer[N,M]m,L,H,din,dout [as] [some] _[T][ ′]_ _[∈]_ [Transformer][M]m,L,H,din,dout [and]
define the output _T_ (X) for _X_ _∈_ R[N] _[×][d][in]_ by letting _Y_ _∈_ R[M] _[×][d][in]_ for _Y1:N_ = X and _YN_ +1:M = _[⃗]0,_ and letting
_T_ (X) = T _[′](Y )._

### B Proofs from Section 3.1

#### B.1 Proof of Lemma 3.2

**Lemma** **3.2.** _For_ _any_ _β, s, N_ _∈_ N, _there_ _exists_ _a_ _transformer_ routeβ,s _∈_ Transformer[N]m,1,1 _[with]_ _[m]_ [=]
_O(s[4]β log N_ ) _satisfying_ routeβ,s(Sent) = Rcvd _for_ _any_ _valid_ (β, s)-routing (Sent, Rcvd).

The proof relies on a _sparse_ _propagation_ sequential primitive, which complements the sparse averaging
primitive of Sanford et al. (2023). For any _Q_ _≤_ _d, N_, on input _X_ = (X1, . . ., XN ) _∈_ R[N] _[×][d]_ with _Xi_ =
(zi, Si) ∈ R[d][−][Q] _× [N_ ][Q] and _bi_ = |{Sj _∋_ _i : j_ _∈_ [N ]}| ≤ _Q,_ we define


sparsePropagateQ,d(X)i =


� _b1i_ �Sj _∋i_ _[z][j]_ if _bi_ _> 0,_

0 otherwise.


Closely following the argument of Sanford et al. (2023), we show in Proposition B.1 that there is a selfattention unit with embedding dimension _m = max(d, O(q log N_ )) that computes sparsePropagateQ,d. This
construction is a key component of the single-layer transformer used in the proof of Lemma 3.2.

**Proposition** **B.1.** _For_ _any_ _b ≤_ _N_ _and_ _d,_ _there_ _exists_ _a_ _self-attention_ _unit_ sparsePropagateQ,d _∈_ Attn[N]m,p _[for]_
_m = d+O(Q log N_ ) and p = O(log N ), which, given any input X _with Xi_ = (zi, Si,[⃗]0) ∈ R[d]×�≤[NQ]�×{0}[m][−][Q][−][d]

_such_ _that_ _bi_ = |{Sj _∋_ _i : j_ _∈_ [N ]}| ≤ _Q_ _for_ _all_ _i,_ _has_ _output_ sparsePropagateQ,d(X) _satisfying_


sparsePropagateQ,d(X)i = _b[1]i_

The proof of Proposition B.1 appears in Appendix F.


� _zj._

_Sj_ _∋i_


_Proof_ _of_ _Lemma_ _3.2._ We construct a single-layer single-headed transformer with query, key, and value
embeddings Q, K, V and output MLP ψ. _Q, K, V_ can be decomposed as Q = Q[′] _◦_ _ϕ,_ _K_ = K _[′]_ _◦_ _ϕ,_ _V_ = V _[′]_ _◦_ _ϕ,_
for some input MLP ϕ and embeddings Q[′], K _[′], V_ _[′]._ We fix Q[′], K _[′], V_ _[′]_ to be the respective embeddings of the selfattention unit with embedding dimension m from Proposition B.1 that computes Y = sparsePropagates,m(X)
for _XSrc_ = (zSrc, SSrc) for every `Src` _∈_ [N ] to be determined. Hence, the proof entails designing elementwise encoders _ϕ_ = (ϕ1, . . ., ϕN ) and decoders _ψ_ = (ψ1, . . ., ψN ) that compute `Rcvd` from `Sent,` using

17


-----

_ϕ_


sparsePropagate _ψ_

(z, S) _Y_ `Rcvd`

```
Sent

```

Machine 2

2Y2 = 2 · sparsePropagate((z, S))2

= z1 + z3

2¯α 2Src 2Dest 2Msg

1 1 0 `hey`

1 3 2 `yes`

2 `asd`

_ψ_


_Y1_ = (0S1 = (0, 2) _, 2)_

_z1_

```
Sent1

```

Machine 1

_α˜_

1

0

_ϕ_ 1

|Dest|Msg|
|---|---|
|0|`hey`|
|2|`yo`|

|α˜|Src<br>g|Dest<br>g|Msg<br>g|
|---|---|---|---|
|1|1|0|`hey`|
|0||||
|1|1|2|`yo`|
|0||||
|1|1|0|`hey`|
|0||||
|1|1|2|yo|

```
Rcvd2

```

Machine 3

_α˜_

0
```
 Sent3

```
1
```
Dest Msg

```
_ϕ_ 1


_S3_ = (2, 4)


_z3_

|Src|Msg|
|---|---|
|0|`yo`|
|2|`yes`|

|2α¯|2Src|2Dest|2Msg|
|---|---|---|---|
|1|1|0|`hey`|
|1|3|2|`yes`|
|2|||`asd`|
|1|3|4|`no`|
|2|||`wyz`|
|0||||
|1|1|2|`yo`|

|Dest|Msg|
|---|---|
|2|`yes`|
|4|`no`|

|α˜|Src<br>g|Dest<br>g|Msg<br>g|
|---|---|---|---|
|0||||
|1|3|2|`yes`|
|1|3|2|`yes`|
|1|3|4|`no`|
|1|3|4|`no`|
|0||||
|0||||


Figure 3: A visualization of the construction used to prove Lemma 3.2 in three phases—the encoding of
each input `SentSrc` as embedding _zSrc_ and subset _SSrc_ with _ϕ;_ the combination of those embeddings into
_YDest_ via the simulation of sparsePropagates,m((z, S)); and the decoding of each _YDest_ into output `RcvdDest`
with _ψ._ The figure provides an example of the encoding and decoding where machines 1 and 3 transmit
messages to machine 2. “Multiple hashing” is used to compute z1 and z3 by encoding each message in multiple
fixed-location “packets” in embedding space space. This redundancy ensures the possibility of machine 2
decoding `Rcvd2` from _Y2,_ due to each message occurring alone at least once in the encoding.

sparsePropagates,m as an intermediate step. A high-level overview of the proof construction is visualized in
Figure 3.
On input SentSrc, we use the encodings QSrc, KSrc, VSrc to specify that all tokens Dest with Dest ∈ `SentSrc`
(or equivalently, all `Dest` with `Src ∈` `RcvdDest)` should receive a copy of the encoding of `SentSrc.` That is, we
set _SSrc_ := {Dest ∈ `SentSrc}` for each `Src ∈` [N ]. This ensures that _Y_ satisfies


1
_YDest_ = _|RcvdDest|_


� _zSrc._
```
Src∈RcvdDest

```

While it’s tempting to simply set each _zSrc_ _∈_ R[m] equal to a (βs)-dimensional vectorization of `SentSrc,` it
is unclear how to extract `RcvdDest` from each _YDest,_ since each average performed by sparsePropagates,m will
combine multiple vector embeddings in a shared space. In order to avoid these troubles, we employ a multiple
_hasing-based_ _encoding_ that treats messages as “packets” identified by a message, a source, a destination, and
a “validity token” that can be used to determine whether a message is uncorrupted. We include multiple

18


-----

copies of each packet in the encoding _zSrc._ For notational ease, we represent each _zSrc_ _∈_ R[m] as a collection of
packets
_zSrc_ = (Msg[�]Src,j, `Src[�]Src,j,` `Dest[�]` `Src,j, αSrc,j)j∈[m′]` _∈_ (Z[β]2[p] _[×][ [][N]_ []][ ×][ [][N] []][ × {][0][,][ 1][}][)][m][′][,]

where _m = m[′](3 + β)._
To sparsely and redundantly encode each `SentSrc` as _zSrc,_ we encode outgoing messages as packets
by utilizing the matrix _A_ guaranteed by the following fact (which we use with _n_ := _N_ [2], _b_ := _s[2],_ and
_m[′]_ := d = O(s[4] log N )).

**Fact** **B.2.** _For_ _any_ _n,_ _b ≤_ _n,_ _and_ _d ≥_ �12b[2] ln n�, _there_ _exists_ _a_ _binary_ _matrix_ _A ∈{0, 1}[n][×][d]_ _such_ _that,_ _for_
_every_ _subset_ _S_ _⊆_ [n] _with_ _|S| ≤_ _b,_ _the_ _columns_ _of_ _the_ _sub-matrix_ _AS_ _∈{0, 1}[|][S][|×][d]_ _contains_ _all_ _S-dimensional_
_elementary_ _vectors,_ _i.e.,_ �e1, . . ., e|S|� _is_ _a_ _subset_ _of_ _the_ _columns_ _of_ _AS._

The proof of Fact B.2 is at the end of the section. We use the following rule to determine which (if any)
message to encode as a packet at each `Src` _∈_ [N ] and _j_ _∈_ [m[′]]. We let _A(Src,Dest),j_ = _AN_ (Src−1)+Dest,j for
notational convenience.


_zSrc,j_ =


(Msg, Src, Dest, 1) if (Msg, Dest) ∈ `SentSrc` and _A(Src,Dest),j_ = 1

and _A(Src,Dest′),j_ = 0, _∀_ `Dest[′]` _∈_ `SentSrc \ {Dest},`

([⃗]0, 0, 0, 0) otherwise.


In Figure 3, this encoding is visualized in the tables of “Machine 1” and “Machine 3,” where the entirety of
each message is encoded in two fixed and distinct locations in the embeddings _z1_ and _z3,_ alongside metadata
about the source of message and the validity _α˜._ Each message is encoded as multiple identical packets in
different embedding dimensions and a large fraction of embedding locations are left blank. These features are
critical for the proper evaluation of the decoding step _ψ._
We analyze the _Y_ = sparsePropagateβ,m(X) outputs, letting

_YDest_ = (YDest,1, . . ., YDest,m′), _YDest,j_ _∈_ (R[β] _× R × R × R)[m][′],_

with all numbers represented with _p-bit_ fixed precision. This analysis shows that there exists an element-wise
decoder MLP _ψ_ satisfying _ψDest(YDest)_ = `RcvdDest` for all `Dest` _∈_ [N ]. For any _j_ _∈_ [m[′]], observe from the
definition of _zSrc_ and sparsePropagates,m that

_YDest,j_ =: �MsgDest,j, SrcDest,j, DestDest,j, ¯αDest,j�

1
= _|RcvdDest|_ `Src∈�RcvdDest` �Msg�Src,j, `Src[�]Src,j,` `Dest[�]` `Src,j, αSrc,j�` _._

Before formally analyzing this construction, we motivate its utility with Figure 3. The encoding 2Y2 of
Machine 2 contains four “clean” rows _j_ with 2α¯2,j = 1, two “corrupted” rows with 2α¯2,j = 2, and one “blank”
row with 2¯α2,j = 0.

 - The **blank** **row** contains no information about any incoming messages, since neither Machine 1 nor
Machine 3 encoded messages as packets in these locations. The fact that 2α¯2,j = 0 certifies the blankness
of this row, and hence, the decoder _ψ_ can ignore it.

 - The **corrupted** **rows** correspond to locations where both Machine 1 and Machine 3 saved messages as
packets. As a result, the corresponding embedding _Y2,j_ = [1]2 [(][z][1][,j] [+][ z][3][,j][)] [is] [an] [average] [of] [two] [non-zero]

embeddings and is hence “corrupted.” Because 2α¯2,j = 2, the decoder _ψ_ detects the corruption and
ignores it when computing `Rcvd2.`

 - The clean **rows are locations where exactly one of Machine 1 and Machine 3 encoded a message.** Hence,
these messages can be cleanly understood by the decoder _ψ,_ which simply validates the “cleanliness” of
the row with 2α¯2,j = 1, determines whether Machine 2 is indeed the target recipient of the respective
message, and saves all such messages in the decoding `Rcvd2.`

19


-----

We prove the validity of this intuition by ensuring that the encoding scheme successfully encodes each
incoming message in a clean row and that the category of each row (blank, corrupted, or clean) can be
detected by the decoder _ψ._ We observe the following sequence of facts about every _YDest._ Let

`RelevantDest` := {(Msg, Src[′], Dest[′]) : Src[′] _∈_ `RcvdDest,` (Msg, Dest[′]) ∈ `SentSrc′}`

denote the set of _all_ messages sent by sources of messages sent to `Dest.`

1. Consider any outgoing message (Msg, Src[′], Dest[′]) ∈ `RelevantDest.` By the property of _A_ guaranteed by
Fact B.2, there exists some j such that A(Src′,Dest′),j = 1 and A(Src′′,Dest′′),j = 0 for every (Src[′′], Dest[′′]) ∈
`RelevantDest` _\_ _{(Src[′], Dest[′])} ._ As a result of the definition of the encoding _z_ and the averaged
representation of _YDest:_

1
_YDest,j_ = (1)
_|RcvdDest|_ [(][Msg][,][ Src][′][,][ Dest][′][,][ 1)][ .]

2. Conversely, if _α¯Dest,j_ = 1/|RcvdDest|, then there exists a unique (Msg, Src[′], Dest[′]) ∈ `RelevantDest` such
that (1) is satisfied.

3. If at least one message is received, then the minimal nonzero value of _α¯Dest_ is 1/|RcvdDest|.

We design _ψDest_ to uniquely identify `RcvdDest` from _YDest_ as follows. If at least one message is received,
then 1/|RcvdDest| can be identified by finding the smallest nonzero value of _α¯Dest._ The decoder _ψ_ inspects
every _YDest,j_ satisfying _α¯Dest,j_ = 1/|RcvdDest|, which therefore satisfies

_|RcvdDest| · (MsgDest,j, SrcDest,j, DestDest,j) ∈_ `RelevantDest.`

Thus, if _|RcvdDest| · DestDest,j_ = Dest, then _|RcvdDest| · (MsgDest,j, SrcDest,j) ∈_ `RcvdDest,` and _ψ_ encodes it as
such.

**Fact** **B.2.** _For_ _any_ _n,_ _b ≤_ _n,_ _and_ _d ≥_ �12b[2] ln n�, _there_ _exists_ _a_ _binary_ _matrix_ _A ∈{0, 1}[n][×][d]_ _such_ _that,_ _for_
_every_ _subset_ _S_ _⊆_ [n] _with_ _|S| ≤_ _b,_ _the_ _columns_ _of_ _the_ _sub-matrix_ _AS_ _∈{0, 1}[|][S][|×][d]_ _contains_ _all_ _S-dimensional_
_elementary_ _vectors,_ _i.e.,_ �e1, . . ., e|S|� _is_ _a_ _subset_ _of_ _the_ _columns_ _of_ _AS._

_Proof._ Let col(A) denote the set of columns of _A._ We use the probabilistic method and consider _A_ with iid
entries _Ai,j_ _∼_ Bernoulli( _b+11_ [).] [We] [bound] [the] [probability] [of] [failure:]


� � [n]
Pr _∃S_ _∈_
_≤_ _b_


� �
s.t. �e1, . . ., e|S|� _̸⊂_ col(AS) _≤_ _b · n[b]_ Pr [ei _̸∈_ col(AS)]

� 1 � 1
_≤_ _n[b][+1]_ 1 − 1 −

_b + 1_ _[·]_ _b + 1_


_b[�][d]_
�


� 1
_≤_ _n[b][+1]_ 1 −
_e(b + 1)_


�d


� _d_
_≤_ _n[b][+1]_ _· exp_ _−_
_e(b + 1)_


�


� _d_
_< exp_ (b + 1) ln n −
3(b + 1)

Therefore, there exists a matrix _A_ with the claimed property.

20


�
_≤_ 1.


-----

Attention head
Attention head

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|||||
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_||||
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|_V_ (_·_)|_V_ (_·_)|_V_ (_·_)|
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_|Sparse propagation<br>`Local`<br>Softmax attention<br>_Q_(_·_)<br>_K_(_·_)_⊤_||||
|<br>Sparse propagation<br>`Local`<br>Softmax attention<br>Attention head<br>`Local`<br>_Q_(_·_)<br>_K_(_·_)_⊤_<br>_V_ (_·_)<br>Multiple<br>hashing|||||||
||||||||
||||||||


Figure 4: To simulate MPC, the local computation within each machine is pushed inside Q(·), K(·), V (·), and
then the pairwise attention matrix performs message routing. To ensure proper routing and also that the
outputs of _Q(·), K(·), V (·)_ are all tall-and-skinny matrices, the construction carefully utilizes both multiple
hashing and sparse propagation.

#### B.2 Proof of Theorem 3.1

We give a generalization of Theorem 3.1 that simulates a broader family of MPC protocol, including those
with more than _n_ machines (i.e. _γ_ _≥_ _δ)._ We accommodate this generalization by simulating MPC protocols
with the generalized transformer family Transformer[N,M]m,L,H [detailed] [in] [Appendix] [A] [with] [supplemental] [blank]
“chain-of-thought” tokens.

**Theorem** **B.3** (Generalization of Theorem 3.1). _For_ _constant_ _γ, δ_ _>_ 0 _and_ _any_ _potentially_ _randomized_
_R-round (γ, δ)-MPC protocol π_ _on nin_ _input words and nout_ _≤_ _nin_ _output words,_ _there exists a transformer T_ _∈_
Transformer[N,M]m,L,H _[with]_ _[N]_ [=][ n][in][, M] [=][ max][(][n][in][, O][(][n][1+]in _[γ][−][δ])), m = O(n[4]in[δ]_ [log][ n][in][)][, L][ =][ R] [+1][, H] [=][ O][(][log log][ n][in][)]
_such_ _that_
_T_ (Input):nout = π(Input).

Theorem 3.1 is an immediate consequence of Theorem B.3 by noting that _M_ = N for sufficiently large
_nin_ when _γ_ _< δ._ Its central construction is summarized in Figure 4.

_Proof._ Consider any MPC protocol _π_ with _q_ = _O(n[1+]in_ _[γ][−][δ])_ machines and _s_ = _O(n[δ]in[)]_ [local] [memory]
that, following the notation of Definition 2.3, maps `Input` _∈_ Z[n]2[p][in] to `Output` _∈_ Z[n]2[p][out] with intermediates `MachineIn[(1)], . . . MachineIn[(][R][)]` and `MachineOut[(1)], . . ., MachineOut[(][R][)]` and deterministic functions
(Localr,i)r∈[R],i∈[q] with

`MachineOut[(]i[r][)]` = Localr,i(MachineIn[(]i[r][)][)][.]

To simulate the protocol, we let every machine i ∈ [q] correspond to a particular position in the transformer’s
context. A transformer that simulates π can then be constructed that consolidates Input onto ⌈nin/s⌉ machines
to match MachineIn[(1)]; computes MachineIn[(][r][+1)] from MachineIn[(][r][)] for each r = 1, . . ., R _−_ 1; and computes
and properly distributes `Output` from `MachineIn[(][r][)].` These three elements of the construction exist due to
the following lemmas, which are proved later.

21


-----

**Lemma** **B.4.** _For_ _any_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _with_ _nin-word_ _inputs,_ _there_
_exists_ _a_ _transformer_ init ∈ Transformer[n]s,[in]1,[,]1[max(],din,d[n]out[in][,q][)] _with_ _din_ = 1 _and_ _dout_ = s, _which,_ _given_ `Input ∈` Z[n]2[p] _[,]_ _[has]_
_output_ _satisfying_ init(Input) = MachineIn[(1)].

**Lemma** **B.5.** _For_ _any_ _R-round_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _and_ _any_ _r_ _∈_ [R − 1],
_there_ _exists_ _a_ _transformer_ round[(][r][)] _∈_ Transformer[q]m,1,H,din,dout _[with]_ _[H]_ [=] _[O][(][log log][ q][)][,]_ _[m]_ [=] _[O][(][s][4][ log][ q][)][,]_ _[and]_
_din_ = dout = s _which,_ _given_ _any_ _valid_ _input_ _X_ = MachineIn[(][r][)] _∈_ Z[q]2[×][p] _[m]_ _under_ _the_ _MPC_ _protocol_ _in_ _vectorized_
_form,_ _has_ _output_ _satisfying_ round[(][r][)](X) = MachineIn[(][r][+1)].

**Lemma** **B.6.** _For_ _any_ _R-round_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _with_ _nout-word_ _output,_
_there_ _exists_ _a_ _transformer_ final _∈_ Transformers,[q,]1[max(],1,din[n][out],dout[,q][)] _for_ _din_ = _s_ _and_ _dout_ = 1, _which,_ _given_ _input_
_X_ = MachineIn[(][R][)], _has_ _output_ final(X) _with_ final(X)i,1 = Outputi _∈_ Z2p.

The proof immediate from the three lemmas. We construct the final transformer _T_ by stacking the
single-layer constructions as a single transformer with embedding dimension _m:_

_T_ = final ◦ round[(][R][−][1)] _◦· · · ◦_ round[(1)] _◦_ init.

The proofs of Lemmas B.4 and B.6 rely on simple constructions with fixed attention matrices and appear
in Appendix F. The proof of Lemma B.5 relies on Lemma 3.2 and is proved in the following section.

**Proof** **of** round[(][r][)] **construction.** To prove the existence single-layer transformer that simulates round[(][r][)],
we separate the computational task into two steps: (i) obtaining `MachineOut[(][r][)]` from `MachineIn[(][r][)]` and
(ii) obtaining `MachineIn[(][r][+1)]` from `MachineOut[(][r][)].` Because the former requires no communication between
machines, we can encode that conversion in the input MLP to the transformer.
The nontrivial part of the reduction is thus the latter step, which we obtain by utilizing multiple singleheaded attention units routeβ,s of Lemma 3.2 to route messages of different sizes to their recipients. The
difficulty in this task is the mismatch in functionality between the two computational models: while the MPC
model ensures that each recipient automatically receives its intended messages, transformers must implement
this functionality manually, while ensuring that multiple messages do not overwrite one another.
The following lemma implements that routing functionality for all messages, using different attention
heads depending on the size of the message. We prove Lemma B.5 at the end of the section as a simple
modification of Lemma B.7.

**Lemma** **B.7.** _For_ _any_ _R-round_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _and_ _any_ _r_ _∈_ [R − 1],
_there_ _exists_ _a_ _transformer_ route[(][r][)] _∈_ Transformer[q]m,1,H _[with]_ _[H]_ [=] _[O][(][log log][ q][)]_ _[and]_ _[m]_ [=] _[O][(][s][4][ log][ q][)][,]_ _[which,]_
_given_ _any_ _valid_ _input_ _X_ = `MachineOut[(][r][)]` _∈_ Z[q]2[×][p] _[m]_ _under_ _the_ _MPC_ _protocol_ _in_ _vectorized_ _form,_ _has_ _output_
_satisfying_ route[(][r][)](X) = MachineIn[(][r][+1)].

Because at most _s_ messages can be shared and received by each machine, and each message is of size at
most _s,_ we can prove an single-headed alternative to Lemma B.7 with a somewhat suboptimal dependence on
embedding dimension. By applying by Lemma 3.2 with message size _β_ = s, bounded number of messages _s,_
and context length _N_ = q, there exists a transformer routes,s with _H_ = 1 and _m = O(s[5]_ log q) that computes
`MachineIn[(][r][+1)]` from `MachineOut[(][r][+1)]` by regarding each outgoing message as belonging to Z[s]2[p] [by] [adding]
padding dimensions as needed.
We improve the embedding dimension to m = O(s[4] log q) by running in parallel _O(log log N_ ) transformers
guaranteed by Lemma 3.2 that encode differently sized messages. The number of heads _H_ increases at a
doubly-logarithmic rate because of a doubling trick employed on the size of message encodings used by
constituent part.

22


-----

_Proof._ We describe an implementation of route[(][r][)] by considering any fixed input `MachineOut[(][r][)]` _∈_ Z[q]2[×][p] _[m]._ For
each _i ∈_ [q] and some integer sequence 1 = β0 _< β1_ _< · · · < βH_ = s + 1, we partition `MachineOut[(]i[r][)]` into _H_
disjoint subsets as follows. For any _h ∈_ [H], let

� �
`Sent[h]i` [:=] (Msg, Dest) ∈ `MachineOut[(]i[r][)]` : dim(Msg) ∈ [βh−1, βh] _,_

� �
`Rcvd[h]i` [:=] (Msg, Src) ∈ `MachineIn[(]i[r][+1)]` : dim(Msg) ∈ [βh−1, βh] _,_

and note that `MachineOut[(]i[r][)]` = [�][˙] _[H]h=1[Sent]i[h]_ [and] `[MachineIn][(]i[r][+1)]` = [�][˙] _[H]h=1[Rcvd]i[h][.]_
For each _h_ _∈_ [H], note that dim(Msg) _≤_ _βh,_ and ��Senthi �� = ��Rcvdhi �� _≤_ _s/βh−1._ As a result, Lemma 3.2
guarantees the existence of a single-headed transformer route[(]h[r][)] such that route[(]h[r][)][(][Sent][h][)] [=] `[Rcvd][h][)]` [with]
embedding dimension _mh_ _≤_ _Cs[4]βh log(q)/βh[4]−1_ [for] [some] [sufficiently] [large] [universal] [constant] _[C][.]_

We defined route[(][r][)] as the computation of route[(]1[r][)][, . . .,][ route]H[(][r][)] [as] _[H]_ [parallel] [heads] [of] [self-attention] [with]
disjoint embeddings concatenated into in _m-dimensional_ embedding space with _m =_ [�]h[H]=1 _[m][h][.]_ [We] [conclude]
by letting

�1 if _h = 0,_

_βh_ =

min(2βh[3]−1[, q][ + 1)] if _h ∈_ [H],


noting that _βH_ = q + 1 for _H_ = O(log log q), and bounding _m:_


_H_
�

_h=1_


_Cs[4]_ log(q)βh _≤_ 2Cs[4] log(q) ·

_βh[4]−1_


_m ≤_


_H_
�

_h=1_


1
_βh−1_


_≤_ 2Cs[4] log(q) ·


_H_
�

_h=1_


1
2[h][−][1] [=][ O][(][s][4][ log][ q][)][.]


_Proof_ _of_ _Lemma_ _B.5._ To simulate a round of MPC protocol π by mapping MachineIn[(][r][)] and ρr to MachineIn[(][r][+1)],
the single-layer transformer round[(][r][)] first computes `MachineOut[(][r][)]` element-wise and then properly routes
messages in MachineOut[(][r][)] to their proper destination. We can define round[(][r][)] = route[(][r][)] _◦_ `Localr` for route[(][r][)]

in Lemma B.7 and `Localr,i(MachineIn[(]i[r][)][, ρ][r,i][) =][ MachineOut]i[(][r][)][.]` [This] [can] [be] [immediately] [constructed] [as] [a]
single-layer transformer by prepending the embeddings _Q, K, V_ of the construction of route[(][r][)] with `Localr,`
using _Q ◦_ `Localr,` _K ◦_ `Localr,` _V_ _◦_ `Localr` as the embeddings of round[(][r][)].

#### B.3 Additional graph problems solvable by log-depth transformers

Theorem 8.1 and Corollary 8.2 of Coy and Czumaj (2022) give efficient MPC protocols for other graph problems
besides connectivity, and therefore, as corollaries of Theorem 3.1, we also obtain log-depth transformers for
these problems.

**Corollary** **B.8** (Spanning forest construction). _For_ _any_ _constant_ _ϵ_ _∈_ (0, 1) _and_ _any_ _D_ _≤_ _N_ _,_ _there_ _exists_
_a_ _transformer_ _in_ Transformer[N]m,L,H _[with]_ _[m][ =][ O][(][N][ ϵ][)][,]_ _[H]_ [=][ O][(][log log][ N] [)][,] _[and]_ _[L][ =][ O][(][log][ D][)]_ _[that]_ _[computes]_ _[a]_
_rooted_ _spanning_ _forest_ _of_ _any_ _input_ _graph_ _G = (V, E)_ _with_ _|V |, |E| = O(N_ ) _where_ _each_ _connected_ _component_
_has_ _diameter_ _at_ _most_ _D._

**Corollary** **B.9** (Minimum spanning forest construction). _For_ _any_ _constant_ _ϵ ∈_ (0, 1) _and_ _any_ _DMSF_ _≤_ _N_ _,_
_there_ _exists_ _a_ _transformer_ _in_ Transformer[N]m,L,H _[with]_ _[m][ =][ O][(][N][ ϵ][)][,]_ _[H]_ [=][ O][(][log log][ N] [)][,] _[and]_ _[L][ =][ O][(][log][ D][MSF]_ [)]
_that_ _identifies_ _the_ _connected_ _components_ _of_ _any_ _input_ _graph_ _G = (V, E)_ _with_ _|V |, |E| = O(N_ ) _and_ poly(N )_bounded_ _integer_ _weights_ _whose_ _minimum_ _spanning_ _forest_ _has_ _diameter_ _at_ _most_ _DMSF ._

23


-----

|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||_Q⊤_<br>_h,iKh,i′_||
||||Inner product machines|


Figure 5: This construction employs _M_ [2] _inner_ _product_ _machines_ to compute the entries of the softmax
matrix, and _M_ _token_ _machines_ to compute all values of _Q(·), K(·), V (·)._ What is most complex about the
construction are the additional machines and message routing needed to propagate these values efficiently
between the inner product machines and the token machines, in particular carefully aggregating the output of
the attention mechanism and computing its normalization. To this end, the protocol uses additional machines,
organized into a tree with branching factor _b = O(N_ _[δ][′][−][δ])_ and depth _D_ = O( _δ[1+][′]−[α]δ_ [).]

### C Proofs from Section 3.2

#### C.1 Proof of Theorem 3.4

As in Appendix B.2, we give and prove a generalized version of Theorem 3.4 that broadens the family of
considered transformers to include masked models and those that contain extra blank chain-of-thought tokens,
using notation from Appendix A.

**Theorem C.1 (Generalization of Theorem 3.4).** _For any transformer T_ _∈_ Transformer[N,M]m,L,H _[(or][ MaskTransformer]m,L,H[N,M]_ _[)]_
_with_ _mH_ = _O(N_ _[δ])_ _for_ _δ_ _∈_ (0, 1) _and_ _M_ = Θ(N [1+][α]) _for_ _α_ _≥_ 0 _and_ _for_ _any_ _δ[′]_ _∈_ (δ, 1), _there_ _exists_ _an_
_O(_ _[L]δ[(1+][′]−δ[α][)]_ [)][-round] [(1 + 2][α][ +][ δ][′][, δ][′][)][-MPC] _[protocol]_ _[with]_ _[q]_ [=][ O][(][M][ 2][)] _[machines]_ _[with]_ _[s][ =][ O][(][N][ δ][′][)]_ _[local]_ _[memory]_

_that_ _outputs_ _the_ _same_ _sequence_ _as_ _T_ (X) _for_ _all_ _X_ _∈_ R[N] _._

Theorem 3.4 is an immediate consequence by setting _M_ := N and _α := 0._


_Proof._ It suffices to show that an _O(_ _δ[1+][′]−[α]δ_ [)-round] [MPC] [protocol] _[π]_ [that] [simulates] [a] [single-layer] [transformer]

_T_ _∈_ Transformer[M]m,m,m,1,H [with] _[m][-dimensional]_ [input] [and] [output] [embeddings] [since] [a] [depth-][L] [transformer]
can be constructed by applying _L_ such protocols sequentially. Moreover, we can ignore the difference between
the input context length _N_ and the context length with padding _M_ by assuming that the input contains _M_
tokens.
Concretely, we consider _H_ heads with embeddings (Qh, Kh, Vh)h∈[H], element-wise output MLP _ψ_ =
(ψ1, . . ., ψM ), and any fixed masks Λ1, . . ., ΛH _∈{−∞, 0}[M]_ _[×][M]_ . We show that there exists some _π_ such that

24


-----

for any `Input = X` _∈_ R[M] _[×][m],_

_π(X) = ψ_


�

_X_ +


_H_ �
� softmax(Qh(X)Kh(X)[T] + Λh)Vh(X)

_h=1_


_,_


where numbers in _X_ and all intermediate products of the transformer computation can be represented with
_p = O(log M_ ) bit precision.
Our MPC protocol _π,_ which will use _q_ = O(M [2]) machines and _s = Θ(N_ _[δ][′])_ words of local memory per
machine, assigns each of the _q_ machines to one of four possible roles: token machine, inner product machine,
query propagation machine, and key/value propagation machine. We describe these machines below. For the
sake of readability, we identify machines with easily interpretable descriptions and use the bijection ID to map
each of those to a token in [q] that is used for routing messages. Our protocol has two important parameters:
_b_ = _⌊s/(4mH)⌋_ = _O(N_ _[δ][′][−][δ])_ is the _branching_ _factor_ of the protocol, and _D_ = _⌈logb(M_ )⌉ = _O(_ _δ[1+][′]−[α]δ_ [)] [is] [the]

_depth_ of the protocol.
At a high level (see Figure 5 for a corresponding diagram), the protocol involves computing all intermediate
products of the of a transformer unit by performing MLP computations in _N_ _token_ _machines,_ computing
inner products in _N_ [2] _inner_ _product_ _machines,_ and using _O(N_ [2]) other _propagation_ _machines_ arranged in trees
to share information between the two in _O(D)_ rounds. The protocol draws inspiration from Appendix C.6.1
of Sanford et al. (2023), which uses a similar construction to simulate transformers with Congest protocols
on fixed graphs. It is also similar to the MPC implementation of the MPI AllReduce functionality (MPICH,
2023) described by Agarwal et al. (2014).

 - Machine _i_ _∈_ [M ] is a _token_ _machine_ that performs all element-wise computation on the _ith_ token
embedding, including the computation of (Qh,i(Xi), Kh,i(Xi), Vh,i(Xi))h∈[H] and the final MLP output
_ψi._ Let `ID(i) = i.`

 - Machine (i, i[′]) ∈ [M ][2] is an inner product machine designed to compute the inner products (Qh,i(Xi)[T]Kh,i[′] (Xi[′] ))h∈[H].

 - Machine (Q, i, d, k) for token _i_ _∈_ [M ], depth _d_ _∈_ [D − 1] and position _k_ _∈_ [b[d]] is a _query_ _propagation_
_machine._ This machine is responsible for handling communication of query tokens (Qh,i(Xi))h∈[H] and
of all partially-computed attention outputs for the ith token between token machine i and inner product
machines (i, i[′]) for

_i[′]_ _∈_ `Descendantsd,k` := �b[D][−][d](k − 1), . . ., b[D][−][d]k� _∩_ [M ].

Concretely, if _ℓ_ = 1, then the machine communicates with token machine _i_ and query propagation
machines (Q, i, d + 1, k[′]) for

_k[′]_ _∈_ `Childrenk` := {b(k − 1) + 1, . . ., bk} .

If _ℓ_ = D − 1, then it communicates with inner product machines (i, i[′]) for _i[′]_ _∈_ `Childrenk ∩` [M ] and
query propagation machine (Q, i, d − 1, ⌊k/b⌋). Otherwise, it communicates with query propagation
machines (Q, i, d − 1, Parentk), for `Parentk` := ⌊k/b⌋, and (Q, i, d + 1, k[′]) for _k[′]_ _∈_ `Childrenk.`

 - Machine (KV, i, d, k) is a _key/value_ _propagation_ _machine._ This machine is analogous to a query
propagation machine, except that it is responsible for the communication of key and value tokens (Qh,i(Xi), Vh,i(Xi))h∈[H] between token machine _i_ and inner product machines (i, i[′]) for _i[′]_ _∈_
```
   Descendantsd,k.

```

Since the total number of machines is _q_ = M + M [2] + M [�]d[D]=1[−][1] _[b][d]_ [=][ O][(][M][ 2][),] [we] [conclude] [that] [the] [global]
memory of the protocol is qs = O(N [2+2][α][+][δ][′] ), which means the protocol is (1 + 2α + δ[′], δ[′])-MPC. We simulate
the transformer using a four stage protocol using 2D + 3 = O( _δ[1+][′]−[α]δ_ [)] [rounds] [of] [MPC] [computation.]

25


-----

**Stage** **1:** **Token** **dispersion.** Because the input to an MPC protocol `Input` = _X_ is divided equally
among machines 1, . . ., ⌈MmH/s⌉, the first round of MPC computation routes each input token _Xi_ to its
respective token machine. This is completed by setting (i, Xi) _∈_ `MachineOut[(1)]i[′]` if (i, Xi) _∈_ `MachineIn[(1)]i[′]` [.]
Thus, `MachineIn[(2)]i` = {(Src, Xi)} for all token machines _i ∈_ [M ].

**Stage** **2:** **Embedding** **propagation.** In rounds 2, . . ., D + 1, _π_ computes the respective key, query, and
value embeddings in each token machine and propagate them to respective inner product machines using the
query and key/value propagation machines. Concretely:

 - In round 2, each token machine _i_ (whose memory contains _Xi)_ computes _m-dimensional_ embeddings
embeddings _Qi_ := (Qh,i(Xi))h∈[H], Ki := (Kh,i(Xi))h∈[H], Vi := (Vh,i(Xi))h∈[H]. It transmits each
embedding to the respective depth-1 query and key/value propagation machine nodes, while also
preserving knowledge of its own _Xi._ (In all further rounds, we assume that ((i, Xi)) ∈ `MachineOut[(]i[r][)]`
to ensure that token machine _i_ can compute the skip-level connection at the end.) That is,

`MachineOut[(2)]i` = {(i, Xi)}

_∪{(ID(Q, i, 1, k[′]), Qi) : k[′]_ _∈_ `Children1}`

_∪{(ID(KV, i, 1, k[′]), (Ki, Vi)) : k[′]_ _∈_ `Children1} .`


Note that the total amount of messages sent is _b · mH_ + 2b · mH + m ≤ _s_ and that the only machines
receiving messages are size _m-messages_ by token machines and size _≤_ 4mH messages by query and
key/value propagation machines.

 - In rounds _r_ _∈{3, . . ., D},_ each query and key/value propagation machine of depth _d_ = _r −_ 2 passes
embeddings onto their successors. That is,

`MachineOut[(]ID[r][)](Q,i,d,k)` [=][ {][(][ID][(][Q][, i, d][ + 1][, k][′][)][, Q][i][) :][ k][′] _[∈]_ `[Children][k][}][,]`

`MachineOut[(]ID[r][)](KV,i,d,k)` [=][ {][(][ID][(][KV][, i, d][ + 1][, k][′][)][,][ (][K][i][, V][i][)) :][ k][′] _[∈]_ `[Children][k][}][ .]`

 - In round _D + 1,_ the depth-(D − 1) query and key/value propagation machines pass their embeddings
onto their respective inner product machines. That is,

`MachineOut[(]ID[D]([+1)]Q,i,D−1,k)` [=][ {][(][ID][(][i, k][′][)][, Q][i][) :][ k][′] _[∈]_ `[Children][k][ ∩]` [[][M] []][}][,]

`MachineOut[(]ID[D]([+1)]KV,i,D−1,k)` [=][ {][(][ID][(][k][′][, i][)][,][ (][K][i][, V][i][)) :][ k][′] _[∈]_ _[k][′]_ _[∈]_ `[Children][k][ ∩]` [[][M] []][}][ .]

**Stage** **3:** **Softmax** **computation.** In rounds D +2, . . ., 2D +2, computes each inner product and iteratively
builds up each attention output by accumulating partial softmax computations. For each query propagation
machine (Q, i, d, k) and _h_ _∈_ [H], we let _Si,d,k,h_ and _Zi,d,k,h_ denote its partial normalization and softmax
computations respectively. That is,

_Zi,d,k,h_ = � exp(Qh,i(Xi)[T]Kh,i′(Xi′))1 {Λi,i′ = 0}

_i[′]∈Descendantsd,k_


=


��k[′]∈Childrenk _[Z][i,d][+1][,k][′][,h]_ if _d ≤_ _D −_ 1,

exp(Qh,i(Xi)[T]Kh,k(Xk))1 {Λi,k = 0} if _d = D._


1
_Si,d,k,h_ =
_Zi,d,k,h_


� exp(Qh,i(Xi)[T]Kh,i′(Xi′))Vh,i′(Xi′)1 {Λi,i′ = 0}

_i[′]∈Descendantsd,k_


=


_Zi,d+1,k′_ _,h_

��k[′]∈Childrenk _Zi,d,k,h_ _· Si,d+1,k′,h_ if _d ≤_ _D −_ 1,

_Vh,k(Xk)1 {Λi,k_ = 0} if _d = D;_


Note that _Si,0,1,h_ = (softmax(Qh(X)Kh(X)[T] + Λh)Vh(X))i and let _Si,d,k_ = (Si,d,k,h)h∈[H] _∈_ R[H][×][m] and
_Zi,d,k_ = (Zi,d,k,h)h∈[H] _∈_ R[H]

26


-----

- In round _D + 2,_ each inner product machine computes its respective inner products and passes its
partial softmax computations to its parent query propagation machine. As a result of round _D + 1,_
each inner product machine (i, i[′]) recently received the embeddings necessary to compute the relevant
inner product:

`MachineIn[(]ID[d][+2)](i,i[′])` [=][ {][(][ID][(][Q][, i, D][ −] [1][,][ Parent][i][)][, Q][i][)][,][ (][ID][(][KV][, i][′][, D][ −] [1][,][ Parent][i][′][)][,][ (][K][i][′][, V][i][′][))][}][ .]

It propagates the respective partial computations _Si,D,i′_ and _Zi,D,i′_ as follows:

`MachineOut[(]ID[D]([+2)]i,i[′])` [=][ {][(][ID][(][Q][, i, D][ −] [1][,][ Parent][i][)][,][ (][S][i,D,i][′][, Z][i,D,i][′][))][}][ .]

Note that each depth-(D−1) query propagation machine receives messages of size at most b·(m+1)H _≤_ _s._

- In rounds _r_ _∈{D + 3, . . ., 2D},_ partial softmax computations are received by query propagation
machines of depth _d = 2D + 1 −_ _r,_ added together, and passed along to their parent machines. That is,
given
`MachineIn[(]ID[r][)](Q,i,d,k)` [=][ {][(][ID][(][Q][, i, d][ + 1][, k][′][)][,][ (][S][i,d][+1][,k][′][, Z][i,d][+1][,k][′][)) :][ k][′] _[∈]_ `[Children][k][}][,]`

each respective machine computes _Si,d,k_ and _Zi,d,k_ recursively and propagates

`MachineOut[(]ID[r][)](Q,i,d,k)` [=][ {][(][ID][(][Q][, i, d][ −] [1][,][ Parent][k][)][,][ (][S][i,d,k][, Z][i,d,k][)][}][ .]

- In round 2D + 1, the top-most query propagation tokens pass their partial sums to the token machines:

`MachineOut[(2]ID[D](Q[+1)],i,1,k)` [=][ {][(][i,][ (][S][i,][1][,k][, Z][i,][1][,k][))][}][ .]

- In round 2D + 2, the token machines compute their respective output of the transformer, _T_ (X)i. Given
input
`MachineIn[(2]i` _[D][+2)]_ = {(k[′], (Si,1,k′, Zi,1,k′)) : k[′] _∈_ `Children1} ∪{(i, Xi)},`

the token machine _i_ computes _Si,0,1_ and _Hi,0,1_ and then


_H_
� _Si,0,1,h_

_h=1_


�


_._


_H_ �
� softmax(Qh(X)Kh(X)[T] + Λh)[T]i _[V][h][(][X][)]_

_h=1_


�

_Xi +_


_T_ (X)i = ψi


�

_Xi +_


= ψi


This quantity is used as an intermediate product for the final phase of computation.

**Stage** **4:** **Token** **compression.** We invert Stage 1 by properly compressing the MPC output in the final
round 2D + 3. That is, we let `MachineOuti[(2][D][+2)]` = {(⌊imH/s⌋ + 1, T (X)i)} for each token machine _i ∈_ [M ],
which ensures that the outputs are condensed in the proper order in machines 1, . . ., ⌈MmH/s⌉.

**Precision** **analysis.** In order for the proof to be fully sound, care must be taken to ensure that the
computation of each self-attention output _Si,0,1,h_ is handled with proper numeric precision, as discussed in
Appendix A. We show that each _Si,0,1,h_ is a _valid_ _implementation_ of its corresponding self-attention unit, per
Definition A.1.
To do so, we let _S[ˆ]i,d,k,h_ and _Z[ˆ]i,d,k,h_ denote the _p-bit_ representations of _Si,d,k,h_ and _Zi,d,k,h,_ where scalars
of _S[ˆ]i,d,k,h_ and log(Z[ˆ]i,d,k,h) are represented as discretized rational numbers _z_ satisfying _|z|_ _≤_ [1]2 [2][p/][2] [and]

_z · 2[p/][2]_ _∈_ Z. For some sufficiently small _p[′]_ = Θ(p), we assume that all embeddings _Qh(X), Kh(X), Vh(X)_
have scalars _z_ satisfying _|z| ≤_ [1]2 [2][p][′][/][2] [and] _[z][ ·][ 2][p][′][/][2]_ _[∈]_ [Z][.] [We] [prove] [that] [for] [each] _[h][ ∈]_ [[][H][],]


� 1
_Si,0,1,h −_ _Sˆi,d,k,h_
��� ���∞ [=][ O] 2[p][′]

27


�
_._


-----

Boundedness of intermediate representations is not an issue because

log(Zi,d,k,h) ≤ _O(log(N_ ) + maxi,i[′] _[|][Q][(][X][)]i[T][K][(][X][)][i][′][|][) = exp(][O][(][p][′][))][,]_

and
_∥Si,d,k,h∥∞_ _≤∥V (X)∥∞_ _≤_ 2[p][′][/][2].

It remains to show that that all intermediate representations are sufficiently close to their exact counterparts.
We prove the following via an inductive argument for _d = D, D −_ 1, . . ., 0:

(2b)D−d
���log(Zi,d,k,h) − log( ˆZi,d,k,h)��� _≤_ 2[p/][2] _,_ (2)

_Si,d,k,h −_ _Sˆi,d,k,h_ _._ (3)
��� ���∞ _[≤]_ [2][p][′][/][2]2[(8][p/][b][2][)][D][−][d]


If (3) holds for _d = 0,_ then the claim holds for sufficiently large _p = Θ(p[′])._
For the base case _D,_ we verify (3) by

1
_Si,D,k,h −_ _Sˆi,D,k,h_ _Vh,k(Xk)1 {Λi,k_ = 0} − _Sˆi,D,k,h_
��� ���∞ [=] ��� ���∞ _[≤]_ 2[p/][2][,]

due to the ability to access _Vh,k(Xk)_ and round it directly. We verify (2) due to the immediate access to and
boundedness of _Qh,i(Xi)[T]Kh,k(Xk):_

_|log(Zi,d,k,h)| ≤_ ��Qh,i(Xi)TKh,k(Xk)�� _≤∥Qh,i(Xi)∥2 ∥Kh,k(Xk)∥2_ _≤_ _N_ _· 2p′/2._

We prove the inductive step for d − 1, assuming that the inductive hypothesis holds for d. We first address
_Zˆi,d−1,k,h_ by employing the Lipschitzness of the log-sum-exp function.


log(Zi,d−1,k,h) − log( ˆZi,d−1,k,h) _≤_ 1 log �� exp(log(Zi,d,k′,h))� _−_ log
��� ��� 2[p/][2] [+]

����� _k[′]_

1
_≤_ � log(Zi,d,k′,h) − log( ˆZi,d,k′,h)
2[p/][2] [+] ��� ���

_k[′]_

_≤_ 2[p/]1 [2] [+][ b][ ·] [(2]2[b][)][p/][D][2][−][d] _≤_ [(2][b][)]2[D][p/][−][2][d][+1] _._


�� exp(log( Z[ˆ]i,d,k′,h))

_k[′]_ ������


To obtain (3) for _d −_ 1, we first note that for sufficiently large _p:_

1 − _Zˆi,d,k′,hZi,d−1,k′,h_ = 1 − exp �log � _Zˆi,d,k′,h_ � + log � _Zi,d−1,k,h_
_Zi,d,k,hZ[ˆ]i,d−1,k′,h_ _Zi,d,k′,h_ _Zˆi,d−1,k,h_
����� ����� �����

_≤_ 1 + 2 log _Zˆi,d,k′,h_ + log Zi,d−1,k,h �

_Zi,d,k[′],h_ _Zˆi,d−1,k,h_
������ ����� ����� �����

_≤_ [4][ ·][ (2][b][)][D][−][d][+1] _._

2[p/][2]

28


�������


-----

We conclude by using the fact that each _Si,d−1,k,h_ is a convex combination of other _Si,d,k,h._


1
���Si,d−1,k,h − _Sˆi,d−1,k,h���∞_ _[≤]_ 2[p/][2] [+] �

_k[′]_

1
_≤_ 2[p/][2] [+] �

_k[′]_

1
_≤_ 2[p/][2] [+] �

_k[′]_


_Zi,d,k′,h_ _Si,d,k′,h −_ _Zˆi,d,k′,h_ _Sˆi,d,k′,h_
_Zi,d−1,k[′],h_ _Zˆi,d−1,k′,h_
����� �����∞

_Zi,d,k′,h_ _Si,d,k′,h −_ _Zˆi,d,k′,hZi,d−1,k′,h_ _Sˆi,d,k′,h_
_Zi,d−1,k[′],h_ _Zi,d,k,hZ[ˆ]i,d−1,k′,h_

�����

_Zi,d,k′,h_ _Si,d,k′,h −_ _Sˆi,d,k′,h_
_Zi,d−1,k′,h_ ��� ���∞


�����∞


+ � _Zi,d,k′,h_ _Sˆi,d,k′,h_ 1 − _Zˆi,d,k′,hZi,d−1,k′,h_

_k[′]_ _Zi,d−1,k′,h_ ��� ���∞ ����� _Zi,d,k,hZ[ˆ]i,d−1,k′,h_ �����

1 _Zi,d,k′,h_ _Zˆi,d,k′,hZi,d−1,k′,h_
_≤_ 2[p/][2] [+] [2][p][′][/][2]2[(8][p/][b][2][)][D][−][d] + 2[p][′][/][2][ �]k[′] _Zi,d−1,k′,h_ �����1 − _Zi,d,k,hZ[ˆ]i,d−1,k′,h_

_≤_ 2 · [2][p][′][/][2][(8][b][)][D][−][d] + 2[p][′][/][2] _·_ [4][ ·][ (2][b][)][D][−][d][+1] _≤_ [2][p][′][/][2][(8][b][)][D][−][d][+1] _._

2[p/][2] 2[p/][2] 2[p/][2]


�����


Owing to the fact that _D_ and _p[′]_ are constants and _b = N_ _[O][(1)],_ a sufficiently large choice of _p_ guarantees
that the implementation is valid.

#### C.2 Proof of Corollary 3.5

**Corollary** **3.5.** _Let_ _ϵ ∈_ (0, 1) _be_ _any_ _constant,_ _and_ _let_ _D_ _≥_ _N_ _[ϵ]._ _Assume_ _Conjecture_ _2.4,_ _and_ _suppose_ _there_
_exists_ _T_ _∈_ Transformer[N]m,L,H _[with]_ _[mH]_ [=][ O][(][D][1][−][ϵ][)] _[that]_ _[decides]_ _[connectivity]_ _[of]_ _[any]_ _[input]_ _[graph]_ _[with]_ _[connected]_
_components_ _having_ _diameter_ _≤_ _D._ _Then_ _L = Ω(log D)._

We prove Corollary 3.5 by combining Theorem C.1 and Conjecture 2.4.

_Proof._ Fix any _D_ _≤_ _N_ with _D_ _≥_ _N_ _[ξ]_ for some _ξ_ _∈_ (0, 1]. Let _C1_ denote a cycle graph on _D_ vertices, and let
_C2_ denote the union of two cycle graphs each with _D/2_ vertices.
Suppose there is a transformer T _∈_ Transformer[N]m,L,H [with] _[mH]_ [=][ O][(][D][1][−][ϵ][)] [that] [determines] [the] [connectiv-]
ity of graphs with at most _N_ edges and connected components with diameter at most _D._ We will show that
it can be used to design an Θ(L)-round MPC protocol _π_ that distinguishes graphs _C1_ and _C2_ with _n = D_
edges.
Let _π[′]_ be an MPC protocol that exactly computes the output of _T_ using taking _R = O(L)_ rounds with
local memory _s = O(D[1][−][ϵ/][2])_ and _q_ = O(N [2]) machines, which is guaranteed to exist by Theorem C.1.
Let _n := 2_ � _D4_ � and _k_ := � _Nn_ �. We design _π_ with the same local memory and machine count to determine
the identity of input graph _G = (V, E) ∈{C1, C2}_ provided as an arbitrary sequence of _n_ edges. Let _u ∈_ _V_
be an arbitrary vertex in _G._
Using a constant number of MPC rounds, π converts G into a graph G[′] = (V _[′], E[′]) with |E[′]| = kn_ + _k_ _≤_ _N_
and diameter _n + 2 ≤_ _D_ such that _G[′]_ is connected if and only if G = C1. We do so by letting _G[′]_ be composed
of k copies G[1], . . ., G[k] of G on separate vertices, along with k extra edges connecting the vertex corresponding
to _u_ in each _G[j]_ (say _u[j]_ _∈_ _G[j])_ to _u[1]_ _∈_ _G1._ This ensures that the connectivity correspondence and edge
count diameter bounds are met. Since _G[′]_ can be produced by simply copying edges from _G_ and adding an
additional edge each time an edge containing _u_ is copied, _π_ can produce _G[′]_ in _O(1)_ rounds.
Then, _π_ simulates _π[′]_ on _G[′]_ and returns its output. Since _G[′]_ is connected if and only if _G_ = _C1,_ this
protocol suffices to distinguish _C1_ and _C2._ Because the protocol uses _s_ = _O(n[1][−][ϵ/][2])_ local memory and
_q_ = O(n[2][/ξ]) machines, Conjecture 2.4 implies that π (and hence T ) only exists if L = Ω(log n) = Ω(log N ).

29


-----

### D Proofs from Section 4.1

#### D.1 Proof of Theorem 4.2

**Theorem** **4.2.** _For_ _any_ _k_ _∈_ N _and_ _alphabet_ Σ _with_ _|Σ|_ _≤_ _N_ _,_ _there_ _exists_ _T_ _∈_ MaskTransformer[N]m,L,H _[that]_
_computes_ hopk : Σ[N] _→_ (Σ ∪{⊥})[N] _with_ _m = O(1),_ _L = ⌊log2 k⌋_ + 2, _and_ _H_ = 1.

_Proof._ We design a masked transformer that implements hopk in two phases. The first two layers compute
find[1]X [(][i][)] [for] [each] _[i][ ∈]_ [[][N] []] [using] [a] [similar] [approach] [to] [the] [induction] [heads] [construction] [of] [Bietti] [et] [al.] [(][2023][).]
The subsequent layers employ a doubling trick to compute each find[2]X[ℓ][−][2](i) after _ℓ_ layers.
To do so we employ two technical lemmas (which are proved in Appendix F.4) that describe the
implementation of masked self-attention units that copy .

**Lemma** **D.1.** _For_ _some_ _m ≥_ _d + 2,_ _τ_ : [N ] × R[m] _→_ [N ], _and_ _ρ : R[m]_ _→_ R[d], _there_ _exists_ _an_ _attention_ _head_
lookUpτ,ρ _∈_ MaskAttn[N]m _[with]_ _[precision]_ _[p][ =][ O][(][log][ N]_ [)] _[and]_ _[m][ ≥]_ _[d]_ [+2] _[satisfying]_ [lookUp]τ,ρ[(][X][)][i,][:][d] [=][ ρ][(][X]τ (i,Xi)[)][.]

**Lemma** **D.2.** _For_ _any_ _finite_ _alphabet_ Σ, _m_ _≥_ _d + 2,_ _µ1, µ2_ : R[m] _→_ Σ, _and_ _ρ_ : R[m] _→_ R[d], _there_ _exists_ _an_
_attention_ _head_ lastOccurrenceµ,ρ _∈_ MaskAttn[N]m _[with]_ _[precision]_ _[p][ =][ O][(log(][N][ |][Σ][|][))]_ _[such]_ _[that,]_


lastOccurrence(X)i,:d =


�ρ([⃗]0) _if_ _∀_ _i[′]_ _< i : µ1(Xi′) ̸= µ2(Xi),_
_ρ(Xi′)_ _if_ _i[′]_ = max {i[′] _< i : µ1(Xi′) = µ2(Xi)} ._


The first layer obtains the previous token _Xi−1_ from each _Xi._ This is accomplished via the self-attention
head lookUpτ,ρ with _τ_ (i, Xi) = i − 1 and _ρ(Xi) = Xi._
The second layer retrieves (find[1]X [(][i][)][, X]find[1]X [(][i][)][)] [for] [each] _[i][ ∈]_ [[][N] []] [by] [finding] [the] [most] [recent] [token] [whose]
_preceding_ token is _Xi._ It does so by employing the lastOccurrenceµ1,µ2,ρ primitive on the intermediate state
_Xi[1]_ [= (][X][i][, X][i][−][1][)] [with] _[µ][1][(][X]i[1][) =][ X][i][−][1][,]_ _[µ][2][(][X]i[1][) =][ X][i][,]_ [and] _[ρ][(][X]i[1][) = (][i, X][i][).]_

 - If find[1]X [(][i][)][ >][ 0,] [then] [lastOccurrence][µ]1[,µ]2[,ρ][(][X]i[1][) = (find]X[1] [(][i][)][, X]find[1]X [(][i][)][).]

 - Otherwise, it obtains _[⃗]0_ and performs no further passing, returning _⊥_ after all _L_ layers.

If _k_ = 1, the transformer returns _T_ (X)i = Xfind1X [(][i][)] [= hop][k][(][X][)][i][.]

inductivelyOtherwise,to ensurelet _k_ :=that[�]j[⌊][log]=0the[2][ k]i[⌋]thkj2output[j] for someof thekjℓ∈{th layer0, 1}, _Xandi[ℓ]_ _[∈]let[R][m]k:ℓ[for]=_ _[ℓ][�][≥]j[ℓ]=0[2]_ [contains][k][j][2][j][.] [Construct][an] [encoding][a] [transformer][of]
� �
_Xi, find[2]X[ℓ][−][2](i), Xfind2Xℓ−2_ (i)[,][ find]X[k][:][ℓ][−][2](i), XfindkX:ℓ−2 (i) _._

Note that the base case holds for _ℓ_ = 2, since find[k]X[:0][(0) = find]X[1] [(0)] [if] _[k][0]_ [= 0] [and] [is] _[i]_ [otherwise.]
For each _ℓ_ = 1, . . ., ⌊log2 k⌋ + 1, we assume that the inductive hypothesis holds up to layer _ℓ_ and prove
that it also holds for layer _ℓ_ + 1. To do so, we use a lookUpτ,ρ self-attention head with _τ_ (i, Xi[ℓ][) =][ find]X[2][ℓ][−][2] (i)
and
_ρ(Xi[ℓ][) = (find]X[2][ℓ][−][2](i), Xfind2Xℓ−2_ (i)[,][ find]X[k][:][ℓ][−][2](i), XfindkX:ℓ−2 (i)[)][,]

which ensures that _Xi[ℓ][+1]_ can encode

find[2]X[ℓ][−][1](i) = find[2]X[ℓ][−][2](find[2]X[ℓ][−][2](i))

_Xfind2Xℓ−1_ (i) [=][ X]find[2]X[ℓ][−][2] (find[2]X[ℓ][−][2] (i))


find[k]X[:][ℓ][−][1](i) =

_XfindkX:ℓ−1_ (i) [=]


�
find[k]X[:][ℓ][−][2](find[2]X[ℓ][−][2](i)) if _kℓ−1_ = 1
find[k]X[:][ℓ][−][2](i) if _kℓ−1_ = 0

X _k:ℓ−2_ if _kℓ−1_ = 1
 findX (find[2]X[ℓ][−][2] (i))

_X_ _k:ℓ−2_ if _kℓ−1_ = 0.

 findX (i)

30


-----

As a result, the output of layer _L = ⌊log2 k⌋_ + 2 contains an encoding of

_XfindXk:L−2_ (i) [=][ X][find]X[k] [(][i][)] [= hop][k][(][X][)][i]

for each _i ∈_ [N ]. This is returned as the output of _T_ (X).

#### D.2 Proof of Corollary 4.3

**Corollary** **4.3.** _Assuming_ _Conjecture_ _2.4,_ _for_ _any_ _constants_ _ξ_ _∈_ (0, 1/2] _and_ _ϵ_ _∈_ (0, 1), _and_ _any_ _even_
_k_ = Θ(N _[ξ]),_ _every_ _transformer_ _T_ _∈_ MaskTransformer[N]m,L,H _[with]_ _[mH]_ [=][ O][(][k][1][−][ϵ][)] _[that]_ _[computes]_ [hop]k _[has]_ _[depth]_
_L = Ω(log k)._

_Proof._ The proof is analogous to that of Corollary 3.5. Let _C1_ be a cycle on _k_ vertices, and _C2_ be the
union of two cycles each on _k/2_ vertices. So both _C1_ and _C2_ have _k_ edges. We show that the existence of
_T_ _∈_ Transformer[N]m,L,H [with] _[mH]_ [=][ O][(][k][1][−][ϵ][)] [such] [that] _[T]_ [(][X][) =][ hop]k[(][X][)] [can] [be] [used] [to] [design] [an] [Θ(][L][)-round]
MPC protocol _π_ to solve the task.
As a result of Theorem C.1, there exists an MPC protocol _π[′]_ that exactly computes _T_ with _R = Θ(L)_
rounds with local memory _s = O(D[1][−][ϵ/][2])_ and _q_ = O(N [2]) machines. On input _G = (V, E) ∈{C1, C2},_ we
design a constant-round protocol that computes an sequence X _∈_ Σ[N] such that hopk(X)N exactly determines
the identity of _G._
Since the _k_ edges are passed to _π_ in an unknown ordering with unknown labelings, we let _V_ = [k]
and denote the edges as _e1_ = _{u1, v1}, . . ., ek_ = _{uk, vk}._ We define an operator next over the domain
_{(u, v), (v, u)_ : _{u, v}_ _∈_ _E}_ as follows: for _{u, v}_ _∈_ _E,_ let next(u, v) := (v[′], u) where _v[′]_ _∈_ _V_ is the unique
vertex _v[′]_ _̸=_ _v_ such that _{u, v[′]}_ _∈_ _E._ Notice that next is well-defined because all vertices in a cycle have
degree 2. If _G = C2,_ then next[k/][2](ui, vi) = (ui, vi) for any _i ∈_ [k].
To set up our encoding of G as a sequence X, we first construct a gadget for each edge ei that will be used
to compute a single next(ui, vi). Under the alphabet Σ = [k] ∪{†, ⋆, _}, we define the nine-token sequence

**ei** = _⋆ui_ _†_ _vi_ _ui_ _†_ _vi_ _⋆_ _.

This gadget ensures that two hops will swap the values of _ui_ and _vi._ That is

find[2]ei◦ui[(10) = find]e[1]i◦ui[(6) = 4][,] _Xfind2ei◦ui_ [(10)] [=][ v][i][,]

find[2]ei◦vi[(10) = find]e[1]i◦vi[(8) = 2][,] _Xfind2ei◦vi_ [(10)] [=][ u][i][.]

Likewise, concatenating sequences corresponding to overlapping edges facilitates multiple hops. For example,
if _e1_ = (1, 2), e2 = (3, 4), e3 = (2, 3), then

find[2]e1◦e2◦e3◦2[(28) = 22][,] _Xfind2e1◦e2◦e3◦2[(28)]_ [= 3][,]

find[4]e1◦e2◦e3◦2[(28) = 13][,] _Xfind4e1◦e2◦e3◦2[(28)]_ [= 4][,]

find[4]e1◦e2◦e3◦3[(28) = 2][,] _Xfind4e1◦e2◦e3◦3[(28)]_ [= 1][.]

Let
**E := (e1 ◦** **e2 ◦· · · ◦** **ek)[k/][2]** _◦_ 1

be a length _Nk_ := 9k · _[k]2_ [+ 1] [sequence] [and] [let] _[X]_ [= (][_][)][N] _[−][N][k]_ _[◦]_ **[E][.]** [We] [show] [that] [hop][k][(][X][)][N] [=][ hop][k][(][E][)][N][k] [= 1]

if and only if _G = C2._
Without loss of generality, let _{j, j + 1} = eij_ _∈_ _E_ for all _j_ _∈_ [ _[k]2_ _[−]_ [1].] [Let] _[e][i][0]_ [=][ {][1][, v][∗][}][,] [where] _[v][∗]_ [=] _[k]2_ [if]

_G = C2_ and _v[∗]_ = k if _G = C1._ Assume without loss of generality that _i1_ _> i0._ We argue inductively that for
any _j_ _∈_ [ _[k]2_ []:]

31


-----

1. Every two hops simulates a single step of next:

hop2j(E)Nk = next[j](1, v[∗])1 =


�j if _j + 1 <_ _[k]2_ [or] _[G][ =][ C][1][,]_

1 if _j_ = _[k]2_ _[,]_ _[G][ =][ C][2][;]_


2. Every two hops never “jumps” by more than one repetition of all edges gadgets:

find[2]E[j][(][N][k][)][ ≥] [find]E[2][j][−][2](Nk) − 9(k − 1);

3. The executed gadget corresponds to the correct edge and the gadget is executed correctly:

find[2]E[j][(][N][k][)][ ∈{][9][kj][′][ + 9][i][j] [+][ ι][ :][ j][′] _[∈]_ [N][, ι][ ∈{][2][,][ 4][}}][ .]

If all three conditions are met, then hopk(X)N = 1 if and only if _G = C1_ from condition 1.
We first show that the base case holds for _j_ = 1. Since _i1_ _> i0,_ the second-last time 1 appears in the **E** is
in the final encoding **ei1.** By the two-case analysis of the **ei1** gadget, we validate that hop2(E)Nk = 2 and
conditions (1) and (3) hold. Since **ei1** cannot be the first edge encoding appearing in **e1 ◦** **e2 ◦· · · ◦** **ek,** owing
to it following **ei0),** condition (2) is satisfied.
Suppose that the inductive hypotheses holds up to _j_ _<_ _k2_ [.] [Then,] [we] [argue] [that] [it] [holds] [for] _[j]_ [+ 1.]
Since hop2j(E)Nk = _j_ + 1 (from condition (1)) and find[2]E[j][(][N][k][)] [resides] [at] [the] [left-most] [side] [of] [the] [gadget]
for **eij** (from condition (3)), the two subsequent findE iterations must occur in the gadget **eij+1.** Because
find[2]E[j][(][N][k][)] _[≥]_ [9][k][(][k] _[−]_ _[j][)]_ [(from] [condition] [(2)),] [all] [edges] [appear] [in] [the] _[k]_ [gadgets] [to] [the] [left] [of] [find][2]E[j][(][N][k][),]
and all other edges (including **eij+1)** must occur before the next occurrence of **eij** . Thus, the two hops
occur in the **eij+1** gadget (within distance 9(k − 1)) and results in a properly positioned find[2]E[j][+2](Nk) with
hop2j+2(E)Nk = next[j][+1](1, v[∗])1.
Since an MPC protocol can convert _G_ to _X_ using a constant number of layers, and because _π[′]_ outputs
_T_ (X)N = 1 if and only if _G = C1,_ we can construct a protocol of _π_ by simulating _π[′]._ Because the protocol _π_
uses _s = O(k[1][−][ϵ/][2])_ local memory and _q_ = O(k[2][/ξ]) machines, Conjecture 2.4 implies that the existence of _T_
requires _L = Ω(log k)._

### E Proofs from Section 5

#### E.1 Multi-player pointer chasing communication complexity

We introduce the multi-pass multi-player blackboard communication model studied by Guha and McGregor
(2009) and Assadi and N (2021) to prove lower bounds for multi-pass streaming algorithms. A protocol in this
model specifies how _k_ players, each possessing a portion of a shared input, can jointly compute a function on
the input over the course of _R_ rounds of communication. In each round, all players take turns to broadcast
an _s-bit_ message to all other players. We provide a formal definition of the model as described in Section 6 of
Assadi and N (2021).

**Definition** **E.1.** A _k-player_ _R-round_ _s-space_ _sequential_ _blackboard_ _communication_ _protocol_ includes _k_ players
_P1, . . ., Pk._ On input Z that can be partitioned into (Z1, . . ., Zk), each player Pj is provided with its respective
_Zj._ In each round, players communicate via a shared blackboard. That is, in round r and in order _Pk, . . ., P1,_
each player _Pj_ writes a message Π[r]j _[∈{][0][,][ 1][}][s]_ [on] [the] [blackboard] [(which] [can] [be] [viewed] [by] [all] [players)] [as] [a]
potentially randomized function of input _Zj_ and all information on the blackboard. After the conclusion of _R_
rounds, the final message Π[R]1 [is] [the] [output] [of] [the] [protocol.]

Assadi and N (2021) proves a lower bound on the round complexity necessary to solve the well-studied
_multi-party_ _pointer_ _chasing_ _problem_ of Nisan and Wigderson (1993). We present the problem as defined by
Assadi and N (2021).

32


-----

**Definition** **E.2.** For _q, k_ _∈_ Z+, let an (q, k)-layered _graph_ _G = (V, E)_ have disjoint vertex layers V1, . . ., Vk+1
with _V_ = V1 ∪· · · ∪ _Vk+1_ and each _|Vj| = q_ and edge layers _E1, . . ., Ek_ with _E_ = E1 ∪· · · ∪ _Ek_ and each _Ej_
being a perfect matching between _Vj_ and _Vj+1._ The _pointer_ _chasing_ task is provides a (q, k)-layered graph _G,_
an arbitrary _v_ _∈_ _V1,_ and an arbitrary equipartition _Vk[1]+1_ [and] _[V]k[2]+1_ [of] _[V][k][+1]_ [as] [input] [and] [asks] [whether] _[v]_ [is]
connected to a vertex in _Vk[1]+1_ [or] _[V]k[2]+1[.]_

Assadi and N (2021) give the following lower bound.

**Proposition** **E.3** (Proposition 4.12 of Assadi and N, 2021). _Consider_ _a_ _k-player_ _R-round_ _s-space_ _sequential_
_blackboard_ _protocol_ _that_ _solves_ _the_ (q, k)-pointer _chasing_ _task_ _where_ _each_ _player_ _Pj_ _is_ _provided_ _with_ _the_
_matching_ _Ej_ _and_ _v_ _and_ _Vk[1]+1[, V]k[2]+1_ _[are]_ _[globally]_ _[known.]_ _[Then,]_ _[the]_ _[protocol]_ _[succeeds]_ _[with]_ _[probability]_ _[at]_ _[least]_ [2]3

_only_ _if_ _R ≥_ _k_ _or_ _s = Ω(_ _k[q][5][ )][.]_

All of the lower bounds in Section 5 are most naturally proved by reducing from hopk, rather than
pointer chasing. So we first prove a lower bound for hopk using the lower bound for pointer chasing from
Proposition E.3.

**Proposition** **E.4.** _Consider_ _a_ _k-player_ _R-round_ _s-space_ _sequential_ _blackboard_ _protocol_ _that_ _computes_
hopk(X)N _on_ _any_ _X_ _∈_ Σ[N] _for_ Σ = [2q + 2] _with_ _q_ = � 2Nk � _where_ _each_ _player_ _Pj_ _is_ _provided_ _with_
_X_ _[j]_ := (X2(k−j)q+1, . . ., X2(k−j+1)q), _except_ _for_ _P1,_ _who_ _is_ _given_ _X_ [1] := (X2(k−1)q+1, . . ., XN ). _Then,_ _the_
_protocol_ _succeeds_ _with_ _probability_ _at_ _least_ [2]3 _[only]_ _[if]_ _[R][ ≥]_ _[k]_ _[or]_ _[s][ = Ω(][ N]k[6][ )][.]_

_Proof._ Assuming the existence of a _k-player_ _R-round_ _s-space_ sequential blackboard protocol for hopk(X)N
as described above, we design a protocol for solving (q, k)-pointer chasing with _R_ rounds and _s-size_ messages.
The claimed lower bound will then follow by Proposition E.3.
Consider any pointer chasing input with universally known _V1, . . ., Vk+1,_ _v_ _∈_ _V1,_ and _Vk[1]+1_ [and] _[V]k[2]+1[,]_ [and]
each player Pj knowing matching Ej. We recursively define v1, . . ., vk +1 such that v1 = v and (vj, vj+1) ∈ _Ej,_
noting that the output hinges on whether _vk+1_ _∈_ _Vk[1]+1[.]_
Without loss of generality, let _v_ = 1 and


_Vj_ =


�{1, . . ., q} if _j_ is odd,
_{q + 1, . . ., 2q}_ if _j_ is even.


Each player independently determines their substring _X_ _[j]_ of a input _X_ to hopk before running the aforementioned protocol:

 - Player _P1_ encodes _X_ [1] by letting _XN_ = s = 1 and for any _i ∈_ 1, . . ., 2q, letting


_Xi[1]_ [=]


� _i+12_ _∈_ _V1_ if _i_ is odd,

_i[′]_ _∈_ _V2_ if _i_ is even, ( 2[i] _[, i][′][)][ ∈]_ _[E][1][.]_


This ensures that that every integer in _{1, . . ., 2q}_ appears exactly once in _X1[1][, . . ., X]2[1]q[,]_ [which] [in] [turn]
guarantees that find[1]X [(][N] [) = (][k][ −] [1 + 1)][q][ + 2] [and] [that] _[X]find[1]X_ [(][N] [)] [=][ v][2] [where] [(1][, i][′][)][ ∈] _[E][1][.]_

- For any _j_ _∈{2, . . ., k −_ 1}, player _Pj_ encodes _Ej_ as _X_ _[j]_ as follows. If _j_ is odd, then for every
_i ∈{1, . . ., 2q},_

_Xi[j]_ [=] � _i+12_ _∈_ _Vj_ if _i_ is odd,

_i[′]_ _∈_ _Vj+1_ if _i_ is even, ( 2[i] _[, i][′][)][ ∈]_ _[E][j][.]_

Alternatively, if _j_ is even,


_Xi[j]_ [=]


�q + _[i][+1]2_ _∈_ _Vj_ if _i_ is odd,

_i[′]_ _∈_ _Vj+1_ if _i_ is even, ( 2[i] _[, i][′][)][ ∈]_ _[E][j][.]_

33


-----

Since every odd token corresponds to a vertex in _Vj_ and each subsequent token corresponds to the
vertex it’s connected to by _Ej,_ we can ensure that for every _i ∈_ [2q]:

(X2(k−j+1)+i, Xfind1X [(2(][k][−][j][+1)+][i][)][)][ ∈] _[E][j][.]_

Hence, it follows inductively that _XfindjX_ [(][N] [)] [=][ v][j][+1][.]

- Player _Pk_ encodes _X_ _[k]_ if _k_ is odd by letting


_Xi[k]_ [=][ X][i] [=]

Likewise, if _k_ is even,


 _i+12_ _∈_ _Vk_ if _i_ is odd,

2q + 1 if _i_ is even, ( 2[i] _[, v][)][ ∈]_ _[E][k][,]_ [and] _[v]_ _[∈]_ _[V]k[1]+1[,]_

2q + 2 if _i_ is even, ( 2[i] _[, v][)][ ∈]_ _[E][k][,]_ [and] _[v]_ _[∈]_ _[V]k[2]+1[.]_


_Xi[k]_ [=][ X][i] [=]

These jointly ensure that


q + _[i][+1]2_ _∈_ _Vk_ if _i_ is odd,

2q + 1 if _i_ is even, ( 2[i] _[, v][)][ ∈]_ _[E][k][,]_ [and] _[v]_ _[∈]_ _[V]k[1]+1[,]_

2q + 2 if _i_ is even, ( 2[i] _[, v][)][ ∈]_ _[E][k][,]_ [and] _[v]_ _[∈]_ _[V]k[2]+1[.]_


hopk(X)N = XfindkX [(][N] [)] [=]


�2q + 1 if _vk+1_ _∈_ _Vk[1]+1[,]_
2q + 2 if _vk+1_ _∈_ _Vk[2]+1[.]_


Therefore, by formatting E1, . . ., Ek appropriately as X, running the protocol for hopk(X)N, and observing
that the final output of player _P_ [1] is 2q + 1 if and only if _vk+1_ _∈_ _Vk[1]+1[,]_ [there] [exists] [a] _[k][-player]_ _[R][-round]_ _[s][-space]_
protocol for pointer chasing. Hence, by Proposition E.3, the protocol for hopk(X)N must use _R ≥_ _k_ rounds
or _s = Ω(_ _k[N][6][ )]_ [space.]

#### E.2 Proofs of Section 5.2


**Corollary** **5.2.** _A_ _multi-layer_ _RNN_ _of_ _depth_ _L_ _and_ _width_ _m_ _as_ _above_ _with_ _YN_ = hopk(X)N _satisfies_ _either_
_L ≥_ _k_ _or_ _m = Ω(_ _k[N][6][ )][.]_

_Proof._ Suppose there exists a multi-layer RNN computing output _Y_ with _YN,1_ = hopk(X)N from input _X_
with intermediate states _Z1, . . ., ZL−1_ and hidden states _H_ [1], . . ., H _[L]._ For any _ℓ_ _∈_ [L] and _i ≤_ _i[′],_ note that
_Zi[ℓ][, . . ., Z]i[ℓ][′]_ [can] [be] [determined] [exactly] [from] _[H]i[ℓ]−1_ [and] _[Z]i[ℓ][−][1], . . ., Zi[ℓ][′][−][1]._ Given this RNN, we provide a multiplayer blackboard communication protocol for solving hopk(X)N under the input model of Proposition E.4.
In round _r,_ we assume inductively that each player _Pj_ knows _Z_ _[ℓ][−][1][,j]_ = (Z2([ℓ][−]k[1]−j)q+1[, . . ., Z]2([ℓ][−]k[1]−j+1)q[),]

except for _P1,_ who knows _Z_ _[ℓ][−][1][,][1]_ = (Z2([ℓ][−]k[1]−1)q+1[, . . ., Z]N[ℓ][−][1]). In descending order, each player _Pj_ computes
_Z_ _[ℓ,j]_ and _H2([ℓ]_ _k−j+1)q[—writing]_ [the] [latter] [on] [the] [blackboard—from] _[Z]_ _[ℓ][−][1][,j]_ [and] _[H]2([ℓ]_ _k−j)q[,which]_ [was] [written] [on]
the blackboard by the previous player. Thus, player _P_ [1] after round _L_ knows and outputs _ZN,[L]_ 1 [=] _[Y][N,][1]_ [=]
hopk(X)N, which provides an _L-round_ protocol _m-space_ protocol.
So the claimed lower bounds on width and depth follow from Proposition E.4.

#### E.3 Proofs of Section 5.3

**Corollary** **5.3.** _Any T_ _∈_ KernelFormer[N]m,m[′],L,H _[with][ T]_ [(][X][)][N] [=][ hop]k[(][X][)][N] _[satisfies either][ L][ ≥]_ _[k]_ _[or][ mm][′][Hp][ =]_
Ω( _k[N][6][ )][.]_

_Proof._ Under the distribution of input _X_ = (X [1], . . ., X _[k])_ to players _P1, . . ., Pk_ stipulated in the statement
of Proposition E.4, we explain how the players can all compute the outcome of a single layer of _H-headed_

34


-----

kernelized attention in a single round of a blackboard protocol. It is immediate that a depth _L_ network can
be simulated in _L_ rounds.
On input _X,_ consider _H_ kernelized self-attention units with embeddings (Q[′]1[, K]1[′] _[, V][1][)][, . . .,][ (][Q][′]H_ _[, K]H[′]_ _[, V][H]_ [)]
and output MLP _ψ._ Each player _Pj_ immediately computes its embeddings (Q[′]h[(][X] _[j][)][, K]h[′]_ [(][X] _[j][)][, V][h][(][X]_ _[j][))][h][∈][[][H][]][,]_
followed by (Kh[′] [(][X] _[j][)][T][V][h][(][X]_ _[j][))][ ∈]_ [R][m][′][×][m] [for] [each] _[h][ ∈]_ [[][H][].] [Because] [the] [object] [is] [to] [compute] [for] [each] _[h]_


_ψ(Q[′]h[(][X][)][K]h[′]_ [(][X][)][T][V][h][(][X][)) =][ ψ][(][Q][′]h[(][X][)]


_k_
� _Kh[′]_ [(][X] _[j][)][T][V][h][(][X]_ _[j][))][,]_

_j=1_


each player writes their (Kh[′] [(][X] _[j][)][T][V][h][(][X]_ _[j][))][h][∈][[][H][]]_ [using] [message] [size] _[s][ = Θ(][mm][′][Hp][).]_ [Each] [can] [then] [construct]
_Kh[′]_ [(][X][)][T][V][h][(][X][))] [by] [reading] [the] [board,] [and] [use] [it] [to] [compute] [its] [respective] [outputs] [without] [requiring]
supplemental communication.
Hence, _T_ (and thus hopk(X)N ) can be simulated using an _L-round_ blackboard protocol with message
size _s = Θ(mm[′]Hp),_ and the corollary follows from Proposition E.4.

**Corollary** **5.4.** _Any_ _T_ _∈_ Λ[w,g]-Attn[N]m,L,H _[with]_ _[T]_ [(][X][)][N] [=][ hop]k[(][X][)][N] _[satisfies]_ _[either]_ _[L][ ≥]_ _[k]_ _[or]_ [(][w] [+] _gk[N]_ [)][mHp][ =]

Ω( _k[N][6][ )][.]_

_Proof._ As in the proof of Corollary 5.3, we explain how each player can compute their respective outputs of a
single unit of self-attention masked by Λ[w,g].
To compute the output corresponding to _Xi,_ note that it is necessary to only know the embeddings
corresponding to _Xi−w, Xi−w+1, . . ., Xi+w_ and _Xg, X2g, . . ., X⌊N/g⌋g._ Thus, player _X_ _[j]_ can compute the
outputs of all of their inputs _X_ _[j]_ = (X2(k−j)q+1, . . ., X2(k−j+1)q) given access to

_X2(k−j)q+1−w, . . ., X2(k−j)q, X2(k−j+1)q+1, . . ., X2(k−j+1)q+w,_

as well as _Xg, X2g, . . ., X⌊N/g⌋g._
Therefore, the protocol can be simulated if each player _X_ _[j]_ writes inputs

_X2(k−j)q+1, . . ., X2(k−j)q+w, X2(k−j+1)q−w+1, . . ., X2(k−j+1)q_ _∈_ R[m],

in addition to all _Xi_ _∈_ _X_ _[j]_ such that _i_ _≡_ 0 (mod _g)._ This can be accomplished by a protocol where each
player writes _s = O((w +_ _gk[N]_ [)][mp][)] [bits] [of] [information] [on] [the] [blackboard.]

By repeating this protocol in parallel for every head and sequentially for every layer, _T_ and hopk(X)N
can be simulated, and hence the claim follows from Proposition E.4.

#### E.4 Proofs of Section 5.4

**Corollary** **5.6.** _Any_ _T_ _∈_ MaskTransformer[N]m,[+]1[N],H[CoT] _that_ _computes_ hopk(X)N _with_ _NCoT_ _tokens_ _of_ _chain-of-_
_thought_ _requires_ _either_ _NCoT_ _≥_ _k_ _or_ _mHp = Ω(_ _k[N][6][ )][.]_

_Proof._ We reduce to Proposition E.4. Consider some input _X_ _∈_ R[N] partitioned into _X_ [1], . . ., X _[j]_ as specified
by the proof of Proposition E.4 with chain-of-thought _XCoT_ and hopk(X)N determined by some masked
transformer T .[7] Suppose T has embeddings (Qh, Kh, Vh)h∈[H] and output MLP ψ. We provide an (NCoT +1)round blackboard protocol to compute hopk(X)N from _X._
Suppose in the _rth_ round of the protocol, all players know _XCoT,1, . . ., XCoT,r−1_ and aim to compute


_T_ (X _◦_ _XCoT)N_ +r−1 =


�XCoT,r if _r_ _≤_ _NCoT_
hopk(X)N if _r_ = NCoT + 1


�Ni=1+r−1 exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][)][T][)][V]i[h][(][X][i][)]
�iN=1+r−1 exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][))]


_H_
�

_h=1_


�


_._


= ψN +r−1


�

_XN_ +r−1 +


7We abuse notation to index _XN_ +i = XCoT,i and let _Xi_ _∈_ _X_ _j_ be true if _i ∈{2(k −_ _j)q + 1, . . ., w(k −_ _j + 1)q}._

35


-----

If we let


_Sr,h,j_ = � exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][)][T][)][V]i[h][(][X][i][)][ ∈] [R][m][,]

_Xi∈X_ _[j]_


_Sr,h,CoT_ =


_N_ +r−1
� exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][)][T][)][V]i[h][(][X][i][)][ ∈] [R][m][,]

_i=N_ +1


_Zr,h,j_ = � exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][)][T][)][ ∈] [R][,]

_Xi∈X_ _[j]_


_Zr,h,CoT_ =

then we observe that


_N_ +r−1
� exp(Q[h]N +r−1[(][X][N] [+][r][−][1][)][T][K]i[h][(][X][i][)][T][)][ ∈] [R][,]

_i=N_ +1


�kj=1 _[S][r,h,j]_ [+][ S][r,h,][CoT]
�kj=1 _[Z][r,h,j]_ [+][ Z][r,h,][CoT]


_H_
�

_h=1_


_._


_T_ (X _◦_ _XCoT)N_ +r−1 = ψN +r−1


�

_XN_ +r−1 +


�


Each player _Pk_ computes (Sr,h,j, Zr,h,j)h∈[H] and writes them on the blackboard with _O(mHp)-bit_ messages.
Since _Sr,h,CoT_ and _Zr,h,CoT_ are known by all players, every player can individually _T_ (X _◦_ _XCoT)N_ +r−1.
By induction, all players know hopk(X)N after _NCoT + 1_ rounds. The claim now follows from Proposition E.4.

### F Proofs of low-level attention constructions

#### F.1 Hardmax simulation proof of Appendix A.1

**Lemma** **A.2.** _Let_ _f_ _∈_ Attn[N]m _[be]_ _[a]_ _[self-attention]_ _[unit]_ _[with]_ _[precision]_ _[p][ = Θ(][log][ N]_ [)] _[and]_ _[embedding]_ _[functions]_
_Q, K, V_ _such_ _that_ _for_ _some_ _fixed_ 1 ≥ _ξ_ = N _[−][O][(1)]_ _and_ _every_ _X_ _∈_ R[N] _[×][m]_ _and_ _i ∈_ [N ]:

_A(X)i,i′_ _≤_ max _[−]_ _[ξ,]_ _[∀][i][′]_ _[̸∈]_ _[I][max][(][A][(][X][)][i][)][,]_
_i[′′]_ _[A][(][X][)][i,i][′′]_

_where_ _A(X) = Q(X)K(X)[T]._ _Then_ _there_ _exists_ _a_ _self-attention_ _unit_ _f_ _[′]_ _∈_ Attn[N]m _[with]_ _[a]_ _[valid]_ _[p][′][-bit]_ _[implemen-]_
_tation_ _with_ _p[′]_ = O(p) _satisfying_
_f_ _[′](X) = hardmax(A(X))V (X)._

_Proof._ For some _p[′]_ = Θ(p + log [1]ξ [)] [and] _[c][ = Θ(][ p][′]ξ[+][ζ]_ _· log N_ ) where _ζ_ is as in Appendix A.1), let _f_ _[′]_ have query

embedding _Q[′](X) = cQ(X)_ and identical key _K_ and value _V_ embeddings as _f_ . Therefore, by construction,
these embeddings can be written with precision _p[′]_ = O(ln(c) + p) = O(log [1]ξ [+ log log][ N] [+][ p][) =][ O][(][p][).]

Let _f[ˆ][′]_ be a valid _p[′]-bit_ implementation of _f_ _[′],_ meaning that the two _∥f[ˆ][′]_ _−_ _f_ _[′]∥∞_ = _O(1/2[p][+1])_ (thus _f[ˆ][′]_
rounds _f_ _[′]_ to _p[′]_ bits of precision), and fix some _X._ We first show that the softmax matrix is sufficiently close
to that of the hardmax and is also a valid _p[′]-bit_ implementation of the hardmax. Without loss of generality,
let 1 ∈ _Imax(A(X)i)._ First, note that


Then,


_N_ 1

� exp(cA(X)i,i′) ≤

exp(cξ) [exp(][cA][(][X][)][i,][1][) =] _N_ _[O][(][p][′][+][ζ][)]_ [exp(][cA][(][X][)][i,][1][)][.]
_i[′]̸∈Imax(A(X)i)_

_|softmax(cA(X))i,1 −_ hardmax(A(X))i,1| = _|Imax(A1(X)i)|_ _[−]_ �Ni[′]exp(=1 [exp(]cA([cA]X)[(]i,[X]1[)])[i,i][′][)]

_≤_ �i[′]̸∈Imax(A(X)i) [exp(][cA][(][X][)][i,i][′][)] = 1

_|Imax(A(X)i)| exp(cA(X)i,1)_ _N_ [Ω(][p][′][+][ζ][)][ .]

36


-----

Likewise, for any _i[′′]_ _̸∈_ _Imax(A(X)i):_

exp(cA(X)i,i′′ ) 1
_|softmax(cA(X))i,i′′_ _−_ hardmax(A(X))i,i′′| ≤ �Ni[′]=1 [exp(][cA][(][X][)][i,i][′][)] = _N_ [Ω(][p][′][+][ζ][)][ .]

Therefore,


_√_
_∥softmax(cA(X))i −_ hardmax(cA(X))i∥2 _≤_


1
_N_ _·maxi[′′]_ _[|][softmax(][cA][(][X][))][i,i][′′]_ _[−]_ [hardmax(][cA][(][X][))][i,i][′′][|][ =] _N_ [Ω(][p][′][+][ζ][)][ .]


We conclude that the approximation is sufficiently close, meaning it is _O(1/2[p][′]_ ), whereby it is exact after
rounding:

_fˆ ′(X) −_ hardmax(Q(X)K(X)T)V (X) _f ′(X) −_ hardmax(Q(X)K(X)T)V (X) _fˆ ′(X) −_ _f ′(X)_
��� ���∞ _[≤]_ �� ��∞ [+] ��� ���∞

_≤_ maxi,j ��softmax(cA(X))Ti _[V][ (][X][)][·][,j]_ _[−]_ [hardmax(][A][(][X][))][T]i _[V][ (][X][)][·][,j]��_ + O � 21[p][′]


�


_≤_ maxi,j ��softmax(cA(X))Ti _[−]_ [hardmax(][A][(][X][))]i[T]��2 _[∥][V][ (][X][)][·][,j][∥]2_ [+][ O] � 21[p][′]


�


1 _√_
_≤_ _N_ [Ω(][p][′][+][ζ][)] _[·]_


� 1
_N_ _· N_ _[ζ]_ + O
2[p][′]


� � 1
= O
2[p][′]


�
_._


Therefore, _f[ˆ][′]_ is a valid _p[′]-bit_ implementation of hardmax(Q(X)K(X)[T])V (X).

#### F.2 Constructions for Appendix B.1

**Proposition** **B.1.** _For_ _any_ _b ≤_ _N_ _and_ _d,_ _there_ _exists_ _a_ _self-attention_ _unit_ sparsePropagateQ,d _∈_ Attn[N]m,p _[for]_
_m = d+O(Q log N_ ) and p = O(log N ), which, given any input X _with Xi_ = (zi, Si,[⃗]0) ∈ R[d]×�≤[NQ]�×{0}[m][−][Q][−][d]

_such_ _that_ _bi_ = |{Sj _∋_ _i : j_ _∈_ [N ]}| ≤ _Q_ _for_ _all_ _i,_ _has_ _output_ sparsePropagateQ,d(X) _satisfying_


sparsePropagateQ,d(X)i = _b[1]i_


� _zj._

_Sj_ _∋i_


_Proof._ Following the proof of Theorem 2 of Sanford et al. (2023), there exist p-bit precision vectors u1, . . ., uN _∈_
_{±1/[√]m}[m]_ and _wS_ with _wS_ _≤_ 2[√]Q for all _S_ _∈_ �≤NQ� such that

_u[T]i_ _[w][S]_ [= 1][,] [for] [all] _[i][ ∈]_ _[S]_

_u[T]i_ _[w][S]_ _[≤]_ [1]2 _[,]_ [for] [all] _[i][ ̸∈]_ _[S.]_

We then design the embeddings of sparsePropagateQ,d with

_Q(X)i_ = (ui, 1),


_K(X)i_ =

_V (X)i_ =


�(wSi, 0) if _i > 0,_
([⃗]0, [3]4 [)] if _i = 0,_

�zi if _i > 0,_
_⃗0_ if _i = 0._


As a result,


_Q(X)[T]i_ _[K][(][X][)][i][′]_ [= 1] if _i ∈_ _Si′, i[′]_ _> 0,_

_Q(X)[T]i_ _[K][(][X][)][i][′]_ _[≤]_ [1] if _i ̸∈_ _Si[′], i[′]_ _> 0,_

2

_Q(X)[T]i_ _[K][(][X][)][0]_ [=] 4[3] _[.]_

37


-----

Hence, the largest inner products for query _i_ correspond to _i[′]_ for all _Si′_ _∋_ _i_ if any exist, and 0 otherwise.
There exists a margin of at least [1]4 [between] [the] [largest] [inner] [product] [in] [each] [row] [and] [all] [others.] [By] [applying]

Lemma A.2, we conclude that there exists a self attention unit _f_ _[′]_ with embedding dimension _p = Θ(log N_ )
that computes
_f_ _[′](X) = hardmax(Q(X)K(X)[T])V (X) = sparsePropagate(X)._

#### F.3 Constructions for Appendix B.2

**Lemma** **B.4.** _For_ _any_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _with_ _nin-word_ _inputs,_ _there_
_exists_ _a_ _transformer_ init ∈ Transformer[n]s,[in]1,[,]1[max(],din,d[n]out[in][,q][)] _with_ _din_ = 1 _and_ _dout_ = s, _which,_ _given_ `Input ∈` Z[n]2[p] _[,]_ _[has]_
_output_ _satisfying_ init(Input) = MachineIn[(1)].

_Proof._ Let _M_ = max(nin, q) and _Q, K, V_ : Z[M]2[p] _[→]_ [R][M] _[×][s]_ [be] [the] [query,] [key,] [and] [value] [embeddings] [of] [the]
attention unit f in init, and let ψ : R[M] _[×][s]_ _→_ Z[s]2[p][ ×][ [][N] []] [be] [its] [output] [MLP.] [Let][ q][in] [=] � _nsin_ � denote the number
of machines used to store the inputs.
Let `Desti′` = � _is[′]_ � _∈_ [qin] denote the machine that stores the input token index _i[′]_ _∈_ [nin] in the MPC

protocol, and let
`Rcvdi` = {(s − 1)i + 1, . . ., min(si, nin)}

denote the set of all input tokens indices belonging to `MachineIn[(1)]i` for machine _i ∈_ [qin].
For each machine _i ∈_ [qin], we define the query embedding as


� � 2πi
_Q(Input)i_ = cos
_M_


� � 2πi
_, sin_
_M_


� � 2πi
_, . . ., cos_
_M_


� � 2πi
_, sin_
_M_


��
_._


Likewise, for each token index _i[′]_ _∈_ [nin], the key and value vectors are


_K(Input)i′,(2ι−1,2ι)_ =

_V (Input)i′,(2ι−1,2ι)_ =


�[�]cos � 2π·DestM _i′_ � _, sin_ � 2π·DestM _i′_ �� if _i[′]_ _≤_ _nin,_ _i[′]_ _≡_ _ι_ (mod _s),_

(0, 0) otherwise,

�(Inputi′, i[′]) if _i[′]_ _≤_ _nin,_ _i[′]_ _≡_ _ι_ (mod _s),_
(0, i[′]) otherwise.


These definitions guarantee that large inner products only occur between machine queries _Q(Input)i_ and
tokens keys _K(Input)i′_ when `Inputi′` is allocated to `MachineIn[(1)]i` [.] [That] [is,]

_Q(Input)[T]i_ _[K][(][Input][)][i][′]_ [= 1][,] if _i[′]_ _∈_ `Rcvdi`


� 1
_Q(Input)[T]i_ _[K][(][Input][)][i][′]_ _[≤]_ [1][ −] [Ω] _M_ [2]


�
_,_ otherwise.


By applying Lemma A.2 with _ξ_ = Ω( _N[1][2][ ),]_ [there] [exists] [some] [self-attention] [unit] _[f][ ′]_ [such] [that]

_f_ _[′](Input)i_ = hardmax(Q(Input)K(Input)[T]) = [(][Input]|Rcvd[i][′][, i][′][)]i[i]|[′][∈][Rcvd][i] _._


A proper choice of _ψ_ and an invocation of the definition of `MachineIn[(1)]` ensures that init(Input)i =
_ψ(f_ (Input))i = MachineIn[(1)]i [.]

**Lemma** **B.6.** _For_ _any_ _R-round_ _MPC_ _protocol_ _π_ _with_ _local_ _memory_ _s_ _and_ _q_ _machines_ _with_ _nout-word_ _output,_
_there_ _exists_ _a_ _transformer_ final _∈_ Transformers,[q,]1[max(],1,din[n][out],dout[,q][)] _for_ _din_ = _s_ _and_ _dout_ = 1, _which,_ _given_ _input_
_X_ = MachineIn[(][R][)], _has_ _output_ final(X) _with_ final(X)i,1 = Outputi _∈_ Z2p.

38


-----

_Proof._ This argument inverts that of Lemma B.4, after applying the `LocalR` to transform `MachineIn[(][R][)]` to
`MachineOut[(][R][)].` Let _Q, K, V_ : Z[M]2[p] _[→]_ [R][M] _[×][s]_ [be] [the] [query,] [key,] [and] [value] [embeddings] [of] [the] [only] [attention]
unit _f_ in final, and let _ψ_ : R[M] _[×][s]_ _→_ Z[s]2[p] _[×][ [][N]_ []] [be] [its] [output] [MLP.] [Let] _[q][out]_ [=] � _nouts_ � denote the number of
machines storing relevant information for the output of the MPC protocol.
For each machine _i[′]_ _∈_ [qout], let

`Senti′` = {(s − 1)i[′] + 1, . . ., min(si[′], nout)}

denote the set of all token indices receiving its output. Likewise, for each token index i ∈ [nout], let Srci = ⌈i/s⌉
be the machine containing its relevant token. We define Q = Q[′] _◦_ `LocalR, K` = K _[′]_ _◦_ `LocalR, V` = V _[′]_ _◦_ `LocalR`
as follows.


_Q[′](MachineOut[(][R][)])i,(2ι−1,2ι)_ =


�[�]cos � 2π⌊MSrci⌋ � _, sin_ � 2π⌊MSrci⌋ �� if _i ≤_ _nout,_ _i ≡_ _ι_ (mod _s)_

(0, 0) otherwise.


� � 2πi′
_K_ _[′](MachineOut[(][R][)])i′_ = cos

_M_

_V_ _[′](MachineOut[(][R][)])i′_ = MsgOut[(]i[′][R][)][.]

Applying Lemma A.2 as before yields


� � 2πi′
_, sin_

_M_


� � 2πi′
_, . . ., cos_

_M_


� � 2πi′ ��
_, sin_ _._

_M_


_f_ (MachineIn[(][R][)])i =


�MachineOut[(]i[′][R][)] if _i ∈_ `Senti′,`
0 otherwise.


A properly chosen _ψ_ ensures that final(MachineIn[(][R][)])i = ψ(f (MachineIn[(][R][)]))i = Outputi.

#### F.4 Constructions for Appendix D.1

**Lemma** **D.1.** _For_ _some_ _m ≥_ _d + 2,_ _τ_ : [N ] × R[m] _→_ [N ], _and_ _ρ : R[m]_ _→_ R[d], _there_ _exists_ _an_ _attention_ _head_
lookUpτ,ρ _∈_ MaskAttn[N]m _[with]_ _[precision]_ _[p][ =][ O][(][log][ N]_ [)] _[and]_ _[m][ ≥]_ _[d]_ [+2] _[satisfying]_ [lookUp]τ,ρ[(][X][)][i,][:][d] [=][ ρ][(][X]τ (i,Xi)[)][.]

_Proof._ We let _V (Xi) = (ρ(Xi),[⃗]0)_ and define sinusoidal embeddings _Q_ and _K_ with


_Q(X)i_ = �cos � 2πτ (i, Xi)
_N_


� _, sin_ � 2πτ (i, Xi)
_N_


� �
_,[⃗]0_ _,_


� � 2πi
_K(X)i_ = cos
_N_


� � 2πi)
_, sin_
_N_


� �
_,[⃗]0_ _._


Note that


_Q(X)[T]i_ _[K][(][X][)][i][′]_ [= 1][,] if _τ_ (i, Xi) = i[′],


� 2π � � 1
_Q(X)[T]i_ _[K][(][X][)][i][′]_ _[≤]_ [cos] _N_ = 1 − Ω _N_ [2]


�
_,_ otherwise.


By applying Lemma A.2 with _ξ_ = Ω( _N[1][2][ ),]_ [we] [conclude] [that] [a] [satisfactory] [self-attention] [unit] [exists.]

**Lemma** **D.2.** _For_ _any_ _finite_ _alphabet_ Σ, _m_ _≥_ _d + 2,_ _µ1, µ2_ : R[m] _→_ Σ, _and_ _ρ_ : R[m] _→_ R[d], _there_ _exists_ _an_
_attention_ _head_ lastOccurrenceµ,ρ _∈_ MaskAttn[N]m _[with]_ _[precision]_ _[p][ =][ O][(log(][N][ |][Σ][|][))]_ _[such]_ _[that,]_


lastOccurrence(X)i,:d =


�ρ([⃗]0) _if_ _∀_ _i[′]_ _< i : µ1(Xi′) ̸= µ2(Xi),_
_ρ(Xi′)_ _if_ _i[′]_ = max {i[′] _< i : µ1(Xi′) = µ2(Xi)} ._

39


-----

_Proof._ Let _N_ _[′]_ = N _|Σ|._ We define token embeddings as follows, including start token “dummy embeddings”
as discussed in Appendix A.1.


_Q(X)i_ = �cos � 2π(NµN2(|ΣX|i) + i)

_K(X)i_ = �cos � 2π(NµN1(|ΣX|i) + i)


� _, sin_ � 2π(Nµ2(Xi) + i)
_N_ _|Σ|_

� _, sin_ � 2π(Nµ1(Xi) + i)
_N_ _|Σ|_


� �
_, 1,[⃗]0_ _,_

� �
_, 0,[⃗]0_ _,_


� � 2π(N _−_ 12 [)]
_K(X)0_ = 0, 0, cos _N_ _|Σ|_

_V (X)i_ = (ρ(Xi),[⃗]0),

_V (X)0_ = _[⃗]0._


� �
_,[⃗]0_ _,_


Taken together, these embeddings provide the following characterization of the inner products (with causal
masking matrix Γ):


� 2π(i − _i′)_
_Q(X)[T]0_ _[K][(][X][)][i][′]_ [+ Γ][i,i][′] [= cos]
_N_ _|Σ|_


�
if _i ≥_ _i[′]_ _> 0,_ _µ1(Xi[′]) = µ2(Xi),_


� 2π
_Q(X)[T]i_ _[K][(][X][)][i][′]_ [+ Γ][i,i][′] _[≤]_ [cos]
_N_


�
if _i ≥_ _i[′]_ _> 0,_ _µ1(Xi′) ̸= µ2(Xi),_


_Q(X)[T]i_ _[K][(][X][)][i][′]_ [+ Γ][i,i][′] [=][ −∞] if _i < i[′],_

� 2π(N _−_ 12 [)] �
_Q(X)[T]i_ _[K][(][X][)][i]_ [+ Γ][i,][0] [= cos] _N_ _|Σ|_ _._

As a result, the largest inner product _Q(X)[T]i_ _[K][(][X][)][i][′]_ [for] [some] _[i]_ [is] [the] [largest] _[i][′]_ [with] _[µ][1][(][X][i][′]_ [) =][ µ][2][(][X][i][)] [if] [one]
exists and _i[′]_ = 0 otherwise. Furthermore, there exists a margin of Ω( _N_ [2]1|Σ|[2][ )] [between] [this] [inner] [product] [and]
all others. We conclude by applying Lemma A.2.

### G Further empirical analysis of k-hop induction heads

This appendix presents in-depth explanations of the empirical results of Section 4.2, along with further
experiments. Taken together, these results suggest that the relationship between the number of hops _k_ and
the depth _L_ of transformers trained on the task is well-characterized by the representational thresholds of
Theorem 4.2 and Corollary 4.3; that the construction described in the proof of Theorem 4.2 is attainable by
trained models; and deep models likely exhibit an inductive bias that favors compositional learning rules in
the finite sample regime.
We define our experimental methodology precisely in Appendix G.1 and provide supporting evidence for
our claims in the subsequent sections.

**Exponential** **powers** **of** **depth.** Our principal empirical claim is that incrementing the depth _L_ of a
transformer exponentially increases the model’s capabilities to learn _k-hop_ induction heads tasks. We explore
this claim primarily in Appendix G.2, where we compare this empirical claim with the relevant theoretical
results (Theorem 4.2 and Corollary 4.3), which suggest a similar dependence. We further study the impacts
of increasing the embedding dimension _m_ of the transformer in Appendix G.3 and find that doubling the
width is roughly equivalent in performance to incrementing the depth by one.

**Empirical** **Claim** **G.1.** A transformer _T_ _∈_ MaskTransformer[N]m,L,H [trained] [with] [Adam] [to] [solve] [hop]k [has]
small token-wise classification error if _L log(m) = Ω(log k)_ and large error if _L log m = O(log k)._

**Mechanistic** **alignment** **with** **theoretical** **construction.** We further demonstrate the empirical salience
of our theoretical construction by conducting a study of the interpretability of learned transformers in
Appendix G.4. This investigation reveals that the attention matrices of sufficiently deep transformers exhibit

40


-----

an implementation of a circuit that relies on the same “doubling” principle of the construction in the proof of
Theorem 4.2. The resulting circuit is comprised of the same intermediate products that are used in that hopk
construction.

**Empirical Claim G.2.** The outputs of individual attention matrices of a transformer T _∈_ MaskTransformer[N]m,L,H
trained with Adam to solve hopk with _L = Ω(log k)_ and evaluated on input _X_ _∈_ Σ[N] (i) correspond to the
find[j]X [intermediate] [products] [of] [the] [Theorem] [4.2] [construction] [and] [(ii)] [demonstrate] [a] [“doubling”] [phenomenon]
where the each head layer _ℓ_ corresponds to find[j]X [for] [some] _[j]_ [=][ O][(2][ℓ][).]

**Beneficial** **inductive** **biases** **of** **depth.** While most of our experiments belong to the “infinite-sample”
regime where new samples are randomly generated on each training step, we also evaluate our models in two
finite-sample regimes in Appendix G.5. We find that a small number of samples is sufficient to approach the
performance of the infinite-sample regime. When the amount of training data is small, we find that deeper
models perform better than shallower models, possibly due to an inductive bias that favors compositional
hypotheses.

**Empirical Claim G.3.** hopk can be learned in a sample-efficient manner by transformers T _∈_ MaskTransformer[N]m,L,H
trained with Adam with _L = Ω(log k)._ If _T_ overfits to hopk tasks for some _k,_ then increasing the depth _L_
while holding _k_ fixed leads superior performance.

The experiments detailed here were conducted under limited computational resources. The authors are
interested in future work that would evaluate whether these scaling rules persist on larger architectures and
more complex tasks.

#### G.1 Experimental details

**Task details.** We study a multi-task variant of k-hop induction heads that predicts hopk(X) = (0, hopk(X _[′]))_
from input _X_ = (k, X _[′])_ for _k_ _∈{0, 1, . . ., kmax}[8]_ and _X_ _[′]_ _∈_ Σ[N] _[−][1]._ We refer to this task as _multi-hop_ and
provide the task hyper-parameters in Table 1.

Hyperparameter Value

Context length _N_ 100
Alphabet size _|Σ|_ 4
Max hops _kmax_ 16

Table 1: Multi-hop task hyper-parameters

We define the distribution _Dmulti−hop_ over labeled samples for the multi-hop task and _DX_ over input
sequences _X_ _∈_ Σ[N] _[−][1]._ We draw a labeled sample (X, hopk(X)) _∼Dmulti−hop_ by independently sampling
_k_ _∼_ Unif({0, 1, . . ., kmax}) and X _[′]_ _∼DX ._ Input sequences X _[′]_ _∼DX_ are drawn uniformly from inputs with _no_
_repeating_ _elements._ That is, we sample _X1[′]_ _[∼]_ [Unif][(Σ)] [and] [each] _[X]j[′]+1_ _[∼]_ [Unif][(Σ][ \] �Xj[′] �). For each _k_ _∈_ [kmax],
let _Dhopk_ denote the conditional distribution ((k[′], X _[′]), (0, hopk′(X_ _[′])))_ _∼Dmulti−hop_ _|_ (k = _k[′])._ Also, let
dom(hopk) = {(k, X _[′]) : Pr [X_ _[′]_ _∼DX ] > 0}._
For Σ := Σ ∪ [kmax], we define the _n-sample_ _empirical_ _token-wise_ _classification_ _error_ of a transformer
_N_ _N_
_T_ : Σ _→_ Σ on a task hopk as


`err[n]k` [(][T] [) =] [1]

_n_


_n_
�

_ι=1_


1
_| {i : hopk(X_ _[ι])i_ ≠ _⊥} |_


_N_
� 1 {T (X _[ι])i_ = hop̸ _k(X_ _[ι])i_ ≠ _⊥},_

_i=1_


for iid samples (X [1], hopk(X [1])), . . ., (X _[n], hopk(X_ _[n]))_ _∼Dhopk_ . We ignore null _⊥_ outputs of hopk when no
_k-hop_ induction head exists in order to avoid inadvertently over-estimating the performance of transformers
on large _k_ tasks, which have a large fraction of null outputs.

8The task hop0 is simply the identity mapping: hop0(X _′) = X_ _′._

41


-----

**Training** **details.** We trained a variety of causally-masked GPT-2 transformers (Radford et al., 2019) from
HuggingFace to solve the multi-hop task. The model has an absolute positional encoding.
The transformers are trained with Adam (Kingma and Ba, 2014) on the cross-entropy loss. In the
infinite-sample regime, we draw 32 new iid samples from _Dmulti−hop_ on each training step. Otherwise, _ntrain_
samples are drawn before training commences and all samples are rotated through batches, before repeating.
We use the hyper-parameters in Table 2 to train all of the models identified in Table 3.

Hyperparameter Value

Embedding dimension _m_ _{128, 256}_
Depth _L_ _{2, 3, 4, 5, 6}_
Number of heads _H_ _{4, 8}_
Vocabulary size 30
Activation function GeLU
Layer norm _ϵ_ 10[−][5]

Training samples _ntrain_ �10[3], 3 · 10[3], ∞�

Learning rate 10[−][4]

Training steps 10[5]

Batch size 32

Table 2: Model and training hyper-parameters

Identifier Heads _H_ Embedding dimension _m_ Depth _L_ Training samples _ntrain_ Total parameters

_T4[∞],2_ 4 128 2 _∞_ 413,440
_T4[∞],3_ 4 128 3 _∞_ 611,712
_T4[∞],4_ 4 128 4 _∞_ 809,984
_T4[∞],5_ 4 128 5 _∞_ 1,008,256
_T4[∞],6_ 4 128 6 _∞_ 1,206,528
_T8[∞],2_ 8 256 2 _∞_ 1,613,312
_T8[∞],3_ 8 256 3 _∞_ 2,403,072
_T8[∞],4_ 8 256 4 _∞_ 3,192,832
_T8[∞],5_ 8 256 5 _∞_ 3,982,592
_T8[∞],6_ 8 256 6 _∞_ 4,772,352
_T4[3000],2_ 4 128 2 3000 413,440
_T4[3000],3_ 4 128 3 3000 611,712
_T4[3000],4_ 4 128 4 3000 809,984
_T4[3000],5_ 4 128 5 3000 1,008,256
_T4[3000],6_ 4 128 6 3000 1,206,528
_T4[1000],2_ 4 128 2 1000 413,440
_T4[1000],3_ 4 128 3 1000 611,712
_T4[1000],4_ 4 128 4 1000 809,984
_T4[1000],5_ 4 128 5 1000 1,008,256
_T4[1000],6_ 4 128 6 1000 1,206,528

Table 3: Hyper-parameters of all MaskTransformer[N]m,L,H [trained] [for] [the] [empirical] [analysis.]

**Computational** **resources.** All experiments were run on a 2021 Macbook Pro with an M1 chip.

42


-----

#### G.2 Exponential increases in k-hop capacity with depth (Empirical Claim G.1; Figures 6 to 8)

Evaluation of L-depth, 4-headed, infinite-sample intransformers on hopk

L = 2

0.7 L = 3

L = 4
L = 5

0.6 L = 6

0.5

0.4

0.3

0.2

0.1

0.0

1 2 4 8 16
k

Figure 6: Zoomed in version of Figure 2. Evaluation of transformers err[n]k [(][T][ ∞]4,L[) with depths][ L][ ∈{][2][,][ 3][,][ 4][,][ 5][,][ 6][}][,]
heads _H_ = 4, and embedding dimension _m = 128_ trained on the multi-hop task. This figure plots `err[n]k` [(][T][ ∞]4,L[)]
on _n = 100_ samples as a function of _k_ for each choice of _L._

We visualize the relationship between the depth _L_ of a transformer and the largest _k_ such that `err[n]k` [(][T] [)]
is small in Figure 6, Figure 7, and Figure 8. We exhibit the relationship in its simplest form by considering
transformers with heads _H_ = 4, embedding dimension _m = 128,_ and new training samples on every epoch.
The figures provide alternate views of `err[n]k` [(][T][ ∞]4,L[)] [for] [each] _[L][ ∈{][2][,][ 3][,][ 4][,][ 5][,][ 6][}]_ [with] _[n][ = 100]_ [samples] [for] [each]
_k_ _∈_ [kmax].
Together, these plots illustrate a sharp phase transition when _D_ = ⌊log2 k⌋ + 2, which identically matches
the depth scaling in Theorem 4.2. Increasing the depth of a transformer by one approximately doubles
the number of values _k_ _∈_ [kmax] with bounded error. For instance, following the theoretical and empirical
intuition of Bietti et al. (2023), the depth _L = 2_ transformer _T4[∞],2_ [succeeds] [in] [solving] [the] [standard] [induction]
heads task, but attains at least 10% error on all other tasks. Likewise, a depth L = 3 model has error bounded
by 1% for _k_ _∈{1, 2},_ which increases rapidly for larger values of _k._
This doubling phenomenon suggests that simple compositional tasks with a larger number of compositions
than the depth of the model are easily learnable if the model can employ a doubling trick, similar to the one
used in the proof of Theorem 4.2. This relationship between compositionality and depth reflects the results
of Zhang et al. (2023), where the learnable task complexity also scales super-linearly in depth.
Given the lower bounds of Corollary 4.3, one may ask why models with depth _L_ _<_ _⌊log2 k⌋_ achieve
non-trivial success on hopk tasks that cannot be represented in a compositional manner. There are several
relevant explanations:

1. In these experiments, the embedding dimension _m = 128_ is actually larger than the context _N_ = 100,
which may enable the model to memorize more of its preceding samples and offload logical work to the
MLP, rather than executing a pointer-doubling strategy. While practical models regularly have the
opposite (and our theoretical results are oriented around that parametric scaling), we used a larger

43


-----

Figure 7: Alternate view of Figure 6 including `err[n]k` [(][T][ ∞]4,L[)] [plotted] [as] [a] [function] [of] _[L]_ [for] [each] _[k][.]_

_m_ than is necessary for representational purpose to improve the optimization landscape and speed
convergence.

2. This is made further plausible by the small alphabet size _|Σ|_ and randomly drawn sequences _X_ _[′],_ which
place effective bounds on how much look-back from each token _i_ is necessary to compute hopk(X)i.

Nonetheless, these results provide strong support that models are substantially easier to train to low
classification error in the regime where the depth is sufficient to implement a pointer-doubling construction.
In the following subsection, we further investigate this phenomenon by examining the intermediate attention
matrices produced by trained models.

44


-----

Figure 8: Alternate views of Figure 6 including `err[n]k` [(][T][ ∞]4,L[)] [as] [a] [table] [with] [one] [cell] [for] [each] [(][L, k][)] [pair.]

#### G.3 Width variation (Empirical Claim G.1; Figure 9)

While the primary focus of these empirical results and the paper as a whole is on the role of depth in the
ability of transformer to learn parallelizable and compositional tasks, we also aim to understand the interplay
of depth and width in learning the multi-hop task. Here, we contrast the previous transformers _T4[∞],L_ [with]
models _T8[∞],L_ [that] [have] [more] [heads] [(][H] [=] [8)] [and] [larger] [embedding] [dimensions] [(][m] [=] [256).] [We] [plot] [the]
classification errors of all 10 architectures over 16 hopk sub-tasks in Figure 9.
Here, we observe a rough correspondence in performance between the transformers TH,L[∞] [and] _[T][2][H,L][−][1]_ [and]
the same doubling phenomenon as is evident models with _H_ = 4 heads. That is, while increasing the width
improves the classification error of learned models, it does so in a far less parameter-efficient manner than
incrementing the depth. As mentioned before, the relative success of wide and shallow transformers is likely
contingent on the relatively short context length _N_ and alphabet size _|Σ|._ However, these results still suggest
an important role for wider models to play beyond representational capabilities of transformers.

45


-----

Figure 9: Comparison between the errors `err[n]k` [(][T][ ∞]H,L[)] [of] [transformers] [with] [embedding] [dimension] [and] [heads]
(m, H) = (4, 128) (dashed line, same plots as Figure 6) and (m, H) = (8, 256) (solid line) trained on the
multi-hop task, evaluated on _n = 100_ samples per hopk task.

46


-----

#### G.4 Mechanistic alignment with theoretical construction (Empirical Claim G.2, Figures 10 to 15)

We use standard attention-based interpretability techniques to better understand what particular logical
circuits are implemented by transformers trained to solve the multi-hop task. By qualitatively inspecting
the attention matrices produced by trained models and by measuring the alignment between those inner
products and partial solutions find[j] of hopk, we uncover a striking correspondence between the behaviors
of the trained models and the transformer construction designed in the proof of Theorem 4.2. We further
observe that trained transformers with high accuracy have “decisive” self-attention units with particularly
strong correlations to some find[j] intermediate, while poorly performing models have less predictable attention
activations.
For a fixed trained model T _∈_ Transformer[N]m,L,H [, we let][ A][ℓ,h][[][T] [](][X][) represent the output of the][ h][th self-self]
attention matrix in the _ℓth_ layer for _h ∈_ [H] and _ℓ_ _∈_ [L], evaluated at some input _X_ _∈_ dom(hopk). That is,
we let
_A[ℓ,h][T_ ](X) = softmax �Q[ℓ,h](X _[ℓ][−][1])K_ _[ℓ,h](X_ _[ℓ][−][1])[T]_ + Γ� _∈_ R[N] _[×][N]_ _,_

where _X_ _[ℓ][−][1]_ is the intermediate state representing the output of layer _ℓ_ _−_ 1 of _T_ on input _X_ and Γ is the
causal masking matrix. Each row _i_ in the matrix represents the coefficients of the convex combination of
value vectors affiliated with each query, which can be used as a signifier of which embeddings _i_ receives
information from.

**Visualization** **of** find[j] **alignment** **for** hop16 **and** **depth** _L = 6_ **(Figure** **10).** The outputs of self-attention
matrices are often highly structured matrices that reveal which relationships between tokens are encoded and
how information is shared within the model (Li and McClelland, 2022; Clark et al., 2019; Rogers et al., 2021).
We plot several self-attention matrices associated with a depth _L = 6,_ heads _H_ = 4 transformer trained in
the infinite-sample regime and evaluated on a single sample _X_ _∈_ dom(hop16) in Figure 10.
By looking at the six self-attention matrices, one can infer that all heads are “decisive” and obtain nearly
all of their relevant information from a single value embedding, rather than averages of a large number
of embeddings. The top-left self-attention matrix, which belongs to the first self-attention head, clearly
associates elements with their predecessors, which is identical the to the function of our lookUp attention
head in the first layer of the hopk construction of Theorem 4.2.
While the roles of the other heads are not immediately obvious, they can be understood by overlaying
colored matrices with non-zero cells at (i, find[j]X [(][i][)) for some][ j] _[≤]_ _[k][.]_ [For instance, the top-right attention matrix]
in layer _ℓ_ = 2 corresponds almost exactly with find[1]X [(as] [suggested] [by] [the] [second-layer] [of] [our] [construction),]
and the others are closely associated with find[1]X [,] [find][2]X [,] [find][3]X [,] [and] [find][8]X [for] [layers] _[ℓ]_ [= 3][,][ 4][,][ 5][,][ 6] [respectively.]
This is a remarkably close correspondence to our construction, which includes a self-attention matrix in the
_ℓth_ layer whose activations correspond to find[2]X[ℓ][−][2].
While not conclusive, this experiment suggests a strong alignment between the behaviors of this particular
transformer and our theoretical construction. This suggests a high likelihood that the transformer successfully
learns to solve hop16 by employing a pointer-doubling primitive. However, these results apply to only a single
model, a single task, and a single input; in the subsequent section, we generalize this interpretability analysis.

**Alignment** **between** **attention** **heads** **and** find[j] **for** **a** **single** hopk **sub-task** **(Figures** **11** **to** **13).** To
broaden and quantify the analysis of the previous section, we measure the extent to which each self-attention
head mimics the functionality of find[j], which are partial computations of hopk that are employed in the proof
of Theorem 4.2. We use cell-wise matrix inner products to quantify the strength of correlation between a
self-attention matrix and a fixed function potentially relevant to interpretability.
For two matrices _A, B_ _∈_ R[N] _[×][N]_, let

_⟨A, B⟩_ = _[∥][A][ ⊙]_ _[B][∥]F[2]_
_∥A∥F_ _∥B∥F_

be their normalized element-wise inner-product, where ∥·∥F is the Frobenius norm and ⊙ denotes element-wise

47


-----

Figure 10: The outputs of several internal self-attention matrices _A[ℓ,h][T4[∞],6[](][X][)]_ _[∈]_ [R][100][×][100] [of] [a] [trained]
multi-task transformer of depth _D_ = 6 evaluated on a single sample _X_ _∼Dhop16_ are plotted in grayscale. In
each cell, the matrix with non-zero entries (find[j]X [(][i][)][, i][)][i][∈][[][N] []] [for] [some] _[j]_ [is] [included] [in] [transparent] [color] [to]
visualize the function of each self-attention unit.

48


-----

multiplication. For some function _g_ : [N ] →{0} ∪ [N ], we let _⟨g, B⟩_ := ⟨A[g], B⟩, where


_A[g]i,j_ [=]


�1 if _g(j) = i,_
0 otherwise.


We use this notation to analyze experimentally how closely the self-attention matrices _A[ℓ,h]_ encode the

_[∈∼D][hop]_
intermediate products of the proof of Theorem 4.2, find[j]X [.] [For] _[n]_ [iid] [samples] _[X]_ [1][, . . ., X] _[n]_ _k_ [,] [let]


_n_
�

_ι=1_


�A[ℓ,h], find[j][�]



[1]
_n,k_ [:=]

_n_


� �
find[j]X _[ι][, A][ℓ,h][(][X]_ _[ι][)]_ _._


Due to the non-negativity of _A[ℓ,h]_ and find[j], �A[ℓ,h], find[j][�]



[and] �A[ℓ,h], find[j][�]
_n,k_ _[∈]_ [[0][,][ 1],]



[only] [if] _[∀][ι][ ∈]_ [[][n][]:]
_n,k_ [= 1]


_A[ℓ,h](X_ _[ι])i,i[′]_ = 1 _⇐⇒_ find[j]X _[ι][(][i][) =][ i][′][.]_

These inner products make it possible to visualize the strength of correlations of all heads in a particular
model _T_ _∈_ MaskTransformer[N]m,L,H [with] [all] [target] [functions] [find][j] [on] [a] [collection] [of] [random] [samples] [drawn]
from some Dhopk . Figure 11 visualizes the functionality of all attention units in the 4-layer, 4-head transformer
_T4[∞],4_ [when] [evaluated] [on] [the] [sub-task] [hop]4[.] [The] [figure] [gives] [several] [clues] [about] [how] [hop]4 [is] [successfully]
computed by the trained model: the second layer and third layer both utilize find[1] to determined find[2] jointly
by the end of the third layer. The fourth layer uses the ability to create a stable find[2] construction to obtain
find[4] and hence hop4.
This plot also indicates the relative stability of this circuit interpretation of the procedure: a large number
of heads are very strongly correlated with find[1] or find[2] across the 10 samples, which indicates they are likely
utilized consistently to compute those intermediates regardless of input.
Figure 12 is a similar plot for the transformer _T4[∞],6_ [with] [depth] _[L][ = 6,]_ [evaluated] [on] [the] [task] [hop]16[.] [The]
functionalities of the heads visualized in Figure 10 can be observed in the corresponding inner products. The
collection of all inner products presents further evidence that the pointer-doubling phenomenon occurs in the
trained models, due to the increase in compositions present in the largest inner products of deeper attention
units.
While Figures 11 and 12 showcase the decisive alignment between self-attention heads and particular
partial computations find[j] in successfully trained models, Figure 13 demonstrates the loss of that decisiveness
in poorly performing transformers. There, we visualize the alignments of the trained depth-4 transformer
_T4[∞],4_ [evaluated] [on] [hop]16[,] [in] [which] [it] [attains] [a] [61%] [token] [error.] [While] [a] [self-attention] [units] [in] [the] [second]
layer coincides with find[1], no strong correlations emerge deeper in the model. Unlike the other figures, the
deeper self-attention units are “indecisive,” lacking any large inner products and failing in particular to
correlate with any highly compositional targets. This provides a visual explanation of the transformer’s
failure, since it lacked the effective representational capacity needed to learn a circuit with consistent and
highly-compositional outputs.[9]

**Alignment** **between** **attention** **heads** **and** find[j] **for** **all** hopk **sub-tasks** **(Figures** **14** **and** **15).** For
an even more global lens on the mechanistic interpretability of these trained models, we visualize how the
maximum inner products of each self-attention unit change for a fixed transformer for different sub-tasks
hopk. Figures 14 and 15 do so for the depth-4 and depth-6 networks respectively. The hue of each cell (and its
numerical label) corresponds to the _j[∗]_ with the most correlated inner product with corresponding attention
unit _A[ℓ,h]_ in samples from dom(hopk), and the opacity corresponds to the magnitude of that inner product.
The takeaways of the previous inner product figures are apparent in these: the approximate doubling
for the depth _L = 6_ transformer can be visualized by the vertically changing opaque colors. Conversely, a

9Since these experiments are in the small alphabet size _|Σ| = 4_ regime, this task performs better than random guessing due to
inferential capabilities that are are powered by the high embedding dimension and do not require implementing a pointer-chasing
algorithm. We suspect that the “checkerboard” patterns are powered by this inference.

49


-----

Figure 11: Plots of all inner products �A[ℓ,h][T4[∞],4[]][,][ find][j][�]10,4 [for] _[n][ = 10]_ [samples] _[X]_ [1][, . . ., X] [10] _[∈]_ [dom][(][hop][4][)] [for]

the 4-layer transformer _T4[∞],4[.]_

separation can be observed between the tasks where the depth _L_ = 4 transformer performs well and has
“decisive” self-attention units deeper in the network and those where it does not.
Moreover, the figures (especially Figure 15) demonstrate that several self-attention units have a consistent
function among samples from the same task, while adapting in function to different hopk tasks. This is most
apparent in head _h = 4_ of layer _ℓ_ = 6, where the self-attention head functions as find[1], find[3], find[5] or find[7]

depending on the complexity of the task.

50


-----

Figure 12: Plots of all inner products �A[ℓ,h][T4[∞],6[]][,][ find][j][�]10,16 [for] _[n][ = 10]_ [samples] _[X]_ [1][, . . ., X] [10] _[∈]_ [dom][(][hop][16][)]

for the 6-layer transformer _T4[∞],6[.]_

51


-----

Figure 13: Plots of all inner products �A[ℓ,h][T4[∞],4[]][,][ find][j][�]10,16 [for] _[n][ = 10]_ [samples] _[X]_ [1][, . . ., X] [10] _[∈]_ [dom][(][hop][16][)]

for the 4-layer transformer _T4[∞],4[.]_

52


-----

Figure 14: Plots of all the maximum inner products �A[ℓ,h][T4[∞],4[]][,][ find][j][�]n,k [for] _[n]_ [=] [10] [fixed] [samples]

_X_ [1], . . ., X [10] _∈_ dom(hopk) for each _k_ _∈_ [16] for the 4-layer transformer _T4[∞],4[.]_ The hue corresponds to
the index of the largest inner product _j[∗]_ = arg maxj �A[ℓ,h][T4[∞],4[]][,][ find][j][�]n,k[,] [while] [the] [opacity] [is] [determined] [by]

the magnitude of the correlation.

53


-----

Figure 15: Plots of all the maximum inner products �A[ℓ,h][T4[∞],6[]][,][ find][j][�]n,k [for] _[n]_ [=] [10] [fixed] [samples]

_X_ [1], . . ., X [10] _∈_ dom(hopk) for each _k_ _∈_ [16] for the 6-layer transformer _T4[∞],6[.]_

54


-----

#### G.5 Finite-sample experiments (Empirical Claim G.3; Figures 16 to 19)

While most of our multi-hop experiments reside in the infinite-sample regime (where new samples are
generated for every batch), we also trained several transformers on _ntrain_ _∈{1000, 3000}_ samples to evaluate
whether generalization is possible in this domain, especially when the number of model parameters far exceeds
the number of training samples. The two training set sizes expose a sharp threshold between two different
generalization modes: low accuracy due to overfitting for most models on most tasks when _ntrain_ = 1000 and
high accuracy approaching the infinite-sample regime when _ntrain_ = 3000.
Figure 16 compares the infinite-sample transformers T4[∞],L [with the 3000-sample models][ T][ 3000]4,L [.] [3000 training]
samples are sufficient to obtain comparable (if slightly worse) generalization error rates across model depths
_L_ and task complexities k. This supports a hypothesis that the existence of a small transformer that perfectly
fits the data enables larger transformers to actually realize such architectures in the over-parameterized
regime.
On the other hand, Figure 17 demonstrates that transformers trained on _ntrain_ = 1000 samples suffer
poor performance on most tasks due to overfitting. While all models perform poorly on hopk sub-tasks for
large _k,_ a depth-separation exists for simpler sub-tasks like hop3. This suggests a positive inductive bias of
deep transformers for simple compositional decision rules, which enables far better performance than other
models in the overfitting regime.
To investigate this gap in performance, we contrast the self-attention inner products of depth-4 _T4[1000],4_ and
depth-6 _T4[1000],6_ on the task hop3 in Figures 18 and 19. The 6-layer model obtains a far superior classification
error on the sub-task, and the interpretability plot establishes a plausible circuit it implements: It uses
self-attention heads with find[1] functionality consecutively in layers 4, 5, and 6, which enables the robust
retrieval of find[3] and hop3. On the other hand, the 4-layer plot exhibits poor performance and only has two
layers with find[1] functionality; this justifies the relatively strong performance of _T4[1000],4_ on hop2 and its poor
performance on hop3.
While neither model learns any kind of pointer-doubling construction, the 6-layer model is still able to
learn a simple construction of hop3 that the 4-layer model misses. The representational suitability of deeper
models to compositional reasoning may thus provide a favorable inductive bias for learning the task in a
setting with little data.

55


-----

Figure 16: Comparison between the errors `err[n]k` [(][T][ n]4,L[)] [of] [transformers] [trained] [in] [the] [infinite] [sample] [regime]
(dashed line) and on _ntrain_ = 3000 samples (solid line) on the multi-hop task, evaluated on _n = 100_ samples
per hopk task.

Evaluation of L-depth, 4-headed, 1000-sample intransformers on hopk

0.8

L = 2, n = 1000
L = 2, n =

0.7 L = 3, n = 1000

L = 3, n =
L = 4, n = 1000

0.6 L = 4, n =

L = 5, n = 1000

0.5 L = 5, n =

L = 6, n = 1000
L = 6, n =

0.4

0.3

0.2

0.1

0.0

1 2 4 8 16
k

Figure 17: Comparison between the errors `err[n]k` [(][T][ n]4,L[)] [of] [transformers] [trained] [in] [the] [infinite] [sample] [regime]
(dashed line) and on _ntrain_ = 1000 samples (solid line) on the multi-hop task, evaluated on _n = 100_ samples
per hopk task.

56


-----

Figure 18: Plots of all inner products �A[ℓ,h][T4[1000],4 []][,][ find][j][�]10,3 [for] _[n][ = 10]_ [samples] _[X]_ [1][, . . ., X] [10] _[∈]_ [dom][(][hop][3][)]

for the 4-layer transformer _T4[1000],4_ [.]

57


-----

Figure 19: Plots of all inner products �A[ℓ,h][T4[1000],6 []][,][ find][j][�]10,3 [for] _[n][ = 10]_ [samples] _[X]_ [1][, . . ., X] [10] _[∈]_ [dom][(][hop][3][)]

for the 6-layer transformer _T4[1000],6_ [.]

58


-----

