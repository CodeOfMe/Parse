**EMBEDDING** **INDUCED** **TREES** **IN** **SPARSE** **EXPANDING** **GRAPHS**


ANTONIO[´] GIRAO[˜] AND EOIN HURLEY

Abstract. Inspired by the network routing literature [ABNC[+]96], we develop what we call a “Pre-Emptive

Greedy Algorithm” to embed bounded degree induced trees in sparse expanders. This generalises a powerful

and central result of Friedman and Pippenger to the induced setting. As corollaries we obtain that a sparse

random graph contains all bounded degree trees of linear order (whp) and that the induced and size induced

Ramsey numbers of bounded degree trees are linear. No such linear bounds were previously known. We also
prove a nearly-tight result on induced forests in bounded degree countable expanders. We expect that our
new result will find many more applications.


1. Background

Over the past century, a central theme in Combinatorics has been to find the right conditions that guarantee
the existence of specific subgraphs. Examples include Dirac’s Theorem [Dir52] from the 40’s, Ramsey theory

[Ram28], universal graphs [Moo65] and multiple questions in random graph theory going back as far as the
birth of the subject itself [ER60]. On the applications side, the travelling salesman problem or network
routing problems can both be phrased in terms of finding or containing appropriate subgraphs.

Dirac’s Theorem is perhaps _the_ classical result in this area, it states that _every_ _n-vertex_ graph with
minimum degree at least _n/2_ contains a Hamilton cycle. Thirty five years later P´osa [P´os76] gave the first
proof of Hamiltonicity (with high probability) in sparse random graphs of average degree Ω(log n). Implicit
in his work was the following fundamental deterministic result (N (X) is the set of vertices with a neighbour
in _X)._


**Theorem** **1.1** (P´osa). _Let_ _G_ _be_ _a_ _graph_ _such_ _that_ _every_ _subset_ _X_ _⊂_ _V (G)_ _with_ _at_ _most_ _n_ _vertices_ _satisfies_
_|N_ (X)| > 3|X| − 1, _then_ _G_ _contains_ _a_ _path_ /cycle _on_ _at_ _least_ 3n − 2 _vertices._

This result (and the method of P´osa rotations) quickly found other applications, most notably in Beck’s
proof [Bec83] that the size ramsey number of paths is linear in its order. This answered a question from
the seminal paper of Erd˝os, Faudree, Schelp and Rousseau [EFRS78] for which Erd˝os had later offered $100

[Erd81]. Beck conjectured that such a linear upper bound should hold not just for paths but for any bounded
degree tree (see Section 1.2), but could not prove it. Finally, Friedman and Pippenger [FP87] confirmed this
by showing the following beautiful generalisation of P´osa’s Theorem.


**Theorem 1.2.** _Let G be a graph such that every set X_ _with at most 2n−2 vertices satisfies |N_ (X)| ≥ (d+2)|X|,
_then_ _G_ _contains_ _every_ _tree_ _with_ _maximum_ _degree_ _at_ _most_ _d_ _on_ _n_ _vertices._

AG: Mathematical Institute, University of Oxford, Oxford OX2 6GG, UK. E-mail: `girao@maths.ox.ac.uk.` Research
supported by EPSRC grant EP/V007327/1 and ERC Advanced Grant no. 883810.
EH: Korteweg de Vries Instituut, Universiteit van Amsterdam, Science Park 904, 1098 XH Amsterdam, The Netherlands.
E-mail: eoin.hurley@umail.ucc.ie.


1


-----

This proved to be a fundamental result with many applications in graph theory and in the network
routing theory [FFP88]. It was used by Alon, Krivelevich and Sudakov [AKS07] to embed almost spanning
trees in sparse random graphs. The method was later refined by Haxell [Hax01] and modified by Glebov,
Johannsen and Krivelevich [GJK] before it was used by Montgomery [Mon19] to show that _G(n, C∆_ log n/n)
contains every spanning tree of maximum degree ∆with high probability, thus resolving a conjecture of Kahn.
Friedman and Pippinger’s method has also been recently used by Dragani´c, Krivelevich and Nenadov [DKN22]
in conjunction with a rolling back technique to embed a variety of very sparse graphs in expanders. It was
used even more recently, (again with rolling back) by Dragani´c, Montgomery, Munh´a Correia, Pokrovskiy and
Sudakov [DMC[+]24] to prove a long standing conjecture of Krivelevich and Sudakov stating that pseudorandom
graphs are Hamiltonian. Finally, it is a key ingredient in many bounds in size ramsey theory for bounded
degree (hyper)trees [HK95], (hyper)graphs of bounded treewidth [BKM[+]21, KLWY21, HS], and other related
families of sparse (hyper)graphs [LPY21].

Finding _induced_ subgraphs, while equally natural, is more challenging than the non-induced case and
our understanding typically lags behind. In the case of bounded degree trees and related sparse graphs, a
technical but significant reason for this gap in our understanding is the lack of any induced analogue of
Friedman and Pippenger’s powerful result. This is our main contribution.

Note that both in P´osa’s and Friedman and Pippenger’s results there is only a lower bound on order of
the vertex boundary of subsets. Thus large cliques satisfy the conditions of both theorems although they
do not contain any non-trivial induced tree. We must therefore add an upper bound on average degree of
small subgraphs, that is, we must forbid small dense spots. We also need a maximum degree condition that
is trivial to satisfy in all of our applications but we believe it to be a mere artifact of our proof (since we use
the Lov´asz Local Lemma). One final difference with P´osa’s and Friedman and Pippenger’s results is an extra
factor of ∆which we conjecture could be removed completely, but removing it will likely require some new
ideas.

**Theorem** **1.3.** _Let_ _G_ _be_ _a_ _graph_ _with_ _minimum_ _degree_ 10[7]∆ _and_ _maximum_ _degree_ _at_ _most_ exp(∆/10[9]) _such_
_that_ _every_ _subgraph_ _on_ _at_ _most_ (10[7]∆+ 1)n _vertices_ _has_ _average_ _degree_ _at_ _most_ 12/5. _Then_ _G_ _contains_ _every_
_tree_ _with_ _maximum_ _degree_ _at_ _most_ ∆ _on_ _n_ _vertices_ _as_ _an_ _induced_ _subgraph._

One drawback of Friedman and Pippenger’s theorem is that it is non-algorithmic. This was rectified by
Dellamonica and Kohayakawa [DJK08] who (under a slightly stronger and more robust expansion condition)
reduced the problem to that of finding a matching in a robustly expanding bipartite graph. They then applied
an algorithmic (also online) result of Aggarwal, Bar-Noy, Coppersmith, Ramaswami, Schieber and Sudan

[ABNC[+]96] on such matchings (the motivation of [ABNC[+]96] was to obtain algorithmic versions of Friedman
and Pippenger’s applications in the network routing literature). It is from this latter proof [ABNC[+]96] that
we take our inspiration and we call the overall strategy the “Pre-Emptive Greedy Algorithm”. Our result
yields an efficient online algorithm for finding such trees.

We anticipate that Theorem 1.3 will find many applications. We present three examples, the first two
make significant progress on two long-standing and central problems in induced Ramsey theory and in the
theory of random graphs, while the third is a near tight result on induced forests in countable (very strong)
expanders.

1.1. **Induced** **Subgraphs** **of** **Random** **Graphs.** One of the oldest problems in the theory of random graphs
is estimating the size of the largest independent set in _G(n, p)._ This value was asymptotically determined for

2


-----

0 < p < 1 constant and _n →∞_ by Grimmett and Mcdiarmid in the 70’s [GM75]. They showed that with
high probability _G(n, p)_ contains an independent set of order

2 + o(1)
(1)
log(1/(1 − _p)) [log][ n.]_

The matching upper bound follows from the first moment method. It is natural to wonder whether there is
anything special about independent sets in this regard or whether any sufficiently sparse induced graph of
the same order should also appear. Indeed, Erd˝os and Palka [EP83] showed in the early 80s that the order
of the largest induced tree in _G(n, p)_ was also given by (1). If one wants find a specific tree on that many
vertices, such as a path or regular tree, then one can turn to Ruci´nski [Ruc87] who proved the same bound
(1), provided the maximum degree of the tree is sub-polynomial in its order.

Another regime of _G(n, p),_ the so called _sparse_ regime, where _pn_ = _d_ is constant, has received a lot of
attention in recent decades. This regime presents mathematical challenges not faced in the dense case (the
classical second moment approach breaks down), and further, most applications coming from computer
science, statistical physics and mathematical modelling use graphs of constant average degree. Erd˝os and
Palka suggested the problem of extending their result to the sparse regime and conjectured that for all _d > 1_
there exists _Cd_ such that _G(n, d/n)_ contains an induced tree of order _Cd · n_ with high probability. In other
words they conjectured that there is an induced tree of linear order in _G(n, d/n)._ In a flurry of activity this
was simultaneously proven by De la Vega [dlV86], Frieze and Jackson [FJ87b], Kuˇcera and R¨odl [KR87], and
�Luczak and Palka [�LP88]. The best constant, _Cd_ = (1+ _o(1)) log d/d,_ was due to De La Vega who proved that
the greedy algorithm succeeds (with high probability) by tracking stochastic differential equations. Frieze
and Jackson [FJ87a] soon showed that one can actually find an induced path (even a cycle) of linear order,
albeit for a smaller constant _Cd_ and only if _d_ is sufficiently large. Suen [Sue92] and �Luczak [�Luc91] improved
the result on induced paths to show that as long as _d > 1,_ _G(n, d/n)_ contains an induced path of linear order
and further that for large _d_ one could take _Cd_ = (1 + o(1)) log d/d, matching De La Vega’s bound for induced
trees. Suen’s result was a particularly elegant use of the depth first search tree that avoided a lot of the
technicalities of the stochastic differential equation method.

Of course, the fact that _G(n, d/n)_ contains an independent set of linear order follows from the average
degree of the graph and a random greedy algorithm finds an independent set of order (1 + o(1))(log d/d)n
(matching the order of De le Vega’s induced tree). To this day this is the largest independent set or induced
tree one can find in _G(n, d/n)_ _efficiently._ This value appears to be an algorithmic barrier related to the
so-called shattering and freezing thresholds [COE15] and different tools are needed to go beyond it. Indeed,
the asymptotic order of the largest independent set in _G(n, d/n)_ was not determined until the brilliant insight
of Frieze [Fri90] who proved that if _d_ is sufficiently large then it is

(2 + o(1)) [log][ d] _n,_ (2)

_d_

extending (1). De La Vega, using Frieze’s result as a black box, showed that the order of the largest induced
tree is also (2). For all these exact asymptotic results, the upper bounds follow from the first moment method
and the first moment gives the same bound for any fixed graph of degeneracy at most 2, such as a matching,
a tree or a cycle. This suggests this bound may be tight for a broader class, provided, say _d/n_ _<_ _p_ _<_ _.99_
for some appropriate constant _d._ However, following the rapid progression of the seventies, eighties and
early nineties, the past three decades have been comparatively slow. Dragani´c [Dra20] extended Ruci´nski’s
results beyond the constant _p_ regime by showing that _G(n, p)_ contains any bounded degree tree of order (1),
provided _n[−][1][/][2]_ log[10][/][9] _n < p < .99._ He also conjectured that the result should apply to all trees of maximum

3


-----

degree ∆provided _d/n < p < .99_ for some _d(∆)_ (recall that (1) and (2) coincide). Cooley, Dragani´c, Kang
and Sudakov [CDKS21] showed that the order of the largest induced matchings is indeed given by (1) and
(2), and they made the same conjecure. Paths are perhaps the simplest case of the conjecture and it was
only recently that Dragani´c, Glock and Krivelevich [DGK22] showed that, so long as _d_ is sufficiently large,
_G(n, d/n)_ contains an induced path of order (2) with high probability. They further re-iterated the conjecture
of Dragani´c. But in spite of this precise conjecture, and the fact that we know asymptotically the order of
the longest induced cycle and the order of the largest induced tree, _no_ _linear_ _bound_ for a general bounded
degree induced tree in _G(n, d/n)_ was known. We remedy this.

**Theorem** **1.4.** _There_ _is_ _C_ _>_ 0, _such_ _that_ _for_ _all_ ∆ _∈_ N _and_ _d_ _>_ 2[20∆], _G(n, d/n)_ _contains_ _all_ _trees_ _with_
_maximum_ _degree_ _at_ _most_ ∆ _and_ _order_ _at_ _most_ _Cn_ _[induced]_ _[subgraphs]_ _[with]_ _[high]_ _[probability][.]_
_d log[2](d)_ _[as]_

We observe our result is essentially tight as a function of _d,_ up to a _C log[3](d)_ factor. We also note that
we could drop the lower bound on _d_ (to a linear function of ∆) provided the order of the tree is at most
_Cn/(d log(d))[2]._

1.2. **Ramsey** **Theory** **of** **Sparse** **Graphs.** One of the most famous recent results in Ramsey theory is the
Burr-Erd˝os Conjecture, proved by Lee [Lee17]. It states that for all _d_ there exists _Cd_ such that any _n-vertex_
graph with degeneracy at most _d_ has ramsey number at most _Cd_ _· n,_ in other words _graphs_ _of_ _bounded_
_degeneracy_ _have_ _linear_ _ramsey_ _numbers._ This extended the central result of Chvat´al, R¨odl, Szemer´edi and
Trotter [CRST83], who proved that bounded degree graphs have linear ramsey numbers. The original bound
from [CRST83] on _C∆_ came from the regularity lemma and was thus huge. This was greatly improved by
Graham, R¨odl and Ruci´nski [GRR00] and further by Conlon, Fox and Sudakov [CFS12] who showed one can
take _C∆_ = 2[c][∆log ∆]. The best lower bound, coming from bipartite graphs is 2[c][∆] (also due to [GRR00]) and
this is conjectured to be tight (up to the constant in the exponent) by Conlon, Fox and Sudakov [CFS15]. If
tight for all ∆(n) then this would given a nice generalisation of the upper bound of Erd˝os and Szekeres (up
to the constant in the exponent).

Two natural generalisations of Ramsey theory are size ramsey theory _rˆ_ and induced Ramsey theory _rind,_
and we further have their common generalisation, size induced ramsey theory _rˆind._

_rind(H) = min{v(G) : in_ any 2-colouring of _E(G)_ there is a monochromatic copy of _H_ that is induced in _G},_

_rˆ(H) = min{e(G) : in_ any 2-colouring of _E(G)_ there is a monochromatic copy of _H},_

_rˆind(H) = min{e(G) : in_ any 2-colouring of _E(G)_ there is a monochromatic copy of _H_ that is induced in _G}._

Note that trivially we have _r(H)_ _≤_ _rind(H), ˆr(H)_ _≤_ _rˆind(H),_ and further _rˆ(H)_ _≤_ �r(2H)� and _rˆind(H)_ _≤_
�rind2(H)�. In all three of the above cases there are trees (which have degeneracy one) that have ramsey number
superlinear in their order, thus the Burr Erd˝os conjecture does not generalise. A fundamental question then
asks:

_For_ _which_ _families_ _of_ _graphs_ _are_ _the_ _ramsey_ _numbers_ _linear_ _in_ _the_ _number_ _of_ _vertices_ ?

Let _T_ be the set of all trees and _T∆_ those of maximum degree at most ∆, and let _G∆_ and _Hd_ be the families
of graphs of degree and degeneracy at most ∆and _d_ respectively. In Table 1 we collect estimates for the
maximum ramsey number of an _n-vertex_ graph from each family.

For size ramsey numbers linear bounds for paths were proven by [Bec83]. This answered a question from
the seminal paper of Erd˝os, Faudree, Schelp and Rousseau [EFRS78] for which Erd˝os had later offered $100

4


-----

Paths _T∆_ _T_ _G∆_ _Hd_
_r(·)_ Θ(n) Θ(n) Θ(n) Θ(n) [CRST83] Θ(n) [Lee17]


Θ(n) Θ(n) _n[2]/4 ≤· ≤_ _n[3]_ log[4] _n_
_rˆ(·)_

[Bec83] [FP87] [Bec90], [Bec90]

Θ(n) _ω(n) ≤· ≤_ _n[2]_ log[2] _n_
_rind(·)_ Θ(n)

[HK�L95] [FS08], [Bec90]

Θ(n) _n[2]/4 ≤· ≤_ _n[3]_ log[4] _n_
_rˆind(·)_ Θ(n)

[HK�L95] [Bec90], [Bec90]


_cne[c][√][log][ n]_ _< · < n[O][(∆)]_

[RS00, Tik22],[CFZ14]


Θ(n[2])

[Bec90],[Lee17]

_ω(n) < · < n[O][(][d][ log][ d][)]_

[FS08],[FS08]

_n[2]/4 < · < n[O][(][d][ log][ d][)]_

[Bec90],[FS08]

|Paths|T∆|T|G∆|
|---|---|---|---|
|Θ(_n_)|Θ(_n_)|Θ(_n_)<br>|Θ(_n_) [CRST83]<br>~~_√ _~~|
|Θ(_n_)<br>[Bec83]|Θ(_n_)<br>[FP87]|_n_2_/_4_ ≤· ≤n_3 log4 _n_<br>[Bec90], [Bec90]<br>|_cnec_log_ n_ _< · < n_2_−_1_/_∆_−o_(1)<br>[RS00, Tik22], [KRSS11]|
|Θ(_n_)<br>[HK L95]|Θ(_n_)|_ω_(_n_)_ ≤· ≤n_2 log2 _n_<br>[FS08], [Bec90]<br>|_· < nO_(∆)<br>[CFZ14]<br>~~_√ _~~|


Table 1. Bounds for the maximum ramsey number of an _n-vertex_ graph from each family;
paths, bounded degree trees, trees, bounded degree graphs, bounded degeneracy graphs.
Original results are in red.

[Erd81]. With an impressive application of the the probabilistic method Beck [Bec83] also proved an upper
bound of _C∆n log[12]_ _n_ for trees of maximum degree ∆, while he conjectured that a bound of _C∆n_ should hold.
Friedman and Pippenger’s [FP87] “beautiful” result proved a linear upper bound and this was tightened
by Haxell and Kohayakawa [HK95] via a subtle anyalysis of Friedman and Pippenger’s method, resolving
Beck’s conjecture[1]. On the other hand it was shown that no such bounds are possible for general trees by
Beck [Bec90] or graphs of maximium degree 3 by R¨odl and Szemer´edi [RS00]. R¨odl and Szemer´edi further
conjectured that for all ∆there exists _ϵ_ _>_ 0 such that for all large _n_ the maximum size ramsey number
of maximum degree ∆graphs on _n_ vertices is between _n[1+][ε]_ and _n[2][−][ε]._ This upper bound was settled by
Kohayakawa, R¨odl, Schact and Szemer´edi [KRSS11] while for the lower bound the best result is due to
Tikhomirov [Tik22] who significantly improved the bound of [RS00] through a clever random twist on their
construction.

For induced ramsey numbers (and in fact size induced ramsey numbers), a linear bound for paths was
proved by Haxell, Kohayakawa and �Luczak [HK�L95]. Fox and Sudakov showed that no linear upper bound for
induced ramsey numbers of trees was possible, while remarkably the case of bounded degree graphs remains
wide open. Indeed, no non-trivial (super-linear) lower bound is known while the best upper bound due to
conlon, Fox and Zhao [CFZ14] is _n[C][∆]._ In [FS08] the authors asked if there exists a constant _C,_ independent
of ∆, such that the induced ramsey number of _n-vertex_ graphs with maximum degree at most ∆is at most a
polynomial in _n_ of degree at most _C_ (the coefficients may depend on ∆).

Of course lower bounds from size ramsey numbers yield lower bounds for induced size ramsey numbers,
thus neither bounded degree graphs nor trees satisfy linear upper bounds. As was remarked in the paper of
Bradac, Dragani´c and Sudakov [BDS23]: ”...for bounded degree trees we know that the size-Ramsey number
is linear in their number of vertices, whereas for its induced counterpart we have no good bounds while we
have every reason to believe that the answer should also be linear”. We prove the first linear bounds for
induced and size induced ramsey numbers of general bounded degree trees.

**Theorem** **1.5.** _For_ _all_ ∆ _∈_ N _there_ _exists_ _C∆_ _such_ _that_ _for_ _any_ _tree_ _T_ _of_ _maximum_ _degree_ _at_ _most_ ∆ _on_ _n_
_vertices_ _rˆind(T_ ) < C∆ _· n._ _One_ _can_ _take_ _C∆_ = 10[25]∆[3] log(∆).

Of course one can replace ˆrind by rind for free. While this paper was in preparation Hunter and Sudakov [HS]
also proved the above theorem. They proceed by very different techniques, cleverly reducing the problem to

1Beck actually made an even more specific conjecture for each tree, but we will not discuss it here.

5


-----

the non-induced case, via carefully constructed subgraphs of blowups. Their methods give worse bounds (an
exponential dependency on ∆) but one nice thing is that it extends to graphs of bounded treewidth. In order
to do this they once again reduce to the non-induced case which was proved by [BKM[+]21] (the two-colour
case was proved simultaneously by [KLWY21]). These results in turn use Friedman and Pippenger’s result as
a black box. Thus, it is very natural to ask if one can use our induced Friedman and Pippenger type result
to avoid any reduction to the non-induced case and improve the quantitative bounds due to Hunter and
Sudakov (which are very large due to the use of sparse regularity).

In fact, we prove something much stronger than the above statement. Given a family of graphs _G_ a
_Universal_ _graph_ for _G_ is a graph that contains all graphs of _G_ as subgraphs. The study of such objects goes
back at least as far as Moon [Moo65], with the central question being bounds on their size and order. The
induced question is also well studied [CG83] and now it is even known that an induced universal graph of
order _O(n)_ exists for the family of trees on _n_ vertices (even unbounded degrees) [ADK17]. The constructions
are far from random. If in any colouring of our graph _G_ we have a monochromatic (induced) universal graph
then we say that our graph is (induced) partition universal. An induced such result (also going by the name
adjacency labelling) for bounded degree graphs is what was actually proved in [KRSS11]. Of course a density
universal result is stronger again, that is, a result that says: in any subgraph of density _ε > 0_ one finds all of
said subgraphs. This is what we have proven; a density universal theorem for bounded degree trees of order
_n_ with an upper bound _on_ _the_ _number_ _of_ _edges_ that is linear in _n._

**Theorem** **1.6.** _For_ _all_ ∆, n ∈ N _and_ _ε > 0_ _there_ _exists_ _a_ _graph_ _G_ _with_ _less_ _than_ _C(∆, ε) · n_ _edges_ _such_ _that_
_any_ _subgraph_ _J_ _⊂_ _G_ _containing_ _ε · e(G)_ _edges_ _contains_ _every_ _tree_ _of_ _maximum_ _degree_ ∆ _and_ _order_ _at_ _most_ _n_
_as_ _an_ _induced_ _subgraph_ _of_ _G._ _One_ _can_ _take_ _C(∆, ε) =_ �10[42]∆[3] log(∆) log( [1]ε [)][3][�]/ε[2].

We remark that Butler [But09] showed that there is no induced universal graph of linear order for bounded
degree graphs, thus one cannot replace bounded degree trees by general bounded degree graphs in the above
theorem, even if one sets _ε = 1_ and replaces the bound on the number of edges by the same bound on the
order. It was also shown by Chung and Graham [CG83], that even with _ε = 1,_ if we remove the bounded
degree condition the theorem above does not work[2]. In order to see the strength of the constant, note that
this implies state of the art bounds even in the case of the multicolour size ramsey number of induced paths
(see Theorem 1.7 and discussion).

1.3. **Dense** **Induced** **Forests** **in** **Countable** **Sparse** **Expanders.** Friedman and Pippenger’s result also
applies to countable graphs, where it is best phrased in terms of Cheeger’s constant. For a countable graph
_G_ we define the vertex Cheeger constant as


�
_,_


_hv(G) := inf_
_X_


� _|N_ (X)\X|
_|X|_


where the infimum is over all finite sets _X_ _⊂_ _V (G)._ Friedman and Pippenger’s result (because it is online)
implies that if _d_ _>_ 3 and _h(G)_ _≥_ _d_ then _G_ contains a tree with _h(T_ ) _≥_ _d −_ 2. Benjamini and Schramm

[BS97] proved the following stronger theorem about _spanning_ forests, which is actually a characterisation in
the case of graphs with integer Cheeger constant.

**Theorem** **1.7** (Benjamini-Schramm [BS97]). _Suppose_ _d ≥_ 0 _is_ _an_ _integer_ _and_ _G_ _is_ _a_ _graph_ _with_ _h(G) ≥_ _d._
_Then_ _G_ _has_ _a_ _rooted_ _spanning_ _forest_ _in_ _which_ _the_ _roots_ _have_ _degree_ _d_ _and_ _all_ _other_ _vertices_ _have_ _degree_ _d + 2._

2Further, a minor adjustment to the proof allows one to prove the above theorem with a larger constant, for bounded degree

forests (instead of trees), see the discussion of rolling back in the conclusion.

6


-----

One can also define the Cheeger constant in terms of edge boundaries (this is perhaps the more classical
quanitity),

� _e(X, G\X)_ �

_h(G) := inf_ _,_
_X_ _|X|_

where again the infimum is over finite sets _X_ _⊂_ _V (G)._ The use of the edge boundary is essentially forced
on you if you are looking for induced structures. For regular graphs _G_ this Cheeger constant is dual to the
supremum of average degree over all finite subgraphs. Our technique extends, as Friedman and Pippenger’s
did, to finding an induced tree in countable graphs and in fact it immediately yields a spanning version,
in the spirit of Benjamini and Schramm. The attentive reader will correctly object that there is only one
induced spanning subgraph of a graph. Our result is not spanning but is _as_ _spanning_ _as_ _possible_ (given the
degrees of the induced forest we embed). A _pseudoforest_ is a graph in which every component contains at
most one cycle. A ∆-ary pseudoforest is a pseudoforest in which all acyclic components are ∆-ary trees and
all other components are (∆+ 1)-regular. A subgraph is _component-wise_ _induced_ if each component is an
induced subgraph.

**Theorem** **1.8.** _There_ _exists_ _ε > 0_ _such_ _that_ _if_ _G_ _is_ _a_ _d-regular_ _graph_ _with_ _h(G) > d −_ 3 + 10[7]∆+11 _[for]_ _[some]_
∆ _< εd,_ _then_ _G_ _contains_ _a_ _spanning_ ∆-ary _pseudoforest_ _F,_ _that_ _is_ _component-wise_ _induced,_ _with_ _the_ _property_
_that_ _one_ _can_ _turn_ _F_ _into_ _an_ _induced_ _forest_ _by_ _deleting_ _one_ _vertex_ _from_ _each_ _of_ _its_ _components._

There is only one _d-regular_ graph with _h(G) > d −_ 2, and that is the _d-regular_ tree, and if _h(G) = d −_ 2
then _G_ is a pseudoforest. Our result says that we can weaken this condition and still find a ”dense” family of
induced trees. The induced forest intersects all neighbourhoods in at least ∆vertices which is essentially
best possible. Further, one cannot weaken the bound on Cheeger’s constant by much (even without the
component-wise induced condition), as is witnessed by blowing up the edges of the _d/2-regular_ tree by _K2,2’s_
(one replaces the vertices with independent sets). This graph has _h(G) = d −_ 4 but does not even contain a
binary spanning pseudoforest with the properties described in Theorem 2.18 [3].

**Notation.** Let _N_ (v) denote the open neighbourhood of _v_ and let _N_ [v] := N (v) ∪ _v_ be the closed. For a set
_X_ _⊂_ _V (G)_ we let _N_ (X) := ∪v∈X _N_ (v) and _N_ [X] = X _∪v∈X_ _N_ (v).

2. Proof Overview

Our approach is inspired by the proof of Aggarwal, Bar-Noy, Coppersmith, Ramaswami, Schieber and
Sudan [ABNC[+]96], who showed that a certain online matching game was winnable in robustly expanding
bipartite graphs. The online matching game goes as follows. We have a graph _H_ = (A, B) and _X_ _⊂_ _A_
matched to a set _Y_ _⊂_ _B_ (initially these will be empty). The adversary picks an element _x ∈_ _A\X_ and we
have to choose an element _y_ _∈_ _B\Y_ and match _X_ _∪_ _x_ to _Y_ _∪_ _y._

**Theorem** **2.1.** _Let_ _H_ = (A, B) _be_ _a_ _bipartite_ _subgraph._ _Suppose_ _that_ _for_ _all_ _subgraphs_ _F_ _⊂_ _H_ _with_
degF (v) > degH (v)/2 _for_ _all_ _v_ _∈_ _V (H)_ _we_ _have_ _|NF (X)| > 2|X|_ _for_ _all_ _X_ _⊂_ _A_ _with_ _|X| ≤_ _n._ _Then_ _there_ _is_
_a_ _polynomial_ _time_ _algorithm_ _to_ _find_ _an_ _online_ _matching_ _of_ _order_ _n_ _in_ _A._

3To see this we will refer to the vertices that are to be deleted in the statement of Theorem 2.18 as roots of _F_ . If there is
any pair _u_ and _v_ of twin vertices with _v_ not a root then _v_ must be in the same component of _F_ as _u_ (deleting _u_ leaves an
induced tree containing neighbours of _u)._ Call this component _C._ By the inducedness of _T_ = C _−_ _r_ (where _r_ is the root of _C)_

all neighbours of _u_ in the tree are neighbours of _v_ in the tree. It follows that there is at least one 4-cycle in _C_ containing _u_ and
_v,_ but then the component must have been 3-regular (by the definition of a binary pseudoforest) in which case there are three
4-cycles, a contradiction. Thus all pairs of twins are both roots, and so are all vertices, which is clearly impossible.

7


-----

We call the high-level strategy used for Theorem 2.1 and in this paper the Pre-emptive Greedy Algorithm.
It was applied to the problem of efficiently finding linear order bounded degree trees in expanders by [DJK08].
While [DJK08] used Theorem 2.1 as a black box, we describe the high level strategy if one were to open
the box and run the argument in the case of bounded degree trees (not necessarily induced). The high level
strategy is the same in our case. The game is as follows.

**The** **Game.** We wish to embed a bounded degree tree _T_ in a graph _G._ The graph _G_ is given to us, but
_T_ is chosen by the adversary one vertex at a time. We start from the empty tree _T_, and in each round
the adversary adds one vertex to _T_, to obtain _T_ _[′],_ maintaining that _T_ _[′]_ has both maximum degree at most
∆and is a tree. In each round we find an embedding of _T_ _[′]_ in _G_ that extends our earlier embedding of _T_ .
Formally we find an injective homomorphism _ϕ[′](T_ _[′])_ such that the restriction of _ϕ[′]_ to _T_ is simply _ϕ,_ the
homomorphism we had from the previous round. Then in each round our adversary asks us to extend our
current tree (embedded in _G)_ from a vertex of degree at most ∆ _−_ 1 (in the embedded tree _T_ ). We lose when
we cannot extend, we win if we play _n_ rounds without losing (and thus embed an _n-vertex_ tree).

We now _informally_ describe the strategy, leaving all formal definitions to later sections.

**The** **Pre-Emptive** **Greedy** **Algorithm.** Let us begin na¨ıvely. If one were to greedily embed the tree, one
could, after some short amount of time, end up in the following situation. The adversary asks you to extend
the tree from a vertex _v,_ but all of _v’s_ neighbours are already in the tree. This prevents you from extending
your tree without creating a cycle, so you lose. You must _pre-empt_ this situation. One thing you could try
would be to watch all the vertices in the graph each time you extend your tree. If a vertex _v_ has too many
neighbours in the tree i.e. it is _critical,_ then one immediately _reserves_ some neighbours of _v_ that are not in
the tree. You _reserve_ these vertices for the eventuality that at some point in the future you are asked to
extend your tree from _v._ You will not use the vertices _unless_ you are extending from _v._ Note that _v_ is not
necessarily in your tree, and it may never be.[4] But if you are at _v_ then you can _escape_ using the reserved
vertices.

Of course, there is an issue here, there is a risk of a _criticality_ _cascade._ That is, when we reserve vertices
for _v_ we may take neighbours away from a different almost critical vertex _u,_ forcing us to reserve vertices for
_u._ But doing that may mean that we make another vertex _w_ critical and so on, raising two issues.

(1) Perhaps the criticality cascade consumes the whole graph (i.e. it makes every vertex critical).
(2) Perhaps in reserving vertices for _u_ and _v_ we have used all the neighbours of some other vertex _w._

Thus we must be more clever. We require expansion properties of _G_ to show that criticality cascades cannot
be much larger than the current tree, this prevents problem (1). For the second problem, rather than fixing
critical vertices one by one, each time we extend our _tree_ (not including reserving vertices) we find all the
vertices that _might_ (we can’t predict the future, but we can bound it) be caught in a criticality cascade and
call this set _C_ (this is not too big!). Because they are only _at_ _risk_ of becoming critical, the vertices in _C_ are
not critical yet and still have many neighbours that are not in the tree. If we require many more neighbours
than ∆to be available, then perhaps we can find a way to reserve ∆ _−_ 1 neighbours for each vertex in _C_
_simultaneously._ Aggarwal, Bar-Noy, Coppersmith, Ramaswami, Schieber and Sudan used a robust expansion
condition, Hall’s Theorem and augmenting paths to show these remedies work in the case of online matchings
(the case of trees reduces to that case). None of those tools work in the case of induced trees however the
high level strategy is similar.

4If one wants to tighten Theorem 1.3 by reducing the factor of ∆, then this wastefulness is one place to start.

8


-----

(1) Extend the tree greedily until some set _C_ of vertices is at risk of being caught in a _criticality_ _cascade._
(2) Simultaneously _reserve_ neighours for each of the vertices in _C._
(3) Consider these _reserved_ vertices as part of the extended “tree” and repeat.

To summarise our earlier analysis, in order for this algorithm to work it suffices that:

(1) Criticality cascades cannot be much larger than the set that they start from (see Lemma 2.9).
(2) If _C_ is not too large and each vertex in _C_ has enough neighbours not in the tree, then we can
_simultaneously_ reserve ∆ _−_ 1 vertices for each vertex of _C_ (see Lemma 2.13).

_Remark._ In the above algorithm, we watch every vertex all the time, and actually ensure that we can always
extend from any vertex that has not been extended ∆times yet. Thus we can actually build a spanning
forest, this is what allows Theorem 2.18.

As previously mentioned none of the arguments that were used in the non-induced case work in the induced
case. Thus we must introduce some tools. These will allow us to prove Lemmas 2.9 and 2.13, before proving
the main theorem.

2.1. **Formal** **Machinery.** We begin by introducing directed graphs, because directing an edge from _u_ to _v_
will be useful for encoding that we have reserved _u_ for _v._ An oriented subgraph _D_ of a simple graph _G_ is a
subgraph _H_ _⊂_ _G_ along with an orientation for each of its edges _E(H)._ We view _D_ as a digraph living on the
same vertex set as _G._ For a vertex _v_ _∈_ _V (G),_ we define the _in-neighbours_ of _v_ as _ND[−][(][v][)][ :][=][ {][u][ :][ uv]_ _[∈]_ _[E][(][D][)][}][.]_
The following two propositions will allow us to extend our tree or pseudoforest while only focusing on local
information. In spite of their simplicity they are crucial to our proof.

**Proposition** **2.2.** _A_ _connected_ _graph_ _T_ _has_ _an_ _orientation_ _such_ _that_ _every_ _vertex_ _has_ _in-degree_ _at_ _most_ 1
_and_ _at_ _least_ _one_ _vertex_ _has_ _in-degree_ 0, _if_ _and_ _only_ _if_ _T_ _is_ _a_ _tree._

_Proof._ The if direction is straightforward because if _T_ is a tree then we can choose a root abitrarily and
orient all edges away from the root.

For the only if direction, let _r_ be a vertex of in-degree 0 and observe that due to the connectedness of _T_
we have a path _Pru_ from _r_ to _u_ for any _u ∈_ _V (T_ )\r. Because _r_ has in-degree 0, the edge _rw_ incident to _r_ in
_Pru_ must be oriented away from _r._ Because every other vertex has in-degree at most 1 we see that all edges
in _Pru_ are oriented away from _r._ Now suppose, for contradiction that there is a cycle _C_ in _T_, and let _u_ be a
vertex in the cycle with shortest distance to _r_ (in the undirected sense). By choosing appropriate paths from
_r_ via _u_ to other vertices in _C,_ we see that every path within _C,_ with end-vertex _u,_ must be oriented away
from _u._ But this is impossible in a cycle. 
**Proposition** **2.3.** _A_ _graph_ _F_ _has_ _an_ _orientation_ _such_ _that_ _every_ _vertex_ _has_ _in-degree_ _at_ _most_ 1 _if_ _and_ _only_ _if_
_F_ _is_ _a_ _pseudo-forest._

_Proof._ For the if direction we orient each component individually. If a component _C_ is a tree then we are
done by Proposition 2.2 and if it contains one cycle then we cyclically orient the cycle and orient all other
edges away from said cycle.

For the only if direction fix such an orientation and consider a component _C_ of _F_ . If _C_ has a vertex of
in-degree 0 then we are done by the previous proposition. Otherwise assume, aiming for a contradiction that
_C_ has two distinct cycles _S_ and _S[′]._ Clearly both cycles must be cyclically oriented. Thus each vertex in _S_
has an in-edge that is in _E(S)._ The same is true for _S[′]_ and it follows that all edges incident to _S_ and _S[′]_ that

9


-----

are not in _E(S)_ or _E(S[′])_ respectively, are oriented away from _S_ or _S[′]_ respectively. In particular this means
that the cycles cannot intersect in any path (including the path that is just a vertex), because all edges of _S[′]_

with exactly one endvertex in a maximal path _P_ in the intersection would both be oriented away from _P_,
implying _S[′]_ is not cyclically oriented. Therefore by the connectedness of _C_ there exists a path with at least
one edge from _S_ to _S[′]_ and so it must start oriented away from _S_ and finish oriented away from _S[′],_ implying
that there is a vertex of in-degree 2 on the path. 
Of course we are interested not only in trees, but in induced trees. Thus we define the following object
which combines the local-witness properties of Propositions 2.2 and 2.3 with a subtle inducedness condition.
The next definition is key to our proof. Let _G_ be a graph and _D_ a (bi-)oriented subgraph of _G._ We define
_Vin(D) and Vout(D) to be the vertices with at least one in-neighbour or at least one out-neighbour respectively._

**Definition** **2.4.** _Given_ _a_ _graph_ _G,_ _an_ _escape-way_ _D_ _is_ _an_ _oriented_ _subgraph_ _of_ _G_ _satisfying_ _that_ _the_ _in-degree_
(in _D)_ _of_ _each_ _vertex_ _of_ _G_ _is_ _at_ _most_ 1 _and_ _such_ _that_ _if_ _x, y_ _∈_ _Vin(D)_ _and_ _xy_ _∈_ _E(G)_ _then_ _exactly_ _one_ _of_ _xy_
_or_ _yx_ _is_ _in_ _E(D)._ _The_ _latter_ _condition_ _implies_ _that_ _Vin(D)_ _induces_ _a_ _subgraph_ _of_ _D_ _in_ _G._

For example, an induced forest, oriented away from its roots is an escape-way. An induced cycle, oriented
to be a directed cycle, is an escape-way. Instructively, the following example is also an escape-way. Take an
induced forest _T1, . . ., Tt_ and orient each tree away from some root _r1, . . ., rt_ (ri _∈_ _Ti)._ Now add to _E(G)_
(but not to _D),_ a clique containing _r1, . . ., rt._ See the discussion following Corolary 2.10 for more motivation.
Note that subgraphs of escape-ways are escape-ways and if _D_ is an escape-way and _u ∈_ _ND[+][(][v][),]_ [we] [say] [that]
_D_ _reserves_ _u_ _for_ _v._

In our pre-emptive greedy algorithm we are interested in extending our escape-way chunk by chunk. We
cannot use Hall’s Theorem or the augmenting path properties that [ABNC[+]96] used. Thus we need to
understand when two escape-ways are compatible. While Proposition 2.2 showed that certain components of
escape-ways are trees, we also require inducedness. Therefore it is tempting to say that if _u_ is reserved for
_v,_ then no neighbours of _u_ are allowed to join the tree. This is almost correct, but its flaw is fatal; if we
block all neighbours of _u_ then we can never extend from _u!_ The correct rule is that no neighbours of _u_ are
allowed to join the tree, _unless_ _we_ _are_ _extending_ _from_ _u._ For a vertex _v_ _∈_ _V (G),_ we define the set of _available_
_neighbours_ of _v_ as _AD(v) := NG(v)\(Vin(D −_ _v) ∪_ _ND[−][(][v][)).]_ [Note] [that] _[A][D][(][v][)][ ⊂]_ _[N][G][(][v][).]_ [A] [neighbour] _[u]_ [of] _[v]_ [in]
_G_ is therefore not available if either _u_ has an in-neighbour that is not _v_ _or_ _u_ is an in-neighbour of _v._

**Definition** **2.5.** _Suppose_ _D_ _is_ _a_ (bi-)oriented _subgraph_ _of_ _G._ _We_ _define_ _K(D)_ _to_ _be_ _the_ (bi-)oriented _subgraph_
_of_ _G,_ _obtained_ _by_ _starting_ _from_ _D_ _and_ _adding,_ _for_ _all_ _edges_ _uv_ _∈_ _E(D),_ _all_ _edges_ _vw_ _for_ _w_ _∈_ _NG(v) \ u._

Observe that if _D_ is an escape-way then _K(D)_ contains no bi-oriented edges. Further it is clear that if
_D_ _⊂_ _D[′]_ then _K(D) ⊂_ _K(D[′])._ We say an escape-way _D_ in _G_ _agrees_ with a (bi-)oriented subgraph _H_ of _G_ if
the following hold:

_•_ For all _x ∈_ _Vin(H) ∩_ _Vin(D)_ we have _NH[−][(][x][) =][ N]D[ −][(][x][),]_ [and]

_•_ For all _xy_ _∈_ _E(H),_ _yx ̸∈_ _E(D)._

**Proposition** **2.6.** _Given_ _escape-ways_ _D_ _and_ _D[′]_ _in_ _G_ _the_ _following_ _are_ _equivalent:_

(1) _D ∪_ _D[′]_ _is_ _an_ _escape-way_ _in_ _G,_
(2) _D[′]_ _agrees_ _with_ _D,_
(3) _For_ _each_ _x ∈_ _V (G)_ _we_ _have_ _ND[+][′][(][x][)][ ⊂]_ _[A][K][(][D][)][(][x][).]_

10


-----

_Proof._ First we show that (1) implies (2) by showing the contrapositive. There are two reasons (2) may not
hold. If there exists _x_ _∈_ _Vin(D[′]) ∩_ _Vin(D)_ but _NH[−][(][x][)]_ _[̸][=]_ _[N]D[ −][(][x][),]_ [then] _[x]_ [has] [in-degree] [at] [least] [2] [in] _[D][ ∪]_ _[D][′]_

meaning that _D ∪_ _D[′]_ is not an escape-way. If _xy_ _∈_ _E(D)_ and _yx ∈_ _E(D[′])_ then _D ∪_ _D[′]_ has bi-oriented edges,
meaning it is not an escape-way.

We also show that (2) implies (3) via the contrapositive. Suppose there exists y _∈_ _ND[+][′]_ [(][x][) and][ y] _[̸∈]_ _[A][K][(][D][)][(][x][).]_
There could be two reasons for _y_ _̸∈_ _AK(D)(x)._ Suppose first _yx_ _∈_ _D._ Then, _xy_ _∈_ _E(D[′])_ and _yx_ _∈_ _E(D),_
hence _D[′]_ does not agrees with _D._ If on the other hand _zy_ _∈_ _E(D)_ for some _z_ _̸= x_ then _y_ _∈_ _Vin(D[′]) ∩_ _Vin(D)_
but _NH[−][(][x][)][ ̸][=][ N]D[ −][(][x][).]_

Finally we show that (3) implies (1), again via the contrapositive. As _D_ and _D[′]_ are escape-ways there
are only two reasons that _D ∪_ _D[′]_ would not be an escape-way. The first is that some vertex _y_ has in-degree
at least 2 in _D ∪_ _D[′]._ But then we must have two distinct edges _xy_ _∈_ _D_ and _zy_ _∈_ _D[′]_ which would imply
that _y_ _∈_ _Vin(K(D) −_ _z)_ and so _y_ _̸∈_ _AK(D)(z)._ The second is that there are _x ∈_ _Vin(D)_ and _y_ _∈_ _Vin(D[′])_ with
_xy_ _∈_ _G_ but _xy, yx ̸∈_ _E(D) ∪_ _E(D[′])._ In this case _xy_ _∈_ _E(K(D))_ and so _y_ _̸∈_ _AK(D)(z)_ for any _z_ _̸= x._

                              
Having established these technical properties we prepare to prove that criticality cascades cannot be too
large. We will use the following bootstrap percolation process to capture all vertices that might be caught in
a criticality cascade from _X._

**Definition** **2.7.** _Given_ _a_ _set_ _of_ _vertices_ _X_ _in_ _a_ _graph_ _G,_ _we_ _define_ _C(X) ⊂_ _V (G),_ _the_ _d-critical_ set of _X_ _as_
_the_ _terminal_ _set_ _of_ _the_ _following_ _boostrap_ _percolation_ _process._ _We_ _let_ _X0_ := X _and_ _given_ _Xi_ _we_ _let_ _Xi+1_ _be_
_the_ _union_ _of_ _Xi_ _with_ _all_ _vertices_ _v_ _that_ _have_ _at_ _least_ _d_ _neighbours_ _within_ _distance_ _at_ _most_ 2 _from_ _Xi_ _in_ _G\v_
(where _x ∈_ _Xi_ _has_ _distance_ 0).

It follows immediately from the definition that if _X_ _⊂_ _X_ _[′]_ then the _d-critical_ set of _X_ _[′]_ contains that of
_X._ Further, the (d − 1)-critical set of _X_ contains the _d-critical_ set. The reason this definition is useful is
because of its relationship to escape-ways. Essentially it provides an upper bounds on how much the available
neighbourhood can be reduced by fixing certain escape-ways.

**Proposition** **2.8.** _Let_ _X_ _be_ _a_ _set_ _of_ _vertices_ _in_ _a_ _graph_ _G,_ _and_ _let_ _C(X)_ _be_ _the_ _d-critical_ _set_ _of_ _X._ _Suppose_
_B_ _is_ _an_ _escape-way_ _with_ _Vout(B) ⊂_ _C(X)._ _Then_ _|AK(B)(v)| ≥_ degG(v) − _d_ _for_ _all_ _vertices_ _v_ _̸∈_ _C(X)._

_Proof._ Let _v_ _̸∈_ _C(X)._ The crucial fact is that if _u ∈_ _N_ (v) has distance at least 3 from _Vout(B)_ in _G\v,_ then
_u ∈_ _AK(B)(v)._ By the definition of _C(X)_ if _v_ _̸∈_ _C(X)_ then at most _d −_ 1 neighbours of _v_ have distance less
than 3 from _X._ It follows that at most that many have distance less than 3 from _Vout(B)_ in _G\v._ 
The following straightforward Lemma shows that large criticality cascades generate sets that are noticeably
denser than trees or cycles. This is useful because it implies they cannot be too large in graphs with no dense
spots (see proof of main theorem).

**Lemma** **2.9.** _Let_ _X_ _be_ _a_ _finite_ _set_ _of_ _vertices_ _in_ _a_ _graph_ _G,_ _and_ _let_ _C(X)_ _be_ _the_ _d-critical_ _set_ _of_ _X._

_•_ _If_ _G[X]_ _is_ _connected_ _and_ _|C(X)|_ _≥_ 2|X| _then_ _there_ _exists_ _a_ _graph_ _H_ _⊂_ _G_ _on_ _at_ _most_ (2d + 2)|X|
_vertices_ _with_ _average_ _degree_ _at_ _least_ 2 + 2[d]d[−]+2[2] [.]

_•_ _If_ _C(X)_ _is_ _unbounded_ _then_ _there_ _exist_ _a_ _sequence_ _of_ _finite_ _graphs_ _H_ _⊂_ _G_ _with_ _average_ _degree_
_approaching_ 3(1 − _d+11_ [).]

11


-----

_Proof._ We analyse the bootstrap percolation that generates _C(X)_ from _X,_ with the added assumption that
the vertices are added one by one. We break ties by a global ordering of the vertices. This does not affect
the terminal set which we call _C_ := _C(X)._ We start from _X_ and let _v1_ be the first vertex added and so
on. By definition _vi+1_ was added because it has at least _d_ neighbours who are at distance at most 2 from
_X_ _∪{v1, . . ., vi}._

For the first case let _C_ _[′]_ = _X_ _∪{v1, . . ., v|X|}._ This exists by assumption. Let the graph _H_ _[′]_ _⊂_ _G_ be a
minimal spanning subgraph such that the boostrap percolation w.r.t. _H_ _[′]_ starting at _X_ reaches all of _C_ _[′]_ and
in the same order as above. We define four groups of vertices in H. The first two are natural, the initial set X
and the newly critical _C_ _[′]\X._ Now for each _v_ _∈_ _C_ _[′]\X,_ we know that there at least _d_ neighbours that caused
_v_ to be in _C_ _[′]._ Choose _d_ of these arbitrarily and call this set _Yv._ We then let _Y_ be the union of all these sets
over _v_ _∈_ _C_ _[′]\X._ Note that _Y_ may intersect both _X_ and _C_ _[′]\X._ Finally we define _Z_ := V (H _[′])\(C_ _[′]_ _∪_ _Y )._ The
role of the vertices in _Z_ (by process of elimination, and the minimality of _H_ _[′])_ is to guarantee that the vertices
in _Y_ are at distance less than 2 from the earlier stages of the percolation. We now construct a surjective
map _f_ : Y _→_ _Z_ in order to show _|Z| ≤|Y |._ We assume _Z_ is non-empty as the inequality is trivial otherwise.
For _w_ _∈_ _Y_ let _j_ be the smallest index such that _w_ _∈_ _Yvj_ . Let _f_ (w) be an arbitrary vertex in _Z_ on a path of
length 2 between _w_ and _X ∪{v1, . . ., vj−1}_ in _H_ _[′]._ If no such vertex exists let _f_ (w) be arbitrary in _Z._ By the
minimality of _H_ _[′]_ this map is a surjection. Indeed otherwise there would exist a vertex _u ∈_ _Z_ whose deletion
does not affect the bootstrap percolation rule. As _Y_ has order at most _d(|C_ _[′]| −|X|)_ this implies that the
order of _H_ _[′]_ is at most _|C_ _[′]| + (2d)(|C_ _[′]| −|X|) = (2d + 2)|X|._

We now iteratively delete leaves (degree one vertices) of _H_ _[′]_ until we are left with a graph _H_ that has no
leaves. Observe that for each _v_ _∈_ _C_ _[′]\X,_ every pair of edges incident to _v_ in _H_ _[′]_ lie on a common cycle in _H_ _[′]._
This is because they both lie on paths from _v_ to _X_ and _G[X]_ is connected. Therefore every edge incident to
_v_ in _H_ _[′]_ is also in _H._ This allows us to lower bound the degree of every vertex in _H_ by 2 and every vertex in
_C_ _[′]\X_ (in _H)_ by _d._ By summing up the degrees in _H_ we obtain a lower bound on the average degree. In
particular

� deg(v)
_≥_ _[d][(][|][C]_ _[′][| −|][X][|][) + 2(][v][(][H][)][ −]_ [(][|][C] _[′][| −|][X][|][))]_ = [2][v][(][H][) + (][d][ −] [2)][|][X][|] (3)
_v(H)_ _v(H)_ _v(H)_

= 2 + _[|][X][|][(][d][ −]_ [2)] _≥_ 2 + _[|][X][|][(][d][ −]_ [2)] _[d][ −]_ [2] (4)

_v(H)_ (2d + 2)|X| [= 2 +] 2d + 2 _[,]_


where in the second step we used that _|C_ _[′]| −|X| = |X|._ Thus _H_ is the desired graph.

For the unbounded case, we can be more wasteful. We construct a sequence of graphs _H0, H1, . . ._ with
_H0_ = G[X] inductively as follows. Given _Hi_ if _vi+1_ _∈_ _Hi_ then let _Hi+1_ = Hi. Otherwise do the following.
By the definition of _vi+1_ it has at least _d_ neighbours _u1, . . ., ud_ that are at distance 0, 1 or 2 from _Hi._
We construct _Hi+1_ in _j_ steps _Hi_ = _H0_ _∪_ _vi+1, . . ., Hi[d]_ [=] _[H][i][+1][,]_ [again] [inductively.] [For] _[j]_ _[∈]_ [[][d][]] [note] [that]
dist(uj, Hi[j][−][1]) _≤_ 2 by assumption. Add a path of length _dist(uj, Hi[j][−][1])_ to _Hi[j][−][1]_ to obtain _Hi[j][.]_ [Thus] [we]
have added _dist(uj, Hi[j][−][1])_ vertices and _dist(uj, Hi[j][−][1]) + 1_ edges. Letting _M_ denote the sum over _j_ of the
aforementioned distances we have that to get from _Hi_ to _Hi+1_ we have added _M_ + 1 vertices and _M_ + d
edges. Thus the ratio (M + d)/(M + 1) of edges added to vertices added is minimised when _M_ is maximised.
_M_ is bounded above by 2d and so we obtain that the ratio of edges added to vertices added is at least


�
_._


3d [3]
2d + 1 _[>]_ 2


� 1
1 −
_d + 1_


Thus if _C(X)_ is unbounded then the average degree of the _Hi’s_ approaches 3d/(d + 1). 
12


-----

Our final lemma before we prove the main theorem states that in graphs of small degeneracy we can find
escape-ways that reserve a large portion of each vertex’ neighbourhood for each vertex. In fact it has a few
more bells and whistles than that so we first state a corollary that captures its essence.

**Corollary** **2.10.** _Let_ _G_ _be_ _a_ _graph_ _with_ _maximum_ _degree_ ∆ _such_ _that_ _every_ _subgraph_ _of_ _G_ _has_ _average_ _degree_
_at_ _most_ 3. _There_ _exists_ _an_ _escape-way_ _D_ _in_ _G_ _with_ deg[out]D [(][v][)][ ≥] _[|][ deg(]10[7][v][)][|]_ _−_ 5 log ∆ _for_ _all_ _v_ _∈_ _V (G)._

This result sheds light on the delicate definition of an escape-way. We have already seen in Propositions
2.2 and 2.3 that the definition is sufficient to encode trees and pseudoforests. We now show that very slight
alterations to the definition of an escape-way make Corollary 2.10 impossible.

First suppose we required _Vin(D) ∪_ _Vout(D)_ (instead of just _Vin(D))_ to induce a subgraph of _D_ in _G._ We
construct _G_ as follows. Consider a cycle _C_ of length _ℓ_ with a single chord and add _d_ pendant edges to each
vertex of the cycle. Then _G_ has maximum degree _d + 3_ and every subgraph has average degree less than 3. If
_d_ is large enough then every vertex of the cycle must be in _Vout(D)_ for any _D_ as in Corollary 2.10. But if
_Vin(D) ∪_ _Vout(D)_ induced a subgraph of _D_ in _G_ then all of the edges in _C_ as well as the chord are in _D._ But
orienting these edges will create a vertex of in-degree 2, which contradicts that _D_ is an escape-way.

Secondly suppose that instead of requiring that _Vin(D)_ induces a subgraph of _D_ in _G_ we required that our
escape-ways are unions of disjoint stars (oriented away from the root). Let _G_ be the _d-ary_ tree of depth 3.
Thus _G_ has max degree _d + 1_ and all subgraphs have average degree less than 2. If _d_ is large then root _r_
must have at least one out-neighbour, say _v._ But then if _v_ also has an out-neighbour _Vin(D)_ the stars (one
rooted _r_ and one at _v)_ are not disjoint. Thus if _d_ is large, there is no _D_ satisfying Corollary 2.10.

The maximum degree condition in Corollary 2.10 is due to our use of the Lov´asz Local Lemma.

**Theorem** **2.11** (Lov´asz Local Lemma). _Suppose_ _there_ _are_ _a_ _set_ _of_ _events_ _such_ _that_ _each_ _event_ _is_ _mutually_
_independent_ _of_ _all_ _but_ _d_ _other_ _events._ _If_ _each_ _event_ _occurs_ _with_ _probability_ _less_ _than_ 1/ed _then_ _the_ _probability_
_that_ _none_ _of_ _the_ _events_ _occur_ _is_ _positive._

We will also require McDiarmid’s Inequality for Lipschitz functions of independent random variables.

**Lemma** **2.12.** _Suppose_ _X1, . . ., Xm_ _are_ _independent_ _binary_ _random_ _variables_ _and_ _f_ : _{0, 1}[m]_ _→R_ _is_ _an_
_s-Lipschitz_ _function._ _Then_ _for_ _any_ _t > 0_


�
P[E[X] − _X_ _> t] < exp_ _−_ [2][t][2]

_ms[2]_


�
_._


We now state the lemma in full detail. Given a subgraph _G[′]_ _⊂_ _G_ we denote by _A[G]D[′]_ [(][v][)][ :][=][ A][D][(][v][)][ ∩] _[N][G][′]_ [(][v][),]
the available neighbours of _v_ in _G_ with respect to _D,_ that are also neighbours of _v_ in _G[′]._

**Lemma** **2.13.** _Suppose_ _F_ _is_ _an_ _undirected_ _graph_ _with_ _maximum_ _degree_ ∆ _such_ _that_ _all_ _subgraphs_ _have_ _average_
_degree_ _at_ _most_ 3. _Further_ _suppose_ _F_ _has_ _a_ _spanning_ _subgraph_ _G_ _and_ _an_ _oriented_ _subgraph_ _H._ _Then_ _F_ _has_ _an_
_escape-way_ _D_ _that_ _agrees_ _with_ _H,_ _such_ _that_ deg[out]D [(][v][)][ ≥] _[|][A]10H[G]_ [(][7][v][)][|] _−_ 5 log ∆ _for_ _all_ _v_ _∈_ _V ._

The reason for two graphs _F_ and _G_ is to prepare for our ramsey theoretic applications (G will be a large
monochromatic subgraph). We cannot just forget about _F_ because in induced ramsey theory one wants a
monochromatic subgraph that is induced in the original graph. For the oriented subgraph _H,_ the motivation
is to encode the contraint that the escape-way _D_ that we seek should combine with an escape-way _D[′]_ that
we already have, to give a bigger escape-way (see proof of main theorem).

13


-----

_Proof._ The idea is to use the average degree condition to bound the degeneracy by C and find an orientation of
_F_ in which all but _C_ of the available neighbours of each vertex are out-neighbours. We then sample a random
subgraph of this orientation and carefully resolve clashes (keeping the earlier edge in the degeneracy ordering)
to turn it into an escape-way called **D.** In expectation this will leave everyone with a large out-neighbourhood
and we complete the proof by showing concentration (once again using the average degree condition) and
applying the LLL.

Fix an ordering of _V (G)_ which witnesses that the degeneracy of _F_ is at most _C._ Namely, writing
_V (J) = V (G) = {v1, . . ., vn}_ we have _|N_ (vj) ∩{v1, . . ., vj−1}| ≤ _C_ for all 2 ≤ _j_ _≤_ _n._ Orient all the edges in
_E(J)\E(H)_ so that for all such edges _vivj_ we have _i < j,_ and copy the orientation of all other edges from
_H._ Call the resulting digraph _G[′′]._ Let _G[′]_ be the subdigraph of _G[′′]_ consisting of edges of the form _vu_ for
_u ∈_ _A[G]H_ [(][v][)] [and] [observe] [that] _[|][N]G[ out][′]_ [(][v][)][| ≥|][A][G]H [(][v][)][| −] _[C]_ [for] _[v]_ _[∈]_ _[V][ (][G][).]_

Now we choose a subdigraph J of G[′] by including each edge of G[′] independently with probability p := 1/C [2].
We then deterministically resolve clashes, choosing a further subdigraph **D** of **J** as follows. For each _i ∈_ [n], if:

(1) deg[in]J [(][v][i][)][ ≥] [2] [or]
(2) deg[in]J [(][v][i][) = 1] [and] [there] [exists] _[u][ ∈]_ _[N]G[ in][′′][(][v][i][)][\][N]J[ in][(][v][i][)]_ [with] [deg][in]J [(][u][)][ ≥] [1,]

then we delete all edges of **J** that are oriented towards _vi._ We observe that **D** is a deterministic function of **J,**
and that by construction **D** is an escape-way. Indeed, (1) ensures that the indegree of any vertex is at most 1
and (2) ensures that the vertices of indegree at least 1 induce (in _F_ ) a subgraph of **D.** We denote by **Jww′**
the event that _{ww[′]_ _∈_ _E(J)},_ for each edge _ww[′]_ _∈_ _E(G[′])._ Thus **J** can be viewed as [�]e∈E(G[′]) **[J][e][.]** [Similarly,]

we denote, for each edge _ww[′]_ _∈_ _E(G[′]),_ by **Dww′,** the event that _{ww[′]_ _∈_ _E(D)}._

**Claim** **2.14.** _For_ _each_ _vu ∈_ _E(G[′])_ _we_ _have_ P[Dvu] ≥ 1/(eC [2]).

_Proof._ Consider an arbitrary edge _vu ∈_ _E(G[′])._ The probability that _vu ∈_ _E(J)_ is simply P[Jvu] = p. Then
the probability of Dvu given Jvu is precisely the probability that for all w _∈_ _NG[in][′]_ [(][u][)][\][v] [we have][ wu][ ̸∈] _[E][(][J][) (i.e.]_
not Jwu) and deg[in]J [(][w][) = 0.] [In other words, the probability that a set of at most][ C][(][C][ −] [1) edges that appeared]
in _G[′]_ do not appear in **J.** This occurs with probability at least (1 − _p)[C][(][C][−][1)]._ Recalling that _p_ = 1/C [2],
and using the relation that 1 − 1/x ≥ e[−][(1][/][(][x][−][1))] which holds for all reals _x > 1,_ we can lower bound this

probability by e[−] _[C]C[(][C][2]−[−]1[1)]_ _≥_ 1/e. Thus the probability of the event Dvu is at least p(1−p)[C][(][C][−][1)] _≥_ 1/(eC [2]). 
**Claim** **2.15.** _For_ _each_ _vertex_ _v_ _∈_ _V (G)_ _and_ _edge_ _e ∈_ _E(G[′]),_ _changing_ _the_ _outcome_ _of_ **Je** _effects_ _the_ _outcome_
_of_ _at_ _most_ 8 _events_ **Dvu.**

_Proof._ Recall that **D** is a deterministic function of **J.** The event **Dvu** depends on only: the event **Jvu,** and
for each _w_ _∈_ _NG[in][′]_ [(][u][)][\][u] [the] [events] **[J][wu]** [and] **[J][w][′][w]** [for] [each] _[w][′]_ _[∈]_ _[N]G[ in][′]_ [(][w][).] [The] [same] [can] [be] [said] [for] **[D][e]** [for] [any]
edge _e ∈_ _E(G[′])._ Thus if **Dvu** depends on **Jqw** then it must be that

_•_ _qw_ = vu or

_•_ _w_ = u or

_•_ _w_ _̸= v_ and _wu ∈_ _E(G[′])._

The first two candidates can hold for at most one vertex _u_ in total. Thus if **Jqw** effects _ℓ_ events **Dvu** then
there must be at least _ℓ_ _−_ 1 vertices in the common neighbourhood of _w_ and _v._ But this implies that there
exists a small subgraph of average degree (4(ℓ−ℓ1)+2−1) [.] [Thus] [by] [the] [condition] [that] [all] [subgraphs] [have] [average]
degree at most 3, we have _ℓ_ _≤_ 8. 
14


-----

**Claim** **2.16.** _For_ _all_ _v_ _∈_ _V (G[′])_ _we_ _have_

P�deg[out]D [(][v][)][ <] _[|][A]H[G]_ [(][v][)][| −] _[C]_

2eC [2]


� � (|A[G]H [(][v][)][| −] _[C][)][2]_
_< exp_ _−_

_|A[G]H_ [(][v][)][| ·][ 200][ ·][ e][2][C] [6]


�
_._


_Proof._ Let _v_ _∈_ _V (G[′])_ be arbitrary. As _|NG[out][′]_ [(][v][)][| ≥|][A][G]H [(][v][)][| −] _[C]_ [we] [have,] [by] [Claim] [2.14] [that] [E][[][|][ deg][out]D [(][v][)][|][]][ ≥]
(|A[G]H [(][v][)][| −] _[C][)][/][(][eC]_ [2][).] [Further] [deg][out]D [(][v][)] [is] [a] [function] [of] _[{][J][vu][}]_ [where] _[u]_ [ranges] [over] _[N]G[ out][′]_ [(][v][).] [Recall] [that]
deg[out]D [(][v][)] [depends] [on] [at] [most] _[m][ :][=][ |][A][G]H_ [(][v][)][|][C] [2] [events.] [Further] [by] [Claim] [2.15] [it] [depends] [on] [these] [events] [in]

a Lipschitz manner with constant 8. Therefore we can apply McDiarmid’s inequality with _t_ = _[|][A]H[G]2[(]eC[v][)][2][|−][C]_,

_m = |A[G]H_ [(][v][)][|][C] [2] [and] _[s][ = 8]_ [to] [get]


P�deg[out]D [(][v][)][ <] _[|][A]H[G]_ [(][v][)][| −] _[C]_

2eC [2]


� �
_< exp_ _−_ [2][ ·][ t][2]

_m · s[2]_


� � 2(|A[G]H [(][v][)][| −] _[C][)][2]_
_≤_ exp _−_

_|A[G]H_ [(][v][)][|][C] [2][ ·][ 8][2][ ·][ 4][e][2][C] [4]


�
_._



                              
Call the quantity inside the exponent of the end last equations _qv._ For the final step we wish to apply the
LLL to conclude that with positive probabilty we have that **D** satisfies the conditions of the Lemma. We
note that if _qv_ _≤_ 5 log ∆then the statement holds trivially for _v_ as deg[out]D [(][v][)][ ≥] [0.] [For] [all] [other] _[v]_ _[∈]_ _[V][ (][G][)]_ [we]

will introduce a bad event **Bv** := {deg[out]D [(][v][)][ <] _[|][A]H[G]2[(]eC[v][)][2][|−][C]_ _}._ Clearly if no bad events occur then the guarantees

of the Lemma are satisfied. We have, by Claim 2.16, that P[Bv] < exp[−qv] and by assumption _qv_ _> 5 log ∆._
It follows that P[Bv] < 1/∆[5].

Now we upper bound the number of dependent bad events per bad event. If for some pair _u, v_ _∈_ _V (G[′]),_
we have that **Bu** and **Bv** are dependent then there must be a vertex _w_ at distance at most 2 from both _u_ and
_v_ in _J._ Thus for each _u_ there are at most ∆[4] vertices _v_ such that **Bu** and **Bv** are dependent. We apply the
LLL via the relation _e · q · d ≤_ e · ∆1[5] _[·][ ∆][4]_ [=] ∆[e] _[≤]_ [1.] [The] [LLL] [then] [tells] [us] [that] [with] [positive] [probability] [none]

of the events **Bv** occur. Further, as observed earlier, with probability one, **D** is an escape-way, and it follows
that there exists at least one escape-way _D_ as guaranteed by the Lemma. Substituting _C_ = 3 completes the
proof with the estimate _qv_ _> |A[G]H_ [(][v][)][|][/][10][7][.] 
We are now ready to prove the following result which is a stronger version of our main result (Theorem 1.3).

**Theorem** **2.17.** _Let_ ∆ _∈_ N _and_ _suppose_ _G_ _is_ _a_ _graph_ _of_ _maximum_ _degree_ _at_ _most_ exp(∆/10[9]) _such_ _that_
_all_ _subgraphs_ _on_ _at_ _most_ (10[7]∆+ 1)n _vertices_ _have_ _average_ _degree_ _at_ _most_ 12/5. _Further_ _suppose_ _J_ _⊂_ _G_
_is_ _a_ _spanning_ _subgraph_ _with_ _minimum_ _degree_ _at_ _least_ 10[7]∆. _Then_ _J_ _contains_ _all_ _trees_ _on_ _n_ _vertices_ _with_
_maximum_ _degree_ _at_ _most_ ∆ _as_ _induced_ _subgraphs_ _of_ _G._

_Proof._ Let _T_ be a fixed tree on _n_ vertices and maximum degree ∆. We pick a root _r_ _∈_ _V (G)._ We will follow
an online process extending locally our current tree _Ti_ whereby at each step an adversary picks a vertex _x_ of
degree at most ∆ _−_ 1 from the current tree and we extend the tree from that _x._

_The_ _Pre-emptive_ _Greedy_ _Algorithm._ Throughout the process we will maintain an escape-way _Bi_ which
contains _Ti_ rooted at _r._ We will write _Di_ := _K(Bi)._ On a high level, one can think of _Bi_ as those edges
which are reserved for (possibly) extending the current tree _Ti._ The directed-ness of an edge in _Bi,_ say from
_u_ to _v,_ is to signal the fact that _if_ the edge _uv_ is ever used, then that is because we extend _from_ _u_ _to_ _v._

In step one our adversary chooses a root _r_ = T1 _⊂_ _T_ and we choose any vertex _r_ _∈_ _V (G)_ to be the root.
The root requires some special treatment (an unimportant technicality). We add all edges from _r_ to _NG(r)_
into _B1_ and we delete all other vertices adjacent to _r_ from _J._ Due to the condition that there are no small
graphs of average degree at least 12/5 co-degrees are bounded 7 and this deletion reduces the degree of any

15


-----

vertex in our graph by at most 7. For convenience and because _d ≥_ 1000, we shall ignore this 7 and continue
to call the remaining graph _J._ We now state the formal properties to be maintained throughout the whole
process. Suppose at step _i,_ we have a tree _Ti_ in _G_ and an escape-way _Bi_ in _both_ _G_ and _J._ We will always
assume there is an isomorphism _f_ from _Ti_ to a sub-tree of _T_ and for ease of notation identify _x ∈_ _Ti_ with
_f_ (x). We let _Xi_ denote the non-leaf vertices of _Ti_ and let _Ci_ := C(Xi) ⊆ _V (J)_ denote the _d-critical_ vertices
of _Xi._ Note that _Xi,_ _Ci_ and _Di_ are determined by _Ti_ and _Bi._ We will have _Bi_ _⊆_ _Bi+1_ and _Ti_ _⊂_ _Ti+1,_ and
thus _Xi_ _⊂_ _Xi+1,_ _Di_ _⊆_ _Di+1_ and _Ci_ _⊆_ _Ci+1_ for all _i ∈_ [n − 1]. We further ensure:

(1) _Ti_ _⊂_ _Bi;_
(2) _Vout(Bi) = Ci;_
(3) For every _v_ _∈_ _Ci_ we have deg[+]Bi[(][v][)][ ≥] [∆] _[−]_ [1] [and] [deg]Bi[(][r][) = deg]G[(][r][).]

As _C1_ = _r_ it is trivial to see that the conditions are satisfied initially. We now proceed by induction and
assume we have _Ti, Bi, Xi, Ci_ and _Di_ as described for some _i ∈_ [n − 1]. At each step, an adversary chooses a
vertex _x_ from the current tree _Ti_ with _dTi(x) < ∆and_ asks us to append a leaf to _x._ We now show how we
can maintain the above properties while extending _Ti._ Let _x ∈_ _Ti_ be the chosen vertex.

2.2. **Case** 1: _x ∈_ _Ci._ We simply choose an out-neighbour of _x,_ say _y_ _∈_ _NB[+]i[(][x][)][\][N][T]i[(][x][).]_ [This] [exists] [by] [(][3][)]
since _dTi_ (x) ≤ ∆ _−_ 1 and because if _x_ is not the root then it has an in-neighbour in Bi. We let _Ti+1_ = Ti ∪ _xy._
We then define _Bi+1_ := Bi and note that it contains _Ti+1._

2.3. **Case** 2: _x_ _∈/_ _Ci._ In this case, _x_ has many available neighbours, and so we would like to simply add
them to our tree. However, informally, adding arbitrary edges to _Bi_ may trigger a criticality cascade. In
order to find out which vertices near _x_ might be so affected we look at the _d-critical_ set _Ci+1_ := C(Ci ∪ _x)_ of
_Ci ∪_ _x._ Because _Ci_ = C(Xi) we have that _Ci+1_ = C(Xi ∪ _x)._

We observe that _|Ci+1|_ _<_ 2|Xi ∪ _x|._ Indeed suppose otherwise, then the 100-critical set of _Xi ∪_ _x_ is at
least as large and so by Lemma 2.9, there would exists a graph _H_ _⊂_ _J_ of order at most 300n and average
degree at least 12/5. This would contradict the assumptions of the theorem. Thus, in particular, we have
that _C_ _[∗]_ := Ci+1\Ci has order most _n._ Further, because _Vout(Bi) = Ci_ by induction, Proposition 2.8 tells us
that _|A[G]Di[(][v][)][| ≥]_ [deg]G[(][v][)][ −] _[d][ ≥]_ _[d]_ [for] [all] _[v]_ _[̸∈]_ _[C][i]_ [(and] [thus] [all] _[v]_ _[∈]_ _[C]_ _[∗][).]_

For each _v_ _∈_ _C_ _[∗]_ we add _d_ vertices from _A[G]Di[(][v][),]_ [along] [with] _[v][,]_ [to] [a] [set] [we] [call] _[C]_ [+][.] [Thus] _[|][C]_ [+][|] _[≤]_
(d + 1)|C _[∗]| ≤_ (d + 1)n. We now give as input to Lemma 2.13, the graph _J[C_ [+]] as _F_, the graph _G[C_ [+]] as _G_
and _Di_ the (bi)oriented subgraph of _J_ as _H._ We input the maximum degree of exp(d/10[9]) and note that this
_F_ has degeneracy at most 3, because it has order most 10[7]∆n (degeneracy is at most the maximum average
degree over all subgraphs). Lemma 2.13 therefore gives us an escape-way _I,_ that agrees with _Di,_ such that
the out-degree of each vertex _v_ _∈_ _C_ _[∗]_ is at least 10d[7] _[−]_ [5] 10[d][9][,] [which] [is] [at] [least] [∆because] _[d >][ 10][7][∆.]_ [We] [obtain]

_I_ _[′]_ from _I_ by deleting all edges whose initial vertex is not in _C_ _[∗]._ We let _Bi+1_ be the union of _Bi_ and _I_ _[′]._

Crucially I _[′]_ agrees with Di = K(Bi) because I agrees with Di and so Bi+1 is an escape-way by Proposition
2.6. We now let _Ti+1_ := Ti ∪ _xy_ for some _y_ _∈_ _NBi+1(x),_ so that clearly _Ti+1_ _⊂_ _Bi+1_ as required by (1). To
see (3) we first note it holds by induction for _v_ _∈_ _Ci_ because _Bi_ _⊂_ _Bi+1_ and for _v_ _∈_ _C_ _[∗]_ by the construction
of _I_ _[′]._ Similarly for (2). This completes the induction step.

Thus we obtain our tree _Tn_ as a subgraph of our escape-way _Bn_ in _G_ and _J._ We must check that is
induced. First note that because deg[+]Bi[(][r][) =][ deg]G[(][r][)] [and] [all] [other] [neighbours] [were] [deleted,] [we] [have] [that]

16


-----

the in-degree of _r_ is zero. By Proposition 2.2 this implies that the component of _Bn_ containing _r_ is a tree.
Further by the definition of escape-way, _Vin(Bn)_ induces a subgraph of _Bi,_ thus all that remains to note is
that any neighbours of _r_ that are in the tree are out-neighbours of _r._ This completes the proof.

                              
**Theorem** **2.18.** _There_ _exists_ _ε > 0_ _such_ _that_ _if_ _G_ _is_ _a_ _d-regular_ _graph_ _with_ _h(G) > d −_ 3 + 10[7]∆+11 _[for]_ _[some]_
_integer_ ∆ _< εd,_ _then_ _G_ _contains_ _a_ _spanning_ ∆-ary _pseudoforest_ _F,_ _that_ _is_ _component-wise_ _induced,_ _with_ _the_
_property_ _that_ _one_ _can_ _turn_ _F_ _into_ _an_ _induced_ _forest_ _by_ _deleting_ _one_ _vertex_ _from_ _each_ _of_ _its_ _components._

_Proof._ We run the same argument as in the proof of Theorem 1.3, with some minor adjustments. Firstly in
the online game, the adversary is now allowed to ask us to add neighbours to any vertex in _G,_ this does not
effect the process it just means that the current _Ti_ may not be connected. We enumerate the vertices of _G_
and assume that the adversary will ask us to extend the vertices in that order. We always extend from a
vertex ∆times in a row. We choose the root _r_ arbitrarily, but we do not do anything to its neighbours and
we just proceed with the induction. Case 1 is identical but in Case 2 we only require that the critical sets
are finite and we show so by applying Lemma 2.9. It is clear that as our adversary will eventually ask us to
extend _Ti_ from every vertex and give it out-degree ∆that we obtain in the limit a spanning escape-way _D_
with all out-degrees ∆. By Proposition 2.3 this is a spanning pseudoforest, and it is straightforward to see
that it is ∆-ary.

We now describe which vertices to delete. By Proposition 2.2 if a component of _D_ has a cycle then it has
no vertex of in-degree 0. For each component we either delete one vertex from the at most one cycle, or we
delete the at most one vertex with in-degree 0 (if there two vertices of indegree 0 in the same component,
then any path between them must have a vertex with indegree at least 2). Thus all remaining vertices are in
_Vin(D)_ and so the resulting graph is induced. Further, there are no cycles because we deleted a vertex from
each. 
3. Induced trees in random graphs

We will show how we can derive easily Theorem 1.4 from Theorem 1.3. We restate it below for convenience
of the reader.

**Theorem** **1.4.** _There_ _is_ _C_ _>_ 0, _such_ _that_ _for_ _all_ ∆ _∈_ N _and_ _d_ _>_ 2[20∆], _G(n, d/n)_ _contains_ _all_ _trees_ _with_
_maximum_ _degree_ _at_ _most_ ∆ _and_ _order_ _at_ _most_ _Cn_ _[induced]_ _[subgraphs]_ _[with]_ _[high]_ _[probability][.]_
_d log[2](d)_ _[as]_

_Proof._ Let _G_ _∼_ _G(n, d/n),_ where _d_ _> 2[20∆]_ and ∆ _≥_ 1. It is a simple observation that the following holds
w.h.p.

(i) For every _S_ _⊂_ _V (G),_ _e(G[S]) =_ _[d][|]2[S]n[|][2]_ _± dn/10;_

(ii) For every _S_ _⊂_ _V (G),_ with 200d log(n _d)_ _[≤|][S][| ≤]_ 100d log(n _d)_ [,] _[e][(][G][[][S][])][ ≤]_ [(1 + 1][/][5)][|][S][|][;]
(iii) There are at most _n/4_ vertices with degree greater than 20d.


Let _G_ satisfy the above. Delete from _G_ all vertices of degree greater than 20d. By (iii), we obtain an induced
subgraph _G[′]_ _⊂_ _G_ on at least 3n/4 vertices. Let _S_ _⊂_ _V (G[′])_ be a maximal subset of size at most 200d log(n _d)_
with _e(G[′][S])_ _>_ (1 + 1/5)|S|. By assumption _|S|_ _≤_ 200d log(n _d)_ [.] [Delete] _[S]_ [and] [let] _[G][′′]_ [:][=] _[G][[][V][ (][G][′][)][ \][ S][].]_ [It] [is]
clear that by maximality and (ii) no _S[′]_ _⊂_ _V (G[′′])_ of size at most 200d log(n _d)_ [spans] [more] [than] [(1 + 1][/][5)][|][S][|]
edges. Furthermore, by (ii), _|G[′′]| ≥_ _n/2_ and by (i), we know _e(G[′′]) ≥_ _[d]8_ _[|][G][′][|][.]_ [Finally,] [passing] [to] [an] [induced]

subgraph _G[′′′]_ _⊂_ _G[′′]_ with minimum degree at least _d/16,_ we have thus constructed an induced subgraph of

17


-----

_G(n, d/n)_ with _δ(G[′′′]) ≥_ _d/16_ and ∆(G[′′′]) ≤ 20d. Moreover, (by (i)), we know _|G[′′′]| ≥_ _n/50._ It is easy to see
_G[′′′]_ satisfies the conditions of Theorem 1.3 with _n :=_ _n_ [∆:= 10][6][ log(][d][).]
10[14]d log[2](d) [and]

                              
2. Induced size Ramsey of trees

In this section, we will prove Theorem 1.6 which trivially implies Theorem 1.5 by taking _ε_ = 1/2. As
above, we restate the theorem for convenience of the reader.

**Theorem** **1.6.** _For_ _all_ ∆, n ∈ N _and_ _ε > 0_ _there_ _exists_ _a_ _graph_ _G_ _with_ _less_ _than_ _C(∆, ε) · n_ _edges_ _such_ _that_
_any_ _subgraph_ _J_ _⊂_ _G_ _containing_ _ε · e(G)_ _edges_ _contains_ _every_ _tree_ _of_ _maximum_ _degree_ ∆ _and_ _order_ _at_ _most_ _n_
_as_ _an_ _induced_ _subgraph_ _of_ _G._ _One_ _can_ _take_ _C(∆, ε) =_ �10[42]∆[3] log(∆) log( [1]ε [)][3][�]/ε[2].

_Proof._ Let _N_ := [10][30][n][ log(∆)∆]ε [2][ log(][ 1]ε [))][3] and _d :=_ [10][12][∆log(]ε [ 1]ε [)] Let _G ∼_ _G(N, d/N_ ) then as above we know that

w.h.p. the following holds.

(i) For every _S_ _⊂_ _V (G),_ _e(G[S]) =_ _[d]2[|][S]N[|][2]_ _[±][ dN/][10;]_

(ii) For every _S_ _⊂_ _V (G),_ with 200dN log(d) _[≤|][S][| ≤]_ 100dN log(d) [,] _[e][(][G][[][S][])][ ≤]_ [(1 + 1][/][5)][|][S][|][;]
(iii) There are at most _N/4_ vertices with degree greater than 20d.


Let _G_ satisfy the above and delete all vertices of degree greater than 20d. We obtain an induced subgraph _G[′]_

on at least 3N/4 vertices with ∆(G1) ≤ 20d. Let _S_ _⊂_ _V (G[′])_ be a maximal subset of size at most 200 log(N _d)_
with _e(G[′][S])_ _>_ (1 + 1/5)|S|. By assumption _|S|_ _≤_ 200d log(n _d)_ [.] [Delete] _[S]_ [and] [let] _[G][′′]_ [:][=] _[G][[][V][ (][G][′][)][ \][ S][].]_ [It] [is]
clear that by maximality and (ii) no _S[′]_ _⊂_ _V (G[′′])_ of size at most 200d log(n _d)_ [spans] [more] [than] [(1 + 1][/][5)][|][S][|]
edges. Furthermore, by (ii), _|G[′′]| ≥_ _N/2_ and by (i), we know _e(G[′′]) ≥_ _[d]8_ _[|][G][′][|][.]_ [Finally,] [passing] [to] [an] [induced]

subgraph _G[′′′]_ _⊂_ _G[′′]_ with minimum degree at least _d/16,_ we have thus constructed an induced subgraph of
_G(N, d/N_ ) with _δ(G[′′′]) ≥_ _d/16_ and ∆(G[′′′]) ≤ 20d. Moreover, (by (i)), we know _|G[′′′]| ≥_ _N/50._ _G[′′′]_ will be
the desired graph.

All we need to show is that given any _J_ _⊂_ _G[′′′]_ with _e(J) ≥_ _εe(G[′′]),_ _J_ contains an induced copy (in _G[′′′])_
of every tree on _n_ vertices and maximum degree ∆. We first let _J_ _[′]_ _⊂_ _J_ be an induced subgraph of _J_ with
minimum degree _εd/20 ≥_ 10[7](10[6]∆ log(1/ε)) = 10[7]f, where _f_ := 10[6]∆ log(1/ε). Moreover, by assumption
∆(G[V (J _[′])]) ≤_ 20d ≤ 2[f/][10][9] and every subset _|S|_ of size at most 10[10]n∆ log(∆) log( [1]ε [)][ ≥] [(10][7][f] [+ 1)][n] [spans]

at most (1 + 1/5)|S| edges. We may now invoke Theorem 2.17 with _G[′′′]_ := G, _J_ := J, ∆ := f and _n := n._ 
We now easily derive an induced size ramsey result for _q_ colours by taking _ε := 1/q_ in Theorem 1.6

**Theorem** **1.7.** _There_ _is_ _C_ _> 0_ _such_ _that_ _the_ _following_ _holds._ _Let_ _q, ∆, n ≥_ 1. _Then,_ _there_ _is_ _a_ _graph_ _G_ _on_
_at_ _most_ _C∆[3]_ log(∆)q[2] log(q)[3]n _edges_ _such_ _that_ _in_ _every_ _q-edge-colouring_ _of_ _G_ _there_ _is_ _a_ _colour_ _class_ _which_
_spans_ _all_ _trees_ _T_ _on_ _at_ _most_ _n_ _vertices_ _and_ ∆(T ) ≤ ∆ _as_ _induced_ _subgraphs_ _of_ _G._

We observe this is almost tight as a function of _q_ since even for the non-induced case one has that the _q_
size ramsey number of a path on _n_ vertices is at least _cq[2]n,_ for some absolute _c > 0._

2. Concluding remarks

We have developed an algorithmic approach to embed bounded degree trees in sparse expanding graphs,
generalising the remarkable result of Friedman and Pippenger [FP87]. We have applied this result to give the

18


-----

state of the art on the bounds for multiple questions. We will now discuss some further avenues of research
and state some open problems.

2.1. **Tightening** **Theorem** **1.3.** There are three main places where one could tighten Theorem 1.3, each of
which would yield slight but growing (as a function of ∆or _d)_ improvements in our applications. Firstly, one
could try remove the maximum degree condition. We use this exclusively when applying the LLL in Lemma
2.13. Secondly, one could try to remove the factor of ∆in the order of the sets upon which we place our
density constraint. Morally, this factor comes from reserving ∆neighbours for critical vertices even though
some critical vertices may never actually be extended from. Finally, one could try to replace the average
degree upper bound of 12/5 by some larger constant. This upper bound is used in three places. It is used
to bound the degeneracy in Lemma 2.13, but there is a lot of slack in this application. That is, one merely
requires that ∆is much smaller than _d/C_ [2], where _d_ is the minimum of the graph given to Lemma 2.13 (we
think the factor of _C_ [2] is roughly tight here). It is also used within Lemma 2.13 when we bound the Lipschitz
constant, but this could be avoided by simply requiring girth at least 5. The place where it is really required
is when it used to show that criticality cascades eventually stop (Lemma 2.9). In the case of Theorem 1.3
there may be an interesting dependency between the average degree upper bound and the order of the sets
that must satisfy it. We find this latter problem particularly intriguing. For example, in a more concrete way
we could not answer the following nice question.

**Problem** **2.1.** _Let_ _d1, d2_ _be_ _positive_ _integers._ _Is_ _there_ _f_ (d1, d2) ≥ 1 _and_ _ε(d1, d2) > 0_ _such_ _that_ _the_ _following_
_holds._ _Let_ _G_ _be_ _a_ _graph_ _with_ _average_ _degree_ _f_ (d1, d2) _such_ _that_ _all_ _subset_ _of_ _size_ _at_ _most_ _n_ _have_ _average_
_degree_ _at_ _most_ _d2._ _Is_ _there_ _an_ _induced_ _subgraph_ _G[′]_ _⊂_ _G_ _of_ _average_ _degree_ _at_ _least_ _d1_ _such_ _that_ **_all_** _subsets_
_S_ _⊂_ _V (G[′])_ _of_ _size_ _at_ _most_ _ε(d1, d2)n_ _in_ _G[′]_ _span_ _at_ _most_ 3/2|S| _edges_ ?

Finally, we mention a related problem. Theorem 2.18 in its current form is almost tight. However, if we
impose girth at least 5 then can we apply it to graphs with _h(G) > d −_ _f_ (d) for any function _f_ that tends to
infinity with _d?_ We see no reason why _f_ could not be taken a polynomial.

2.2. **Rolling** **backwards.** In [DKN22], Friedman-Pippenger result was cleverly combined with a “rollingback” technique which allowed them to find different structures in expanding graphs. It is natural to ask
whether our method also allows for roll-backs and indeed, it does. The key points are that in the proof of the
main result we induct on properties of _Ti,_ _Ci_ and _Bi,_ while _Ci_ can be viewed as a monotone function of _Ti._
When rolling back, one _deletes_ some vertices of the current embedded tree _Ti_ to obtain _T_ _[′]_ and only keep the
directed edges of _Bi_ that start from the new _Ci_ which is in turn a function of _T_ _[′]._ The desired properties are
maintained. One can even delete non-leaf vertices when rolling back although the bound on the order of the
critical bootstrap percolation will be worse (larger) than the factor of 2 in Lemma 2.8 (which uses that _G[X]_
is connected).

2.3. **Induced** **ramsey** **and** **induced** **size** **ramsey.** Recently, Dragani´c and Keevash [KD24] gave a bound
on the induced size ramsey number of paths, _rˆind[q]_ [(][P][n][) =][ O][(][nq][3][ log][2][(][q][)).] [Theorem] [1.7] [gives] [an] [improvement]
on this by essentially a factor of _q._ We believe however the size ramsey and induced size ramsey of bounded
degree trees should not behave very differently _as_ _a_ _function_ _of_ _the_ _number_ _of_ _colours._

**Conjecture** **2.2.** _Let_ ∆ _≥_ 1 _and_ _T_ _be_ _a_ _tree_ _on_ _n_ _vertices_ _with_ ∆(T ) ≤ ∆. _Then,_ _for_ _every_ _q_ _≥_ 1,

_rˆind[q]_ [(][T] [) =][ O][∆][(ˆ][r][q][(][T] [))][.]

19


-----

We reiterate a central problem in the area regarding the induced ramsey number of bounded degree graphs.
The best upper bound on the induced ramsey number for graphs of bounded degree is _n[O][(∆)]_ proved by
Conlon, Fox and Zhao [CFZ14]. It is therefore remarkable that the possibility of induced ramsey numbers of
bounded degree graphs being linear remains open.

**Problem** **2.3.** _Is_ _there_ ∆ _≥_ 1, _such_ _that_ _for_ _every_ _C_ _> 0,_ _there_ _is_ _a_ _graph_ _G_ _on_ _n_ _vertices_ _and_ ∆(G) ≤ ∆
_with_ _rind(G) > Cn?_

2.4. **Induced** **structures** **in** **random** **graphs.** We are confident our main result will be very useful in
finding other large induced structures in random graphs. For example, one could ask what is the largest _k_ for
which _G(n, p)_ contains an _induced_ subdivision of a _Kk_ whp? This would be a induced version of a classical
result of Ajtai, K´omlos and Szemer´edi [AKS79] which guarantees that whp _G(n, p)_ where _p = o(_ _√[1]_ ) contains
_n_
a subdivison of _K(1+o(1))∆_ where ∆is the maximum degree of _G(n, p)._ Finally, we think our methods could
be helpful in proving essentially tight bounds for the size of induced bounded degree trees in _G(n, p)._

**Conjecture** **2.4.** _For_ _every_ ∆ _≥_ 1 _there_ _is_ _C∆_ _>_ 0 _such_ _that_ _the_ _following_ _holds._ _For_ _all_ _[C]n[∆]_ _[≤]_ _[p]_ _[≤]_ [0][.][1,]

� log(pn) �
_G(n, p)_ _contains_ _w.h.p._ _all_ _trees_ _of_ _order_ Ω _p_ _with_ _maximum_ _degree_ ∆.

We do not even know the above result for any bounded degree tree (including a path).

References

[ABNC[+]96] Alok Aggarwal, Amotz Bar-Noy, Don Coppersmith, Rajiv Ramaswami, Baruch Schieber, and
Madhu Sudan. Efficient routing in optical networks. _Journal_ _of_ _the_ _ACM_ (JACM ), 43(6):973–
1001, 1996. 1, 2, 7, 10

[ADK17] Stephen Alstrup, Søren Dahlgaard, and Mathias Knudsen. Optimal induced universal graphs
and adjacency labeling for trees. _Journal_ _of_ _the_ _ACM_ (JACM ), 64(4):1–22, 2017. 6

[AKS79] Mikl´os Ajtai, J´anos Koml´os, and Endre Szemerdi.[´] Topological complete subgraphs in random
graphs. _Studia_ _Scientiarum_ _Mathematicarum_ _Hungarica,_ 14:293–297, 1979. 20

[AKS07] Noga Alon, Michael Krivelevich, and Benny Sudakov. Embedding nearly-spanning bounded
degree trees. _Combinatorica,_ 27(6):629–644, 2007. 2

[BDS23] Domagoj Bradaˇc, Nemanja Dragani´c, and Benny Sudakov. Effective bounds for induced
size-Ramsey numbers of cycles. _arXiv_ _preprint_ _arXiv:2301.10160,_ 2023. 5

[Bec83] J´ozsef Beck. On size ramsey number of paths, trees, and circuits. i. _Journal_ _of_ _Graph_ _Theory,_
7(1):115–129, 1983. 1, 4, 5

[Bec90] J´ozsef Beck. On size Ramsey number of paths, trees and circuits. ii. _Mathematics_ _of_ _Ramsey_
_theory,_ pages 34–45, 1990. 5

[BKM[+]21] S¨oren Berger, Yoshiharu Kohayakawa, Giulia Satiko Maesaka, Ta´ısa Martins, Walner Mendon¸ca,
Guilherme Oliveira Mota, and Olaf Parczyk. The size-Ramsey number of powers of bounded
degree trees. _Journal_ _of_ _the_ _London_ _Mathematical_ _Society,_ 103(4):1314–1332, 2021. 2, 6

[BS97] Itai Benjamini and Oded Schramm. Every graph with a positive cheeger constant contains a
tree with a positive cheeger constant. _Geometric_ _&_ _Functional_ _Analysis_ _GAFA,_ 7(3):403–419,
1997. 6

[But09] Steve Butler. Induced-universal graphs for graphs with bounded maximum degree. _Graphs_ _and_
_Combinatorics,_ 25(4):461, 2009. 6

20


-----

[CDKS21] Oliver Cooley, Nemanja Dragani´c, Mihyun Kang, and Benny Sudakov. Large induced matchings
in random graphs. _SIAM_ _Journal_ _on_ _Discrete_ _Mathematics,_ 35(1):267–280, 2021. 4

[CFS12] David Conlon, Jacob Fox, and Benny Sudakov. On two problems in graph Ramsey theory.
_Combinatorica,_ 32:513–535, 2012. 4

[CFS15] David Conlon, Jacob Fox, and Benny Sudakov. Recent developments in graph Ramsey theory.
_Surveys_ _in_ _combinatorics,_ 424(2015):49–118, 2015. 4

[CFZ14] David Conlon, Jacob Fox, and Yufei Zhao. Extremal results in sparse pseudorandom graphs.
_Advances_ _in_ _Mathematics,_ 256:206–290, 2014. 5, 20

[CG83] Fan Chung and Ronald Graham. On universal graphs for spanning trees. _Journal_ _of_ _the_ _London_
_Mathematical_ _Society,_ 2(2):203–211, 1983. 6

[COE15] Amin Coja-Oghlan and Charilaos Efthymiou. On independent sets in random graphs. _Random_
_Structures_ _&_ _Algorithms,_ 47(3):436–486, 2015. 3

[CRST83] V´aclav Chvat´al, Vojtech R¨odl, Endre Szemer´edi, and William Trotter. The Ramsey number of a
graph with bounded maximum degree. _Journal of Combinatorial Theory,_ _Series B, 34(3):239–243,_
1983. 4, 5

[DGK22] Nemanja Dragani´c, Stefan Glock, and Michael Krivelevich. Short proofs for long induced paths.
_Combinatorics,_ _Probability_ _and_ _Computing,_ 31(5):870–878, 2022. 4

[Dir52] Gabriel Andrew Dirac. Some theorems on abstract graphs. _Proceedings_ _of_ _the_ _London_ _Mathe-_
_matical_ _Society,_ 3(1):69–81, 1952. 1

[DJK08] Domingos Dellamonica Jr and Yoshiharu Kohayakawa. An algorithmic friedman–pippenger
theorem on tree embeddings and applications. _The_ _Electronic_ _Journal_ _of_ _Combinatorics,_ 2008.
2, 8

[DKN22] Nemanja Dragani´c, Michael Krivelevich, and Rajko Nenadov. Rolling backwards can move
you forward: on embedding problems in sparse expanders. _Transactions_ _of_ _the_ _American_
_Mathematical_ _Society,_ 375(7):5195–5216, 2022. 2, 19

[dlV86] Wanceslas Fernandez de la Vega. Induced trees in sparse random graphs. _Graphs_ _and_ _Combina-_
_torics,_ 2:227–231, 1986. 3

[DMC[+]24] Nemanja Dragani´c, Richard Montgomery, David Munh´a Correia, Alexey Pokrovskiy, and Benny
Sudakov. Hamiltonicity of expanders: optimal bounds and application. _ArXiv:2402.06603,_ 2024.
2

[Dra20] Nemanja Dragani´c. Large induced trees in dense random graphs. _ArXiv:2004.02800,_ 2020. 3

[EFRS78] Paul Erd˝os, Ralph Faudree, Cecil Rousseau, and Richard Schelp. The size Ramsey number.
_Periodica_ _Mathematica_ _Hungarica,_ 9(1-2):145–161, 1978. 1, 4

[EP83] Paul Erd˝os and Zbigniew Palka. Trees in random graphs. _Discrete_ _Mathematics,_ 46(2), 1983. 3

[ER60] Paul Erd˝os and Alfr´ed R´enyi. On the evolution of random graphs. _Publications_ _Mathematical_
_Institute_ _Hungarian_ _Academiy_ _Sciences,_ 5:17–61, 1960. 1

[Erd81] Paul Erd˝os. On the combinatorial problems which i would most like to see solved. _Combinatorica,_
1(1):25–42, 1981. 1, 5

[FFP88] Paul Feldman, Joel Friedman, and Nicholas Pippenger. Wide-sense nonblocking networks. _SIAM_
_Journal_ _on_ _Discrete_ _Mathematics,_ 1(2):158–173, 1988. 2

[FJ87a] Alan Frieze and Bill Jackson. Large holes in sparse random graphs. _Combinatorica,_ 7:265–274,
1987. 3

21


-----

[FJ87b] Alan Frieze and Bill Jackson. Large induced trees in sparse random graphs. _Journal_ _of_
_Combinatorial_ _Theory,_ _Series_ _B,_ 42(2):181–195, 1987. 3

[FP87] Joel Friedman and Nicholas Pippenger. Expanding graphs contain all small trees. _Combinatorica,_
7:71–76, 1987. 1, 5, 18

[Fri90] Alan Frieze. On the independence number of random graphs. _Discrete_ _Mathematics,_ 81(2):171–
175, 1990. 3

[FS08] Jacob Fox and Benny Sudakov. Induced Ramsey-type theorems. _Advances_ _in_ _Mathematics,_
219(6):1771–1800, 2008. 5

[GJK] Roman Glebov, D Johannsen, and Michael Krivelevich. Hitting time appearance of certain
spanning trees in the random graph process. _manuscript_ _in_ _preparation._ 2

[GM75] Geoffrey Grimmett and Colin McDiarmid. On colouring random graphs. In _Mathematical_
_Proceedings_ _of_ _the_ _Cambridge_ _Philosophical_ _Society,_ volume 77, pages 313–324. Cambridge
University Press, 1975. 3

[GRR00] Ronald Graham, Vojtech R¨odl, and Andrzej Ruci´nski. On graphs with linear Ramsey numbers.
_Journal_ _of_ _Graph_ _Theory,_ 35(3):176–192, 2000. 4

[Hax01] Penny Haxell. Tree embeddings. _Journal_ _of_ _Graph_ _Theory,_ 36(3):121–130, 2001. 2

[HK95] Penny Haxell and Yoshiharu Kohayakawa. The size-Ramsey number of trees. _Israel_ _Journal_ _of_
_Mathematics,_ 89:261–274, 1995. 2, 5

[HK�L95] Penny Haxell, Yoshiharu Kohayakawa, and Tomasz �Luczak. The induced size-Ramsey number
of cycles. _Combinatorics,_ _Probability_ _and_ _Computing,_ 4(3):217–239, 1995. 5

[HS] Zach Hunter and Benny Sudakov. Induced ramsey problems for trees and graphs with bounded
treewidth. _Manuscript_ _in_ _preparation._ 2, 5

[KD24] Peter Keevash and Nemanja Dragani´c. Long induced paths in expanders. _ArXiv:2402.02256v_ 1,
2024. 19

[KLWY21] Nina Kamˇcev, Anita Liebenau, David Wood, and Liana Yepremyan. The size Ramsey number of
graphs with bounded treewidth. _SIAM_ _Journal_ _on_ _Discrete_ _Mathematics,_ 35(1):281–293, 2021.
2, 6

[KR87] Ludˇek Kuˇcera and Vojtˇech R¨odl. Large trees in random graphs. _Commentationes_ _Mathematicae_
_Universitatis_ _Carolinae,_ 28(1):7–14, 1987. 3

[KRSS11] Yoshiharu Kohayakawa, Vojtˇech R¨odl, Mathias Schacht, and Endre Szemer´edi. Sparse partition
universal graphs for graphs of bounded degree. _Advances_ _in_ _Mathematics,_ 226(6):5041–5065,
2011. 5, 6

[Lee17] Choongbum Lee. Ramsey numbers of degenerate graphs. _Annals of Mathematics, 185(3):791–829,_
2017. 4, 5

[�LP88] Tomasz �Luczak and Zbigniew Palka. Maximal induced trees in sparse random graphs. In _Annals_
_of_ _Discrete_ _Mathematics,_ volume 38, pages 257–265. Elsevier, 1988. 3

[LPY21] Shoham Letzter, Alexey Pokrovskiy, and Liana Yepremyan. Size-ramsey numbers of powers of
hypergraph trees and long subdivisions. _ArXiv:2103.01942,_ 2021. 2

[�Luc91] Tomasz �Luczak. Cycles in a random graph near the critical point. _Random_ _Structures_ _&_
_Algorithms,_ 2(4):421–439, 1991. 3

[Mon19] Richard Montgomery. Spanning trees in random graphs. _Advances_ _in_ _Mathematics,_ 356:106793,
2019. 2

22


-----

[Moo65] John Moon. On minimal _n-universal_ graphs. _Glasgow_ _Mathematical_ _Journal,_ 7(1):32–33, 1965.

1, 6

[P´os76] Lajos P´osa. Hamiltonian circuits in random graphs. _Discrete_ _Mathematics,_ 14(4):359–364, 1976.

1

[Ram28] Frank Ramsey. On a problem of formal logic. _Proceedings_ _of_ _London_ _Mathematical_ _Society,_
48:122–160, 1928. 1

[RS00] Vojtˇech R¨odl and Endre Szemer´edi. On size Ramsey numbers of graphs with bounded degree.
_Combinatorica,_ 20(2):257–262, 2000. 5

[Ruc87] Andrzej Ruci´nski. Induced subgraphs in a random graph. In North-Holland _Mathematics_ _Studies,_
volume 144, pages 275–296. Elsevier, 1987. 3

[Sue92] Stephen Suen. On large induced trees and long induced paths in sparse random graphs. _Journal_
_of_ _Combinatorial_ _Theory,_ _Series_ _B,_ 56(2):250–262, 1992. 3

[Tik22] Konstantin Tikhomirov. On bounded degree graphs with large size-Ramsey numbers. _arXiv_
_preprint_ _arXiv:2210.05818,_ 2022. 5

23


-----

