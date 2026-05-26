## Average-Case Deterministic Query Complexity of Boolean Functions with Fixed Weight


#### Yuan Li
```
yuan li@fudan.edu.cn
 Fudan University

```

#### Haowei Wu
```
hwwu21@m.fudan.edu.cn
 Fudan University

```

#### Yi Yang
```
 yyang1@fudan.edu.cn
 Anqing Normal University Fudan University

```

**Abstract**

We study the _average-case_ _deterministic_ _query_ _complexity_ of boolean functions under a _uni-_
_form_ _input_ _distribution,_ denoted by Dave(f ), the minimum average depth of zero-error decision
trees that compute a boolean function _f_ . This measure has found several applications across
diverse fields, yet its understanding is limited.
We study boolean functions with fixed weight, where weight is defined as the number of inputs
on which the output is 1. We prove Dave(f ) ≤ max �log [wt(]log n[f] [)] [+][ O][(log log] [wt(]log n[f] [)] [)][, O][(1)]� for every

_n-variable_ boolean function f, where wt(f ) denotes the weight. For any 4 log n ≤ _m(n) ≤_ 2[n][−][1],
we prove the upper bound is tight up to an additive logarithmic term for almost all _n-variable_
boolean functions with fixed weight wt(f ) = m(n).
H˚astad’s switching lemma or Rossman’s switching lemma [Comput. Complexity Conf. 137,

� 1 � � 1 �
2019] implies Dave(f ) _≤_ _n_ 1 − _O(w)_ or Dave(f ) _≤_ _n_ 1 − _O(log s)_ for CNF/DNF formulas of

width w or size s, respectively. We show there exists a DNF formula of width w and size ⌈2[w]/w⌉

� �
such that Dave(f ) = n 1 − Θ([log]w[ n]) for any _w_ _≥_ 2 log n.


**_Keywords— average-case query complexity, decision tree, weight, criticality, switching lemma_**

### 1 Introduction


The _average-case_ _deterministic_ _query_ _complexity_ of a boolean function _f_ under a _uniform_ _input_
_distribution[1],_ denoted by Dave(f ), is the minimum average depth of zero-error decision trees that
compute _f_ . This notion serves as a natural average-case analogy of the classic deterministic query
complexity D(f ) and has found applications in query complexity, boolean function analysis, learning
algorithms, game theory, and percolation theory. Besides that, Dave(f ) is a measure with limited
understanding, since Dave(f ) falls outside the class of polynomially-related measures, which includes
D(f ), R(f ), C(f ), bs(f ), and s(f ) (see the summaries in [BW02; ABK16; Aar+21] and Huang’s
proof of the Sensitivity Conjecture [Hua19]). This work is also inspired by Rossman’s circuit lower
bounds of detecting _k-clique_ on Erd˝os–R´enyi graphs in the average case [Ros08; Ros14]. Through
this paper, we hope to initiate a comprehensive study on Dave(f ), exploring its implications and
applications.

1In [AW01], the complexity under distribution _µ_ is denoted by Dµ(f ), and _µ_ could be arbitrary. In this paper, we
assume the input distribution _µ_ is uniform.

1


-----

#### 1.1 Background

To our knowledge, Ambainis and de Wolf were the first to introduce the concept of _average-case_
query complexity [AW01]. They showed super-polynomial gaps between average-case deterministic query complexity, average-case bounded-error randomized query complexity, and average-case
quantum query complexity.
Prior to the conceptualization by Ambainis and de Wolf, average-case query complexity had
been studied implicitly since the early days of computer science. Yao [Yao77] noticed that D[µ]ave[(][f] [)]
(with respect to any distribution µ) lower bounds the zero-error randomized query complexity, i.e.,
D[µ]ave[(][f] [)] _[≤]_ [R][0][(][f] [).] [Furthermore,] [Yao’s] [minimax] [principle] [says] [the] [maximum] [value] [of] [D][µ]ave[(][f] [)] [over]
all distributions _µ_ equals R0(f ).
O’Donnell, Saks, Schramm, and Servedio established the OSSS inequality [ODo+05; Lee10;
JZ11]: D[µ]ave[p] [(][f] [)] _[≥]_ maxVar[i Inff ]i[f ] [for] [any] [boolean] [function] _[f]_ [,] [where] _[µ][p]_ [is] [the] _[p][-biased]_ [distribution] [and]
Inf _i[f_ ] is the influence of coordinate _i._ By applying the inequality, O’Donnell et al. [ODo+05]
showed that R0(f ) _≥_ D[µ]ave[p] [(][f] [)] _[≥]_ [(][n/]�4p(1 − _p))[2][/][3]_ for any nontrivial monotone _n-vertex_ graph
property _f_ with critical probability _p._ This result made progress on Yao’s conjecture [Yao77],
which asserts that R0(f ) = Ω(n) for every nontrivial monotone graph property. When _p_ = 1/2,
we have R0(f ) _≥_ Dave(f ) _≥_ _n[2][/][3];_ Benjamini et al. proved that the lower bound Dave(f ) _≥_ _n[2][/][3]_ is
almost tight [BSW05].
While studying learning algorithms, O’Donnell and Servedio [OS06] discovered the OS inequality: ([�]i _[f][ˆ][(][{][i][}][))][2]_ _[≤]_ [D][ave][(][f] [)] _[≤]_ [log DT][size][(][f] [)] [for] [any] [boolean] [function] _[f]_ [,] [where] _[f][ˆ][(][·][)]_ [denotes] _[f]_ [’s]
Fourier coefficient and DTsize(f ) denotes the decision tree size. The OS inequality plays a crucial
role in learning monotone boolean functions (under the uniform distribution).
The most surprising connection (application) arose in game theory. Peres, Schramm, Sheffield,
and Wilson [Per+07] studied the _random-turn_ HEX game, in which two players determine who
plays next by tossing a coin before each round. They proved that the expected playing time (under
optimal play) coincides with Dave(f ), where _f_ is the _L × L_ hexagonal cells connectivity function.
Using the OS inequality and the results of Smirnov and Werner on percolation [SW01], Peres et al.
proved a lower bound _L[1][.][5+][o][(1)]_ on the expected playing time on an _L × L_ board.

#### 1.2 Our results

The _weight_ of a boolean function, defined as the number of inputs on which the output is 1, is
related to its query complexity. For instance, Ambainis et al. [Amb+16] proved that the quantum
� log m �
query complexity of almost all n-variable functions with fixed weight m is Θ,
_c+log n−log log m_ [+][ √][n]
where _c_ _>_ 0 is a constant. In contrast, the hardest function with weight _m_ has quantum query

�� log m �1/2 �
complexity Θ _n ·_ + √n . Ambainis et al. [Amb+16] also proved that almost
_c+log n−log log m_

all functions with fixed weight _m ≥_ 1 have randomized query complexity Θ(n) as the hardest one.
Our first result proves that Dave(f ) _≤_ log logm n [+] _[O][(log log]_ logm n [)] [for] [any] _[n][-variable]_ [boolean]
function _f_ with weight _m ≥_ 4 log n.

**Theorem** **1.1.** For every boolean function _f_ : {0, 1}[n] _→{0, 1},_ if the weight wt(f ) ≥ 4 log n, then


�
_._ (1)


Otherwise, Dave(f ) = O(1).


�

Dave(f ) ≤ log [wt(][f] [)] log log [wt(][f] [)]

log n [+][ O] log n

2


-----

We prove Theorem 1.1 by designing a recursive query algorithm that attains the query complexity given in (1). The algorithm queries an arbitrary bit until the subfunction’s weight becomes
sufficiently small, or more specifically, smaller than the logarithm of its input length; once this
border condition is met, we invoke another algorithm which, on average, takes _O(1)_ bits to query
the subfunction.
Next, we prove Theorem 1.2, complementing our first result, which says that Theorem 1.1 is
tight up to a lower order term for almost all fixed-weight functions.

**Theorem** **1.2.** Let _m_ : N _→_ N be a function such that 4 log n _≤_ _m(n)_ _≤_ 2[n][−][1]. For almost all
boolean functions _f_ : {0, 1}[n] _→{0, 1}_ with fixed weight wt(f ) = m(n),


�

Dave(f ) ≥ min log log [wt(][f] [)]
_x∈{0,1}[n][ C][x][(][f]_ [)][ ≥] [log wt(]log n[f] [)] _[−]_ _[O]_ log n


�
_,_ (2)


where Cx(f ) denotes the size of the smallest certificate on input _x._

**Remark 1.3.** For boolean functions with wt(f ) ≥ 2[n][−][1], we can obtain a similar bound by replacing
_f_ with _¬f_ .

Beyond fixed-weight functions, we also examine CNFs, circuits, and formulas, studying the
connection between Dave(f ) and criticality.
Rossman introduced the notion of _criticality,_ defined as the minimum value _λ_ _≥_ 1 such that
the following property holds: Prρ∼Rp[D(f _|ρ)_ _≥_ _t]_ _≤_ (pλ)[t] for any _p_ _∈_ [0, 1] and _t_ _∈_ N. In terms of
criticality, H˚astad’s switching lemma says every width-w CNF is _O(w)-critical_ [H˚as86]; Rossman’s
switching lemma says every size-s CNF is O(log s)-critical [Ros17; Ros19]; Rossman proved depth-d
size-s AC[0] circuits are O(log s)[d][−][1]-critical [Ros19]; Harsha et al. proved depth-d size-s AC[0] formulas
are _O(_ [1] [[HMS23].]

_d_ [log][ s][)][d][−][1][-critical]
For any _λ-critical_ function _f_, applying a ( [1] [restriction] [and] [then] [querying] [the] [result-]

2λ [)-random]

� � �� _n_ �
ing subfunction via its optimal decision tree yields Dave(f ) _≤_ _n_ 1 − _λ[1]_ + O _λ_ (Lemma 4.1

from Section 4). Hence, criticality bounds imply average-case query complexity bounds for CNFs,
formulas, and circuits.
For CNFs, circuits, or formulas, it is meaningful to understand whether the upper bounds on
Dave(f ) are tight or not. For example, consider a _w-CNF_ _f_ . By Lemma 4.1 from Section 4,
� 1 �
we have Dave(f ) _≤_ _n_ 1 − _O(w)_ . If the bound were indeed tight, it would suggest that the _p-_

random restriction, with _p_ = 1 [is] [essentially] [an] [optimal] [query] [algorithm] [for] [generic] _[w][-CNFs.]_
10w [,]
Otherwise, either a better query algorithm exists, or a stronger version of the switching lemma can
be established. Either way, the answer would be interesting.
Along this line, we show that there exists a DNF formula of width _w_ and size _⌈2[w]/w⌉_ with
Dave(f ) = _n(1 −_ Θ([log]w[ n]) [).] [It] [indicates] [that] [even] [if] [there] [is] [a] [better] [query] [algorithm,] [the] [room] [for]

improvement is limited when _w_ is large.

**Theorem** **1.4.** For any integer _w_ _∈_ [2 log n, n], there exists a boolean function f : {0, 1}[n] _→{0, 1}_
computable by a DNF formula of width _w_ and size _⌈2[w]/w⌉_ such that


�
Dave(f ) = n 1 − [log][ n]

Θ(w)

3


�
_._


-----

Lastly, we define _penalty_ _shoot-out_ _functions_ in Appendix A, which are monotone balanced
functions, such that the gap between D(f ) and Dave(f ) is arbitrarily large. Moreover, unlike the
worst-case measures D(f ), R(f ), C(f ), bs(f ), s(f ), which are known to be polynomially related

[BW02; ABK16; Hua19; Aar+21], no such polynomial relation holds between _any_ two of the
average-case analogues[2] Dave(f ), Rave(f ), Cave(f ), bsave(f ), save(f ), even for monotone balanced
functions[3].

### 2 Preliminaries

#### 2.1 Boolean functions

Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be a boolean function. The _weight,_ denoted by wt(f ), is the number of
inputs on which _f_ outputs 1. Let _Bn,m_ = _{f_ : _{0, 1}[n]_ _→{0, 1}_ _|_ wt(f ) = _m}_ denote the set of all
_n-variable_ boolean functions with weight _m._
A _restriction_ _ρ_ : _{1, . . ., n}_ _→{0, 1, ⋆}_ is a mapping fixing some variables to 0 or 1. We write
_f_ _|ρ_ for the subfunction of _f_ obtained by fixing its input by _ρ._ Let supp(ρ) = _ρ[−][1]({0, 1})_ denote
the support of the restriction _ρ._ The weight of _x,_ denoted by _|x|,_ is the number of 1’s in _x._ The
bitwise negation of _x_ is denoted by _¬x = (1 −_ _x1, . . ., 1 −_ _xn)._
Let _F_ : _{0, 1}[n]_ _→{0, 1}_ and _G_ : _{0, 1}[m]_ _→{0, 1}_ be two boolean functions. Define the
_composition_ _F_ _◦_ _G_ by
(F _◦_ _G)(x) = F_ (G(x[(1)]), . . ., G(x[(][n][)]))

for _x = (x[(1)], . . ., x[(][n][)]) ∈{0, 1}[nm]_ and _x[(][i][)]_ _∈{0, 1}[m]._
Define _x_ _⪯_ _y_ if and only if _xi_ _≤_ _yi_ for all _i_ _∈{1, 2, . . ., n}._ Say _f_ is _monotone_ if and only if
_f_ (x) ≤ _f_ (y) for all inputs _x ⪯_ _y._
Given a boolean function f : {0, 1}[n] _→{0, 1}, a certificate on input x is a subset S_ _⊆{1, . . ., n}_
such that _f_ (x) = _f_ (y) for any input _y_ _∈{0, 1}[n]_ satisfying _xi_ = _yi_ for all _i_ _∈_ _S._ The certificate
complexity Cx(f ) on input _x_ is the size of the smallest certificate on input _x._

#### 2.2 Decision trees

A (deterministic) decision tree _T_ is a binary tree. Each internal node is labeled by some integer
_i_ _∈{1, 2, . . ., n},_ and the edges and the leaves are labeled by 0 or 1. Repeatedly querying _xi_
and following the edge labeled by _xi,_ the decision tree _T_ finally reaches a leaf and outputs the
leaf’s label, called the value _T_ (x) of _T_ on input _x._ The cost of deciding the value _T_ (x), denoted
by cost(T, x), is the length of the root-to-leaf path which _T_ passes through. The depth of _T_ is
the maximum cost maxx∈{0,1}n cost(T, x). We say _T_ computes _f_ (with zero error) if _T_ (x) = _f_ (x)
for every _x_ _∈{0, 1}[n]._ A query algorithm queries some variables and determines the value of the
function; a query algorithm can be viewed as a family of decision trees.

2These average-case counterparts are defined in the uniform distribution.
3Super-polynomial gaps can be demonstrated using the threshold function [AW01], the tribes function, and Maj ◦
AND, all of which are monotone. An extra trick can make them balanced: given a monotone _f_, let _g_ = Maj(f, f _[†], z)_
for _z_ _∈{0, 1},_ where _f_ _[†]_ denotes _f_ ’s dual [ODo14].

4


-----

#### 2.3 Circuits and formulas

A _clause_ is a logical OR of variables or their negations, and a _term_ is a logical AND of variables
or their negations. A _conjunctive_ _normal_ _form_ _(CNF)_ formula is a logical AND of clauses, and a
_disjunctive_ _normal_ _form_ _(DNF)_ formula is a logical OR of terms. The _size_ of a CNF (respectively,
DNF) formula is the number of the clauses (respectively, the terms). The _width_ of a CNF (respectively, DNF) formula is the maximum variable number of the clauses (respectively, the terms).
A _circuit_ _F_ is a directed acyclic graph with _n_ nodes of no incoming edge, called _sources,_ and
a node of no outgoing edge, called _sink._ Apart from the sources, the other nodes are called _gates._
Each gate is labeled by AND, OR or NOT, and each AND (respectively, OR, NOT) node computes
the logical AND (respectively, OR, NOT) of the values of its incoming nodes. The fan-in of a gate
is the number of its incoming edges, and the _fan-out_ of a gate is the number of its outgoing edges.
The fan-in of NOT gates is fixed to 1. The _size_ of _F_ is the number of the gates. The _depth_ is
the length of the longest path between the sink and the sources. Obviously, every CNF or DNF
formula can be represented as a circuit. An AC[0] circuit is a circuit of polynomial size, constant
depth and AND/OR gates with unbounded fan-in. A _formula_ is a circuit of gates with fan-out 1.

#### 2.4 Query complexities

Let _f_ : {0, 1}[n] _→{0, 1}_ be a boolean function. The worst-case deterministic query complexity of a
boolean function f, denoted by D(f ), is the minimum depth of decision trees that compute f . The
average-case deterministic query complexity of a boolean function _f_, denoted by Dave(f ), is the
minimum average depth of zero-error deterministic decision trees that compute _f_ under a uniform
input distribution.

**Definition** **2.1.** The average-case deterministic query complexity of _f_ : _{0, 1}[n]_ _→{0, 1}_ under a
uniform distribution is defined by

Dave(f ) = min E
_T_ _x∈{0,1}[n][[cost(][T, x][)]][,]_

where _T_ is taken over all zero-error deterministic decision trees that compute _f_ .

Dave(f ) turns out to equal the average-case zero-error _randomized_ query complexity, defined
as minT Ex∈{0,1}n[cost(T, x)], where _T_ is taken over all zero-error _randomized_ decision trees that
compute _f_ (see Remark 8.63 in [ODo14]).

### 3 Fixed-weight functions

#### 3.1 Upper bound

As a warm-up, the following proposition gives the exact value of Dave(f ) for boolean functions _f_
with weight 1, such as the AND function. For convenience, we say input _x_ is a _black_ _point_ (with
respect to _f_ ) if _f_ (x) = 1.

**Proposition** **3.1.** Dave(f ) = 2(1 − 2[1][n][ )] [for] [any] _[n][-variable]_ [boolean] [function] _[f]_ [with] [wt(][f] [) = 1.]


_Proof._ Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be a boolean function with a unique black point _z_ _∈{0, 1}[n]._ We
prove Dave(f ) = 2(1 − 2[1][n][ )] [by] [induction] [on] _[n][.]_ [When] _[n][ = 1,]_ [we] [have] [D][ave][(][f] [) = 2(1][ −] 2[1][n][ ) = 1.]

5


-----

Suppose an optimal query algorithm queries _xi_ first. If _xi_ _̸=_ _zi,_ the algorithm outputs 0.
Otherwise, the algorithm continues on the subfunction _f_ _|xi=zi._ Therefore,

Dave(f ) = 1 + Pr _[̸][=][ z][i][]][ ·][ 0 +]_ Pr [=][ z][i][]][ ·][ D][ave][(][f] _[|][x][i][=][z][i][)]_
_x∈{0,1}[n][ [][x][i]_ _x∈{0,1}[n][ [][x][i]_


� 1

= 1 + [1]2 _[·][ 2]_ 1 − 2[n][−][1]


�


�
= 2 1 − [1]

2[n]


�
_,_


where the second step is by the induction hypothesis.

Next, we show a simple bound Dave(f ) ≤ log wt(f ) + O(1) for any _f_ . Say a query algorithm is
_reasonable_ if it terminates as soon as the subfunction becomes constant.

**Lemma** **3.2.** Dave(f ) ≤ log wt(f ) + 2 for any non-zero boolean function _f_ .

_Proof._ Let _m_ = wt(f ). We prove by induction on _m_ and _n._ When _m_ = 1, we have Dave(f ) =
2(1 − [1] [by] [Proposition] [3.1.] [When] _[n][ = 1,]_ [D][ave][(][f] [)][ ≤] [1.]

2[n][ )][ ≤] [2]
Suppose _xi_ is queried first. Let _m0_ = wt(f _|xi=0)_ and _m1_ = wt(f _|xi=1)._ If _mb_ = 0 for some
_b ∈{0, 1},_ a reasonable algorithm will stop on a constant subfunction _f_ _|xi=b._ Thus,

Dave(f ) ≤ 1 + [1]

2 [(log][ m][ + 2)][ ≤] [log][ m][ + 2]

by the induction hypothesis. Otherwise, by the induction hypothesis and the AM-GM inequality,
we have

Dave(f ) ≤ 1 + [1] [1]

2 [(log][ m][0][ + 2) +] 2 [(log][ m][1][ + 2)]

= log(2[√]m0m1) + 2


_≤_ log m + 2.

We introduce concepts that will be used later. Suppose f : {0, 1}[n] _→{0, 1}_ has _m_ black points
_x[(1)], . . ., x[(][m][)]_ _∈{0, 1}[n]_ in lexicographical order. We call _ci_ = (x[(1)]i _[, . . ., x]i[(][m][)])_ the _column_ _pattern_
of coordinate _i._ Coordinates _i, j_ are _positively_ _(negatively)_ _correlated_ if _ci_ = _cj_ (ci = _¬cj)._ An
_equivalent_ _coordinate_ _set_ (ECS) is a set of correlated coordinates.
Say a coordinate set _S_ _⊆{1, . . ., n}_ is _pure_ if each _ci_ for _i_ _∈_ _S_ is either all-zero or all-one;
otherwise, _S_ is _mixed._ For example, in Table 1, the set _{5, 9, 11}_ is pure, since _c5_ = c9 = (0, 0) and
_c11_ = (1, 1); the set _{1, 2, 3}_ is mixed, since _c1_ = c3 = (0, 1) and _c2_ = (1, 0).

black points _x1_ _x2_ _x3_ _x4_ _x5_ _x6_ _x7_ _x8_ _x9_ _x10_ _x11_
_x[(1)]_ 0 1 0 1 0 1 0 0 0 0 1
_x[(2)]_ 1 0 1 0 0 1 0 1 0 1 1

Table 1: A 11-variable boolean function with weight 2.

6

|black points|x x x x x x x x x x x<br>1 2 3 4 5 6 7 8 9 10 11|
|---|---|
|_x_(1)<br>|0<br>1<br>0<br>1<br>0<br>1<br>0<br>0<br>0<br>0<br>1|
|_x_(2)|1<br>0<br>1<br>0<br>0<br>1<br>0<br>1<br>0<br>1<br>1|


-----

**Proposition** **3.3.** If coordinates i, j are positively (negatively) correlated, then for any x ∈{0, 1}[n]

with _f_ (x) = 1, we have _xi_ = xj (xi _̸= xj)._

**Proposition** **3.4.** Let _S_ be a mixed ECS. For any coordinate _i_ _∈_ _S_ and _v_ _∈{0, 1},_ we have
wt(f _|xi=v) < wt(f_ ).

**Proposition** **3.5.** Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be a boolean function such that _n_ _>_ _k · 2[wt(][f]_ [)][−][1], then
_f_ has an ECS of size at least _k + 1._

It is straightforward to prove Propositions 3.3 and 3.4. Proposition 3.5 follows from the pigeonhole principle, since there are 2[m][−][1] distinct equivalence classes with respect to correlation. Using
these facts, one can prove that any boolean function of weight _O(log n)_ has constant Dave(f ).

**Lemma** **3.6.** Let _f_ : {0, 1}[n] _→{0, 1},_ where wt(f ) < log n. We have Dave(f ) ≤ 5.

_Proof._ Let _m = wt(f_ ) ≥ 3. (Lemma 3.6 follows directly from Lemma 3.2 if wt(f ) < 3.) We prove
Dave(f ) ≤ 5 by induction on _n._
By Proposition 3.5, there exists a maximal ECS I = {i1, . . ., ik} of size k _≥_ 3, since n > 2[wt(][f] [)] =
2 · 2[wt(][f] [)][−][1]. Without loss of generality, assume that coordinates _i1, . . ., ik_ are positively correlated.
By Proposition 3.3, we have _xi1_ = · · · = xik for any black point _x ∈{0, 1}[n]._
If _|I|_ = _n,_ then any black point _x_ must satisfy _x1_ = _· · ·_ = _xn._ Therefore, the only possible
black points are the all-zero vector and the all-one vector, so there are at most 2 black points. By
Lemma 3.2, Dave(f ) ≤ log wt(f ) + 2 ≤ 3.
From now on, we assume _|I| < n,_ and thus there exists a coordinate j _̸∈_ _I._ Let _J_ be a maximal
ECS that contains _j._ Notice that _I_ or _J_ is mixed, since at most one of _f_ ’s maximal ECSs can be
pure.
For notational convenience, let _ρu,v_ denote the restriction fixing _xi1_ and _xi2_ to _u,_ _xj_ to _v,_ and
leaving all other variables free. Our query algorithm _T_ is defined as follows:

(1) query _xi1, xi2;_

(2) output 0 if _xi1_ _̸= xi2;_

(3) if _xi1_ = xi2, then query _xj,_ and apply the query algorithm recursively on the subfunction.

The query algorithm _T_ correctly computes _f_ . Input _x_ cannot be a black point if _xi1_ _̸= xi2,_ since _i1_
and _i2_ are positively correlated (Proposition 3.3).
For any _u, v_ _∈{0, 1},_ the number of inputs of _f_ _|ρu,v_ is _n −_ 3, and wt(f _|ρu,v_ ) ≤ _m −_ 1 since _I_ or
_J_ is mixed (Proposition 3.4). Observe that _n −_ 3 > 2[m] _−_ 3 > 2[m][−][1] _≥_ 2[wt(][f] _[|][ρu,v][ )]_ for any _m ≥_ 3. By
the induction hypothesis, we have Dave(f _|ρu,v_ ) ≤ 5.
Let us analyze the average cost of _T_ . First, notice that the probability of querying exactly 2
variables is [1] [the] [probability] [of] [querying] [at] [least] [3] [variables] [is] [1] [Thus,] [we] [conclude] [that]

2 [;] 2 [.]

Dave(f ) ≤ E _T, x)] ≤_ [1]
_x [cost(_ 2 _[·][ 2 + Pr]x_ [[cost(][T, x][)][ ≥] [3]][ ·][ (3 + 5) = 5][.]


**Corollary** **3.7.** Let _f_ : {0, 1}[n] _→{0, 1}_ be a boolean function. If wt(f ) ≤ 4 log n, then Dave(f ) ≤
40.

7


-----

_Proof._ We have Dave(f1(x) ∨· · · ∨ _fk(x)) ≤_ Dave(f1) + · · · + Dave(fk) for any _f1, f2, . . ., fk._ This is
because we can query f1(x), _f2(x), . . .,_ _fk(x) one by one and compute f1(x)_ _∨· · ·∨_ _fk(x) afterward._
The expected number of variables queried is at most the sum of the individual expectations.
Let _k_ = 8 and _Bf_ denote _f_ ’s on-set (the set of inputs on which _f_ outputs 1). Partition _Bf_
into _k_ disjoint sets _B1 . . ., Bk,_ where _|Bi|_ _≤⌈_ [4 log]8 _[ n]_ _⌉._ Each _Bi_ is the on-set of some function _fi._

It can be verified that _f_ (x) = _f1(x) ∨· · · ∨_ _fk(x)_ and that wt(fi) = _|Bi|_ _≤⌈_ [4 log]8 _[ n]_ _⌉._ Note that

wt(fi) _≤⌈_ [4 log]8 _[ n]_ _⌉_ _<_ [1]2 [log][ n][ + 1] _[≤]_ [log][ n] [when] _[n]_ _[≥]_ [4.] [(When] _[n]_ _[<]_ [4,] [the] [corollary] [holds] [clearly.)]

Thus, by Lemma 3.6, Dave(fi) ≤ 5 for any _i,_ implying that Dave(f ) ≤ [�][8]i=1 [D][ave][(][f][i][)][ ≤] [40.]

To prove Theorem 1.1, we design a query algorithm and analyze its cost. Recall that in the
proof of Lemma 3.2, we considered _any_ reasonable query algorithm, which queries an arbitrary bit
and terminates until the remaining function becomes constant. Similarly, to prove Theorem 1.1,
we design a more sophisticated algorithm, which queries an arbitrary variable until the subfunction
satisfies the following border condition: Dave(f ) = O(1) if wt(f ) ≤ 4 log n (Corollary 3.7); then the
query algorithm used in the proof of Corollary 3.7 is invoked.

_Proof_ _of_ _Theorem_ _1.1._ Let _m_ = wt(f ). If _m_ _≤_ 4 log n, we have Dave(f ) = _O(1)_ by Corollary 3.7.
We will prove
_m_ _m_
Dave(f ) ≤ log (3)
log n [+ log log] log n [+ 87][,]

by induction on _n_ for any _f_ : {0, 1}[n] _→{0, 1}_ with wt(f ) = m ≥ 4 log n.
First, when _m_ _≥_ _n,_ the inequality (3) directly follows from Lemma 3.2 because log m + 2 _≤_
log _m_ _m_ [if] [and] [only] [if] _[m][ ≥]_ _[n][2][−][85][ log][ n][.]_ [From] [now] [on,] [we] [assume] [that] _[m][ ≤]_ _[n][.]_
log n [+ log log] log n [+ 87]
Our query algorithm is as follows:

(1) If wt(f ) ≥ _n,_ apply Lemma 3.2, which implies (3) as we have shown.

(2) Otherwise, query any

� _m_ _m_ �
_ℓ_ = log
log n [+ log log] log n [+ 3]

variables, say, _xi1, . . ., xiℓ._

(3) Given the values of xi1, xi2, . . ., xiℓ, say xi1 = c1, . . ., xiℓ = cℓ, apply our algorithm recursively
to the subfunction _f_ _|xi1_ =c1,...,xiℓ =cℓ.

Let _ρ1, . . ., ρ2ℓ_ enumerate all restrictions that fix _xi1, . . ., xiℓ,_ while leaving all remaining variables undetermined. Averaging over all Dave(f _|ρi),_ we have

Dave(f ) ≤ _ℓ_ + E
_i_ [[D][ave][(][f] _[|][ρ][i][)]]_

_≤_ _ℓ_ + Pr
_i_ [[wt(][f] _[|][ρ][i][)][ ≤]_ [4 log][ n][]][ ·][ E]i [[D][ave][(][f] _[|][ρ][i][)][ |][ wt(][f]_ _[|][ρ][i][)][ ≤]_ [4 log][ n][]]

+ Pr (4)
_i_ [[wt(][f] _[|][ρ][i][)][ >][ 4 log][ n][]][ ·][ E]i_ [[D][ave][(][f] _[|][ρ][i][)][ |][ wt(][f]_ _[|][ρ][i][)][ >][ 4 log][ n][]][ .]_

We have wt(f ) = [�]i[2]=1[ℓ] [wt(][f] _[|][ρ]i[)]_ [and] [E][i][[wt(][f] _[|][ρ]i[)] =]_ 2[m][ℓ] _[≤]_ [1]8 _[·]_ logloglog nm n [.] [By] [Markov’s] [inequality,]


1

Pr _≤_ [1] _m_ _._
_i_ [[wt(][f] _[|][ρ][i][)][ >][ 4 log][ n][]][ ≤]_ [E][i][[wt(]4 log[f] n[|][ρ][i][)]] 32 _[·]_ log

log n

8


-----

We bound Dave(f _|ρi)_ based on the weight of _f_ _|ρi._
**Case** **1:** wt(f _|ρi)_ _≤_ 4 log n. Since _m_ _≤_ _n,_ we have _ℓ_ _≤_ log n + log log n + 4. So the number of
variables in _f_ _|ρi_ is _n −_ _ℓ_ _≥_ _n −_ (log n + log log n + 4) ≥ _[√]n_ (when _n_ is large enough). Notice that
wt(f _|ρi) ≤_ 4 log n ≤ 8 log(n − _ℓ)._ Thus, by Corollary 3.7, we have Dave(f _|ρi) ≤_ 80.
**Case** **2:** wt(f _|ρi)_ _>_ 4 log n. Note that wt(f _|ρi)_ _≥_ 4 log n _>_ 4 log(n − _ℓ)._ By the induction
hypothesis, we have

_m_ _m_
Dave(f _|ρi) ≤_ log log(n − _ℓ)_ [+ log log] log(n − _ℓ)_ [+ 87]

_m_
_≤_ 2 log
log(n − _ℓ)_ [+ 87]

_m_
_≤_ 2 log
log n [+ 89][,]

where the second step is because log log _m_ _m_
log(n−ℓ) _[≤]_ [log] log(n−ℓ) [, and the third step is because][ n] _[−]_ _[ℓ]_ _[≥]_
_n −_ (log n + log log n + 4) ≥ _[√]n._
Combining the two cases and plugging them into (4), we have

Dave(f )

_m_ _m_ 1 � _m_ �
_≤_ log [1] _m_ _·_ 2 log
log n [+ log log] log n [+ 4 + 80 +] 32 _[·]_ log log n [+ 89]

log n


_m_ _m_ 89
= log log n [+ log log] log n [+ 84 +] 16[1] [+] 32 · log _m_

log n

_m_ _m_
_≤_ log
log n [+ log log] log n [+ 87][.]

To conclude, if _m_ _≥_ 4 log n, the right-hand side of (3) is at most log _m_ _m_
log n [+][ O][(log log] log n [),]
completing the proof of Theorem 1.1.

#### 3.2 Lower bound

In this section, we prove Theorem 1.2, showing that Theorem 1.1 is tight up to an additive logarithmic term.
To illustrate the idea of the proof, let us take the XORn function as an example. Regardless
of which variable is queried next, the black points are evenly partitioned, and the subfunction’s
weight is exactly halved. Since XORn has weight 2[n][−][1] and the algorithm must continue until the
subfunction becomes constant, it must query all n variables for every input. Thus, Dave(XORn) = n.
Similarly, the key idea of our proof is to show that most boolean functions exhibit a similar property:
regardless of which variable is queried next, the black points are split into two roughly equal halves.
In other words, for almost all _f_ _∈Bn,m,_ where _Bn,m_ = {f : {0, 1}[n] _→{0, 1} | wt(f_ ) = m}, we shall
prove wt(f _|P )_ is “close” to 2[−][k]m for _any_ tree path _P_ querying _k_ = _ϵ log m_ variables. (Ignoring
the output of _P_, we view a tree path _P_ as a restriction, with _f_ _|P_ representing the subfunction
restricted to _P_ .)
Now, we explain the proof strategy in more detail. To sample f _∈Bn,m_ uniformly, a straightforward approach proceeds as follows: (1) randomly select _m_ distinct inputs _x[(1)], · · ·_ _, x[(][m][)]_ _∈{0, 1}[n];_
(2) set _f_ (x[(][i][)]) = 1 for all _i;_ and (3) set the remaining inputs to 0. This can also be done by
repeatedly drawing _m_ vectors from _{0, 1}[n]_ without replacement and placing them into _m_ vectors

9


-----

_y[(1)], . . ., y[(][m][)]_ _∈{0, 1}[n]._ However, to estimate the probability that wt(f _|P ),_ where _f_ _∈Bn,m,_ is
close to 2[−][k]m for any length-k tree path, we adopt a different sampling method.
Fix a tree path _P_, viewed as a restriction. Instead of sampling a random _f_ _∈Bn,m_ and then
estimating wt(f _|P ),_ we choose to sample wt(f _|P )_ directly, where _f_ _∈Bn,m,_ using the following
method.
Fix a tree path _P_ of length _k,_ where

_P_ = xi1 _−→v1_ _xi2_ _−→· · · −→v2_ _xik_ _−→vk_ _c_ (5)

and _c_ _∈{0, 1}_ is the output of the path. We denote the restriction _f_ _|x1=v1,···,xk=vk_ by _f_ _|P ._ In
_k_ rounds, for _j_ = 1, . . ., k, we sample without replacement from a box with 2[n][−][j] 0’s and 2[n][−][j]

1’s; place the numbers in the _ij-th_ position of each vector, and discard vectors with (¬vj) at _ij-th_
position. (We can safely discard these vectors, because they are not counted in the weight of f _|P .)_
At the end of _k_ rounds, wt(f _|P )_ vectors remain.
Specifically, we sample wt(f _|P )_ as follows, given a fixed path _P_ defined in (5), where _f_ _∈Bn,m_
uniformly:

(1) Let _y[(1)], . . ., y[(][m][)]_ _∈{0, 1, ⋆}[n]_ be the _m_ vectors, where all elements are set to _⋆_ initially.

(2) In the first round, we sample _t0_ = _m_ numbers without replacement from a box with 2[n][−][1]

zeros and 2[n][−][1] ones. We then assign these numbers sequentially to the positions yi[(1)]1 _[, . . ., y]i[(]1[m][)][,]_
that is, the _i1-th_ position of all the _m_ vectors. After that, discard the vectors with (¬v1) at
position _i1,_ that is, _yi[(]1[p][)]_ [=][ ¬][v][1][.] [Let] _[t][1]_ [be] [number] [of] [remaining] [vectors,] [where] _[t][1]_ [is] [a] [random]
variable equal to wt(f _|xi1_ =v1).

(3) In the second round, we sample _t1_ numbers without replacement from a box with 2[n][−][2] zeros
and 2[n][−][2] ones, since f _|xi1_ =v1,xi2 =0 and f _|xi1_ =v1,xi2 =1 have 2[n][−][2] inputs. Assign these numbers
sequentially to the _i2-th_ position of the remaining _t1_ vectors, and discard the vectors with
(¬v2) at position _i2,_ i.e., _yi[(]2[p][)]_ = _¬v2._ Let _t2_ be number of remaining vectors, where _t2_ is a
random variable equal to wt(f _|xi1_ =v1,xi2 =v2).

(4) Proceed for _k_ rounds. The number of remaining vectors _tk_ is a random variable equal to
wt(f _|P ) = wt(f_ _|xi1_ =v1,...,xik =vk ).


Recall that tj is the number of vectors remaining after the j-th round, and tk = wt(f _|P )._ If path
_P_ correctly computes _f_ (on input _x ∈{0, 1}[n]_ such that _xi1_ = v1, _. . .,_ _xik_ = vk), then we must have
wt(f _|P )_ = 0 or wt(f _|P )_ = 2[n][−][k]. Intuitively, in each round, _ti_ _≈_ [1]2 _[t][i][−][1]_ [holds] [with] [high] [probability]

by Hoeffding’s inequality, so it takes Ω(log n) rounds to make _tk_ = 0. Thus, it is unlikely that a
“short” path computes _f_ .

**Definition** **3.8** (δ-parity path). Let _P_ = xi1 _−→v1_ _xi2_ _−→· · · −→v2_ _xik_ _−→vk_ _c_ be a path of length _k._ Let
_ρj_ denote the restriction that fixes _xip_ to _vp_ for _p = 1, . . ., j,_ leaving other variables undetermined.
The path _P_ is called _δ-parity_ with respect to _f_ : {0, 1}[n] _→{0, 1}_ if


for each _j_ = 1, . . ., k.


1 wt(f _|ρj_ )
2 [(1][ −] _[δ][)][ ≤]_ wt(f _|ρj−1)_ _[≤]_ [1]2 [(1 +][ δ][)]

10


-----

**Lemma** **3.9** (Hoeffding’s inequality[Hoe63; Ser74]). Let _X1, X2, . . ., Xm_ be independent random
variables such that 0 ≤ _Xi_ _≤_ 1, and let _Sm_ = X1 + X2 + . . . + Xm. For any _t > 0,_ we have


_._ (6)


Pr[|Sm − E[Sm]| ≥ _t] ≤_ 2 exp


�

_−_ [2][t][2]

_m_


�


The inequality (6) also holds when _X1, . . ., Xm_ are obtained by sampling without replacement.

**Lemma** **3.10.** Let f : {0, 1}[n] _→{0, 1} be a boolean function with wt(f_ ) = m. Let P be a decision
tree path of length at most _ϵ log m._ For any _δ_ _∈_ (0, 1 [and] _[ϵ][ ∈]_ [(0][,][ 1),]
2ϵ log m []]

� �
Pr [is] [not] _[δ][-parity]_ [for] _[f]_ []][ <][ 2][ϵ][ log][ m][ ·][ exp] _−_ [1] _._
_f_ _∼Bn,m_ [[][P] 2 _[·][ δ][2][m][1][−][ϵ]_


_Proof._ Let _b_ _≤_ _ϵ log m_ denote the length of _P_ . The random variable _tj_ equals wt(f _|ρj_ ), where _f_
is sampled uniformly at random from _Bn,m._ Let _Xj,1,_ _. . .,_ _Xj,tj−1_ _∈{0, 1}_ be random variables
indicating whether each of the _tj−1_ vectors is in the on-set of _f_ _|ρj_, i.e., whether it remains after
the _j-th_ round. Random variables _Xj,1, . . ., Xj,tj−1_ _∈{0, 1}_ are obtained from 2[n][−][j] zeros and 2[n][−][j]

ones by sampling without replacement. We have _tj_ = Xj,1 + · · · + Xj,tj−1 and E[tj] = 2[1] _[t][j][−][1][.]_

Let _α =_ [1] [and] _[β]_ [=] [1] [We] [have]

2 [(1][ −] _[δ][)]_ 2 [(1 +][ δ][).]

Pr [is] [not] _[δ][-parity]_ [with] [respect] [to] _[f]_ []]
_f_ _∼Bn,m_ [[][P]


= 1 − Pr




 _b_ 

 � _tj_ _∈_ [αtj−1, βtj−1]

_j=1_


j−1 

 � _tk_ _∈_ [αtk−1, βtk−1]

_k=1_




_._ (7)



= Pr


∃1 ≤ _j_ _≤_ _b_ s.t. _tj_ _̸∈_ [αtj−1, βtj−1] ∧


Let _Aj_ be the event that _tj_ _∈_ [αtj−1, βtj−1], and let _Bj_ be the event _tj_ _∈_ �αjm, βjm�. By a union
bound, (7) is at most


j−1

 � _Ak_

_k=1_




_≤_




_b_
� Pr [¬Aj _∧_ _Bj−1] ._ (8)

_j=1_


_b_
� Pr

_j=1_




¬Aj _∧_


If event _Aj_ does not occur, we have _tj_ _>_ _βtj−1_ = [1]2 [(1 +][ δ][)][t][j][−][1] [or] _[t][j]_ _[<]_ _[αt][j][−][1]_ [=] [1]2 [(1][ −] _[δ][)][t][j][−][1][,]_

which implies _|tj_ _−_ E[tj]| > [1]2 _[δt][j][−][1][.]_ [By] [Hoeffding’s] [inequality] [(Lemma] [3.9),]


� � � �
Pr [¬Aj _∧_ _Bj−1] ≤_ 2 exp _−_ [1] _≤_ 2 exp _−_ [1] _,_ (9)

2 _[δ][2][t][j][−][1]_ 2 _[α][j][−][1][δ][2][m]_


since _tj−1_ _≥_ _α[j][−][1]m_ by _Bj−1._


11


-----

Finally, plugging (9) into (8), we have

Pr [is] [not] _[δ][-parity]_ [with] [respect] [to] _[f]_ []]
_f_ _∼Bn,m_ [[][P]


_≤_


_b_

� �

� 2 exp _−_ [1]2 _[·][ α][j][−][1][δ][2][m]_

_j=1_


� � � �
_≤_ 2b · exp _−_ [1] _α =_ [1]

2 _[·][ α][b][−][1][δ][2][m]_ 2 [(1][ −] _[δ][)][ <][ 1]_


� 1
1 −
2ϵ log m


�ϵ log m−1 � � 1
_δ[2]m_ _δ_ _≤_

2ϵ log m


_≤_ 2b · exp


� 1

_−_ [1]

2 _[·]_ 2[b][−][1]


�


�x−1
_>_ [1]

2


�


�
_≤_ 2b · exp _−_ [1] _[m]_

2 _[δ][2][ ·]_ 2[b]


� ��
1 − [1]

2x


� � � �
_≤_ 2ϵ log m · exp _−_ [1]2 _[·][ δ][2][m][1][−][ϵ]_ _._ _b ≤_ _ϵ log m_


**Definition** **3.11** ((t, δ)-parity function). Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be a boolean function. The
function _f_ is called (t, δ)-parity if any path of length at most t is a _δ-parity_ path with respect to f .

**Lemma** **3.12.** Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be a (t, δ)-parity function with wt(f ) _≤_ 2[n][−][1] satisfying
1 ≤ _t ≤_ log wt(f ) − 1 and _δ_ _≤_ [1] [Then,] [min][x][∈{][0][,][1][}][n][ C][x][(][f] [)][ ≥] _[t][.]_

2t [.]

_Proof._ Let _P_ = _xi1_ _−→v1_ _xi2_ _−→· · ·v2_ _−→_ _xik_ _−→vk_ _c_ be any decision tree path of length _k_ _≤_ _t._ By
Definition 3.8, we have

1 _[≤]_ [wt(][f] _[|][ρ][j]_ [)]

= [wt(][f] _[|][ρ][1][)]_ _·_ [wt(][f] _[|][ρ][2][)]_ [wt(][f] _[|][ρ][j]_ [)] (10)

2[j][ (1][ −] _[δ][)][j]_ wt(f ) wt(f ) wt(f _|ρ1)_ _[· · ·]_ wt(f _|ρj−1)_ _[≤]_ 2[1][j][ (1 +][ δ][)][j]


for _j_ = 1, 2, . . ., k, where _ρj_ is the restriction that fixes _xip_ to _vp_ for _p = 1, 2, . . ., j._ Thus, we have
1 _[≤]_ [wt(][f] _[|][ρ][j]_ [)] _[≤]_ 1 [Note] [that] [(10)] [holds] [for] [any] [decision] [tree] [path] [of]
2[j][ (1][ −] _[δ][)][j][ wt(][f]_ [)] 2[j][ (1 +][ δ][)][j][ wt(][f] [).]
length _j._ So we have
1

(11)

2[j][ (1][ −] _[δ][)][j][ wt(][f]_ [)][ ≤] [wt(][f] _[|][ρ][)][ ≤]_ 2[1][j][ (1 +][ δ][)][j][ wt(][f] [)]

for any restriction _ρ_ fixing _j_ variables, where _j_ _≤_ _t._
On one hand, from (11), we have

�
wt(f _|ρ) ≤_ (1 + δ)[t]2[n][−][j][−][1] wt(f ) ≤ 2[n][−][1][�]


� �t � �

_≤_ [1] 1 + [1] 2[n][−][j] _δ_ _≤_ [1]

2 2t 2t

_√_
_≤_ _e_ _[≤]_ [0][.][8244][ ·][ 2][n][−][j][.] ��1 + [1]

2 _[·][ 2][n][−][j]_ 2n


�n �
_≤_ _[√]e_


12


-----

On the other hand, by (11), we have

� �
wt(f _|ρ) ≥_ (1 − _δ)[t]2[t][−][j][+1]_ _t ≤_ log wt(f ) − 1


� �t � �
_≥_ 2 1 − [1] _δ_ _≤_ [1] _[j]_ _[≤]_ _[t]_

2t 2t [and]

�� �x �
_≥_ 1. 1 − [1] _≥_ [1] _[x][ ≥]_ [1]

2x 2 [when]


Thus, wt(f _|ρ)_ is less than 2[n][−][j] and larger than 0, which implies _f_ _|ρ_ cannot be constant for any
restriction _ρ_ fixing at most _t_ variables. Therefore, we conclude minx∈{0,1}n Cx(f ) ≥ _t._

Finally, one can prove Theorem 1.2 using Lemmas 3.10 and 3.12.

_Proof_ _of_ _Theorem_ _1.2._ Let _m = m(n)_ and


1
_ϵ = 1 −_
log m


� _m_ �
log log n + 3 log log _._
log n [+ 5]


Since cost(T, x) ≥ Cx(f ) for any _x ∈{0, 1}[n],_ Dave(f ) ≥ minx∈{0,1}n Cx(f ), Our goal is to prove


Pr
_f_ _∼Bn,m_


� _m_ _m_ �

min
_x∈{0,1}[n][ C][x][(][f]_ [)][ < ϵ][ log][ m][ = log] log n _[−]_ [3 log log] log n _[−]_ [5]


_→_ 0


as _n →∞._ Since 4 log n ≤ _m ≤_ 2[n][−][1], _m, n_ tend to infinity simultaneously.
Let _t_ = _ϵ log m_ and _δ_ = (2 log _m_ _[≤]_ [(2][ϵ][ log][ m][)][−][1] [=] 1 [Let] [len(][P] [)] [denote] [the] [length] [of] [a]
log n [)][−][1] 2t [.]
tree path _P_ . By Lemma 3.12, if minx∈{0,1}n Cx(f ) _<_ _t,_ then _f_ is not (b, δ)-parity. That is, there
exists a path _P_ being not _δ-parity._ Thus,


�

min
_x∈{0,1}[n][ C][x][(][f]_ [)][ < ϵ][ log][ m]


Pr
_f_ _∼Bn,m_


�


_≤_ Pr [path] _[P]_ [with] [len(][P] [)][ < t] [such] [that] _[P]_ [is] [not] _[δ][-parity]][ .]_ (12)
_f_ _∼Bn,m_ [[][∃] [tree]

By a union bound, (12) is at most

� Pr [is] [not] _[δ][-parity]_ [for] _[f]_ []]

_f_ _∼Bn,m_ [[][P]
len(P )<t


_≤_


_ϵ log m−1_
�

_k=0_


�n
_k_


� � �

_k!2[k]_ _· exp_ _−_ [1] (Lemma 3.10)

2 _[δ][2][m][1][−][ϵ]_


� �
_≤_ _n[ϵ][ log][ m](ϵ log m)[2][ϵ][ log][ m]_ _· (2ϵ log m) · exp_ _−_ [1]

2 _[δ][2][m][1][−][ϵ]_


�


_._ (13)


_≤_ exp (ln n · 4ϵ log m) · exp


� _m[1][−][ϵ]_

_−_ 8(log _m_

log n [)][2]

13


-----

Then, since _ϵ log m ≤_ log _m_ [(13)] [is] [at] [most]
log n [,]


exp


� _m_ _m[1][−][ϵ]_

log n · (4 ln 2) · log _m_
log n _[−]_ 8(log

log n [)][2]


�


� _m_ �
= exp _−4(1 −_ ln 2) log log n _[·][ log][ n]_

_≤_ exp (−8(1 − ln 2) log n), (m ≥ 4 log n)

which tends to zero as _n →∞._ Therefore, we conclude

_m_ _m_
Dave(f ) ≥ min
_x∈{0,1}[n][ C][x][(][f]_ [)][ ≥] _[ϵ][ log][ m][ = log]_ log n _[−]_ [3 log log] log n _[−]_ [5]

for almost all functions _f_ _∈Bn,m._ That is, Dave(f ) is at least log logm n _[−]_ _[O][(log log]_ logm n [),] [since]
_m ≥_ 4 log n.

### 4 DNFs, circuits, and formulas

In this section, we study Dave(f ) of circuits that consist of AND, OR, NOT gates with unbounded
fan-in.
As a warm-up, we show Dave(F ) = O(s) for general size-s circuits _F_ . The bound is tight up to
a multiplicative factor, since Dave(XORn) = n, and XORn is computable by a circuit of size _O(n)_
and depth _O(log n)._

**Proposition** **4.1.** Dave(F ) ≤ 2s for every circuit _F_ of size _s._

_Proof._ Notice that the average cost of evaluating each AND/OR/NOT gate does not exceed 2
(Proposition 3.1). Therefore, it takes at most 2s queries on average to evaluate _s_ gates.

**Definition** **4.2.** A _p-random_ _restriction,_ denoted by _Rp,_ is a distribution over restrictions leaving
_xi_ unset with probability _p_ and fixing _xi_ to 0 or 1 with equal probability [1]2 [(1][ −] _[p][)]_ [independently]

for each _i = 1, 2, . . ., n._

**Definition** **4.3** ([Ros17; Ros19]). A boolean function _f_ is _λ-critical_ if

Pr
_ρ∼Rp_ [[D(][f] _[|][ρ][)][ ≥]_ _[t][]][ ≤]_ [(][pλ][)][t]

for any _p ∈_ [0, 1] and _t ∈_ N.

The next lemma gives an upper bound on Dave(f ) for _λ-critical_ functions.

**Lemma** **4.4.** Let _f_ : {0, 1}[n] _→{0, 1}_ be _λ-critical._ Then


�
Dave(f ) ≤ _n_ 1 − [1]

_λ_

14


� � _n_
+ 2 (14)
_λ_ _[.]_


-----

_Proof._ Let _ϵ_ _>_ 0 and _p_ = 1 [Since] _[f]_ [is] _[λ][-critical,]_ [we] [have] [Pr][ρ][∼R][p][[D(][f] _[|][ρ][)]_ _[≥]_ _[t][]]_ _[≤]_ [(1 +][ ϵ][)][−][t][.]
(1+ϵ)λ [.]
Consider a query algorithm that queries each variable independently with probability 1 − _p,_ and
then applies a worst-case optimal query algorithm to _f_ _|ρ._ We have

Dave(f ) ≤ E [| supp(ρ)| + D(f _|ρ)]_
_ρ∼Rp_


_n_

= n(1 − _p) +_ � Pr

_ρ∼Rp[[D(][f]_ _[|][ρ][)][ ≥]_ _[t][]]_
_t=1_

_∞_

1

_≤_ _n (1 −_ _p) +_ �

(1 + ϵ)[t]

_t=0_


� 1
= n 1 −
(1 + ϵ)λ


�
+ [1 +][ ϵ]

_ϵ_


�
= n 1 − [1]

_λ_


� + _[n]_ _ϵ_ [1 +][ ϵ] _._

_λ_ _[·]_ 1 + ϵ [+] _ϵ_


Let α = _[n]_ [The function][ h][(][ϵ][) =] _αϵ_ [1+][ϵ] attains its minimum at ϵ = _√_ 1, where h( _√_ 1 ) =

_λ_ _[≥]_ [1.] 1+ϵ [+] _ϵ_ _α−1_ _α−1_

2[√]α. Thus,

� � � _n_
Dave(f ) ≤ _n_ 1 − [1] + 2

_λ_ _λ_ _[.]_

**Remark** **4.5.** Alternatively, one can prove Dave(f ) _≤_ _n(1_ _−_ 21λ [)] [+] _[O][(1)]_ [by] [combining] [the] [OS]
inequality Dave(f ) ≤ log DTsize(f ) [OS06] and the bound DTsize(f ) ≤ _O(2[n][(1][−]_ 2[1]λ [)]) [Ros19].

By combining Lemma 4.4 with the existing bounds on criticality for CNFs, bounded-depth
circuits, and formulas [H˚as86; Ros17; Ros19; H˚as14; HMS23], the following upper bounds on
Dave(f ) can be derived.

**Corollary** **4.6** ([H˚as86]). Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be computable by a CNF/DNF of width _w._
Then
� 1 �
Dave(f ) ≤ _n_ 1 − _._
_O(w)_

**Corollary** **4.7** ([Ros17; Ros19]). Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be computable by a CNF/DNF of size
_s._ Then
� 1 �
Dave(f ) ≤ _n_ 1 − _._
_O(log s)_

**Corollary** **4.8** ([H˚as14; Ros17]). Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be computable by a circuit of depth _d_
and size _s._ Then
� 1 �
Dave(f ) ≤ _n_ 1 − _._
_O(log s)[d][−][1]_

**Corollary** **4.9** ([HMS23]). Let _f_ : _{0, 1}[n]_ _→{0, 1}_ be computable by a formula of depth _d_ and
size _s._ Then

� 1 �

Dave(f ) ≤ _n_ 1 − _O(_ [1] _._

_d_ [log][ s][)][d][−][1]


15


-----

It is natural to ask whether the upper bounds above are tight. A positive answer would suggest
that random restrictions with the same probability p (as was used in the proof of the aforementioned
results) are optimal. Toward this goal, we prove Theorem 1.4, which says there exists a DNF of
width _w_ and size _⌈2[w]/w⌉_ such that Dave(f ) = n(1 − Θ([log]w[ n]) [).]

Here, we provide an outline of the proof and briefly explain how to find such a DNF formula. In
contrast to the O(1) average cost to determine the output of the OR function under a uniform input
distribution (Proposition 3.1), it costs _n(1 −_ _o(1))_ on average under a _p-biased_ input distribution
when _p_ = _o(1/n)_ (Exercise 8.65 in [ODo14]). Our approach is to employ a biased function _g_
(given by Theorem 1.2) with _p_ = Prx∈{0,1}n[g(x) = 1] and Dave(g) = _n(1 −_ _o(1))_ as a “simulator”
of _p-biased_ variable. Then, we show the composition OR _◦_ _g_ is hard to query under a uniform
distribution and is computable by a somewhat small DNF formula. As such, Theorem 1.4 follows.

_Proof_ _of_ _Theorem_ _1.4._ Let _m = ⌈_ [2][w] _[h][ =][ ⌊]_ _[n]_ _[s][ =][ ⌈][2][w][/w][⌉][.]_ [Observe] [that]

2n _[⌉]_ [and] _w_ _[⌋]_ [and]


� 2w � � 2w
_mh ≤_ _·_ _[n]_ [2][w] _[n]_

2n [+ 1] _w_ [=] 2w [+] _w_ _[≤]_ _w_


�
= s.


Since _n ≥_ _w_ _≥_ 2 log n, we have _m ≤_ [2][w] [and] _[m][ ≥]_ [2][w] _[n]_ [By] [Theorem]

2n [+ 1][ ≤] [2][n][−][1] 2n _[≥]_ 2[n]n[2] [=] 2 _[≥]_ [4 log][ w.]

1.2, there exists _g_ _∈Bw,m_ such that


_m_ � _m_
_d =_ min log log
_y∈{0,1}[w][ C][y][(][g][)][ ≥]_ [log] log w _[−]_ _[O]_ log w


�
= w − log n − _O(log w)._


Let _p =_ 2[m][w] [denote] [the] [probability] [that] _[g]_ [outputs] [1.] [Note] [that] _[p][ ≤]_ 21n [+] 21[w] _[≤]_ _n[1]_ [.]

Let

_h_

_f_ (x) = � _g(x[(][k][)]),_ (15)

_k=1_

where _x = (x[(1)], . . ., x[(][h][)]) ∈{0, 1}[n]_ and _x[(1)], . . ., x[(][h][)]_ _∈{0, 1}[w]._ It is obvious that _f_ is computable
by a DNF formula of width _w_ and size _mh ≤_ _s,_ because each individual g is computable by a DNF
formula of width _w_ and size _m._
Let _T_ be any query algorithm computing _f_ . Let us condition on the event _g(x[(1)])_ = _· · ·_ =
_g(x[(][h][)])_ = 0, which happens with probability (1 − _p)[h]._ The algorithm _T_ needs to query at least _d_
variables to evaluate each clause _g(x[(][k][)])._ If _g(x[(1)]) = · · · = g(x[(][h][)]) = 0,_ then _T_ queries at least _hd_
variables. Thus,


Dave(f ) ≥ _hd ·_ Pr
_x∈{0,1}[n]_


� �
_g(x[(1)]) = · · · = g(x[(][h][)]) = 0_


�
_≥_ _n_ 1 − [log][ n][ +][ O][(log][ w][)]

_w_

�
_≥_ _n_ 1 − [log][ n][ +][ O][(log][ w][)]

_w_


� � �
(1 − _p)[h]_ _d ≥_ _w −_ log n − _O(log w)_

� �
(1 − _ph)_ 1 − _ph ≤_ (1 − _p)[h][�]_


�
= _n_ 1 − [log][ n]

Ω(w)


� �
_._ _ph ≤_ [1]

_w_


�
(16)


On the other hand, we can query _f_ by evaluating all the clauses one by one. By Lemma 3.2,
Dave(g) ≤ log m + 2 ≤ _w −_ log n + 2. Thus,


�
Dave(f ) ≤ _h · Dave(g) ≤_ _n_ 1 − [log][ n][ −] [2]

_w_

16


� �
= n 1 − [log][ n]

_O(w)_


�
_._ (17)


-----

�
Finally, combining (16) and (17), we conclude Dave(f ) = n 1 − Θ([log]w[ n])

### 5 Conclusion


�
.


In this paper, we studied the average-case query complexity of boolean functions under the uniform
distribution. We prove an upper bound on Dave(f ) in terms of its weight; on the other hand, we
prove that for almost all fixed-weight boolean functions, the upper bound is tight up to an additive
logarithmic term. We show that, for any _w_ _≥_ 2 log n, there exists a DNF formula of width _w_ and
size _⌈2[w]/w⌉_ such that Dave(f ) = n(1 − Θ([log]w[ n]) [),] [which] [suggests] [that] [the] [criticality] [bounds] _[O][(][w][)]_ [and]

_O(log s)_ are tight up to a multiplicative log n factor.
Theorems 1.1 and 1.2 essentially relate Dave(f ) to the zero-order Fourier coefficient _f[�]({∅})._
Establishing an upper bound on Dave(f ) in terms of higher-order Fourier coefficients (such as
influences) would be valuable. For example, it is unclear whether the lower bound Dave(HexL×L) ≥
_L[1][.][5+][o][(1)]_ is tight [Per+07]; bounding Dave(f ) in terms of Fourier coefficients might shed light on
the open problem.
It remains open to prove tight upper bounds on Dave(f ) for _k-DNF,_ as well as for bounded
depth formulas and circuits.

#### Acknowledgements

We are grateful to the anonymous reviewers for their valuable feedback.

### References

[ABK16] Scott Aaronson, Shalev Ben-David, and Robin Kothari. “Separations in query complexity using cheat sheets”. In: Proceedings _of_ _the_ _forty-eighth_ _annual_ _ACM_ _symposium_
_on_ _Theory_ _of_ _Computing._ 2016, pp. 863–876.

[Aar+21] Scott Aaronson, Shalev Ben-David, Robin Kothari, Shravas Rao, and Avishay Tal.
“Degree vs. approximate degree and quantum implications of Huang’s sensitivity theorem”. In: _Proceedings_ _of_ _the_ _53rd_ _Annual_ _ACM_ _SIGACT_ _Symposium_ _on_ _Theory_ _of_
_Computing._ 2021, pp. 1330–1342.

[Amb+16] Andris Ambainis, Kazuo Iwama, Masaki Nakanishi, Harumichi Nishimura, Rudy Raymond, Seiichiro Tani, and Shigeru Yamashita. “Quantum query complexity of almost
all functions with fixed on-set size”. In: _computational_ _complexity_ 25 (2016), pp. 723–
735.

[AW01] Andris Ambainis and Ronald de Wolf. “Average-case quantum query complexity”. In:
_Journal_ _of_ _Physics_ _A:_ _Mathematical_ _and_ _General_ 34.35 (2001), p. 6741.

[BSW05] Itai Benjamini, Oded Schramm, and David B. Wilson. “Balanced boolean functions
that can be evaluated so that every input bit is unlikely to be read”. In: _Proceedings_
_of_ _the_ _thirty-seventh_ _annual_ _ACM_ _symposium_ _on_ _Theory_ _of_ _computing._ 2005.

[BW02] Harry Buhrman and Ronald de Wolf. “Complexity measures and decision tree complexity: a survey”. In: _Theoretical_ _Computer_ _Science_ 288.1 (2002), pp. 21–43.

17


-----

[HMS23] Prahladh Harsha, Tulasimohan Molli, and Ashutosh Shankar. “Criticality of AC[0]Formulae”. In: _38th_ _Computational_ _Complexity_ _Conference_ _(CCC_ _2023)._ Vol. 264.
Leibniz International Proceedings in Informatics (LIPIcs). 2023, 19:1–19:24.

[H˚as86] Johan H˚astad. “Computational limitations for small depth circuits”. PhD thesis. Massachusetts Institute of Technology, 1986.

[H˚as14] Johan H˚astad. “On the correlation of parity and small-depth circuits”. In: SIAM _Jour-_
_nal_ _on_ _Computing_ 43.5 (2014), pp. 1699–1708.

[Hoe63] Wassily Hoeffding. “Probability Inequalities for Sums of Bounded Random Variables”.
In: _Journal_ _of_ _the_ _American_ _Statistical_ _Association_ 58.301 (1963), pp. 13–30. doi:
```
     10.1080/01621459.1963.10500830.

```
[Hua19] Hao Huang. “Induced subgraphs of hypercubes and a proof of the sensitivity conjecture”. In: _Annals_ _of_ _Mathematics_ 190.3 (2019), pp. 949–955.

[JZ11] Rahul Jain and Shengyu Zhang. “The influence lower bound via query elimination”.
In: _arXiv_ _preprint_ _arXiv:1102.4699_ (2011).

[Lee10] Homin K Lee. “Decision trees and influence: An inductive proof of the OSSS inequality”. In: _Theory_ _of_ _Computing_ 6.1 (2010), pp. 81–84.

[ODo+05] R. O’Donnell, M. Saks, O. Schramm, and R.A. Servedio. “Every decision tree has an
influential variable”. In: _46th_ _Annual_ _IEEE_ _Symposium_ _on_ _Foundations_ _of_ _Computer_
_Science_ _(FOCS’05)._ 2005, pp. 31–39.

[ODo14] Ryan O’Donnell. _Analysis_ _of_ _boolean_ _functions._ Cambridge University Press, 2014.

[OS06] Ryan O’Donnell and Rocco A. Servedio. “Learning monotone decision trees in polynomial time”. In: 21st _Annual_ _IEEE_ _Conference_ _on_ _Computational_ _Complexity_ _(CCC’06)_
(2006), pp. 213–225.

[Per+07] Yuval Peres, Oded Schramm, Scott Sheffield, and David B Wilson. “Random-turn hex
and other selection games”. In: _The_ _American_ _Mathematical_ _Monthly_ 114.5 (2007),
pp. 373–387.

[Ros08] Benjamin Rossman. “On the constant-depth complexity of k-clique”. In: _Proceedings_
_of_ _the_ _fortieth_ _annual_ _ACM_ _symposium_ _on_ _Theory_ _of_ _computing._ 2008, pp. 721–730.

[Ros14] Benjamin Rossman. “The monotone complexity of k-clique on random graphs”. In:
_SIAM_ _Journal_ _on_ _Computing_ 43.1 (2014), pp. 256–279.

[Ros17] Benjamin Rossman. _An_ _entropy_ _proof_ _of_ _the_ _switching_ _lemma_ _and_ _tight_ _bounds_ _on_ _the_
_[decision-tree size of AC0. 2017. url: https://users.cs.duke.edu/˜br148/logsize.](https://users.cs.duke.edu/~br148/logsize.pdf)_
```
     pdf.

```
[Ros19] Benjamin Rossman. “Criticality of Regular Formulas”. In: _34th_ _Computational_ _Com-_
_plexity_ _Conference_ _(CCC_ _2019)._ Vol. 137. 2019, 1:1–1:28.

[Ser74] Robert J Serfling. “Probability inequalities for the sum in sampling without replacement”. In: _The_ _Annals_ _of_ _Statistics_ (1974), pp. 39–48.

[SW01] Stanislav Smirnov and Wendelin Werner. “Critical exponents for two-dimensional percolation”. In: _Mathematical_ _Research_ _Letters_ 8 (2001), pp. 729–744.

18


-----

[Yao77] Andrew Chi-Chin Yao. “Probabilistic computations: Toward a unified measure of complexity”. In: 18th Annual Symposium on Foundations of Computer Science (sfcs 1977).
1977, pp. 222–227.

### A Penalty shoot-out function

Besides the AND/OR functions, which are highly biased, there are monotone balanced functions
such that the gap between Dave(f ) and D(f ) is arbitrarily large.
Let us define the _penalty_ _shoot-out_ _function_ PSOn : _{0, 1}[2][n][+1]_ _→{0, 1}_ as follows. Consider
an _n-round_ penalty shoot-out in a football game. In each round, two teams, A and B, each take a
penalty kick in turn, with team A going first. Let x2i−1 = 1 indicate the event team A scores in the
_i-th round, and let x2i_ = 0 — this is to make the function monotone — indicate the event that team
B scores in the _i-th_ round for _i_ = 1, 2, . . ., n. The game continues until one team scores _and_ the
other does not (within the same round). If no winner is declared after 2n kicks, an additional kick
by team A decides the game. In this final kick, if team A scores, team A wins and PSOn(x) = 1;
otherwise, team B wins and PSOn(x) = 0. To the best of our knowledge, PSOn is first studied
here.
Assume both teams have equal probabilities of scoring, that is, PSOn is defined under a uniform
distribution. The function PSOn is a monotone balanced function with Dave(PSOn) = _O(1)_ and
D(PSOn) = Θ(n).

**Proposition** **A.1.** D(PSOn) = 2n + 1 and Dave(PSOn) = 4 − 2[3][n][ .]


_Proof._ In the worst case, the winner cannot be declared until the last round is finished, so D(PSOn) =
2n + 1.
We prove Dave(PSOn) = 4 − 23[n] [by] [induction] [on] _[n][.]_ [When] _[n]_ [=] [0,] [D][ave][(PSO][n][)] [=] [1,] [because]
one kick by team A decides the game. Assuming Dave(PSOn−1) = 4 − 2[n]3[−][1] [holds,] [let] [us] [prove]
Dave(PSOn) = 4 − 2[3][n][ .] [In] [each] [round,] [there] [are] [four] [cases:]

(1) Team A scores, and team B does not.

(2) Team B scores, and team A does not.

(3) Both teams score.

(4) Neither scores.

Each case happens with equal probability [1] [If] [case 1)] [or 2)] [happens,] [the winner] [is decided;] [if case]

4 [.]
3) or 4) happens, the game will continue. Therefore, we have


Dave(PSOn) = [1] [1]

2 _[·][ 2 +]_ 2 _[·][ (D][ave][(PSO][n][−][1][) + 2)]_

� 3 �

= [1] 4 − + 2

2 2[n][−][1]

= 4 − [3]

2[n][,]

completing the induction proof.


19


-----

