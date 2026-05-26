# Flexible Regularized Estimating Equations: Some New Perspectives

#### Yi Yang [∗], Yuwen Gu [†], Yue Zhao [‡], Jun Fan[§]


#### November 3, 2021


Abstract

In this note, we make some observations about the equivalences between regularized
estimating equations, fixed-point problems and variational inequalities. A summary of
our findings is given below.



- A regularized estimating equation is equivalent to a fixed-point problem, specified
by the proximal operator of the corresponding penalty.

- A regularized estimating equation is equivalent to a generalized variational inequality.

- Both equivalences extend to any estimating equations and any penalty functions.


To our knowledge, these observations have never been presented in the literature before.
We hope our new findings can lead to further research in both computational and
theoretical aspects of the regularized estimating equations.

### 1 Introduction


Suppose U(β) = (U1(β), . . ., Up(β))[⊤] is an estimating function for β = (β1, . . ., βp)[⊤] based on
a random sample of size n, where U( ) : R[p] R[p] is a vector-valued function. For example,

                             - →
in maximum likelihood estimation, U(β) is the negative score function. In general, U(β)
may not necessarily correspond to the negative gradient of an objective function, such as a
likelihood. Consider the standard estimating equation

U(β) = 0. (1)


Assume that the solution of (1) exists, which is denoted by β[ˆ]. Note that for any τ - 0,

U(β[ˆ]) = 0 βˆ = βˆ τ U(ˆβ).
⇐⇒ −


∗Department of Mathematics and Statistics, McGill University (yi.yang6@mcgill.ca)
†Department of Statistics, University of Connecticut
‡Department of Mathematics, University of York
§Corresponding author, Department of Mathematics and Statistics, McGill University


-----

This motivates us to rewrite (1) as a fixed-point problem:

find βˆ R[p] such that β[ˆ] = f (β[ˆ]), with f (β) β τ U(β). (2)
∈ ≡ −

We can also show that (1) is equivalent to the following variational inequality problem:

find βˆ R[p] such that U(β[ˆ])[⊤](β β) 0, for all β R[p]. (3)
∈ − [ˆ] ≥ ∈

This is because if U(β[ˆ]) = 0, then inequality (3) holds with equality for all β. Conversely,
if β[ˆ] satisfies (3), we can choose β = β[ˆ] U(β[ˆ]), which implies that U(β[ˆ])[⊤]U(β[ˆ]) 0 and
− − ≥
therefore U(β[ˆ]) = 0.
These results may have very little practical relevance, but it raises an interesting question,
that is, whether the equivalences between estimating equations, fixed-point problems and
variational inequalities carry over to the regularization setting?

### 2 Regularized estimating equations

In this section, we extend the results to the more interesting regularization cases. Existing literature on regularized estimating equations (Fu, 2003; Johnson et al., 2008) typically
considers the following formulation:

U(β) + qλ(|β|) ⊙ sgn(β) = 0, (4)

where sgn(β) = (sgn(β1), . . ., sgn(βp))[⊤] and qλ(|β|) = (qλ(|β1|), . . ., qλ(|βp|))[⊤] with qλ(·)
being a continuous function. Here denotes the component-wise product. The tuning
⊙
parameter λ - 0 determines the amount of regularization. Johnson et al. (2008) mainly
considered the case where qλ(|βj|) = [dp]dt[λ][(][t][)] ��t=|βj| [≡] [p]λ[′] [(][|][β][j][|][)] [is] [the] [derivative] [of] [some] [penalty]

function pλ(·) evaluated at |βj| for j = 1, . . ., p. Some example penalties include (a) the lasso
penalty (Tibshirani, 1996), pλ(|t|) = λ|t| ; (b) the elastic net penalty (Zou and Hastie, 2005),
pλ(|t|) = λ1|t| + λ2|t|[2]; and (c) the SCAD penalty (Fan and Li, 2001), defined by

� �
p[′]λ[(][|][t][|][) =][ λ] I(|t| < λ) + [(][aλ][ −|][t][|][)][+],

(a 1)λ [I][(][|][t][| ≥] [λ][)]
−

for a > 2.
Note that formulation (4) only works for penalties with element-wise separability and
cannot be directly applied to many other commonly-used penalties, such as the group lasso
(Yuan and Lin, 2006) and the sparse group lasso (Simon et al., 2013). In this article, we
consider the regularized estimating equation in a slightly more general form:

0 U(β) + λ∂Ω(β), (5)
∈

where Ω( ) : R[p] R is a general convex penalty and ∂Ω(β) denotes the set of all subgradients

       - →
of Ω( ) at β. A subgradient of Ω( ) at β R[p] is defined as any vector g R[p] such that

    -     - ∈ ∈

Ω(β[′]) Ω(β) + g[⊤](β[′] β) for all β[′].
≥ −

Note that ∂Ω(β) is a closed and convex set. Several examples of formulation (5) follow.


-----

Ridge. If Ω( ) is a convex and differentiable function, then ∂Ω(β) = Ω(β), i.e., the

            - {∇ }
gradient of Ω(β) at β is its only subgradient. Therefore, for the ridge penalty Ω(β) = ∥β∥2[2][,]
the sub-differential set contains only the regular gradient ∂Ω(β)/∂β = 2β and thus (5) reduces
to the regular estimating equation U(β) + 2λβ = 0.

Lasso. If Ω(·) is the lasso penalty Ω(β) = ∥β∥1, then β must satisfy the equation

U(β) + λv = 0, (6)

where v ∈ ∂∥β∥1 is a subgradient of ∥β∥1, evaluated at β. The jth element of v is

�
sgn(βj), if βj = 0̸,

vj = (7)

∈ [−1, 1], if βj = 0,

for j = 1, . . ., p. The estimating equation (6) yield the following equivalent conditions
�
Uj(β) + λ sgn(βj) = 0, if βj = 0̸,
(8)
|Uj(β)| ≤ λ, if βj = 0.

Note that the first condition in (8) for βj ≠ 0 coincides with the original formulation (4)
by Johnson et al. (2008) with qλ(|βj|) = λ, but (4) did not explicitly handle the scenario
βj = 0. When U(β) = −X [⊤](y − Xβ) is the negative gradient of the least squares objective
L(β) = [1] [(8)] [corresponds] [to] [the] [KKT] [conditions] [of] [the] [lasso] [regularized] [least]

2[∥][y][ −] [Xβ][∥][2][,]
squares problem.

Group lasso. Suppose the p predictors are divided into several non-overlapping groups.
Let G = {g1, . . ., g|G|} be a partition of the index set {1, . . ., p}. Each group gj is a subset
of the index set G, with no overlaps with other groups, gj ∩ gk = ∅ for k ≠ j. Let |G| be
the number of groups and let mg be the size of group g. The union of all |G| groups covers
the entire index set such that ∪j[T]=1[g][j] [=] [G][.] [For] [the] [coefficient] [vector] [β] [=] [(][β][1][, . . ., β][p][)][⊤][,] [we]
let βg denote the sub-vector of β whose indices are within g. Yuan and Lin (2006) proposed
the group lasso regularization Ω(β) = [�]g∈G √mg∥βg∥2. For ease of notation, we omit the

weights [√]mg in the penalty term. The corresponding regularized estimating equation is

�� �

0 ∈ U(β) + λ∂ ∥βg∥2 . (9)

g∈G

Denote by [x]g the sub-vector of x for group g. The solution to (9) satisfies the following
equation, for each group g:

[U(β)]g + λug = 0,

where ug is the subgradient of ∥βg∥2 evaluated at βg with

�

ug = ∥ββgg∥2 [,] if βg ≠ 0, (10)

∈{x : ∥x∥2 ≤ 1}, if βg = 0.


The subgradient equation (9) yields the following equivalent conditions
�

[U(β)]g + λ ∥ββgg∥2 [=][ 0][,] if βg ≠ 0,

∥[U(β)]g∥2 ≤ λ, if βg = 0.


-----

Sparse group lasso. As an important extension of the group lasso, Simon et al. (2013)
proposed the sparse group lasso which allows both group-wise and within-group sparsity. The
penalty is a convex combination of the lasso and group-lasso penalties, Ω(β) = [�]g∈G[(1][ −]

α)∥βg∥2+α∥β∥1, where α ∈ [0, 1]. For each group g, the corresponding regularized estimating
equation is

[U(β)]g + λ(1 − α)ug + λαvg = 0, (11)

where ug is a subgradient as defined in (10) and vg is the sub-vector of a subgradient v as
defined in (7).

### 3 Fixed-point formulation

In this section, we provide a connection between regularized estimating equations and fixedpoint problems. Assume that the solution of (5) exists, which is denoted by β[ˆ]. Then we have
the following equivalent conditions for τ - 0:

0 U(β[ˆ]) + λ∂Ω(β[ˆ])
∈
0 β (β[ˆ] τ U(β[ˆ])) + τλ∂Ω(β[ˆ]) (12)
⇐⇒ ∈ [ˆ] − −

⇐⇒ 0 ∈ [1] 2��� ���

2 [∇][β][∥][β][ −] [(ˆ][β][ −] [τ] [U][(ˆ][β][))][∥][2] β= β[ˆ] [+][ τλ∂][β][Ω(][z][)] β= β[ˆ][,]


where the differentiation ∇β and subdifferential ∂β are with respect to β. If Ω(β) is a convex
penalty, the last line of (12) characterizes the necessary and sufficient condition for β[ˆ] to be
a minimizer of the penalized quadratic function:

1
ˆ
β = arg min 2 [+][ τλ][Ω(][β][)][.] (13)
β 2[∥][β][ −] [(ˆ][β][ −] [τ] [U][(ˆ][β][))][∥][2]

Let proxΩ : R[p] → R[p] be the proximal operator (Parikh and Boyd, 2014) of the convex penalty
function Ω,

1
proxΩ(v) = arg min 2 [+ Ω(][z][)][.] (14)
z 2 [∥][z][ −] [v][∥][2]

Since the regularized quadratic function on the right-hand side of (14) is strongly convex, it
has a unique minimizer for every v R[p]. Now we can rewrite (13) as a fixed-point problem:
∈

find βˆ ∈ R[p] such that β[ˆ] = f (β[ˆ]), with f (β) ≡ proxτλΩ(β − τ U(β)). (15)

Therefore β[ˆ] is a solution to (5) if and only if β[ˆ] is a solution to (15). Note that if λ = 0, the
operator proxτλΩ(v) reduces to v, thus (15) simplifies to (2).
Evaluating the proximal operator of a function requires solving a small strongly convex
optimization problem (14). In many cases, these problems often have closed form solutions
or can be solved very efficiently using specialized algorithms. We present several examples
below.

Lasso. When the penalty is lasso, the j-th element of the proximal operator is

[proxτλ∥·∥1(v)]j = sgn(vj)(|vj| − τλ)+ ≡ Sτλ(vj),

which is the soft-thresholding rule.


-----

Group lasso. The group lasso penalty has a closed form proximal operator (Parikh and
Boyd, 2014): for group g,


�
τλ

[proxτλΩ(v)]g = 1 −
∥zg∥2


�


vg,
+


where [x]g is the sub-vector corresponding to group g of x.

Sparse group lasso. The sparse group lasso also has a closed form proximal operator
(Simon et al., 2013): for group g,

1

[proxτλΩ(v)]g = arg min 2 [+][ τλ][ [(1][ −] [α][)][∥][z][g][∥][2] [+][ α][∥][z][g][∥][1][]]

��zg 2[∥][z][g][ −] [v][g][∥][2]� �
= 1 − [(1][ −] [α][)][τλ] Sατλ(vg),

∥Sατλ(vg)∥2 + g


where Sατλ(vg) ≡ (Sατλ([vg]1), . . ., Sατλ([vg]mg ))[⊤] with Sατλ(x) = sgn(x)(|x| − ατλ)+.

### 4 Variational inequality formulation

After establishing the equivalences between regularized estimating equations and fixed-point
problems, we also show a connection between regularized estimating equations and generalized variational inequalities. This is not surprising since equivalences between fixed-point
problems and variational inequalities are well known (see, e.g., Malitsky, 2019).
Following (5), a solution β[ˆ] should satisfy U(β[ˆ])/λ ∂Ω(β[ˆ]). This implies that U(β[ˆ])/λ
− ∈ −
is a subgradient of Ωat β[ˆ]. Thus, by the definition of a subgradient,

Ω(β) Ω(β[ˆ]) U(β[ˆ])[⊤](β β)/λ (16)
≥ − − [ˆ]

for any β. It follows that (16) can be rewritten as a variational inequality problem:

find βˆ R[p] such that U(β[ˆ])[⊤](β β) + λ(Ω(β) Ω(β[ˆ])) 0, for all β R[p]. (17)
∈ − [ˆ] − ≥ ∈

Note that if λ = 0, (17) reduces to (3). Unlike formulations (5) and (15), which require either
specification of the subgradient or evaluation of the proximal operator for Ω, formulation (17)
only needs us to specify U(β) and Ω(β).

### 5 Extensions to constrained forms

Alternative to the Lagrangian form (5), one may also consider the constrained form of the
regularized estimating equations

U(β) = 0, such that β, (18)
∈C

where is a convex set. For example, can be a normed ball β : Φ(β) r with the norm
C C { ≤ }
function Φ(·) and radius r - 0. One can set Φ(β) to be ∥β∥1 for the lasso constraint, and


-----

�

g∈G [∥][β][g][∥][2] [for] [the] [group] [lasso] [constraint,] [etc.] [Intriguingly,] [(18)] [can] [still] [be] [viewed] [as] [an]
instance of (5). Let I (β) : R[p] R be an indicator function
C →

�
0 if β
∈C

I (β) =
C if β / .

∞ ∈C

Assume the solution of (18) exists, then (18) is equivalent to (5) with Ω(β) = I (β) and
C
λ = 1. Let β[ˆ] be the solution of (18), the fixed-point formulation thus apply

0 U(β[ˆ]) + ∂I (β[ˆ])
∈ C
⇐⇒ βˆ = proxτIC (β[ˆ] − τ U(β[ˆ]))

1
ˆ
⇐⇒ β = arg min 2
β∈C 2 [∥][β][ −] [(ˆ][β][ −] [τ] [U][(ˆ][β][))][∥][2]

βˆ = P (β[ˆ] τ U(β[ˆ])), (19)
⇐⇒ C −

where the projection operator onto is defined as
C

1
PC(y) = arg minx∈C 2 [∥][x][ −] [y][∥]2[2][.]

From (19) we can see that the proximal operator associated with the constraint I (β[ˆ]) becomes
C
the projection on the convex set, which shows that (18) can be rewritten as the fixed-point
C
problem
find βˆ R[p] such that β[ˆ] = f (β[ˆ]), with f (β) P (β τ U(β)).
∈ ≡ C −

On the other hand, (18) can also be represented as the variational inequality problem

find βˆ R[p] such that U(β[ˆ])[⊤](β β) 0, for all β .
∈ − [ˆ] ≥ ∈C

To see this, let (β) be the normal cone of at β,
NC C

(β) = g R[p] : g[⊤](β[′] β) 0 for all β[′] .
NC { ∈ − ≤ ∈C}

For β, we know that ∂I (β) = (β), this gives
∈C C NC

0 U(β[ˆ]) + ∂I (β[ˆ]) = U(β[ˆ]) + (β[ˆ])
∈ C NC

U(β[ˆ]) (β[ˆ])
⇐⇒ − ∈NC

U(β[ˆ])[⊤](β[′] β) 0 for all β[′],
⇐⇒ − [ˆ] ≥ ∈C

as desired.

### 6 Computation

Formulations (15) and (17) reveal interesting connections between regularized estimating
equations, fixed-point problems and variational inequalities. To solve large-scale regularized
estimating equations, it might be worth pursuing computation from (15) and (17). While
fast computational algorithms are less developed for (4), there are many efficient solvers for
fixed-point problems and variational inequalities. In this regard, we apply some efficient and
scalable solvers to (15) and (17), and examine their performance against existing algorithms
for regularized estimating equations.


-----

#### 6.1 Existing approaches

To solve (4), many existing works (e.g., Johnson et al., 2008) adopted the local quadratic
approximation (LQA) approach proposed by Fan and Li (2001). Specifically, they considered
a local quadratic approximation to the penalty function

p[′]λ[(][|][β][˜][j][|][)]

pλ(|βj|) ≈ pλ(|β[˜]j|) + [1]2 |β[˜]j| (βj[2] [−] [β][˜]j[2][)]


around an iterate β[˜]j. This yields the following approximation to the subgradient of pλ(|βj|)
when β[˜]j = 0:̸

∂ λ[(][|][β][˜][j][|][)]
pλ(|βj|) = p[′]λ[(][|][β][j][|][) sgn(][β][j][)][ ≈] [p][′] βj. (20)
∂βj |β[˜]j|

By this local quadratic approximation, the Newton–Raphson algorithm was used to solve the
following equation
Qβ�(β) := U(β) + Λλ(β[�]) ⊙ β = 0, (21)

where Λλ(β[�]) = diag{p[′]λ[(][|][β][˜][1][|][)][/][|][β][˜][1][|][, . . ., p]λ[′] [(][|][β][˜][p][|][)][/][|][β][˜][p][|}][.] [Let] [β][(][k][)] [= (][β]1[(][k][)][, . . ., β]p[(][k][)][)][⊤] [be] [the] [k][-th]
iterate of β. The algorithm finds the next update β[(][k][+1)] using

�∂Qβ(k) (β(k)) �−1
β[(][k][+1)] = β[(][k][)] − Qβ(k)(β[(][k][)])
∂β[⊤]

�∂U(β(k)) �−1
= β[(][k][)] − + Λλ(β[(][k][)]) Qβ(k)(β[(][k][)]). (22)
∂β[⊤]


Note that once a component βj[(][k][)] becomes zero during the iteration, the term p[′]λ[(][|][β]j[(][k][)] j

[|][)][/][|][β][(][k][)][|]
in Λλ(β[(][k][)]) becomes illy defined. To continue the iteration, the algorithm would have to
stop updating those zero components and simply set their final estimates to zero, and then
works only with the nonzero components of β. This treatment, however, creates a potential
problem, that is, once a component becomes zero, it is permanently deleted and will never
again receive updates. To fix this, Hunter and Li (2005) replaced (p[′]λ[(][|][β][˜][j][|][)][/][|][β][˜][j][|][)][β][j] [in (20) with]
(p[′]λ[(][|][β][˜][j][|][)][/][(][|][β][˜][j][|][ +][ ǫ][))][β][j] [for] [some] [ǫ >][ 0.] [This] [leads] [to] [a] [modified] [Λ][λ][(][β][�][) = diag][{][p][′]λ[(][|][β][˜][1][|][)][/][(][|][β][˜][1][|][ +]
ǫ), . . ., p[′]λ[(][|][β][˜][p][|][)][/][(][|][β][˜][p][|][ +][ ǫ][)][}] [in] [(21)] [and] [(22).]
The algorithms by Fan and Li (2001) and Hunter and Li (2005) suffer from some significant
drawbacks: (a) they cannot easily handle more complex penalty functions, such as the group
and sparse group lassos; (b) the Newton–Raphson update in (22) involves the inversion of the
p × p matrix ∂U(β[(][k][)])/∂β[⊤] + Λλ(β[(][k][)]), so if p ≫ n, the computational cost of (22) becomes
(p[3]), which renders the algorithm extremely impractical for high-dimensional data, when,
O
e.g. p = 100, 000; (c) the update in (22) does not directly produce a sparse solution, so one
needs to manually truncate the β[ˆ]j’s to zero when |β[ˆ]j| < c for some threshold c, but there is
no theoretical guideline on how to choose the value of c, and in practice it is just set to an
arbitrarily small number; and (d) the convergence properties of the algorithm in (22) were
studied only for the maximum penalized likelihood (Hunter and Li, 2005), but have never
been established for the regularized estimating equations.


-----

#### 6.2 Computation for the fixed-point formulation

Suppose f : R[p] R[p] has Lipschitz constant L > 0 such that
→

∥f (β) − f (β[′])∥2 ≤ L∥β − β[′]∥2, for all β, β[′] ∈ R[p]. (23)

When L = 1, f is referred to as a nonexpansive mapping and its set of fixed points = β :
P {
f (β) = β is closed and convex (can be empty or contain many points; see Ryu and Boyd,
}
2016). Instead, if L < 1, f is called a contraction and has exactly one fixed point (Ryu and
Boyd, 2016, page 6).
A very straightforward algorithm for solving (15) is the fixed-point iteration (Picard, 1890;
Lindel¨of, 1894; Banach, 1922), also called the Picard iteration:

β[(][k][+1)] = f (β[(][k][)]), k = 0, 1, 2, . . ., (24)

with an initial value β[(0)]. One can show that if f is a contraction with Lipschitz constant
L < 1, the fixed-point iteration described in Algorithm 1 can converge to the unique fixedpoint β[ˆ] of f with a geometric rate (p15, Ryu and Boyd, 2016):

β[(][k][)] β L[k] β[(0)] β .
∥ − [ˆ]∥≤ ∥ − [ˆ]∥

However, if f is only nonexpansive, the fixed-point iteration (24) may not converge to
the set of fixed-points . Alternatively, we can use the Krasnosel’skii–Mann iteration (KM,
P
Mann, 1953; Krasnosel’skiı, 1955):

β[(][k][+1)] = (1 ρ)β[(][k][)] + ρf (β[(][k][)]), k = 0, 1, 2, . . ., (25)
−

with ρ (0, 1). Assume the set of fixed-points is nonempty. Then the KM iteration
∈ P
detailed in Algorithm 2 will yield updates β[(][k][)] β for some β[ˆ], that satisfy Fej´er
→ [ˆ] ∈P
monotonicity
inf β[(][k][)] β 0.
βˆ∈P ∥ − [ˆ]∥→

Moreover, the points yielded by the KM iteration satisfies the fixed-point condition (15)
arbitrarily closely,
∥f (β[(][k][)]) − β[(][k][)]∥2 → 0,

with rate O(1/k). Specifically, we have

β[(0)] β
∥ − [ˆ]∥
min 2 (26)
j=0,...,k [≤] (k + 1)ρ(1 ρ) [.]

[∥][f] [(][β][(][j][)][)][ −] [β][(][j][)][∥][2] −

Choosing ρ = 1/2 can maximize ρ(1 ρ), and therefore minimizes the righthand side of the
−
inequality (26). This suggests a possible choice ρ = 1/2, which gives the simple iteration

β[(][k][+1)] = (1/2)β[(][k][)] + (1/2)f (β[(][k][)]), k = 0, 1, 2, . . ..


-----

Algorithm 1: fixed-point iteration.

Input: Regularization parameter λ > 0, function U, τ  - 0

1 Initialize β[(0)];

2 for k = 1, 2, . . . do

3 β[(][k][+1)] = proxτλΩ(β[(][k][)] − τ U(β[(][k][)]));

4 end


Algorithm 2: Krasnosel’skii iteration.

Input: Regularization parameter λ > 0, function U, ρ (0, 1), τ  - 0
∈

1 Initialize β[(0)];


2 for k = 1, 2, . . . do

3 β[(][k][+1)] = (1 − ρ)β[(][k][)] + ρ proxτλΩ(β[(][k][)] − τ U(β[(][k][)]));

4 end

#### 6.3 Computation for variational inequality formulation

We can solve the variational inequality (17) using the Golden Ratio Algorithm (GRA) proposed by Malitsky (2019). At each iteration, the algorithm only requires the evaluation of
U and proxλΩ. Algorithm 3 provides the computational details of this method with a fixed
stepsize.

Algorithm 3: Golden ratio algorithm with a fixed step size.


Input: Lipschitz constant L, function U.

√
1 Initialize β[(1)] and β[¯][(0)], golden ratio φ =

2 for k = 1, 2, . . . do

3 Compute β[¯][(][k][)] = [(][φ][−][1)][β][(][k][)][+ ¯][β][(][k][−][1)] ;

φ

4 β[(][k][+1)] = proxtλΩ(β[¯][(][k][)] − tU(β[(][k][)]));

5 end


5+1, fixed step size t (0, [φ]

2 ∈ 2L [];]


Followed from Theorem 1 of Malitsky (2019), we know that if U in (17) is monotone, i.e.

U(β) U(β[′]), β β[′] 0, for all β, β[′] R[p],
⟨ − − ⟩≥ ∈

and is L-Lipschitz continuous, i.e.

∥U(β) − U(β[′])∥2 ≤ L∥β − β[′]∥2, for all β, β[′] ∈ R[p],

then with arbitrary initialization β[(1)], β[¯][(0)] ∈ R[p] and a fixed stepsize t ∈ (0, 2[φ]L [], the sequences]

(β[(][k][)]) and (β[¯][(][k][)]) generated by Algorithm 3 converge to the solution of (17) with rate O(1/k).
Algorithm 3 uses a fixed stepsize t ∈ (0, 2[φ]L [], which requires the knowledge of the Lipschitz]

constant L. If the value of L is not available, one can adopt an adaptive stepsize version of the
GRA algorithm for solving (17) (see details in Algorithm 4). This approach does not require
a line-search. The adaptive GRA computes the stepsizes in each iteration by approximating


-----

an inverse local Lipschitz constant of U, which has the same computational cost as the fixed
stepsize version. Malitsky (2019) showed that, even when U is only locally Lipschitz (that is,
on every bounded set S ⊂ R[p], for each β0 ∈ S, there is a constant L0 - 0 and a δ0 - 0 such
that ∥β − β0∥2 < δ0 implies ∥U(β) − U(β0)∥2 ≤ L0∥β − β0∥2), with arbitrary initialization
β[(1)] and β[¯][(0)] R[p], the updating sequences (β[(][k][)]) and (β[¯][(][k][)]) generated by Algorithm 4 can
∈
converge to a solution of (17) with rate O(1/k).

Algorithm 4: Adaptive golden ratio algorithm.

√
Input: golden ratio t >[¯] 0, φ = 5+12, ϕ ∈ (1, φ], ρ = ϕ[1] [+] ϕ1[2] [,] [function] [U][.]

1 Initialize β[(0)] and β[(1)] = β[¯][(0)], stepsize t0 = ∥U(∥ββ[(1)][(1)])−−βU[(0)](β∥[(0)])∥ [,] [θ][0] [= 1;]

2 for k = 1, 2, . . . do

3 Find the step size


�
tk = min ρtk−1, [ϕθ]4tk[k]−[−]1[1]


�

β[(][k][)] β[(][k][−][1)]
∥ − ∥[2]

.
U(β[(][k][)]) U(β[(][k][−][1)])
∥ − ∥[2] [,][ ¯][t]


4 Update

5 Update θk = tkt−k 1 [ϕ][.]

6 end



[β][¯][(][k][−][1)]
¯
β[(][k][)] = [(][ϕ][ −] [1)][β][(][k][)][ +],

ϕ

β[(][k][+1)] = proxtkλΩ(β[¯][(][k][)] − tkU(β[(][k][)])).


-----

### References

Banach, S. (1922). Sur les op´erations dans les ensembles abstraits et leur application aux
´equations int´egrales. Fund. math 3, 133–181. 6.2

Fan, J. and Li, R. (2001). Variable selection via nonconcave penalized likelihood and its
oracle properties. Journal of the American Statistical Association 96, 1348–1360. 2, 6.1,
6.1

Fu, W. J. (2003). Penalized estimating equations. Biometrics 59, 126–132. 2

Hunter, D. R. and Li, R. (2005). Variable selection using mm algorithms. Ann. Statist.
33, 1617–1642. 6.1

Johnson, B. A., Lin, D. Y. and Zeng, D. (2008). Penalized estimating functions and
variable selection in semiparametric regression models. Journal of the American Statistical
Association 103, 672–680. 2, 2, 2, 6.1

Krasnosel’skiı, M. A. (1955). Two remarks on the method of successive approximations.
Usp. Mat. Nauk 10, 123–127. 6.2

Lindel¨of, E. (1894). Sur l’application de la m´ethode des approximations successives aux
´equations diff´erentielles ordinaires du premier ordre. Comptes rendus hebdomadaires des
s´eances de l’Acad´emie des sciences 116, 454–457. 6.2

Malitsky, Y. (2019). Golden ratio algorithms for variational inequalities. Mathematical
Programming, 1–28. 4, 6.3, 5

Mann, W. R. (1953). Mean value methods in iteration. Proceedings of the American
Mathematical Society 4, 506–510. 6.2

Parikh, N. and Boyd, S. (2014). Proximal algorithms. Foundations and Trends in optimization 1, 127–239. 3, 3

Picard, E. (1890). Memoire sur la theorie des equations aux derivees partielles et la methode
des approximations successives. Journal de Math´ematiques pures et appliqu´ees 6, 145–210.
6.2

Ryu, E. K. and Boyd, S. (2016). A primer on monotone operator methods (survey).
Applied and Computational Mathematics 15, 3–43. 6.2, 6.2

Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). A sparse-group lasso.
Journal of computational and graphical statistics 22, 231–245. 2, 2, 3

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the
Royal Statistical Society, Series B (Methodological) 58, 267–288. 2

Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped
variables. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68,
49–67. 2, 2


-----

Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net.
Journal of the Royal Statistical Society: Series B (Statistical Methodology) 67, 301–320. 2


-----

