THE WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS: MATCHINGS,
MULTIPATHS, AND MAGNITUDE COHOMOLOGY

LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS


ABSTRACT. Building upon previous works of Proudfoot and Ramos, and using the categorical framework of Sam and Snowden, we extend the weak categorical minor theorem from undirected graphs to quivers. As case of study, we investigate the
consequences on the homology of multipath complexes; eg. on its torsion. Further, we prove a comparison result: we show
that, when restricted to directed graphs without oriented cycles, multipath complexes and matching complexes yield functors
which commute up to a blow-up operation on directed graphs. We use this fact to compute the homotopy type of matching
complexes for a certain class of bipartite graphs also known as half-graphs or ladders. We complement the work with a study
of the (representation) category of cones, and with analysing related consequences on magnitude cohomology of quivers.


1. INTRODUCTION

The graph minor theorem of Robertson and Seymour [RS04] states that the undirected graphs, partially ordered by
the graph minor relationship, form a well-quasi-ordering. If one restricts to graphs with bounded combinatorial genus,
this fact has a categorical enhancement which was recently explored by Miyata and the third author in [MR23]. It is
shown in [MR23] that the (weak) categorical minor theorem is related to Noetherian properties of the (representation)
category of undirected graphs with bounded genus. Remarkably, this relationship has non-trivial topological and
combinatorial consequences. With a view to such consequences, in this paper we extend the (weak) categorical version
of the graph minor theorem to quivers and directed graphs. Inspired by previous works, and in particular by [Bar15,
PR19, PR22], we borrow techniques from representation theory of categories. Sam and Snowden, in fact, developed
a fascinating and powerful machinery to prove that a certain representation category is Noetherian [SS17]. Their
approach makes use of combinatorial properties – called (quasi-) Gröbner properties – on the base category: if a
category C is quasi-Gröbner, then its category of representations RepRC, over a Noetherian ring R, is Noetherian.
The category Graph[op]≤g [of undirected graphs with] [bounded genus][ g][,] [and opposite minor morphisms, was] [shown]
to be quasi-Gröbner in [PR22]. As a consequence, all subrepresentations of finitely generated representations of
Graph[op]≤g [are also finitely generated.] [This fact has some striking applications to the study of certain simplicial com-]
plexes associated to graphs, such as those arising from monotone properties of graphs, cf. [PR22, MR23]. The study of
such simplicial complexes and, in particular, of matching complexes is a well-established [Wac03, SW04, Jon08a] and
yet vibrant area of research [[˘]IPVZ20, MJMV22, Sin22, Mat22, MMS23]. In particular, the behaviour of the torsion
appearing in the homology of matching complexes has been a subject of considerable attention – see [SW04, Jon10],
and references therein. Using the fact that the category Graph[op]≤g [is quasi-Gröbner, Miyata and the third author ap-]
proached the study of torsion of matching complexes from a different perspective. They proved, in [MR23], that if one
considers graphs of bounded genus, then the homology of their matching complexes has universally bounded torsion.
It is an open conjecture whether a similar statement holds true when considering the whole category of graphs, with no
restriction on the genus – see [MR23, Conjecture 3.3].
In this work, we go beyond the undirected case, and prove that also the category of quivers Quiver, and minor
morphisms, is combinatorially well-behaved;


Theorem 1.1 (Weak categorical quiver minor theorem). The category Quiver[op]≤g [is quasi-Gröbner.]

As a consequence, simplicial complexes associated to quivers of bounded genus, such as those arising from monotone properties of directed graphs, are finitely generated (under some additional mild assumptions). As case of study
we consider multipath complexes [CCDTS24], which also include cycle-free chessboard complexes [VŽ09]. The
interest in studying the homology of multipath complexes stems from their subtle relations with Hochschild homology [TW12] and symmetric homology of algebras [AF07, Aul10]. Furthermore, multipath complexes are related to,
and sometimes coincide with, matching complexes, see [CCC]. In the spirit of [MR23], we use Theorem 1.1 to investigate the global behaviour of the homology of multipath complexes. Let X : Quiver[op] → SimplComp be the
functor associating to each quiver its multipath complex. First, we prove that, for any fixed i, g ∈ N, the homology


1


-----

2 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS


p1

p2

p3


q1

q2

q3


FIGURE 1. The bipartite graph B3.

Hi(X(−); Z): Quiver[op]≤g [→] [Mod][R][ is a finitely generated][ Quiver]≤[op]g[-module (Proposition 4.6). Then, we show that]
the torsion in Hi(X(−); Z) is universally bounded for quivers of bounded genus (Proposition 4.7).
In a different direction, we analyse further the connection between matching and multipath complexes. To this
end, we introduce the notion of blow-up B(G) of a directed graph G, see Definition 4.15. We show, in Theorem 4.20,
that the multipath complex of a directed graph G without oriented cycles and the matching complex of the underlying
undirected graph of B(G) are isomorphic. The blow-up construction defines a functor B from the category of directed
graphs without oriented cycles (and injective morphisms of digraphs) Digrapho to the opposite of the category of
undirected graphs (and minor morphisms) Graph[op]. Then, Theorem 4.20 can be restated by saying that the diagram

B

Digrapho Graph[op]
(1) M

X


SimpCompl

is commutative up to isomorphism of simplicial complexes; see also Corollary 4.22. Here, we have denoted with M
the functor associating to each undirected graph its matching complex. We infer the following direct comparison result:

Theorem 1.2. Multipath complexes of directed graphs without oriented cycles are isomorphic to (joins of) matching
complexes of bipartite graphs.

Diagram (1) and Theorem 1.2 show that, when restricting to digraphs without oriented cycles, computations of
matching complexes can be obtained by equivalent computations of multipath complexes, in turn providing a way to
compute the homotopy type of new families of matching complexes. We give here a main example. Let Bn be the
undirected graph – sometimes called half-graph or ladder [NORS21] – on 2n vertices p1, . . ., pn and q1, . . ., qn, and
edges (pi, qj) for all i ≤ j – see Figure 1.

Theorem 1.3. The matching complex of Bn is either contractible or homotopy equivalent to a wedge of spheres.

We complement the paper with two more main directions. The first, regarding the theoretical description of the combinatorial properties of the category of quivers, aims to shed light on the analog of [MR23, Conjecture 3.3] for quivers.
Trying to enlarge the category for which the weak categorical minor theorem is true, we prove in Proposition 3.10 that
the category of cones of graphs with bounded genus is also quasi-Gröbner. This is beneficial to (partially) get around
the restriction of bounded genus. For instance, the cone category includes such families as the wheel graphs, which are
cones over cycles, and the Thagomizer graphs, which are cones over star trees. The second direction concerns some applications of the categorical machinery of Sam and Snowden to homology theories of digraphs which do not (directly)
arise from monotone properties of digraphs. Our main example is magnitude homology of graphs [HW17]. Magnitude
homology is the categorification of magnitude, as defined by Leinster – see, for example, the book [Lei21] – capturing
many interesting properties of graphs [Gom19, Asa21, Asa23, AHK23, TY23]. In [CC23], the first two authors proved
that magnitude cohomology of undirected graphs is finitely generated; in turn, getting new insight about the torsion in
magnitude homology – see also [KY21, SS21]. We extend in Section 5 the results of [CC23] to the category of quivers.
Using the work of Asao [Asa23], we also get in Corollary 5.11 some insight on the torsion of the path homology of
quivers, introduced in [GMVY18].
We conclude with a perspective on homology theories of digraphs and quivers, whose interest in recent years has
skyrocketed due to connections to persistence and topological data analysis. If [MR23, Conjecture 3.3] were true,
we could extend the results of this paper to all (directed) graphs, without restriction on the genus. In this work, we
focused on two main examples of cohomology theories of simplicial complexes deriving from quivers. However,
in view of a better understanding of the (weak) categorical minor theorem, it would be interesting to investigate also
further cohomology theories of graphs, quivers and categories. To name a few, some recent intriguing examples, related
to magnitude homology, are for example reachability homology [HR23, CR23], spectral homology [Iva23], or those
introduced in [IP23, Rof23].


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 3

Conventions and notation. All (directed) graphs and quivers are assumed to be finite, and are denoted in typewriter
font, e.g. Q, G. Unless otherwise specified, R will denote a Noetherian commutative ring with identity. All categories
are in bold font. We mainly use the following categories of graphs:

Graph undirected graphs and minor morphisms
Digraph digraphs and minor morphisms
Quiver quivers and minor morphisms
CGraph undirected graphs and contractions
CDigraph digraphs and contractions
CQuiver quivers and contractions
Digrapho directed graphs without directed cycles and injective morphisms of digraphs
Graphg undirected graphs of genus g and contractions
Digraphg digraphs of genus g and contractions
Quiverg quivers of genus g and contractions
CGraph≤g undirected graphs of genus at most g and contractions
CDigraph≤g digraphs of genus at most g and contractions
CQuiver≤g quivers of genus at most g and contractions
Graph≤g undirected graphs of genus at most g and minor morphisms
Digraph≤g digraphs of genus at most g and minor morphisms
Quiver≤g quivers of genus at most g and minor morphisms
Cone(Quiver≤g) cone over quivers of genus at most g and minor morphisms of the base quiver

While we listed them here for the readers’ convenience, the precise definition of these categories will be given throughout the paper.

2. BASIC NOTIONS

In this section, we recall some basic notions and set the notations needed throughout.

2.1. Quivers. Recall that a (finite) quiver Q consists of two finite sets V (Q) and E(Q), together with a pair of functions
s, t : E(Q) → V (Q). The elements of V (Q) are called vertices, while the elements of E(Q) are called edges. For
each e ∈ E(Q) the vertices s(e) and t(e) are the source and the target of e, respectively, and they are collectively
called the endpoints of e. A morphism of quivers f : Q → Q[′] is given by a pair of functions fV : V (Q) → V (Q[′]) and
fE : E(Q) → E(Q[′]), such that the two diagrams


E(Q) s V (Q)

fE fV and

s[′]

E(Q[′]) V (Q[′])


E(Q) t V (Q)


t[′]

E(Q[′]) V (Q[′])


fE


fV


commute. Note that morphisms of quivers can send an edge to a self-loop.
Special cases of quivers are digraphs. These are quivers such that each edge e is completely determined by the
(ordered) pair of its endpoints (s(e), t(e)). An undirected graph G is a quiver together with a map r : E(G) → E(G),
called reflection, which is an involution such that s ◦ r = t. An undirected edge in an undirected graph refers to a pair
of edges in the underlying quiver which are swapped by the reflection. Note that any morphism of quivers between two
undirected graphs commutes with the respective reflections.

Remark 2.1. We will often identify each edge e in a digraph with the ordered pair (s(e), t(e)). Similarly, if the quiver
underlying an undirected graph is a digraph, we will identify each undirected edge with the unordered collection of its
endpoints.

A subquiver H of a quiver Q is a quiver such that V (H) ⊆ V (Q), E(H) ⊆ E(Q), and both source and target functions
are given by restriction. If H is a subquiver of Q, we write H ≤ Q. If H is a subquiver of Q, and V (H) = V (Q), then
we say that H is spanning in Q. A subquiver of a digraph is automatically a digraph. However, a subquiver of an
undirected graph may not be an undirected graph. Subquivers of undirected graphs which are undirected graphs with
reflection defined by restriction are called subgraphs. To be coherent with the classical terminology, we will refer also
to subquivers of digraphs as subgraphs.
A quiver Q has a geometric realization |Q|. This is the geometric realization of the CW complex whose 0-cells are
the vertices of the quiver, 1-cells are the edges of the quiver, and the attaching maps are given by the source and target


-----

4 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

maps. In the case of undirected graphs we will take the quotient by the action of the reflection. We define the genus of a
quiver (resp. undirected graph) as the first Betti number of its geometric realisation. A (directed) tree T is an undirected
graph (resp. quiver) whose genus is 0 or, equivalently, whose geometric realization |T| is contractible.
Given a quiver Q and an edge e ∈ E(Q), we can define two quivers Q \ e and Q/e, called respectively the deletion
and the contraction of e. The former is the subquiver of Q obtained by removing e from the set of edges. The latter
is the quiver whose edges are E(Q) \ e, whose vertices are the quotient of V (Q) by the identification t(e) = s(e), and
whose source and target maps are defined as follows: s(e[′]) = [s(e[′])] and t(e[′]) = [t(e[′])], where the brackets denote
the equivalence class. Less formally, Q/e is obtained from (the geometric realization of) Q by contracting e to a point.
Contraction and deletion of undirected graphs are defined similarly; the only change is that we remove both e and r(e),
and then we define the reflection as the map induced by the original reflection.
Note that the operation of contracting edges does not change the homotopy type of the geometric realization, unless
the edge contracted is a self loop. We are not going to consider this latter case, and only allow contractions of edges with
distinct endpoints. Similarly, when dealing with deletions, we will only allow deletions whose geometric realization is
connected.
A minor of a quiver (resp. undirected graph) Q[′] is a quiver (resp. undirected graph) Q that is isomorphic to a quiver
(resp. undirected graph) obtained from Q[′] by iterative contractions and deletions. More formally, we have the following
definition of minor morphisms of quivers.

Definition 2.2. A minor morphism φ : G[′] → G of quivers is a map of sets

φ : V (G[′]) ⊔ E(G[′]) ⊔{⋆} → V (G) ⊔ E(G) ⊔{⋆},

such that:

   - φ(V (G[′])) = V (G) and φ(⋆) = ⋆;

   - if an edge e ∈ E(G[′]) has endpoints (s(e), t(e)) = (v, w), and φ(e) ̸= ⋆, then either φ(e) = φ(v) = φ(w) is a
vertex of G, or φ(e) is an edge of G with endpoints s(φ(e)) = φ(v) and t(φ(e)) = φ(w);

   - there is a bijection between φ[−][1](E(G)) and E(G);

   - for each vertex v ∈ G, the preimage φ[−][1](v) as a subquiver of G[′] is a directed tree.
If there is a minor morphism φ : G[′] → G we will say that G is a minor of G[′]. The definition of minor morphism of
undirected graphs is the same, but the words “edge”, “directed tree”, and “subquiver” are replaced by “undirected
edge”‘, “tree”, and “subgraph”, respectively.

The preimage of ⋆ under φ consists of deleted edges, whereas the edges that are mapped to vertices of G represent
the contracted ones. Furthermore, the last item in the definition implies that self loops cannot be contracted, but only
deleted.
A simple path, or directed path, in a quiver Q is subquiver whose edges can be ordered e1, ..., en in such a way that
(i) s(ei+1) = t(ei) for each i < n, (ii) no vertex is encountered twice, i.e. if s(ei) = s(ej) or t(ei) = t(ej) then i = j,
and (iii) s(e1) ̸= t(en). A subquiver of Q satisfying properties (i) and (ii), but not (iii), in the definition of simple path
is called oriented cycle. We call alternating any quiver Q such that the sets t(E(Q)), s(E(Q)) ⊂ V (Q) are disjoint.

Example 2.3. The alternating quiver An in Figure 2 has a simple path Im, with m ≤⌊ [n]2 [⌋] [edges, as a minor (but not as]

a subquiver). More generally, the simple path I1 is a minor of any directed tree.

We get the categories Graph and Quiver of undirected graphs and quivers, respectively, with minor morphisms.
The category Digraph is the full subcategory of Quiver spanned by digraphs. Analogously, we denote by CGraph
and CQuiver the categories of graphs and quivers, respectively, with only contractions as allowed morphisms. The
category CDigraph is also defined, as the full subcategory of CQuiver spanned by digraphs.

Remark 2.4. There is a pair of functors ι : Quiver → Graph and ρ : Graph → Quiver. The latter is just the
forgetful functor which forgets the reflection. Intuitively, ρ makes each undirected edge of a graph bidirectional. The
functor ι instead replaces each edge with an undirected edge. We will refer to ι(Q) as the underlying graph of Q.

v0 v1 v2 v3 v4 v5 . . . vn−1 vn

FIGURE 2. The alternating linear quiver An on n + 1 vertices. The edge between vn−1 and vn can
be oriented either way depending on the parity of n.


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 5

2.2. Finitely generated C-modules. Let C be a (essentially) small category, and A be a ring. A representation of C,
or C-module, over A is a functor M : C → A-Mod with values in the category of (left) A-modules. Denote by
RepA(C) the category of C-modules over A where morphisms are given by natural transformations.
A submodule of a C-module M is a C-module N such that N (c) is a A-submodule of M(c), for each object c ∈ C.
If S is a subset of [�]c∈C [M][(][c][)][, the][ span of][ S][, denoted by][ span(][S][)][, is the minimal][ C][-submodule of][ M][ containing][ S][.]

Definition 2.5. A C-module M is finitely generated if there is a finite set S ⊆ [�]c∈C [M][(][c][)][, such that][ span(][S][) =][ M][.]

We can also characterise finitely generated modules in terms of simpler modules. For each object c of C, define a
principal projective C-module Pc, as follows

Pc(c[′]) := A⟨HomC(c, c[′])⟩, for each c[′] ∈ C,

i.e. the free (left) A-module spanned by HomC(c, c[′]), and Pc is then defined on morphisms by (post)composition.
Given a morphism γ : c → c[′], we denote by eγ the corresponding element in Pc(c[′]).

Lemma 2.6 ([CEFN14, Proposition 2.3]). A C-module M is finitely generated if and only if there exists a surjection

n
�

Pci →M
i=1


for some objects c1, . . ., cn of C.

For a finitely generated C-module M, we refer to the objects c1, . . ., cn of C in the lemma as generators of M.

Definition 2.7. A C-module M is Noetherian if all its submodules are finitely generated. The category RepA(C) is
(locally) Noetherian if all finitely generated C-modules over A are Noetherian.

Observe that, in discussing properties related to finite generation, it is often possible to restrict to principal projective
modules. Indeed, by [SS17, Proposition 3.1.1], the category RepA(C) is Noetherian if and only if every principal
projective module Pc is Noetherian.

Example 2.8. Let FI be the category of finite sets and injective maps. Then, by [CEFN14, Theorem A], the category RepR(FI) is Noetherian, for any (commutative Noetherian) ring R.

Noetherian properties of various other categories have been extensively investigated, in particular thanks to the
techniques developed by Sam and Snowden in [SS17]. One of the main results in the latter paper is that, for a given
category C (under some combinatorial assumptions), and a (possibly non-commutative) left Noetherian ring R, the
associated category of representations is Noetherian as well. Before recalling the combinatorial conditions to be required on the category C, and stating the main result in this section, we need a definition. Let F : C → D be a functor
between essentially small categories.

Definition 2.9. We say that F satisfies property (F) if for every object d ∈ D there exist finitely many objects
c1, . . ., cn of C, and morphisms δi : d →F (ci), such that: for any object c in C, and morphism δ : d →F (c), there
exists a morphism γi : ci → c satisfying δ = F (γi) ◦ δi.

Observe that a functor that is surjective on both objects and morphisms satisfies property (F). Property (F) allows us
to transfer finitely generated properties through functors. In fact, the following holds:

Proposition 2.10 ([SS17, Proposition 3.2.3]). If a functor F : C → D satisfies property (F), and M : D → R-Mod
is finitely generated, then F [∗]M := M ◦F : C → R-Mod is finitely generated.

We need to introduce some further notation and terminology; see also [SS17, Section 4.1] for a more extensive
overview. Let S : C → Set be a functor with values in the category of sets. We associate a poset |S| to S as follows.
First, take S[�] to be the union [�]c∈C [S][(][c][)][.] [Then, for an element][ f] [in][ S][(][c][)][ and an element][ g] [in][ S][(][c][′][)][, set] [f] [≤] [g] [if and]
only if there exists a morphism h : c → c[′] in C such that S(h)(f ) =h∗(f ) = g. Consider the equivalence relation ∼
defined by f ∼ g if and only if f ≤ g and g ≤ f . Then, the poset |S| is defined as the quotient of the set S[�] with respect
to ∼, equipped with the partial order induced from ≤.

Definition 2.11. An ordering on S : C → Set is a choice of a well-order on S(c) for each c in C, such that for every
morphism c → c[′] the induced map S(c) → S(c[′]) is strictly order-preserving; in such a case, we say that S is orderable.

We say that a poset P is Noetherian if, for every infinite sequence x1, x2, . . . in P, there exist indices i < j such
that xi ≤ xj (cf. [SS17, Proposition 2.1]). We can now recall the fundamental definition of Gröbner category.


-----

6 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

Definition 2.12. An essentially small category C is Gröbner if, for all objects c of C, the functor Sc := HomC(c, −)
is orderable, and the associated poset |Sc| is Noetherian. An essentially small category C is quasi-Gröbner if there
exists a Gröbner category C[�] and an essentially surjective functor C[�] → C satisfying property (F).

Roughly speaking, a category is said to be Gröbner if its slice categories are restrictive enough, so that representations of the category allow for a theory of “Gröbner bases”. The category FI from Example 2.8 is not a Gröbner
category, as its automorphism groups are symmetric groups, hence non-trivial. However, FI is quasi-Gröbner:

Remark 2.13. Let OI be the category of linearly ordered finite sets and ordered inclusions. This is a “rigidified” version
of the category FI; in particular, it has trivial automorphism groups. It was shown in [SS17, Theorem 7.1.2] that OI
is indeed a Gröbner category, and that the functor OI → FI is an essentially surjective functor satisfying property (F).
As a consequence, FI is quasi-Gröbner.

The next result follows readily from the definitions:

Proposition 2.14. Let F : C → D be a functor satisfying property (F). If C is a quasi-Gröbner category, and F is
essentially surjective, then D is a quasi-Gröbner category.

Proof. Composition of essentially surjective functors is essentially surjective. Moreover, the composition of functors
satisfying property (F) satisfies property (F) by [SS17, Proposition 3.2.6]. 
Assume now that R is a left Noetherian ring. The following is one of the main results connecting combinatorial
properties of a category with the Noetherianity of the category of representations.

Theorem 2.15 ([SS17, Theorem 1.1.3]). If C is a quasi-Gröbner category, then the category RepA(C) is Noetherian.

In particular, if R is a left Noetherian ring and C is quasi-Gröbner, all submodules and quotients of finitely generated
functors M : C → R-Mod are also finitely generated.

3. GRÖBNER PROPERTIES OF Quiver AND Digraph

The primary goal of this section is to prove that certain subcategories of Quiver[op] and Digraph[op] are Noetherian.

Definition 3.1. Let g be a non-negative integer. The category Quiverg (resp. Quiver≤g) is the full subcategory
of Quiver whose objects are quivers of genus g (resp. at most g). The categories Digraphg and Digraph≤g are
defined in the same fashion.

Observe that in both Quiverg and Digraphg, the only minor morphisms that appear are automorphisms and contractions, as deletions never preserve genus. On the other hand, among the morphisms in Quiver≤g and Digraph≤g
there are both deletions and contractions.
For technical reasons, we need to define “rigidified” versions of the above categories. These will play a prominent
role in the proof of this section’s main theorem. First, recall that a rooted spanning tree for a quiver G is a pair (T, vr),
where T is a spanning subquiver of G which is a directed tree, and vr is a fixed vertex of T called root. Note that, just as
any connected subquiver, T is obtained from G by subsequent deletions. We set P Quiverg to be the category whose
objects are undirected graphs of genus g, enhanced with the following extra information:

(1) a choice of rooted spanning tree;
(2) a choice of planar embedding for the aforementioned spanning tree;
(3) a choice of orientation for the edges outside of this spanning tree;
(4) at each non-root vertex, a label indicating whether that vertex is the source or target of the (unique) edge of the
spanning tree leading from the vertex to the root.

The morphisms of this category will be edge contractions (and automorphisms) of the underlying graphs, which
preserve all of the above structure. We similarly define P Digraphg.

Remark 3.2. The reader might find it strange that we defined the category P Quiverg by starting with undirected
graphs, and then adding extra information that effectively orients them. The reason for this relates with our ultimate
proof that this category is Gröbner. The idea is that we start with planar trees, and then add the data of the edge
orientations and the extra edges in such a way that they can be encoded into the vertices of the tree. This way, we will
be able to use a version of Kruskal’s Tree theorem, which allows for the vertices of the tree to be labelled.


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 7

Remark 3.3. We observe that our category P Quiverg is a minor modification of the category PGg in [PR22]. Indeed,
the main difference between the two categories lies in the labels on the vertices of the spanning tree. The important
point is that this extra data is finite in nature; this means that our vertices are being given labels from a finite set (source
and target). This will allow us to apply [PR22, Corollary 3.7].

To prove that representations of Quiver[op]≤g [and][ Digraph]≤[op]g [have our desired Noetherian property, we will use the]
Gröbner methods of Sam and Snowden [SS17]. To summarize, for each fixed g, we will proceed as follows:

   - We show that the category P Quiver[op]g [is Gröbner;]

   - we show that the forgetful functor P Quiver[op]g [→] [Quiver]g[op] [has Property (F) and is essentially surjective;]

   - finally, we argue that the natural inclusion [�]q≤g [Quiver]q[op] ֒→ Quiver[op]≤g [has Property (F) and is essentially]

surjective.
Once we have completed each of the above steps, [SS17, Theorem 1.1.3] (cf. Theorem 2.15) will immediately imply
our desired Noetherian Property.
Note that in our outline we have not said anything about the category Digraph≤g! To justify why, we have the
following standard proposition.

Proposition 3.4. The category P Digraph[op]≤g [is Gröbner provided that][ P] [Quiver]≤[op]g [is.]

Proof. By construction the category P Digraph≤g is a full subcategory of P Quiver≤g. The Gröbner property is
inherited by full subcategories per [SS17, Proposition 4.4.2]. 
The above proposition justifies why the first step in our outline need not be repeated for the two categories. We will
see that the latter two steps can be proven identically in the digraph case, and therefore it is unnecessary to repeat them.
In particular, the above represents the crux of why we will only be working with the quiver categories from this point
forward in this section.

Proposition 3.5. For any g ≥ 0 the category P Quiver[op]g [is Gröbner.]

Proof. In [PR22, Theorem 3.10] it is proven that the category PGg,S[op] [is Gröbner where][ S] [is any finite (or even well-]
quasi-ordered) set of permissible vertex labels. Consider the case wherein S = {s, t}. P Quiverg is a full subcategory
of PGg,S[op] [in this case.] [It follows that][ PG]g,S[op] [is Gröbner by the same argument given in the prior proposition.] 
Following our outline, our next goal is to now show that the forgetful functor P Quiverg → Quiverg has Property (F) and is essentially surjective. It is clear that every quiver in Quiverg is in the image of the forgetful functor.
Property (F) follows immediately from [PR22, Lemma 3.11]. To conclude our proof it therefore remains to show that
the embedding [�]q≤g [Quiver]q[op] ֒→ Quiver[op]≤g [is essentially surjective with property (F).]

Lemma 3.6. Fix g ≥ 0. The embedding [�]q≤g [Quiver]q[op] ֒→ Quiver[op]≤g [is essentially surjective with property (F).]

Proof. Let Q be a quiver of genus at most g, and let {φi : Qi → Q} be an exhaustive list of all possible deletions that
terminate at Q. Note that by a deletion we mean a minor morphism for which no edge is contracted and at least one
edge is deleted. Further note that because our genus g is fixed, and because deletions always strictly decrease genus
when applied, this collection is finite.
To prove that our embedding has (F), we must argue that for any quiver Q[′] of genus at most g and any minor
morphism ψ : Q[′] → Q we can find an i and an edge contraction η : Q[′] → Qi, such that ψ = φi ◦ η. Indeed, this just
amounts to the statement that any minor morphism can be written as a composition of a deletion followed by an edge
contraction. 
Let going forward, we will write CQuiver≤g to denote the category [�]q≤g [Quiver]q[op][, of quivers of genus at most][ g]
and contractions. To summarize the work of this section, we have proven the following.

Theorem 3.7. The categories Quiver[op]≤g[,][ CQuiver][op]≤g[, and][ Digraph][op]≤g [are quasi-Gröbner.]

Consider the edge module
Eg : (Quiver≤g)[op] → ModR
which associates to a quiver Q the free R-module Eg(Q) generated by the set E(Q) of edges of Q. For any minor
morphism, there is a well-defined inclusion of edges in the opposite direction of the contraction. This in turn induces
maps between the aforementioned free modules.


-----

8 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

... m2

m1 m3
. . . . . .

... m[′]2

FIGURE 3. The graph S(m1, m2, m[′]2[, m][3][)][.] [The edge in blue goes from][ v][1] [to][ v][2][.]

Proposition 3.8. Let d be a natural number. Then, the module Eg[⊗][d] is a finitely generated Quiver[op]≤g[-module.]

Proof. The proof follows the same arguments of [MR23, Lemma 4.2]; by replacing the sets of edges with the sets of
directed edges. 
Although defined in a similar way, note that the vertex module

V : Quiver[op]≤g [→] [Mod][R]

is not finitely generated. However, we have the following:

Proposition 3.9. Let R be a ring with identity. Then, Hom(V [⊕][k], R) is finitely generated, as Quiver[op]≤g[-module, for]
each k.

Proof. The proof follows the same arguments of [CC23, Theorem 3.11], with only two small changes: the edges
of R(m) have to be oriented, and the quiver S(m1, m2, m[′]2[, m][3][)] [in] [Figure] [3] [replaces the] [graph][ S][(][m][1][, m][2][, m][3][)] [used]
in [CC23]. 
3.1. Categories of directed cones. For a given quiver G, a cone over G, denoted Cone(G), is a quiver obtained by
adding a new vertex to G, while also adding (directed) edges to connect it with every vertex of G. Note that for our
purposes, the added cone point is connected to each vertex of G by exactly one edge. In other words, we do not allow
for multi-edges connected to the cone point.
Let G, G[′] be two quivers. Then a base contraction from Cone(G) to Cone(G[′]) is a minor morphism Cone(G) →
Cone(G[′]) involving the contraction of edges of G, thought of as living inside of Cone(G), followed by the deletion of
the newly created multiedges that are connected to the cone point of Cone(G[′]). Similarly, a base deletion is a minor
morphism Cone(G) → Cone(G[′]) involving the deletion of edges of G, thought of as a subquiver of Cone(G). More
generally, a base minor morphism is any minor morphism Cone(G) → Cone(G[′]) that can be written as a composition
of base contractions and base deletions. We will write Cone(Quiver≤g) for the category of directed cones whose base
graph has genus at most g, and whose morphisms are base minor morphisms.
Our first result will tell us that representations of the cone category satisfy a Noetherian Property.

Proposition 3.10. The category Cone(Quiver≤g)[op] is quasi-Gröbner. In particular, subrepresentations of finitely
generated representations of Cone(Quiver≤g)[op] are also finitely generated.

Proof. We begin with the category P Quiver[op]≤g[, and modify its objects to add an extra label to each vertex.] [Specifi-]
cally, this label will be either ’u’ (for up) or ’d’ (for down). We also modify our morphisms to insist that the newly added
label must be preserved. This new category will be Gröbner following an essentially identical proof to Proposition 3.5.
Consider now the functor from our new category to Cone(Quiver≤g)[op], which sends every object to the cone
over its underlying graph, while orienting the cone edges in the direction dictated by the vertex labels. A morphism
in our category naturally defines a morphism in Cone(Quiver≤g)[op] by performing the analogous base deletions and
contractions. In the case of a contraction, the cone edges which are deleted by the corresponding base contraction are
precisely those whose vertices were originally further from the selected spanning tree’s root.
Our functor is essentially surjective by constuction, and it is immediate that every base deletion and contraction
appears in the image of some map. In particular, our functor has property (F), as desired. 
One benefit of the cone category is that it allows us to (partially) get around the restriction of bounded genus.
For instance, Cone(Quiver≤g)[op] includes such families as the Wheel graphs, which are cones over cycles, and the
Thagomizer graphs, which are cones over star trees.


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 9

Remark 3.11. The idea to expand categories of bounded genus by considering cones was first done by Proudfoot
and the third author in [PR19]. In that work, only undirected trees are considered. This work will therefore expand
previous applications both by allowing for higher genera bases, and also allowing for the edges - cone or otherwise to be directed.

For our applications in the sections that follow, we will need to state an analogous result to Proposition 3.8. Once
again the proof is similar to that of [MR23].

Proposition 3.12. The Cone(Quiver≤g)[op]-module,

G �→ Z[|][E][(][G][)][|]

is finitely generated. Moreover, the same is true of its tensor powers.

4. APPLICATIONS TO MULTIPATH COMPLEXES

In this section, we apply the abstract technology of Section 3 to simplicial complexes arising from monotone properties of directed graphs [Jon08b]. More specifically, we will deal with multipath complexes [CCDTS24].

4.1. Multipath complexes and their torsion. We are interested in the monotone property of digraphs on disjoint sets
of simple paths; following [TW12, CCDT], we call them multipaths:

Definition 4.1. A multipath of a quiver G is a spanning subquiver such that each connected component is either a vertex
or a simple path. The length of a multipath is the number of its edges.

The path poset of a quiver G is the poset (P (G), ≤), where P (G) is the set of multipaths in G (including the multipath
with no edges), ordered by the relation of being a subgraph. To the path poset P (G) we associate a simplicial complex:

Definition 4.2 ([CCDT23, Definition 6.2]). For a quiver G, the multipath complex X(G) is the simplicial complex
whose face poset (augmented to include the empty simplex ∅) is the path poset P (G).

(v2, v3)

v0 v1 v2 v3

(v0, v1) (v1, v2)

FIGURE 4. The coherently oriented linear graph I3 (top left), the multipath complex X(I3) (top
right), and the path poset P (I3) (bottom).

Equivalently, X(G) is the simplicial complex on the edges of G whose k-simplices are given by the multipaths in G
of length k − 1. When G is the complete digraph, the associated simplicial complex is also called cycle free chessboard
complex, denoted Ωn in [VŽ09]. Since being a multipath is a monotone property of digraphs, that is it is closed under
subgraphs – cf. [BW99] or [Jon08b], it follows that X(G) is a well-defined simplicial complex.

Lemma 4.3. The association G �→ X(G) yields a functor

X : Quiver[op] → SimplComp

to the category of simplicial complexes and simplicial maps.


-----

10 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

Proof. Let φ : G[′] → G be a minor morphism of quivers. Then, the morphism φ[op] is injective on the set of edges: for e
an edge of G, φ[op](e) is again an edge, of G[′]. Furthermore, φ[op] preserves the adjacency relation, hence it sends simple
paths to simple paths, and, in turn, multipaths to multipaths. Lengths and containment relations are also preserved,
from which we get a simplicial map X(G) → X(G[′]). As the construction preserves identities and compositions, the
statement follows. 
Denote by ∗ the join operation of simplicial complexes – cf. [Koz08, Definition 2.16]. Then, for quivers G and H,
we have isomorphisms

(2) X(G ⊔ H) [∼]= X(G) ∗ X(H),

where ⊔ denotes the disjoint union; see also [CCDTS24, Remark 2.6]. We observe that the disjoint union on Quiver
and the join on SimplComp yields monoidal structures on the respective categories. Equaition (2) implies that the
functor X is symmetric monoidal, up to isomorphism of simplicial complexes.

Example 4.4. Consider the coherently oriented linear digraph In – see Figure 4 for an illustration of the quiver I3.
The path poset (P (In), ≤) is isomorphic to the Boolean poset B(n). Thence, the associated multipath complex is an
(n − 1)-dimensional simplex. Likewise, consider the coherently oriented polygonal digraph Pn with n edges, obtained
from In by identifying the vertices v0 and vn. Then, the path poset (P (Pn), ≤) is isomorphic to the Boolean poset
B(n) minus its maximum, and the corresponding multipath complex is a (n − 2)-dimensional sphere. Further, if we
consider the minor morphism φ : In → In−1 given by the contraction of the edge (vn−1, vn), then X(φ[op]) is the
inclusion of the (n − 2)-dimensional simplex into the boundary of the (n − 1)-dimensional simplex as the subcomplex
of the latter spanned by all vertices, but one.

As often happens to simplicial complexes associated to monotone properties of graphs, it is easy to find multipath
complexes which are homotopy equivalent to wedges of spheres, and, in particular, torsion free. Examples of such
complexes are those associated to directed trees. In fact, as we shall see below, the multipath complex of a directed tree
is homotopy equivalent to the join of matching complexes of undirected trees, cf. Proposition 4.11. On the other hand,
matching complexes of trees are contractible or homotopy equivalent to wedges of spheres [MT08, Theorem 4.13]. It
follows that the multipath complex of a directed tree is either contractible or homotopy equivalent to a wedge of spheres.
Further, the complexity of wedges of spheres which can be realised, up to homotopy, as multipath complexes of directed
trees can be arbitrarily high; for instance, for each n, multipath complexes of directed trees homotopy equivalent to the
wedge of an arbitrary number of n-dimensional spheres are given in the proof of [CCDT23, Proposition 6.12].
The fact that the multipath complex of an alternating digraph coincides with the matching complex of the underlying
undirected graph [CCC] can be used to find multipath complexes whose homology is not torsion free:

Example 4.5. Let Kn,m be the complete bipartite (undirected) graph on m, n vertices. When equipped with an alternating orientation, the multipath complex X(Kn,m) coincides with the matching complex of Kn,m and its homology
groups can have torsion. The minimal values of m and n for which the homology groups of X(Kn,m) have torsion are
m = n = 5, and in such case there is 3-torsion [SW04].

The torsion in the homology of matching complexes of undirected graphs with genus less or equal than a fixed
number was shown to be bounded in [MR23, Theorem 1.2]. Therefore, when restricting to complete bipartite graphs
with alternating orientation, the torsion of multipath complexes is also bounded. We want to extend this result to
categories of quivers with bounded genus. In the spirit of the techniques presented in Section 3, we first need to show
that the homology of multipath complexes yields a finitely generated module.

Proposition 4.6. For each i ∈ N, the homology functor

Hi(X(−); Z): Quiver[op]≤g [→] [Mod][R]

is a finitely generated Quiver[op]≤g[-module.]

Proof. The proof follows the same ideas as [MR23, Theorem 4.3]. First, we observe that taking chain complexes
and homology in a fixed degree is functorial with respect to taking minors. Using Lemma 4.3, we infer that the
functor Hi(X(−); Z) is a Quiver[op]≤g[-module.] [Furthermore,][ H][i][(][X][(][−][);][ Z][)][ is a subquotient of the tensor product][ E]g[⊗][i]
of the edge module; this module is finitely generated over Quiver[op]≤g [by Proposition 3.8.] [The] [statement] [follows by]
Theorem 2.15. 
As a consequence of Proposition 4.6, we have that torsion of multipath complexes is bounded:


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 11

Proposition 4.7. For every pair of integers i, g ≥ 0, there exists m = m(g, i) ∈ Z which annihilates the torsion
subgroup of Hi(X(G); Z), for each quiver G of genus at most g.

Proof. The proof follows the arguments of [MR23, Theorem 3.20], which we include for the sake of completeness.
Denote by Ti(G) the torsion submodule of the homology of multipath complex of G in (homological) degree i. The
functor Ti is a Quiver[op]≤g[-submodule] [of] [H][i][(][X][(][−][);][ Z][)][.] [Then,] [in] [virtue] [of] [Proposition] [4.6] [and] [by] [Theorem] [2.15,] [it]
follows that Ti is a finitely generated Quiver[op]≤g[-module. By Lemma 2.6, there is a finite number of quivers][ G][1][, . . .,][ G][k][,]
such that there is a surjective group homomorphism

k
� Ti(Gj) ։ Ti(G),

j=1

for each quiver G of genus at most g. Thus, the least common multiple of the exponents of Ti(G1), . . ., Ti(Gk) gives
the desired m(g, i). 
We wish to point out here that the statements in Proposition 4.6 and Proposition 4.7 can be extended to monotone
properties P of directed graphs, or quivers, which satisfy the following additional property: if φ : G → G[′] is a minor
morphism and σ[′] is a simplex in XP (G[′]), then the subgraph of G induced by the image φ[∗](E(σ[′])) is also in P . Here
we have denoted by XP (G) the simplicial complex associated to G via the monotone proeperty P . In the case of
the multipaths, note that the corresponding monotone property satisfies this condition. Another interesting monotone
property of digraphs is given by (the complex of) directed forests. In such case, as by [Eng09], the complex of directed
forests of graphs without oriented cycles is shellable; hence, it has no torsion. This observation leads to the following
question, whose answer is not known to the authors:

Question 4.8. Does the complex of directed forests of graphs with oriented cycles contain torsion?

The terminology of this last part of the section is taken from [CCDTS24], and we will limit our discussion to
digraphs (rather than all quivers).
Let G be a digraph, and let G[′] ≤ G be a subgraph. The complement CG(G[′]) of G[′] in G is the subgraph of G spanned
by the edges in E(G) \ E(G[′]). The boundary ∂GG[′] of G[′] in G, or simply ∂G[′] when G is clear from the context, is defined
as ∂GG[′] = V (G[′]) ∩ V (CG(G[′])).

Definition 4.9. Let G be a connected digraph with at least one edge and without self-loops. A vertex v ∈ V (G) is called
stable if it is not, at the same time, a source and a target (i.e. if it does not belong to the intersection s(E(G))∩t(E(G))),
and is unstable otherwise. A dynamical region in G is a connected subgraph R ≤ G, with at least one edge, such that:

(a) all vertices in the boundary of R are unstable in G, but stable in both R and CG(R);
(b) no edge of R belongs to any oriented cycle in G which is not contained in R.

A dynamical region is called stable if all its non-boundaryvertices are stable. It is called unstable if all its non-boundary
vertices are unstable, and it has at least one non-boundary vertex. A dynamical region which has no proper subgraph
which is itself a dynamical region is called a dynamical module.

Lemma 4.10. Let M be a dynamical module of a connected directed tree. Then, M is stable.

Proof. Let T be a directed tree, and M ≤ T a dynamical module. Assume, by contradiction, that M is not stable. Then,
there is at least an unstable vertex v not in the boundary of M. Since v is an unstable vertex in M there are some edges,
say e1, . . ., eh ∈ E(M), such that s(ei) = v, and some edges, say f1, . . ., fk ∈ E(M), such that t(fj) = v, for k, h ≥ 1.
Since T is a directed tree, so is M. In particular, deleting the edges e1, ..., eh disconnects M. Denote by R the
connected component containing the edges f1, ..., fk. We argue that R is a dynamical region of M, and this yields the
desired contradiction. First, R being a subgraph of M ≤ T implies that item (b) in the definition of dynamical region is
automatically satisfied. Observe that v is the unique vertex in the boundary of R in M. In fact, given w ̸= v in V (R) all
edges of M incident in w are also edges of R. Otherwise, one of the ei’s would be incident to w and we would have a
closed loop in (the geometric realisation of) T, e.g. by taking an (unoriented) path from v to w in R and then adding ei.
Since v is unstable in M and is, by construction, stable both in R and in CM(R), the statement follows. 
Stable dynamical regions are alternating digraphs. Therefore, the multipath complex of a stable dynamical region R
in a directed graph G is the matching complex of the underlying undirected graph of R – cf. [CCDTS24, Lemma 4.8].
As a consequence of this fact, we get the following:


-----

12 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

Proposition 4.11. Let T be a connected directed tree. Then, X(T) is homotopy equivalent to a join of matching
complexes of (undirected) trees.

Proof. By [CCDTS24, Theorem 1], there is a unique (up to re-ordering) decomposition of G into dynamical modules
M1, ..., Mk, and we have an isomorphism

X(G) [∼]= X(M1) ∗· · · ∗ X(Mk) .

By Lemma 4.10, each Mi is stable, hence each X(Mi) is the matching complex of the underlying undirected tree of Mi.
The statement follows. 
A straightforward corollary follows, strengthening some of the results contained in [CCDTS24].

Corollary 4.12. The multipath complex of a forest is either contractible or homotopy equivalent to a wedge of spheres.

Hence, a direct consequence of Proposition 4.11 is that the homology of multipath complexes of directed trees does
not contain any torsion. In order to get interesting torsion, one needs to consider non-acyclic quivers. However, when
the quivers contain directed cycles, computations of the homology groups are more involved – and often computationally exponential. Therefore, it would be interesting to know more about the global growth of the ranks of the homology
groups. By applying similar arguments as in [PR22, Proposition 4.3], the ranks have a polynomial growth, and they
could be packed into a Hilbert series. Let us recall the definition of Hilbert series. Denote by C a(n essentially) small
category. A norm on C is a function ν from the objects of C modulo isomorphism to N.

Definition 4.13. Given a finitely generated C-module (over R) F and a norm ν on C, the Hilbert series (with respect
to F and ν) is defined as
�
SF,ν(t) = rankR(F (x))t[ν][(][x][)] .

x∈C

If C is a category of graphs we will take ν to be the number of edges, and omit it from the notation.

We know from [Ram21] that the Hilbert series of matching complexes of undirected trees is algebraic. By Proposition 4.11, multipath complexes of directed trees decompose, up to isomorphism of simplicial complexes, as join
products of matching complexes of trees. As a consequence, we get that their homology over a field can be expressed
in terms of the Künneth formula. This might suggest that, in complete analogy to what happens with matching complexes, also the Hilbert series of multipath complexes is algebraic.

Question 4.14. Denote by Tk the subcategory of CDigraph spanned by trees with at most k unstable vertices. Is the
Hilbert series with respect to Hi(X−; Z): Tk → R-Mod algebraic for each choice of i, k ∈ N?

4.2. Multipath complexes and matching complexes. In the previous section we used the observation, from [CCC],
that the multipath complexes of trees are strictly related to matching complexes. The aim of this section is to extend
the relation between multipath complexes and matching complexes to a wider class of quivers, and, as a consequence,
to generalise Proposition 4.11. We will make use of a blow-up operation. To avoid further technical assumptions, we
work also in this section with digraphs rather than with quivers.
Recall that the indegree (resp. outdegree) of a vertex v is the number of edges with target (resp. source) v.

Definition 4.15. Let G be a digraph, and let v ∈ V (G) with both indegree and outdegree different from 0. The blow-up
of G at v is the digraph B(G, v) obtained from G as follows: the vertices of B(G, v) are the same as G except for v,
which is replaced by vin and vout. Given v[′], v[′′] ∈ V (B(G, v)) the edges from v[′] to v[′′] are in bijection with
(a) the edges between the corresponding vertices in G, if {v[′], v[′′]} ∩{vin, vout} = ∅;
(b) the edges from v[′] to v, if v[′] ̸= vout and v[′′] = vin;
(c) the edges from v to v[′′], if v[′] = vout and v[′′] ̸= vin;
(d) the empty set, in the remaining cases.
If the indegree or the outdegree of v are zero, then we set B(G, v) := G. The blow-up of G is the digraph B(G) obtained
from G by iteratively blowing-up all vertices in G one after the other; see Figure 6 for an illustrative example.

Note that the blow-up of digraphs does not depend on the order we blow-up the vertices. Furthermore, blowing-up
a vertex v yield a pair of vertices vin and vout. The blow-up of either vin or vout yields just the identity. Therefore, the
blow-up is a trivial operation on the class of alternating digraphs.

Remark 4.16. Blow-ups of digraphs are alternating digraphs, and, in particular, bipartite.


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 13

Note that blow-ups of digraphs might yield disconnected digraphs. In fact, it is not difficult to check that the
connected components of the blow-up B(G) of a digraph G without oriented cycles are in one-to-one correspondence
with the dynamical modules of G.

Example 4.17. The blow-up of In is the disconnected digraph [�]i[n]=1 [I][1][.] [While the blow-up of][ A][n] [is][ A][n][.]

Locally, blowing-up a vertex corresponds to blowing-up a dandelion subdigraph at its central vertex:

Example 4.18. Consider the dandelion Dn,m on (n + m + 1) vertices, and (m + n) edges, defined as follows:
(1) V (Dn,m) = {v0, w1, ...., wn, x1, ..., xm};
(2) E(Dn,m) = {(wi, v0), (v0, xj) | i = 1, ..., n; j = 1, ..., m}.


w1

w2

w3


x1

x2


FIGURE 5. The graph D3,2.

In other words we have a single (m + n)-valent vertex v0, all remaining vertices are univalent, there are n edges with
target v0, and there are m edges with source v0 – cf. Figure 5. Then, the blow-up of Dn,m is the blow-up B(Dn,m, v0)
of Dn,m at the central vertex v0, which is the disjoint union of a source with m edges, and a sink with n edges. Note
that maximal multipaths of Dn,m are of type {(wi, v0), (v0, xj)}, whereas maximal multipaths of B(Dn,m, v0) are of
type {{(wi, v0)}, {(v0, xj)}}.


1out


1in

2in

3


1


0


2


1in

0


2in


0

1out

2out


FIGURE 6. Transitive tournament T3 (in red on the left), its blow-up (in blue at the center), and the
associated bipartite graph B3 (in orange on the right).

Proposition 4.19. Let G be a digraph. Suppose that v ∈ V (G) does not belong to any oriented cycle in G. Then, the
multipath complex X(G) is canonically isomorphic, as a simplicial complex, to X(B(G, v)).

Proof. By definition, there is a natural bijection b : E(G) → E(B(G, v)) given by


b((v1, v2)) =



(v1, vin) if v2 = v,

(vout, v2) if v1 = v,

(v1, v2) otherwise


for each (v1, v2) ∈ E(G). Given a multipath M in G, we define b(M) as the spanning subgraph of B(G, v) spanned by
the edges in b(E(M)) ⊆ E(B(G, v)). To prove the statement is sufficient to show that b gives a well-defined bijection
between multipaths in G and multipaths in B(G, v).
The image of a multipath in G is a multipath in B(G, v); if at most one edge in the multipath has v as endpoint, then
the statement is clear. If two edges of the multipath have v as endpoint, then the connected component of the multipath
containing these two edges splits into the disjoint union of two paths (one containing vin and the other containing vout).
We claim that every multipath M[′] in B(G, v) is the image of a multipath in G. To this end we show that there is at
most one edge in each connected component of M[′] that has one endpoint in {vin, vout}. Assume, by contradiction, that
there are two edges of the multipath, say e1 and e2, incident in vout and vin, respectively. If e1 and e2 belong to the


-----

14 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

same component γ of M[′] we get a contradiction. By construction, there is no edge e in B(G, v) such that t(e) = vout
or s(e) = vin. Thus, e1 is the first edge in γ and e2 the last edge in γ. Therefore, b[−][1](γ) is an oriented cycle in G
which has v as a vertex. This is absurd since v does not belong to any oriented cycle in G. It follows that b[−][1](M[′]) is a
well-defined multipath, and the claim follows.
Since b is a bijection between the edges of the two digraphs, the induced map on multipaths is injective. This
concludes the proof. 
We are now ready to state the main technical result of the section. Recall by Remark 2.4 that the functor ι associates
to a quiver the underlying undirected graph.

Theorem 4.20. Let G be a digraph without oriented cycles. Then, the multipath complex X(G) is isomorphic, as
simplicial complex, to the matching complex of ι(B(G)).

Proof. Since no vertex of G is contained in any oriented cycle, by Proposition 4.19, X(G) is canonically isomorphic
to X(B(G)). The digraph B(G), and, in particular, each connected component of B(G), is alternating – cf. Remark 4.16.
Hence, by [CCC, Theorem 4.1], for each connected component H of B(G), we have an isomorphism of simplicial
complexes M (ι(H)) [∼]= X(H). Let H1, . . ., Hn be the connected components of B(G). Then, by Equation (2), we have

X(G) ≡ X(H1) ∗· · · ∗ X(Hn) [∼]= M (ι(H1)) ∗· · · ∗ M (ι(Hn)),

where M denotes the matching complex. As also the matching complex of a disjoint union is, up to isomorphism, the
join product of the matching complexes, this fact implies the statement. 
Let Digrapho be the category of directed graph without oriented cycles, and regular morphisms of directed graphs;
that means, injective maps φ : G1 → G2 of digraphs such that (v, w) ∈ E(G1) =⇒ φ(v) ̸= φ(w). Note, that this is
the same as the opposite category of digraphs without cycles, and deletions.

Remark 4.21. The blow-up construction lifts to a functor

B := B ◦ ι : Digrapho → Graph[op] .

In fact, it is easy to see that the blow-up extends to regular morphims of digraphs, and preserves compositions and
identities. Note that we do not allow contractions here because contractions may generate new oriented cycles.

Corollary 4.22. The following diagram

B

Digrapho Graph[op]

M
X

SimpCompl

is commutative up to isomorphism of simplicial complexes.

Proof. The statement follows in view of Theorem 4.20 and Remark 4.21. 
We can use the identification with the matching complex given in Theorem 4.20 to get new computations of the
homotopy type of matching complexes. In fact, let Bn be the undirected graph on 2n vertices p1, . . ., pn and q1, . . ., qn,
and edges (pi, qj) for all i ≤ j. Sometimes, these are called half-graphs or ladders [NORS21].

Theorem 4.23. The matching complex of Bn is either contractible or homotopy equivalent to a wedge of spheres.

Proof. Let Tn be the transitive tournament on n + 1 vertices (with only edges of type (i, j) for i < j). Then, we
can identify its blow-up with the bipartite graph Bn – see also Figure 6. In light of Theorem 4.20, the multipath
complex of Tn is isomorphic to the matching complex of ι(Bn). The statement now follows directly from [CCDTS24,
Theorem 5.1]. 
Remark 4.24. In view of Theorem 4.23 and [CCDTS24, Lemma 5.7], we get that the matching complexes of bipartite graphs obtained from Bn by deletion of a subset of vertices in {p1, . . ., pn} are either contractible or homotopy
equivalent to wedges of spheres.

Let G be a digraph without oriented cycles. Assume that, for each embedded cycle (possibly of length 2), the induced
orientation on it is not alternating. In such case, we say that G is a digraph without oriented and alternating cycles.
In particular, G does not have pairs of edges of the form (v, w) and (w, v). With this terminology, we directly get the
following:


-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 15

Theorem 4.25. Let G be a digraph without any oriented or alternating cycle. Then, the multipath complex of G is either
contractible or a wedge of spheres.

Proof. By assumption, the induced orientation on each embedded cycle is not alternating. This means that, for each
embedded cycle H, we can find a vertex vH with both indegree and outdegree non zero. Blowing-up G at vH breaks
the cycle H. Therefore, the blow-up of G is a directed forest. The statement follows by Theorem 4.20 and [MT08,
Theorem 4.13]. 
The class of digraphs without oriented and alternating graphs is not big. For example, cones of cycles or of trees
with at least an unstable vertex contain oriented or alternating cycles. Therefore, we can not apply Theorem 4.25 to
such digraphs, and cone graphs may contain torsion. However, in complete analogy with Proposition 4.7, we get that
such torsion is also bounded:

Proposition 4.26. For every pair of integers k, g ≥ 0, there exists m = m(g, k) ∈ Z which annihilates the torsion
subgroup of Hi(X(Cone(Q)); Z), for each quiver Q of genus at most g.

Proof. Use Proposition 3.12 and then proceed as for [MR23, Theorem 1.2]. 
5. APPLICATIONS TO MAGNITUDE HOMOLOGY

We start by recalling the definition of magnitude homology, following [HW17, Asa23].
A directed graph G can be seen as a metric space with the path metric d, where d(v, w) is the minimum length across
all directed paths between v and w. If v and w are not connected by a directed path, we set d(v, w) := ∞. Given
a non-negative integer k and a commutative ring R, let Λk(G; R) := R⟨(v0, . . ., vk) | vi ∈ V (G)⟩ be the R-module
freely generated by (k + 1)-tuples of vertices of G. We admit the empty tuple, and set Λk(G; R) := 0 for k ≤−2. For
a k-tuple (v0, . . ., vk) of vertices of G, with vi ̸= vi+1 and d(vi, vi+1) < ∞ for each i, the length of (v0, . . ., vk) in G
is the number


ℓ(v0, . . ., vk) :=


k−1
�

d(vi, vi+1) .
i=0


Furthermore, we define a differential on Λk(G; R) by setting


δ(v0, . . ., vk) :=


k−1
�

(−1)[i]δi(v0, . . ., vk),
i=1


where δi(v0, . . ., vk) = (v0, . . ., vi−1, vi+1, . . ., vk) if ℓ(v0, . . ., vk) = l = ℓ(v0, . . ., vi−1, vi+1, . . ., vk), and it is set
to 0 otherwise.
Consider the submodule Ik(G; R) := R⟨(v0, . . ., vk) | vi = vi+1 for some i⟩ of Λk(G; R), where I0(G) is set
to 0. The family of such modules can be equipped with the differential δ, yielding a chain complex. As we have
the inclusions Ik(G; R) ⊆ Λk(G; R) for all k, we can form the quotient chain complex with modules Rk(G; R) :=
Λk(G; R)/Ik(G; R). The magnitude chain complex MCk,l(G; R) is defined as the submodule of Rk(G; R) given by all
those tuples of length l; this is compatible with the chain complex structure – cf. [Asa23, Lemma 2.14]. Hence, the
pair (MC∗,l(G; R), δ) is a chain complex, and its homology called magnitude homology (of G) with coefficients in R.

Remark 5.1. A contraction φ : G → H of directed graphs induces a chain map

φ# : MC∗,∗(G; R) → MC∗,∗(H; R)

which, to the tuple (v0, . . ., vk) of G, associates the tuple (φ(v0), . . ., φ(vk)) if ℓ(φ(v0), . . ., φ(vk)) = ℓ(v0, . . ., vk),
and it is set to be 0 otherwise. The map φ# is a chain map, as it commutes with the differential δ, and it induces a map
in magnitude homology.

For the following result, we refer the reader to [HW17, Proposition 3.3] or [Asa23, Lemma 3.6].

Proposition 5.2. Magnitude homology is a functor

MH∗,∗ : CDigraph → BiGrModR

from the contraction category of digraphs to the category of bigraded R-modules.


-----

16 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

The dual version of magnitude homology, called magnitude cohomology

MC[k]l [(][G][;][ R][) := Hom(MC][l,k][(][G][;][ R][);][ R][)][,]

was introduced in [Hep22]. It also yields a functor MH[∗]∗ [:] [CDigraph][op] [→] [BiGrMod][R][.]
We have recalled here the definition of magnitude homology of directed graphs. Almost verbatim we could have
defined magnitude homology of undirected graphs. In fact, the two notions are related:

Remark 5.3. Let G be an (undirected) graph. Then, in the notation of Remark 2.4, we have that the magnitude
(co)homology of the digraph ρ(G) coincides with the magnitude (co)homology (of undirected graphs) of G – see also

[Asa23, Remark 2.16].

More generally, the definitions of magnitude homology and magnitude cohomology can also be extended to quivers,
yielding functors on the whole category CQuiver of quivers and contractions. Let R be a commutative Noetherian
ring, with identity. Then, we have the following extension of [CC23, Theorem 3.11]:

Theorem 5.4. The CQuiver[op]≤g[-module][ MH]l[k][(][−][;][ R][):] [CQuiver][op]≤g [→] [Mod][R] [is finitely generated.]

Proof. By Theorem 3.7, the (opposite) category CQuiver[op]≤g [of quivers of bounded genus, and contractions, is quasi-]
Gröbner. Hence, subquotients of finitely generated CQuiver[op]≤g[-modules are finitely generated.] [By Theorem 3.9, the]
CQuiver[op]≤g[-module] [Hom(][V] [⊕][k][, R][)] [is] [finitely] [generated.] [The] [statement] [now] [follows] [using] [the] [same] [arguments of]

[CC23, Proposition 3.10], adapted to the case of quivers. 
Using Theorem 5.4, we can extend the results of [CC23] to the more general setting of quivers. In particular, a
straightforward adaptation of the arguments in [CC23] yields:

Corollary 5.5. For every pair of integers k, g ≥ 0, there exists m = m(g, k) ∈ Z which annihilates the torsion
subgroup of MH[k]∗[(][G][;][ Z][)][, for each quiver][ G][ of genus at most][ g][.]

Corollary 5.6. Let K be a field, and g ≥ 0. Then, there exists a polynomial f ∈ Z[t] of degree at most g + 1, such
that, for all quivers G of genus at most g, we have

dimK MH[k]∗[(][G][;][ K][)][ ≤] [f] [(#][E][(][G][))][,]

where #E(G) is the number of edges of G.

Corollary 5.5 says that says that the order of torsion classes in integral magnitude (co)homology of quivers of genus
at most g, in a fixed (co)homological degree k, is bounded. By Remark 5.3 and using the results of [KY21] and, in
particular, [SS21, Theorem 3.13], we know that magnitude homology and cohomology of quivers can contain torsion.
However, we do not know if such torsion always comes from undirected graphs via the functor ρ, or not.

Question 5.7. Is it possible to extend the constructions of [KY21] and [SS21] so to get torsion of directed graphs which
does not come from the essential image of ρ : Graph → Quiver of Remark 2.4?

In view of the algebraicity results of Section 4, we ask the following:

Question 5.8. Is the Hilbert series of the magnitude homology of (directed) graphs algebraic?

Question 5.9. The categories Graph and Quiver are related by the functors ι and ρ. What is the relation between
generators of magnitude homology of graphs and magnitude homology of quivers?

Magnitude homology has strong connections with another homological invariant of digraphs, the so-called path
homology – cf. [GLMY20]. More precisely, path homology is the diagonal component of a certain spectral sequence
involving magnitude homology [Asa23]. Despite torsion in magnitude homology of digraphs is yet unclear, when we
turn to path homology we have more information; to be more precise, we have the following:

Remark 5.10. Path homology of digraphs can contain arbitrary torsion: if X is a simplicial complex, and GX the Hasse
digraph associated to X, then the path homology of GX is isomorphic to the (simplicial) homology of X – see, e.g.,

[GMY14], or [GMY16] for the cohomological version.

The following corollary tells us that also in the case of path homology of digraphs, torsion has to be bounded:

Corollary 5.11. For each g, k positive integers, there exists a d = d(g, k) ∈ Z such that, for each digraph G of genus g,
the torsion part of the path cohomology PH[k](G, Z) has exponent at most d.

Proof. Path (co)homology groups appear as groups in the second page of a spectral sequence whose 0-th page features
magnitude chain groups. Turning the pages of the spectral sequence corresponds to taking subsequent subquotients of
the 0-th page. Therefore, the statement follows thanks to Theorem 5.4. 

-----

WEAK CATEGORICAL QUIVER MINOR THEOREM AND ITS APPLICATIONS 17

REFERENCES

[AF07] S. Ault and Z. Fiedorowicz. Symmetric homology of algebras, 2007. ArXiv:0708.1575.

[AHK23] Y. Asao, Y. Hiraoka, and S. Kanazawa. Girth, magnitude homology and phase transition of diagonality. Proceedings of the Royal
Society of Edinburgh Section A: Mathematics, page 1–27, 2023.

[Asa21] Y. Asao. Magnitude homology of geodesic metric spaces with an upper curvature bound. Algebr. Geom. Topol., 21(2):647–664, 2021.

[Asa23] Y. Asao. Magnitude homology and path homology. Bull. Lond. Math. Soc., 55(1):375–398, 2023.

[Aul10] S. Ault. Symmetric homology of algebras. Algebr. Geom. Topol., 10(4):2343–2408, 2010.

[Bar15] D. Barter. Noetherianity and rooted trees, 2015.

[BW99] A. Björner and V. Welker. Complexes of directed graphs. SIAM J. Discrete Math., 12:413–424, 10 1999.

[CC23] L. Caputi and C. Collari. On finite generation in magnitude (co)homology, and its torsion, 2023. ArXiv:2302.06525.

[CCC] L. Caputi, D. Celoria, and C. Collari. Monotone cohomologies and oriented matchings. Homology, Homotopy & Applications. In press.
ArXiv:2203.03476.

[CCDT] L. Caputi, C. Collari, and S. Di Trani. Multipath cohomology of directed graphs. Algebraic & Geometric Topology. In press.
ArXiv:2108.02690.

[CCDT23] L. Caputi, C. Collari, and S. Di Trani. Combinatorial and topological aspects of path posets, and multipath cohomology. J. Algebr.
Comb., 57(2):617–658, 2023.

[CCDTS24] L Caputi, C Collari, S Di Trani, and J P. Smith. On the homotopy type of multipath complexes. Mathematika, 70(1):e12235, 2024.

[CEFN14] T. Church, J. S. Ellenberg, B. Farb, and R. Nagpal. FI-modules over Noetherian rings. Geometry & Topology, 18(5):2951 – 2984,
2014.

[CR23] L. Caputi and H. Riihimäki. On reachability categories and commuting algebras of quivers, 2023.

[Eng09] A. Engström. Complexes of directed trees and independence complexes. Discrete Math., 309(10):3299–3309, 2009.

[GLMY20] A. A. Grigor’yan, Yong Lin, Y. V. Muranov, and S.-T. Yau. Path complexes and their homologies. Journal of Mathematical Sciences,
248:564–599, 2020.

[GMVY18] A. A. Grigor’yan, Y. V. Muranov, V. Vershinin, and S.-T. Yau. Path homology theory of multigraphs and quivers. Forum Mathematicum,
30(5):1319–1337, 2018.

[GMY14] A. Grigor’yan, Yu. V. Muranov, and Shing-Tung Yau. Graphs associated with simplicial complexes. Homology, Homotopy and Applications, 16(1):295 – 311, 2014.

[GMY16] A. Grigor’yan, Y. Muranov, and S.-T. Yau. On a cohomology of digraphs and Hochschild cohomology. Journal of Homotopy and
Related Structures, 11:209–230, 2016.

[Gom19] K. Gomi. Magnitude homology of geodesic space, 2019.

[Hep22] R. Hepworth. Magnitude cohomology. Math. Z., 301(4):3617–3640, 2022.

[HR23] R. Hepworth and E. Roff. The reachability homology of a directed graph, 2023.

[HW17] R. Hepworth and S. Willerton. Categorifying the magnitude of a graph. Homology Homotopy Appl., 19(2):31–60, 2017.

[IP23] S. O Ivanov and F. Pavutnitskiy. Simplicial approach to path homology of quivers, subsets of groups and submodules of algebras.
Journal of the LMS, 2023.

[[˘]IPVZ20] D. [˘]Ioich, G. Panina, S. T. Vrechitsa, and R. T. Zhivalevich. Generalized chessboard complexes and discrete Morse theory. Chebyshevski˘ı
Sb., 21(2):207–227, 2020.

[Iva23] S. O. Ivanov. Nested homotopy models of finite metric spaces and their spectral homology, 2023.

[Jon08a] J. Jonsson. Exact sequences for the homology of the matching complex. Journal of Combinatorial Theory, Series A, 115(8):1504–1526,
2008.

[Jon08b] J. Jonsson. Simplicial complexes of graphs, volume 1928 of Lecture Notes in Mathematics. Springer-Verlag, Berlin, 2008.

[Jon10] J. Jonsson. More torsion in the homology of the matching complex. Experiment. Math., 19(3):363–383, 2010.

[Koz08] D. N. Kozlov. Combinatorial algebraic topology, volume 21 of Algorithms and computation in mathematics. Springer, Berlin

[KY21] R. Kaneta and M. Yoshinaga. Magnitude homology of metric spaces and order complexes. Bull. Lond. Math. Soc., 53(3):893–905,
2021.

[Lei21] T. Leinster. Entropy and diversity. The axiomatic approach. Cambridge: Cambridge University Press, 2021.

[Mat22] T. Matsushita. Matching complexes of polygonal line tilings. Hokkaido Mathematical Journal, 51(3):339 – 359, 2022.

[MJMV22] M. J. Milutinovi´c, H. Jenne, A. McDonough, and J. Vega. Matching complexes of trees and applications of the matching tree algorithm.
Ann. Comb., 26(4):1041–1075, 2022.

[MMS23] A. Mondal, S. Mukherjee, and K. Saha. Discrete morse theoretic computations in the matching complex of k7, 2023.

[MR23] D. Miyata and E. Ramos. The graph minor theorem in topological combinatorics. Advances in Mathematics, 430:109203, 2023.

[MT08] M. Marietti and D. Testa. A uniform approach to complexes arising from forests. Electron. J. Combin., 15(1):Research Paper 101, 18,
2008.

[NORS21] J. Nešetˇril, P. Ossona de Mendez, R. Rabinovich, and S. Siebertz. Classes of graphs with low complexity: The case of classes with
bounded linear rankwidth. European Journal of Combinatorics, 91:103223, 2021. Colorings and structural graph theory in context (a
tribute to Xuding Zhu).

[PR19] N. Proudfoot and E. Ramos. Functorial invariants of trees and their cones. Selecta Mathematica, 25:1–28, 2019.

[PR22] N. Proudfoot and E. Ramos. The contraction category of graphs. Representation Theory of the American Mathematical Society,
26(23):673–697, 2022.

[Ram21] E. Ramos. Hilbert series in the category of trees with contractions. Math. Z., 298(3-4):1831–1852, 2021.

[Rof23] E. Roff. Iterated magnitude homology, 2023.

[RS04] N. Robertson and P. D. Seymour. Graph minors. XX: Wagner’s conjecture. J. Comb. Theory, Ser. B, 92(2):325–357, 2004.

[Sin22] A. Singh. Higher matching complexes of complete graphs and complete bipartite graphs. Discrete Mathematics, 345(4):112761, 2022.


-----

18 LUIGI CAPUTI, CARLO COLLARI, AND ERIC RAMOS

[SS17] S. Sam and A. Snowden. Gröbner methods for representations of combinatorial categories. Journal of the American Mathematical
Society, 30(1):159–203, 2017.

[SS21] R. Sazdanovic and V. Summers. Torsion in the magnitude homology of graphs. J. Homotopy Relat. Struct., 16(2):275–296, 2021.

[SW04] J. Shareshian and M. L. Wachs. Torsion in the matching complex and chessboard complex. Advances in Mathematics, 212:525–570,
2004.

[TW12] P. Turner and E. Wagner. The homology of digraphs as a generalisation of Hochschild homology. Journal of Algebra and Its Applications, 11(02):1250031, 2012.

[TY23] Y. Tajima and M. Yoshinaga. Causal order complex and magnitude homotopy type of metric spaces, 2023.

[VŽ09] S. T. Vre´cica and R. T. Živaljevi´c. Cycle-free chessboard complexes and symmetric homology of algebras. European J. Combin.,
30(2):542–554, 2009.

[Wac03] M. L. Wachs. Topology of matching, chessboard, and general bounded degree graph complexes. volume 49, pages 345–385. 2003.
Dedicated to the memory of Gian-Carlo Rota.


-----

