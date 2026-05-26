## On the universality class of the special adsorption point of two-dimensional lattice polymers

Nathann T. Rodrigues,[1, 2,][ ∗] Tiago J. Oliveira,[2,][ †] and Thomas Prellberg[3,][ ‡]


_1Instituto_ _de_ _F´ısica,_ _Universidade_ _Federal_ _Fluminense,_
_Avenida_ _Litorˆanea_ _s/n,_ _24210-346_ _Niter´oi,_ _Rio_ _de_ _Janeiro,_ _Brazil_
_2Departamento_ _de_ _F´ısica,_ _Universidade_ _Federal_ _de_ _Vi¸cosa,_ _36570-900,_ _Vi¸cosa,_ _Minas_ _Gerais,_ _Brazil_
_3School_ _of_ _Mathematical_ _Sciences,_ _Queen_ _Mary_ _University_ _of_ _London,_ _London_ _E1_ _4NS,_ _United_ _Kingdom_
(Dated: May 18, 2023)

In recent work [PRE **100,** 022121 (2019)] evidence was found that the surface adsorption transition of interacting self-avoiding trails (ISATs) placed on the square lattice displays a non-universal
behavior at the special adsorption point (SAP) where the collapsing polymers adsorb. In fact, different surface exponents _φ[(][s][)]_ and 1/δ[(][s][)] were found at the SAP depending on whether the surface
orientation is horizontal (HS) or diagonal (DS). Here, we revisit these systems and study other ones,
through extensive Monte Carlo simulations, considering much longer trails than previous works. Importantly, we demonstrate that the different exponents observed in the reference above are due to
the presence of a previously unseen surface-attached-globule (SAG) phase in the DS system, which
changes the multicritical nature of the SAP and is absent in the HS case. By considering a modified
horizontal surface (mHS) where the trails are forbidden of having two consecutive steps along it,
resembling the DS situation, a stable SAG phase is found in the phase diagram, and both DS and
mHS systems present similar 1/δ[(][s][)] exponents at the SAP, being 1/δ[(][s][)] _≈_ 0.44, whilst 1/δ[(][s][)] _≈_ 0.34
in the HS case. Intriguingly, while _φ[(][s][)]_ _≈_ 1/δ[(][s][)] is found for the DS and HS scenarios, as expected,
in the mHS case _φ[(][s][)]_ is about 10% smaller than 1/δ[(][s][)]. These results strongly indicate that at least
two universality classes exist for the SAPs of adsorbing ISATs on the square lattice.


PACS numbers:


**I.** **INTRODUCTION**

The study of dilute polymers in solution has been a
long enterprise with great deal of efforts in both theoretical and experimental fields [1–4]. Besides the vast number of polymer applications [5], this has been motivated
also by the interesting fundamental physical properties
of these systems. Of interest here is the polymer conformation, which is highly affected by the solvent conditions
and temperature (T ) of the solution [1]. For instance, in
a good solvent (and/or at high _T_ ) flexible polymers are
usually found in a swollen coil phase, and by decreasing
the solvent quality (and/or _T_ ) the polymer may eventually collapse into a dense globule at the so-called _θ-_
point [1]. An even richer phase behavior can be observed
when the polymer is close to an attracting surface, where
it may adsorb depending on _T_, the solvent quality, and
surface properties [6, 7].

In this context, lattice models have been widely used
to investigate the various thermodynamic phases, phases
transitions and critical properties of dilute polymers [8].
In these coarse-grained models, details such as the polymer chemical composition and chemical bonds are not explicitly taken into account, while complex effects such as
excluded volume and hydrophobicity are represented in


a simple manner through in-lattice interactions [1–3, 8].
For instance, the canonical lattice model for collapsing
polymers is the interacting self-avoiding walk (ISAW) [9]:
walks where each lattice site (edge) can be visited by at
most one monomer (bond), with an energy _−εb_ _<_ 0 associated with each pair of non-bonded nearest neighbor
(NN) monomers. This model does indeed present a _θ-_
point, where a continuous coil-globule transition takes
place, which is found to be of tri-critical nature in a
grand-canonical description of the system [10], as theoretically predicted by De Gennes [11]. Therefore, in
three-dimensional (3D) lattices, the _θ_ exponents assume
mean-field values, with logarithmic corrections to scaling [2]. In the 2D case, these exponents are non-classical
and believed to be those found by Duplantier and Saleur
(DS) [12] in the exact solution of the ISAW on a hexagonal lattice with hidden hexagons, as confirmed in several
numerical works [13].

A different model for collapsing polymers, receiving
also a considerable attention in the literature, is the interacting self-avoiding trail (ISAT) [14]. In this case, each
site of a _q-coordinated_ lattice can be visited by up to
_⌊q/2⌋_ monomers, respecting the restriction of only one
bond per edge, and the (bulk) self-attraction interaction
is associated with multiply visited sites. On the square
lattice, for example, an energy _−εb_ _<_ 0 is associated
with each doubly visited site, regardless of being a crossing or a ‘collision’ of the trail. If crossings are forbidden
in this system, one recovers the vertex-interacting SAW
(VISAW) model by Bl¨ote and Nienhuis (BN) [15], whose
solution presents a tri-critical point with exponents dif

_∗Electronic_ address: [nathan.rodrigues@ufv.br](mailto:nathan.rodrigues@ufv.br)
_†Electronic_ address: [tiago@ufv.br](mailto:tiago@ufv.br)
_‡Electronic_ address: [t.prellberg@qmul.ac.uk](mailto:t.prellberg@qmul.ac.uk)


-----

fering from the DS ones. Such a difference triggered a
long debate on what the generic exponents for the θ-point
of 2D polymers are (see, e.g., Refs. [16–18] for clear discussions on this). For the continuous collapse transition
of the ISAT model, several controversial results have been
reported in the literature, with works indicating that it
belongs to the BN [19] or some other undetermined universality class [20], while recent mean-field solutions on
hierarchical lattices have suggested that it may be of bicritical nature [21].
To investigate the adsorption transition, it is common
to introduce a (flat, homogeneous and impenetrable) surface in the systems above, such that one end of the
polymer is tethered to it, and giving a (surface) energy
_−εs_ _<_ 0 to either each monomer or each bond touching
the surface. When defined on the square lattice (where
the “surface” is a line), the adsorbing ISAW system is
known to present four phases, as indicated in Fig. 1(a).
The desorbed coil and globule phases are stable for small
_εs/kBT_ (where _kB_ is Boltzmann’s constant); polymers
have a negligible number, ns, of surface contacts, and the
two phases are separated by a line of _θ-points._ For large
_εs/kBT_ the system is found in an adsorbed phase, forming quasi-one-dimensional configurations where _ns_ _∼_ _n,_
with _n_ being the chain length. The coil-adsorbed transition is critical and the associated line meets the _θ-line_
at a multi-critical point known as the _special_ _adsorption_
_point_ (SAP) [23–26]. The change from the globule to the
adsorbed phase [e.g., by increasing the surface interaction
_εs/kBT_, while keeping the bulk one (εb/kBT ) fixed] is a
bit more complex. While early works reported a direct
globule-adsorbed transition [22, 23], further studies have
revealed the existence of an intermediary phase, known
as surface-attached-globule (SAG) [24–26], which is characterized by a simultaneous maximization of monomersurface and (non-adjacent) monomer-monomer contacts.
Continuous globule-SAG and SAG-adsorbed transitions
were found in Refs. [25, 26], yielding the phase diagram
displayed in Fig. 1(a) for the square lattice. We notice
that, on the cubic lattice, the adsorbed phase can be in
an extended (2D coil) or collapsed (2D globule) configuration, so that an additional transition exists in the phase
diagram of Fig. 1(a) [25, 26]. Some controversy on the
existence of a SAG phase is also found in the literature
for this 3D case (see, e.g., Refs. [25–28]).
The phase behavior of adsorbing ISATs has also been
considered in several works. For example, the model defined on the triangular lattice display a very rich phase
diagram, with desorbed coil, globule and crystal phases,
besides two types of adsorbed phases [29]. However, no
evidence of a stable SAG phase was found in Ref. [29].
In the same fashion, a SAG phase has never been observed in previous studies of this model on the square
lattice [30–33]. For instance, recent flatPERM simulations of this system [33] for the case where the surface
is in the horizontal direction of the square lattice and _εs_
is associated with monomer-surface contacts (let us refer
to it as ‘HS case’), revealed a phase diagram analogous


FIG. 1: Qualitative phase diagrams, as suggested by previous
works in the literature, for the adsorbing (a) ISAW and (b)
ISAT models on the square lattice, in terms of the strengths
of surface (εs/kBT ) versus bulk (εb/kBT ) interactions. The
solid (dashed) lines are continuous (discontinuous) transition
lines. The red squares indicate the SAPs.

to the one depicted in Fig. 1(b), with a discontinuous
globule-adsorbed transition, beyond the continuous coiladsorbed and coil-globule ones. A similar behavior was
found in a transfer matrix study by Foster [31], where
the surface interaction was associated with bond-surface
contacts (let us call it ‘BS case’).
The critical properties of the adsorption transitions
are also subject of much recent interest, in part motivated by the numerical studies by Plascak _et_ _al._ [34]
suggesting that the critical exponents for the ordinary
adsorption (i.e., the coil-adsorbed transition) may depend on the strength of bulk interactions, being thus
non-universal. Subsequent works have, however, provided evidence to the contrary, indicating that the exponent variation may be a consequence of strong finite-size
corrections [33, 35, 36]. The possibility of non-universal
behavior has recently been raised also for the special adsorption of ISATs on the square lattice [33]. Three scenarios for the surface-trail interactions were analyzed in
Ref. [33]: the HS and BS cases discussed just above, and
a ‘DS case’ where the surface is in the diagonal direction
and thus _εs_ is associated with each monomer touching
it. Intriguingly, different surface exponents were found
in each case, with 1/δ[(][s][)] = _φ[(][s][)]_ _≈_ 0.44 for the DS scenario (in good agreement with a previous study of this
system [30]) and appreciably smaller values for the BS
and HS cases. It is noteworthy that trails with up to
_n = 10240_ steps were analyzed in [33]; no indication was
found that such differences are due to finite-size effects.
In order to understand this very interesting issue, we
revisit the ISAT adsorption here, via extensive flatPERM
and PERM simulations, focusing on square lattices in
the HS and DS scenarios. Detailed analyses of the phase
diagrams of these systems reveal that the multi-critical
nature of the SAP of the ISAT model does indeed depend
on surface details. In fact, while we confirm the phase
behavior of Fig. 1(b) for the HS scenario, in the DS case a
diagram analogous the one for the ISAW [Fig. 1(a)] is obtained. As a means to explain the origin of this difference,
we investigate also a modified horizontal surface (mHS)
system — where the trails have to leave the surface af

-----

ter each one step on it (resembling the DS situation) —
and it also has the phase behavior of Fig. 1(a). For the
three (HS, DS and mHS) scenarios, the critical surface
exponents at the SAP were carefully estimated, for trails
with up to 102400 steps. While the exponent 1/δ[(][s][)] suggests the existence of two universality classes for the special adsorption, depending on whether the SAG phase is
present or absent, a more complex behavior is found for
the exponent _φ[(][s][)]._

The rest of the paper is organized as follows. In Sec.
II we define the model and surface scenarios analyzed,
as well as the Monte Carlo methods and quantities of
interest. The phase diagrams of these systems are investigated in the Sec. III, while the critical behavior at the
SAP is analyzed in Sec. IV. Our final discussions and
conclusion are presented in Sec. V.

**II.** **MODELS,** **SIMULATIONS** **AND**
**QUANTITIES** **OF** **INTEREST**

**A.** **Models** **and** **simulation** **methods**

A self-avoiding trail (SAT) is a lattice path where
each lattice edge can be visited only once. This restriction introduces an excluded volume effect, mimicking the
one present in dilute polymers, so that the SAT can be
regarded as a model for such polymers in a good solvent. By placing the monomers on the lattice sites, ⌊q/2⌋
monomers are allowed per site on a lattice of coordination _q._ The interacting SAT (ISAT) model is obtained
by assigning attractive on-site interactions among the
monomers in multiply occupied sites. Thereby, on the
square lattice, which is the case of interest here, each site
can be visited upmost twice and an energy _−εb_ _< 0_ will
be associated with each of such doubly occupied sites.

To investigate the polymer adsorption, we consider
that the square lattice is limited by a sticking boundary “surface” (a straight line, actually) where one end of
the polymer is tethered. The attractive polymer-surface
interaction will be introduced in the ISAT model by assigning an energy _−εs_ _<_ 0 to each monomer lying on
the surface. Beyond the cases of horizontal surface (HS)
and diagonal surface (DS) [see Fig. 1 of Ref. [33] for a illustration of them], we investigate the ISAT considering
also a modified horizontal surface (mHS) which does not
allow two consecutive steps of the trail on it. Namely, the
trail is forced to leave this surface after each step there,
as illustrated in Fig. 2. This leads to an adsorbed state
that resembles the one of the DS scenario, where the trail
naturally has to leave the surface after each contact with
it, forming a stair-like configuration in the fully adsorbed
(ground) state.

By defining the Boltzmann weights _κ_ = _e[ε][s][/k][B]_ _[T]_ and
_ω_ = _e[ε][b][/k][B]_ _[T]_, we may write the partition function of the


where the sum is evaluated over all _n-step_ trails _ψn._

To estimate these averages, we have performed numerical simulations with the PERM [37] and flatPERM [38]
algorithms. They both are methods where the trails
are stochastically grown starting from a monomer at the
origin, which is placed on the surface here. Strategies
for pruning and enriching trails based on their statistical weights are employed to avoid trapped configurations
and large dispersion of the weights; common problems of
more simple methods such as the Rosenbluth-Rosenbluth
one [39]. The flatPERM algorithm, in its more general
form, is used to determine _Cm[(][n]s[)],mb_ [,] [hence] [it] [is] [a] [more]
suitable method for exploring large portions of the parameter space (ω, κ). We used a form of flatPERM where
one of the parameters is kept fixed, which reduces the dimension of the density of states, allowing us to sample
longer trails. Indeed, trails with up to 1024 steps were
simulated, for several values of fixed ω (or κ), with ∼ 10[9]

samples being generated in each case. The version of the
PERM considered here is equivalent to flatPERM with
both parameters kept fixed. We performed these PERM
simulations along the coil-globule line (i.e., at ω = 3 [30])
for several values of _κ_ in the vicinity of the special adsorption point. In this case, we were able to sample much
longer trails, with up to 102400 steps. For each point
(ω = 3, κ), a total of _≈_ 10[10] trails were sampled for each
scenario.


FIG. 2: Example of a trail configuration in the mHS scenario,
where the trail has to leave the surface after each step on it.
A Boltzmann weight _ω_ is associated with each doubly visited
site in the bulk (regardless of being a crossing or a ‘collision’),
while each monomer at the surface has a weight _κ._ The dot
represents the origin, where the trail starts.

system as

_Zn(κ, ω) =_ � _Cm[(][n]s[)],mb_ _[κ][m][s]_ _[ω][m][b]_ _[,]_ (1)

_ms,mb_

where _Cm[(][n]s[)],mb_ [is] [the] [number] [of] _[n][-step]_ [trails] [with] _[m]s_
surface contacts and _mb_ doubly visited sites. Then, the
expected value of any quantity _Q_ is given by


_⟨Q⟩(κ, ω) =_ [1]

_Zn_


� _κ[m][s][(][ψ][n][)]ω[m][b][(][ψ][n][)]Q(ψn),_ (2)

_ψn_


-----

**B.** **Quantities** **of** **interest**

An important quantity to characterize the adsorption
transition is the surface internal energy, _un,_ defined as

_un(κ, ω) =_ _[⟨][m][s][⟩]_ _,_ (3)

_n_

where _⟨ms⟩_ is the average number of polymer-surface
contacts. This energy is also the order parameter for
the adsorption transition. Close to the adsorption point
it is expected to behave as [40, 41]

_un_ _∼_ _n[φ][−][1]f_ (τn[1][/δ]), (4)

where _τ_ = T _−_ _Ta_ is the temperature relative to the adsorption transition point Ta and the scaling function f (x)
is a constant at _x_ = 0. Thereby, the exponent _φ_ can be
estimated from the scaling _un_ _∼_ _n[φ][−][1]_ at _T_ = Ta. Moreover, the crossover exponent 1/δ is related to the finitesize scaling of the pseudo-critical temperature, _Ta(n),_
through

_Ta(n) = Ta + const. × n[−][1][/δ]._ (5)

Hence, the exponent 1/δ can be obtained from the relative fluctuation in the number of monomers at the surface:

Γn(T ) = _[d][ log]dTa[ u][n]_ = _[⟨][m]s[2][⟩−⟨]⟨ms⟩[m][s][⟩][2]_ _,_ (6)


whose maximum scales as

Γn,max _∼_ _n[1][/δ]._ (7)

At the normal adsorption transition, these exponents
are given by _φ_ = 1/δ = 1/2 in two-dimensions [42],
whereas a different value is expected for the special adsorption transition, as indeed observed in several works
(see, e.g., Refs. [30, 31, 33]). We recall that the special
transition takes place at (ω, κ) = (3, κs), since _ωs_ = 3 is
the critical parameter for the coil-globule transition for
the ISAT on the square lattice [30, 31, 33]. So, in order
to determine the exponents at the special point it is imperative to first estimate _κs_ with a good precision. One
of the best ways to do this is through the components
of the mean squared end-to-end distance, _Rn[2]_ [,] [parallel]
and perpendicular to the surface [35, 36]. For horizontal
surfaces, these components are simply given by

_R⊥[2]_ _,n[(][ω, κ][) =]_ �yn[2] � (8)

_R∥[2],n[(][ω, κ][) =]_ �x[2]n� _,_ (9)

where _xn_ and _yn_ are the end-point components of the
_n-step_ trail (since the starting point is located at the
origin). The definition is slight different for a diagonal
surface, being

_R⊥[2]_ _,n[(][ω, κ][) =]_ [1] �(x[2]n [+][ y]n[2] [)]� (10)

2


_R∥[2],n[(][ω, κ][) =]_ [1] �(x[2]n _[−]_ _[y]n[2]_ [)]� _,_ (11)

2

Similarly to other metric quantities, these components
present a scaling behavior determined by the Flory exponents, _ν⊥_ and _ν∥_ in this case, so that

_R⊥[2]_ _/∥,n_ _[∼]_ _[n][2][ν][⊥][/][∥]_ _[.]_ (12)

In the non-adsorbed phases _ν⊥_ = _ν∥,_ whereas in the
(quasi-one-dimensional) adsorbed phase _ν⊥_ _→_ 0 and
_ν∥_ _→_ 1. Due to finite-size effects, these exponents
cross at some intermediate temperature, and the crossing
point can be identified as the pseudo-critical temperature
_Ta(n)._

**III.** **PHASE** **BEHAVIOR** **OF** **THE** **ADSORBING**
**ISATS**

**A.** **HS** **and** **DS** **systems**

We will start our study of the adsorbing ISATs by revisiting and completing the phase diagrams for the HS
and DS systems. Since the collapse transition of the ISAT
on the square lattice is exactly know to be located at
_ωs_ = 3 [30] and this is a bulk transition — not affected
by the presence of a weakly interacting surface —, a coilglobule line is expected in the phase diagrams for the
adsorbing ISATs at _ω_ = _ωs_ = 3 for small _κ,_ as indeed
observed in Refs. [30, 31, 33]. This line ends at the special adsorption point (SAP), where the collapsing trails
adsorb. For _ω_ _<_ 3, upon increasing _κ_ one finds a continuous coil-adsorbed transition, often refereed to as the
ordinary adsorption transition. A detailed analysis of
this transition for the HS and DS systems was reported
in Ref. [33], revealing that the surface exponents along
the corresponding critical lines agree with those for the
SAW universality class: _φ_ = 1/δ = 1/2 [42]. Some evidence for a first-order globule-adsorbed transition was
also found in Ref. [33] for _ω_ _>_ 3, indicating phase diagrams analogous to the one in Fig. 1(b) for both the HS
and DS scenarios. However, this analysis was based on
very short trails (with up to 128 steps) in some cases and,
thus, we will concentrate in this region here to determine
how the collapsed chains (for small _κ)_ become adsorbed
(for large _κ)._
Figure 3 shows the variation of the fluctuation in the
number of monomers at the surface with κ, for ω = 3.50.
In the HS case [Fig. 3(a)], one finds a single peak in
this quantity, confirming the existence of a direct globuleadsorbed transition, as suggested in Ref. [33]. Moreover,
the fast increase of these peaks with length _n_ is a clear
signature of the first-order nature of this transition. We
have confirmed this also through the density of states
_Cm[(][n]b[)],ms_ [versus] _[κ]_ [(not] [shown),] [which] [displays] [a] [bimodal]
behavior around the transition point. Similar results are
found for other values of _ω_ _>_ 3, yielding a first-order
globule-adsorbed transition line in the phase diagram,


-----

60

50

40

30

20

10

0


1


0.5

0


2 4 6
### κ


-0.5

1 2 3 4
### κ

1


5


4

3


0.75

0.5


0.25


2


(b)
κ κ*
c c

Adsorbed
SAG

2 4 6 8
### κ


0

1 2 3 4 5 6
### κ

|(b)<br>Globule|SAG<br>n<br>128<br>256<br>512<br>1024|Adsorbed|
|---|---|---|

|c<br>SAG|κ*<br>c<br>Adsorbed|
|---|---|


FIG. 3: Fluctuation Γn as function of _κ_ for fixed _ω_ = 3.50 for
the HS (a) and DS (b) scenarios and lengths (from bottom to
top) _n_ = 128, 256, 512 and 1024. The vertical dotted lines in
(b) indicate the location of the globule-SAG (κc) and SAGadsorbed (κ[∗]c [)] [phase] [transition] [for] _[n][ = 1024.]_

which starts at the SAP and extends to large values of ω.
Hence, the diagram for the HS case is indeed analogous
to the one shown in Fig. 1(b), as can be seen in Fig. 4 of
Ref. [33], where the HS scenario was denoted as MS.
A quite different behavior is found here for the adsorption of the collapsed phase in the DS case. As shown in
Fig. 3(b), upon increasing _κ_ the polymer-surface-contact
fluctuations present now two peaks in the region of ω _> 3._
Therefore, two transitions take place in the system there,
one at κc and another one at κ[∗]c _[> κ][c][.]_ [Note that,] [in con-]
trast with the HS case, the maxima of both peaks in
Fig. 3(b) display a slow increase with the length _n,_ indicating that these transitions are continuous. The analysis of the density of states confirms this and, moreover, it
shows that for weak surface interactions (i.e., for κ < κc)
the trail has few polymer-surface contacts and a large
number of doubly visited sites, as expected for the globule phase of ISAT. In the opposite regime of very strong
interactions (i.e., for _κ_ _>_ _κ[∗]c_ [)] [the] [chains] [are] [adsorbed,]
displaying a large surface energy _un_ _→_ 1 and few sites
occupied by two monomers. For _κc_ _<_ _κ_ _<_ _κ[∗]c_ [,] [one] [finds]
an intermediate phase, which is dense (similarly to the
globule one) but has a macroscopic number of monomers


FIG. 4: Effective surface exponent _φ(n)_ as function of _κ,_ for
_ω_ = 3.50 and several lengths _n,_ for the HS (a) and DS (b)
systems. The asymptotic values of _κc_ and _κ[∗]c_ [,] [for] [this] _[ω][,]_ [are]
indicated by the vertical dotted lines in (b).

at the surface, characterizing a SAG phase.
Evidence for the existence (absence) of a SAG phase
in the DS (HS) system is found also in the _φ_ exponent,
which is expected to assume the values _φ = 0_ in the desorbed phase, _φ_ = 1/2 in the SAG phase, and _φ_ = 1 in
the fully adsorbed phase [24, 25]. Following the scaling in
Eq. 4, effective exponents, _φ(n),_ were estimated here as
_φ(n) = ln(un/un/2)/ ln 2._ Figures 4(a) and 4(b) compare
the variation of _φ(n)_ with _κ,_ for fixed _ω_ = 3.50, for the
HS and DS systems, respectively. In the former case, the
exponents rapidly change from φ ≈ 0 to φ ≈ 1, presenting
an abrupt variation close to the transition point, where
it becomes negative (for long trails) due to the discontinuous nature of the globule-adsorbed transition in the HS
system. On the other hand, a smooth variation of _φ(n)_
is seen in Fig. 4(b) for the DS case, where approximate
plateaus are observed for intermediate values of κ, which
get close to φ = 1/2 as n increases. This confirms the existence of a continuous globule-SAG transition at _κc_ and
a SAG-adsorbed one at _κ[∗]c_ [in] [the] [DS] [scenario,] [whereas]
a single discontinuous globule-adsorbed transition exists
in the HS case.
From the crossing points of the curves of _φ(n)_ _×_ _κ_
for different values of _n_ in Fig. 4(b), we may obtain es

-----

timates for the effective critical exponents, _φc(n),_ and
for the pseudo-critical points _κc(n)_ and _κ[∗]c_ [(][n][).] [It] [turns]
out that in the vicinity of the SAP these two critical
points become very close (since the globule-SAG and
SAG-adsorbed critical lines meet at the SAP, as discussed
below), and crossings start appearing only for curves for
very large _n._ Moreover, it is hard to sample characteristic configurations of trails for large _ω_ (i.e., deep inside
the collapsed phase) with growth methods such as PERM
and flatPERM, and, consequently, the data present considerable fluctuations in this region. For these reasons,
we were unable to reliably extrapolate the outcomes from
the crossing points for _n →∞._ Anyhow, from the crossings of curves for _n_ = 896 and _n_ = 1024, we found _φc_
in the intervals: 0.2 < φc _< 0.3_ for the globule-SAG and
0.7 < φc _< 0.8_ for the SAG-adsorbed transition.
Since the globule-SAG transition is a critical adsorption transition, the corresponding critical line can be
determined by using the same procedure employed in
Ref. [33] for the coil-adsorbed transition. Namely, for a
fixed value of _ω,_ we firstly determine the maxima in the
Γn × _κ curves and, then, estimate the crossover exponent_
1/δ from Eq. 7. Exponents in the range 0.2 < 1/δ _< 0.3_
were found for the values of ω analyzed here, in fair agreement with the values of _φc_ estimated above. Finally,
by using these exponents in Eq. 5, we extrapolate the
pseudo-critical points _κc(n)_ obtained from the crossing
points of the curves of the Flory exponents _ν⊥_ and _ν∥_
versus _κ,_ to determine the critical point. As _ω_ tends to
_ωs_ = 3, we find _κc_ approximating the exact value for the
SAP in the DS case (κ[(]c[s][)] = 3 [30]), strongly indicating
that the globule-SAG transition line starts at the SAP.
As demonstrated in Fig. 5(a), which presents the phase
diagram for the DS system, the line κc(ω) is a decreasing
function of _ω,_ in agreement with the behavior found in
Refs. [24, 25] for adsorbing ISAWs.
The procedure above does not work for determining
the SAG-adsorbed transition line, because it happens between two adsorbed phases and, thus, the end-to-end distance (and related Flory exponents) gives no clue about
the location of the transition. Hence, the clearest signature of this transition is observed in the Γn _× κ_ curves,
as those in Fig. 3(b), where the values of _κ_ at the maxima can be regarded as the pseudo-critical points _κ[∗]c_ [(][n][).]
Along the SAG-adsorbed line we estimate crossover exponents in a broad interval 0.3 < 1/δ _< 0.6._ Once again,
it is hard to obtain reliable extrapolations of _κ[∗]c_ [(][n][)] [to]
the _n_ _→∞_ limit, due to strong finite-size corrections
and fluctuations in their values, which seems to be more
prominent in the SAG-adsorbed transition. For this reason, we are simply considering _κ[∗]c_ [(][n][)] [for] _[n]_ [=] [1024] [as]
the transition point in the phase diagram of the DS system, in Fig. 5(a). These results strongly indicate that
the SAG-adsorbed line meets the globule-SAG one at the
SAP, giving rise to a phase diagram qualitatively analogous to the one in Fig. 1(a) for adsorbing ISAWs in
two-dimensions.
In order to understand why the SAG phase appears


FIG. 5: Phase diagrams for the DS (a) and mHS (b) systems. The symbols are the estimated transition points, while
the continuous lines connecting them are guides-to-eye. Our
results indicate that all transition lines are continuous. The
green squares denote the special adsorption points.

in the DS system, but is absent in the HS scenario, we
start remarking that it has the same bulk properties of
the globule phase, _i.e.,_ it is a compact configuration rich
in doubly visited sites. However, at the same time it
has to have a macroscopically large number of surface
contacts. It turns out that when the globule phase adsorbs onto the horizontal surface, the layer immediately
above the surface can not be fully populated with doubly visited sites. Indeed, in the HS case the adsorption of
a straight segment creates a kind of depletion zone just
above it, as illustrated in Fig. 6(a), hindering the formation of compact configurations in this region. On the
other hand, as shown in Fig. 6(b), in the DS case the trail
cannot have straight segments along the surface, since it
has to move away from it after each contact. Therefore,
the configuration closest to the straight one consists in
a zigzag (or laddered) structure, with the trail leaving
and returning to the surface after each two steps, producing a sequence of visited sites on the layer just above
it. This allows for the formation of doubly visited sites
in this layer and any other close to the surface, such that


7


-----

8


FIG. 6: Illustrations of adsorbing globule configurations in
the (a) HS and (b) DS systems. The starting points of the
trails are denoted by the black circles.

the globule phase can adsorb (simultaneously maximizing the monomer-monomer and monomer-surface interactions), yielding the SAG phase in the DS case.


**B.** **mHS** **system**

The reasoning above lead us immediately to inquiry
whether the SAG phase can be induced in the HS scenario
if in some way the adsorbing portion of the trail also
visits a similar number of sites twice in the layer just
above the surface. This can be achieve by considering
a modified HS (mHS) system where, after each step on
the horizontal surface, the trail is forced to move away
from it, somewhat mimicking the DS scenario. In fact,
the completely adsorbed configuration in this case has
a square wave form, while in the DS case it is triangle
wave-like.
The thermodynamic behavior of the mHS case was obtained following the same procedures as above for the
other systems, once again for trails with up to 1024 steps.
Analogously to the other cases, a desorbed coil phase is
found in the region of small _κ_ and _ω._ By increasing _ω,_


FIG. 7: (a) Fluctuation Γn as a function of _κ_ for the mHS
case, for fixed _ω_ = 3.50 and several lengths _n._ (b) Effective surface exponents _φ(n)_ versus _ω,_ for _n = 1024_ and three
values of _κ,_ as indicated by the legend.

a coil-globule transition is observed at _ωs_ = 3 for small
values of _κ,_ as expected. Moreover, upon increasing _κ,_
for fixed _ω_ _< 3,_ the system undergoes a continuous coiladsorbed transition. For a given _ω_ _<_ 3, we find that
this transition occurs for κmHS _> κHS._ This is indeed expected, since the trails are forced to leave the surface in
the mHS case — decreasing thus the number of polymersurface contacts when compared with the HS scenario
— it requires a larger _κ_ to adsorb. Critical exponents
1/δ _≈_ _φ_ _≈_ 1/2 were found along the coil-adsorbed transition line, though some deviations from this value were
observed close to _ωθ,_ which are certainly due to strong
finite-size corrections in this region, as also observed for
the DS and HS systems in Ref. [33].
For _ω_ _>_ 3, the globule phase is observed in the mHS
system for small values of _κ,_ as just mentioned above.
By increasing _κ,_ for a fixed _ω,_ a behavior very similar
to the one for the DS case is found. As demonstrated in
Fig. 7(a), the fluctuation Γn displays two peaks, indicating the presence of three phases in this region. Figure
7(b) shows the variation of the effective exponent _φ(n)_
with _ω,_ for _n_ = 1024 and fixed _κ_ = 1.0, _κ_ = 2.5 and
_κ = 4.0._ While for _κ = 1.0_ the exponents always remain
close to _φ_ = 0, consistently with the expected behavior


-----

for the coil and globule phases, for _κ_ = 2.5 they change
from _φ ≈_ 0 to _φ ∼_ 1/2, indicating the presence of a SAG
phase for large _ω._ For _κ_ = 4.0 one finds _φ_ _≈_ 1 in the
adsorbed phase, for small _ω,_ and then a decreasing to
_φ_ _≈_ 1/2, confirming the existence of the SAG phase in
the mHS system.
The faster increasing of the maxima of Γn at the SAGadsorbed transition, as well as the somewhat abrupt variation in _φ(n) × ω_ in this case, as observed in Fig. 7, may
indicate that this is a discontinuous transition. However,
no bimodal behavior was found in the density of states
close to the SAG-adsorbed transition; as well as close
to the globule-SAG transition. Therefore, it seems that
these are both continuous transitions, similarly to the
DS case. The resulting phase diagram for the mHS system is depicted in Fig. 5(b), which is qualitatively analogous to the DS one, with all transition lines meeting
at the SAP. This demonstrates that the phase behavior
of these adsorbing models can be strongly affected by
simple changes in the surface features. Moreover, this
confirms that the depletion effect caused by the adsorption of straight segments in the HS scenario is indeed the
reason for the absence of the SAG phase in this system.

**IV.** **SPECIAL-ADSORPTION** **POINT**

Next, we explore in detail the special adsorption point
(SAP), determining its location for each scenario and the
surface exponents _φ[(][s][)]_ and 1/δ[(][s][)] on it.
We remark that the existence of a SAG phase in the
DS and mHS systems modifies the multi-critical nature
of their SAPs, when compared with the one for the HS
scenario. Indeed, while in the former case four continuous transition line meet at the SAPs (see Fig. 5), in the
HS case there are only two of such lines connecting with
a coexistence one there. Hence, the surface exponents
do not necessarily have to be the same for these systems at their SAPs, which explains the different values
reported for _φ[(][s][)]_ and 1/δ[(][s][)] for the DS and HS scenarios
in Refs. [30, 33]. This also raises the interesting question
of whether the exponents for the mHS and DS systems
are the same, belonging to a possible universality class for
systems with a SAG phase. To answer this, besides characterizing the SAP behavior for the mHS case, we also
improve the results for the other systems, by performing
extensive PERM simulations at _ω_ = 3, for several values
of _κ_ in the vicinity of the SAPs. Since these simulations
were carried out keeping both _ω_ and _κ_ fixed, we were
able to investigate trails with up to 102400 steps (which
are 10× longer than those considered in Ref. [33]) and a
very large statistics, with up to ∼ 10[10] trails being grown
for each scenario and set of parameters.
To determine the value of κs, we estimate first pseudocritical points _κs(n)_ from three distinct quantities: the
points of maxima in the curves of Γn; the crossing points
of the parallel and perpendicular Flory exponents; and
the crossing points of the surface exponents _φ,_ all of


FIG. 8: (a) Effective crossover exponents 1/δ[(][s][)] as function
of _n[−][0][.][5]_ for all scenarios considered. (b) Extrapolation of
the pseudo-critical estimates of _κs_ for the mHS system from
the three indicated quantities. (c) Effective exponents _φ[(][s][)]_

against _n[−][0][.][5]_ for all analyzed systems. The dotted line in
(a) indicates the value 1/δ[(][s][)] = 0.44 previously found in the
literature, while the dashed lines in (b) and (c) correspond to
the best linear fits of the data in each case.

them measured as function of _κ._ We obtain the asymptotic values of κs following the same procedure employed
in the previous section, _i.e.,_ by firstly determining the
crossover exponent for each model and, then, using such
exponents to extrapolate κs(n) to the n →∞ limit. Figure 8(a) shows the finite-size estimates of 1/δ[(][s][)], where


-----

one sees that these exponents have very similar values
for the DS and mHS systems, which fluctuate in the interval 0.430 _<_ 1/δ[(][s][)] _<_ 0.448. For the HS model, on
the other hand, significantly smaller values are found,
in the range 0.332 _<_ 1/δ[(][s][)] _<_ 0.352. Since these exponents fluctuate around constant values without any
clear tendency to increase or decrease, we may simply
take their average for long trails to obtain an estimate of
their asymptotic values. Considering lengths _n > 10000,_
this yields: 1/δDS[(][s][)] [=] [0][.][439(3),] [1][/δ]mHS[(][s][)] [=] [0][.][438(5)] [and]
1/δHS[(][s][)] [=] [0][.][343(6),] [strongly] [indicating] [that] [the] [DS] [and]
mHS exponents are indeed the same. We notice that our
result for the HS case is slightly larger than the previous estimate from Ref. [33] [1/δ[(][s][)] = 0.303(22)], which is
likely due to the much longer trails considered here.

With the crossover exponents at hand, we may use
them in a finite-size scaling Ansatz analogous to Eq. 5
to find the asymptotic values of _κs_ at the SAPs. Since
_κs_ is exactly known for the DS case [30], it is interesting to start the analysis with this system, to benchmark our method. In fact, this gives the extrapolated
results: _κs_ = 3.005(5) [from the crossings of the _φ(n)],_
_κs_ = 3.002(3) (from the crossings of the Flory exponents)
and _κs_ = 2.998(8) (from the maxima of Γn), in striking
agreement with the expected result κs = 3 [30]. Although
the three quantities return very similar values for _κs,_
the pseudo-critical values from the maxima of Γn present
much stronger corrections than those from the crossing
points of _φ(n)_ and _ν⊥/∥._ The very same behavior is observed in Fig. 8(b) for the mHS system, as well as in the
HS case (not shown). Similarly to Fig. 8(b), in all cases
the data are very well linearized when plotted against
_n[−][1][/δ][(][s][)]_ with the values of 1/δ[(][s][)] found above. This
demonstrates the reliability of these exponents and gives
further confirmation that the DS and mHS systems behave in the same way. Since the extrapolated values of κs
obtained from different quantities are always very close to
each other, we have determined the location of the SAPs
by taking their average, which gives: _κ[(]s[DS][)]_ = 3.002(5),
_κ[(]s[HS][)]_ = 1.927(2) and _κ[(]s[mHS][)]_ = 2.602(5).

To determine the exponent _φ[(][s][)],_ we have considered
two different methods: (i ) the crossing points of curves
of _φ(n)_ and _φ(n + ∆n)_ versus _κ_ (considering trails sizes
in the interval 10240 _≤_ _n_ _≤_ 102400 with ∆n = 10240);
and (ii ) the scaling of the surface internal energy in Eq.
4 at the SAP (i.e., for _ωs_ = 3 and the values of _κs_ just
estimated above). Although approach (i ) has the advantage of allowing us to estimate _φ[(][s][)]_ without knowing the SAP location, we observed that it usually gives
not so precise results, because even small fluctuations in
the curves of _φ(n)_ _×_ _κ_ can produce appreciable variations in their crossing points. Method (ii ) yields more
precise estimates for effective _φ[(][s][)](n),_ calculated by averaging the slopes of several linear fits of log un _× log n_
curves at the SAP for _n-length_ trails. Figure 8(c) shows
the extrapolation of the exponents calculated in this way
at the central values of the estimates for _κs_ above. It


is important to notice that the outcomes from this procedure are very sensitive to the values of _κs_ used. In
fact, a variation of such values at the third decimal place
(within their error bars) may yield a change at the second figure in the final results for _φ[(][s][)]._ Despite these
caveats, consistent exponents were obtained from both
methods, when the central values of _κs_ are used in procedure (ii ). For example, in the DS system method
(i ) gives exponents in the range 0.432 _<_ _φ[(][s][)]_ _<_ 0.461,
whose average yields _φ[(]DS[s][)]_ [=] [0][.][449(14),] [while] [from] [ap-]
proach (ii ) we obtain _φ[(]DS[s][)]_ [=] [0][.][446(4).] Both results
agree quite well among them, as well as with previous
estimates _φ[(][s][)]_ _≈_ 0.44 [30] and _φ[(][s][)]_ _≈_ 0.447(18) [33]. In
the HS case, we find 0.323 _<_ _φ[(]HS[s][)]_ _[<]_ [0][.][359] [with] [the]
average value _φ[(]HS[s][)]_ [=] [0][.][347(16)] [in] [approach] [(][i] [),] [which]
agrees quite well with _φ[(]HS[s][)]_ [=] [0][.][349(5)] [obtained] [from]
method (ii ). In a similar fashion, for the mHS system method (i ) gives 0.387 _<_ _φ[(]mHS[s][)]_ _[<]_ [0][.][421] [with] [the]
average _φ[(]mHS[s][)]_ [=] [0][.][398(12),] [while] [approach] [(][ii] [)] [yields]
_φ[(]mHS[s][)]_ [=] [0][.][390(6).] [Intriguingly,] [for] [the] [mHS] [system] [our]
results suggest that _φ[(]mHS[s][)]_ _[̸][= 1][/δ]mHS[(][s][)]_ [,] [whereas] [an] [equal-]
ity between these exponents is found in the other cases.

**V.** **CONCLUSION**

By performing extensive flatPERM and PERM simulations, we have studied the thermodynamic properties of
adsorbing ISATs for three surface scenarios on the square
lattice: horizontal surface (HS), diagonal surface (DS)
and a modified HS (mHS) case where the trail is forced
to leave a horizontal surface after each one step on it. Our
careful analyzes uncover key properties of these systems
not reported in the literature.
We found a surface-attached-globule (SAG) phase in
the DS system, between the globule and adsorbed phases,
which has never been observed in previous studies of
this scenario [30, 33]. Therefore, similarly to adsorbing ISAWs, the DS system’s phase diagram presents four
stable phases: desorbed coil and globule, SAG and adsorbed. Our results strongly indicate that all transition
lines separating them are continuous, so that its special
adsorption point (SAP) is featured by the meeting of four
continuous transition lines: coil-globule, coil-adsorbed,
SAG-adsorbed and globule-SAG. The same phase behavior is found in the mHS case. Indeed, due to the geometric restrictions imposed by the surfaces in both DS
and mHS scenarios, their fully adsorbed phases consist
in wave-like conformations that equally visit the surface
and the first layer of sites just above it. Out of these
ground states, this property allows the collapsed phase
to partially wet these surfaces, yielding a stable SAG
phase.
In contrast, in the HS system, the adsorbed configurations are featured by long straight segments on the
surface, which creates a kind of “depletion zone” in the


-----

layer just above them (where the sites cannot be doubly
occupied), preventing a simultaneous maximization of
monomer-monomer and monomer-surface contacts. For
this reason, the SAG phase is not observed in the HS
case and a direct (first-order) globule-adsorbed transition is found in its phase diagram. This explains also why
the SAG phase was not found in previous works [31, 33]
for the BS system, where bonds (rather than sites) of
the trail interact with a horizontal surface. It is worth
remarking also that no surface-induced depletion effect
exists for ISAWs, where the sites are always visited by
at most one monomer. Thereby, we may expect that different surface scenarios shall not change the topology of
the ISAW phase diagrams.
These results are in agreement with a recent reasoning
of Foster _et_ _al._ [32] to explain the different universality
classes for the collapse transition in the ISAW and ISAT
models. Indeed, it was argued in Ref. [32] that, while
long enough ISAWs do not “see” the underlying lattice
(so that their critical properties only depend on dimensionality), the ISAT behavior may depend on the lattice
where they are placed. In the same token, it is somewhat expected that the thermodynamic properties of adsorbing ISATs may be indeed sensitive to details of the
surface. We emphasize, however, that for all scenarios
considered here and elsewhere [33], critical surface exponents consistent with the expected value _φ_ = 1/δ = 1/2
were found for the ordinary adsorption transition.
A different situation is observed for these exponents at
the SAPs, whose multi-critical nature in the HS case is
different from that in the DS and mHS scenarios. This
certainly explains the different values of _φ[(][s][)]_ and 1/δ[(][s][)]

found in previous works [30, 32, 33] and confirmed here.
In fact, giving the very long trails considered in our analysis (with up to 102400 steps), it seems very unlikely
that the appreciable difference between the crossover exponents in Fig. 8(a) [yielding 1/δDS/mHS[(][s][)] _[≈]_ [0][.][44] [and]

1/δHS[(][s][)] _[≈]_ [0][.][34]] [is] [due] [to] [finite-size] [corrections.] [Instead,]

[1] P. Flory, _Principles_ _of_ _Polymer_ _Chemistry_ (Cornell University, Ithaca, 1953).

[2] P.-G. De Gennes, _Scaling_ _Concepts_ _in_ _Polymer_ _Physics_
(Cornell University Press, Ithaca, NY, 1979).

[3] J. des Cloizeaux and G. Jannink, _Polymers_ _in_ _Solu-_
_tion:_ _Their Modelling and Structure_ (Clarendon, Oxford,
1990).

[4] L. S. Hirst, _Fundamentals_ _of_ _Soft_ _Matter_ _Science_ (2nd
edition, CRC Press, Boca Raton, 2019).

[5] _Polymer Science and Innovative Applications:_ _Materials,_
_Techniques,_ _and_ _Future_ _Developments,_ edited by M. A.
Ali Al-Maadeed, D. Ponnamma, M. A. Carignano (Elsevier, 2020).

[6] S. Pavlukhina, and S. Sukhishvili, _Polymer_ _Adsorption._
In _Encyclopedia_ _of_ _Polymer_ _Science_ _and_ _Technology,_ _4th_
_ed.,_ edited by H. F. Mark (Wiley, New York, 2014).

[7] B. Kronberg, K. Holmberg, B. Lindman, _Surface_ _Chem-_


it strongly indicates that two universality classes exist
for the SAPs of ISATs on the square lattice, depending
on whether the SAG phase is present (as in the DS and
mHS systems) or absent in the phase diagram (as in HS
case).
The _φ[(][s][)]_ exponents, notwithstanding, seems to suggest a different picture, since different values are found
for each system [Fig. 8(c)], with _φ[(]HS[s][)]_ _[<]_ _[φ][(]mHS[s][)]_ _[<]_ _[φ][(]DS[s][)]_ [.]
A possible explanation for this intriguing finding is the
sensitivity of these estimates with the value used for the
SAP coordinate _κs,_ so that an inaccuracy in _κs_ for the
mHS case could be yielding an underestimated result for
_φ[(]mHS[s][)]_ [.] Note that in the DS and HS scenarios we obtained _φ[(][s][)]_ _≈_ 1/δ[(][s][)], as expected. It is important to
remark, however, that the phase diagrams in Fig. 5 indicates a subtle difference in the way the globule-SAG
critical line arrives at the SAPs for the DS and mHS
scenarios. In fact, in the latter case it appears to connect tangentially to the coil-adsorbed line, while in the
DS system it forms a very small angle with the (vertical) coil-globule line, suggesting that they both may get
parallel at the SAP. Although it is unclear to us how
this could yield _φ[(]mHS[s][)]_ _[<]_ _[φ][(]DS[s][)]_ [=] [1][/δ]mHS[(][s][)] [=] [1][/δ]DS[(][s][)] [,] [this]
may be a clue to explain such behavior. Of course, additional studies of these systems are needed to determine
how many sets of critical exponents exist for the special
adsorption of two-dimensional polymers.

**Acknowledgments**

T.J.O. and N.T.R. acknowledge financial support from
CNPq, FAPEMIG and FAPERJ (Brazilian agencies).
This research utilized Queen Mary’s Apocrita HPC facility, supported by QMUL Research-IT [43], on which
all simulations have been performed.

_istry_ _of_ _Surfactants_ _and_ _Polymers_ (Wiley, New York,
2014).

[8] C. Vanderzande, Lattice _Models_ _of_ _Polymers_ (Cambridge
University Press, Cambridge, 1998).

[9] W. J. C. Orr, Trans. Faraday Soc. **43,** 12 (1947).

[10] For summaries of estimates for the _θ_ point of the ISAW
on the square and cubic lattices see, e.g., J. H. Lee, S.Y. Kim and J. Lee, J. Chem. Phys. **133,** 114106 (2010);
Phys. Rev. E 86, 011802 (2012); S.-S. Huang, Y.-H. Hsieh
and C.-N. Chen, Polymers **14,** 4536 (2022).

[11] P.-G. De Gennes, J. Physique Lett. **36,** 55 (1975).

[12] B. Duplantier and H. Saleur, Phys. Rev. Lett **59,** 539
(1987); **60,** 1204 (1988);

[13] F. Seno and A. L. Stella, J. Phys. (Paris) 49, 739 (1988);
B. Duplantier and H. Saleur, Phys. Rev. Lett. **62,** 1368
(1989); T. Prellberg and A. L. Owczarek, J. Phys. A:
Math. Gen. **27,** 1811 (1994); P. Grassberger and R. Heg

-----

ger, J. Phys. I France **5,** 597 (1995); N. T. Rodrigues
and T. J. Oliveira, J. Phys. A: Math. Theor. **47,** 405002
(2014).

[14] A. R. Massih and M. A. Moore, J. Phys. A: Math. Gen.
**8,** 237 (1975).

[15] H. W. J. Bl¨ote and B. Nienhuis, J. Phys. A **22,** 1415
(1989); M. T. Batchelor, B. Nienhuis, and S. O. Warnaar, Phys. Rev. Lett. **62,** 2425 (1989); B. Nienhuis, Int.
J. Mod. Phys. B **4,** 929 (1990); S. O. Warnaar, M. T.
Batchelor, and B. Nienhuis, J. Phys. A: Math. Gen. **25,**
3077 (1992).

[16] D. P. Foster, Phys. Rev. E **84,** 032102 (2011).

[17] E. Vernier,[´] J. L. Jacobsen, and H. Saleur, J. Stat. Mech.:
Theory Exp. (2015) P09001.

[18] A. Nahum, Phys. Rev. E **93,** 052502 (2016).

[19] D. P. Foster, J. Phys. A **42,** 372002 (2009).

[20] Y. Shapir and Y. Oono, J. Phys. A **17,** L39 (1984);
H. Meirovitch and H. A. Lim, Phys. Rev. A **38,** R1670
(1988); I. Chang and H. Meirovitch, Phys. Rev. Lett. 69,
2232 (1992); J. Lyklema, J. Phys. A **18,** L617 (1985); A.
Guha, H. A. Lim, and Y. Shapir, _ibid._ **21,** 1043 (1988);
A. L. Owczarek and T. Prellberg, J. Stat. Phys. **79,** 951
(1995); Physica A **373,** 433 (2007).

[21] T. J. Oliveira and J. F. Stilck, Phys. Rev. E **93,** 012502
(2016); W. G. Dantas, T. J. Oliveira, J. F. Stilck and T.
Prellberg, Phys. Rev. E 95, 022132 (2017); T. J. Oliveira,
W. G. Dantas, T. Prellberg and J. F. Stilck, J. Phys. A:
Math. Theor. **51,** 054001 (2018).

[22] A. R. Veal, J. M. Yeomans and G. Jug, J. Phys. A: Math.
Gen. **24,** 827 (1991).

[23] D. P. Foster, E. Orlandini, and M. C. Tesi, J. Phys. A
**25,** L1211 (1992).

[24] Singh Y., Giri D. and Kumar S., J. Phys. A: Math. Gen.
**34,** L67 (2001).

[25] Rajesh R., Dhar D., Giri D., Kumar S. and Singh Y.,
Phys. Rev. E. **65,** 056124 (2002).



[26] P. K. Mishra, D. Giri, S. Kumar and Y. Singh, Physica
A **318,** 171 (2003).

[27] Krawczyk J, Owczarek A L, Prellberg T and Rechnitzer
A 2005 Europhys. Lett. 70 726–32

[28] A. L. Owczarek, A. Rechnitzer, J. Krawczyk, and T.
Prellberg, J. Phys. A: Math. Theor. 40, 13257 (2007).

[29] N. T. Rodrigues, T. J. Oliveira, T. Prellberg and A. L.
Owczarek, Phys. Rev. E **100,** 062504 (2019).

[30] A. L. Owczarek and T. Prellberg, J. Stat. Phys. **79,** 951
(1995).

[31] D. P. Foster, . Phys. A: Math. Theor. **43,** 335004 (2010).

[32] D. P. Foster, R. Kenna, C. Pinettes, Entropy **21,** 153
(2019).

[33] N. T. Rodrigues, T. Prellberg and A. L. Owczarek, Phys.
Rev. E **100,** 022121 (2019) .

[34] J. A. Plascak, P. H. L. Martins, and M. Bachmann, Phys.
Rev. E **95,** 050501(R) (2017); P. H. L. Martins, J. A.
Plascak, and M. Bachmann, J. Chem. Phys. 148, 204901
(2018).

[35] C. J. Bradly, A. L. Owczarek, and T. Prellberg, Phys.
Rev. E 97, 022503 (2018).

[36] C. J. Bradly, A. L. Owczarek, and T. Prellberg, Phys.
Rev. E 98, 062141 (2018).

[37] P. Grassberg; Phys. Rev. E. **56,** 3682 (1997).

[38] T. Prellberg and J. Krawczyk; Phys.Rev.Lett. 92, 120602
(2004).

[39] M. N. Rosenbluth and A. W. Rosenbluth; J.Chem.Phys.
**23,** 2 (1955).

[40] P.G.De Gennes, J. Phys. (France) **37,** 1445 (1976).

[41] C. Vanderzande, A. L. Stella, and F. Seno, Phys. Rev.
Lett. **67,** 2757 (1991).

[42] Burkhardt T W, Eisenriegler E and Guim I, Nucl. Phys.
B, **316** 559–72 (1989).

[43] `http://doi.org/10.5281/zenodo.438045`


-----

