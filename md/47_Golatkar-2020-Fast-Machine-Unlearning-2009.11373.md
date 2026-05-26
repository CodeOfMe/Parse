## Fermionic atoms in a spin-dependent optical lattice potential: topological insulators with broken time-reversal symmetry

Igor Kuzmenko[1], Miros�law Brewczyk[2], Grzegorz �Lach[3], Marek Trippenbach[3], Y. B. Band[1]

1 Department of Chemistry, Department of Physics,
Department of Electro-Optics, and the Ilse Katz Center for Nano-Science,
Ben-Gurion University, Beer-Sheva 84105, Israel
2 Wydzia�l Fizyki, Uniwersytet w Bia�lymstoku, ul. K. Cio�lkowskiego 1L, 15-245 Bia�lystok, Poland
3 Faculty of Physics, University of Warsaw, ul. Pasteura 5, 02-093 Warsaw, Poland

We propose a novel approach to study the topological properties of matter. In this approach,
fermionic atoms are placed in an external magnetic field and in a two-dimensional spin-dependent
optical lattice (SDOL) created by intersecting laser beams with a superposition of polarizations. To
demonstrate the utility of the SDOL-based technique we compute the topological invariants (Chern
numbers) for the SDOL bands as a function of an external magnetic field, and show the existence of a
rich topology of the energy bands for this system which does not have parity-time-reversal symmetry.
We explicitly consider [6]Li F = 1/2 atoms. Using a projection matrix method we observe topological
phase transitions between an ordinary insulator, an abelian topological insulator, and a non-abelian
topological insulator as the external magnetic field strength is varied. Upon introducing edges for
the SDOL we find topological edge states (that are correlated with the band Chern numbers) that
simultaneously exhibit non-trivial density and spin currents with both a rotational flow contribution
and flow along the edge of the SDOL.


I. INTRODUCTION

A topological insulator is a material whose interior is
unable to conduct an electric current while its edges support such a flow [1–5]. A topological insulator (TI) differs from an ordinary insulator in that it is not possible
to continuously deform spin-orbit-induced topological insulators into an ordinary one without closing the bulk
gap, i.e., without undergoing a topological phase (TP)
transition. The presence of time-reversal symmetry is
crucial for that [6]. TIs have been theoretically studied [1, 3, 7] and experimentally realized in a variety of
systems, including HgTe/CdTe semiconductor quantum
wells [8], BiSb alloys [9], and Bi2Se3 crystals [10, 11].
Significant progress has been made in the realization
of band structures with non-trivial topology in ultracold
atomic gas experiments [4, 5]. The use of cold atoms in
the study of TPs of matter offers the advantage of wellcontrolled experimental parameters. Furthermore, recent
advances have also been made in the generation of fictitious magnetic fields and spin-orbit coupling for ultracold
neutral atoms in optical potentials (see Refs. [12–16]).
In this paper we propose another promising method for
the study of topological properties of matter that does
not possess parity-time-reversal symmetry. In contrast
to existing ultracold-atom methods, which are based on
periodically shaken lattices [17–19] or Raman coupling
of internal states [20, 21], our approach involves placing
atoms in a two-dimensional spin-dependent optical lattice potential and an external magnetic field. A simplified
study of fermionic and bosonic atoms in a spin-dependent
optical lattice (SDOL) in the limit of singly occupied sites
was recently carried out in Refs. [22, 23]. Here we use a
Bloch band model that goes beyond the two-band tightbinding approximation [24] to describe the SDOL potential, which allows us to determine the topology of the


high-energy bands, which exhibit multiple tangled bulk
band gaps [25]. We find a sequence of TP transitions
upon increasing the external magnetic field strength, involving abelian and non-abelian TPs. The rich topology
of the high-energy bands is also reflected in the characteristics of the edge states.
Below we consider a cold fermionic gas of [6]Li F = 1/2
atoms in a SDOL and an external magnetic field. We
show that the properties of the system are significantly
enriched relative to the system without external magnetic field, and that a radical change of the topological properties can be observed as the strength of the
external magnetic field is varied. We calculate the bandstructure and the Chern numbers (the topological invariants that classify bands in topological materials [1–5])
of the bands. Then, we apply blue-detuned lasers which
introduce edges to the SDOL, and we investigate their
topological character.
The paper is organized as follow: Section II introduces
the Hamiltonian of atoms trapped in a two-dimensional
spin-dependent optical lattice potential (SDOLP) in the
presence of an external magnetic field perpendicular to
the SDOLP plane. Section III calculates the Chern numbers, denoted by Cn of Bloch bands, which are parameterized by an integer band number n. In Sec. IV, energies and wave functions of edge states are calculated.
The numerical calculation of the energies and wave functions of trapped atoms is performed for a variety of external magnetic field strengths. Section V presents a
generalization of the concept of an eigenvector frame rotation in a non-abelian topological phase employing the
projection matrices. The transition from abelian to nonabelian is indicated by a discontinuity in the Frobenius
norm of the projection matrix. In Section VI, we analyze
the atom probability density, spin density, current density, and spin-current density in order to characterize the


-----

edge states. A summary of the results obtained is provided in Section VII. The appendices contain a number of
technical details. Appendix A addresses the symmetries
of the SDOLP Hamiltonian in both the absence and presence of an external magnetic field. The TP transitions
observed in the presence of an external magnetic field are
described in Appendix B. In Appendix C a detailed calculation of the Chern numbers is presented, while edge
state degeneracy is discussed in Appendix D.

II. SPIN-DEPENDENT OPTICAL LATTICE
POTENTIAL

The spin-dependent optical lattice potential is generated by two pairs of counter-propagating linearly polarized laser beams, which are tightly bound in the zdirection and form two-dimensional lattice. The polarization of the beams is varied such that the complex slowly
varying envelope E(r) of the electric field is given by

E(r, t) = [1] (1)

2 [(][E][(][r][)][e][−][iω][l][t][ + c][.][c][.][)][,]

�4 � �

E(r) = √[E][0] zˆ + k[ˆ]n × ˆz e[i][k][n][·][r], (2)

2 n=1


Here F is the atomic hyperfine angular momentum, µB
is the Bohr magneton, and for a ground state alkali
atom with L = 0, J = S = 1/2, I = 1 nuclear spin,
gF = gS F (F +1)−2FI( (IF+1)+ +1) S(S+1), and gS is the electron

spin g-factor. The SDOLP is defined as follows V (r) =
− [α][s][(]4[ω][l][)] E[∗](r)·E(r) and Bfic(r) = 2iF 4αgvF( µωlB) [E][∗][(][r][)][×][E][(][r][),]

where αs and αv are scalar and vector polarizabilities
which depend upon the detunings of the laser frequency
ωl from the D1 and D2 resonance lines of Li. Explicitly,

� � � � ��

V (r) = − [V][0] 2 + cos q0x + cos q0y, (6)

2

� � �
Bfic(r) = −B0xˆ sin q0x cos[2][ �] [q][0][y]

2

� � �
−B0yˆ sin q0y cos[2][ �] [q][0][x] . (7)

2


where r = (x, y), zˆ is the polarization unit vector along
the z-axis, the wavevectors are

� � � � � �
(2n + 1)π (2n + 1)π

kn = kl cos, sin, 0, (3)

4 4

kl = 2π/λl is the laser wavenumber, λl is the laser wavelength, and k[ˆ]n is the unit vector in the direction of kn.
The total atomic Hamiltonian in the SDOLP can be
written as

H = (4)
− [ℏ][2]

2M

[∇][2][ +][ H][Stark][(][r][) +][ H][Z][.]

The first term on the right hand side of Eq. (4) is the
kinetic energy operator of an atom, where M is the
atomic mass. The last term on the right hand side of
Eq. (4), HZ = −gF µB BextFz, is the Zeeman interaction Hamiltonian of the atom with an external magnetic
field Bext = Bextzˆ. The middle term, HStark(r), is the
optical lattice Stark interaction Hamiltonian. Some symmetries of the SDOLP Hamiltonian in both the absence
and presence of an external magnetic field are discussed
in Appendix A.
For 6Li, the total electronic angular momentum is
J = 1/2, hence, the effective interaction of an atom
with the electromagnetic field can be described using a
scalar potential V and vector potential containing what
Cohen-Tannoudji and Dupont-Roc termed the fictitious
magnetic field Bfic [22, 26–28], and the tensor potential
vanish. The Stark Hamiltonian in the presence of a fictitious magnetic field is given by

HStark(r) = V (r) − gF µB Bfict(r) · F . (5)


√
Here V0 = αs(ωl)E0[2][,] [B][0] [=] 2F gF2 µB [α][v][(][ω][l][)][E]0[2] [and] [q][0] [=]

√

2 kl. Note that the divergence of the fictitious magnetic
field does not vanish; Bfic(r) corresponds to a radially
distributed magnetic monopole density. The Hamiltonian (4) has square lattice symmetry with lattice vectors
a1 = (a0, 0) and a2 = (0, a0), where the lattice period is
a0 = 2π/q0, hence the reciprocal lattice has square symmetry with reciprocal lattice vectors q1 = (q0, 0) and
q2 = (0, q0).
A [6]Li atom in the ground state has F = 1/2, and
the scalar potential V (r) and the fictitious magnetic field
Bfic(r) in the vector potential are asigned as previously
described, with polarizabilities αs(ωl) = α[s]nJF [(][ω][l][)] [and]
αv(ωl) = α[v]nJF [(][ω][l][)] [defined] [by] [[26]]

1
α[s]nJF [(][ω][l][)] [=] � α[(0)]nJ [(][ω][l][)][,] (8)
3(2J + 1)

�
� �J+I+F 2F (2F + 1)
α[v]nJF [(][ω][l][)] [=] − 1

F + 1

� �
F 1 F
× J I J α[(1)]nJ [(][ω][l][)][,] (9)


� �
F 1 F
where are the Wigner 6-j symbols. The reJ I J

duced scalar and vector polarizabilities are


� �K+J+J [′]+1
1
−


α[(]nJ[K][)][(][ω][l][)] [=] ℏ[1]


�

√

2K + 1


n[′]J [′]

�
1 K 1 ��� n′J ′ d nJ ��2
× J J [′] J ⟨ ∥ ∥ ⟩

�
1
Re
× ωn′J ′nJ − ωl − 2[i] [γ][n][′][J] [′][nJ]

�

( 1)[K]
−
+, (10)

ωn′J ′nJ + ωl + 2[i] [γ][n][′][J] [′][nJ]


where n[′]J [′] d nJ is reduced matrix element of the elec⟨ ∥ ∥ ⟩
tric dipole transition nJ → n[′]J [′], ωn[′]J [′]nJ is the transition
frequency, and γn′J ′nJ is the transition line-width.


-----

(a)


(b)

-0.4 -0.2 0.0 0.2 0.4


The ground electronic state of the lithium atom is the
2 [2]S1/2 state, and the excited electronic states are the
2 [2]P1/2 state and the 2 [2]P3/2 state. The D1 transition,
2 [2]S1/2 → 2 [2]P1/2, has the transition frequency ωD1 ≡
ω2 12 [2][ 1]2 [=] [2][π] [×] [446][.][789634] [THz] [[29],] [the] [linewidth] [is]

γD1 ≡ γ2 12 [2][ 1]2 [= 2][π][×][5][.][8724 MHz [30] and the reduced ma-]

trix element dD1 ≡⟨2 2[1] [∥][d][∥][2][ 1]2 [⟩] [=][ −][8][.][433][×][10][−][18][ esu][×][cm.]

The D2 transition, 2 [2]S1/2 → 2 [2]P3/2, has the transition
frequency ωD2 ≡ ω2 32 [2][ 1]2 [=] [2][π] [×] [446][.][799677] [THz] [[29],]

linewidth γD2 ≡ γ2 32 [2][ 1]2 [=] [2][π] [×] [5][.][8724] [MHz] [[30]] [and]

reduced matrix element dD2 ≡⟨2 2[3] [∥][d][∥][2][ 1]2 [⟩] [=] [11][.][925][ ×]

10[−][18] esu cm.
×

2.5


-4.50

-4.55

-4.60

-4.65

-4.70


-4.75
-0.4 -0.2 0.0 0.2 0.4

_kx/q0_


-4.50
-4.55
-4.60
-4.65
-4.70
-4.75


_kx/q0_


-4.5

-4.6

-4.7

-4.8

-4.9 (c)
-0.4 -0.2 0.0 0.2 0.4


-4.4
-4.5
-4.6
-4.7
-4.8
-4.9

(d)

-5.0
-0.4 -0.2 0.0 0.2 0.4

_kx/q0_

|(d)|Col2|
|---|---|


_kx/q0_


2.0

1.5


1.0

0.5


0 2 4 6 8 10


ω-ωD1 (2π×GHz)

FIG. 1. The electromagnetic energy flux per unit area S0
versus detuning from the D1 line. The curves of equal V0
(blue): V0 = 2.5 E0 (dashed), V0 = 5 E0 (solid) and V0 = 7.5 E0
(dot-dashed), where E0 is the recoil energy. The curves of
equal B0 (red): B0 = 5 B0 (dot-dashed), B0 = 10 B0 (solid),
and B0 = 15 B0 (dashed), where B0 = E0/(gF µB). The green
dot is a crossing point of the blue and red solid curves.


Note that V0 and B0, which are proportional to the
scalar polarizability αs(ωl) and the vector polarizability
αv(ωl) respectively, depend on the laser frequency ωl and
the electromagnetic energy flux per unit area S0 = 4cπ [E]0[2]
(in Gaussian units). Figure 1 shows the curves of the
constant V0 and the curves of the constant B0, where the
x-axis is the detuning of the laser frequency ω from the
resonance frequency ωD1 of the D1 transition, and the
y-axis is the electromanetic energy flux per unit area S0.
The green dot is the point ω = ωD2 − 2π × 5.11678 GHz
and S0 = 1.73328 × 10[7] erg s[−][1] cm[−][2], where V0 = 5 E0
and B0 = 10 B0, where E0 = 2ℏMc[2]ωl[2][2] [is] [the] [recoil] [energy,]
B0 = E0/(gF µB). Thus, the laser frequency ωl which is
used is red-detuned from the Li D2 line by ∆= −2π ×
5.117 GHz.
The quantum states of the atoms in the SDOLP are
parametrized by wavevector k = (kx, ky) belonging to the
first Brillouin zone (BZ) of the SDOLP: |kx, ky| < q0/2,
and the energy band number n, where n is a positive
integer. The corresponding energies ǫn,k and wave functions ψn,k(r) are determined from the Sch¨odinger equation Hψn,k(r) = ǫn,kψn,k(r). The Sch¨odinger equation


FIG. 2. Band energies ǫn,kx versus kx numbered by n = 5
(orange), n = 6 (red), n = 7 (gold), n = 8 (magenta), for
V0 = 5 E0, B0 = 10 B0, and four values of the external magnetic field Bext: (a) Bext = 0.05 B0, (b) Bext = 0.2 B0, (c)
Bext = 0.7 B0, and (d) Bext = 1.0 B0. Also shown are the edgestate energies (purple) that result when the blue-detuned potential
VBD(y)(y) = VBD,0Θ(|y|−Ly/2) is introduced where VBD,0 is taken
to be infinite and Ly = 25a0. To experimentally probe high-lying
bands and edge states, one would fill the Fermi sea to a desired
energy.

was solved numerically using the Mathematica command
NDEigensystem to find radial wave functions ψn,k(r) and
eigenenergies ǫn,k. Figure 2 shows the bands calculated
for specific values of V0, B0, and Bext as specified in the
figure caption (energy and magnetic field are in units
of E0 and B0, respectively). The figure also shows edge
states in a finite width strip produced by blue-detuned
lasers, see below.
The scalar potential V (r) has minima at r = a
≡
n1a1 + n2a2 where n1 and n2 are integers. Bfic(r) vanishes at r = a, and at the edges of the Wigner-Seitz cells,
r = a + 2[1] [a][1] [and] [r] [=] [a][ +] [1]2 [a][2][.] [Hence] [the] [minimum] [of]

V (r) at a [where V (a) = −2V0] and the nearest-neighbor
minima at a + aj (j = 1, 2) are separated by a barrier
of height V0. Hence, the bands with ǫn,k < −V0 can
be well described using a tight-binding model. However,
for the bands which are above the barriers, ǫn,k - −V0,
tight-binding is a poor approximation. All bands shown
in Fig. 2 are above −V0, hence tight-binding cannot be
used for them. Note that for [6]Li, singlet s-wave scattering length is small [31] and a single particle picture can
be used.


III. CHERN NUMBERS

Chern numbers Cn for the Li atoms in the SDOLP
are defined on the bands in wavevector space [1, 4]. For
a 2D periodic system, Cn for the nth band is determined by integrating the Berry curvature, Fxy(n, k) =
∂kxAy(n, k) − ∂ky Ax(n, k), over the first BZ,

� � �

Cn = [1] dk ∂Ay(n, k) − [∂A][x][(][n,][ k][)], (11)

2π BZ ∂kx ∂ky


-----

where Aµ(n, k) = −i⟨n, k| ∂k∂µ [the] [Berry] [con-]

[|][n,][ k][⟩] [is]
nection. In our analysis we used an efficient numerical
method proposed in Ref. [32]. The details are presented
in Appendix C.
Increasing the external magnetic field strength we find
a sequence of TP transitions. Phase transitions occur
when bandgaps between Bloch bands close and open.
The energy gap can vanish at several points in the
BZ: the Γ point, the vertices of the BZ, and the edge
centers of the BZ. Some of the gapless spectra have
Dirac cones and some have quadratic wavevector dependence. The summary of all TP transitions found
for external magnetic fields Bext < 2 B0 is given in
Appendix B. The lowest critical points at which the
TP transitions occur are {Bc,1, Bc,2, Bc,3, Bc,4, Bc,5} =
{0.11, 0.66, 0.89, 1.25, 1.35}B0. All the Chern numbers
vanish below Bc,1. At the lowest TP transition the gap
between the 6th and 7th bands closes at the center of the
BZ and forms a Dirac cone, see Fig. 9 in Appendix B.
For Bc,1 < Bext < Bc,2, C6 = −1 and C7 = 1. Indeed,
in this case the bulk-boundary correspondence leads to
the appearance of two topologically protected chiral edge
states in the bandgap, see discussion below. Above Bc,2
the second TP transition appears and the gap between
7th and 8th energy bands closes, again at k = 0, and C7
changes from +1 to −1, and C8 changes from 0 to +2.
Note that the sum of all the Chern numbers in the eight
lowest energy bands is always equal to zero.
Since the Chern numbers Cn are non-zero, this system
is a TI. For abelian TIs, the Chern numbers are such that
Cn = −Cn+1, and the number of chiral edge states on
each edge between bands n and n + 1 is |Cn| [1, 3]. For
non-abelian TIs the bulk-edge correspondence is more
subtle than the abelian case [33]; multiple tangled bulk
bandgaps are present and the system supports non-trivial
edge states.

IV. EDGE STATES

In order to study edge states for the atoms in the
SDOLP we introduce a blue-detuned potential of the
form VBD(y) = VBD,0Θ(|y| − Ly/2) where Θ(•) is the
Heaviside Theta function. This potential mimics the effects of a blue-detuned laser that repels atoms from the
region |y| - Ly/2. For convenience we take VBD,0 very
large so we can apply Dirichlet boundary conditions at
the edge. To compute the edge states we again use the
Mathematica command NDEigensystem with Dirichlet
boundary conditions at y = ±Ly/2 and periodic boundary conditions in x.
Figure 2 shows the edge states of the finite-width strip,
|y| ≤ Ly/2, as purple curves that lie within the bandgaps
(and within the bands) of the fully periodic system. Not
all the edge states shown in purple in Fig. 2 are topological edge states (TESs). Clearly those that connect adjacent bands and lie in the gap between them are TESs.
The edge states in Fig. 2(a) do not link different bands


and are not topological. Furthermore, all the Chern numbers for the bands are zero for Bext < Bc,1. The lowest
energy edge states in Fig. 2(b) connect bands 5 and 6
but there is no gap between the 5th and 6th bands. The
projection of the edge states onto the bulk states of the
same energy differs from zero allowing transitions to the
system’s interior, which would result in a damage of the
edge current. Hence these edge states might not be TESs.
In contrast, the upper set of edge states in Fig. 2(b) that
connect bands 6 and 7 are topological (the Chern number C6 = −1 and C7 = 1 and the other Chern numbers
are zero). The edge states that connect the 6th and 7th
bands have regions in kx that lie within the bandgap. The
bulk-boundary correspondence leads to the emergence of
one topologically protected chiral edge state within the
bandgap on every edge of the TP material. There are
edge states between the 5th and 6th bands, 6th and 7th,
and 7th and 8th bands in Fig. 2(c). The Chern numbers
C6 = C7 = −1 and C8 = 2. There is one pair of TESs
between the 6th and 7th bands, and two pairs between
the 7th and 8th bands. The latter two pairs are situated
between the 7th and 8th bands near the kx = 0 point, but
are not seen well in Fig. 2(c), but are seen in Fig. 2(d).
The former pair connecting the 6th and 7th bands is located in the interval 0.3q0 < |kx| < 0.4q0. In Fig. 2(d)
there are edge states between all the bands. The Chern
numbers are C6 = 1, C7 = −3 and C8 = 2, indicating
that the edge states between the 6th and 7th bands and
the 7th and 8th bands are topological. A crossing of one
pair of edge states between the 6th and 7th bands occurs
at kx = 0. It seems plausible that another pair, situated
in close proximity to kx = ±0.4q0, represents a continuation of the edge states observed between the 5th and 6th
bands. The edge states between the 7th and 8th bands
contain two pairs.

V. NON-ABELIAN TOPOLOGICAL PHASE

In both Fig. 2(c) and (d), the coefficients C6, C7 and
C8 are non-vanishing and Cn ≠ −Cn+1. Therefore the
TP of the system might be non-abelian. Here, we generalize the concept of an eigenvector frame rotation, as
previously defined in reference [33–35], using the projection matrices, and show that indeed the appearance of
such a sequence of bands implies non-abelian TP.
We follow the line of argument presented in Refs. [33–
35], wherein the non-abelian topological properties are
investigated in momentum space, with a focus on eigenvectors and frame rotations. This concept has been
demonstrated to be a valuable tool for the description of
one- and two-dimensional non-abelian topological insulator, as evidenced by the findings presented in Ref. [33]
where a specific model of a TI is considered. The system exhibits space-time inversion symmetry, which allows the Hamiltonian to be gauge transformed in such a
way that it is real in the momentum space and possesses
real eigenstates. If it is assumed that three bands are


-----

involved, the eigenvectors can be mapped onto a threedimensional space. This allows for an examination of the
system’s properties by considering the rotational characteristics of the band states as they traverse the Brillouin
zone. The TI studied in Ref. [33] exhibits non-abelian
band topology, despite the bands being flat. This is due
to the non-abelian quaternion group governing the rotational properties of its bands. It is evident that at
least three bands must be considered in order to identify the non-abelian properties, given that rotations in
two-dimensional space are abelian.
However, our spin-dependent optical lattice Hamiltonian is not PT symmetric, which results in the eigenvectors being complex. It thus follows that the concept
of eigenvector frame rotations must be generalized. We
therefore introduce a projection matrix, whose complex
elements are defined as the inner product between eigenstates of the system, Un,n′ (k) = ⟨un,k0|un′,k⟩, where
un,k(r), according to the Bloch’s theorem, is periodic
function with period equal to the lattice constant and
the integration is over a Wigner-Seitz cell. k0 is an arbitrarily chosen reference momentum and the subscript
n specifies the band. Note that the eigenstates |un,k0⟩
are mutually orthogonal. In our case the reference momentum k0 is taken to be the Γ point. For an external
magnetic field Bext such that Bext - Bc,2, the sixth, seventh, and eighth bands possess non-zero Chern numbers.
The appropriate projection matrix Un,n′ (k) is therefore
a 3 3 matrix with n, n[′] 6, 7, 8 and with each row
× ∈{ }
normalized to 1. The properties of the projection matrix
(k) play a pivotal role in determining the character of
U
the TP.
To demonstrate how this works we compare the structure of the projection matrix for the TPs for Bc,1 <
Bext < Bc,2 and Bc,2 < Bext < Bc,3. The primary difference between these two regions is that in the first region
the projection matrices have the form of 2 2 and 1 1
× ×
blocks, while in the second they form single 3 3 block
×
matrices. To illustrate this difference we use the Frobenius norm defined as follows: ||U||F[2] [=] [�]n,n[′][ |U][n,n][′][|][2][.]
Note that, the Frobenius norm (or part thereof) is gauge
invariant.
Figure 3 depicts the ratio R of part of the Frobenius norm where we include only the following elements:
(n, n[′]) = (6, 8), (7, 8), (8, 6), and (8, 7) to the total Frobenius norm. In the top frame this norm is small, below 0.2,
which is indicative of the emergence of a two-block structure for Bext < Bc,2, typical for the topological abelian
phase. In contrast, the norm remains large in the bottom frame, indicating that the projection matrix is a full
3 3 matrix. In the trivial TP, where the external mag×
netic field Bext < Bc,1 the projection matrix possess a
three 1 1 block structure, which is numerically nearly
×
diagonal.
The observation that in the abelian TP the projection matrix has 2 2 and 1 1 block structure implies
U × ×
that 6th and 7th bands decouple from the 8th band.
Therefore, the 6th and 7th band structure may be con

FIG. 3. Ratio R of the part of the Frobenius norm of the
projection matrix to the total Frobenius norm, as indicated
in the text. It is plotted as a function of momentum for the
following values of the external magnetic field: Bc,1 < Bext =
0.5 < Bc,2 (top frame) and Bc,2 < Bext = 0.7 < Bc,3 (bottom
frame). For Bext < Bc,2 value of R is relatively small and
the 6th and 7th bands separate from the 8th band, while for
Bext - Bc,2 R is of order of one and all three bands become
entangled and their topology is non-abelian.

sidered separately. The eigenstates |u6,k0⟩, |u7,k0⟩, and
|u8,k0⟩ are orthonormal and can be represented in a threedimensional space by mutually orthogonal unit basis vectors β[ˆ], γˆ, and δ[ˆ]. As the Brillouin zone is traversed,
|Un,n′(k)| = |⟨un,k0|un′,k⟩|, which are gauge independent,
can be monitored. The block structure of the projection
matrix in the abelian TP (see Fig. 3) indicates that the
states |u6,k⟩ and |u7,k⟩ are primarily confined to the twodimensional subspace spanned by the vectors β[ˆ] and γˆ
Even though the block diagonal structure is not ideal, we
can follow a procedure common in quantum information

[36, 37], where the projection of the unitary evolution in
the Hilbert space onto the small subspace is performed
and then forced to be again unitary. This can be achieved
with the aid of a technique based on the singular value


-----

decomposition, which allows for the identification of the
closest unitary matrix to a given matrix.

To get insight into the character of state transformations let us consider moving from a point k1 in the BZ
to the point k2. The transformation from the states
|u6,k1⟩ and |u7,k1⟩ to |u6,k2⟩ and |u7,k2⟩ is realized using the product operator U(k2) U [†](k1), which is the
2×2 unitary matrix U n,n[′](k1, k2) = ⟨un,k1|un[′],k2⟩ with
n, n[′] 6, 7 . In order to trace the evolution of the
∈{ }
band states we consider the gauge-independent quantities |U n,n′ (k1, k2)| = |⟨un,k1|un′,k2⟩|. However, the 2×2
unitary matrix after its elements are replaced by their absolute values becomes a real matrix with equal diagonal
elements and equal off-diagonal elements. Such matrices
form an abelian group, a subgroup of GL2(R), thereby
justifying the designation of the TP.

In contrast, for an external magnetic field strength exceeding the critical field strength for the second phase
transition, Bext - Bc,2 (Fig. 3, bottom frame), all three
bands remain coupled and the band states undergo a full
three-dimensional transformation as the system traverses
the Brillouin zone. Consequently, for Bc,2 < Bext < Bc,3
the TP is non-abelian, since 6th, 7th, and 8th bands
transform according to the non-abelian GL3(R) group.
The transition from abelian to non-abelian TP is sharp,
as demonstrated in Fig. 4, and occurs at the value of the
external magnetic field equal to Bc,2 (see Fig. 10).

Similarly, for Bc,3 < Bext < Bc,4 the projection matrix exhibits a single 3 3 block structure, thereby demon×
strating that the phase is also a non-abelian TP. It should
be noted that in the Bc,4 < Bext < Bc,5 region, the phase
reverts to abelian. This is due to the fact that the sum
of components of the Frobenius norm corresponding to
the off-diagonal elements: (6, 7), (6, 8), (7, 6), and (8, 6)
become small with magnitude below 0.1. This results in
the projection matrix assuming the form of a 1 1 by 2 2
× ×
block. As a result, the states in the seventh and eighth
bands become decoupled from the sixth band.

In the final phase, Bext  - Bc,5, the Chern numbers
are non-zero for the bands 5th, 6th, 7th, and 8th (see
Appendix B). However, the sum of the Chern numbers for
any two successive bands is zero. The projection matrix,
which is now a 4 4 matrix, has the form of 2 2 and
× ×
2 2 blocks (with the part of the Frobenius norm located
×
off the blocks being smaller than 0.1). This indicates
that the phase is again an abelian phase. Even though
four successive bands have nonzero Chern numbers, the
system remains in the abelian four-band TP.

Therefore, the Chern numbers themselves cannot be
used to distinguish between abelian and non-abelian TPs.
The transition from abelian to non-abelian is indicated
rather by a discontinuity of the Frobenius norm of the
projection matrix as shown in Fig. 4.


FIG. 4. Ratio R, as in Fig. 3, of the projection matrix, as a
function of momentum along the diagonal in the BZ and an
external magnetic field Bc,1 < Bext < Bc,3. Appearance (or
disappearance) of a block diagonal structure has a character of
a phase transition. It occurs at Bext = Bc,2 in correspondence
with the TP transition observed at Bc,2.

VI. EDGE STATE DENSITY, SPIN DENSITY,
CURRENT DENSITY, AND SPIN-CURRENT
DENSITY

The density of a representative edge state with negative kx is shown in Fig. 5(a). The density is restricted to
the region near the y = Ly/2 edge, and the atoms have
negative group velocity vg(kx) = ℏ[−][1]dǫ/dkx. The nodes
in the edge state density are due to the excited state nature of the edge state. Edge states near y = −Ly/2 (not
shown) have positive kx and positive vg(kx).
Figure 5(b) shows the edge state spin density S(r) =
ψk[†]x[(][r][)][F][ψ][k][x] [(][r][)] [(which] [depends] [on] [k][x][),] [the] [arrows] [show]
the 2-dimensional vector Sxy(r) ≡ (Sx�(r), Sy(r)), and

the color of the arrows show the length Sx[2](r) + Sy[2](r).

The density plot in Fig. 5(b) shows the z-component of
the spin density vector, Sz(r) which is negative everywhere except where the wave function has nodes (vortices). The texture of the 2D spin vector Sxy(r) is due
to the r-dependence of Bfic(r): Sx(r) is an odd function
of x since Bx,fic(r) is odd, and Sy(r) is an even function
of x, since By,fic(r) is even.
Figure 6 shows the atomic current density Jkx (r) =
ℏ
M [Im [][ψ]k[†]x [(][r][)][ ∇][ψ][k][x] [(][r][)],] [which] [can] [be] [separated] [into] [two]
parts: Jkx(r) = Jk[av]x,x[(][y][) ˆ][x][ +] [J][vor]kx [(][r][).] Here Jk[av]x,x[(][y][)] [=]

a10 �−a0a/02/2 [J][k][x][,x][(][r][)][ dx] [is] [an] [‘average’] [current] [propagating]
along the edge of the SDOL, and J[vor]kx [(][r][) which] [describes]
a vortex current flow, i.e., a rotational part of the flow.
The net atomic current along the edge can be calculated
as Jk[tot]x,x [≡] �−LLy/y2/2 [J]k[av]x,x[(][y][)][dy][; the total current is negative]
and depends on kx and Jk[tot]x,y [vanishes] [for] [all] [k][x][.]
The atomic spin-current density can be denoted by
ℏ
Jkx,α,α′ (r) = M [Im [][ψ]k[†]x [(][r][)][ F][α][∂][α][′] [ψ][k][x] [(][r][)].] [The] [subscript]
α = x, y, z, indicates the spin polarization, and the subscript α[′] = x, y, specifies the current propagation direc

-----

0.4

0.2

0.0


-0.2

0

-0.4

10.5 11.0 11.5 12.0 12.5

|kx,x (<br>6<br>4<br>2<br>0|a)|
|---|---|


_y/a0_


0.4

0.2

0.0


-0.2

0

-0.4

10.5 11.0 11.5 12.0 12.5

|kx,y (<br>6<br>4<br>2<br>0|b)|
|---|---|


_y/a0_


FIG. 5. (a) Edge state atom density and (b) spin density
for Bext = 0.2 E0/(gF µB) and kx = −0.2 q0 and Ly = 25a0

[see edge state connecting the 6th and 7th bands in Fig. 2(b)].
The colors of the arrows are shown in the Sxy legend, and the
background color which gives the Sz component of the spin
density is shown in the left legend.

tion. One can decompose the tensor Jkx,α,α[′](r) into three
2D vectors J kx,α(r) = (Jkx,α,x(r), Jkx,α,y(r)) which describe spin-polarized currents and are shown in Fig. 7.

|kx,z (<br>5<br>4<br>3<br>2<br>1<br>0|c)|
|---|---|


_y/a0_

FIG. 7. Spin-current density for the parameter values used
in Fig. 5. (a) J kx,x(r), (b) J kx,y(r), and (c) J kx,z(r). The
color of the arrows show Jkx,α(r) = |J kx,α(r)|, where α =
x, y, z.


0.4

0.2

0.0


-0.2 1

0

-0.4

10.5 11.0 11.5 12.0 12.5


0.4

0.2

0.0

-0.2

-0.4

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|_Jkx_<br>0.<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5|_Jkx_<br>0.<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5|_Jkx_<br>0.<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5|_Jkx_<br>0.<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5|_Jkx_<br>0.<br>2.5<br>5.0<br>7.5<br>10.0<br>12.5|
||||||


10.5 11.0 11.5 12.0 12.5


_y/a0_

FIG. 6. Atom current density Jkx(r) for the parameter values
used in Fig. 5. Note that the current density flows clockwise,
i.e., it has vorticity.


Both J kx,y(r) (in Fig. 7(b)) and J kx,z(r) (in Fig. 7(c))
spin-current densities have ‘average’ currents along the
edge of the SDOL, and ‘vorticity’ currents due to the
rotational character of the edge states. The nonvanishing total spin-currents are given by Jk[tot]x,y,x [=]
��−−LLLLyy/y/y22//22 [J][J]kk[ av][ av]xx,y,x,z,x[(][(][y][y][)][)][dy][dy] [=][=][ −][−][0][.][0][236][.][526][ ℏ][/][ ℏ][(][/][Ma][(][Ma][2]0[).]0[2][)] [and] [J]k[ tot]x,z,x [≡]
J kx,x(r) in Fig. 7(a) exhibits rotational motion and
the net spin-current in the x-direction vanishes. It is
unexpected that a non-vanishing spin-current exists in
the y-direction since the atomic current vanishes in the
y-direction. To explain why this happens, let us note
that the y-component of the current in Fig. 6 is an odd


function of x, and thus the net atomic current vanishes in
the y-direction. But the y-component of the spin-current
density in Fig. 7(a) is an even function of x, and the net
spin-current density in the y-direction does not vanish
because both the y-component of the atomic current and
the x-component of the spin density in Fig. 5 are odd
functions of x, and their product is even.

VII. SUMMARY AND CONCLUSION


We studied the topological properties of cold fermionic
Li atoms in a 2D SDOLP in the presence of an external
magnetic field perpendicular to the lattice. Both scalar
and vector (fictitious magnetic field) potentials are built
into the SDOLP, and both magnetic fields break timereversal symmetry (see Appendix A). Topological phases,
protected by spatial symmetries, appear upon increasing
the external magnetic field. The calculated Chern numbers for the lowest energy bands (n 8) are all zero for
≤
Bext < Bc,1. For Bext - Bc,1, the Chern numbers take a
series of nonzero values as Bext increases, revealing both
abelian and non-abelian topological states and topolog

-----

ical phase transitions. For a finite width lattice, we observe edge states in the energy gaps between successive
bands, some of which are topologically protected. Thus,
the atoms in the SDOLP behave as a topological insulator. The atom current density and spin-current density
of the TESs have vorticity and average flow along the
edge. We believe that using the SDOL technique opens
new possibilities for studying a wide range of topological
phenomena in ultracold atomic systems.

ACKNOWLEDGMENTS

MB was supported by the NCN Grant
No. 2019/32/Z/ST2/00016 through the project MAQS
under QuantERA funded by the European Union’s
Horizon 2020 research and innovation program under
grant No. 731473. Some numerical results were obtained
using Center of University of Bia�lystok computers.

Appendix A: SDOLP Hamiltonian symmetries

Here we discuss some symmetries of the Hamiltonian
in Eq. (4). For the sake of completeness we first discuss
the case without the external magnetic field, then we
introduce the symmetries for the case with Bext = 0.̸

1. Symmetries for Bext = 0

Consider first symmetries of the system in the absence
of the external magnetic field, that is, Bext = 0. In this
case, the system is not a time-reversal invariant, since

T HStark(x, y)T [−][1] = HStark(−x, −y) ̸= HStark(x, y),
(A1)
where T = e[iπF][y] K, and K is the operator for complex
conjugation. The system is not a P invariant, since

PHStark(x, y)P [−][1] = HStark(−x, −y) ̸= HStark(x, y).
(A2)
But HStark(x, y) is a PT invariant,

PT HStark(x, y)(PT )[−][1] = HStark(x, y). (A3)

Moreover, HStark(x, y) is invariant under twofold rotation
around the z axis,

C2,zHStark(x, y)C2[−],z[1] [=][ H][Stark][(][x][′][, y][′][)][,] (A4)

where C2,z = e[iπF][z] and (x[′], y[′]) = (−x, −y). The product
of the time reversal and twofold rotation gives

(C2,zT )HStark(x, y)(C2,zT )[−][1] = HStark(x, y), (A5)

hence the Hamiltonian is not symmetric under the product operator C2,zT . However, there is fourfold rotational
symmetry C4,z = e[iπF][z] [/][2] of the optical lattice,

C4,zHStark(x, y)C4[−],z[1] [=][ H][Stark][(][x][′][, y][′][)][,] (A6)


where (x[′], y[′]) = (y, x) are the rotated coordinates. Sim−
ilarly, there are also two-fold rotational symmetries about
the x-axis and y-axis,

C2,xHStark(x, y)C2[−],x[1] [=] [H][Stark][(][x,][ −][y][)][,] (A7)

C2,yHStark(x, y)C2[−],y[1] [=] [H][Stark][(][−][x, y][)][,] (A8)

generated by C2,x = e[iπF][x] and C2,y = e[iπF][y] .

2. Symmetries for Bext ̸= 0

Now let us consider the system in the presence of
a finite external magnetic field applied in the z-axis,
Bext = Bexteˆz. The Zeeman interaction Hamiltonian
HZ is not time-reversal invariant, T HZT [−][1] ≠ HZ, it
is P invariant, PHZP [−][1] = HZ, therefore it is not PT
invariant, (PT )HZ(PT )[−][1] ≠ HZ . Both the Stark interaction Hamiltonian HStark(x, y) and the Zeeman interaction Hamiltonian HZ are symmetric with respect
to the transformations C2,z and C4,z but is not invariant under C2,x and C2,y (since both of them flip
Bext). Therefore, only two- and fourfold rotation around
the z-axis remain valid symmetries of the Hamiltonian
HStark,Z(r) = HStark(r) + HZ, i.e.

C2,zHStark,Z(x, y)C2[−],z[1] [=][ H][Stark][,][Z][(][−][x,][ −][y][) =][ H][Stark][,][Z][(][x][′][, y][′][)]
(A9)
with (x[′], y[′]) = ( x, y), and
− −

C4,zHStark,Z(x, y)C4[−],z[1] [=][ H][Stark][,][Z][(][y,][ −][x][) =][ H][Stark][,][Z][(][x][′][, y][′][)]
(A10)
with (x[′], y[′]) = (y, x).
−

Appendix B: Chern numbers and topological phase
transitions

In the main text we classify the TPs of a twodimensional SDOLP. The SDOLP TP in the presence
of a finite external magnetic field that is perpendicular
to the plane of the SDOLP breaks time-reversal symmetry. The time-reversal symmetry is broken by both
fictitious magnetic field and the external magnetic field.
Therefore different TPs can be observed with increasing
the strength of the external magnetic field Bext. Figure 8
shows the Chern numbers of 5th, 6th, 7th, and 8th bands
verus external magnetic field strength for V0 = 5 E0 and
B0 = 10 E0/(gF µB). Bext = 1.0 E0/(gF µB) corresponds
to the external magnetic field equal to 78.8 mG. For lower
bands all the Chern numbers equal zero.
The first TP transition occurs at the external magnetic field Bext = Bc,1. The 6th and 7th Bloch bands at
this external magnetic field touch each other at momentum k = 0. Near the touching point the dispersion of
the states is linear, hence the dispersion forms a Dirac
cone, see Fig. 9. For Bext - Bc,1 the Chern numbers of
6th and 7th Bloch bands change from C6 = C7 = 0 to


-----

2

1

0

-1

-2

-3
0.0 0.5 1.0 1.5


_C5_

_C6_

_C7_

_C8_


_Be_ �� [[] 0[/(][g]F _B[)]]_

FIG. 8. The Chern numbers of 5th, 6th, 7th, and 8th
bands as a function of external magnetic field. The critical points at which the some of the Chern numbers at which
the TP transitions occur are {Bc,1, Bc,2, Bc,3, Bc,4, Bc,5} =
{0.11, 0.66, 0.89, 1.25, 1.35} E0/(gF µB).

FIG. 9. The 6th and 7th Bloch bands at the external magnetic
field Bext = Bc,1. The bands form a Dirac cone at k = 0 so
the gap between bands is closed at this value of Bext.

C6 = −1 and C7 = 1. The Chern numbers of all other
bands remain zero. For Bc,1 < Bext < Bc,2 the atoms in
a spin-dependent optical lattice potential remain in the
phase of an abelian TP. At Bext = Bc,2 the second TP
transition occurs (see Fig. 8). Again, the gap is closed at
the Γ point in the Brillouin zone (Fig. 10). However, this
time 7th and 8th Bloch bands touch each other and the
dispersion does not form a Dirac cone. After the second
phase transition the Chern numbers of involved bands
become C7 = −1 and C8 = 2 an the system of atoms
in a spin-dependent optical lattice potential becomes essentially (since C6 = −1) a multi-band TP. The system

FIG. 10. The 7th and 8th Bloch bands are closed at the
external magnetic field Bext = Bc,2 at k = 0 but not in the
form of a Dirac cone.


enters the non-abelian TP [33].

FIG. 11. The 6th and 7th Bloch bands are closed at the
external magnetic field Bext = Bc,3 at the centers of the edges
of the Brillouin zone.

FIG. 12. The 6th and 7th Bloch bands are closed at the
external magnetic field Bext = Bc,4 at the corners of the edges
of the Brillouin zone.

FIG. 13. The 5th and 6th Bloch bands are closed at the external magnetic field Bext = Bc,5 at the corners of the Brillouin
zone.

Two successive TP transitions involve the 6th and 7th
bands and the inter-band gap is closed at four points in
the BZ. At Bext = Bc,3 the bands touch at the centers
of the edges of the BZ and the Chern numbers change
to C6 = 1 and C7 = −3 (Figs. 11 and 8). The system
still remains in the non-abelian TP. However, after the
fourth TP transition at Bext = Bc,4 (here the gap is
closed at the corners of the BZ, see Fig. 12) the atoms in
a spin-dependent optical lattice potential enter back the
abelian TP, the only non-zero Chern numbers are those
for bands 7th and 8th: C7 = −2 and C8 = 2. The last TP
transition we studied occurs at Bext = Bc,5 and the band
gap between 5th and 6th bands is closed at the corners
of the BZ (see Fig. 13). After the transition the Chern
numbers become C5 = −2 and C6 = 2, and C7 = −2 and


-----

C8 = 2 remain unchanged. Even though four successive
bands have nonzero Chern numbers, the system remains
in the abelian four-band TP (see Sec. V).
Figure 14 shows the energies ǫ6,k and ǫ7,k of the 6th
and 7th Bloch bands respectively. For all wave vectors k,
ǫ7,k - ǫ6,k, i.e., the 6th and 7th bands are separated by
a gap. However, the maximum energy of the 6th band,
ǫ6,(0,q0/2) = −4.615 E0, is above the minimum energy of
the 7th band, ǫ7,(q0/2,q0/2) = −4.673 E0. The finite width
of the SDOLP in the strip geometry considered in the
main text lifts the translational symmetry in the y direction, hence ky is not a good quantum number. Projecting
the 3-dimensional image in Fig. 14 onto the kx-ǫ plane,
we get an overlap of the 6th and 7th bands, hence the
band-gap closes, see the red and gold bands in Fig. 2(d).

FIG. 14. The 6th and 7th Bloch bands for external magnetic
field Bext = E0/(µB gF ).

Appendix C: Calculating Chern numbers

To calculate the Chern numbers of the bands for
fermionic [6]Li atoms in the SDOLP we switch from position space to momentum space. Since [6]Li atoms are in
the hyperfine state with F = 1/2, the Sch¨odinger equation Hψn,k(r) = ǫn,kψn,k(r), with periodic Hamiltonian
having the lattice period a0 = 2π/q0 (see the main text),
can be written in a 2 2 matrix form as
×
� �� � � �
H+ W ψ 12 [,n,][k][(][r][)] ψ 12 [,n,][k][(][r][)]
W [∗] H− ψ− 12 [,n,][k][(][r][)] = ǫk ψ− 12 [,n,][k][(][r][)], (C1)


constant, it can be expanded in a Fourier series,

�
un,k(r) = e[i][q][·][r] u˜n,k(q), (C4)

q

where q = mxq1 + myq2 (mx and my are integer numbers) are the reciprocal lattice vectors and u˜n,k(q) are
Fourier coefficients. Similarly, the scalar potential V (r)
and the fictitious magnetic field Bfic(r) behave periodically with the lattice constant (see the main text) and
can be written as


�
V (r) = Vq e[i][q][·][r]

q

�
W (r) = Wq e[i][q][·][r] . (C5)

q

Inserting the expansions Eqs. (C4) and (C5) into
Eq. (C1) one obtains the Schr¨odinger equation in momentum space, which in what follows will be split into
two coupled equations with Fourier coefficients u˜ 12 [,n,][k]

and u˜− 21 [,n,][k] [for the hyperfine spin projections] [1]2 [and][ −] [1]2 [,]

respectively:


�
ℏ2 k + q 2
| |

− [g][F][ µ][B][B][ext]
2m 2


�

�

u˜ 12 [,n,][k][(][q][) +] Vq−q′ u˜ 12 [,n,][k][(][q][′][)]

q[′]


�
+ Wq−q′ u˜− 12 [,n,][k][(][q][′][) =][ ǫ][k][ ˜][u][ 1]2 [,n,][k][(][q][)][,] (C6)

q[′]


where


H± = − [ℏ][2] Bext,

2M 2

[∇][2][ +][ V][ (][r][)][ ∓] [g][F] [µ][B]

W = − [g][F] [µ][B] (Bfic,x(r) − iBfic,y(r)), (C2)

2

and the wave function ψn,k(r) is a two-component vector
ψn,k(r) = (ψ 12 [,n,][k][(][r][)][, ψ][−] 2[1] [,n,][k][(][r][))][T][ .]

The wave functions ψn,k(r), which are the solutions of
the Sch¨odinger equation in a periodic potential, take the
form of a plane wave modulated by a periodic functions,
according to the Bloch’s theorem,

ψn,k(r) = e[i][k][·][r] un,k(r), (C3)

where n is a band index and k is the momentum vector.
Since un,k(r) is periodic with period equal to the lattice


and
� �
ℏ2 k + q 2
| 2m | + [g][F][ µ][B]2[B][ext] u˜− 21 [,n,][k][(][q][)]

�
+ Vq−q′ u˜− 21 [,n,][k][(][q][′][)]

q[′]

�
+ Wq[∗][′]−q [u][˜][ 1]2 [,n,][k][(][q][′][) =][ ǫ][k][ ˜][u][−] 2[1] [,n,][k][(][q][)][.] (C7)

q[′]

Here the Fourier coefficients of scalar and vector potentials are given by integrals over an elementary cell


�
e[−][i][(][q][−][q][′][)][r] V (r) dx dy


Vq−q′ = a[1][2]0

Wq−q′ = a[1][2]0


�
e[−][i][(][q][−][q][′][)][r] W (r) dx dy . (C8)


The solutions of Eqs. (C6) and (C7), i.e., the eigenstates
of the Bloch Hamiltonian H(k), are the Bloch states
n, k in momentum space,
| ⟩




...



u˜ 12 [,n,][k][(][q][)]

u˜− 12 [,][k][(][q][)], (C9)

...


n, k =
| ⟩











-----

where q is an infinite set of reciprocal lattice vectors. In
the position representation one has

�
uα,n,k(r) ≡⟨α, r|n, k⟩ = u˜α,k(q) e[i][q][·][r], (C10)

q


where α = 1/2 is the spin projection on the z-axis.
±
The Chern numbers are found from the formula


Appendix D: Edge states

Topological edge states are symmetry-protected [1, 2].
In the discussion of the SDOLP in the main text (see also
Appendix A), we point out that there is no time-reversal
symmetry, but there are other symmetries, such as twoand fourfold rotational symmetries, C2,z and C4,z. For
an abelian TP [1], a pair of edge states propagating on
opposite edges has a degeneracy point, i.e., a value of kx
where they have the same energy. For a finite strip width
with topological edge states, there is a gap at the degeneracy point, hence the edge states are not topological, but
the gap decreases exponentially with increasing width of
the strip [3]. The degeneracy point can be clearly seen
(because the width is sufficiently large) in Fig. 2(b) between the 5th and 6th bands at kx = 0 and between
the 6th and 7h bands at the edge of the Brillouin zone.
This is illustrated in Fig. 15 in the upper left frame, for a
SDOLP with a strip-width of 3 lattice periods, where the
lattice period is a0 = 2π/q0. In this figure one can easily see the gap between the edge states connecting 5th
and 6th bands at kx = 0, as well as the one occurring
at the edge of the Brillouin zone for the states between
the 6h and 7th bands. Both avoided crossings disappear
when the width of the SDOLP is increased (see Fig. 15
for increasing number of cells).
In Figs. 2(c) and 2(d) there are two pairs of edge states
between the 7th and 8th bands, both of which are located
near the kx = 0 point. One of them has no degeneracy
point, and the other has degeneracy point at kx = 0.
These edge states are topological. For non-abelian topological edge states without a degeneracy point the finite
width of the strip doesn’t destroy the nature of the TES.
For non-abelian topological edge states with a degeneracy point, similar to the abelian case, the gap opens when
the width of a strip becomes smaller.


1
Cn =
2πi


�

Fxy(n, k), (C11)
k∈BZ


where the Berry curvature Fxy(n, k) is (see Ref. [32])

Fxy(n, k) = ln Uk,k+ˆkx(n) Uk+ˆkx,k+ˆkx+ˆky (n)

Uk,k+ˆky (n) Uk+ˆky,k+ˆkx+ˆky (n) [.]


(C12)

Here the link variables Uk′,k′′ (n) necessary to calculate
the curvature are defined as Uk′,k′′ (n) ≡⟨n, k[′]|n, k[′′]⟩,
where the wavevectors k[′] and k[′′] belong to the set k, k+
{
kˆx, k + kˆy, k + kˆx + kˆy}, where kˆ is a unit vector in
direction of one of the Cartesian axes, kx, ky. Note that
k[′] and k[′′] are both near the vector k, i.e., k[′′] k
| − | ≪
q0 and |k[′′] − k| ≪ q0, where q0 is a reciprocal lattice
wavenumber. This yields:


�
Uk′,k′′(n) =


q


�
u˜ 12 [,n,][k][′][(][q][) ˜][u][ 1]2 [,n,][k][′′][(][q][)]


�

+ u˜− 12 [,n,][k][′][(][q][) ˜][u][−] 2[1] [,n,][k][′′][(][q][)] . (C13)


We calculate the Chern numbers, Eq. (C11), numerically on the discretized Brillouin zone, and check the
convergence of the results by increasing the size of the
numerical grid. However, special care must be taken to
satisfy the periodicity of the Bloch Hamiltonian H(k)
in momentum space. Let’s assume that the Brillouin
zone is defined by the set of discrete points (kx, ky)
in the first Brillouin zone with the values of both momenta on a square grid with period ∆k, i.e., kx and
ky ∈{0, ∆k, 2∆k, ..., 2π/a0 − ∆k}. Then, calculating
the link variables and the curvature at the points near
the boundary of the BZ, when kx = 2π/a0 − ∆k or
ky = 2π/a0 − ∆k, we need solutions for the points
(kx = 2π/a0, ky) or (kx, ky = 2π/a0). These points are
equivalent to the points (kx = 0, ky) or (kx, ky = 0) because of periodicity. Unfortunately, in numerical calculations, the periodicity of the Hamiltonian H(k) is broken
because of the finite number of terms used in expansion
(C4). The remedy is to neglect the points closest to the
edge of the Brillouin zone when calculating the sum in
Eq. (C11), while simultaneously increasing the size of the
numerical grid. As a result of this procedure, the sum in
Eq. (C11) approaches an integer number which is the
Chern number for the band.


- 4.45
- 4.50
- 4.55
- 4.60
- 4.65
- 4.70
- 4.75
- 4.80
   - 0.4    - 0.2 0.0 0.2 0.4

kx/q0

- 4.45
- 4.50
- 4.55
- 4.60
- 4.65
- 4.70
- 4.75
- 4.80
   - 0.4    - 0.2 0.0 0.2 0.4

kx/q0


- 4.45
- 4.50
- 4.55
- 4.60
- 4.65
- 4.70
- 4.75
- 4.80
   - 0.4    - 0.2 0.0 0.2 0.4

kx/q0

- 4.45
- 4.50
- 4.55
- 4.60
- 4.65
- 4.70
- 4.75
- 4.80
   - 0.4    - 0.2 0.0 0.2 0.4

kx/q0


FIG. 15. Energies ǫn,kx of the energy bands n = 5, 6, 7, and
8 for V0 = 5 E0, B0 = 10 B0, and for the value of the external
magnetic field Bext = 0.2 B0. Successive frames (from left to
right and top to bottom) show the spectra for finite width
SDOLP strips having 3, 5, 7, and 9 elementary lattice period.


-----

[1] M. Z. Hasan and C. L. Kane, “Colloquium: Topological
insulators”, Rev. Mod. Phys. 82, 3045 (2010).

[2] X.-L. Qi and S.-C. Zhang, “Topological insulators and
superconductors”, Rev. Mod. Phys. 83, 1057 (2011).

[3] J. K. Asb´oth, L. Oroszl´any, A. P´alyi,A Short Course on
Topological Insulators, (Springer International Publishing, 2016).

[4] N. R. Cooper, J. Dalibard, and I. B. Spielman “Topological bands for ultracold atoms”, Rev. Mod. Phys. 91,
015005 (2019).

[5] N. Goldman, G. Juzeli¯unas, P. Ohberg[¨] and I. B. Spielman, “Light-induced gauge fields for ultracold atoms”,
Rep. Prog. Phys. 77 126401 (2014).

[6] C.-K. Chiu, J. C. Y. Teo, A. P. Schnyder, and S. Ryu,
“Classification of topological quantum matter with symmetries”, Rev. Mod. Phys. 88, 035005 (2016).

[7] B. A. Bernevig, T. L. Hughes, and S.-C. Zhang, “Quantum Spin Hall Effect and Topological Phase Transition
in HgTe Quantum Wells”, Science 314, 1757 (2006).

[8] M. K¨onig, S. Wiedmann, C. Br¨une, A. Roth, H. Buhmann, L. W. Molenkamp, X.-L. Qi, and S.-C. Zhang,
“Quantum spin hall insulator state in HgTe quantum
wells”, Science 318, 766 (2007).

[9] D. Hsieh, D. Qian, L. Wray, Y. Xia, Y. S. Hor, R. J.
Cava, and M. Z. Hasan, “A topological Dirac insulator
in a quantum spin Hall phase”, Nature (London) 452,
970 (2008).

[10] D. Hsieh, Y. Xia, D. Qian, L. Wray, J. H. Dil, F. Meier, J.
Osterwalder, L. Patthey, J. G. Checkelsky, N. P. Ong, A.
V. Fedorov, H. Lin, A. Bansil, D. Grauer, Y. S. Hor, R.
J. Cava, and M. Z. Hasan, “A tunable topological insulator in the spin helical Dirac transport regime”, Nature
(London) 460, 1101 (2009).

[11] Y. Xia, D. Qian, D. Hsieh, L. Wray, A. Pal, H. Lin, A.
Bansil, D. Grauer, Y. S. Hor, R. J. Cava, and M. Z.
Hasan, “Observation of a large-gap topological-insulator
class with a single Dirac cone on the surface”, Nat. Phys.
5, 398 (2009).

[12] H. Zhai, “Degenerate quantum gases with spin-orbit coupling: a review”, Rep. Prog. Phys. 78, 026001 (2015).

[13] J. Dalibard, Quantum Matter at Ultralow Temperatures
edited by M. Inguscio, W. Ketterle, and S. Stringari (IOS
Press, Amsterdam, 2016).

[14] M. Aidelsburger, S. Nascimbene, and N. Goldman, “Artificial gauge fields in materials and engineered systems”,
C. R. Physique 19, 394 (2018).

[15] Z. Wu, L. Zhang, W. Sun, X.-T. Xu, B.-Z. Wang, S.-C. Ji,
Y. Deng, S. Chen, X.-J. Liu, and J.-W. Pan, “Realization
of two-dimensional spin-orbit coupling for Bose-Einstein
condensates”, Science 354, 83 (2016).

[16] L. Huang, Z. Meng, P. Wang, P. Peng, S.-L. Zhang, L.
Chen, D. Li, Q. Zhou, and J. Zhang, “Experimental realization of two-dimensional synthetic spin-orbit coupling
in ultracold Fermi gases”, Nat. Phys. 12, 540 (2016).

[17] J. Struck, M. Weinberg, C. Olschl¨ager, P. Windpassinger,[¨]
J. Simonet, K. Sengstock, R. H¨oppner, P. Hauke, A.
Eckardt, M. Lewenstein, and L. Mathey “Engineering
Ising-XY spin-models in a triangular lattice using tunable artificial gauge fields”, Nat. Phys. 9, 738 (2013).

[18] G. Jotzu, M. Messer, R. Desbuquois, M. Lebrat, T.
Uehlinger, D. Greif, and T. Esslinger, “Experimental re

alization of the topological Haldane model with ultracold
fermions”, Nature (London) 515, 237 (2014).

[19] N. Fl¨aschner, B. S. Rem, M. Tarnowski, D. Vogel, D.S. L¨uhmann, K. Sengstock, and C. Weitenberg, “Experimental reconstruction of the Berry curvature in a Floquet
Bloch band”, Science 352, 1091 (2016).

[20] M. Mancini, G. Pagano, G. Cappellini, L. Livi, M. Rider,
J. Catani, C. Sias, P. Zoller, M. Inguscio, M. Dalmonte,
L. Fallani, “Observation of chiral edge states with neutral
fermions in synthetic Hall ribbons”, Science 349, 1510
(2015).

[21] B. K. Stuhl, H.-I. Lu, L. M. Aycock, D. Genkina, and
I. B. Spielman, “Visualizing edge states with an atomic
Bose gas in the quantum Hall regime”, Science 349, 1514
(2015).

[22] I. Kuzmenko, T. Kuzmenko, Y. Avishai, and Y. B. Band,
“Atoms trapped by a spin-dependent optical lattice potential: Realization of a ground-state quantum rotor”,
Phys. Rev. A 100, 033415 (2019).

[23] P. Szulim, M. Trippenbach, Y. B. Band, M. Gajda, and
M. Brewczyk, “Atoms in a spin dependent optical potential: ground state topology and magnetization”, New J.
Phys. 24, 033041 (2022).

[24] S. Mao, A. Yamakage, and Y. Kuramoto, “Tight-binding
model for topological insulators: Analysis of helical surface modes over the whole Brillouin zone”, Phys. Rev. B
84, 115413 (2011).

[25] T. Jiang, Q. Guo, R.-Y. Zhang, Z.-Q. Zhang, B. Yang,
and C. T. Chan, “Four-band non-Abelian topological insulator and its experimental realization”, Nat. Commun.
12, 6471 (2021).

[26] F. Le Kien, P. Schneeweiss, and A. Rauschenbeutel, “Dynamical polarizability of atoms in arbitrary light fields:
general theory and application to cesium”, Eur. Phys. J.
D 67, 92 (2013).

[27] C. Cohen-Tannoudji and J. Dupont-Roc, “Experimental
Study of Zeeman Light Shifts in Weak Magnetic Fields”,
Phys. Rev. A 5, 968 (1972).

[28] A. M. Dudarev, R. B. Diener, I. Carusotto, and Q. Niu,
“Spin-Orbit Coupling and Berry Phase with Ultracold
Atoms in 2D Optical Lattices”, Phys. Rev. Lett. 92,
153005 (2004).

[29] W. Scherf, O. Khait, H. J¨ager, and L. Windholz, “Remeasurement of the transition frequencies, fine structure splitting and isotope shift of the resonance lines
of lithium, sodium and potassium”, Z. Phys. D 36, 31
(1996).

[30] W. I. McAlexander, E. R. I. Abraham, and R. G. Hulet,
“Radiative lifetime of the 2P state of lithium”, Phys. Rev.
A 54, R5 (1996).

[31] E. R. I. Abraham, W. I. McAlexander, J. M. Gerton,
R. G. Hulet, R. Cˆot`e, and A. Dalgarno “Triplet s-wave
resonance in 6Li collisions and scattering lengths of 6Li
and 7Li” Phys. Rev. A55, R3299 (1997).

[32] T. Fukui, Y. Hatsugai, and H. Suzuki, “Chern Numbers
in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances”, J. Phys. Soc. Jpn. 74,
1674 (2005).

[33] T. Jiang, R.-Y. Zhang, Q. Guo, B. Yang, and C. T.
Chan, “Two-dimensional non-Abelian topological insulators and the corresponding edge/corner states from an


-----

eigenvector frame rotation perspective”, Phys. Rev. B
106, 235428 (2022).

[34] Q. Wu, A. A. Soluyanov, and T. Bzduˇsek, “Non-Abelian
band topology in noninteracting metals”, Science 365,
1273 (2019).

[35] A. Bouhon, Q. Wu, R.-J. Slager, H. Weng, O. V. Yazyev,
and T. Bzduˇsek, “Non-Abelian reciprocal braiding of
Weyl points and its manifestation in ZrTe”, Nat. Phys.


16, 1137 (2020).

[36] D.M. Reich, “Characterisation and Identification of Unitary Dynamics Maps in Terms of Their Action on Density
Matrices”, (unpublished).

[37] JX. Cui, ZD. Wang, “Measuring the degree of unitarity for any quantum process”, Eur. Phys. J. D 68, 248
(2014).


-----

