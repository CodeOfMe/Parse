## Numerical Approximations and Error Analysis of the Cahn-Hilliard Equation with Dynamic Boundary Conditions

### Xuelian Bao[*], Hui Zhang[†]


**Abstract** We consider the numerical approximations of the Cahn-Hilliard equation with
dynamic boundary conditions (C. Liu et. al., Arch. Rational Mech. Anal., 2019). We
propose a first-order in time, linear and energy stable numerical scheme, which is based on
the stabilized linearly implicit approach. The energy stability of the scheme is proved and the
semi-discrete-in-time error estimates are carried out. Numerical experiments, including the
comparison with the former work, the accuracy tests with respect to the time step size and
the shape deformation of a droplet, are performed to validate the accuracy and the stability of
the proposed scheme.

## 1 Introduction


The Cahn-Hilliard equation is one of the most fundamental models which describe the phase
separation processes of binary mixtures. The classical Cahn-Hilliard equation, first introduced in

[1], can be written as follows:


φt = ∆µ, in Ω × (0, T ),

µ = −ε∆φ + [1] in Ω × (0, T ),

ε _[F][′][(][φ][)][,]_


(1.1)





where T ∈ (0, ∞) and Ω ⊆ R[d] (d = 2, 3) is a bounded domain with the smooth boundary Γ = ∂Ω.
The phase-field order parameter φ represents the difference of two local relative concentrations
to describe the binary mixtures. In the domain Ω, φ = ±1 correspond to the pure phases of
the materials, which are separated by a interfacial region whose thickness is proportional to the
parameter ε. µ represents the chemical potential in Ω, which can be expressed as the Fr´echet
derivative of the bulk free energy:


�
(1.2) _E[bulk](φ) =_


Ω


ε [1]
2 [|∇][φ][|][2][ +] ε _[F][(][φ][)d][x][,]_


where F stands for the bulk potential, which usually has a double-well structure with two minima
at -1 and 1 and a local unstable maximum at 0. A classical choice is the regular double-well

2010 Mathematics Subject Classification. 65M12; 65M06; 65N12; 65M22.
_Key words and phrases._ Cahn-Hilliard equation; Dynamic boundary conditions; Error estimates; Linear numerical
scheme; Energy stability.

*Corresponding author, School of Mathematical Sciences, Beijing Normal University, Beijing 100875, China (email: xlbao@mail.bnu.edu.cn).
†Laboratory of Mathematics and Complex Systems, Ministry of Education and School of Mathematical Sciences,
Beijing Normal University, Beijing 100875, China.


-----

2 Xuelian Bao, Hui Zhang

potential

(1.3) _F(x) =_ [1] _x ∈_ R.

4 [(][x][2][ −] [1)][2][,]

When the time-evolution of φ is confined in a bounded domain, suitable boundary conditions
should be considered for the system (1.1). The homogeneous Neumann conditions are the classical
boundary conditions:

(1.4) ∂nµ = 0, on Γ × (0, T ),

(1.5) ∂nφ = 0, on Γ × (0, T ),

where n = n(x) denotes the unit outer normal vector and ∂n denotes the outward normal derivative
on Γ. The Cahn-Hilliard equation with the boundary conditions (1.4) and (1.5) can be viewed as
an H[−][1]-gradient flow of the bulk free energy.
The no-flux boundary condition (1.4) guarantees the conservation of mass in the bulk (i.e., in
Ω):


�
(1.6)


�
φ(t)dx =
Ω


φ(0)dx, _t ∈_ [0, T ].
Ω


Moreover, the boundary conditions (1.4) and (1.5) imply that the bulk free energy E[bulk] (Eq. (1.2))
is decreasing with respect to time, namely,


_d_ �
(1.7) |∇µ|[2]dx = 0, _t ∈_ (0, T ).
_dt [E][bulk][(][φ][(][t][))][ +]_ Ω


However, the Cahn-Hilliard equation with homogeneous Neumann conditions neglects the effects of the boundary to the bulk dynamics. Thus, it is not suitable for some applications (for
instance, hydrodynamic applications such as contact line problems). In order to describe the effective interactions between the solid wall and the binary mixture, physicists added the suitable
surface free energy functional into the system [4, 5, 14]:

(1.8) _E[total](φ) =_ _E[bulk](φ) + E_ _[sur f]_ (φ),


�
(1.9) _E_ _[sur f]_ (φ) =


Γ


δκ [1]

2 [|∇][Γ][φ][|][2][ +] δ _[G][(][φ][)d][S]_ [,]


where ∇Γ represents the tangential or surface gradient operator on Γ, G is the surface potential, the
parameter κ is related to the surface diffusion and δ denotes the thickness of the interfacial region
on Γ. When κ = 0, it is related to the moving contact line problem [21]. Recently, for the total
free energy (1.8), various dynamic boundary conditions for the Cahn-Hilliard equation have been
proposed and investigated, see for instance, see [14, 9, 17, 15, 16], and references therein.


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 3

In the present work, the Cahn-Hilliard equation with the dynamic boundary conditions, which
was derived by an energetic variational approach by Liu and Wu (Liu-Wu model, for short) [17],
is considered. It reads as follows:


(1.10)


φt = ∆µ, in Ω × (0, T ),

µ = −ε∆φ + [1] in Ω × (0, T ),

ε _[F][′][(][φ][)][,]_

∂nµ = 0, on Γ × (0, T ),

φ|Γ = ψ, on Γ × (0, T ),

ψt = ∆ΓµΓ, on Γ × (0, T ),

µΓ = −δκ∆Γψ + [1] on Γ × (0, T ),

 δ _[G][′][(][ψ][)][ +][ ε∂][n][φ]_


where ∆Γ denotes the Laplace-Beltrami operator on Γ. The dynamic boundary conditions (with δ,
κ - 0) turns out to be a surface Cahn-Hilliard type equation for the trace of φ on Γ, coupled with
the bulk evolution in terms of ∂nφ. The existence and uniqueness of weak and strong solutions
of the Liu-Wu model have been established in [17]. A different approach to construct the weak
solutions of the Liu-Wu model is proposed in [8].
The numerical approximations of the Cahn-Hilliard equation and its variants have been intensively investigated. The stabilized linearly implicit approach [11, 23], the approaches based on the
convex-concave splitting [19, 10], the invariant energy quadratization [24, 25] and the scalar auxiliary variable (SAV) [20] method are efficient techniques for the time discretization. Recently, there
have been numerous contributions on the numerical approximation of the Cahn-Hilliard equation
with dynamic boundary conditions [2, 3, 13, 7, 22]. For the numerical approximations of LiuWu model, the first finite element scheme was proposed in [22] and the corresponding numerical
results were presented in [8], where the straightforward discretization based on piecewise linear
finite element functions was utilized to simulate Liu-Wu model, and the corresponding nonlinear system was solved by Newton’s method. A recent contribution on the numerical analysis for
the Liu-Wu model can be found in [18], where a different discrete scheme was proposed and the
connection between φ and the chemical potentials was investigated. However, the backward implicit Euler method was used for time discretization in the above discrete schemes, which lead to
nonlinear systems at each time step.
In the present work, a first-order in time, linear and energy stable scheme for solving the Liu-Wu
model is proposed based on the stabilized linearly implicit approach. At each time step, one only
needs to solve one linear equation and thus, the scheme is highly efficient. The energy stability of
the scheme is proved and various numerical simulations in two-dimensional spaces are performed
to validate the accuracy and stability of the scheme by comparing with the former work. The error
estimates in semi-discrete-in-time for the scheme are also carried out. To the best of the authors’
knowledge, the proposed scheme in this paper is the first linear and energy stable scheme for
solving the Liu-Wu model and it is the first work to give the semi-discrete-in-time error estimates
for the model.
The rest of the paper is organized as follows. In Section 2, we first recall some notions and
notation appearing in this article. In Section 3, a simple derivation of Liu-Wu model and the stabilized scheme with the energy stability are derived. In Section 4, we construct the error estimates.


-----

4 Xuelian Bao, Hui Zhang

The accuracy tests and numerical examples are presented in Section 5. Finally, some concluding
remarks are presented in Section 6.

## 2 Preliminaries

Before giving the stabilized scheme and the corresponding error analysis, we make some definitions in this section. The norm and inner product of L[2](Ω) and L[2](Γ) are denoted by ∥· ∥Ω, (·, ·)Ω
and ∥· ∥Γ, (·, ·)Γ respectively. The usual norm in _H[k](Ω)_ and _H[k](Γ)_ are denoted by ∥· ∥Hk(Ω) and
∥· ∥Hk(Γ) respectively.
We consider a finite time interval [0, T ] and a domain Ω ⊂ R[d] (d = 2, 3), which is a bounded
domain with sufficient smooth boundary Γ = ∂Ω and n = **n(x) is the unit outer normal vector on**
Γ.
Let τ be the time step size. For a sequence of functions _f_ [0], _f_ [1], . . ., _f_ _[N]_ in some Hilbert space E,
we denote the sequence by { fτ} and define the following discrete norm for { _fτ}:_


(2.1) ∥ _fτ∥l[∞](E)_ = max
0≤n≤N


� �
∥ _f_ _[n]∥E_ .


We denote by C a generic constant that is independent of τ but possibly depends on the data and
the solution, and use _f_ ≲ _g to say that there is a generic constant C such that_ _f_ ⩽ _Cg._

## 3 Derivation of the Cahn-Hilliard equation with dynamic boundary conditions and its numerical scheme

In this section, we first propose a simple derivation of the Liu-Wu model, indicating that it
satisfies the energy dissipation law and mass conservation.
Since φ is the phase-field order parameter in the bulk, denote its trace ψ := φ|Γ as the order
parameter on the boundary. When the mass conservation holds true in the bulk Ω and on the
boundary Γ respectively, φ and ψ satisfy the following continuity equations [6]:

φt + ∇· (φu) = 0, in Ω × (0, T ),
(3.1)

ψt + ∇Γ · (φv) = 0, on Γ × (0, T ),

where u is the microscopic effective velocity and v is the microscopic effective tangential velocity
field on the boundary. Assume that there is no mass exchange between the bulk and the boundary,
**u satisfies the following boundary condition:**

(3.2) **u · n = 0,** on Γ × (0, T ).

Since the boundary is closed, there is no need to impose any boundary condition on v.
For the total free energy (1.8), we consider the following energy dissipation law:

_d_
(3.3)
_dt [E][total][(][t][)][ =][ −D][total][(][t][)][,]_


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 5

which is based on the first and second laws of thermodynamics [12]. And the rate of energy
dissipation D[total](t) also consists of two contributions from the bulk and the boundary, namely,

(3.4) D[total](t) = D[bulk](t) + D[sur f] (t).

Assume that


� �
(3.5) D[bulk](t) = φ[2]|u|[2]dx, D[sur f] (t) =

Ω

and substitute (1.2) and (1.9) into (3.3), we obtain


ψ[2]|v|[2]dS,
Γ


δκ [1] �

2 [|∇][Γ][φ][|][2][ +] δ _[G][(][φ][)d][S]_


(3.6)


_d_ � [�] ε [1] �
_dt_ Ω 2 [|∇][φ][|][2][ +] ε _[F][(][φ][)d][x][ +]_ Γ

� �
= − φ[2]|u|[2]dx − ψ[2]|v|[2]dS .

Ω Γ


The left part of (3.6) can be written as


δκ [1] �

2 [|∇][Γ][φ][|][2][ +] δ _[G][(][φ][)d][S]_


ε [1] �
2 [|∇][φ][|][2][ +] ε _[F][(][φ][)d][x][ +]_


_d_
_dt_


� [�]


(3.7)


_dt_ Ω 2 ε Γ 2 δ

� �
= [−ε∆φ + [1] [ε∂nφ − δκ∆Γψ + [1]

Ω ε _[F][′][(][φ][)]][φ][t][d][x][ +]_ Γ δ _[G][′][(][ψ][)]][ψ][t][d][S]_

� �
= φ∇[−ε∆φ + [1] ψ∇Γ[ε∂nφ − δκ∆Γψ + [1]

Ω ε _[F][′][(][φ][)]][ ·][ u][d][x][ +]_ Γ δ _[G][′][(][ψ][)]][ ·][ v][d][S]_

� �
= φ∇µ · udx + ψ∇Γ[ε∂nφ + µs] · vdS .

Ω Γ


Here, µ and µs are the chemical potentials in Ω and on Γ, respectively, which can be expressed as
the Fr´echet derivative of the bulk free energy E[bulk] and the surface free energy E _[sur f]_, namely,

µ = [δ][E][bulk] = −ε∆φ + [1]

δφ ε _[F][′][(][φ][)]_

(3.8)

µs = [δ][E] _[sur f]_ = −δκ∆Γψ + [1]

δψ δ _[G][′][(][ψ][)][.]_


From (3.6) and (3.7), we obtain

φu = −∇µ,
(3.9)

ψv = −∇Γ(ε∂nφ + µs).

Substitute (3.9) into (3.1), we obtain the Liu-Wu model (Eq. (1.10)) with µΓ = ε∂nφ + µs.
Obviously, the Liu-Wu model satisfies the mass conservation law in the bulk and on the boundary:


� �
(3.10) φ(t)dx =

Ω


�
φ(0)dx and
Ω


�
ψ(t)dS = ψ(0)dS, _t ∈_ [0, T ],
Γ Γ


-----

6 Xuelian Bao, Hui Zhang

Moreover, from the energy dissipation law, it is easy to see that the total free energy E[total](φ, ψ) =
_E[bulk](φ) + E_ _[sur f]_ (ψ) is decreasing in time:

d
(3.11) Ω [−∥∇][Γ][µ][Γ][∥]Γ[2] [≤] [0][.]
dt [E][total][(][φ, ψ][)][ =][ −∥∇][µ][∥][2]

Now we present the stabilized scheme for the Cahn-Hilliard equation with dynamic boundary
conditions (namely, (1.10)). The scheme can be written as follows,

φ[n][+][1] − φ[n]

(3.12) = ∆µ[n][+][1], in Ω,
τ

(3.13) µ[n][+][1] = −ε∆φ[n][+][1] + [1] in Ω,
ε _[F][′][(][φ][n][)][ +][ s][1][(][φ][n][+][1][ −]_ [φ][n][)][,]

(3.14) ∂nµ[n][+][1] = 0, on Γ,


(3.15) φ[n][+][1]|Γ = ψ[n][+][1], on Γ,

ψ[n][+][1] − ψ[n]

(3.16) τ = ∆Γµ[n]Γ[+][1], on Γ,

(3.17) µ[n]Γ[+][1] = −δκ∆Γψ[n][+][1] + [1]δ _[G][′][(][ψ][n][)][ +][ ε∂][n][φ][n][+][1][ +][ s][2][(][ψ][n][+][1][ −]_ [ψ][n][)][,] on Γ.

Here, T is an arbitrary and fixed time, N is the number of time steps and τ = T/N is the step size.
We have the energy stability as follows.

**Theorem 3.1.** _Assume that s1_ ≥ 2[1]ε [max][ξ][∈][R][ F][′′][(][ξ][)][, s][2] [≥] 2[1]δ [max][η][∈][R][ G][′′][(][η][)][, the scheme][ (3.12)][-][(3.17)]

_is energy stable in the sense that_


_E(φ[n][+][1], ψ[n][+][1]) −_ _E(φ[n], ψ[n])_
(3.18) ≤−∥∇µ[n][+][1]∥[2]Ω [−∥∇][Γ][µ]Γ[n][+][1]∥[2]Γ[,]

τ


_where_

�
(3.19) _E(φ[n], ψ[n]) =_


Γ


Ω


1 [ε] �
ε _[F][(][φ][n][)][ +]_ 2 [|∇][φ][n][|][2][dx][ +]


1 [δκ]
δ _[G][(][ψ][n][)][ +]_ 2 [|∇][Γ][ψ][n][|][2][dS]


_Proof._ By taking inner product of (3.12) with µ[n][+][1] in Ω, we have

(3.20) ( [φ][n][+][1][ −] [φ][n], µ[n][+][1])Ω = (∆µ[n][+][1], µ[n][+][1])Ω = −∥∇µ[n][+][1]∥[2]Ω[.]

τ

By using (3.13), we have


(3.21) ( [φ][n][+][1][ −] [φ][n], µ[n][+][1])Ω = ( [φ][n][+][1][ −] [φ][n], −ε∆φ[n][+][1] + [1]

τ τ ε _[F][′][(][φ][n][)][ +][ s][1][(][φ][n][+][1][ −]_ [φ][n][))][Ω][,]

(3.22) ( [φ][n][+][1][ −] [φ][n], −ε∆φ[n][+][1])Ω = −ε(∂nφ[n][+][1], [φ][n][+][1][ −] [φ][n] )Γ + ε(∇φ[n][+][1], [∇][φ][n][+][1][ −∇][φ][n] )Ω.

τ τ τ


For the boundary integral term in (3.22), by taking the inner product of (3.16) with µ[n]Γ[+][1] on Γ,
we obtain

(3.23) ( [ψ][n][+][1]τ[ −] [ψ][n], µ[n]Γ[+][1])Γ = (∆Γµ[n]Γ[+][1], µ[n]Γ[+][1])Γ = −∥∇Γµ[n]Γ[+][1]∥[2]Γ[.]


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 7

By using (3.17), we have

(3.24) ( [ψ][n][+][1]τ[ −] [ψ][n], µ[n]Γ[+][1])Γ = ( [ψ][n][+][1]τ[ −] [ψ][n], −δκ∆Γψ[n][+][1] + [1]δ _[G][′][(][ψ][n][)]_

+ ε∂nφ[n][+][1] + s2(ψ[n][+][1] − ψ[n]))Γ,


(3.25) ( [ψ][n][+][1][ −] [ψ][n], −δκ∆Γψ[n][+][1])Γ = ( [∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n], δκ∇Γψ[n][+][1])Γ.

τ τ

To handle the nonlinear term associated with F[′] and G[′] in (3.21) and (3.24), we need the following
identities

_F[′](φ[n])(φ[n][+][1]_ − φ[n]) = _F(φ[n][+][1]) −_ _F(φ[n]) −_ _[F][′′][(][η][)]_ (φ[n][+][1] − φ[n])[2],

2

(3.26)

_G[′](φ[n])(φ[n][+][1]_ − φ[n]) = G(φ[n][+][1]) − _G(φ[n]) −_ _[G][′′][(][ζ][)]_ (φ[n][+][1] − φ[n])[2].

2

Combining the equations mentioned above, we get


and


( [φ][n][+][1][ −] [φ][n], µ[n][+][1])Ω + ( [ψ][n][+][1][ −] [ψ][n], µ[n]Γ[+][1])Γ = −∥∇µ[n][+][1]∥[2]Ω [−∥∇][Γ][µ]Γ[n][+][1]∥[2]Γ[,]

τ τ

( [φ][n][+][1]τ[ −] [φ][n], µ[n][+][1])Ω + ( [ψ][n][+][1]τ[ −] [ψ][n], µ[n]Γ[+][1])Γ

= ε(∇φ[n][+][1], [∇][φ][n][+][1][ −∇][φ][n] )Ω + [1] )Ω + _[s][1]_ Ω

τ ε [(][F][′][(][φ][n][)][, φ][n][+][1]τ[ −] [φ][n] τ [∥][φ][n][+][1][ −] [φ][n][∥][2]

+ (δκ∇Γψ[n][+][1], [∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n] )Γ + [1] )Γ + _[s][2]_ Γ

τ δ [(][G][′][(][ψ][n][)][, ψ][n][+][1]τ[ −] [ψ][n] τ [∥][ψ][n][+][1][ −] [ψ][n][∥][2]

= ε(∇φ[n][+][1], [∇][φ][n][+][1][ −∇][φ][n] )Ω + [1], 1)Ω − [1] )Ω

τ ε [(] _[F][(][φ][n][+][1][)]τ[ −]_ _[F][(][φ][n][)]_ 2ε [(][F][′′][(][η][)][,][ (][φ][n][+][1][ −]τ [φ][n][)][2]

+ _[s][1]_ Ω [+][ δκ][(][∇][Γ][ψ][n][+][1][,][ ∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n] )Γ + [1], 1)Γ

τ [∥][φ][n][+][1][ −] [φ][n][∥][2] τ δ [(][G][(][ψ][n][+][1][)]τ[ −] _[G][(][ψ][n][)]_

− [1] )Γ + _[s][2]_ Γ

2δ [(][G][′′][(][ζ][)][,][ (][ψ][n][+][1][ −]τ [ψ][n][)][2] τ [∥][ψ][n][+][1][ −] [ψ][n][∥][2]

= [ε] Ω [−∥∇][φ][n][∥][2]Ω [+][ ∥∇][φ][n][+][1][ −∇][φ][n][∥][2]Ω[)][ +] [1]

2τ [(][∥∇][φ][n][+][1][∥][2] ετ [(][F][(][φ][n][+][1][)][ −] _[F][(][φ][n][)][,][ 1)][Ω]_

+ [1] Ω [+] [δκ] Γ

τ [(][s][1][ −] 2[1]ε _[F][′′][(][η][))][∥][φ][n][+][1][ −]_ [φ][n][∥][2] 2τ [(][∥∇][Γ][ψ][n][+][1][∥][2]

−∥∇Γψ[n]∥[2]Γ [+][ ∥∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n][∥]Γ[2][)][ +] [1]

δτ [(][G][(][ψ][n][+][1][)][ −] _[G][(][ψ][n][)][,][ 1)][Γ]_

+ [1]τ [(][s][2][ −] 2[1]δ[G][′′][(][ζ][))][∥][ψ][n][+][1][ −] [ψ][n][∥]Γ[2]

= [1]τ [[][E][(][φ][n][+][1][, ψ][n][+][1][)][ −] _[E][(][φ][n][, ψ][n][)]][ +]_ 2[ε]τ [∥∇][φ][n][+][1][ −∇][φ][n][∥]Ω[2]

+ [δκ] Γ [+] [1] Ω

2τ [∥∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n][∥][2] τ [(][s][1][ −] 2[1]ε _[F][′′][(][η][))][∥][φ][n][+][1][ −]_ [φ][n][∥][2]

+ [1] Γ[.]

τ [(][s][2][ −] 2[1]δ[G][′′][(][ζ][))][∥][ψ][n][+][1][ −] [ψ][n][∥][2]


-----

8 Xuelian Bao, Hui Zhang

Thus, we have

1τ [[][E][(][φ][n][+][1][, ψ][n][+][1][)][ −] _[E][(][φ][n][, ψ][n][)]][ +]_ 2[ε]τ [∥∇][φ][n][+][1][ −∇][φ][n][∥]Ω[2]

+ [δκ] Γ [+] [1] Ω

2τ [∥∇][Γ][ψ][n][+][1][ −∇][Γ][ψ][n][∥][2] τ [(][s][1][ −] 2[1]ε _[F][′′][(][η][))][∥][φ][n][+][1][ −]_ [φ][n][∥][2]

+ [1] Γ [=][ −∥∇][µ][n][+][1][∥]Ω[2] [−∥∇][Γ][µ]Γ[n][+][1]∥[2]Γ [≤] [0][.]

τ [(][s][2][ −] 2[1]δ[G][′′][(][ζ][))][∥][ψ][n][+][1][ −] [ψ][n][∥][2]


Therefore, under the conditions that


_s1_ ≥ [1]

2ε [max]ξ∈R _[F][′′][(][ξ][)]_

and
_s2_ ≥ [1]

2δ [max]η∈R _[G][′′][(][η][)][,]_


we have
1
τ [[][E][(][φ][n][+][1][, ψ][n][+][1][)][ −] _[E][(][φ][n][, ψ][n][)]][ ≤]_ [0][,]

namely, the scheme (3.12)-(3.17) is energy stable. 
## 4 Error estimates for the stabilized semi-discrete scheme

In this section, we establish the error estimates for the phase functions φ and ψ for the stabilized
scheme (3.12)-(3.17).
Assume that the Lipschitz properties hold for the derivatives of F[′] and G[′],

′′
(4.1) max (φ)| ≤ _L1,_
φ∈R [|][F]

′′
(4.2) max (ψ)| ≤ _L2,_
ψ∈R [|][G]

which are necessary for error estimates.
The PDE system (1.10) can be rewritten as the following truncated form,

φ(t[n][+][1]) − φ(t[n])

(4.3) τ = ∆µ(t[n][+][1]) + R[n]φ[+][1], in Ω,

(4.4) µ(t[n][+][1]) = −ε∆φ(t[n][+][1]) + [1]ε _[F][′][(][φ][(][t][n][))][ +][ s][1][(][φ][(][t][n][+][1][)][ −]_ [φ][(][t][n][))][ +][ R]µ[n][+][1], in Ω,

(4.5) ∂nµ(t[n][+][1]) = 0, on Γ,

(4.6) φ(t[n][+][1])|Γ = ψ(t[n][+][1]), on Γ,

ψ(t[n][+][1]) − ψ(t[n])

(4.7) = ∆ΓµΓ(t[n][+][1]) + R[n]ψ[+][1][,] on Γ,
τ

µΓ(t[n][+][1]) = −δκ∆Γψ(t[n][+][1]) + [1]

δ _[G][′][(][ψ][(][t][n][))][ +][ ε∂][n][φ][(][t][n][+][1][)]_


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 9

(4.8) +s2(ψ(t[n][+][1]) − ψ(t[n])) + R[n]Γ[+][1], on Γ,

where

(4.9) _R[n]φ[+][1]_ = [φ][(][t][n][+][1][)]τ[ −] [φ][(][t][n][)] − φt(t[n][+][1]),


(4.10) _R[n]ψ[+][1]_ = [ψ][(][t][n][+][1][)]τ[ −] [ψ][(][t][n][)] − ψt(t[n][+][1]),

(4.11) _R[n]µ[+][1]_ = [1]ε _[F][′][(][φ][(][t][n][+][1][))][ −]_ [1]ε _[F][′][(][φ][(][t][n][))][ −]_ _[s][1][(][φ][(][t][n][+][1][)][ −]_ [φ][(][t][n][))][,]


(4.12) _R[n]Γ[+][1]_ = [1]δ _[G][′][(][ψ][(][t][n][+][1][))][ −]_ [1]δ _[G][′][(][ψ][(][t][n][))][ −]_ _[s][2][(][ψ][(][t][n][+][1][)][ −]_ [ψ][(][t][n][))][.]

We assume that the exact solutions of the system (1.10) possesses the following regularity:


φ, φt, φtt ∈ _L[∞](0, T_ ; H[3](Ω));

ψ, ψt, ψtt ∈ _L[∞](0, T_ ; H[3](Γ));

µ ∈ _L[∞](0, T_ ; H[2](Ω));

µΓ ∈ _L[∞](0, T_ ; H[2](Γ)).


(4.13) (A) :





From the Taylor expansion, it’s easy to prove that

**Lemma 4.1.** _Under the Assumption (A), the truncation errors satisfy_

∥Rφ,τ∥l∞(H1(Ω)) + ∥Rµ,τ∥l∞(H1(Ω)) ≲ τ,

∥Rφ,τ∥l∞(L2(Ω)) + ∥Rµ,τ∥l∞(L2(Ω)) ≲ τ,

(4.14)

∥Rψ,τ∥l∞(H1(Γ)) + ∥RΓ,τ∥l∞(H1(Γ)) ≲ τ,

∥Rψ,τ∥l∞(L2(Γ)) + ∥RΓ,τ∥l∞(L2(Γ)) ≲ τ.

_Here, the truncation errors are defined as Eq._ (4.9)-(4.12). _And the corresponding sequences are_
_denoted as {Rφ,τ},_ {Rψ,τ}, {Rµ,τ} and {RΓ,τ} with τ the time step size. _Moreover,_ _the discrete norm_
∥· ∥l[∞](·) is defined as Eq. (2.1).

Thus we can establish the estimates for the stabilized scheme as follows.

**Theorem 4.2.** _Provided that the exact solutions are sufficiently smooth or under the assumption_
_(A), then for 0 ≤_ _m ≤_ [ _[T]τ_ []][−][1][, the solution][ (][φ][m][, ψ][m][)][ of the scheme][ (3.12)][-][(3.17)][ satisfy the following]

_error estimates_


∥eφ,τ∥l∞(H1(Ω)) + ∥eψ,τ∥l∞(H1(Γ)) ≲ τ,
(4.15)

∥eφ,τ∥l∞(L2(Ω)) + ∥eψ,τ∥l∞(L2(Γ)) ≲ τ.


-----

10 Xuelian Bao, Hui Zhang

_Here, the error functions are defined as_

_e[n]φ_ [=][ φ][(][t][n][)][ −] [φ][n][,] _e[n]µ_ [=][ µ][(][t][n][)][ −] [µ][n][,]


(4.16)


_e[n]ψ_ [=][ ψ][(][t][n][)][ −] [ψ][n][,] _e[n]Γ_ [=][ µ][Γ][(][t][n][)][ −] [µ]Γ[n][,]

_e[n]φ[|][Γ]_ [=][ e][n]ψ[.]


_The_ _corresponding_ _sequences_ _of_ _error_ _functions_ _are_ _denoted_ _as_ _eφ,τ,_ _eψ,τ,_ _eµ,τ_ _and_ _eΓ,τ,_ _and_ _the_
_discrete norm ∥· ∥l[∞](·) is defined as Eq._ (2.1).

_Proof._ We use the mathematical induction to prove this theorem. When m = 0, we have e[0]φ [=][ e]ψ[0] [=]
∇e[0]φ [=] [∇][Γ][e][0]ψ [=] [0.] [Obviously, (4.15) holds.] [Assuming that the error estimate holds for all][ n] [≤] _[m][,]_
we need to show that the error estimate holds for e[m]φ [+][1] and e[m]ψ [+][1]. For each n ≤ _m, by subtracting_
(4.3)-(4.8) from the corresponding scheme (3.12)-(3.17), we derive the error equations as follows,

1
(4.17) τ [(][e]φ[n][+][1] − _e[n]φ[)][ = ∆][e]µ[n][+][1]_ + R[n]φ[+][1], in Ω,

(4.18) _e[n]µ[+][1]_ = −ε∆e[n]φ[+][1] + [1]ε [(][F][′][(][φ][(][t][n][))][ −] _[F][′][(][φ][n][))][ +][ s][1][(][e]φ[n][+][1]_ − _e[n]φ[)][ +][ R]µ[n][+][1],_ in Ω,

(4.19) ∂ne[n]µ[+][1] = 0, on Γ,

(4.20) _e[n]φ[+][1]|Γ_ = e[n]ψ[+][1][,] on Γ,

1
(4.21) ψ − _e[n]ψ[)][ = ∆][Γ][e][n]Γ[+][1]_ + R[n]ψ[+][1][,] on Γ,
τ [(][e][n][+][1]

_e[n]Γ[+][1]_ = −δκ∆Γe[n]ψ[+][1] + [1]δ [(][G][′][(][ψ][(][t][n][))][ −] _[G][′][(][ψ][n][))][ +][ ε∂][n][e]φ[n][+][1]_

(4.22) +s2(e[n]ψ[+][1] − _e[n]ψ[)][ +][ R][n]Γ[+][1],_ on Γ.

By taking the L[2] inner product of (4.17) with τe[n]µ[+][1] in Ω, we obtain

(e[n]φ[+][1] − _e[n]φ[,][ e]µ[n][+][1])Ω_ + τ∥∇e[n]µ[+][1]∥[2]Ω [=][ τ][(][R]φ[n][+][1], e[n]µ[+][1])Ω.

By taking the L[2] inner product of (4.17) with ετe[n]φ[+][1] in Ω, we obtain

ε
φ ∥[2]Ω [−∥][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω[)][ =][ −][ετ][(][∇][e]µ[n][+][1], ∇e[n]φ[+][1])Ω_ + ετ(R[n]φ[+][1], e[n]φ[+][1])Ω,
2 [(][∥][e][n][+][1]

where the boundary terms vanish due to ∂ne[n]µ[+][1] = 0. By taking the L[2] inner product of (4.18) with
−(e[n]φ[+][1] − _e[n]φ[) in][ Ω][, we obtain]_

− (e[n]µ[+][1], e[n]φ[+][1] − _e[n]φ[)][Ω]_ [+] [ε] φ ∥[2]Ω [−∥∇][e]φ[n][∥][2]Ω [+][ ∥∇][e]φ[n][+][1] −∇e[n]φ[∥][2]Ω[)][ +][ s][1][∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω_ [=]

2 [(][∥∇][e][n][+][1]

ε(∂ne[n]φ[+][1], e[n]φ[+][1] − _e[n]φ[)][Γ][ −]_ [1] φ − _e[n]φ[)][Ω]_ [−] [(][R]µ[n][+][1], e[n]φ[+][1] − _e[n]φ[)][Ω][.]_

ε [(][F][′][(][φ][(][t][n][))][ −] _[F][′][(][φ][n][)][,][ e][n][+][1]_


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 11

By combining the equations above, we derive


(4.23)


ε
2 [(][∥∇][e]φ[n][+][1]∥[2]Ω [−∥∇][e]φ[n][∥][2]Ω [+][ ∥∇][e]φ[n][+][1] −∇e[n]φ[∥][2]Ω[)][ +][ s][1][∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω_

+ 2[ε] [(][∥][e]φ[n][+][1]∥[2]Ω [−∥][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω[)][ +][ τ][∥∇][e]µ[n][+][1]∥[2]Ω_

= τ(R[n]φ[+][1], e[n]µ[+][1])Ω − ετ(∇e[n]µ[+][1], ∇e[n]φ[+][1])Ω + ετ(R[n]φ[+][1], e[n]φ[+][1])Ω


− (R[n]µ[+][1], e[n]φ[+][1] − _e[n]φ[)][Ω]_ [+][ ε][(][∂][n][e][n]φ[+][1], e[n]φ[+][1] − _e[n]φ[)][Γ][ −]_ [1] φ − _e[n]φ[)][Ω][.]_

ε [(][F][′][(][φ][(][t][n][))][ −] _[F][′][(][φ][n][)][,][ e][n][+][1]_

For the boundary term, by taking the L[2] inner product of (4.21) with τe[n]Γ[+][1] on Γ, we obtain

(e[n]ψ[+][1] − _e[n]ψ[,][ e][n]Γ[+][1])Γ + τ∥∇Γe[n]Γ[+][1]∥[2]Γ_ [=][ τ][(][R]ψ[n][+][1][,][ e]Γ[n][+][1])Γ.

By taking the L[2] inner product of (4.21) with δκτe[n]ψ[+][1] on Γ, we obtain

δκ

ψ [∥]Γ[2] [−∥][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[)][ =][ −][δκτ][(][∇][Γ][e][n]Γ[+][1], ∇Γe[n]ψ[+][1][)][Γ][ +][ δκτ][(][R]ψ[n][+][1][,][ e]ψ[n][+][1][)][Γ][,]_
2 [(][∥][e][n][+][1]

where the boundary terms vanish due to Γ is closed. By taking the L[2] inner product of (4.22) with
−(e[n]ψ[+][1] − _e[n]ψ[) on][ Γ][, we obtain]_


− (e[n]Γ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ][ +]_ [δκ] ψ [∥]Γ[2] [−∥∇][Γ][e]ψ[n] [∥][2]Γ

2 [(][∥∇][Γ][e][n][+][1]

+ ∥∇Γe[n]ψ[+][1] −∇Γe[n]ψ[∥][2]Γ[)][ +][ s][2][∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_

= −ε(∂ne[n]φ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ][ −]_ [1]δ [(][G][′][(][ψ][(][t][n][))][ −] _[G][′][(][ψ][n][)][,][ e]ψ[n][+][1]_ − _e[n]ψ[)][Γ][ −]_ [(][R][n]Γ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ][.]_

By combining the equations above, we derive


(4.24)


δκ

ψ [∥]Γ[2] [−∥∇][Γ][e]ψ[n] [∥][2]Γ [+][ ∥∇][Γ][e]ψ[n][+][1] −∇Γe[n]ψ[∥][2]Γ[)][ +][ s][2][∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_
2 [(][∥∇][Γ][e][n][+][1]

+ [δκ] ψ [∥]Γ[2] [−∥][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[)][ +][ τ][∥∇][Γ][e][n]Γ[+][1]∥[2]Γ_

2 [(][∥][e][n][+][1]

= τ(R[n]ψ[+][1][,][ e]Γ[n][+][1])Γ − δκτ(∇Γe[n]Γ[+][1], ∇Γe[n]ψ[+][1][)][Γ][ +][ τδκ][(][R]ψ[n][+][1][,][ e]ψ[n][+][1][)][Γ]

− (R[n]Γ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ][ −]_ [ε][(][∂][n][e][n]φ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ][ −]_ [1]δ [(][G][′][(][ψ][(][t][n][))][ −] _[G][′][(][ψ][n][)][,][ e]ψ[n][+][1]_ − _e[n]ψ[)][Γ][.]_


-----

12 Xuelian Bao, Hui Zhang

By combining (4.23) and (4.24) together, we derive


(4.25)


ε
φ ∥[2]Ω [−∥∇][e]φ[n][∥][2]Ω [+][ ∥∇][e]φ[n][+][1] −∇e[n]φ[∥][2]Ω[)][ +] [ε] φ ∥[2]Ω [−∥][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω[)]_
2 [(][∥∇][e][n][+][1] 2 [(][∥][e][n][+][1]

+ [δκ] ψ [∥]Γ[2] [−∥∇][Γ][e]ψ[n] [∥][2]Γ [+][ ∥∇][Γ][e]ψ[n][+][1] −∇Γe[n]ψ[∥][2]Γ[)]

2 [(][∥∇][Γ][e][n][+][1]

+ [δκ] ψ [∥]Γ[2] [−∥][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[)][ +][ s][1][∥][e]φ[n][+][1]_ − _e[n]φ[∥][2]Ω_ [+][ s][2][∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_

2 [(][∥][e][n][+][1]

+ τ∥∇e[n]µ[+][1]∥[2]Ω [+][ τ][∥∇][Γ][e]Γ[n][+][1]∥[2]Γ
= τ(R[n]φ[+][1], e[n]µ[+][1])Ω + τ(R[n]ψ[+][1][,][ e]Γ[n][+][1])Γ (:= term A1)

− (R[n]µ[+][1], e[n]φ[+][1] − _e[n]φ[)][Ω]_ [−] [(][R][n]Γ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ]_ (:= term A2)

− ετ(∇e[n]µ[+][1], ∇e[n]φ[+][1])Ω − δκτ(∇Γe[n]Γ[+][1], ∇Γe[n]ψ[+][1][)][Γ] (:= term A3)

+ ετ(R[n]φ[+][1], e[n]φ[+][1])Ω + τδκ(Rψ[n][+][1][,][ e]ψ[n][+][1][)][Γ] (:= term A4)


− [1] φ − _e[n]φ[)][Ω]_ [−] [1] ψ − _e[n]ψ[)][Γ]_

ε [(][F][′][(][φ][(][t][n][))][ −] _[F][′][(][φ][n][)][,][ e][n][+][1]_ δ [(][G][′][(][ψ][(][t][n][))][ −] _[G][′][(][ψ][n][)][,][ e][n][+][1]_

(:= term A5)

For simplicity, we define H[n] = _F[′](φ(t[n])) −_ _F[′](φ[n])._ It can be rewritten as


(4.26) _H[n]_ = e[n]φ


� 1

_F[′′](sφ(t[n]) + (1 −_ _s)φ[n])ds._
0


We have ∥H[n]∥Ω ≲ ∥e[n]φ[∥][Ω] [since][ F][′′][ is bounded.] [By taking the gradient of][ H][n][, we have]

(4.27) ∇H[n] = _F[′′](φ(t[n]))∇φ(t[n]) −_ _F[′′](φ[n])∇φ[n]_ = (F[′′](φ(t[n])) − _F[′′](φ[n]))∇φ(t[n]) + F[′′](φ[n])∇e[n]φ_

Since F[′′] is bounded and Lipschitz and assumption (A), we have

∥∇H[n]∥Ω ≲ ∥e[n]φ[∥][Ω][∥][φ][(][t][n][)][∥]H[3](Ω) [+][ ∥∇][e]φ[n][∥][Ω]
(4.28)

≲ ∥e[n]φ[∥][Ω] [+][ ∥∇][e][n]φ[∥][Ω][.]

Similarly, we define _H[˜]_ _[n]_ = _G[′](ψ(t[n])) −_ _G[′](ψ[n]) for simplicity._ Since G[′′] is bounded and Lipschitz
and assumption (A), we have


∥H[˜] _[n]∥Γ_ ≲ ∥e[n]ψ[∥][Γ][,]
(4.29)

∥∇ΓH[˜] _[n]∥Γ_ ≲ ∥e[n]ψ[∥][Γ][ +][ ∥∇][Γ][e][n]ψ[∥][Γ][.]


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 13

For the term A1, we have

τ(R[n]φ[+][1], e[n]µ[+][1])Ω + τ(R[n]ψ[+][1][,][ e]Γ[n][+][1])Γ


(4.30)


= τ(R[n]φ[+][1], −ε∆e[n]φ[+][1] + [1] φ − _e[n]φ[)][ +][ R]µ[n][+][1])Ω_

ε _[H][n][ +][ s][1][(][e][n][+][1]_

+ τ(R[n]ψ[+][1][,][ −][δκ][∆][Γ][e]ψ[n][+][1] + [1] _H˜_ _[n]_ + ε∂ne[n]φ[+][1] + s2(e[n]ψ[+][1] − _e[n]ψ[)][ +][ R][n]Γ[+][1])Γ_

δ

= ετ(∇R[n]φ[+][1], ∇e[n]φ[+][1])Ω + [τ] φ )Ω + s1τ(R[n]φ[+][1], e[n]φ[+][1] − _e[n]φ[)][Ω]_

ε [(][H][n][,][ R][n][+][1]

+ τ(R[n]φ[+][1], R[n]µ[+][1])Ω + τδκ(∇ΓR[n]ψ[+][1][,][ ∇][Γ][e]ψ[n][+][1][)][Γ][ +] [τ] _[H][˜]_ _[n][,][ R][n]ψ[+][1][)][Γ]_

δ [(]

+ s2τ(R[n]ψ[+][1][,][ e]ψ[n][+][1] − _e[n]ψ[)][Γ][ +][ τ][(][R][n]ψ[+][1][,][ R]Γ[n][+][1])Γ_

≤ ετ∥∇R[n]φ[+][1]∥Ω∥∇e[n]φ[+][1]∥Ω + [τ] φ ∥Ω + s1τ∥R[n]φ[+][1]∥Ω∥e[n]φ[+][1] − _e[n]φ[∥][Ω]_

ε [∥][H][n][∥][Ω][∥][R][n][+][1]

+ τ∥R[n]φ[+][1]∥Ω∥R[n]µ[+][1]∥Ω + τδκ∥∇ΓR[n]ψ[+][1][∥][Γ][∥∇][Γ][e]ψ[n][+][1][∥][Γ][ +] [τ] ψ [∥][Γ]

δ [∥][H][˜] _[n][∥][Γ][∥][R][n][+][1]_

+ s2τ∥R[n]ψ[+][1][∥][Γ][∥][e]ψ[n][+][1] − _e[n]ψ[∥][Γ][ +][ τ][∥][R][n]ψ[+][1][∥][Γ][∥][R]Γ[n][+][1]∥Γ_

≤ [ετ] φ ∥[2]Ω [+] [ετ] φ ∥[2]Ω [+] [τ] Ω [+] [τ] φ ∥[2]Ω

2 [∥∇][R][n][+][1] 2 [∥∇][e][n][+][1] 2ε [∥][H][n][∥][2] 2ε [∥][R][n][+][1]

+ _[s][1][τ]_ φ ∥[2]Ω [+] _[s][1][τ]_ φ − _e[n]φ[∥][2]Ω_ [+] [τ] φ ∥[2]Ω [+] [τ] µ ∥[2]Ω

2 [∥][R][n][+][1] 2 [∥][e][n][+][1] 2 [∥][R][n][+][1] 2 [∥][R][n][+][1]

+ [τδκ]2 [∥∇][Γ][R]ψ[n][+][1][∥]Γ[2] [+] [τδκ]2 [∥∇][Γ][e]ψ[n][+][1][∥]Γ[2] [+] 2[τ]δ [∥][H][˜] _[n][∥]Γ[2]_ [+] 2[τ]δ [∥][R]ψ[n][+][1][∥]Γ[2]

+ _[s]2[2][τ]_ [∥][R]ψ[n][+][1][∥]Γ[2] [+] _[s]2[2][τ]_ [∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_ [+] 2[τ] [∥][R]ψ[n][+][1][∥]Γ[2] [+] 2[τ] [∥][R]Γ[n][+][1]∥[2]Γ

≤ _C1τ[3]_ + [ετ] φ ∥[2]Ω [+][ C][2][τ][∥][e]φ[n][∥][2]Ω [+] _[s][1][τ]_ φ − _e[n]φ[∥][2]Ω_

2 [∥∇][e][n][+][1] 2 [∥][e][n][+][1]

+ [τδκ]2 [∥∇][Γ][e][n]ψ[+][1][∥]Γ[2] [+][ C][3][τ][∥][e]ψ[n] [∥][2]Γ [+] _[s]2[2][τ]_ [∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[,]_


where Ci (i = 1, 2, 3) are constants independent of τ. Here, we use the estimates for _H[n]_ and _H[˜]_ _[n]_

and the truncation terms R[n]φ[+][1], R[n]ψ[+][1][,][ R]µ[n][+][1] and R[n]Γ[+][1].

For the terms in A2, we have

_e[n]φ[+][1]_ − _e[n]φ_
− (R[n]µ[+][1], e[n]φ[+][1] − _e[n]φ[)][Ω]_ [=][ −][τ][(][R]µ[n][+][1], )Ω
τ

= −τ(R[n]µ[+][1], ∆e[n]µ[+][1] + Rφ[n][+][1])Ω = τ(∇R[n]µ[+][1], ∇e[n]µ[+][1])Ω − τ(R[n]µ[+][1], R[n]φ[+][1])Ω


(4.31)


≤ τ∥∇R[n]µ[+][1]∥Ω∥∇e[n]µ[+][1]∥Ω + τ∥R[n]µ[+][1]∥Ω∥R[n]φ[+][1]∥Ω

≤ 2τ∥∇R[n]µ[+][1]∥[2]Ω [+] 8[τ] [∥∇][e]µ[n][+][1]∥[2]Ω [+] 2[τ] [∥][R]µ[n][+][1]∥[2]Ω [+] 2[τ] [∥][R]φ[n][+][1]∥[2]Ω

≤ _C4τ[3]_ + [τ] µ ∥[2]Ω[,]

8 [∥∇][e][n][+][1]


-----

14 Xuelian Bao, Hui Zhang

and

_e[n]ψ[+][1]_ − _e[n]ψ_
− (R[n]Γ[+][1], e[n]ψ[+][1] − _e[n]ψ[)][Γ]_ [=][ −][τ][(][R][n]Γ[+][1], τ )Γ

= −τ(R[n]Γ[+][1], ∆Γe[n]Γ[+][1] + R[n]ψ[+][1][)][Γ] [=][ τ][(][∇][Γ][R][n]Γ[+][1], ∇Γe[n]Γ[+][1])Γ − τ(R[n]Γ[+][1], R[n]ψ[+][1][)][Γ]

(4.32)

≤ 2τ∥∇ΓR[n]Γ[+][1]∥[2]Γ [+] [τ] Γ ∥[2]Γ [+] [τ] Γ ∥[2]Γ [+] [τ] ψ [∥]Γ[2]

8 [∥∇][Γ][e][n][+][1] 2 [∥][R][n][+][1] 2 [∥][R][n][+][1]

≤ _C5τ[3]_ + 8[τ] [∥∇][Γ][e]Γ[n][+][1]∥[2]Γ[,]

where Ci (i = 4, 5) are constants independent of τ. Here, we use the estimates for the truncation
terms R[n]φ[+][1], R[n]ψ[+][1][,][ R][n]µ[+][1] and R[n]Γ[+][1].
We estimate A3 as follows

− ετ(∇e[n]µ[+][1], ∇e[n]φ[+][1])Ω − δκτ(∇Γe[n]Γ[+][1], ∇Γe[n]ψ[+][1][)][Γ]

(4.33) ≤ ετ∥∇e[n]µ[+][1]∥Ω∥∇e[n]φ[+][1]∥Ω + δκτ∥∇Γe[n]Γ[+][1]∥Γ∥∇Γe[n]ψ[+][1][∥][Γ]

≤ 2ε[2]τ∥∇e[n]φ[+][1]∥[2]Ω [+] [τ] µ ∥[2]Ω [+][ 2][δ][2][κ][2][τ][∥∇][Γ][e]ψ[n][+][1][∥]Γ[2] [+] [τ] Γ ∥[2]Γ[.]

8 [∥∇][e][n][+][1] 8 [∥∇][Γ][e][n][+][1]

For the term A4, we have

ετ(R[n]φ[+][1], e[n]φ[+][1])Ω + τδκ(R[n]ψ[+][1][,][ e]ψ[n][+][1][)][Γ]

(4.34) ≤ ετ∥R[n]φ[+][1]∥Ω∥e[n]φ[+][1]∥Ω + τδκ∥R[n]ψ[+][1][∥][Γ][∥][e]ψ[n][+][1][∥][Γ]

≤ _C6ετ[3]_ + [ετ]2 [∥][e]φ[n][+][1]∥[2]Ω [+] [δκτ]2 [3] + [δκτ]2 [∥][e]ψ[n][+][1][∥]Γ[2][.]


Here, C6 is a constant independent of τ and we use the estimates for the truncation terms R[n]φ[+][1] and
_R[n]ψ[+][1][.]_
For the terms in A5, we have

− [1] φ − _e[n]φ[)][Ω]_

ε [(][F][′][(][φ][(][t][n][))][ −] _[F][′][(][φ][n][)][,][ e][n][+][1]_

_e[n]φ[+][1]_ − _e[n]φ_

= − ε[τ] [(][H][n][,] τ )Ω = − ε[τ] [(][H][n][,][ ∆][e]µ[n][+][1] + R[n]φ[+][1])Ω

(4.35) = ε[τ] [(][∇][H][n][,][ ∇][e]µ[n][+][1])Ω − ε[τ] [(][H][n][,][ R]φ[n][+][1])Ω

≤ ε[τ] [∥∇][H][n][∥][Ω][∥∇][e]µ[n][+][1]∥Ω + ε[τ] [∥][H][n][∥][Ω][∥][R]φ[n][+][1]∥Ω

≤ _C7τ(∥e[n]φ[∥][Ω]_ [+][ ∥∇][e][n]φ[∥][Ω][)][∥∇][e]µ[n][+][1]∥Ω + C8τ∥e[n]φ[∥][Ω][∥][R][n]φ[+][1]∥Ω

≤ _C9τ∥e[n]φ[∥][2]Ω_ [+][ 2][C]7[2][τ][∥∇][e]φ[n][∥][2]Ω [+] [τ] µ ∥[2]Ω [+][ C][10][τ][3][,]

4 [∥∇][e][n][+][1]

where Ci (i = 7, 8, 9, 10) are constants independent of τ and C9 = 2C7[2] [+][ C][8][/][2.] [Here, we use the]
estimates for H[n], ∇H[n] and R[n]φ[+][1].


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 15


(4.36)


− [1] ψ − _e[n]ψ[)][Γ]_

δ [(][G][′][(][ψ][(][t][n][))][ −] _[G][′][(][ψ][n][)][,][ e][n][+][1]_

_e[n]ψ[+][1]_ − _e[n]ψ_

= − [τ] _[H][˜]_ _[n][,]_ )Γ = − [τ] _[H][˜]_ _[n][,][ ∆][Γ][e][n]Γ[+][1]_ + R[n]ψ[+][1][)][Γ]

δ [(] τ δ [(]

= [τ] _[H][˜]_ _[n][,][ ∇][Γ][e][n]Γ[+][1])Γ −_ [τ] _[H][˜]_ _[n][,][ R][n]ψ[+][1][)][Γ]_

δ [(][∇][Γ] δ [(]

≤ [τ] _[H][˜]_ _[n][∥][Γ][∥∇][Γ][e][n]Γ[+][1]∥Γ +_ [τ] ψ [∥][Γ]

δ [∥∇][Γ] δ [∥][H][˜] _[n][∥][Γ][∥][R][n][+][1]_

≤ _C11τ(∥e[n]ψ[∥][Γ][ +][ ∥∇][Γ][e][n]ψ[∥][Γ][)][∥∇][Γ][e][n]Γ[+][1]∥Γ + C12τ∥e[n]ψ[∥][Γ][∥][R][n]ψ[+][1][∥][Γ]_

≤ _C13τ∥e[n]ψ[∥][2]Γ_ [+][ 2][C]11[2] [τ][∥∇][Γ][e]ψ[n] [∥][2]Γ [+] [τ] Γ ∥[2]Γ [+][ C][14][τ][3]

4 [∥∇][Γ][e][n][+][1]


where Ci (i = 11, 12, 13, 14) are constants independent of τ and C13 = 2C11[2] [+] _[C][12][/][2.]_ [Here, we use]
the estimates for _H[˜]_ _[n], ∇ΓH[˜]_ _[n]_ and R[n]ψ[+][1][.]
Combine (4.25) with (4.30), (4.31), (4.32), (4.33), (4.34), (4.35) and (4.36), we derive


(4.37)


ε
φ ∥[2]Ω [−∥∇][e]φ[n][∥][2]Ω [+][ ∥∇][e]φ[n][+][1] −∇e[n]φ[∥][2]Ω[)][ +] [ε] φ ∥[2]Ω [−∥][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω[)]_
2 [(][∥∇][e][n][+][1] 2 [(][∥][e][n][+][1]

+ [δκ] ψ [∥]Γ[2] [−∥∇][Γ][e]ψ[n] [∥][2]Γ [+][ ∥∇][Γ][e]ψ[n][+][1] −∇Γe[n]ψ[∥][2]Γ[)]

2 [(][∥∇][Γ][e][n][+][1]

+ [δκ] ψ [∥]Γ[2] [−∥][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[)][ +][ s][1][∥][e]φ[n][+][1]_ − _e[n]φ[∥][2]Ω_ [+][ s][2][∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_

2 [(][∥][e][n][+][1]

+ [τ] µ ∥[2]Ω [+][ ∥∇][Γ][e]Γ[n][+][1]∥[2]Γ[)]

2 [(][∥∇][e][n][+][1]

≲ τ[3] + τ(∥∇e[n]φ[+][1]∥[2]Ω [+][ ∥∇][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1]∥[2]Ω [+][ ∥][e]φ[n][∥][2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω_
+ ∥∇Γe[n]ψ[+][1][∥]Γ[2] [+][ ∥∇][Γ][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1][∥]Γ[2] [+][ ∥][e]ψ[n] [∥][2]Γ [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ[)][.]_


Summing (4.37) together for n = 0 to m, we derive


(4.38)


ε
2 [∥∇][e]φ[m][+][1]∥[2]Ω [+] 2[ε] [∥][e]φ[m][+][1]∥[2]Ω [+] [δκ]2 [∥∇][Γ][e]ψ[m][+][1]∥[2]Γ [+] [δκ]2 [∥][e]ψ[m][+][1]∥[2]Γ

_m_
� � ε

+ φ −∇e[n]φ[∥][2]Ω [+][ (] [ε] φ − _e[n]φ[∥][2]Ω_

2 [∥∇][e][n][+][1] 2 [+][ s][1][)][∥][e][n][+][1]

_n=0_

+ [δκ] ψ −∇Γe[n]ψ[∥][2]Γ [+][ (] [δκ] ψ − _e[n]ψ[∥][2]Γ_

2 [∥∇][Γ][e][n][+][1] 2 [+][ s][2][)][∥][e][n][+][1]

�

+ [τ] µ ∥[2]Ω [+][ ∥∇][Γ][e]Γ[n][+][1]∥[2]Γ[)]

2 [(][∥∇][e][n][+][1]

_m_
� �

≤ _C[˜](m + 1)τ[3]_ + _C[˜]τ_ ∥∇e[n]φ[+][1]∥[2]Ω [+][ ∥][e]φ[n][+][1]∥[2]Ω [+][ ∥][e]φ[n][+][1] − _e[n]φ[∥][2]Ω_

_n=0_


�
+ ∥∇Γe[n]ψ[+][1][∥]Γ[2] [+][ ∥][e]ψ[n][+][1][∥]Γ[2] [+][ ∥][e]ψ[n][+][1] − _e[n]ψ[∥][2]Γ_,

where we use e[0]φ [=][ e]ψ[0] [=][ ∇][e]φ[0] [=][ ∇][Γ][e]ψ[0] [=][ 0.]


-----

16 Xuelian Bao, Hui Zhang

Denote

_Im_ = 2[ε] [∥∇][e]φ[m][+][1]∥[2]Ω [+] 2[ε] [∥][e]φ[m][+][1]∥[2]Ω [+] [δκ]2 [∥∇][Γ][e]ψ[m][+][1]∥[2]Γ [+] [δκ]2 [∥][e]ψ[m][+][1]∥[2]Γ
(4.39)

+ ( [ε] φ − _e[m]φ_ [∥][2]Ω [+][ (] [δκ] ψ − _e[m]ψ_ [∥][2]Γ

2 [+][ s][1][)][∥][e][m][+][1] 2 [+][ s][2][)][∥][e][m][+][1]


and

_m_
� � ε

_S m_ = 2 [∥∇][e]φ[n][+][1] −∇e[n]φ[∥][2]Ω [+] [δκ]2 [∥∇][Γ][e]ψ[n][+][1] −∇Γe[n]ψ[∥][2]Γ
(4.40) _n=0_

�

+ [τ] µ ∥[2]Ω [+][ ∥∇][Γ][e]Γ[n][+][1]∥[2]Γ[)] .

2 [(][∥∇][e][n][+][1]


Then we have

(4.41) _Im + S m_ ≲ τ[2] + τ


_m_
�

_In._
_n=0_


According to the discrete Gronwall’s inequality, there exists constants _c˜0 and C0, such that_

(4.42) _Im + S m_ ≤ _c˜0τ[2],_

where _c˜0_ is independent of τ and τ ≤ _C0._ And thus the error estimate (4.15) holds for _e[m]φ_ [+][1] and
_e[m]ψ_ [+][1]. 
**Remark 4.3.** When the surface diffusion is absent, namely, when κ = 0, the scheme (3.12)-(3.17)
is also valid and the energy stability and error estimates also hold. In this case, we only need to
let κ = 0 in Eq. (3.17) to get the corresponding numerical scheme. Moreover, the proof for the
stability and error estimates are similar to those mentioned above. The only difference is to let
κ = 0 in the proof. Thus, we omit the details here and leave it to interested readers.

## 5 Numerical simulations

In this section, we present numerical experiments of the Liu-Wu model by implementing the
developed scheme (3.12)-(3.17). The numerical examples include the comparison with the numerical results in [15] and [8], accuracy tests with respect to the time step size, and the simulation of
the shape deformation of a square shaped droplet.
The discrete energy and mass are defined as


_E(φ[n], ψ[n]) =_ _Ebulk(φ[n]) + Esur f (ψ[n])_
(5.1) = � 1 [ε] �

Ω ε _[F][(][φ][n][)][ +]_ 2 [|∇][φ][n][|][2][d][x][ +]


Γ


1 [δκ]
δ _[G][(][ψ][n][)][ +]_ 2 [|∇][Γ][ψ][n][|][2][d][S] [,]


_M(φ[n], ψ[n]) =_ _M[bulk](φ[n]) + M_ _[sur f]_ (ψ[n])
(5.2) � �

= φ[n]dx + ψ[n]dS .

Ω Γ


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 17

Figure 1: The initial data of (5.3) and (5.4).

The time evolutions of energy and mass are plotted in this section to validate the stability of the
numerical scheme and the conservation of mass.
In this section, we present the numerical simulations in two dimensions. For the spatial operators, we use the second-order central finite difference method to discretize them over a uniform
spatial grid.

### 5.1 Comparison with former work

We reconstruct the numerical experiments in [15] and [8] to validate the accuracy and robustness of our scheme.

**5.1.1** **Comparison with numerical experiments in [15]**

Firstly, we consider the initial condition


(5.3) φ0(x, y) =


1 if _x > 1/2,_

− 1 if _x ≤_ 1/2,




where x = (x, y) ∈ [0, 1][2], which is plotted in Fig. 1. The time step size τ = 10[−][5] and the spatial
step size _h_ = 0.01. The parameters are set the same as those in Section 7.2.1 of [15]: ε = 1,
δ = 0.1, κ = 1. _F_ and G are chosen to be the classical double-well potential (1.3). We set _s1_ = 1,
_s2_ = 10 to make sure that the scheme (3.12)-(3.17) is energy stable.
We observe that the numerical solutions are almost constant in the orthogonal direction. Thus,
the projection of the numerical solution on the line y = 1/2 after 200 time steps is plotted in Fig.
2, indicating the dissipation in the bulk. It is consistent with the results in [15]. The energy and
mass evolution with respect to time are also shown in Fig. 2, revealing the energy stability and the
mass conservation both in the bulk and on the boundary.
Next, we consider the initial data

(5.4) φ0(x, y) = sin(4πx) cos(4πy),

where **x** = (x, y) ∈ [0, 1][2], which is also plotted in Fig. 1. The time step size τ = 10[−][5] and the
spatial step size h = 0.01.


-----

18 Xuelian Bao, Hui Zhang

Figure 2: For the initial data (5.3), the projection of the numerical solution on the line _y_ = 1/2
after 200 time steps (left), energy evolution (middle) and mass evolution (right).

Figure 3: Numerical results of Liu-Wu model with the initial data (5.4) after 100 time steps (left),
energy evolution (middle) and mass evolution (right).

The parameters are set the same as those in Section 7.2.2 of [15]: ε = δ = 0.02, κ = 1. _F_ and
_G are chosen to be the classical double-well potential (1.3)._ We set s1 = _s2_ = 50 to make sure that
the scheme (3.12)-(3.17) is energy stable. The numerical solution after 100 time steps is plotted in
Fig. 3, which is qualitatively consistent with the numerical results in [15]. The energy and mass
evolutions are also plotted in Fig. 3, showing the energy stability of the numerical scheme and the
mass conservation both in the bulk and on the boundary.

**5.1.2** **Comparison with numerical experiments in [8]**

Firstly, the parameters are the same as those of the first numerical simulation in [8]: ε = δ =
0.02, κ = 0.02. Ω is the unit square and the spatial step size h = 0.01. _F_ and G are chosen to be
the classical double-well potential (1.3), and we set _s1_ = _s2_ = 100 to make sure that the scheme
(3.12)-(3.17) is energy stable. The time step size is set as τ = 8 × 10[−][6]. The initial data φ0 is set to
be zero in the bulk and set to be one on the boundary.
The energy evolution and the mass evolutions are plotted in Fig. 4 and 5. The energy decreases
with respect to time, indicating the stability of the scheme. The masses in the bulk and on the
boundary are conserved respectively, which is consistent with the properties of Liu-Wu model.
The numerical solutions after 5, 15, 80, 200, 500 and 2500 time steps are shown in Fig. 6. Due
to the mass conservation on the boundary, the numerical solution remain to be 1 on the boundary.
A wave-like structure arises during the phase separation in the bulk, and ultimately, a circle of the
phase of value -1 appears at the center of Ω. The numerical results are consistent with the former
work [8].


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 19

Figure 4: Energy evolution of Liu-Wu model with the initial data of 0 in the bulk and 1 on the
boundary (left) and the initial data of random values between -0.1 and 0.1 in the bulk and random
values between 0.4 and 0.6 on the boundary (right).

Figure 5: Mass evolution of Liu-Wu model with the initial data of 0 in the bulk and 1 on the
boundary: mass in the bulk(left), mass on the boundary(middle) and total mass(right).

Figure 6: Numerical results of Liu-Wu model with the initial data of 0 in the bulk and 1 on the
boundary.


-----

20 Xuelian Bao, Hui Zhang

Figure 7: Mass evolution of Liu-Wu model with the initial data of random values between -0.1 and
0.1 in the bulk and random values between 0.4 and 0.6 on the boundary: mass in the bulk(left),
mass on the boundary(middle) and total mass(right).

Figure 8: Numerical results of Liu-Wu model with the initial data of random values between -0.1
and 0.1 in the bulk and random values between 0.4 and 0.6 on the boundary.

Secondly, the parameters are the same as those of the second numerical simulation in [8]:
ε = δ = 0.02, κ = 0.075, Ω= [0.5, 0.5][2] and the spatial step size h = 0.005. The initial data is set
as random values between -0.1 and 0.1 in the bulk and random values between 0.4 and 0.6 on the
boundary. The numerical solutions after 5, 15, 50, 150, 300 and 3000 time steps are shown in Fig.
8. The energy evolution and the mass evolutions are plotted in Fig. 4 and 7. The numerical results
are consistent with [8].

### 5.2 Accuracy test

We present in this section numerical accuracy test using the scheme (3.12)-(3.17) to support
our error analysis. Let Ω to be the unit square, the spatial step size _h_ = 0.01 and the time steps
τ = 0.1, 0.05, 0.025, 0.0125, 6.25 × 10[−][3], 3.125 × 10[−][3]. The parameters are chosen as ε = δ = 0.02,
κ = 1 and _s1_ = _s2_ = 100. The initial data is set to be zero in the bulk and set to be one on the


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 21

Figure 9: The L[2] numerical errors for φ and ψ at T = 0.5.

boundary. In this section, we choose F and G to be


(φ − 1)[2] φ > 1,
1
− 1 ≤ φ < 1,
4 [(][φ][2][ −] [1)][2]

(φ + 1)[2] φ ≤−1,


(5.5) _F(φ) = G(φ) =_





which is modified from the classical double-well potential (1.3). Thus, the Lipschitz property
holds for their derivatives

(5.6) max
φ∈R [|][F][′′][(][φ][)][|][ =][ max]ψ∈R [|][G][′′][(][ψ][)][| ≤] [2][,]

which is necessary for the error estimates.
The errors are calculated as the difference between the solution of the coarse time step and
that of the reference time step τ[∗] = 10[−][4]. In Fig. 9, we plot the _L[2]_ errors of φ and ψ between
the numerical solution and the reference solution at _T_ = 0.5 with different time step sizes. The
results show clearly that the convergence rate of the numerical scheme is asymptotically first-order
temporally for φ and ψ, which is consistent with our numerical analysis in Section 4.

### 5.3 Shape deformation of a droplet

In this section, we consider the domain Ω= [0, 1][2] and place a square shaped droplet with
center at (0.5, 0.25) and the length of each side to be 0.5 (see Fig. 10). The phase inside the
droplet is set to be 1 and outside the droplet to be -1. _F_ and G are chosen to be of the form (1.3).
And the parameters are set as

ε = δ = 0.02, κ = 1, _s1_ = _s2_ = 100.

We simulate the behaviour of the droplet from t = 0 to T = 0.5 with the time step τ = 2 × 10[−][4]

and the spatial step size h = 0.01.


-----

22 Xuelian Bao, Hui Zhang

Figure 10: The initial data of the square shaped droplet.

Figure 11: Numerical results of Liu-Wu model with the initial data of a square shaped droplet.

The energy evolution and the mass evolutions are shown in Fig. 12, revealing the decrease of
the total energy and the conservation of mass in the bulk and on the boundary, respectively. The
time evolution of the droplet after 10, 50, 100, 500, 1000 and 2500 time steps are plotted in Fig.
11. It’s shown that the square shaped droplet evolves to attain the circular shape with constant
mean curvature. Moreover, the contact area of the droplet and the boundary doesn’t change due to
the conservation of mass, which is consistent with the previous work [16].

### 5.4 Comparison with different potentials

In the numerical experiments mentioned above, the surface potential G is chosen polynomial.
In this section, we simulate the shape deformation of the droplet with different surface potentials.
For simplicity, we denote the classical double-well potential as G1, namely,

_G1(φ) =_ [1]

4 [(][φ][2][ −] [1)][2][.]

And we denote the typical potential for moving contact line problems as G2, namely,


_G2(φ) =_ [γ]

2 [cos(][θ][s][) sin(] [π]2 [φ][)][,]


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 23

Figure 12: Energy and mass evolution of Liu-Wu model with the initial data of a square shaped
droplet: energy evolution(left) and mass evolution(right).

Figure 13: Numerical results of Liu-Wu model with the surface potential G1.

where θs stands for the static contact angle. In this section, we simulate the shape deformation of
the droplet (with the initial data as in Fig. 10) with the surface potentials G1 and G2.

In Section 5.3, we have shown the shape deformation of the droplet, the energy and mass
evolution with the surface potential G1. In order to make comparisons between√ the two surface
potentials, we choose the same parameters as those in Section 5.3 with γ = [2] 2 and τ = 1e − 5.

3
The energy and mass evolutions of the cases with the surface potential G2 (with cos θs = ± 2[1] [) are]

shown in Fig. 16, indicating the decrease of the total energy and the conservation of mass. The
time evolution of the droplet with G1 and G2 after 50, 100, 200, 500, 800 and 1000 time steps are
plotted in Fig. 13, 14 and 15. In the case of G2, the square shaped droplet also evolves to attain the
circular shape, which is the same as the case that G is polynomial. However, note that the contact
area of the droplet and the boundary changes, which is different from the case of G1. Thus, due to
the conservation of mass both in the bulk and on the boundary (as shown in Fig. 16), the values of
the phase-field order parameter φ and ψ are not confined in [−1, +1].


-----

24 Xuelian Bao, Hui Zhang

Figure 14: Numerical results of Liu-Wu model with the surface potential G2 (cos θs = [1]2 [).]


Figure 15: Numerical results of Liu-Wu model with the surface potential G2 (cos θs = − [1]2 [).]

Figure 16: Energy evolution of Liu-Wu model with the surface potential G2 (left). Mass evolution
of Liu-Wu model with the surface potential G2: cos θs = [1]2 [(middle) and cos][ θ][s] [=][ −] [1]2 [(right).]


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 25

## 6 Conclusions

In the present work, we consider numerical approximations for the Cahn-Hilliard equation with
dynamic boundary conditions (C. Liu et. al., Arch. Rational Mech. Anal., 2019). To solve the
model, we develop an efficient scheme based on the stabilized linearly implicit approach, which
is first-order in time, linear and energy stable. The stabilization terms are used to enhance the
stability of the scheme. To the best of our knowledge, this is the first linear and energy stable
scheme for solving the Liu-Wu model. The semi-discretized-in-time error estimates for the scheme
are also derived. The energy stability and the accuracy of the developed scheme are demonstrated
numerically by constructing numerical experiments, including the comparison with the former
work, accuracy tests with respect to the time step size and the shape deformation of a droplet.

Acknowledgment

The authors would like to thank Prof. Chun Liu for some useful discussions on the subject of
this article. X. Bao is thankful to Prof. Chun Liu, Prof. Yiwei Wang and Prof. Qing Cheng for
some stimulating discussions during the visit of Illinois Institute of Technology. X. Bao is partially
supported by China Scholarship Council (No. 201906040019). H. Zhang was partially supported
by the National Natural Science Foundation of China (Nos. 11971002 and 11471046).

## References

[1] J.W. Cahn and J.E. Hilliard, Free energy of a nonuniform system I. Interfacial free energy, J.
Chem. Phys., 2:205-245, 1958.

[2] L. Cherfils, M. Petcu and M. Pierre, A numerical analysis of the Cahn-Hilliard equation with
_dynamic boundary conditions, Discrete Contin. Dyn. Syst., 27:_ 1511-1533, 2010.

[3] L. Cherfils and M. Petcu, _A_ _numerical_ _analysis_ _of_ _the_ _Cahn-Hilliard_ _equation_ _with_ _non-_
_permeable walls, Numer. Math., 128:_ 517-549, 2014.

[4] H.P. Fischer, P. Maass, and W. Dieterich, _Novel_ _Surface_ _Modes_ _in_ _Spinodal_ _Decomposition,_
Phys. Rev. Lett., 79:893-896, 1997.

[5] H.P. Fischer, J. Reinhard, W. Dieterich, J. F. Gouyet, P. Maass, A. Majhofer, and D. Reinel,
_Timedependent density functional theory and the kinetics of lattice gas systems in contact with_
_a wall, J. Chem. Phys., 108(7):3028-3037, 1998._

[6] J. Forster, _Mathematical_ _Modeling_ _of_ _Complex_ _Fluids,_ Master’s Thesis, University of
W¨urzburg, 2013

[7] T. Fukao, S. Yoshikawa and S. Wada, _Structure-preserving_ _finite_ _difference_ _schemes_ _for_ _the_
_Cahn-Hilliard equation with dynamic boundary conditions in the one-dimensional case, Com-_
mun. Pure Applied Anal., 16: 1915-1938, 2017.


-----

26 Xuelian Bao, Hui Zhang

[8] H. Garcke and P. Knopf, Weak Solutions of the Cahn-Hilliard System with Dynamic Boundary
_Conditions:_ _A Gradient Flow Approach, SIAM J. Math. Anal., 52(1):340-369, 2020._

[9] G.R. Goldstein, A. Miranville, and G. Schimperna, A Cahn-Hilliard model in a domain with
_nonpermeable walls, Physica D, 240:754-766, 2011._

[10] G. Gr¨un, _On_ _convergent_ _schemes_ _for_ _diffuse_ _interface_ _models_ _for_ _two-phase_ _flow_ _of_ _incom-_
_pressible fluids with general mass densities, SIAM J. Numer. Anal., 51(6):_ 3036-3061, 2013.

[11] Y.N. He, Y.X. Liu and T. Tang, On large time-stepping methods for the Cahn-Hilliard equa_tion, Appl. Numer. Math., 57:_ 616-628, 2007.

[12] Y. Hyon, D. Y. Kwak, C. Liu, _Energetic variational approach in complex fluids:_ _maximum_
_dissipation principle, Discrete Contin. Dyn. Syst., 26(4):_ 1291-1304, 2010.

[13] H. Israel, A. Miranville and M. Petcu, Numerical analysis of a Cahn-Hilliard type equation
_with dynamic boundary conditions, Ricerche Mat., 64:_ 25-50, 2015.

[14] R. Kenzler, F. Eurich, P. Maass, B. Rinn, J. Schropp, E. Bohl, and W. Dietrich, _Phase sep-_
_aration_ _in_ _confined_ _geometries:_ _Solving_ _the_ _Cahn-Hilliard_ _equation_ _with_ _generic_ _boundary_
_conditions, Comp. Phys. Comm., 133:139-157, 2001._

[15] P. Knopf and K.F. Lam, _Convergence_ _of_ _a_ _Robin_ _boundary_ _approximation_ _for_ _a_ _Cahn-_
_Hilliard_ _system_ _with_ _dynamic_ _boundary_ _conditions,_ Accepted in Nonlinearity, Preprint:
[arXiv:1908.06124 [math.AP], 2019.](http://arxiv.org/abs/1908.06124)

[16] P. Knopf, K. F. Lam, C. Liu and S. Metzger, _Phase-field_ _dynamics_ _with_ _transfer_ _of_ _materi-_
_als:_ _The_ _Cahn–Hillard_ _equation_ _with_ _reaction_ _rate_ _dependent_ _dynamic_ _boundary_ _conditions,_
Preprint: [arXiv:2003.12983 [math.AP], 2020.](http://arxiv.org/abs/2003.12983)

[17] C. Liu and H. Wu, An energetic variational approach for the Cahn-Hilliard equation with dy_namic boundary condition:_ _model derivation and mathematical analysis, Arch. Ration. Mech._
Anal., 233(1):167-247, 2019.

[18] S. Metzger. _An_ _efficient_ _and_ _convergent_ _finite_ _element_ _scheme_ _for_ _Cahn-Hilliard_ _equations_
_with dynamic boundary conditions, Preprint arXiv:_ 1908.04910 [math.NA], 2019.

[19] J. Shen, C. Wang, X. M. Wang and S. M. Wise, _Second-order convex splitting schemes for_
_gradient_ _flows_ _with_ _Ehrlich-Schwoebel_ _type_ _energy:_ _application_ _to_ _thin_ _film_ _epitaxy,_ SIAM J.
Numer. Anal., 50(1): 105-125, 2012.

[20] J. Shen, J. Xu and J. Yang, The scalar auxiliary variable (SAV) approach for gradient flows,
J. Comput. Phys., 353: 407-416, 2018.

[21] P.A. Thompson and M.O. Robbins, Simulations of contact-line motion: slip and the dynamic
_contact angle, Phys. Rev. Lett., 63:766-769, 1989._

[22] D. Trautwein, _Finite-Elemente_ _Approximation_ _der_ _Cahn-Hilliard-Gleichung_ _mit_ _Neumann-_
_und dynamischen Randbedingungen, Bachelor thesis, University of Regensburg, 2018._


-----

Numerical Approximations of the Cahn-Hilliard Equation with Dynamic Boundary Conditions 27

[23] Z. Xu, X. F. Yang and H. Zhang, Error Analysis of a Decoupled, Linear Stabilization Scheme
_for_ _the_ _Cahn-Hilliard_ _Model_ _of_ _Two-Phase_ _Incompressible_ _Flows,_ J. Sci. Comput., 83: 57,
2020.

[24] X. F. Yang, Linear, first and second-order, unconditionally energy stable numerical schemes
_for the phase field model of homopolymer blends, J. Comput. Phys., 327:_ 294-316, 2016.

[25] X. F. Yang, J. Zhao and X. M. He, _Linear,_ _second order and unconditionally energy stable_
_schemes for the viscous Cahn-Hilliard equation with hyperbolic relaxation using the invariant_
_energy quadratization method, J. Comput. Appl. Math., 343:_ 80-97, 2018.


-----

