**ELLIPTIC** **CURVES** **ASSOCIATED** **WITH** **HERON** **TRIANGLES**
**WITH** **HIGH** 2-SELMER **RANK**

VINODKUMAR GHALE, MD IMDADUL ISLAM, AND DEBOPAM CHAKRABORTY


Abstract

Rank computation of elliptic curves has deep relations with various unsolved questions

in number theory, most notably in the congruent number problem for right-angled tri
angles. Similar relations between elliptic curves and Heron triangles were established

later. In this work, we explicitly compute the 2-Selmer rank of the elliptic curves associ
ated with Heron triangles of even area, which eventually sheds light on the Mordell-Weil


rank of those curves.


1. Introduction

For a given positive integer n, the congruent number problem is to determine whether a

positive integer n can be observed as the area of a right-angle triangle of rational sides.

The problem eventually boils down to the rank computation of the congruent number

elliptic curve corresponding to a positive integer n, defined as En : y[2] = x[3] − n[2]x. Rank

computation of an elliptic curve is often done by the method of m-descent, which gives

rise to the m-Selmer group. In particular, the 2-Selmer group plays an important role

in the rank computation of an elliptic curve. Works of D. R. Heath-Brown described


the 2-Selmer group structure for a congruent number elliptic curve En in [7] and [8].

A Heron triangle is a triangle with rational sides with no restrictions on the angles.

The works of Goins and Maddox in [6] resulted in an elliptic curve En,τ : y[2] = x(x −

nτ )(x + nτ [−][1]), corresponding to a Heron triangle of area n and of the angles θ such

that τ = tan [θ] [This] [curve] [is] [known] [as] [the] [Heronian] [elliptic] [curve][,] [which] [is] [the] [same]

2 [.]

as the congruent number elliptic curve when θ = [π]2 [.] [Goins] [and] [Maddox] [showed] [in] [[6]]

that infinitely many Heron triangles with area n and one of the angles θ such that

τ = tan 2[θ] [exist] [if] [the] [rank] [of] [E][n,τ] [is] [positive.] [Buchholz] [and] [Rathbun] [[1]] [proved] [the]

existence of infinitely many Heron triangles with two rational medians. The existence


2010 Mathematics Subject Classification. Primary 11G05, 11G07; Secondary 51M04.

Key words and phrases. Elliptic curve; Selmer group; Heron triangle.


1


-----

HERON TRIANGLES AND ELLIPTIC CURVES 2

of Heron triangles with three rational medians was discussed in [2]. The existence of

infinitely many Heron triangles with the area n was studied in [9]. The work [4] of

Dujella and Peral showed the existence of elliptic curves of higher ranks associated with

Heron triangles. In a recent work [5], Ghale et al. constructed a family of elliptic curves

of rank at most one from a certain Diophantine equation via Heron triangles. In [3],

the 2-Selmer group structure of the Heronian elliptic curve associated with the Heron

triangles of area 2[m] - p for primes p ≡ 7 (mod 8).

As a generalization to the triangle considered in [3], we focus on Heron triangles of area

2[m] - n for square-free odd n and with one of the angles θ such that τ = tan [θ]2 [=][ n][.] [The]

elliptic curve E : y[2] = x(x − 2[m]n[2])(x + 2[m]) is associated with such triangles (see [6]).

Let S denote the set consisting of all finite places at which E has bad reductions, the

infinite places, and the prime 2. We define

� �
(1.1) Q(S, 2) = b ∈ Q[∗]/(Q[∗])[2] : ordl(b) ≡ 0 (mod 2) for all primes l ̸∈ S

= ⟨±2, ±pi, ±q⟩

where n = p1p2...pk is a square-free odd number such that n[2] + 1 = 2q for some prime

q. By the method of 2-descent (see [10]), there exists an injective homomorphism

β : E(Q)/2E(Q) −→ Q(S, 2) × Q(S, 2)


defined by


(x, x − 2[m]n[2]) if x ̸= 0, 2[m]n[2],

(−1, −2[δ][(][m][)]) if x = 0,

(2[δ][(][m][)], 2q) if x = 2[m]n[2],

(1, 1) if x = ∞, i.e., if (x, y) = O,


β(x, y) =
























where O is the fixed base point and δ(m) = 0 for even m and δ(m) = 1 otherwise.

If (b1, b2) is a pair which is not in the image of O, (0, 0), (2[m]n[2], 0), then (b1, b2) is the

image of a point P = (x, y) ∈ E(Q)/2E(Q) if and only if the equations

(1.2) b1z1[2] [−] [b][2][z]2[2] [= 2][m][ ·][ n][2][,]

(1.3) b1z1[2] [−] [b][1][b][2][z]3[2] [=][ −][2][m][,]

(1.4) b1b2z3[2] [−] [b][2][z]2[2] [= 2][m][+1][ ·][ q.]

have a solution (z1, z2, z3) ∈ Q[∗] × Q[∗] × Q. We note that (1.4) is obtained by subtracting

(1.3) from (1.2), and is only included here due to its use later in this work. The image of


-----

HERON TRIANGLES AND ELLIPTIC CURVES 3

E(Q)/2E(Q) under the 2-descent map is contained in a subgroup of Q(S, 2) × Q(S, 2)

known as the 2-Selmer group Sel2(E/Q), which fits into an exact sequence

(1.5) 0 −→ E(Q)/2E(Q) −→ Sel2(E/Q) −→ X(E/Q)[2] −→ 0.

Throughout the work, b denotes all possible products of the prime factors of n and 1.

By p, we denote an arbitrary factor of n, unless otherwise mentioned. The main results

of this work are now as follows;

**Theorem** **1.1.** Let E : y[2] = x(x − 2[m]n[2])(x + 2[m]) be the elliptic curve corresponding

to Heron triangles of area 2[m] - n and one of the angles θ such that τ = tan [θ]2 [=] [n][.]

We assume n = p1p2..pk as a square-free odd number and n[2] + 1 = 2q for some prime

number q. Then for odd m,


⟨(2, 2)⟩ if pi ≡±3 (mod 8), q ≡ 1 (mod 8),

0 if pi ≡±3 (mod 8), q ≡ 5 (mod 8),

⟨(b, b)⟩ if pi ≡±1 (mod 8), q ≡ 5 (mod 8),

⟨(b, b), (2b, b), (2, 2)⟩ if pi ≡±1 (mod 8), q ≡ 1 (mod 8),


Sel2(E) =
























for all i ∈{1, 2, ..., k}, where (b, b) ∈ Sel2(E) implies b ≡ 1 (mod 8) and (2b, b) ∈

Sel2(E) implies b ≡ 7 (mod 8).

**Theorem** **1.2.** Let E : y[2] = x(x − 2[m]n[2])(x + 2[m]) be the Heronian elliptic curve as

mentioned above. Then for even m, Sel2(E) = ⟨(b, b), (b, 2b)⟩ where

(i) (b, b) ∈ Sel2(E) =⇒ b ≡±1 (mod 8).

(ii) (b, 2b) ∈ Sel2(E) implies every prime factor of n is of the form ±1 modulo 8.

2. l-adic solution to the homogeneous spaces

Throughout this work, we denote an l-adic solution for (1.2) and (1.3) as zi = ui - l[t][i]

for i = 1, 2, 3 where ui ∈ Z[∗]l [.] [We] [note] [that] [this] [implies] [v][l][(][z][i][)] [=] [t][i][.] [In] [this] [section,] [we]

focus on l-adic solutions that are not in Zl, i.e. ti < 0 for all i = 1, 2, 3. We start with

the following result relating t1 and t2.

**Lemma** **2.1.** Suppose (1.2) and (1.3) have a solution (z1, z2, z3) ∈ Ql × Ql × Ql for any

prime l. If vl(zi) < 0 for any one i ∈{1, 2}, then vl(z1) = vl(z2) = −t < 0 for some

integer t.


-----

HERON TRIANGLES AND ELLIPTIC CURVES 4

Proof. With the notations above, for t1 < 0, (1.2) implies b1 · u[2]1 [−] [b][2] [·][ u][2]2 [·][ l][2(][t][2][−][t][1][)] [=]

2[m] - n[2] - l[−][2][t][1] =⇒ b1 ≡ 0 (mod l[2]) if t2 - t1, a contradiction as b1 is square-free. Hence

t2 ≤ t1 < 0. Now if t2 < t1 < 0 then again from (1.2) one gets

b1 · u[2]1 [·][ l][2(][t][1][−][t][2][)][ −] [b][2] [·][ u][2]2 [= 2][m][ ·][ n][2][ ·][ l][−][2][t][2][,]

which implies l[2] must divide b2, a contradiction again. Hence if t1 < 0, then we have

t1 = t2 = −t < 0 for some integer t. For t2 < 0, one similarly gets t1 = t2 = −t < 0. 
The following result gives a correspondence between vl(z1) and vl(z3) for all primes l.

**Lemma** **2.2.** Suppose (1.2) and (1.3) have a solution (z1, z2, z3) ∈ Ql × Ql × Ql for

any prime l. Then vl(z1) < 0 =⇒ vl(z3) = vl(z1) = −t < 0.

Proof. For t1 < 0, from (1.3), we have

b1 · u[2]1 [−] [b][1][b][2] [·][ u][2]3 [·][ l][2(][t][3][−][t][1][)] [=][ −][2][m][ ·][ l][−][2][t][1] [.]

If t3 - t1, then l[2] must divide b1, a contradiction. Hence t3 ≤ t1 < 0. Now for t3 < t1 < 0,

one can see from above that

(2.1) b1 · u[2]1 [·][ l][2(][t][1][−][t][3][)][ −] [b][1][b][2] [·][ u][2]3 [=][ −][2][m][ ·][ l][−][2][t][3] [,]

which implies l[2] must divide b1b2, i.e., l = 2, p or q where p denotes any of the primes that

divide n. Noting b1, b2 are square-free, one can get that t3 ≤−2 =⇒ b1b2 ≡ 0 (mod l[3]),

a contradiction from the above equation. Hence, t3 = −1, but then t3 < t1 =⇒ t1 ≥ 0,

contradiction again as t1 < 0. Together, now we obtain t1 = t3 = −t < 0 if t1 < 0. 
**Lemma** **2.3.** Suppose (1.2) and (1.3) have a solution (z1, z2, z3) ∈ Ql × Ql × Ql for

any prime l. If p denotes an arbitrary prime factor of n, then

(i) For all primes l ̸= p, vl(z3) < 0 =⇒ vl(z3) = vl(z1). The same conclusion holds

true for l = p also if b1b2 ̸≡ 0 (mod p[2]).

(ii) For l = p, b1b2 ≡ 0 (mod p[2]), and vl(z3) < 0, either vl(z3) = −1 and vl(z1) =

vl(z2) = 0, or vl(z3) = vl(z1) = −t < 0.

Proof. (i) For vl(z3) = t3 < 0, from (1.3), we get

b1 · u[2]1 [·][ l][2(][t][1][−][t][3][)][ −] [b][1][b][2] [·][ u][2]3 [=][ −][2][m][ ·][ l][−][2][t][3] [.]

Hence t1 - t3 implies l[2] divides b1b2 i.e. l = 2, p or q. l = 2 will imply 2[3] divides

b1b2, a contradiction. For l = q, from (1.4), we again get q[3] divides b1b2 if t2 - t3, a


-----

HERON TRIANGLES AND ELLIPTIC CURVES 5

contradiction. Hence t2 ≤ t3 < 0 =⇒ t2 = t1 ≤ t3 from Lemma 2.1. Now t1 < t3 =⇒

b1 ≡ 0 (mod l[2]) from (1.3), a contradiction. Hence t3 < 0 =⇒ t3 = t1 = −t < 0. This

proves the first part of the result.

(ii) For the second part, l = p, b1b2 ≡ 0 (mod p[2]), and t1 - t3 will imply t3 = −1 from

(1.3), and hence t1 ≥ 0. This, in turn, will also imply t2 ≥ 0. As vp(2[m] - n[2]) = 2, one

can easily observe that t1, t2 ≥ 0 =⇒ t1 = t2 = 0 from (1.2). Hence for l = p and

t1 - t3 implies t3 = −1, and t1 = t2 = 0. For l = p and t1 ≤ t3 < 0 =⇒ t1 < 0, and

hence from Lemma 2.2, we conclude t1 = t3. This concludes the proof. 
The following result discusses the non-existence of l-adic solutions for different homoge
neous spaces and different l. This, in turn, makes the upper bound of the size of Sel2(E)

smaller. Without loss of generality, we assume b2 ̸≡ 0 (mod q), as (2[δ][(][m][)], 2q) belongs in

the image of the E(Q)tors under the 2-descent map β.

**Lemma** **2.4.** Let (b1, b2) ∈ Q(S, 2) × Q(S, 2). Then

(i) The corresponding homogeneous space has no l-adic solution for the case l = ∞ if

b1b2 < 0.

(ii) If b1 ≡ 0 (mod q), then the corresponding homogeneous space has no q-adic solution.

(iii) The homogeneous space corresponding to (b1, b2) has no p-adic solution if b1b2 ≡ 0

(mod p) but b1b2 ̸≡ 0 (mod p[2]).

Proof. (i) Let the homogeneous space corresponding to (b1, b2) ∈ Q(S, 2) × Q(S, 2) has

real solutions. Then b1 - 0 and b2 < 0 implies −2[m] - 0 in (1.3), which is absurd.

Similarly, b1 < 0 and b2 - 0 implies 2[m] - n[2] < 0 in (1.2), a contradiction. Thus,

the homogeneous space corresponding to (b1, b2) has no l-adic solutions for l = ∞ if

b1b2 < 0.

(ii) Let us assume b1 ≡ 0 (mod q). From (1.4), we notice that vq(zi) < 0 =⇒ b2 ≡ 0

(mod q), a contradiction. Now for vq(zi) ≥ 0, (1.3) will imply that −2[m] ≡ 0 (mod q),

a contradiction again. Hence the result follows.

(iii) Assuming b1 ≡ 0 (mod p), we get b2 ̸≡ 0 (mod p) from the given condition. If

vp(zi) < 0, from (1.2), we get b2 · u[2]2 [≡] [0] [(mod] [p][)][ where][ u][2] [∈] [Z]p[∗][, a contradiction.] [Hence]

vp(zi) ≥ 0, but then (1.3) implies −2[m] ≡ 0 (mod p), a contradiction. The case b2 ≡ 0

(mod p), b1 ̸≡ 0 (mod p) can be done in a similar manner. 

-----

HERON TRIANGLES AND ELLIPTIC CURVES 6

3. Size of the Sel2(E) when m is an odd integer

We start this section under the assumption m is odd. From Lemma (2.4), it is evident

that (b, b), (2b, b), (b, 2b), (2b, 2b) are the only possible elements of Sel2(E) where b runs

over all possible square-free combinations of prime factors of n and 1. Throughout this

section, we assume m is odd.

**Lemma** **3.1.** Let (b1, b2) ∈ Q(S, 2) × Q(S, 2). Then

(i) The homogeneous spaces corresponding to (2b, b) and (b, 2b) have no 2-adic solutions

if b ̸≡ 7 (mod 8). The homogeneous space corresponding to (2b, 2b) has no 2-adic solu
tion if b ̸≡ 1 (mod 8).

(ii) For p ≡±3 (mod 8), the homogeneous spaces corresponding to (b, b), (2b, b), (b, 2b)

and (2b, 2b) have no p-adic solutions where p is any prime factor of b.

(iii) For q ≡ 5 (mod 8), the homogeneous space corresponding to (2, 2) has no q-adic

solution. Moreover, if p ≡±1 (mod 8), then the homogeneous spaces corresponding to

(2b, b), (b, 2b) and (2b, 2b) also have no q-adic solution.

Proof. (i) For the case (2b, b), one can immediately observe v2(zi) ≥ 0, as otherwise

b ≡ 0 (mod 2) from (1.2). For v2(zi) ≥ 0, looking into the parity of v2(zi) from (1.2)

and (1.4), we get v2(z1) = [m]2[−][1] and v2(z2) = [m]2[+1] [.] [Using] [(1.2),] [this,] [in] [turn,] [implies]


b · u[2]1 [−] [2][b][ ·][ u]2[2] [≡] [b][ −] [2][b][ =][ n][2] [≡] [1] (mod 8) =⇒ b ≡ 7 (mod 8).

For the case (b, 2b), in a similar way, we note that from (1.2) and (1.3), v2(z2) = [m]2[−][1] =

v2(z3). From (1.4), we can now observe that b[2] - u[2]3 [−] [b][ ·][ u]2[2] [=] [2][q] =⇒ b[2] − b ≡ 2

(mod 8) =⇒ b ≡ 7 (mod 8). Hence the result follows.

Now for the case (2b, 2b), v2(zi) < 0 =⇒ 2b·u[2]1 [≡] [0] [(mod] [4)][ from (1.3), a contradiction.]

Hence v2(zi) ≥ 0 for all i = 1, 2, 3. Now from (1.3) and (1.4), one can immediately

observe that v2(z1) = v2(z3) = [m]2[−][1] [,] [which] [in] [turn,] [implies] [b][ ·][ u][2]1 [−] [2][b][2][ ·][ u]3[2] [=] [−][1] [=][⇒]

b ≡ 1 (mod 8) from (1.3). Hence the result follows.

(ii) For all the four types of pairs, in this case, one can first note from Lemma (2.3)

that vp(z3) = −1, and vp(z1) = vp(z2) = 0, as otherwise from (1.3), either 2[m] ≡ 0

� m+1 �
(mod p) or b ≡ 0 (mod p[2]). Now from (1.4), for (b, b) and (2b, 2b), we get 2 p q = 1,

� � � �
a contradiction as n[2] + 1 = 2q =⇒ 2pq = 1 whereas m is odd, and 2p = −1 as

p ≡±3 (mod 8).

For (2b, b) and (b, 2b), noting again that vp(z3) = −1, and vp(z1) = vp(z2) = 0, one gets


-----

HERON TRIANGLES AND ELLIPTIC CURVES 7

� �
p2 = 1 from (1.2), a contradiction again as p ≡±3 (mod 8). Hence the result follows.

(iii) For (2, 2), from (1.4), we obtain vq(zi) > 0 =⇒ 2[m][+1] ≡ 0 (mod q), vp(z2) = vz3 =

� �
0 =⇒ 2q = 1 contradiction both the time. Also, one can trivially observe from (1.4)

that vp(z2) = 0 if and only if vp(z3) = 0. Hence vq(zi) < 0 for all i ∈{1, 2, 3}, and from

� �
Lemma (2.1), one now gets 2q = 1, a contradiction when q ≡ 5 (mod 8). For p ≡±1

(mod 8), homogeneous spaces corresponding to (2b, b), (b, 2b) and (2b, 2b) provide the

� �
same contradiction, after we note that n[2] + 1 = 2q =⇒ qb = 1. This concludes the

proof. 
4. Size of Sel2(E) when m is even

We focus on the elliptic curve E and its corresponding 2-Selmer group Sel2(E) with

the assumption m is an even integer in this section. From Lemma 2.3 and Lemma 2.4,

it is again evident that {(b, b), (b, 2b), (2b, b), (2b, 2b)} are the only possible elements in

Sel2(E).

**Lemma** **4.1.** Let (b1, b2) ∈ Q(S, 2) × Q(S, 2). Then

(i) The homogeneous spaces corresponding to (2b, b) and (2b, 2b) have no 2-adic solu
tions.

(ii) If p ̸≡±1 (mod 8) and n ≡ 0 (mod p), then the homogeneous spaces corresponding

to (b, 2b) has no p-adic solutions.

(iii) The homogeneous space corresponding to (b, b) has no 2-adic solution if b ̸≡±1

(mod 8).

(iv) The homogeneous spaces corresponding to (b, b) and (b, 2b) have no q-adic solution

� �
if qb = −1.

Proof. (i) For l = 2 and (b1, b2) = (2b, b), it is evident from Lemma 2.1 and Lemma 2.2

that v2(zi) ≥ 0 for the homogeneous space described by (1.2) and (1.3). Noting that

m is even and comparing the parity of the exponent of 2 on both sides of the equation

(1.2) and (1.4), we get v2(z3) = v2(z2) = [m]2 [.] [From] [(1.4),] [we] [now] [get] [2][b][ ·][ u][2]3 [−] [b][ ·][ u]2[2] [=]

2q =⇒ b ≡ 2 (mod 8), hence a contradiction and the result then follows for the case

(b1, b2) = (2b, b).

For the case (b1, b2) = (2b, 2b) a similar method shows from (1.3) and (1.4) that v2(z2) =

m2 [, v][2][(][z][3][) =] [m]2[−][2] [.] [From] [(1.4),] [one] [can] [now] [get] [2][b][ ≡] [7] [(mod] [8)][,] [a] [contradiction.] [Hence]


-----

HERON TRIANGLES AND ELLIPTIC CURVES 8

the result follows.

(ii) We first prove the case when b ≡ 0 (mod p). If vp(zi) ≥ 0 for all i, then from

(1.3) for (b1, b2) = (b, 2b), one can observe that 2[m] ≡ 0 (mod p), a contradiction.

Similarly if vp(zi) = −t < 0 for all i, again from (1.3), one can see b · u[2]1 [≡] [0] [(mod] [p][2][)][,]

a contradiction. From Lemma 2.3, now the only remaining case is vp(z3) = −1, and

� �
vp(z1) = vp(z2) = 0. Noting m is even, from (1.3), this implies p2 = 1 =⇒ p ≡±1

(mod 8).

Now for the case when p ̸≡±1 (mod 8) divides n but not b, one can immediately observe

� �
that vp(zi) ≥ 0 as otherwise p2 = 1 from (1.2), a contradiction if p ̸≡±1 (mod 8).

� �
Now for vp(zi) ≥ 0, from (1.2), we get that vp(z2) = 0 =⇒ vp(z1) = 0 =⇒ 2p = 1, a

contradiction. Hence vp(z2) > 0 which implies vp(z1) > 0 and that leads to vp(z3) = 0.


Now from (1.4), we get

�
2
2b[2]       - u[2]3 [−] [2][b][ ·][ z]2[2] [= 2][m][ ·][ 2][q] [=][⇒] [b][2][ ·][ u][2]3 [≡] [2][m][−][1] (mod p) =⇒
p

The result now follows as 2 is a quadratic non-residue modulo p here.


�
= 1.


(iii) For homogeneous space corresponding to (b, b), if v2(zi) < 0 in (1.2) and (1.3),

then from (1.3), it is evident that b ≡ 1 (mod 8). Otherwise, for the case v2(zi) ≥ 0,

from (1.4), one can notice that

v2(z2) = v2(z3) =⇒ b[2]      - u[2]3 [−] [b][ ·][ u]2[2] [≡] [2][q] [or] [0][ ≡] [2] [or] [0] (mod 8).

This implies b ≡±1 (mod 8), and the result follows.

(iv) For homogeneous spaces corresponding to both (b, b) and (b, 2b), if the correspond
ing equations (1.2) and (1.3), have q-adic solutions with vq(zi) ≥ 0, then from (1.3), we

� �
get 2[m] ≡ 0 (mod q), a contradiction. Hence vq(zi) = −t < 0 =⇒ qb = 1 from (1.4).

                        
5. Everywhere local solution for certain homogeneous spaces

We now look into concluding the computation of the 2-Selmer rank for E. The following

result focuses on the homogeneous spaces with local solutions everywhere.

**Lemma** **5.1.** Equations (1.2) and (1.3) have a local solution in Ql for every prime l

for (b1, b2) = (2, 2) when m is odd, q ≡ 1 (mod 8).


-----

HERON TRIANGLES AND ELLIPTIC CURVES 9

Proof. The Jacobian of the intersection of (1.2) and (1.3) for (2, 2) is
 


(5.1) [4][ ·][ z][1] −4 · z2 0 

4 · z1 0 −8 · z3

which one can easily observe has rank 2 whenever l ̸= 2, p, q where p ≡±1 (mod 8),

q ≡ 1 (mod 8). Hence except for those l’s, the topological genus becomes the same as

the arithmetic genus, which is 1 by the degree-genus formula, and Hasse-Weil bound for

a genus one curve can be used for all but finitely many primes. For small primes l = 2, 3

and for l = p, q we check directly for the local solutions.

For l ̸= p, q, l ≥ 5, using Hasse bound we choose a solution (z1, z2, z3) ∈ Fl × Fl × Fl

such that not all three of them are zero modulo l. Now z1 ≡ z2 ≡ 0 (mod l) implies

l[2] divides 2[m] - n[2] =⇒ l = p, a contradiction. Similarly, z1 ≡ z3 ≡ 0 (mod l) implies

−2[m] ≡ 0 (mod l) =⇒ l = 2, contradiction again. One can now suitably choose two of

z1, z2 and z3 to convert equations (1.2) and (1.3) into one single equation of one variable

with a simple root over Fl. That common solution can then be lifted to Ql via Hensel’s

lemma.

For l = 2, we first note that any solution (z1, z2, z3) of (1.2) and (1.4) implies v2(z1) =

v2(z3) = [m]2[−][1] [.] [Using] [these] [we] [convert] [the] [equations] [(1.2)] [and] [(1.3)] [into] [the] [following;]

(5.2) u[2]1 [−] [z]2[2] [=][ n][2][,]

(5.3) u[2]1 [−] [2][u]3[2] [=][ −][1][.]

Fixing z2 = 0 and u3 = 1, one can see that u[2]1 [≡] [1] [(mod] [8)] [is] [a] [solution] [to] [the] [above]

equations and can be lifted to Z2 via Hensel’s lemma. Multiplying both sides by 2[m]

then gives rise to a solution of (1.2) and (1.3) in Z2.

For l = 3, if n ≡ 0 (mod 3), fixing z2 = z3 = 1 gives rise to z1 ̸≡ 0 (mod 3) as a solution

to (1.2) and (1.3), that can be lifted to Z3 via Hensel’s lemma. If 3 does not divide n,

fixing z2 = 0 and z3 = 1 implies z1 ̸≡ 0 (mod 3) is a solution that Hensel’s lemma can

again lift.

For l = p ≡±1 (mod 8), instead of looking for p-adic solution of (1.2) and (1.3), we

� �
focus on equations (1.3) and (1.4). Fixing z1 = z2 = 0, and noting that p2 = 1, one

can choose z3 such that z3[2] [≡] [2][m][−][2] [(mod] [p][)][,] [as] [a] [solution] [that] [can] [be] [lifted] [to] [Z][p] [via]

Hensel’s lemma.

For l = q ≡ 1 (mod 8), a very similar argument as for l = p, shows that z1 = z2 = 0


-----

HERON TRIANGLES AND ELLIPTIC CURVES 10

gives rise to z3 as a solution such that z3[2] [≡−][2][m][−][1] [(mod] [q][)][,] [that] [can] [lifted] [to] [Z][q] [via]

Hensel’s lemma. This concludes the proof. 
The following result now completely determine the 2-Selmer group of E for odd integer

m when all the prime factors of n is of the form ±1 modulo 8.

**Lemma 5.2.** Let pi ≡±1 (mod 8) for all i ∈{1, 2, ..., k}. Then for odd m, the equations

(1.2) and (1.3) have a local solution in Ql for

(i) every prime l for (b1, b2) = (b, b), when b ≡ 1 (mod 8).

(ii) every prime l for (2b, b) when b ≡ 7 (mod 8) if q ≡ 1 (mod 8).

Proof. The method adopted to prove the result is similar to that used in the proof

of Lemma 5.1. Hence, we mostly provide direct solutions that can be lifted to l-adic

rational numbers via Hensel’s lemma. Like the previous case, we explicitly check the

cases l = 2, 3, q, p ≡±1 (mod 8), and the case q ≡ 1 (mod 8) if q is of such form. For all

other primes l, we use the Hasse-Weil bound and can immediately observe that there

exists a solution (z1, z2, z3) for homogeneous spaces corresponding to both (b, b) and

(2b, b) modulo l such that z1 ≡ z2 ≡ 0 (mod l) or z1 ≡ z3 ≡ 0 (mod l) is not possible.

Hence can be lifted to Ql via Hensel’s lemma. We now start with the case (b, b).

(i) For l = 2, choosing v2(zi) < 0, one can immediately see z1[2] [≡] [1] [(mod] [8)][ is a solution]

if z2 = z3 = 1 and hence can be lifted to Z2.

For l = 3, we note that n ̸≡ 0 (mod 3) in this case, and so is true for b. For b ≡ 1

(mod 3), we look for solutions such that vq(zi) < 0. Fixing z2 = z3 = 1, one can see

z1 ̸≡ 0 (mod 3) is then a solution modulo 3 that can be lifted to Q3 via Hensel’s lemma.

For b ≡ 2 (mod 3), fixing z2 = 0 and z3 = 1 gives rise to z1 ̸≡ 0 (mod 3), a solution

that can gain be lifted by Hensel’s lemma.

� �
For l = p, where p varies over all prime factors of n, we start with noting p2 = 1 in

this case. We start with those primes p such that b ≡ 0 (mod p). Using Lemma 2.3,

we look for solutions (z1, z2, z3) such that vp(z3) = −1 and vp(z1) = vp(z2) = 0. Fixing

z1 = z2 = 1, we get p[b][2][2] [·][ z]3[2] [= 2][m][.] [This] [z][3] [can] [be lifted] [to][ p][-adic] [solutions] [using] [Hensel’s]

lemma.

� �
Now for l = p, such that p divides n but not b, noting that 2p = 1, we observe that

(0, 0, z3) is a solution for (1.3) and (1.4) where b[2]z3[2] [≡] [2][m] [(mod] [p][)][.] [This] [solution] [can]

be lifted to Zp using Hensel’s lemma.

For l = q, we first note that q ≡ 5 (mod 8) implies the Jacobian described earlier has


-----

HERON TRIANGLES AND ELLIPTIC CURVES 11

rank 2 and hence is already considered using Hasse-Weil bound. So q ≡ 1 (mod 8), and
� �
qb = 1 here. Now (z1, 0, 0) gives rise to a solution that can be lifted by Hensel’s lemma,

where b · z1[2] [≡−][2][m] [(mod] [q][)][.] [This] [concludes] [the] [proof] [for] [the] [case] [(][b][1][, b][2][) = (][b, b][)][.]

(ii) We now look into the case (b1, b2) = (2b, b) where b ≡−1 (mod 8). As above, we

will only focus on the primes l = 2, 3, p and q where p ≡±1 (mod 8). q ≡ 1 (mod 8) is

a necessary condition in this case, as otherwise, the corresponding homogeneous space

does not have any q-adic solution from Lemma 3.1 and hence does not belong to Sel2(E).

For l = 2, we again notice that v2(z1) = v2(z2) − 1 = [m]2[−][1] which implies dividing both

sides of (1.2) and (1.3) by 2[m], (z1, 1, 0) is a solution modulo 8, where z1[2] [≡] [1] [(mod] [8)][,]

and hence can be lifted to Q2 via Hensel’s lemma.

For l = 3, we again note that n ̸≡ 0 (mod 3), and hence so is b. For b ≡ 1 (mod 3),

fixing z1 = 0 gives rise to z2, z3 ̸≡ 0 (mod 3) as solutions that can be lifted. For b ≡ 2

(mod 3), Fixing z2 = 1, z3 = 0, we get z1 ̸≡ 0 (mod 3) as a solution that Hensel’s

lemma can lift.

For l = p, when p ≡±1 (mod 8), p divides n but not b, fixing z1 = z2 = 0, one can

choose z3 such that b[2]z3[2] [≡] [2][m][−][1] [(mod] [p][)][.] [If] [b] [≡] [0] [(mod] [p][)][,] [we] [look] [for] [the] [solution]

with vp(z1) = vp(z2) = 0, vp(z3) = −1. Now fixing z1, z2 to any arbitrary values modulo

p, one can see (z1, z2, z3) is a solution where p[b][2][2] [·][ z]3[2] [≡] [2][m][−][1] [(mod] [p][)][.] [The] [solution] [then]

can be lifted to Qp by Hensel’s lemma.

� �
For l = q ≡ 1 (mod 8), noting qb = 1 here, one can fix z2 = z3 = 0 and get z1 as a

solution to (1.2) and (1.3) such that b · z1[2] [≡−][2][m] [(mod] [q][)][.] [This] [can] [be] [lifted] [to] [Q][q] [via]

Hensel’s lemma. This concludes the proof for this case. 
Now we focus on the 2-Selmer group Sel2(E) of the elliptic curve E when m is an even

integer. Before proving the results below, we first note that b ≡±1 (mod 8) implies
� �
qb = 1 here.

**Lemma** **5.3.** Equations (1.2) and (1.3) have a local solution in Ql for every prime l

for (b1, b2) = (b, b) when m is even, b ≡±1 (mod 8).

Proof. The Jacobian of the intersection of (1.2) and (1.3) for (b, b) is
 


(5.4) [2][b][ ·][ z][1] −2b · z2 0 

2b · z1 0 −2b[2]                   - z3


-----

HERON TRIANGLES AND ELLIPTIC CURVES 12

For l ̸∈{2, p, q}, the Jacobian has rank 2 modulo l and hence represent a smooth curve of

genus one. Using Hasse-Weil bound for such l ≥ 5, one can guarantee the existence of at

least two solutions (z1, z2, z3) for (1.2) and (1.3) modulo l. It is now a simple observation

that either of z1 ≡ z2 ≡ 0 (mod l) or z1 ≡ z3 ≡ 0 (mod l) implies l ∈{2, p, q}. Hence

the solutions can not be pairwise zero. Fixing two of z1, z2 and z3 suitably, now one can

use Hensel’s lemma to lift the solutions modulo l to Ql for all such primes l.

For l = 2, we start with the case b ≡ 1 (mod 8). Fixing z2 = z3 = 1 for v2(zi) < 0,

z1[2] [≡] [1] [(mod] [8)] [is] [a] [solution] [that] [can] [be] [lifted] [to] [Q][2] [by] [Hensel’s] [lemma.] [For] [b] [≡−][1]

(mod 8), we first observe from (1.2) and (1.4) that [m]2 [=][ v][2][(][z][2][) =][ v][2][(][z][3][)][ < v][2][(][z][1][)] [is] [the]

only possibility in this case for equations (1.2) and (1.3). Dividing both sides of (1.2)

and (1.3) by 2[m] then yields in

(5.5) b · z1[2] [−] [b][ ·][ u]2[2] [=][ n][2] [≡] [1] (mod 8),

(5.6) b · z1[2] [−] [b][2][ ·][ u]3[2] [=][ −][1][ ≡−][1] (mod 8).

Fixing z1 = 0 results in u2 = u3 = 1 as solutions that can be lifted to Q2 via Hensel’s

lemma.

For l = 3 and b ≡ 1 (mod 3), one can note that fixing z2 = z3 = 1 for v3(zi) < 0 gives

rise to z1 = 1 as a solution that can be lifted to Q3. Similarly for b ≡ 2 (mod 3), if

n ̸≡ 0 (mod 3), fixing z1 = 0 leads to z2 = z3 = 1 as solutions that can be lifted. In

case b ≡ 2 (mod 3) and n ≡ 0 (mod 3), fixing z2 = 1, z3 = 0 gives rise z1 ̸≡ 0 (mod 3)

as a solution that can be lifted to Z3.

For l = p where p divides b, it can be seen that fixing z1 and z2 to any arbitrary values

modulo p and choosing z3 as the solution of p[b][2][2] [·][ z]3[2] [≡] [2][m] [(mod] [p][)] [gives] [the] [required]

solution of (1.2) and (1.3) that can be lifted to Qp for vp(z1) = vp(z2) = 0, vp(z3) = −1.

For l = p where p divides n but not b, (0, 0, z3) is a solution of (1.3) and (1.4), that can

be lifted to Qp where z3 satisfies b[2] - z3[2] [≡] [2][m] [(mod] [p][)][.]

For l = q, fixing z2 = z3 = 0 in (1.2) and (1.3), we can see that (z1, 0, 0) is a solution

� �
that can be lifted to Qq where b·z1[2] [≡−][2][m] [(mod] [q][)][. This follows from the fact] qb = 1.

Now the result follows. 
**Lemma** **5.4.** Equations (1.2) and (1.3) have a local solution in Ql for every prime l

for (b1, b2) = (b, 2b) when m is even, and the the prime factors of n are of the form ±1

modulo 8.


-----

HERON TRIANGLES AND ELLIPTIC CURVES 13

Proof. The Jacobian of the intersection of (1.2) and (1.3) for (b, b) is
 


(5.7) [2][b][ ·][ z][1] −4b · z2 0 

2b · z1 0 −4b[2]                   - z3

Similar to the method used in Lemma 5.3, we note that for l ̸∈{2, p, q}, the Jacobian

has rank 2 modulo 2 and hence represent a curve of genus one modulo such primes.

Using Hasse-Weil bound, in a similar way, one can then notice the homogeneous space

corresponding to (b, 2b) represented by (1.2) and (1.2) has l-adic solution for all l ≥ 5

and l ̸= 2, p, q.

For l = 2, b ≡ 1 (mod 8), we first note that [m]2 [=][ v][2][(][z][1][)][ < v][2][(][z][2][)] [is] [the] [only] [possibility.]

Dividing both sides of (1.2) and (1.3) by 2[m], we can see that (z1, 0, 1) where z1[2] [≡] [1]

(mod 8) is a solution to the reduced equations and can be lifted to Q2 by Hensel’s

lemma. For b ≡−1 (mod 8), again one can note that [m]2 [=] [v][2][(][z][1][)] [<] [v][2][(][z][3][)][.] [This] [gives]

rise to (z1, 1, 0) as a solution modulo 8 such that z1[2] [≡] [1] [(mod] [8)] [and] [can] [be] [lifted] [to]

Z2.

For l = 3, for z1 ̸≡ 0 (mod 3), we get (z1, 0, 1) as solution modulo l to (1.2) and (1.3)

when b ≡ 1 (mod 3) that can be lifted to Q3 via Hensel’s lemma. For b ≡ 2 (mod 3),

the solution (z1, 1, 0) with z1 ̸≡ 0 (mod 3) does the same thing.

For l = p, if b ≡ 0 (mod p), then fixing z1, z2 to any arbitrary non-zero values modulo

l, (z1, z2, u3) becomes a solution for the case vl(z1) = vl(z2) = 0, vl(z3) = −1 where

� �

u3 satisfies p[b][2][2] [·][ u][2]3 [≡] [2][m][−][1] [(mod] [p][)][.] [This] [happens] [as] p2 = 1. Hensel’s lemma can lift

the solution because after fixing z1 and z2, u3 becomes a simple root for equations (1.2)

and (1.3). For the case when p divides n but not b, (0, 0, z3) is a solution to (1.3) and

(1.4) that can be lifted to Qp where 2b[2] - z3[2] [≡] [2][m] [(mod] [p][)][.]

� �
Now for l = q, noting that m is even and qb = 1, (z1, 0, 0) is a solution modulo q

that can be lifted to Qq where z1 satisfies b · z1[2] [≡−][2][m] [(mod] [q][)][.] [This] [concludes] [the]

proof. 
6. Elliptic curves with arbitrarily large 2-Selmer rank

Now we are in a position to prove Theorem 1.1 and Theorem 1.2. We note that these the
orems give rise to a construction of elliptic curves with arbitrarily high 2-Selmer ranks,

and possibly high Mordell-Weil ranks depending on the structure of the Shafarevich
Tate groups.

**Proof** **of** **Theorem** **1.1:** From Lemma 3.1, one can immediately conclude that for


-----

HERON TRIANGLES AND ELLIPTIC CURVES 14

p ≡±3 (mod 8), Sel2(E) = 0 if q ≡ 5 (mod 8) and for q ≡ 1 (mod 8), the 2-Selmer

group can possibly only be generated by (2, 2) which was proved in Lemma 5.1.

For the case p ≡±1 (mod 8), from Lemma 3.1, one can identify (b, b) as only possible

generators of Sel2(E) if q ≡ 5 (mod 8) and b ≡ 1 (mod 8), the assertion later veri
fied in Lemma 5.2. For q ≡ 1 (mod 8), again from Lemma 3.1, the possible elements

of Sel2(E) are identified as (2, 2), (b, b), (2b, b), (b, 2b), (2b, 2b). In Lemma 5.2, the exis
tence of (b, b), (2b, b) in the 2-Selmer group is proved, which along with the existence of

(2, 2) ∈ Sel2(E) proved in Lemma 5.1, proves the result of Theorem 1.1.

We enlist a table of examples for m odd case when m = 1, 3, 5.

Table 1. Examples for m odd

n q Sel2(E) n q Sel2(E)
3 5 trivial 3 · 5 113 ∗⟨(2, 2)⟩
11 · 19 21841 ∗⟨(2, 2)⟩ 3 · 5 · 11 13613 trivial
3 · 5 · 13 19013 trivial 79 3121 ∗⟨(2, 2), (79, 2 · 79)⟩
17 · 23 76441 ⟨(2, 2), ∗(34, 34), (23, 46)⟩ 7 · 17 · 31 6804361 ⟨(2, 2), (17, 17), (14, 7), (62, 31)⟩

**Proof** **of** **Theorem** **1.2:** From Lemma 4.1, we conclude that Sel2(E) possibly only

contain (b1, b1) or (b2, 2b2) when b1 ≡±1 (mod 8), every prime factors of b2 is of the

form ±1 modulo 8, and bi is a quadratic residue modulo q and any prime factor of

n that does not divide bi. Lemma 5.3 and Lemma 5.4 proves that Sel2(E) contains

(b1, b1), (b2, 2b2) by showing the existence of l-adic solution for all primes l. Hence the

result follows.

We enlist a table of examples for m even case when m = 2, 4, 6.

Table 2. Examples for m even

n q Sel2(E) n q Sel2(E)
3 5 trivial 3 · 5 113 ∗⟨(3 · 5, 3 · 5)⟩
11 · 19 21841 ∗⟨(11 · 19, 11 · 19)⟩ 3 · 5 · 11 13613 ⟨(15, 15), (55, 55)⟩
3 · 5 · 13 19013 ⟨(15, 15), (65, 65)⟩ 71 2521 ⟨(71, 71), (1, 2)⟩
79 3121 ∗⟨(79, 79), (1, 2)⟩ 17 · 23 76441 ⟨∗(17, 17), (23, 23), (1, 2)⟩
5 · 7 · 11 · 17 21418513 ⟨(7, 7), (17, 17), (55, 55)⟩ 7 · 17 · 31 6804361 ⟨(7, 7), (17, 17), (31, 31), (1, 2)⟩

References

[1] R.H. Buchholz, and L.R. Randall, "An infinite set of Heron triangles with two rational medians."

The American Mathematical Monthly 104.2 (1997): 107-115.


-----

HERON TRIANGLES AND ELLIPTIC CURVES 15

[2] R.H. Buchholz, and R.P. Stingley, "Heron triangles with three rational medians." The Rocky

Mountain Journal of Mathematics 49.2 (2019): 405-417.

[3] D. Chakraborty, V. Ghale, and A. Saikia. "Construction of an infinite family of elliptic curves of

2-Selmer rank 1 from Heron triangles." Research in Number Theory 8, no. 4 (2022): 101.

[4] A. Dujella, and J.C. Peral, "Elliptic curves coming from Heron triangles." The Rocky Mountain

Journal of Mathematics 44.4 (2014): 1145-1160.

[5] V. Ghale, S Das, and D. Chakraborty, "A Heron triangle and a Diophantine equation." Periodica

Mathematica Hungarica (2022): 1-8.

[6] E. H. Goins, and D. Maddox, "Heron triangles via elliptic curves." The Rocky Mountain Journal

of Mathematics (2006): 1511-1526.

[7] D. R. Heath-Brown, "The size of Selmer groups for the congruent number problem." Inventiones

Mathematicae 111.1 (1993): 171-195.

[8] D.R. Heath-Brown, "The size of Selmer groups for the congruent number problem, II." Inventiones

Mathematicae 118 (1994): 331-370.

[9] D. J. Rusin, "Rational triangles with equal area." New York J. Math 4.1 (1998): 15.

[10] J. H. Silverman, The arithmetic of elliptic curves. Vol. 106. New York: Springer, 2009.

Department of Mathematics, BITS-Pilani, Hyderabad campus, Hyderabad, INDIA

Email address: p20180465@hyderabad.bits-pilani.ac.in

Department of Mathematics, BITS-Pilani, Hyderabad campus, Hyderabad, INDIA

Email address: p20200059@hyderabad.bits-pilani.ac.in

Department of Mathematics, BITS-Pilani, Hyderabad campus, Hyderabad, INDIA

Email address: debopam@hyderabad.bits-pilani.ac.in


-----

