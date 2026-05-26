#set page(
  paper: "a4",
  flipped: true,
  margin: 0.5cm,
  columns: 3,
)
#set columns(gutter: 4pt)

#set text(size: 8pt)
#set par(leading: 4pt, spacing: 6pt)

#show heading.where(level: 1): it => [
  #set text(size: 10pt, weight: "bold")
  #it.body
]
#show heading.where(level: 2): it => [
  #set text(size: 9pt, weight: "bold")
  #it.body
]
#show heading.where(level: 3): it => [
  #set text(size: 8pt, weight: "bold")
  #it.body
]

#show heading: set block(above: 6pt, below: 3pt)
#show math.equation.where(block: true): set block(above: 4pt, below: 4pt)
#show list: set block(above: 3pt, below: 3pt)
#set list(indent: 6pt, body-indent: 4pt)

#place(
  top + center,
  scope: "parent",
  float: true,
  block(width: 100%, align(center)[
    #text(size: 14pt, weight: "bold")[Höhere Mathematik 1 und 2]
  ])
)

= Rechnerarithmetik

== Gleitkommazahlen / Maschinenzahlen

- Darstellung: $M = {x in RR | x = plus.minus 0.m_1 m_2 m_3 dots m_n dot B^(e_(l-1) dots e_1 e_0 - "bias")} union {0}$
- Basis $B$, Mantissenlänge $n$, Exponentenbereich $[e_"min", e_"max"]$
- Normalisierte Zahl: $m_1 != 0$
- Bias: $"bias" = B^(l-1) - 1$ für $l$-stellige Exponentendarstellung
- Grösster Exponent: $e_"max" = B^"Stellen von e" - 1 - "bias"$
- Kleinster Exponent: $e_"min" = -"bias"$
- Kleinste positive Zahl: $x_"min" = B^(e_"min" - 1)$
- Grösste Zahl: $x_"max" = (1 - B^(-n)) dot B^(e_"max")$
- Kleinste Zahl: $x_"denormalisiert" = B^(e_"min" - n + 1)$
- Anzahl der Maschinenzahlen (mit 0): $|M| = 2 dot (B-1) dot B^(n+l-1) + 1$

= Fehleranalyse

- Gerundete Zahl: $tilde(x) = "rd"(x)$
- Absoluter Rundungsfehler: $|tilde(x) - x| <= 1/2 B^(e-n+1)$
- Relative Rundungsfehler: $|(tilde(x) - x) / x| <= "eps"$
- Maschinengenauigkeit: $"eps" = max_(x in M) |("rd"(x) - x) / x| = 1/2 B^(1-n)$

== Fehlerfortpflanzung bei Funktionen

- Absoluter Fehler: $|f(tilde(x)) - f(x)| approx |f'(x)| dot |tilde(x) - x|$
- Relativer Fehler: $|(f(tilde(x)) - f(x)) / f(x)| approx |(x dot f'(x)) / f(x)| dot |(tilde(x) - x) / x|$
- Kondition (Je kleiner desto besser): $K = |(x dot f'(x)) / f(x)|$

= Nullstellenprobleme

== Fixpunktiteration

- Fixpunktgleichung: $F(x) = x$
- Fixpunkte $(overline(x))$ sind Nullstellen von $f(x) = F(x) - x$
- Iterationsvorschrift: $x_(n+1) = F(x_n)$
- Konvergenz: $|F'(overline(x))| < 1$ $arrow$ anziehender Fixpunkt
- Divergenz: $|F'(overline(x))| > 1$ $arrow$ abstoßender Fixpunkt

=== Banach'scher Fixpunktsatz

- Sei $F: [a,b] arrow [a,b]$ und es existiert eine Konstante $0 < alpha < 1$ mit
  $ |F(x) - F(y)| <= alpha |x - y| quad forall x,y in [a,b] $
- Dann besitzt $F$ genau einen Fixpunkt $overline(x) in [a,b]$ und die Iteration $x_(n+1) = F(x_n)$ konvergiert für jeden Startwert $x_0 in [a,b]$ gegen $overline(x)$. (Lipschitz-Stetigkeit mit Lipschitz-Konstante $alpha$)
- a-priori-Fehlerabschätzung:
  $ |x_n - overline(x)| <= alpha^n / (1 - alpha) |x_1 - x_0| $
- a-posteriori-Fehlerabschätzung:
  $ |x_n - overline(x)| <= alpha / (1 - alpha) |x_n - x_(n-1)| $
- Umformung: $|(F(x) - F(y)) / (x - y)| <= alpha$

== Newton-Verfahren

$ x_(n+1) = x_n - f(x_n) / f'(x_n) $

Für den Startwert $x_0$ muss gelten, dass $f(x_0) != 0$ und $f'(x_0) != 0$. Ein geeigneter Startwert kann durch die Nullstellenbestimmung der Ableitungsfunktion gefunden werden.

=== Vereinfachtes Newton-Verfahren

- Fixierter Ableitungswert: $x_(n+1) = x_n - f(x_n) / f'(x_0)$
- Sekantenverfahren: $x_(n+1) = x_n - f(x_n) dot (x_n - x_(n-1)) / (f(x_n) - f(x_(n-1)))$

#colbreak()

== Fehlerabschätzung

=== Konvergenzordnung

- Sei $(x_n)$ eine konvergierende Folge gegen $overline(x)$
- Die Folge hat die Konvergenzordnung $q$, wenn eine Konstante $c > 0$ existiert, so dass
  $ |x_(n+1) - overline(x)| <= c dot |x_n - overline(x)|^q $
- $q = 1$ lineare Konvergenz, $q = 2$ quadratische Konvergenz

=== Nullstellensatz von Bolzano

- Sei $f: [a,b] arrow RR$ stetig mit $f(a) dot f(b) < 0$ oder $f(a) >= 0 >= f(b)$
- Dann existiert mindestens ein $overline(x) in (a,b)$ mit $f(overline(x)) = 0$

= Lineare Gleichungssysteme

== LR-Zerlegung

- Zerlegung einer Matrix $A$ in das Produkt $A = L dot R$
- $L$: untere Dreiecksmatrix mit Einsen auf der Diagonale
- $R$: obere Dreiecksmatrix
- Lösen von $A dot x = b$ durch Lösen von $L dot y = b$ und $R dot x = y$
- Voraussetzung: Keine Null-Pivotelemente während der Zerlegung / kein Zeilentausch notwendig (sonst $P dot A = L dot R$)

=== Beispiel

$
A = mat(-1, 1, 1; 1, -3, -2; 5, 1, 4)
= underbrace(mat(1, 0, 0; -1, 1, 0; -5, -3, 1), L)
dot underbrace(mat(-1, 1, 1; 0, -2, -1; 0, 0, 6), R)
$

=== Permutationsmatrix

- Vertauscht Zeilen oder Spalten einer Matrix
- $P dot A$: Vertauschen der Zeilen von $A$
- $A dot P$: Vertauschen der Spalten von $A$
- $P_1, P_2$ sind Permutationsmatrizen $=>$ $P = P_1 dot P_2$ ist auch eine Permutationsmatrix
- $P^(-1) = P^T$

== QR-Zerlegung

- Zerlegung einer Matrix $A$ in das Produkt $A = Q dot R$
- $Q$: orthogonale Matrix ($Q^T dot Q = I$)
- $R$: obere Dreiecksmatrix

=== Householder-Matrizen

- $H$ ist orthogonal und symmetrisch ($H^T = H$)
- $H$ spiegelt an der Hyperebene orthogonal zu $u$
- $H = H^T = H^(-1) => H dot H = I_n$
- Wenn $u$ ein normierter Vektor ist, dann gilt:
  $ H = I_n - 2 u u^T $

=== QR-Zerlegung mit Householder-Matrizen

$
A = mat(1, 2, -1; 4, -2, 6; 3, 1, 0), quad b = mat(9; -4; 9)
$

+ $v_1 = a_1 + "sign"(a_(11)) dot |a_1| dot e_1$
+ $u_1 = v_1 / (||v_1||_2)$
+ $H_1 = I - 2 u_1 u_1^T$
+ $A^((1)) = H_1 dot A$, $b^((1)) = H_1 dot b$

$
A^((1)) = mat(-5.0990, 0.5883, -4.5107; 0, -2.9258, 3.6976; 0, 0.3056, -1.7268)
$

Rekursiv mit $A^((1))$ ($2 times 2$-Matrix) durchführen bis $A^((n-1))$ obere Dreiecksmatrix ist.

#colbreak()

== Vektor- und Matrixnormen

- Vektornormen:
  $ "1-Norm, Summennorm: " ||x||_1 = sum_(i=1)^n |x_i| $
  $ "2-Norm, euklidische Norm: " ||x||_2 = sqrt(sum_(i=1)^n |x_i|^2) $
  $ "Unendlich-Norm, Maximumnorm: " ||x||_infinity = max_(1 <= i <= n) |x_i| $
- Matrixnormen ($n times m$-Matrizen):
  $ "1-Norm, Spaltensummennorm: " ||A||_1 = max_(1 <= j <= m) sum_(i=1)^n |a_(i j)| $
  $ "2-Norm, Spektralnorm: " ||A||_2 = sqrt(rho(A^T A)) $
  $ "Unendlich-Norm, Zeilensummennorm: " ||A||_infinity = max_(1 <= i <= n) sum_(j=1)^m |a_(i j)| $

=== Spektralradius

- $rho(A) = max { |lambda| | lambda "ist Eigenwert von" A }$
- Für jede Matrixnorm gilt: $rho(A) <= ||A||$

== Fehlerrechnung Vektoren

- $|| dot ||$: gewählte Matrixnorm
- Konditionszahl: $"cond"(A) = ||A|| dot ||A^(-1)||$
- Abschätzung des relativen Lösungsfehlers:
  $ (||tilde(x) - x||) / (||x||) <= "cond"(A) dot (||tilde(b) - b||) / (||b||) ", falls" ||b|| != 0 $
- Je größer $"cond"(A)$, desto schlechter ist das LGS konditioniert

== Fehlerrechnung Matrizen

- Gegeben: $tilde(A) = A + E$ mit $E$ als Störmatrix ($E = A - tilde(A)$)
- Abschätzung des relativen Lösungsfehlers:
  $ (||tilde(x) - x||) / (||x||) <= ("cond"(A)) / (1 - "cond"(A) dot (||E||) / (||A||)) dot ((||E||) / (||A||) + (||b - tilde(b)||) / (||b||)) $
- Gilt nur, wenn $"cond"(A) dot (||E||) / (||A||) < 1$

== Aufwandsabschätzung

- LR-Zerlegung: $2/3 n^3$ Operationen
- Vorwärts- und Rückwärtseinsetzen: $2n^2$ Operationen
- QR-Zerlegung mit Householder: $2n^3$ Operationen
- Vorwärts- und Rückwärtseinsetzen: $2n^2$ Operationen
- Allgemein:
  $ sum_(i=1)^n i = (n(n+1)) / 2 approx n^2 / 2 $
  und
  $ sum_(i=1)^n i^2 = 1/6 n + 1/2 n^2 + 1/3 n^3 approx 1/3 n^3 $

#colbreak()

= Iterative Verfahren für LGS

== Jacobi-Verfahren

- Zerlegung: D ist die Diagonalmatrix von A, L die untere Dreiecksmatrix und R die obere Dreiecksmatrix
  $
  A x &= b \
  (L + D + R) x &= b \
  D x &= -(L + R)x + b \
  x &= D^(-1)(L + R)x + D^(-1) b
  $
- Iterationsvorschrift: $x^((k+1)) = D^(-1)(L + R) x^((k)) + D^(-1) b$
- Iterationsmatrix: $B_J = D^(-1)(L + R)$

== Gauß-Seidel-Verfahren

- Iterationsvorschrift: $x^((k+1)) = (D - L)^(-1) R x^((k)) + (D - L)^(-1) b$
- Iterationsmatrix: $B_(G S) = (D - L)^(-1) R$

== Konvergenz und Abschätzung

- Beide Verfahren konvergieren, wenn $||B|| < 1$
- Symmetrisch positiv definite Matrizen (nur Gauß-Seidel)
- Siehe Fixpunktiteration ($alpha = ||B||$)

=== Diagonaldominanz

- Eine Matrix ist diagonaldominant wenn eines der folgenden Kriterien erfüllt ist:
  - $forall i = 1, dots, n: |a_(i i)| > sum_(j=1, j!=i) |a_(i,j)|$ (Zeilensummenkriterium)
  - $forall j = 1, dots, n: |a_(j j)| > sum_(i=1, i!=j) |a_(i,j)|$ (Spaltensummenkriterium)
- Beide Verfahren konvergieren für diagonaldominante Matrizen

= Komplexe Zahlen

== Darstellung

- Normalform / kartesische Form: $z = x + i y$, $x, y in RR$
- Polardarstellung / Trigonometrische Form: $z = r(cos phi + i sin phi)$
- Eulersche Darstellung: $z = r e^(i phi)$
- Betrag: $|z| = r = sqrt(x^2 + y^2) = sqrt(z dot z^*)$
- Argument: $phi = arg(z) = tan^(-1)(y / x) = cos^(-1)(x / r) = sin^(-1)(y / r)$

== Rechenregeln

- Addition: $z_1 + z_2 = (x_1 + x_2) + i(y_1 + y_2)$
- Multiplikation: $z_1 dot z_2 = (x_1 + i y_1)(x_2 + i y_2)$
- Multiplikation in Polardarstellung:
  $ z_1 dot z_2 = r_1 r_2 [cos(phi_1 + phi_2) + i sin(phi_1 + phi_2)] $
- Division:
  $ z_1 / z_2 = (z_1 dot z_2^*) / (z_2 dot z_2^*), quad z_2 != 0 $
- Division in Polardarstellung:
  $ z_1 / z_2 = r_1 / r_2 [cos(phi_1 - phi_2) + i sin(phi_1 - phi_2)], quad z_2 != 0 $
- Potenzieren in der Polardarstellung:
  $ z^n = r^n [cos(n phi) + i sin(n phi)] = r^n e^(i n phi) $

== Wurzeln

- $z = r(cos phi + i sin phi) = r e^(i phi)$
- $n$-te Wurzel:
  $ z_k = r^(1\/n) [cos((phi + 2 k pi) / n) + i sin((phi + 2 k pi) / n)] = r e^(i (phi + 2 k pi) \/ n) $
  $ k = 0, 1, dots, n-10 $

#colbreak()

= Eigenwerte und Eigenvektoren

== Definition

Sei $A in RR^(n times n)$. Ein Vektor $v in RR^n, v != 0$ heißt Eigenvektor von $A$ zum Eigenwert $lambda in RR$, wenn gilt:
$ A v = lambda v $

== Bestimmung

- $lambda$ ist Eigenwert von $A$ $<=> det(A - lambda I_n) = 0$
- Charakteristisches Polynom: $P_A (lambda) = det(A - lambda I_n)$
- Eigenwerte: Nullstellen des charakteristischen Polynoms
- Eigenvektoren: Lösen von $(A - lambda I_n) v = 0$ für jeden Eigenwert $lambda$
- $lambda^(-1)$ ist Eigenwert von $A^(-1)$ mit demselben Eigenvektor $v$, falls $A$ invertierbar ist
- Das Spektrum von $A$ ist die Menge aller Eigenwerte: $sigma(A) = {lambda_1, lambda_2, dots, lambda_k}$

== Eigenräume

- Eigenraum zu $lambda$: $E_lambda = {v in RR^n | A v = lambda v}$
- Dimension des Eigenraums: geometrische Vielfachheit
- Vielfachheit eines Eigenwerts als Nullstelle des charakteristischen Polynoms: algebraische Vielfachheit
- $1 <= "geometrische Vielfachheit" <= "algebraische Vielfachheit"$
- Algebraische Vielfachheit 2 $=>$ Determinante $= 0$ $=> Delta = b^2 + 4 a c$

== Diagonalisierbarkeit / Ähnlichkeit

- Es seien $A, B in RR^(n times n)$. $A$ und $B$ heißen ähnlich, wenn eine invertierbare Matrix $S in RR^(n times n)$ existiert, so dass gilt:
  $ B = S^(-1) A S $
- $A$ ist diagonalisierbar, wenn $S = D$ eine Diagonalmatrix ist
- $D$ enthält die Eigenwerte von $A$ auf der Diagonale
- $S$ enthält die zugehörigen Eigenvektoren als Spalten
- Wenn $A, B$ ähnlich sind, gilt:
  - A und B haben dieselben Eigenwerte mit derselben algebraischen Vielfachheit
  - Die Eigenvektoren von A und B hängen über die Matrix S zusammen

== von-Mises-Iteration

- Iterationsvorschrift: $v^((k+1)) = (A v^((k))) / (||A v^((k))||)$
- Konvergenz gegen den Eigenvektor zum betragsmäßig größten Eigenwert
- Approximierter Eigenwert: $lambda^((k)) = ((v^((k)))^T A v^((k))) / ((v^((k)))^T v^((k)))$

= Ableitungsregeln

- *Summenregel:* $(f + g)' = f' + g'$
- *Kettenregel:* $(f compose g)' = f'(g) dot g'$
- *Produktregel:* $(f dot g)' = f' dot g + f dot g'$
- *Quotientenregel:* $(f / g)' = (f' dot g - f dot g') / g^2$
- *Quotientenregel f = 1:* $(1 / g)' = -g' / g^2$
- *Potenzregel:* $(x^n)' = n dot x^(n-1)$
- *Exponentialfunktion:* $(e^x)' = e^x$
- *Logarithmus:* $(ln(x))' = 1 / x$
- *Sinus:* $(sin(x))' = cos(x)$
- *Kosinus:* $(cos(x))' = -sin(x)$
- *Tangens:* $(tan(x))' = 1 / cos^2(x)$
- *Umkehrfunktionen:*
  - $(arcsin(x))' = 1 / sqrt(1 - x^2)$
  - $(arccos(x))' = -1 / sqrt(1 - x^2)$
  - $(arctan(x))' = 1 / (1 + x^2)$
  - Generell: $(f^(-1)(y))' = 1 / (f'(f^(-1)(y)))$

#colbreak()

== Grenzwertregel von Bernoulli-de l'Hôpital

- Die Grenzwertregel von Bernoulli-de l'Hôpital ist ein Verfahren zur Bestimmung von Grenzwerten, die in der Form $0/0$ oder $infinity/infinity$ vorliegen.
  $ lim_(x -> x_0) f(x) / g(x) = lim_(x -> x_0) f'(x) / g'(x) $
- Falls der Grenzwert $lim_(x -> x_0) f(x)/g(x)$ in der Form $0 dot infinity$ vorliegt, kann die Regel wie folgt umgeformt werden:
  $ lim_(x -> x_0) f(x) dot g(x) = lim_(x -> x_0) f(x) / (1 / g(x)) $
- Falls der Grenzwert in der Form $infinity - infinity$ vorliegt, kann die Regel wie folgt umgeformt werden:
  $ lim_(x -> x_0) f(x) - g(x) = lim_(x -> x_0) (f(x) dot g(x)) / g(x) $

== Basiswechsel

- Umwandlung Dezimal in ein anderes Zahlensystem mit Basis B:
  - Ganzzahliger Teil: wiederholte Division durch B, Reste notieren (von unten nach oben lesen)
  - Dezimalteil: wiederholte Multiplikation mit B, ganzzahlige Teile notieren (von oben nach unten lesen)
  - Beispiel: Dezimal 13.625 in Binär
    - Ganzzahliger Teil: $13 div 2 = 6$ Rest 1, $6 div 2 = 3$ Rest 0, $3 div 2 = 1$ Rest 1, $1 div 2 = 0$ Rest 1 $arrow$ 1101
    - Dezimalteil: $0.625 times 2 = 1.25$ Ganzzahliger Teil 1, $0.25 times 2 = 0.5$ Ganzzahliger Teil 0, $0.5 times 2 = 1.0$ Ganzzahliger Teil 1 $arrow$ 101
    - Ergebnis: $13.625_(10) = 1101.101_2$
- Umwandlung von Basis B in Dezimal:
  - Ganzzahliger Teil: $Z = sum_(i=0)^n z_i dot B^i$
  - Dezimalteil: $D = sum_(i=1)^m z_(-i) dot B^(-i)$

#pagebreak()

= Iterative Verfahren für nichtlineare Gleichungssysteme

== Paritelle Ableitung
- Die partielle Ableitung von $f: RR^n arrow RR$ wird wie folgt definiert:
$ partial f / partial x_i (x) = lim_(h -> 0) (f(x_1, dots, x_(i-1), x_i + h, x_(i+1), dots, x_n) - f(x)) / h $
- Beispiel: $f(x,y) = x^2 y + sin(x y)$
  - $partial f / partial x = 2 x y + y cos(x y)$
  - $partial f / partial y = x^2 + x cos(x y)$
== Jacobimatrix
- Für $F: RR^n arrow RR^m$ mit $F(x) = (f_1(x), f_2(x), dots, f_m(x))^T$ ist die Jacobimatrix von $F$ an der Stelle $x$ definiert als:
  $ J_F(x) = mat(partial f_1 / partial x_1, partial f_1 / partial x_2, dots, partial f_1 / partial x_n; partial f_2 / partial x_1, partial f_2 / partial x_2, dots, partial f_2 / partial x_n; dots; partial f_m / partial x_1, partial f_m / partial x_2, dots, partial f_m / partial x_n) $
- Beispiel: $F(x,y) = (x^2 y + sin(x y), x^2 + y^2)^T$
  - $J_F(x,y) = mat(2 x y + y cos(x y), x^2 + x cos(x y); 2 x, 2 y)$

== Newton-Verfahren für Systeme
- Gegeben: $F: RR^n arrow RR^n$, $F(x) = 0$
- Iterationsvorschrift: $x^((k+1)) = x^((k)) - J_F(x^((k)))^(-1) F(x^((k)))$ oder äquivalent: Löse $J_F(x^((k))) delta^((k)) = F(x^((k)))$ und setze $x^((k+1)) = x^((k)) - delta^((k))$
- $J_F(x)$ ist die Jacobimatrix von $F$ an der Stelle $x$
- Konvergenz: Quadratische Konvergenz, wenn $F$ zweimal stetig differenzierbar ist und $J_F(overline(x))$ invertierbar ist

== Vereinfachtes Newton-Verfahren für Systeme
- Fixierter Jacobimatrix: $x^((k+1)) = x^((k)) - J_F(x^((0)))^(-1) F(x^((k)))$
- Sekantenverfahren: $x^((k+1)) = x^((k)) - J_F(x^((k)))^(-1) F(x^((k)))$ mit $J_F(x^((k)))$ approximiert durch finite Differenzen:
  $ J_F(x^((k))) approx (F(x^((k))) - F(x^((k-1)))) / (x^((k)) - x^((k-1))) $

== Gedämpftes Newton-Verfahren für Systeme
- Iterationsvorschrift: Berechne $delta^((k))$ aus $J_F(x^((k))) delta^((k)) = -F(x^((k)))$
- Finde minimales $k in {0,1,...,k_max}$ mit $||F(x^((k)) + delta^((k)) / 2^k)||_2 < ||F(x^((k)))||_2$
- Setze $x^((k+1)) = x^((k)) + delta^((k)) / 2^k$
- Falls kein $k$ gefunden: rechne mit $k=0$ weiter
- Faustregel: $k_max = 4$

= Ausgleichsrechnung

== Interpolation

=== Lagrange-Interpolationsformel
- Gegeben: $n+1$ Stützpunkte $(x_i, y_i)$, $i=0,...,n$ mit $x_i != x_j$ für $i != j$
- Eindeutiges Interpolationspolynom vom Grad $<= n$:
  $ P_n(x) = sum_(i=0)^n l_i(x) y_i $
- Lagrange-Basispolynome:
  $ l_i(x) = product_(j=0, j!=i)^n (x - x_j) / (x_i - x_j) $

#colbreak()

=== Kubische Splineinterpolation
- Für $n+1$ Stützpunkte: $n$ kubische Polynome $S_i$ auf $[x_i, x_{i+1}]$:
  $ S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3 $
- Bedingungen (12 für 4 Punkte):
  - Interpolation: $S_i(x_i) = y_i$, $S_(n-1)(x_n) = y_n$
  - Stetiger Übergang: $S_i(x_(i+1)) = S_(i+1)(x_(i+1))$
  - Keine Knicke: $S'_i(x_(i+1)) = S'_(i+1)(x_(i+1))$
  - Gleiche Krümmung: $S''_i(x_(i+1)) = S''_(i+1)(x_(i+1))$
  - +2 Randbedingungen nach Typ:
- Natürliche kubische Spline: $S''_0(x_0) = 0$, $S''_(n-1)(x_n) = 0$
- Periodische kubische Spline: $S'_0(x_0) = S'_(n-1)(x_n)$, $S''_0(x_0) = S''_(n-1)(x_n)$
- Not-a-knot: $S'''_0(x_1) = S'''_1(x_1)$, $S'''_(n-2)(x_(n-1)) = S'''_(n-1)(x_(n-1))$

== Lineare Ausgleichsrechnung

=== Problemstellung
- Gegeben: $n$ Datenpunkte $(x_i, y_i)$
- Gesucht: Funktion $f(x) = lambda_1 f_1(x) + ... + lambda_m f_m(x)$ (Linearkombination von Basisfunktionen)
- Minimiere Fehlerfunktional (kleinste Fehlerquadrate / least squares):
  $ E(f) = ||y - f(x)||_2^2 = sum_(i=1)^n (y_i - f(x_i))^2 arrow min $

=== Normalgleichungssystem
- Designmatrix $A in RR^(n times m)$ mit $A_(i j) = f_j(x_i)$
- Normalgleichungssystem: $A^T A lambda = A^T y$
- Lösung gibt optimale Parameter $lambda = (lambda_1, ..., lambda_m)^T$
- Stabiler via QR-Zerlegung: $A = Q R arrow R lambda = Q^T y$

=== Ausgleichsgerade
- $f(x) = a x + b$, Designmatrix: $A = mat(x_1, 1; dots.v, dots.v; x_n, 1)$
- Normalgleichungssystem:
  $ mat(sum x_i^2, sum x_i; sum x_i, n) vec(a, b) = vec(sum x_i y_i, sum y_i) $

== Nichtlineare Ausgleichsrechnung

=== Gauss-Newton-Verfahren
- Gegeben: $g(lambda) = y - f(lambda)$, Jacobimatrix $D g(lambda)$ (partielle Ableitungen nach $lambda$ bzw. Fitparametern)
- Iterationsvorschrift: Löse lineares Ausgleichsproblem
  $ min ||g(lambda^((k))) + D g(lambda^((k))) delta^((k))||_2^2 $
  via QR-Zerlegung von $D g(lambda^((k))) = Q^((k)) R^((k))$:
  $ R^((k)) delta^((k)) = -Q^((k)T) g(lambda^((k))) $
- Setze $lambda^((k+1)) = lambda^((k)) + delta^((k))$

=== Gedämpftes Gauss-Newton-Verfahren
- Wie Gauss-Newton, aber akzeptiere $delta^((k))$ nur wenn:
  $ ||g(lambda^((k)) + delta^((k)) / 2^p)||_2^2 < ||g(lambda^((k)))||_2^2 $
- Finde minimales $p in {0,1,...,p_max}$ und setze:
  $ lambda^((k+1)) = lambda^((k)) + delta^((k)) / 2^p $

#colbreak()

= Numerische Integration

== Quadraturformeln (Newton-Cotes)

=== Einfache Formeln ($[a,b]$)
- Rechteckregel: $R f = f((a+b)/2) dot (b-a)$
- Trapezregel: $T f = (f(a)+f(b))/2 dot (b-a)$
- Simpsonregel: $S f = (b-a)/6 [f(a) + 4 f((a+b)/2) + f(b)]$

=== Summierte Formeln 
- *$h = (b-a)/n$, $n = (b-a)/h$, $x_i = a + i h$*
- Summierte Rechteckregel:
  $ R f(h) = h sum_(i=0)^(n-1) f(x_i + h/2) $
- Summierte Trapezregel:
  $ T f(h) = h [(f(a)+f(b))/2 + sum_(i=1)^(n-1) f(x_i)] $
- Summierte Simpsonregel:
  - $n$ muss gerade sein
  $ S f(h) = h/3 [1/2 f(a) + sum_(i=1)^(n-1) f(x_i) + 2 sum_(i=1)^(n) f((x_(i-1)+x_i)/2) + 1/2 f(b)] $
  Alternativ: $S f(h) = (T f(h) + 2 R f(h)) / 3$

== Fehlerabschätzung Quadraturformeln

$ |integral_a^b f(x) d x - R f(h)| <= h^2/24 (b-a) max |f''| $
$ |integral_a^b f(x) d x - T f(h)| <= h^2/12 (b-a) max |f''| $
$ |integral_a^b f(x) d x - S f(h)| <= h^4/2880 (b-a) max |f^((4))| $

== Gauss-Formeln

Nicht-äquidistante Stützstellen für höhere Genauigkeit. Für $integral_a^b f(x) d x approx (b-a)/2 sum a_i f(x_i)$:
- $G_1 f = (b-a) dot f((a+b)/2)$
- $G_2 f = (b-a)/2 [f((a+b)/2 - (b-a)/(2sqrt(3))) + f((a+b)/2 + (b-a)/(2sqrt(3)))]$
- $G_3 f = (b-a)/2 [5/9 f(x_(-)) + 8/9 f((a+b)/2) + 5/9 f(x_(+))]$ mit $x_(plus.minus) = (a+b)/2 plus.minus sqrt(0.6) dot (b-a)/2$


== Romberg-Extrapolation

- Berechne Trapezwerte $T_(j 0) = T f(h_j)$ mit $h_j = (b-a)/2^j$ für $j = 0,1,...,m$
- Anzahl Teilintervalle: $n_j = (b-a)/h_j = 2^j$ (aus $h = (b-a)/n$ folgt $n = (b-a)/h$)
- Extrapoliere via Rekursion:
  $ T_(j k) = (4^k T_(j+1,k-1) - T_(j,k-1)) / (4^k - 1) $
- Schema (z.B. $m=3$):
  $T_(00) arrow T_(01) arrow T_(02) arrow T_(03)$

  $T_(10) arrow T_(11) arrow T_(12)$

  $T_(20) arrow T_(21)$

  $T_(30)$

#colbreak()

= Gewöhnliche DGL

== Definition
- DGL $n$-ter Ordnung: $y^((n))(x) = f(x, y(x), y'(x), ..., y^((n-1))(x))$
- Anfangswertproblem (AWP): DGL + $n$ Anfangsbedingungen bei $x_0$

== Richtungsfeld
- $y'(x) = f(x, y(x))$ gibt Steigung der Lösungskurve in jedem Punkt $(x, y)$
- Lösungskurven verlaufen stets tangential zu den Richtungspfeilen

== Einschrittverfahren (1. Ordnung)
- Allgemein: $x_(i+1) = x_i + h$, $y_(i+1) = y_i + "Steigung" dot h$
- Schrittweite: $h = (b-a)/n$, Gitterstellen $x_i = a + i h$

=== Euler-Verfahren ($p=1$)
$ y_(i+1) = y_i + h dot f(x_i, y_i) $

=== Mittelpunkt-Verfahren ($p=2$)
$ x_(h\/2) = x_i + h/2, quad y_(h\/2) = y_i + h/2 dot f(x_i, y_i) $
$ y_(i+1) = y_i + h dot f(x_(h\/2), y_(h\/2)) $

*Schritte:* (1) $f$ am aktuellen Punkt auswerten, (2) halben Euler-Schritt → Mittelpunkt $(x_(h\/2), y_(h\/2))$, (3) $f$ am Mittelpunkt auswerten, (4) ganzen Schritt mit dieser Steigung

*Beispiel System (RLC-Kreis):* $L=1, R=80, C=1/2500, U=100$, $z_1=q, z_2=dot(q)$, $bold(z)^((0))=(0,0)^T$, $h=0.01$
$ f(t, bold(z)) = vec(z_2, 100 - 80 z_2 - 2500 z_1) $
$ f(0, bold(y)^((0))) = vec(0, 100), quad bold(y)_(h\/2) = vec(0,0) + 0.005 vec(0,100) = vec(0, 0.5) $
$ f(x_(h\/2), bold(y)_(h\/2)) = f(0.005, vec(z_1=0, z_2=0.5)) $
$ = vec(z_2, 100 - 80 z_2 - 2500 z_1) = vec(0.5, 100 - 80 dot 0.5 - 2500 dot 0) = vec(0.5, 60) $
$ bold(y)^((1)) = vec(0,0) + 0.01 vec(0.5, 60) = vec(0.005, 0.6) $

=== Modifiziertes Euler-Verfahren / Heun ($p=2$)
$ k_1 = f(x_i, y_i), quad y^"Euler"_(i+1) = y_i + h k_1 $
$ k_2 = f(x_(i+1), y^"Euler"_(i+1)) $
$ y_(i+1) = y_i + h dot (k_1 + k_2)/2 $

=== Klassisches Runge-Kutta ($p=4$)
$ k_1 = f(x_i, y_i) $
$ k_2 = f(x_i + h/2, y_i + h/2 k_1) $
$ k_3 = f(x_i + h/2, y_i + h/2 k_2) $
$ k_4 = f(x_i + h, y_i + h k_3) $
$ y_(i+1) = y_i + h/6 (k_1 + 2k_2 + 2k_3 + k_4) $

== DGL höherer Ordnung $arrow$ System 1. Ordnung
- DGL $k$-ter Ordnung: Einführen von Hilfsfunktionen $z_1 = y$, $z_2 = y'$, ..., $z_k = y^((k-1))$
- System: $z' = f(x, z)$ mit $z(x_0) = (y(x_0), y'(x_0), ..., y^((k-1))(x_0))^T$
- Lösung $y(x)$ steht in der ersten Komponente $z_1(x)$
- Alle Einschrittverfahren direkt auf Vektoren anwendbar ($y_i, f$ werden vektorwertig)

== Stabilität
- Stabilitätsbedingung für Euler auf $y' = -alpha y$ ($alpha > 0$):
  $ |1 - h alpha| < 1 quad arrow quad 0 < h < 2/alpha $
- Stabilitätsfunktion $g(z)$: $y_(i+1) = g(h alpha) dot y_i$
- Stabilitätsintervall: $z in (0, alpha)$ mit $|g(z)| < 1$
- Für $s$-stufige explizite Runge-Kutta: $g(z)$ ist Polynom vom Grad $s$

