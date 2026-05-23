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
    #text(size: 14pt, weight: "bold")[Höhere Mathematik 1]
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

