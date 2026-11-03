# Theory: What Bayesian Truth Serum Actually Measures in Clinical Multi-Agent Diagnosis

**Status:** draft v0.1, theory section for *"Truth Serum or Popularity Contest? Peer-Prediction Mechanisms in Clinical Multi-Agent Diagnosis"*
**Date:** 2026-11-03
**Scope:** Propositions 1, 1.2, 2, their corollaries, full proofs, and the design principle that follows.

---

## 0. What this section claims, and what it does not

Three claims, in order of importance:

1. **Under common evidence, BTS is a popularity/rarity statistic, not a truth statistic.** Its score is conditionally independent of correctness given the answer the agent gave. It ranks *answers*, not agents, and the ranking is a strictly decreasing function of how popular the answer is under the team's shared posterior.
2. **Model heterogeneity does not fix this. Only evidence heterogeneity does.** Different temperatures, prompts, or model families change the distribution of answers, but they do not create a channel from truth to score. Only private *evidence* — information not derivable from the shared case file — makes disagreement informative.
3. **Severity weighting and strict propriety cannot coexist in one additive score.** Multiplying a proper score by a severity weight tilts the optimal report to the severity-tilted posterior. The escape is architectural: separate belief elicitation from the treatment decision.

What this section does **not** claim:

- It does not claim BTS is useless. It claims BTS is a *bias detector*: it amplifies whatever the minority says. That is helpful when the team's bias makes the truth rare (sycophancy, anchoring) and harmful when the truth is common (calibrated diagnosis).
- It does not contradict Prelec (2004). Prelec's incentive result is a large-population statement under truthful forecasts. We characterise the finite-team, common-evidence regime that clinical multi-agent systems actually operate in.
- It does not assume LLM agents are Bayesian. Assumption A2 (truthful forecasts) is the load-bearing idealisation, and §7 explains how the pilot tests it rather than assuming it.

---

## 1. Setup

### 1.1 Team, evidence, truth

- A case has true diagnosis $\theta \in \Theta = \{1,\dots,m\}$.
- $E$ is the **common evidence**: history, examination, and available results visible to every agent.
- A team of $G$ agents produces answers $x_1,\dots,x_G \in \Theta$ and forecasts $y_1,\dots,y_G \in \Delta(\Theta)$, where $y_i(k)$ is agent $i$'s reported probability that a randomly chosen teammate answers $k$.
- $\mu(k) := \Pr(x_j = k \mid E)$ is the **group marginal answer distribution**. It is the object BTS implicitly estimates.

### 1.2 The BTS statistic

We use Prelec's (2004) population form, in the include-self version used by the recent LLM work (2608.25267, Definition 3):

$$
\bar x(k) = \frac{1}{G}\sum_{j=1}^{G} \mathbf{1}\{x_j = k\},
\qquad
\bar y(k) = \exp\!\left(\frac{1}{G}\sum_{j=1}^{G} \ln y_j(k)\right).
$$

The **information score** and **prediction score** of agent $i$ are

$$
I_i = \ln\frac{\bar x(x_i)}{\bar y(x_i)},
\qquad
P_i = -\sum_{k\in\Theta} \bar x(k)\ln\frac{\bar x(k)}{\bar y(k)},
\qquad
r_i = I_i + P_i .
$$

$\bar y$ is a geometric mean, which is Prelec's choice and is not normalised when forecasts disagree. The arithmetic variant $\bar y(k)=\frac1G\sum_j y_j(k)$ is common in code; §2.6 explains exactly what changes.

**Lemma 0 (zero-sum and answer-level scoring).**
For the include-self form above:

$$
\frac1G\sum_{i=1}^{G} I_i = -P_i
\quad\text{for every } i,
\qquad\text{hence}\qquad
\frac1G\sum_{i=1}^{G} r_i = 0 .
$$

Moreover $P_i$ is the same number for every agent, so for all $i,j$:

$$
r_i - r_j = I_i - I_j = \ln\frac{\bar x(x_i)}{\bar y(x_i)} - \ln\frac{\bar x(x_j)}{\bar y(x_j)} .
$$

*Proof.*
$\frac1G\sum_i \ln \bar x(x_i) = \sum_k \bar x(k)\ln\bar x(k)$ and $\frac1G\sum_i \ln \bar y(x_i) = \sum_k \bar x(k)\ln\bar y(k)$. Subtracting gives the first identity. $P_i$ depends on $i$ only through the group aggregates $(\bar x,\bar y)$, which are identical for all $i$, so it cancels in differences. $\square$

Lemma 0 is the structural fact behind everything that follows: **BTS assigns one score per answer, not one score per agent.** All agents who give the same answer receive the same reward. The mechanism can reweight factions; it cannot identify which member of a faction reasoned better.

### 1.3 Assumptions

**A1 (Common evidence).** Each answer is conditionally independent of $\theta$ given $E$:
$$
x_i \perp\!\!\!\perp \theta \mid E \qquad \text{for all } i.
$$
Equivalently: the only thing an agent knows about the case is the shared file $E$. Temperature, seed, and sampling noise are independent of $\theta$.

**A2 (Truthful forecasts).** Every agent reports the group marginal:
$$
y_j(k) = \mu(k) := \Pr(x_j = k\mid E) \qquad \text{for all } j,k.
$$
This is the idealisation that an LLM asked "what will your colleagues answer?" reports the actual distribution of colleague answers rather than its own belief.

**A3 (Exchangeability).** Given $E$, the answers $x_1,\dots,x_G$ are i.i.d. with law $\mu$.

**A4 (Full support).** $\mu(k) > 0$ for all $k\in\Theta$, so logarithms are finite.

A1–A4 describe a team that shares a case file, answers by sampling from a common posterior, and forecasts its own consensus correctly. This is the *best case* for BTS. If the mechanism fails here, it fails for a reason, not for noise.

### 1.4 Two regimes

The theory distinguishes two clinical regimes, and the empirical section should too:

- **Calibrated regime.** The team is unbiased and its shared posterior is roughly correct. Common diseases are common; rare diseases are rare and usually not the answer.
- **Biased regime.** The team is anchored, sycophantic, or otherwise pulled toward a wrong consensus. The correct answer is a minority report.

BTS behaves oppositely in the two. That is the paper.

---

## 2. Proposition 1 — Common-evidence degeneracy

### 2.1 Statement

**Proposition 1.** Assume A1–A4. Then for every agent $i$:

**(i) No truth channel.** Conditional on the answer and the case,
$$
r_i \perp\!\!\!\perp \mathbf{1}\{x_i = \theta\} \mid (x_i, E).
$$
The score carries no information about correctness beyond which answer was given.

**(ii) Consensus collapse.** If the team returns a common answer $k$ (i.e. $x_j=k$ for all $j$), then
$$
r_i = 0 \quad \text{for every } i,
$$
for every $k$, including $k\neq\theta$. A unanimously wrong diagnosis and a unanimously correct diagnosis are scored identically.

**(iii) Population limit.** If $G\to\infty$, then $I_i\to 0$ and $P_i\to 0$ almost surely, so $r_i\to 0$. Consequently $\mathrm{AUC}(r;\mathbf{1}\{x=\theta\})\to \tfrac12$.

**(iv) Rarity premium.** For finite $G$, define
$$
h_G(u) := \mathbb{E}\!\left[\ln\frac{1+\mathrm{Bin}(G-1,u)}{G}\right] - \ln u .
$$
Then
$$
\mathbb{E}[I_i \mid x_i=k, E] = h_G(\mu(k)),
\qquad
\mathbb{E}[r_i \mid x_i=k, E] = h_G(\mu(k)) + c_G(\mu,k),
$$
where $c_G(\mu,k)=\mathbb{E}[P_i\mid x_i=k,E]$ satisfies $|c_G(\mu,k)-c_G(\mu)|=O(1/G)$ with $c_G(\mu)=\mathbb{E}[P_i\mid E]=\mathbb{E}[H(\bar x)]-H(\mu)\le 0$. Crucially, $c_G$ depends on $k$ only through $\mu(k)$ (by label symmetry), never through $\mathbf{1}\{k=\theta\}$. $h_G$ is strictly decreasing on $(0,1)$, with $h_G(1)=0$ and $h_G(u)\to+\infty$ as $u\downarrow 0$.

Read (iv) carefully: the expected reward depends on the answer **only through its popularity** $\mu(k)$, and it is *larger* for rarer answers. Truth enters nowhere.

### 2.2 Proof

**(i)** Fix $(x_i,E)$. The only randomness left in $r_i$ is $\{x_j\}_{j\ne i}$ and $\{y_j\}$. By A2, $y_j=\mu(\cdot\mid E)$ is a deterministic function of $E$. By A1, $x_j\perp\!\!\!\perp\theta\mid E$ for every $j\ne i$. Hence the conditional law of $r_i$ given $(x_i,E)$ does not depend on $\theta$ at all. Since $\mathbf{1}\{x_i=\theta\}$ is a function of $(\theta,x_i)$, it is conditionally independent of $r_i$ given $(x_i,E)$. $\square$

**(ii)** If $x_j=k$ for all $j$, then $\bar x=e_k$. By A2, $\bar y=\mu$. Therefore
$$
I_i=\ln\frac{1}{\mu(k)}=-\ln\mu(k),
\qquad
P_i=-\sum_m e_k(m)\ln\frac{e_k(m)}{\mu(m)}=-\ln\frac{1}{\mu(k)}=\ln\mu(k),
$$
so $r_i=0$. This is exact, not asymptotic, and holds for every $k$. $\square$

**(iii)** By A3 and the strong law, $\bar x(k)\to\mu(k)$ a.s. for each $k$. By A4, $\mu(k)>0$, so $\ln$ is continuous at the limit. Hence
$$
I_i\to \ln\frac{\mu(x_i)}{\mu(x_i)}=0,
\qquad
P_i\to -\sum_k \mu(k)\ln\frac{\mu(k)}{\mu(k)}=0 .
$$
Since $r_i\to0$ for every agent, the score is constant in the limit and cannot rank correct above incorrect answers. $\square$

**(iv)** Condition on $x_i=k$. By A3, $S:=\sum_{j\ne i}\mathbf{1}\{x_j=k\}\sim\mathrm{Bin}(G-1,\mu(k))$, independent of $\theta$, and
$$
\bar x(k)=\frac{1+S}{G},
\qquad
\bar y(k)=\mu(k)\ \text{(A2)} .
$$
Therefore
$$
I_i=\ln\frac{1+S}{G}-\ln\mu(k),
\qquad
\mathbb{E}[I_i\mid x_i=k,E]=h_G(\mu(k)).
$$
For the full reward, $P_i=-\sum_m\bar x(m)\ln\bar x(m)+\sum_m\bar x(m)\ln\mu(m)$. Its unconditional expectation, using $\mathbb{E}[\bar x(m)]=\mu(m)$, is
$$
\mathbb{E}[P_i\mid E]=\mathbb{E}[H(\bar x)]+\sum_m\mu(m)\ln\mu(m)=\mathbb{E}[H(\bar x)]-H(\mu)=:c_G(\mu)\le 0,
$$
where $H$ is Shannon entropy and the inequality is Jensen. Conditional on $x_i=k$, the only change is that $\mathbb{E}[\bar x(m)\mid x_i=k]=\frac1G\mathbf{1}\{m=k\}+\frac{G-1}{G}\mu(m)$, which perturbs $c_G$ by $O(1/G)$ and, by label symmetry, can depend on $k$ only through $\mu(k)$. Since $\theta$ enters neither expression, neither term can depend on $\mathbf{1}\{k=\theta\}$. $\square$

It remains to prove the monotonicity of $h_G$. Write $n=G-1$ and $\varphi(u)=\mathbb{E}[\ln(1+\mathrm{Bin}(n,u))]$. Differentiating the binomial expectation,

$$
\varphi'(u)=n\,\mathbb{E}\!\left[\ln\frac{2+V}{1+V}\right],
\qquad V\sim\mathrm{Bin}(n-1,u).
$$

Using $\ln(1+z)\le z$,

$$
\varphi'(u)\le n\,\mathbb{E}\!\left[\frac{1}{1+V}\right].
$$

For $V\sim\mathrm{Bin}(n-1,u)$ the identity $\mathbb{E}[1/(1+V)]=\bigl(1-(1-u)^n\bigr)/(nu)$ holds (integrate $(1-u+uz)^{n-1}$ over $z\in[0,1]$). Hence

$$
h_G'(u)=\varphi'(u)-\frac1u
\le \frac{1-(1-u)^n}{u}-\frac1u
=-\frac{(1-u)^n}{u}<0
$$

for $u\in(0,1)$. So $h_G$ is strictly decreasing. At the endpoints, $h_G(1)=\ln 1-\ln G+\ln G=0$, and $h_G(u)\to\ln 1-\ln(Gu)\to+\infty$ as $u\downarrow0$. $\square$

**Table 1 — $h_G(u)$ for representative team sizes.** Values are nats of information score per agent. Read down a column: a rarer answer pays more. Read across a row: larger teams pay less in total (the premium is a finite-team artefact).

| $G$ | $u=0.01$ | $0.05$ | $0.10$ | $0.20$ | $0.50$ | $0.80$ | $0.95$ |
|---|---|---|---|---|---|---|---|
| 3 | 3.520 | 1.966 | 1.340 | 0.777 | 0.216 | 0.050 | 0.010 |
| 5 | 3.023 | 1.521 | 0.954 | 0.491 | 0.116 | 0.027 | 0.006 |
| 10 | 2.364 | 0.981 | 0.533 | 0.234 | 0.053 | 0.013 | 0.003 |
| 64 | 0.833 | 0.162 | 0.073 | 0.032 | 0.008 | 0.002 | 0.0004 |

### 2.3 Corollary 1.1 — model heterogeneity is not information heterogeneity

**Corollary 1.1.** Suppose agents use $M$ different models (or temperatures, or role prompts), so that agent $i$'s answer is drawn from $\mu_{m(i)}(\cdot\mid E)$ rather than a common $\mu$, but A1 still holds (every model conditions only on $E$). Then:

**(a)** With truthful forecasts, every agent reports the group mixture $\bar\mu=\frac1G\sum_j\mu_{m(j)}$, and Proposition 1(i)–(iii) hold verbatim with $\mu$ replaced by $\bar\mu$. Model diversity changes the mixture; it does not create a truth channel.

**(b)** If instead each agent forecasts its *own* model's distribution ($y_i=\mu_{m(i)}$), the reference $\bar y$ becomes a geometric average of the individual models rather than the group marginal $\bar\mu$, and the expected score becomes
$$
\mathbb{E}[I_i]\approx\sum_k\mu_{m(i)}(k)\ln\frac{\bar\mu(k)}{\bar y(k)}\ \ge 0,
$$
which rewards answers that are rare under the group reference, not answers that are correct. A confidently wrong minority agent can outscore a correct majority.

*Proof.* (a) is Proposition 1 with $\mu:=\bar\mu$; A1 is unchanged because each model's answer is $\sigma(E)$-measurable. For (b), in the population limit $\bar x\to\bar\mu$ while $\bar y$ is the geometric mean of the individual models; substituting into the definition of $I_i$ and taking expectations gives the displayed expression. $\square$

**Numerical example (a confidently wrong minority wins).** Take $G=3$, $\Theta=\{A,B\}$, true $\theta=A$. Agents 1 and 2 believe $(0.9,0.1)$; agent 3 believes $(0.1,0.9)$. Each forecasts its own belief (A2 violated), and each answers its mode, so $x=(A,A,B)$. Then

$$
\bar x=(2/3,1/3),\qquad
\bar y=\bigl((0.9^2\cdot0.1)^{1/3},(0.1^2\cdot0.9)^{1/3}\bigr)=(0.4327,0.2080).
$$

$$
I_1=\ln\frac{0.6667}{0.4327}=0.432,\qquad
I_3=\ln\frac{0.3333}{0.2080}=0.472,
$$
$$
P_i=-\bigl[0.6667\ln(0.6667/0.4327)+0.3333\ln(0.3333/0.2080)\bigr]=-0.445 \ \text{ for all } i,
$$
$$
r_1=r_2=-0.013,\qquad r_3=+0.027 .
$$

The wrong agent beats both correct agents. No truth signal was needed; only forecast heterogeneity.

### 2.4 Corollary 1.2 — the score is a rarity premium, and rarity is anti-correlated with correctness in a calibrated team

**Corollary 1.2.** Assume A1–A4 and, in addition, that answers are drawn from the shared posterior, so that $\mu=q$ and an agent who answers $k$ is correct with probability $q(k)$. Then, within a case, the expected BTS information score is a strictly decreasing function of the agent's probability of being correct.

Consequently, in the homogeneous two-answer case with $q=(q,1-q)$ and $q>\tfrac12$, every correct answer is the common one and earns $h_G(q)$, while every incorrect answer is the rare one and earns $h_G(1-q)>h_G(q)$. The score is therefore a decreasing function of correctness and the AUC is below $\tfrac12$; in the well-separated regime ($q$ far from $\tfrac12$, moderate $G$) it is close to $0$. In heterogeneous case mixes the deficit is smaller but the sign is the same whenever correct answers are not systematically rarer than incorrect ones. The score is **anti-proper**: it pays more for less likely answers.

*Proof.* Immediate from Proposition 1(iv): $\mathbb{E}[I_i\mid x_i=k]=h_G(q(k))$ and $h_G$ is strictly decreasing, while $\Pr(\theta=k\mid x_i=k)=q(k)$. $\square$

Two consequences matter clinically.

1. **In a calibrated team, BTS rewards the zebra hunter.** A wrong rare diagnosis earns more than a correct common one. The prediction score $P_i$ is common to all agents (Lemma 0) and cannot correct this.
2. **In a biased team, BTS looks like it works.** If the team is anchored on a wrong common answer, the correct answer is rare, so the score correlates positively with correctness. This is exactly the sycophancy setting of 2608.25267, where the minority is the honest answer. The mechanism is not detecting truth; it is detecting the team's bias, and the bias happens to point away from the truth.

This is the central empirical prediction: **BTS's apparent accuracy is a function of the minority-truth rate, not of the mechanism's intelligence.** Define
$$
\rho := \Pr\!\left(x_i=\theta \,\middle|\, x_i \text{ is a minority answer in its case}\right).
$$
BTS's ranking correlates positively with correctness if and only if $\rho$ exceeds the accuracy of the majority answer. Evidence partitioning (§5.3) is the only design change that raises $\rho$ by construction.

### 2.5 Corollary 1.3 — BTS is answer-level, not agent-level

**Corollary 1.3.** Under the include-self form, if $x_i=x_j$ then $r_i=r_j$ exactly. BTS cannot distinguish two agents who gave the same answer, however different their reasoning or evidence.

*Proof.* $r_i=I_i+P_i$ with $I_i$ a function of $x_i$ and the group aggregates, and $P_i$ a function of the group aggregates only (Lemma 0). $\square$

Clinical consequence: BTS cannot be used for individual accountability ("which clinician was right?"). It can only reweight factions. For a diagnostic team, that is the wrong granularity: the clinically important question is usually *which* differential to pursue, and the mechanism's answer is "the rarest one", not "the best-supported one".

### 2.6 Remark — geometric versus arithmetic aggregation

Everything in Proposition 1 holds identically for both variants, because under A2 all forecasts equal $\mu$ and the geometric mean of identical numbers is their common value.

The variants diverge only when forecasts disagree (§3). With the arithmetic mean, $\bar y=\mu$ by the tower property, the disagreement term vanishes, and the score collapses to the pure rarity premium $h_G$. With the geometric mean, $\bar y\le\mu$ by AM–GM, and the gap $\ln(\mu/\bar y)$ is a **disagreement premium**. Prelec's geometric choice is what gives BTS its "information" interpretation. Implementations that use arithmetic means are measuring pure rarity and should be described as such.

**Practical instruction for the pilot:** record which aggregation the code uses. It changes the interpretation of every AUC.

---

## 3. Proposition 1.2 — what private information actually buys

Proposition 1 assumes no private information. Real clinical teams do have it: a specialist notices a sign, a model has memorised a rare presentation, a test result is read by only one agent. This section characterises what BTS does with it.

### 3.1 Setup

Replace A1 by:

**A1′ (Private signals).** Each agent observes a private signal $s_i\sim P(\cdot\mid\theta,E)$, conditionally independent across agents given $(\theta,E)$, and answers $x_i\sim p_i(\cdot):=P(\cdot\mid E,s_i)$.

**A2′ (Truthful forecasts).** Each agent reports its own posterior predictive: $y_j=p_j$.

**A3′ (Exchangeable answer rules).** Conditional on $(\theta,E)$, the pairs $(s_i,x_i)$ are i.i.d. across agents.

Let

$$
\mu(k)=\mathbb{E}[p_j(k)\mid E]
\qquad\text{(arithmetic marginal)},
\qquad
g(k)=\exp\!\bigl(\mathbb{E}[\ln p_j(k)\mid E]\bigr)
\qquad\text{(geometric mean)}.
$$

By AM–GM, $g(k)\le\mu(k)$.

### 3.2 Statement

**Proposition 1.2 (surprise decomposition).** Assume A1′, A2′, A3′. Then, in the population limit and up to $O(1/G)$,

$$
\mathbb{E}[I_i\mid E]
=\sum_{k\in\Theta}\mu(k)\ln\frac{\mu(k)}{g(k)}
=: \mathcal{D}(\mu\,\|\,g)\ \ge 0,
$$

and, conditional on the agent's own signal,

$$
\mathbb{E}[I_i\mid s_i=t,E]
=\sum_{k\in\Theta}p_t(k)\ln\frac{\mu(k)}{g(k)}+O(1/G).
$$

The per-answer quantity
$$
\mathrm{surprise}(k):=\ln\frac{\mu(k)}{g(k)}
$$
is the **disagreement premium**. It is zero for every $k$ if and only if all agents' predictives coincide almost surely, and it is strictly positive for an answer about which the team disagrees.

### 3.3 Proof

By A3′ and the law of large numbers, $\bar x(k)\to\mu(k)$ and $\ln\bar y(k)\to\mathbb{E}[\ln p_j(k)]=\ln g(k)$. Hence
$$
I_i\to\ln\frac{\mu(x_i)}{g(x_i)}=\mathrm{surprise}(x_i).
$$
Taking expectations, and using $\Pr(x_i=k)=\mu(k)$ by definition of the marginal,
$$
\mathbb{E}[I_i\mid E]=\sum_k\mu(k)\ln\frac{\mu(k)}{g(k)}.
$$
Non-negativity follows pointwise from AM–GM: $g(k)\le\mu(k)$ for every $k$, so each log term is $\ge0$. Equality holds iff $\mu(k)=g(k)$ for every $k$ with $\mu(k)>0$, which is equality in AM–GM and therefore equivalent to $p_j(k)$ being almost surely constant across $j$ for each $k$ — i.e. no disagreement. Conditioning on $s_i=t$ replaces $\mu(k)$ by $p_t(k)$ in the outer weight and leaves $\mathrm{surprise}(k)$ unchanged. $\square$

### 3.4 What the decomposition says

The score has exactly two channels:

$$
\text{score} \;=\; \underbrace{\text{rarity premium}}_{\text{how uncommon the answer is}} \;+\; \underbrace{\text{disagreement premium}}_{\text{how contested the answer is}} .
$$

Neither channel contains $\theta$. Both are properties of the **distribution of beliefs**, not of the truth. This yields the sharpest statement of the paper:

> BTS cannot tell private evidence from private idiosyncrasy, because both produce the same statistic: disagreement about an uncommon answer.

Private evidence is *necessary* for the score to carry truth signal (without it, $\mathcal{D}=0$), but it is not *sufficient*: a team of three different model families disagreeing about a rare disease produces exactly the same score as a team of three specialists who each hold a piece of the puzzle. The mechanism is blind to which is which.

### 3.5 Corollary 1.2.1 — confirming a common diagnosis is punished

**Corollary 1.2.1 (comparative static: confirming a common diagnosis is punished).** Hold the group marginal $\mu$ and the forecasts fixed (A2). Consider two teams that differ only in one agent: in team U the agent is uninformed and answers from $\mu$; in team I the agent holds a private belief $p_t\ne\mu$. Then
$$
\mathbb{E}[I_i\mid s_i=t]=\sum_k p_t(k)\,h_G(\mu(k)),
\qquad
\mathbb{E}[I_{\text{U}}]=\sum_k \mu(k)\,h_G(\mu(k)).
$$
If $p_t$ shifts mass from rarer answers toward a common answer $\theta$ with large $\mu(\theta)$, then team I's informed agent earns **less** than team U's uninformed agent.

*Proof.* Substitute $p_t$ into Proposition 1(iv) and use that $h_G$ is strictly decreasing. A mean-preserving shift of mass toward a high-$\mu$ (low-$h_G$) coordinate lowers the weighted average. $\square$

**Numerical example (G=5, two answers).** Let $\mu=(0.95,0.05)$ (common vs rare). From the table in §2.2, $h_5(0.95)=0.0057$ and $h_5(0.05)=1.5207$.

| Agent | Belief $(p_C,p_R)$ | Expected $I_i$ | Relative to uninformed |
|---|---|---|---|
| Uninformed (answers from $\mu$) | $(0.95,0.05)$ | $0.0814$ | $1.00\times$ |
| Confirms common disease | $(0.99,0.01)$ | $0.0208$ | $0.26\times$ |
| Coin-flip on rare disease | $(0.50,0.50)$ | $0.7630$ | $9.4\times$ |

A 99%-confident correct diagnosis of the common disease earns **74% less** than an uninformed guess, while a coin-flip on the rare disease earns **37 times** as much. This is not a pathology of an adversarial example; it is the mechanism working as designed. BTS is a rarity detector.

The table holds the group marginal $\mu$ fixed (partial equilibrium: the agent is one of many). If the team is small, the agent's own answer moves $\mu$ by $O(1/G)$; the qualitative conclusion is unchanged and the numeric values shift slightly.

---

## 4. Proposition 2 — severity weighting breaks strict propriety

### 4.1 Proper scoring rules

Let $S:\Delta(\Theta)\times\Theta\to\mathbb{R}$ be a scoring rule for a reported belief $r$. $S$ is **strictly proper** if, for every prior $p$,
$$
\mathbb{E}_{\theta\sim p}[S(p,\theta)] > \mathbb{E}_{\theta\sim p}[S(r,\theta)]
\qquad\text{for all } r\ne p .
$$
The log score $S(r,\theta)=\ln r(\theta)$ is the canonical example; the Brier score is another.

### 4.2 Statement and proof

**Proposition 2 (severity tilt).** Let $w:\Theta\to(0,\infty)$ be a severity weight (e.g. harm of missing $\theta$, or cost of treatment). Define the weighted score $S_w(r,\theta):=w(\theta)S(r,\theta)$ and the tilted distribution
$$
q(\theta):=\frac{p(\theta)w(\theta)}{W},
\qquad W:=\sum_{\theta}p(\theta)w(\theta).
$$
Then
$$
\mathbb{E}_{\theta\sim p}[S_w(r,\theta)]
= W\,\mathbb{E}_{\theta\sim q}[S(r,\theta)] .
$$
If $S$ is strictly proper, the unique maximiser of the weighted score is $r^\star=q$, not $p$. Truthful reporting $r=p$ is optimal if and only if $w$ is constant on $\mathrm{supp}(p)$.

*Proof.* Linearity of expectation:
$$
\sum_\theta p(\theta)w(\theta)S(r,\theta)
= W\sum_\theta \frac{p(\theta)w(\theta)}{W}S(r,\theta)
= W\,\mathbb{E}_{q}[S(r,\theta)].
$$
Since $S$ is strictly proper, the right-hand side is uniquely maximised at $r=q$. The final claim follows because $q=p$ iff $w$ is proportional to a constant on the support. $\square$

**What this means clinically.** The weight is supposed to say "missing this disease is worse". In a single additive score it says "this disease is more likely than it is". A 5× severity weight on a disease with true probability 0.2 turns it into the most likely disease ($q=(0.444,0.556)$ in a two-disease example), and the agent's optimal report flips from the common disease to the severe one. The mechanism does not become harm-aware; it becomes **biased toward severe diagnoses**. In a rare-disease clinic that is over-investigation; in an emergency department it may be defensible, but it is a *prior shift*, not a utility calculation, and it should be chosen and reported as such.

### 4.3 Corollary 2.1 — impossibility

**Corollary 2.1.** Fix $p$ in the interior of the simplex. No scoring rule $R$ can satisfy both:

**(a) strict propriety:** $\arg\max_r \mathbb{E}_{p}[R(r,\theta)]=\{p\}$, and
**(b) severity invariance:** $\arg\max_r \mathbb{E}_{p}[w(\theta)R(r,\theta)]=\{p\}$ for every $w>0$.

*Proof.* Suppose (b) holds for all $w>0$. Put $c(\theta)=p(\theta)w(\theta)$, which ranges over $(0,\infty)^m$ as $w$ does. Then for every $c$ in an open set, $p$ maximises $F_c(r)=\sum_\theta c_\theta R(r,\theta)$. Since $p$ is interior, the first-order condition gives
$$
\nabla_r F_c(p)=\sum_\theta c_\theta \nabla_r R(p,\theta)=0
\qquad\text{for all } c \text{ in an open set.}
$$
The vectors $\{\nabla_r R(p,\theta)\}_\theta$ therefore satisfy a linear relation with every coefficient vector $c$, which forces $\nabla_r R(p,\theta)=0$ for each $\theta$. Since $p$ was arbitrary, $\nabla_r R(p,\theta)=0$ for all $p,\theta$, so $R$ is constant in $r$ and cannot be strictly proper. Contradiction. $\square$

(For non-differentiable rules, replace the gradient condition by a supporting-hyperplane argument; the conclusion is unchanged.)

**Scope.** The impossibility is about *multiplicative severity weighting of a single score*. It does not rule out (i) scoring beliefs and deciding actions in separate layers, (ii) restricting $w$ to a one-parameter family, or (iii) making the weight a function of the *action* rather than the *disease*. Those are the escapes, and §5 uses the first.

### 4.4 What this means for BTS specifically

BTS does not score a belief report; it scores an answer. So "multiply by $w(\theta)$" is not well defined until we say what is being weighted. There are two coherent readings, and both fail to do what severity weighting is supposed to do.

**Proposition 2.2 (BTS severity is a no-op or a distortion).** Assume A1–A4.

**(a) Reward weighting.** If the realised BTS reward is multiplied by $w(\theta)$, then
$$
\mathbb{E}[w(\theta)r_i\mid x_i=k,E]=W\bigl(h_G(\mu(k))+c_G(\mu,k)\bigr),
\qquad W=\mathbb{E}[w(\theta)\mid E],
$$
so the weight rescales every agent's reward in a case by the same constant $W$. It does not change the **within-team ranking**, and therefore cannot make the mechanism prefer severe diagnoses. It does change the *across-case* scale of the reward, which is a training-gradient effect, not a decision effect.

**(b) Reference tilting.** If instead the forecasts are computed under the tilted marginal $\mu_w\propto\mu w$ (so that "severe diseases are treated as more likely"), then
$$
\mathbb{E}[I_i\mid x_i=k]\approx\ln\frac{\mu(k)}{\mu_w(k)}+\frac{1-\mu(k)}{2G\,\mu(k)} .
$$
Because $\mu_w(k)>\mu(k)$ for severe $k$, the score of a severe answer **falls**. The tilt penalises exactly the answers it was meant to promote.

*Proof.* (a) By A1, $\theta\perp\!\!\!\perp x_i\mid E$, so $\mathbb{E}[w(\theta)\mid x_i=k,E]=W$; then apply Proposition 1(iv) and Proposition 1(i). (b) Substitute $\bar y=\mu_w$ into $\mathbb{E}[I_i\mid x_i=k]=\mathbb{E}[\ln\bar x(k)]-\ln\mu_w(k)$ and use the expansion in the proof of Proposition 1(iv). $\square$

The clinical lesson is not "weight harder". It is that **severity belongs in the decision, not in the score**. That is the architectural fix.

---

## 5. The fix: separate elicitation from decision

The theory points to a three-part design. None of it rescues BTS as a truth serum; together they make the team's disagreement *mean something*.

### 5.1 Layer 1 — elicit beliefs with a proper score

Ask each agent for a distribution $b_i\in\Delta(\Theta)$ over diagnoses, not just a top-1 answer. Score it with the unweighted log score:
$$
S(b_i,\theta)=\ln b_i(\theta).
$$
By strict propriety, the agent's optimal report is its true posterior. No severity weight enters here (Proposition 2 is not triggered).

Why this matters for LLM training: if the reward is the proper score, the policy's best response is truthful reporting regardless of what the decision layer later does, *provided the reward is the objective*. That proviso is true for RL fine-tuning and false for a self-interested human expert whose utility is the clinical outcome. State it explicitly.

### 5.2 Layer 2 — put severity in the decision

Pool the beliefs (normalised geometric mean is the maximum-likelihood pool under conditional independence):
$$
\bar b(\theta)\propto\prod_i b_i(\theta)^{1/G}.
$$
Choose the action (test, treat, refer, discharge) by Bayes rule under the weighted posterior:
$$
a^\star=\arg\min_{a}\sum_\theta \bar b(\theta)\,w(\theta)\,L(a,\theta).
$$
Severity now does what it is supposed to do: it changes the *action*, not the *belief*. A 5× weight on a severe disease makes the team order a test, not hallucinate a diagnosis.

### 5.3 Evidence partitioning — make disagreement informative

Proposition 1.2 says the score's truth channel exists only if agents hold private evidence. So build the team that way:

- Partition the workup into disjoint slices $E_1,\dots,E_G$ (e.g. history, exam, labs, imaging, prior records).
- Give each agent only its slice plus a shared problem statement. No agent sees the full file.
- Require each agent to cite the slice item supporting its answer.
- Aggregate with Layer 1–2 above.

Now a minority report is evidence-backed by construction, so rarity and information are aligned. This is the only intervention in the theory that raises the minority-truth rate $\rho$ without changing the scoring rule.

### 5.4 BTS as a bias monitor, not a reward

Keep BTS, but use it diagnostically:

- Compute the surprise decomposition of §3.2 for every case.
- Flag cases where surprise is concentrated on answers with no cited private evidence. That is a popularity contest, not a consultation.
- Track the minority-truth rate $\rho$ and forecast calibration (ECE). If $\rho$ is at or below the base rate, BTS's ranking is anti-correlated with correctness and should not be used to select answers.

### 5.5 What the fix buys, and what it does not

| | Buys | Does not buy |
|---|---|---|
| Layer 1 | truthful belief reports; no severity tilt | accuracy; a team of identical beliefs adds nothing |
| Layer 2 | severity-aware actions with clinical utility | truthful reporting by self-interested experts |
| Partitioning | disagreement that is evidence-backed | a guarantee that evidence is used correctly |
| BTS monitor | early warning of popularity contests | a truth serum |

The honest summary: **BTS is a useful instrument for measuring how a team disagrees, and a bad instrument for deciding who is right.**

---

## 6. Falsifiable predictions

Each prediction maps to one cell of the 2-week pilot. "Score" means the BTS statistic; "confidence" means the agent's own top-1 probability.

| # | Cell | Prediction | Test statistic | Falsified if |
|---|---|---|---|---|
| P1 | H0: one model, one prompt, T=0 | Consensus collapse; score variance ≈ 0; AUC undefined | variance, agreement rate | AUC > 0.55 with non-zero variance |
| P2 | H1: one model, T=0.7, 5 samples | $\Delta\mathrm{AUC}=\mathrm{AUC}(\text{score})-\mathrm{AUC}(\text{confidence})\approx 0$ | DeLong test | $\Delta$AUC > 0.10, significant |
| P3 | H2: one model, 5 role prompts | $\Delta\mathrm{AUC}\approx 0$; role prompts are not private evidence | DeLong test | $\Delta$AUC > 0.10 |
| P4 | H3: 3 model families | $\Delta\mathrm{AUC}>0$ but small, and explained by forecast dispersion | regression of AUC on ECE | $\Delta$AUC > 0.15 unexplained by ECE |
| P5 | H4: disjoint evidence slices | $\Delta\mathrm{AUC}\ge 0.15$; $\rho$ rises vs H3 | DeLong + $\rho$ | no recovery |
| P6a | All cells | **Rarity premium:** correct rare diagnoses score above correct common ones | stratified mean difference | premium < 5 points |
| P6b | All cells | **Discrimination gap:** within the same rarity stratum, correct and wrong answers score alike | score(correct rare) − score(wrong rare) | \|gap\| > 0.10 nats |
| P7 | Sycophancy control | Under user pressure, minority = truth, so AUC > 0.5 and BTS-guided selection reduces flip rate | flip rate | no reduction (fails to replicate 2608.25267's direction) |
| P8 | Oracle forecasts | Replacing elicited forecasts with the empirical answer distribution drives $\Delta$AUC→0 in H1–H3, but not in H4 | ablation | $\Delta$AUC survives oracle forecasts in H1–H3 |

P8 is the most important ablation. If the apparent truth signal disappears when forecasts are oracle, then the signal was **forecast miscalibration**, not information. That is the single result that would most damage the paper if it went the other way, so run it first.

**Decision rule.** GO if P1, P2, P6a and P8 hold and P5 shows recovery. Report P6b as an equivalence test (the gap is *predicted* to be near zero), not as a null result. The paper is then: *BTS measures rarity, not truth; here is the decomposition, the impossibility, and the fix.* If P5 fails, the paper is the negative result plus the monitor, which is a workshop paper. If P2 fails (score tracks truth under common evidence), the theory's A1 is violated and the paper pivots to "BTS measures forecast miscalibration", which is still publishable but a different paper.

---

## 7. Limitations and threats to validity

1. **A2 is strong.** LLM forecasts are elicited by prompt, not computed from a posterior. If they are miscalibrated, the "disagreement premium" includes forecast error, and the score can look informative for the wrong reason. The pilot must measure ECE and run P8.
2. **A1 is an idealisation.** LLM answers depend on memorised pretraining data, which is a form of private information. This is exactly the confound between H3 and H4; the experiment is designed to separate them, and the theory does not assume they are separable without the partition.
3. **Posterior sampling is a model, not a fact.** Temperature sampling is not posterior sampling and greedy decoding is deterministic. We treat the answer distribution as the object of interest and do not claim it is calibrated.
4. **Small teams.** Clinical teams have $G=3$–$7$, far from the population limit. The $O(1/G)$ rarity premium is *larger* in small teams, which strengthens the critique but weakens any "truth serum" reading of the score. Prelec's incentive properties are known to fail at $G=3$ (Witkowski & Parkes 2012); our claims are about measurement, not equilibrium.
5. **Severity weights are normative.** We do not claim a correct $w$. We claim that whatever $w$ is chosen cannot be folded into a proper score without distortion. Who sets $w$ (patient, clinician, payer) is a governance question, not a statistical one.
6. **Corollary 2.1 assumes unconstrained $w$ and differentiability.** Constrained families (e.g. a single risk-aversion parameter) may escape the impossibility; we state the scope and do not overclaim.
7. **Single-case analysis.** Multi-turn teams update beliefs; the theory applies to the terminal round. Herding and anchoring are the bias channel the audit is meant to detect, not a violation of the model.
8. **No equilibrium claim.** We do not prove that truthful reporting is a Bayes–Nash equilibrium for finite $G$. We characterise what the statistic measures ex post. This is deliberate: the paper is an audit, not a mechanism-design proposal.

---

## 8. Relation to prior work

| Work | What it establishes | What this section adds |
|---|---|---|
| Prelec (2004), *A Bayesian Truth Serum for Subjective Data* | truthfulness is optimal for large populations under truthful forecasts | the finite-team, common-evidence regime; the rarity decomposition; the clinical failure modes |
| Prelec, Seung & McCoy (2017), *A solution to the single-question crowd wisdom problem* | the "surprisingly popular" algorithm; rarity carries signal | the same rarity logic, but shows it is *anti*-correlated with correctness in a calibrated diagnostic team |
| Witkowski & Parkes (2012) | BTS fails IC/IR at $G=3$ | cited for the small-team caveat; our result is about measurement, not incentives |
| 2608.25267 (BTS + GRPO for sycophancy) | BTS reward cuts answer-flip rate 23%→4% | explains *why* it works there (sycophancy makes truth rare) and predicts non-transfer to calibrated diagnosis; P7 is the replication |
| 2601.20299 (peer prediction for LLM training) | peer-prediction signals can train LLMs | our decomposition applies to this family as well: the signal is disagreement, not truth |
| 2610.33728 (Audited Pivot Scoring) | auditing LLM multi-agent systems | complementary: they audit agents, we audit the *mechanism* |
| 2509.25184 (peer prediction with conflicting sources) | peer prediction under source conflict | our setting is the clinical special case where conflict is base-rate-driven |
| 2606.13003 (Illusion of Multi-Agent Advantage), 2604.02460 (equal-token budgets) | MAS gains are often not what they seem | a mechanism-level explanation for one such illusion |

**Novelty statement (one paragraph).** The individual ingredients — BTS's rarity premium, the proper-scoring impossibility under reweighting, and the AM–GM decomposition — are each known in isolation. The contribution here is to put them together for clinical multi-agent diagnosis and derive the consequence that matters: under common evidence the mechanism is a popularity statistic that is conditionally independent of correctness, model heterogeneity does not repair it, severity weighting either does nothing or backfires, and the only repair is to partition the evidence. That yields a testable prediction (the minority-truth rate $\rho$) that explains both the sycophancy successes and the diagnostic failures of the same mechanism.

---

## Appendix A — Auxiliary facts used

**A.1 (Binomial reciprocal identity).** For $V\sim\mathrm{Bin}(n,u)$,
$$
\mathbb{E}\!\left[\frac{1}{1+V}\right]=\frac{1-(1-u)^{n+1}}{(n+1)u}.
$$
*Proof.* Integrate the probability generating function: $\int_0^1(1-u+uz)^n\,dz=\frac{1-(1-u)^{n+1}}{(n+1)u}$, and expand $(1-u+uz)^n=\sum_k\binom{n}{k}u^k(1-u)^{n-k}z^k$; the integral of $z^k$ is $1/(k+1)$. $\square$

(In §2.2 the identity is applied with $n$ replaced by $G-2$, i.e. to $V\sim\mathrm{Bin}(G-2,u)$.)

**A.2 (Entropy contraction).** $\mathbb{E}[H(\bar x)]\le H(\mu)$, with equality iff $\mu$ is a point mass. Used for $c_G(\mu)\le0$; this is Jensen's inequality for the concave entropy.

**A.3 (AM–GM gap).** For a probability vector $\mu$ and the geometric mean $g$ of a family of probability vectors with arithmetic mean $\mu$,
$$
\sum_k\mu(k)\ln\frac{\mu(k)}{g(k)}\ge0,
$$
with equality iff the family is degenerate. *Proof.* AM–GM applies coordinatewise: $g(k)\le\mu(k)$ for every $k$, so every term in the sum is non-negative. Equality in every term forces $p_j(k)$ to be constant across $j$ for each $k$. $\square$

---

## Appendix B — Summary of symbols

| Symbol | Meaning |
|---|---|
| $\theta$ | true diagnosis |
| $E$ | common evidence |
| $\mu(k)$ | group marginal answer distribution |
| $g(k)$ | geometric mean of agents' forecasts |
| $x_i$ | agent $i$'s answer |
| $y_i$ | agent $i$'s forecast |
| $\bar x,\bar y$ | realised answer frequency, geometric mean forecast |
| $I_i,P_i,r_i$ | information score, prediction score, total reward |
| $h_G(u)$ | expected information score of an agent answering an option of popularity $u$ |
| $c_G(\mu,k)$ | prediction-score term; depends on the answer only through $\mu(k)$ |
| $\mathrm{surprise}(k)$ | $\ln(\mu(k)/g(k))$, the disagreement premium |
| $\rho$ | minority-truth rate |
| $w(\theta)$ | severity weight |
| $q$ | severity-tilted posterior $q\propto pw$ |
