# Whose Efficiency? The Undefended Normative Core of Algorithmic Refugee Assignment

Julian Weide  
Fairness and Algorithms (851-0746-00L), ETH Zurich, Spring 2026

*Replication code:* <https://github.com/egal21348i/geomatch-mapping-function>

---

## 1. Introduction

Debates about algorithmic fairness are usually organised around a trade-off: more accuracy or more equity, but not both, and the question is where along the frontier to sit. Corbett-Davies et al. (2017) give the canonical formulation, and critics of algorithmic refugee assignment have adopted the same grammar, reading these tools' efficiency as something purchased at the price of equity (Alajak et al. 2026).

The grammar is wrong for this case, and getting it wrong has concealed something more troubling than a hard trade-off. GeoMatch, developed by the Immigration Policy Lab at Stanford and ETH Zurich and piloted by the Swiss State Secretariat for Migration (SEM) since 2020, does not maximise employment. It maximises a household-level threshold: the probability that *at least one* member of a refugee family finds work. Two consequences follow from the shape of that formula rather than from the data it is fed. It prefers concentrating employment prospects in one household member to spreading them evenly, regardless of which is better for the household; and it directs the benefits of optimisation away from large households, whose scores grow insensitive to location as they add members. The rule sacrifices both the aggregate quantity it is praised for maximising and the distribution its critics accuse it of ignoring.

The claim is not speculative. Bansak et al.'s own supplementary materials (fig. S8) report that replacing the at-least-one metric with the within-case mean raises the employment gain over the status quo from 41% to 47% in the U.S. backtest: the distribution-neutral alternative is more efficient. There is no frontier to sit on, and the designers do not claim there is; they justify the choice, as Section 2 shows, by the programme's goal of generating self-sufficient refugee families. That answers the question in my title, if only by inference from their justification: the beneficiary is not the refugee whose employment probability the tool nominally maximises but the fiscal position of the host state, for which one earner suffices — a legitimate goal, but not the one the tool's public case is built on, which is the employment gains it delivers to refugees.

Two things this paper does not claim. It is not that the tool harms the people it assigns: at the specification calibrated to its own published gain, every group I can construct does better under GeoMatch than under the legal default, and Section 5 says so. And it is not that the mean is the right objective; it is that four candidates were available, their consequences differ by more than the amount at stake in most of the fairness literature's usual trade-offs, and the choice among them was never argued. Section 4 calibrates the simulation to the pair of numbers just quoted — 41% under the deployed rule, 47% under its mean variant — so that the magnitudes reported there belong to the problem GeoMatch actually solves rather than to a convenient one.

## 2. Where the Normative Decision Sits

GeoMatch has three stages (Bansak et al. 2018). The *modeling* stage fits gradient-boosted trees separately per location, predicting for each individual and location the probability of the target outcome — in Switzerland, employment two to four years after arrival in the ZEMIS registry. The *mapping* stage aggregates those predictions to the level at which assignment actually occurs, the case or family, via a function $\varphi$; the *matching* stage then maximises the average case-level score subject to capacity constraints.

Critical attention has gone to the first stage, and formal fairness work to the last (Strasser Ceballos and Kern 2025). Alajak et al. (2026) focus on the predictors: nationality, gender, age, marital status and religion enter the Swiss feature vector directly, and a model built on historically stratified outcomes will reproduce that stratification. Strasser Ceballos and Kern (2025) show how sensitive the pipeline is to the target variable — optimising German data for social rather than economic integration sends 88% of refugees elsewhere. Both critiques concern what goes *into* the model.

That values enter through the formulation of the objective rather than through the estimator is the general lesson of the problem-formulation literature (Passi and Barocas 2019), and Obermeyer et al. (2019) is its canonical demonstration: a health-risk model that tracked costs instead of illness produced a large racial disparity from a label chosen upstream of any modelling decision. The argument below is that literature applied to a case in which the object quietly substituted is not the label but the unit whose welfare is counted — and to a tool for which the substitution has been visible in a public archive for eight years.

The middle stage is where the decisive normative commitment is made, and it is the one stage nobody argues about: the modeling stage produces individual-level objects, the matching stage consumes case-level ones, and $\varphi$ is the bridge whose selection determines whose welfare the optimisation tracks. It is not hidden: Strasser Ceballos and Kern (2025, Appendix C) tabulate it for both deployed tools of this family, and the comparison is instructive, because the other chose differently. Annie MOORE, running at a U.S. resettlement agency since 2018, aggregates by the *sum* of individual probabilities and so maximises the expected number of employed refugees (Ahani et al. 2021). Two systems on the same three stages optimise different things, and neither paper remarks on it. GeoMatch's choice is explicit in a four-line function in the authors' replication archive (Bansak et al. 2018, Harvard Dataverse doi:10.7910/DVN/MS8XES, `func_M_to_Mstar.R`):

```r
compute_femp_prob <- function(x){
  return(1-(prod(1-x)))
}
```

This is $\varphi(p) = 1 - \prod_i (1 - p_i)$, the probability that at least one member is employed under within-case independence. It is the default for both the U.S. and Swiss backtests, not an option tried once: the authors tested the mean, maximum and minimum, and chose this one because it "best reflects the underlying goal of the refugee resettlement program to generate self-sufficient refugee families."

That is a candid statement that the objective encodes a policy goal rather than a welfare measure: household self-sufficiency is satisfied once a family stops drawing social assistance, and not further by a second earner. Nor is the choice stable across the designers' own later work — Acharya et al. (2022), extending GeoMatch to refugees' preferences and co-authored by two of the original team, use the mean without remarking on the switch. Not because the choice is immaterial, as Sections 3 and 4 show, but because $\varphi$ has never been treated, even by its designers, as a commitment needing defence in either direction.

One limitation should be stated plainly: whether the SEM's live pipeline still uses this function is not published, and neither are the pilot's results (Strasser Ceballos and Kern 2025). What is secure is that it produced the founding study's results and was justified by the programme's goals; if a different $\varphi$ is used today, that change is equally undocumented — an instance of the problem described here, not an exception.

## 3. Inequality Within Households, Disadvantage Between Them

Two properties of $\varphi(p) = 1 - \prod_i (1 - p_i)$ do the work below, each following in a line. First, $\log(1-p)$ is concave, so $\sum_i \log(1 - p_i)$ is largest when the $p_i$ are equal; and since $\varphi = 1 - \exp\left(\sum_i \log(1-p_i)\right)$ is *decreasing* in that sum, among locations offering the same *average* prospect $\varphi$ ranks the one that spreads it evenly lowest and the most concentrated one highest. Second, $\partial\varphi/\partial p_i = \prod_{j \neq i}(1 - p_j)$: the weight the objective places on any member falls as the other members' prospects rise. The rule therefore attends least to whoever is least likely to find work, for any size and any probabilities, and an assignment maximising it can be dominated in expected employment by one that does not:

| | adult 1 | adult 2 | at-least-one score | mean | expected employed adults |
|---|---|---|---|---|---|
| Canton A | 0.80 | 0.10 | **0.820** | 0.450 | 0.90 |
| Canton B | 0.50 | 0.50 | 0.750 | 0.500 | 1.00 |

GeoMatch sends this household to A, worse both in expected employment and in equality, not through error but through faithful maximisation of its stated objective.

The rule is, in this respect, a formal implementation of the unitary household model — the household treated as one agent with one welfare level, satisfied once its need is met — which development economics abandoned some time ago, and for this reason: aggregating to the household conceals who inside it gains, and the concealment is not distributionally neutral where bargaining power and labour-market access differ by gender (Sen 1990; Agarwal 1997). Targeting a benefit at the household rather than at the person is a recognised mechanism of gender inequality in that literature. The novelty here is not the mechanism but its site: it is running inside an optimiser, where the aggregation is not an approximation forced by data but a line of code with three documented alternatives.

Who bears this? In Swiss refugee households the secondary earner is overwhelmingly a woman: seven years after arrival, 64% of men but only 32% of women aged 16–55 at entry are in employment (Staatssekretariat für Migration 2026). Two qualifications belong with that figure. It is a seven-year rate, whereas the Swiss target variable is employment two to four years after arrival, so it overstates the levels the tool works with but not the direction of the gap, which is present at every horizon the SEM reports. And it is a population average, not a household-level one; the claim below needs only that the woman is more often the lower-probability member, which this gap makes true on average rather than in every case. That gap is the raw material the formula exploits. Alajak et al. (2026) allege discriminatory outcomes by ethnicity, gender, age and marital status but cannot identify a mechanism; the argument above supplies one, located not in the training data but in a design choice downstream of it, and it reaches every group they name: whoever starts behind is served worse the further behind they start.

The second property compounds across members, which produces a distinct effect on size. Each additional member multiplies the derivative by a factor below one, so a large household's score barely moves with location: a four-adult household with middling probabilities scores above 0.93 in every canton, while a single adult's score ranges over the full cantonal spread. The matching stage allocates capacity by marginal gain — a slot in a well-matched canton goes to whichever case's score rises most from receiving it — so households whose scores barely move are the natural candidates for displacement. The objective thus steers the benefits of optimisation toward small cases for a reason unrelated to need or prospects: in the vocabulary of Black et al. (2022), a failure of vertical equity.

The two harms need different names. Sacrificing one member's prospects for another's, where the member sacrificed is disproportionately a woman, has the structure of indirect discrimination: a facially neutral rule with a foreseeable adverse effect distributed along a protected characteristic. It does not follow that the legal test is met, and Section 4 will show that it is not on the comparison that matters: indirect discrimination requires a particular disadvantage relative to a comparator, and relative to the legal default women are not disadvantaged — they gain, only less than they would under a rule that counted them once. What survives is the structure without the violation, which is why the objection is pressed below as one about efficiency and the distribution of the gain rather than as a claim of unlawfulness. Displacing a household for its size is a different matter again, since nobody inside the case is sacrificed for anyone else, which is why vertical equity — needing no protected characteristic, only that likes be treated alike — is the right frame for it.

## 4. Magnitude

### 4.1 Calibration

Probabilities are generated on the logit scale from the SEM gender base rates (men 0.64, women 0.32) plus an individual effect, a cantonal effect, a canton-by-gender interaction, and a person-by-canton term. Capacities are the statutory population-proportional shares of all twenty-six cantons (Annex 3 AsylV 1), and the matching stage replicates the published code: each canton is duplicated by its slots and a one-to-one optimal assignment on cost $1 - \varphi$ is solved. The comparison is against the characteristics-blind key of Article 21, over draws of the cantonal parameters, so that no conclusion rests on one draw.

The last term carries the analysis and so should not be chosen freely. It is the extent to which a canton that suits one person does not suit another, which is the entire resource an assignment algorithm has: with no such term, everyone of a given gender ranks the cantons identically, optimisation can only shuffle people between cantons they agree about, and the deployed rule gains 1.4% over the status quo where the best of the four candidates manages 2.3% — two orders of magnitude below what GeoMatch is documented to deliver. A simulation in that regime is not a model of the problem the tool solves, and the results it produces turn out to depend on the degeneracy (below, and Section 4.4).

So the term is pinned to the published effect size rather than assumed. Setting its standard deviation to 1.0 on the logit scale, the deployed at-least-one rule gains 39.4% over the status quo and its mean variant 47.3% — against the 41% and 47% that Bansak et al. report for the U.S. backtest and its fig. S8 comparison. One parameter reproduces both, which is as much external validation as a public-data replication of this tool admits. The gender intercepts are re-solved for every specification so that the status-quo employment rates stay at the SEM base rates throughout; specifications therefore differ in synergy alone. Results are reported at this calibration, with the no-synergy limit and a doubled upper bracket carried alongside.

### 4.2 Efficiency and the Gender Gap

Over 400 draws with 1,000 two-adult households each; at equal household size the sum and the mean coincide, leaving three distinct rules.

| $\varphi$ | employed adults per household | gain over status quo | P(employed), women | within-household gap |
|---|---|---|---|---|
| status quo | 0.962 | — | 0.325 | 0.373 |
| at-least-one | 1.338 | +39.1% | 0.471 | **0.450** |
| mean | 1.415 | **+47.1%** | 0.600 | 0.266 |
| minimum | 1.371 | +42.5% | **0.629** | **0.160** |

90% Monte Carlo intervals over the parameter draws, for the at-least-one and mean rows: gain +36.8 to +41.4% and +45.1 to +49.3%; P(employed), women 0.436–0.508 and 0.570–0.631. These describe dispersion across stylised cantonal parameters, not sampling error.

At-least-one is the worst of the three on every column: a smaller gain than either alternative, a lower employment probability for women, and the only row of the four that widens rather than narrows the within-household gap. Mean wins on efficiency, minimum on equity, at-least-one on neither, so no efficiency-equity frontier remains on which to locate it. Note what the table does not show: women are not left absolutely worse off than under the status quo, at 0.471 against 0.325 — the tool helps them, and substantially. The objection is the narrower one: the rule forgoes employment, and spends the gain it does achieve on the member who already had the better prospects.

The magnitudes are not small. Choosing at-least-one over mean gives up 8.0 percentage points of the gain over the status quo, 17.0% of what is achievable (90% interval 15.2–19.1%), in 100% of replications rather than merely on average. Women's employment probability is 12.9 points lower under the deployed rule than under the mean (90% interval 11.6–14.2), and the within-household gap widens by 7.6 points relative to the legal default where the mean narrows it by 10.8. Ranking cantons by each individual's own prospects rather than the household's makes the same point at the level where it is felt: the deployed rule sends men to their 4.3rd-best canton of 26 and women to their 8.8th, and 13% of women to a canton in their own worst quartile against 2% of men. Under the mean the order reverses, to 6.7 for men and 4.5 for women. Unconstrained by capacity, at-least-one strictly prefers a canton with a lower within-household mean for 50% of households, a share rising from 47% to 56% as the canton-by-gender interaction strengthens.

Three checks. First, the ordering of the rules on every column is the same in the no-synergy limit, where the same table reads +1.4%, +2.2% and +1.4% and the forgone share is 37.8%: what the calibration changes is magnitude, not sign, and it changes it in both directions — the *share* of the achievable gain given up falls as synergies grow, while the absolute cost of the choice rises. Second, doubling the person-by-canton term strengthens the efficiency and gender findings (18.9 points of gain forgone, 26.0% of the achievable, women 23.0 points behind) and weakens the gap finding without reversing it: the widening falls from 7.6 to 2.6 points and holds in 96% of replications rather than 100%. Third, letting household members resemble each other, which is what the formula's independence assumption denies, leaves the efficiency loss where it was (8.0 to 7.9 points) and makes the distributional findings slightly worse: the gap widening rises from 7.8 to 8.1 points and women's shortfall from 12.9 to 13.2 as members' shared variance rises to 0.6.

### 4.3 The Household-Size Effect

The mechanism is not merely algebraic. I extend the simulation to a stylised mix of one- to four-adult households (shares 35/30/20/15) and rank, for each household, all 26 cantons by its own prospects — the outcome it experiences, rather than the compressed score it is matched on. Ranking needs a yardstick, and the yardstick must not be the objective of one of the rules being compared, so I report two. The first is the household's own mean employment probability; because the expected number of employed members is that mean times a household size which is fixed within a household, the two rank cantons identically, so this yardstick does not privilege the mean rule over the sum rule. The second is the employment probability of the household's least employable member, which no compared rule maximises. Tables report the average rank of the canton received, 1 being the best of 26.

Per-person yardstick:

| household size | at-least-one | mean | sum | lottery |
|---|---|---|---|---|
| 1 | 1.7 | 2.5 | 4.0 | 13.5 |
| 2 | 3.8 | 2.6 | 2.6 | 13.5 |
| 3 | 5.6 | 3.2 | 2.1 | 13.5 |
| 4 | **7.8** | 3.6 | 1.9 | 13.5 |
| all sizes | 4.0 | 2.8 | 2.9 | 13.5 |

Worst-off-member yardstick, with the share of households sent to a canton in their own worst quartile:

| household size | at-least-one | mean | sum | lottery | own-worst-quartile: at-least-one |
|---|---|---|---|---|---|
| 1 | 1.7 | 2.5 | 4.0 | 13.5 | 0% |
| 2 | 7.1 | 3.5 | 3.5 | 13.5 | 9% |
| 3 | 9.9 | 5.1 | 4.0 | 13.5 | 16% |
| 4 | **11.7** | 6.4 | 4.6 | 13.5 | **21%** |
| all sizes | 6.5 | 3.9 | 3.9 | 13.5 | 9% |

Take the negative result first. The lottery averages rank 13.5, the midpoint of 26, and sends 27% of households to their own worst quartile at every size. No rule leaves any size below it: the deployed objective's worst cell is a four-adult household at rank 11.7 on the yardstick least favourable to it, with a 21% chance of its weakest member landing in that member's own worst quartile against the lottery's 27%. Optimisation, including this optimisation, helps every size.

What it does not do is help them equally, and the gradient is steep. At-least-one takes a single adult to rank 1.7 and a four-adult household to 7.8 on the first yardstick and 11.7 on the second: the benefit of being optimised for falls by a factor of five, then seven, as a household adds members. The mean rule, which weights each person once, runs from 2.5 to 3.6 and from 2.5 to 6.4 — graded too, since a large household is harder to place well, but at a fifth of the slope on the first yardstick and two fifths on the second. That difference is not about how hard large households are to place; it is the score compression of Section 3, a four-adult household with middling probabilities scoring above 0.93 in every canton and so registering almost no preference for the optimiser to act on. Single adults gain from the same mechanism: for them the compressed score and the true score nearly coincide, which is why at-least-one places them better than the mean rule does.

In probability units rather than ranks, the same gradient reads as regret — the household's best available canton minus the one it received, an upper bound since not everyone can have their best at once. Under at-least-one, regret rises from 1.6 points for a single adult to 13.2 for a four-adult household on the first yardstick and to 27.0 on the second; under the mean it rises from 3.7 to 5.6 and to 14.8. The deployed rule buys single adults about two points of probability and charges four-adult households eight to twelve for them.

The sum rule confirms that no aggregation is neutral, though less tidily than I once put it. On the per-person yardstick it inverts the pattern, buying four-adult households rank 1.9 and pushing single adults to 4.0; on the worst-off yardstick it is close to flat across sizes, from 4.0 to 4.6. Each rule implies a different weighting of persons within a case, and the pooled row hides all of it: every rule beats the lottery on the population average, and at-least-one, mean and sum are within 1.2 ranks of each other there. Only the size-disaggregated rows show the objective choosing who benefits.

One further result, and a correction. In the no-synergy limit — no idiosyncratic person-by-canton match quality, every household ranking the cantons alike — four-adult households fall to rank 17.5 under at-least-one, below the lottery's 13.5, with 44% in their own worst quartile. That was the headline of an earlier version of this analysis. It does not survive calibration, for the reason given in Section 4.4: when all households want the same cantons, being deprioritised means receiving the cantons nobody wanted, which is a property of the degeneracy rather than of the objective. The grid in the replication code locates where the crossing goes. At three tenths of the calibrated synergy the four-adult household's excess over the lottery has already fallen from 4.1 ranks to −0.2 on the first yardstick and to 0.7 on the second; at the calibration it is −5.8 and −1.9, and at twice the calibration −6.7 and −2.3. The gradient reported above is what survives at every point on that grid, which makes this a vertical-equity finding and not an absolute one.

### 4.4 A Note on the Simulated Data

Three inputs are real — the gender gap, the capacity key, and the published employment gain the person-by-canton term is fitted to — and the rest is stylised: cantonal effects, the interaction, the size mix, and an additive logit in place of location-specific gradient-boosted trees, none of which I can estimate, since no individual-level ZEMIS data is public. The figures are therefore illustrative of the mechanism at plausible parameter values, not estimates of what the SEM pilot did.

One limit deserves recording rather than burying, because an earlier version of this analysis omitted the person-by-canton term altogether and drew a stronger conclusion from its absence. Without it every household ranks the cantons in nearly the same order, so a household the matching stage deprioritises does not merely lose its best option, it receives the options nobody wanted; large households then fell below the lottery, and I read that as a *primum non nocere* violation and as a conservative reading besides, on the thought that real synergies would give large households more to lose. The reasoning was wrong in sign. Restoring the term, the residual slots a deprioritised household receives are no longer systematically bad *for it*, and the crossing disappears (Section 4.3). What the omission inflated was the absolute claim; what survives it unchanged is the size gradient and everything in Section 4.2. That is the order of the argument now, and the grid over the term's magnitude is reported so the reader can see which claim lives where.

Three limits remain, all pushing in knowable directions. The idiosyncratic term is drawn independently across members of a household, whereas real household members share location-specific factors — language region, diaspora, an existing contact — so the true within-household correlation is positive; the check on correlated *individual* effects suggests this makes the objection stronger, not weaker, but it is not the same check. Gender is the only characteristic modelled, so the tool's other predictors, which Alajak et al. (2026) put at the centre, do not appear. And the lottery baseline is the statutory key alone, while the real status quo also honours constraints the simulation ignores, so it flatters neither the algorithm nor the comparison in any direction I can sign. None of this bears on Section 3, which is a property of the formula established without simulated data at all.

## 5. Against the Programme's Own Benchmark

Sections 3 and 4 measured the objective against its own stated goal and against the status quo rather than the field's standard criteria, none of which fits: individual fairness (Dwork et al. 2012) is unattainable for any deterministic assignment under hard capacity, since two identical individuals must receive different cantons once the last slot fills; equalised odds (Hardt et al. 2016) presupposes a binary decision; and statistical parity would forbid exactly the synergies the tool exists to exploit.

What does fit is the criterion Freund et al. (2023) formalise as the Random fairness rule: is each group's average outcome under optimisation at least as good as its expectation under random assignment? It needs no contested similarity metric, and its baseline is not one I have selected: Article 21 of the Asylum Ordinance 1 makes the characteristics-blind proportional key the legal default, and the Swiss pilot is a randomised trial evaluating the algorithm against that key. "Is any group left worse off than the legal default?" is the question the institution poses through its own study design, not an external standard imported to embarrass it.

Section 4 ran that test on the two partitions available to me, and the deployed objective passes both. Women do better under at-least-one than under the legal default, and so does every household size: at the calibrated specification even four-adult households receive cantons they themselves rank well above the lottery's midpoint. The violation appears only in the degenerate limit in which no household has idiosyncratic reasons to prefer one canton to another, and that limit also implies a gain for the deployed rule of 1.4% against the 41% it is documented to deliver, so it is not a description of the problem GeoMatch solves. The conclusion has to be stated the other way round from the one I expected: on the institution's own benchmark the objective does no group in my two partitions an absolute harm.

That is a real limit on how much the Random fairness rule can be asked to do. It is a floor, and a floor is silent about the distribution of whatever lies above it. Section 4's size gradient — a fivefold difference in received rank between one- and four-adult households, where the mean rule's is well under twofold — is entirely inside the region the criterion licenses, and so is the finding that the gain is spent on the member who needed it least. A criterion that asks only whether anyone was left worse off than under a lottery cannot distinguish an optimiser that shares its gains from one that concentrates them, which is why vertical equity (Black et al. 2022) has to be carried alongside it rather than reduced to it.

Neither objection points to an unavoidable cost of fairness. Switching to the mean answers both — narrowing the within-household gap rather than widening it, and flattening the size gradient to a fifth of its slope — while raising the employment gain from 39% to 47%, so the repair costs nothing. That is not a demonstration that the mean satisfies the criterion in general: I have tested two group partitions, not all of them. The remaining bill is bounded, which is what makes Freund et al.'s figure useful: in their study, binding group-fairness requirements cost at most 2% of total employment relative to the unconstrained algorithm, on U.S. and Dutch data with groups defined by origin, age or education. That is a dynamic setting rather than the static assignment simulated here, so it indicates an order of magnitude rather than transferring as a bound. Zero for what is documented here, a few per cent for group fairness in general: between those numbers no cost-based defence survives. And the repair belongs where Kleinberg et al. (2018) argue an equity preference belongs, in the decision rule rather than the estimator — GeoMatch's mapping stage.

None of this makes the mean correct — distribution-neutrality is a choice too, and Section 4 shows the sum is not neutral either, only biased the other way where each person counts once, and roughly size-neutral where the weakest member counts. If the goal is household self-sufficiency, the honest implementation is an explicit threshold on the number of earners; if total employment, the sum, as Annie MOORE uses; if employment per person, the mean; if protecting the least employable member, the minimum, at the cost Bansak et al.'s backtest already reports. Each is defensible; none is defensible silently. Rawls's (1958) requirement that an inequality be justifiable to those it disadvantages does not demand that this one be abolished, but it does demand that someone state it and argue for it.

## 6. Conclusion

GeoMatch is a genuine advance over a lottery, and the case for algorithmic assignment here is strong: at the specification calibrated to its own published gain, every group I can define does better under it than under the legal default. The argument of this paper is narrower and survives that concession intact. The deployed objective is not the one the tool is defended in terms of; it is dominated on the very metric it claims to maximise; it is biased against equality within households in a formally demonstrable way; and it distributes the gain from optimisation in inverse proportion to household size, for reasons internal to its algebra rather than to anyone's need. Its designers justify it by household self-sufficiency, a fiscal objective of the host state, and neither they nor their critics have treated that substitution as needing an argument.

The practical implication is small and specific: report the mapping function, justify it, and publish the distributional consequences of the alternatives, which the designers have already computed. Bennett and Keyes (2020) argue that fairness metrics can crowd out the prior question of what a system is for; here that question was not crowded out by metrics but disappeared into an implementation detail nobody read as normative. Bansak et al. did check the aggregate cost of their default and disclosed it candidly, in supplementary materials never foregrounded in the tool's public defence, and the mapping functions of both deployed tools have since been tabulated side by side (Strasser Ceballos and Kern 2025) without anyone asking why they differ. What nobody checked is who absorbs that cost inside the household, and how it falls across sizes. The most consequential normative decision in a system that has shaped the lives of asylum seekers across three countries since 2020 sits in a four-line helper function: disclosed, but never argued for.

---

## References

Acharya, A., Bansak, K., & Hainmueller, J. (2022). Combining outcome-based and preference-based matching: A constrained priority mechanism. *Political Analysis*, 30(1), 89–112.

Agarwal, B. (1997). "Bargaining" and gender relations: Within and beyond the household. *Feminist Economics*, 3(1), 1–51.

Ahani, N., Andersson, T., Martinello, A., Teytelboym, A., & Trapp, A. C. (2021). Placement optimization in refugee resettlement. *Operations Research*, 69(5), 1468–1486.

Alajak, K., Burnazoglu, M., Leurs, K., & van Schie, G. (2026). GeoMatch/MisMatch: A critical investigation of a refugee resettlement and labour market integration algorithm in the Netherlands. *Social Inclusion*, 14, Article 10923.

Asylverordnung 1 über Verfahrensfragen (AsylV 1) of 11 August 1999, SR 142.311, Art. 21, Annex 3.

Bansak, K., Ferwerda, J., Hainmueller, J., Dillon, A., Hangartner, D., Lawrence, D., & Weinstein, J. (2018). Improving refugee integration through data-driven algorithmic assignment. *Science*, 359(6373), 325–329, Supplementary Materials, and replication data (Harvard Dataverse, doi:10.7910/DVN/MS8XES).

Bennett, C. L., & Keyes, O. (2020). What is the point of fairness? *Interactions*, 27(3), 35–39.

Black, E., Elzayn, H., Chouldechova, A., Goldin, J., & Ho, D. (2022). Algorithmic fairness and vertical equity: Income fairness with IRS tax audit models. *FAccT '22*, 1479–1503.

Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017). Algorithmic decision making and the cost of fairness. *KDD '17*, 797–806.

Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. *ITCS '12*, 214–226.

Freund, D., Lykouris, T., Paulson, E., Sturt, B., & Weng, W. (2023). Group fairness in dynamic refugee assignment. arXiv:2301.10642.

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NIPS '16*.

Kleinberg, J., Ludwig, J., Mullainathan, S., & Rambachan, A. (2018). Algorithmic fairness. *AEA Papers and Proceedings*, 108, 22–27.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.

Passi, S., & Barocas, S. (2019). Problem formulation and fairness. *FAT\* '19*, 39–48.

Rawls, J. (1958). Justice as fairness. *The Philosophical Review*, 67(2), 164–194.

Sen, A. (1990). Gender and cooperative conflicts. In I. Tinker (Ed.), *Persistent Inequalities: Women and World Development* (pp. 123–149). Oxford University Press.

Staatssekretariat für Migration (2026). Erwerbssituation von vorläufig Aufgenommenen und Flüchtlingen: Einreisekohorte 2018.

Strasser Ceballos, C., & Kern, C. (2025). Location matching on shaky grounds: Re-evaluating algorithms for refugee allocation. *FAccT '25*.
