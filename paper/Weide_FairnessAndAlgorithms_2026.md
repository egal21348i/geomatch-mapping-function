# Whose Efficiency? The Undefended Normative Core of Algorithmic Refugee Assignment

Julian Weide  
Fairness and Algorithms (851-0746-00L), ETH Zurich, Spring 2026

*Replication code:* <https://github.com/egal21348i/geomatch-mapping-function>

---

## 1. Introduction

Debates about algorithmic fairness are usually organised around a trade-off: more accuracy or more equity, but not both, and the interesting question is where along the frontier to sit. Corbett-Davies et al. (2017) give the canonical formulation — fairness constraints are constraints, and constraints cost something — and critics of algorithmic refugee assignment have adopted the same grammar, reading the efficiency of these tools as something purchased at the price of equity (Alajak et al. 2026).

This paper argues that the grammar is wrong for this case, and that getting it wrong has concealed something more troubling than a hard trade-off. GeoMatch, developed by the Immigration Policy Lab at Stanford and ETH Zurich and piloted by the Swiss State Secretariat for Migration (SEM) since 2020, does not maximise employment. It maximises a household-level threshold: the probability that *at least one* member of a refugee family finds work. Two consequences follow from the shape of that formula rather than from the data it is fed. It prefers assignments that concentrate employment prospects in one household member over assignments that spread them evenly, regardless of which is better for the household; and it directs the benefits of optimisation away from large households, whose scores grow insensitive to location as they add members. The rule therefore sacrifices both the aggregate quantity it is praised for maximising and the distribution its critics accuse it of ignoring.

The claim is not speculative. Bansak et al.'s own supplementary materials report that replacing the at-least-one metric with the within-case mean raises the employment gain over the status quo from 41% to 47% in the U.S. backtest: the distribution-neutral alternative is more efficient. There is no frontier here to sit on, and the designers do not claim there is — they justify their choice by appeal to "the underlying goal of the refugee resettlement program to generate self-sufficient refugee families."

Once this is visible, the question in my title has an answer. The beneficiary of the deployed rule is not the refugee whose employment probability the tool nominally maximises, but the fiscal position of the host state, for which one earner per household suffices — a legitimate policy goal, but not the one the tool is publicly defended in terms of, and one that has never been argued for.

## 2. Where the Normative Decision Sits

GeoMatch has three stages (Bansak et al. 2018). In the *modeling* stage, gradient-boosted trees are fitted separately per location on historical registry data, yielding for each individual and location a predicted probability of the target outcome; in Switzerland the target is employment two, three and four years after arrival, estimated on the ZEMIS database. In the *mapping* stage, individual predictions are aggregated to the level at which assignment actually occurs — the case, typically a family — via a mapping function $\varphi$. In the *matching* stage, cases are assigned to locations so as to maximise the average case-level score subject to capacity constraints.

Almost all critical attention has gone to the first stage. Alajak et al. (2026) focus on the predictors, noting that nationality, gender, age, marital status and religion enter the Swiss feature vector directly, and that a model built on historically stratified labour-market outcomes will reproduce that stratification. Strasser Ceballos and Kern (2025) show how sensitive the pipeline is to the target variable: optimising German data for social rather than economic integration sends 88% of refugees elsewhere. Both critiques are well-founded, and both concern what goes *into* the model.

The mapping stage has been almost entirely overlooked, and it is where the decisive normative commitment is made. The modeling stage produces individual-level objects; the matching stage consumes case-level ones. $\varphi$ is the bridge, and its selection determines whose welfare the optimisation tracks. The authors' replication archive (Bansak et al. 2018, Harvard Dataverse doi:10.7910/DVN/MS8XES, file `func_M_to_Mstar.R`) makes the choice explicit in a four-line function named `compute_femp_prob`:

```r
compute_femp_prob <- function(x){
  return(1-(prod(1-x)))
}
```

This is $\varphi(p) = 1 - \prod_i (1 - p_i)$, the probability that at least one member of the case is employed under within-case independence. It is the default reported for both the United States and Switzerland backtests, not an option tried once for the paper. The authors tested the mean, maximum and minimum as alternatives and give their reason for the default: it "best reflects the underlying goal of the refugee resettlement program to generate self-sufficient refugee families."

That is a candid statement that the objective encodes a policy goal rather than a welfare measure: household self-sufficiency is satisfied once a family stops drawing social assistance, and not satisfied any further by a second earner. The choice is not even stable across the designers' own subsequent work: Acharya et al. (2022), extending GeoMatch to incorporate refugees' preferences and co-authored by two of the original team, use the mean instead without remarking on the switch. That inconsistency is not evidence that the choice is immaterial — Sections 3 and 4 show it is anything but. It is evidence that $\varphi$ has never been treated, even by its designers, as a commitment needing defence in either direction.

One limitation should be stated plainly: whether the SEM's live pipeline still uses this exact function is not published, and neither are its evaluation results (Strasser Ceballos and Kern 2025). What is secure is that this function produced the founding study's results and was justified by the programme's stated goals rather than offered as incidental convenience. If a different $\varphi$ is used today, that change and its rationale are equally undocumented — an instance of the problem described here, not an exception to it.

## 3. Inequality Within Households, Disadvantage Between Them

Given two locations offering a household the same *average* employment prospect — one spreading it evenly across members, the other concentrating it in a single member — Bansak et al.'s formula, one minus the chance that everybody fails, always ranks the second higher. Once one member already has a good chance of work, that alone nearly guarantees the case a positive outcome, so improving anyone else's prospects adds almost nothing to the score. This holds for households of any size and any probabilities, and it means an assignment maximising the score can be dominated in expected employment by one that does not:

| | adult 1 | adult 2 | at-least-one score | mean | expected employed adults |
|---|---|---|---|---|---|
| Canton A | 0.80 | 0.10 | **0.820** | 0.450 | 0.90 |
| Canton B | 0.50 | 0.50 | 0.750 | 0.500 | 1.00 |

GeoMatch sends this household to A — the canton that is worse both in expected employment and in equality — not through error, but through faithful maximisation of its stated objective.

Who bears this? In Swiss refugee households the secondary earner is overwhelmingly a woman: seven years after arrival, 64% of men but only 32% of women aged 16–55 at entry are in employment (Staatssekretariat für Migration 2026). That gap is the raw material the formula exploits, since the larger the within-household spread a location can generate, the higher its score. Alajak et al. (2026) allege "disproportionate discriminatory outcomes on the basis of ethnicity, gender, age, and marital status," but their document-based method cannot identify a mechanism. The argument above supplies one, located not in the training data but in a design choice downstream of it, and it reaches every group the allegation names: whoever is least likely to find work is whom the objective has least reason to consider, so a group that starts behind is served worse the further behind it starts.

A second consequence of the same formula concerns household size. As a household grows, its score becomes less sensitive to any one member's location-specific prospects, because the "at least one" condition is nearly satisfied by the others alone. A single-adult household's score ranges over the full spread of employment probabilities across the twenty-six cantons; a four-adult household with middling probabilities scores above 0.93 almost everywhere, so almost nothing is at stake in where it is sent. This matters because the matching stage allocates scarce capacity by marginal gain: a slot in a well-matched canton goes to whichever case's score rises most from receiving it, and households whose scores barely move are the natural candidates for displacement to a worse one. The objective therefore steers the benefits of optimisation toward small cases for a reason unrelated to need, integration prospects, or any defensible distributive principle. In the vocabulary of Black et al. (2022), this is a failure of vertical equity.

The two distortions sit at different levels, and the distinction matters for what to call them. The gender effect is a *within*-household harm: the metric is satisfied by sacrificing one member's prospects for another's, and the member sacrificed is, given the current employment gap, disproportionately a woman — a facially neutral rule with a foreseeable adverse effect on a protected characteristic, the ordinary structure of indirect discrimination. The household-size effect is a *between*-household harm: nobody inside the case is sacrificed for anybody else, but the case as a whole is sent somewhere worse than an otherwise identical smaller household would be. That is why vertical equity, not discrimination, is the right frame for it: it needs no protected characteristic, only that likes be treated alike.

## 4. Magnitude

To gauge how much this matters, I run the pipeline twice: once calibrated to the observed gender gap, to quantify the within-household mechanism; once extended to households of variable size, to quantify the between-household one.

### Efficiency and the Gender Gap

Individual probabilities are generated on the logit scale with a gender base rate calibrated to the SEM figures above (men 0.64, women 0.32), an individual random effect, a cantonal main effect, and a canton-by-gender interaction representing the synergies the tool exists to exploit. Capacities are the statutory population-proportional shares of all twenty-six cantons from Annex 3 of the Asylum Ordinance 1, and the matching stage replicates the published code: each canton is duplicated by its number of slots, cost is set to $1 - \varphi$, and a one-to-one optimal assignment is solved. I compare three mapping functions against the characteristics-blind key of Article 21 of that ordinance, averaging over 400 draws of the cantonal parameters with 1,000 households each. That key is not a historical baseline: the Swiss deployment is a randomised trial in which only some arrivals receive an algorithmic recommendation and the rest are still placed proportionally, so "status quo" below names the live control arm of the same programme.

| $\varphi$ | expected employed adults per household | gain over status quo | P(employed), women | within-household gap |
|---|---|---|---|---|
| status quo | 0.964 | — | 0.335 | 0.309 |
| at-least-one | 0.977 | +1.3% | 0.342 | **0.336** |
| mean | 0.985 | **+2.2%** | 0.346 | 0.302 |
| minimum | 0.978 | +1.4% | 0.347 | 0.284 |

At-least-one is the worst of the three candidate rules on every column: lower expected employment and a smaller gain over the status quo than either alternative, a lower employment probability for women, and the only row of the four — including the status quo — that widens rather than narrows the within-household gap. Mean wins on efficiency, minimum on equity, at-least-one on neither, so no efficiency-equity frontier remains on which to locate it. The gap-widening is moreover a violation of what Freund et al. (2023) call the Random fairness rule, the *primum non nocere* requirement that no group be worse off than under the status quo, on the very dimension the objective was chosen to address.

The full Monte Carlo adds magnitude and robustness. Choosing at-least-one over mean forgoes 39.8% of the achievable efficiency gain (90% interval 31.2–47.7%), and the gap-widening holds in 100% of replications rather than merely on average. Bansak et al.'s own backtest comparison runs in the same direction, though my synergy structure is deliberately thinner than a full model's and the proportion should not be read as an estimate of theirs. A third result concerns the assignment rather than outcome levels: at-least-one and mean send 85.5% of households to different cantons, a reorganisation on the scale of the 88% Strasser Ceballos and Kern (2025) obtain by changing the target variable. Across interaction strengths, reversals range from 14.8% of households to 57.6% while the forgone-gain share stays near 40%.

### The Household-Size Effect

The household-size mechanism is not merely algebraic; it is large. I extend the simulation to a stylised mix of one- to four-adult households (shares 35/30/20/15) and rank, for each household, all 26 cantons by its own true mean employment probability — the outcome it experiences, rather than the compressed score it is matched on. The table reports, for each size under each rule, the average rank of the canton actually received (1 = best of 26) and the "own-worst-quartile" share, the percentage of households sent to one of the seven worst cantons *on their own ranking*.

| household size | rank: at-least-one | rank: mean | rank: lottery | own-worst-quartile: at-least-one | own-worst-quartile: lottery |
|---|---|---|---|---|---|
| 1 | 6.8 | 7.9 | 13.45 | 3% | 26.7% |
| 2 | 7.8 | 10.3 | 13.46 | 4% | 26.7% |
| 3 | 12.9 | 10.9 | 13.44 | 18% | 26.7% |
| 4 | 17.6 | 11.4 | 13.45 | **45%** | 26.6% |
| all sizes | **9.9** | 9.8 | 13.45 | **13%** | 26.7% |

The lottery column is the yardstick: a household-blind lottery averages rank 13.5, the midpoint of 26. Against it, single adults and couples do far better, at average ranks of 6.8 and 7.8, because for a small case the compressed score and the true score nearly coincide. Four-adult households do worse than the lottery that ignores them entirely: rank 17.6, and a 45% chance of landing in their own worst quartile against the lottery's 26.7%. The mean rule shows no such reversal (rank 11.4), so the failure is specific to the deployed $\varphi$ rather than an inherent cost of optimising.

Two checks confirm this is not an artefact of the size mix or of measuring rank rather than magnitude. The pooled row shows what a population-wide average alone would suggest — an unambiguous improvement over the lottery — so only disaggregation reveals the transfer from large households to small ones, the size-level analogue of what aggregating individuals into households does to the gender effect. Regret, each household's own best-canton probability minus what it receives, measures every size against its own achievable range and tells the same story in probability points: under at-least-one it rises from 0.09 for singles to 0.17 for four-adult households, crossing the lottery's own regret — which falls from 0.17 to 0.14 — between three and four adults.

### A Note on the Simulated Data

These figures are illustrative, not estimates of the pilot's effects. Only two inputs are real: the gender employment gap and the cantonal capacity key. The cantonal main effect, the canton-by-gender interaction, the household-size mix and the functional form — an additive logit rather than the location-specific gradient-boosted trees GeoMatch fits — all stand in for a model I cannot estimate, since no individual-level ZEMIS data is public; the Monte Carlo over parameter draws and the sensitivity analysis exist so that no conclusion rests on one such choice. One structural limit matters: individual heterogeneity enters only as a shift constant across cantons, so everyone of a given gender ranks the cantons identically, and the simulation can establish that a class of households is treated differently but not that one woman is disadvantaged relative to another. None of this bears on Section 3, which is a property of the formula established without simulated data at all; the simulation shows only that its consequences are large at plausible parameter values rather than negligible.

## 5. Against the Programme's Own Benchmark

Sections 3 and 4 measured the objective against its own stated goal and against the status quo, not against the field's standard fairness criteria. Checked directly, none of the three fits, and the reasons are instructive rather than evasive. Individual fairness (Dwork et al. 2012) requires similar individuals to receive similar outcomes; under deterministic assignment with hard capacity, two identical individuals must receive different cantons once the last slot fills — a structural fact about scarce indivisible goods, not a GeoMatch defect, and recoverable only by randomising the assignment, which GeoMatch by design does not do. Statistical parity fails differently: cantons are not ranked treatments, since a canton good for one profile can be poor for another — the entire premise of exploiting synergies — so equalising assignment rates is not obviously desirable, and Dwork et al.'s own objection, that parity permits compensating unfairness across subgroups, applies in full. Equalised odds (Hardt et al. 2016) presupposes a binary decision that a twenty-six-way allocation does not have.

What does fit is the comparative criterion Freund et al. (2023) formalise: is each group's average outcome under optimisation at least as good as under the status quo? What recommends it here is not that it is easy to satisfy, but that accepting it requires no philosophical commitment. It needs no contested similarity metric, and its baseline is not one I have selected: Article 21 of the Asylum Ordinance 1 makes the characteristics-blind proportional key the legal default, and the Swiss pilot is a randomised trial evaluating the algorithm against that very key. "Is any group left worse off than under the legal default?" is therefore the question the institution poses through its own study design, not an external standard imported to embarrass it. It is also, precisely, the question Section 4 showed the deployed objective failing twice — on the within-household gap and on household size — against the comparator its own trial was built around.

Neither failure is an unavoidable cost of fairness. Switching to mean removes both — it narrows the within-household gap rather than widening it, and it does not push large households below the lottery baseline — while raising expected employment, so the removal costs nothing. That is not a demonstration that mean satisfies the criterion in general: satisfying it requires that no group be worse off, and I have tested two group partitions, not all of them. The remaining bill is bounded, though, which is what makes Freund et al.'s figure useful here: a fairness constraint binding across all groups costs at most 2% of total employment on an assignment problem of this kind. Zero for the violations documented here, at most 2% for full group fairness — between those two numbers, no cost-based defence of the deployed rule survives. And the repair would belong exactly where Kleinberg et al. (2018) argue an equity preference belongs, in the decision rule rather than the estimator: GeoMatch's mapping stage, which is also the one stage its designers never discuss.

None of this makes the mean correct — distribution-neutrality is a choice too. If the goal is household self-sufficiency, the honest implementation is an explicit threshold on the number of earners; if individual employment, use the mean; if protecting the least employable member, use the minimum and accept the roughly quarter of the achievable gain Bansak et al.'s backtest shows it costs. Each is defensible; none is defensible silently. Rawls's (1958) requirement that an inequality be justifiable to those it disadvantages does not demand that this inequality be abolished, but it does demand that someone state it and argue for it. The objection is not to the choice made but to its being made invisibly, inside a helper function, under a description that does not match it.

## 6. Conclusion

GeoMatch is a genuine advance over a lottery, and the case for algorithmic assignment in this domain is strong. The argument here is narrower and, I think, harder to dismiss for being narrow: the objective function deployed is not the one the tool is defended in terms of, it is dominated on the very metric it claims to maximise, it is biased against equality within households in a formally demonstrable way, it deprioritises large households for reasons internal to its algebra, and on both counts it leaves those it disadvantages worse off than the characteristics-blind key it is measured against. Its designers justify it by reference to household self-sufficiency, a fiscal objective belonging to the host state, and neither they nor their critics have treated that substitution as requiring an argument.

The practical implication is small and specific: report the mapping function, justify it, and publish the distributional consequences of the alternatives, which the designers have already computed. The theoretical implication is larger. Bennett and Keyes (2020) argue that fairness metrics can crowd out the prior question of what a system is for; here that question was not crowded out by metrics but disappeared into an implementation detail nobody read as normative. Bansak et al. did check the aggregate cost of their own default and disclosed it candidly, in a comparison buried in supplementary materials that was never foregrounded in the tool's public defence and never picked up by critics looking at the training data rather than the aggregation stage. What nobody checked is who absorbs that cost inside the household, or how it falls across household sizes. The most consequential normative decision in a system that has shaped the lives of asylum seekers across three countries since 2020 sits in a four-line helper function: disclosed, but never argued for, and never connected to the fairness debate it went on to trigger.

---

## References

Acharya, A., Bansak, K., & Hainmueller, J. (2022). Combining outcome-based and preference-based matching: A constrained priority mechanism. *Political Analysis*, 30(1), 89–112.

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

Rawls, J. (1958). Justice as fairness. *The Philosophical Review*, 67(2), 164–194.

Staatssekretariat für Migration (2026). Erwerbssituation von vorläufig Aufgenommenen und Flüchtlingen: Einreisekohorte 2018.

Strasser Ceballos, C., & Kern, C. (2025). Location matching on shaky grounds: Re-evaluating algorithms for refugee allocation. *FAccT '25*.
