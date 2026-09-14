# Whose Efficiency? The Undefended Normative Core of Algorithmic Refugee Assignment

Julian Weide  
Fairness and Algorithms (851-0746-00L), ETH Zurich, Spring 2026

*Replication code:* <https://github.com/egal21348i/geomatch-mapping-function>

---

## 1. Introduction

Debates about algorithmic fairness are usually organised around a trade-off: accuracy or equity, but not both, and the question is where along the frontier to sit. Corbett-Davies et al. (2017) give the canonical formulation, and critics of algorithmic refugee assignment have adopted the same grammar, reading these tools' efficiency as purchased at the price of equity (Alajak et al. 2026).

The grammar is wrong for this case, and getting it wrong has concealed something worse than a hard trade-off. GeoMatch, developed at Stanford and ETH Zurich's Immigration Policy Lab and piloted by the Swiss State Secretariat for Migration (SEM) since 2020, does not maximise employment. It maximises a household-level threshold: the probability that *at least one* member of a refugee family finds work. Two consequences follow from the shape of that formula, not from the data it is fed: it prefers concentrating employment prospects in one household member to spreading them evenly, and it directs the benefits of optimisation away from large households, whose scores grow insensitive to location as they add members. The rule sacrifices both the aggregate quantity it is praised for maximising and the distribution its critics accuse it of ignoring.

The claim is not speculative. Bansak et al.'s own supplementary materials (fig. S8) report that replacing the at-least-one metric with the within-case mean raises the employment gain over the status quo from 41% to 47% in the U.S. backtest: the distribution-neutral alternative is more efficient. There is no frontier to sit on, and the designers do not claim there is; they justify the choice, as Section 2 shows, by the programme's goal of household self-sufficiency. That answers the question in my title, if only by inference: the beneficiary is not the refugee whose employment probability the tool nominally maximises but the fiscal position of the host state, for which one earner suffices — a legitimate goal, but not the one the tool's public case is built on.

This is not a claim that the tool harms those it assigns, which Section 5 shows it does not, nor that the mean is the right objective; only that four candidates were available, that their consequences differ widely, and that the choice among them was never argued.

## 2. Where the Normative Decision Sits

GeoMatch has three stages (Bansak et al. 2018). The *modeling* stage fits gradient-boosted trees separately per location, predicting for each individual and location the probability of the target outcome — in Switzerland, employment two to four years after arrival in the ZEMIS registry. The *mapping* stage aggregates those predictions to the level at which assignment actually occurs, the case or family, via a function $\varphi$; the *matching* stage then maximises the average case-level score subject to capacity constraints.

Critical attention has gone to the first stage and formal fairness work to the last. Alajak et al. (2026) focus on the predictors: nationality, gender, age, marital status and religion enter the Swiss feature vector directly, and a model built on historically stratified outcomes reproduces that stratification. Strasser Ceballos and Kern (2025) show how sensitive the pipeline is to the target variable — optimising German data for social rather than economic integration sends 88% of refugees elsewhere. Both critiques concern what goes *into* the model.

That values enter through the objective rather than the estimator is the lesson of the problem-formulation literature (Passi and Barocas 2019), demonstrated canonically by Obermeyer et al. (2019), where a health-risk model tracking costs instead of illness produced a racial disparity from a label chosen upstream of any modelling. What is substituted here is not the label but the unit whose welfare is counted, and the substitution happens in the middle stage: the modeling stage produces individual-level objects, the matching stage consumes case-level ones, and $\varphi$ is the bridge whose selection determines whose welfare the optimisation tracks. That is the one stage nobody argues about, and it is not hidden. Strasser Ceballos and Kern (2025, Appendix C) tabulate it for both deployed tools of this family, and the comparison is instructive, because the other chose differently: Annie MOORE, deployed at a U.S. resettlement agency, aggregates by the *sum* of individual probabilities and so maximises the expected number of employed refugees (Ahani et al. 2021). Two systems on the same three stages optimise different things, and neither paper remarks on it. GeoMatch's choice is explicit in a four-line function in the authors' replication archive (Bansak et al. 2018, Harvard Dataverse doi:10.7910/DVN/MS8XES, `func_M_to_Mstar.R`):

```r
compute_femp_prob <- function(x){
  return(1-(prod(1-x)))
}
```

This is $\varphi(p) = 1 - \prod_i (1 - p_i)$, the probability that at least one member is employed under within-case independence. It is the default for both the U.S. and Swiss backtests, not an option tried once: the authors tested the mean, maximum and minimum, and chose this one because it "best reflects the underlying goal of the refugee resettlement program to generate self-sufficient refugee families."

That is a candid statement that the objective encodes a policy goal rather than a welfare measure: household self-sufficiency is satisfied once a family stops drawing social assistance, and not further by a second earner. Nor is the choice stable across the designers' own later work — Acharya et al. (2022), extending GeoMatch to refugees' preferences and co-authored by two of the original team, use the mean without remarking on the switch. Not because the choice is immaterial, as Sections 3 and 4 show, but because $\varphi$ has never been treated, even by its designers, as a commitment needing defence. Whether the SEM's live pipeline still uses it is unpublished, as are the pilot's results (Strasser Ceballos and Kern 2025); a different $\varphi$ today would be an equally undocumented change.

## 3. Inequality Within Households, Disadvantage Between Them

Two properties of $\varphi(p) = 1 - \prod_i (1 - p_i)$ do the work below, each following in a line. First, $\log(1-p)$ is concave, so $\sum_i \log(1 - p_i)$ is largest when the $p_i$ are equal; and since $\varphi = 1 - \exp\left(\sum_i \log(1-p_i)\right)$ is *decreasing* in that sum, among locations offering the same *average* prospect $\varphi$ ranks the evenly spread one lowest and the most concentrated one highest. Second, $\partial\varphi/\partial p_i = \prod_{j \neq i}(1 - p_j)$: the weight the objective places on any member falls as the other members' prospects rise, so the rule attends least to whoever is least likely to find work, for any size and any probabilities. An assignment maximising it can therefore be dominated in expected employment by one that does not:

| | adult 1 | adult 2 | at-least-one score | mean | expected employed adults |
|---|---|---|---|---|---|
| Canton A | 0.80 | 0.10 | **0.820** | 0.450 | 0.90 |
| Canton B | 0.50 | 0.50 | 0.750 | 0.500 | 1.00 |

GeoMatch sends this household to A, worse both in expected employment and in equality, not through error but through faithful maximisation of its stated objective. This is the unitary household model implemented as an optimiser, and the reason development economics abandoned that model applies here: aggregating to the household conceals who inside it gains, and not neutrally, where bargaining power and labour-market access differ by gender (Sen 1990; Agarwal 1997). What is new is the site: here the aggregation is not an approximation forced by data but a line of code.

Who bears this? In Swiss refugee households the secondary earner is overwhelmingly a woman: seven years after arrival, 64% of men but only 32% of women aged 16–55 at entry are in employment (Staatssekretariat für Migration 2026). That is a population rate at a longer horizon than the Swiss target variable, so it overstates the levels the tool works with but not the direction of the gap. Alajak et al. (2026) allege discriminatory outcomes by ethnicity, gender, age and marital status but cannot identify a mechanism. The argument above supplies one, located not in the training data but in a design choice downstream of it, and it reaches every group they name: whoever starts behind is served worse the further behind they start.

The second property compounds across members, which produces a distinct effect on size. Each additional member multiplies the derivative by a factor below one, so a large household's score barely moves with location: a four-adult household with middling probabilities scores above 0.93 in every canton, while a single adult's score ranges over the full cantonal spread. Since the matching stage allocates capacity by marginal gain, households whose scores barely move are the natural candidates for displacement. The objective steers the benefits of optimisation toward small cases for a reason unrelated to need or prospects: in the vocabulary of Black et al. (2022), a failure of vertical equity.

The two harms need different names. Sacrificing one member's prospects for another's, where the member sacrificed is disproportionately a woman, has the structure of indirect discrimination, though not its legal test: that requires disadvantage relative to a comparator, and relative to the legal default women gain, only less than under a rule that counted them once. The objection is therefore pressed below as one about efficiency and the distribution of the gain. Displacing a household for its size is different, since nobody inside the case is sacrificed for anyone else — there vertical equity, needing only that likes be treated alike, is the frame.

## 4. Magnitude

### 4.1 Design and Calibration

Probabilities are generated on the logit scale from the SEM gender base rates (men 0.64, women 0.32) plus an individual effect, a cantonal effect, a canton-by-gender interaction, and a person-by-canton term. Capacities are the statutory population-proportional shares of all twenty-six cantons (Annex 3 AsylV 1), and the matching stage replicates the published code: each canton is duplicated by its slots and a one-to-one optimal assignment on cost $1 - \varphi$ is solved. The comparison is against the characteristics-blind key of Article 21, over draws of the cantonal parameters.

The last term carries the analysis and so should not be chosen freely: it is the extent to which a canton that suits one person does not suit another, which is the entire resource an assignment algorithm has. Without it, everyone of a given gender ranks the cantons identically and the deployed rule gains 1.4% over the status quo where the best candidate manages 2.3% — two orders of magnitude below what GeoMatch is documented to deliver. So the term is pinned to the published effect size rather than assumed: at a standard deviation of 1.0 on the logit scale, at-least-one gains 39.4% and its mean variant 47.3%, against the 41% and 47% Bansak et al. report for the U.S. backtest and its fig. S8 comparison. One parameter reproduces both. Gender intercepts are re-solved for every specification so that status-quo employment stays at the SEM base rates; specifications therefore differ in synergy alone.

Three inputs are real, then — the gender gap, the capacity key and the published gain the last term is fitted to — and the rest is stylised: cantonal effects, the interaction, the size mix, and an additive logit in place of location-specific gradient-boosted trees, since no individual-level ZEMIS data is public. Gender is also the only characteristic modelled, so the tool's other predictors, which Alajak et al. (2026) put at the centre, do not appear. The figures illustrate the mechanism at plausible parameter values; they are not estimates of what the SEM pilot did, and none bears on Section 3, a property of the formula established without simulated data at all.

### 4.2 Efficiency and the Gender Gap

Over 400 draws with 1,000 two-adult households each; at equal household size the sum and the mean coincide, leaving three distinct rules.

| $\varphi$ | employed adults per household | gain over status quo | P(employed), women | within-household gap |
|---|---|---|---|---|
| status quo | 0.962 | — | 0.325 | 0.373 |
| at-least-one | 1.338 | +39.1% | 0.471 | **0.450** |
| mean | 1.415 | **+47.1%** | 0.600 | 0.266 |
| minimum | 1.371 | +42.5% | **0.629** | **0.160** |

At-least-one is the worst of the three on every column: a smaller gain than either alternative, a lower employment probability for women, and the only row that widens rather than narrows the within-household gap. Mean wins on efficiency, minimum on equity, at-least-one on neither, so no efficiency-equity frontier remains on which to locate it. Note what the table does not show: women are not left absolutely worse off than under the status quo, at 0.471 against 0.325 — the tool helps them, and substantially. The objection is narrower: the rule forgoes employment, and spends the gain it does achieve on the member who already had the better prospects.

The magnitudes are not small. Choosing at-least-one over mean gives up 8.0 percentage points of the gain over the status quo, 17.0% of what is achievable (90% interval 15.2–19.1% across parameter draws), in 100% of replications rather than merely on average. Women's employment probability is 12.9 points lower than under the mean, and the within-household gap widens by 7.6 points relative to the legal default where the mean narrows it by 10.8. Ranked by each individual's own prospects, the deployed rule sends men to their 4.3rd-best canton of 26 and women to their 8.8th, with 13% of women against 2% of men in a canton in their own worst quartile.

Two checks. The ordering of the rules is unchanged in the no-synergy limit, where the table reads +1.4%, +2.2% and +1.4%, so calibration changes magnitude, not sign; and doubling the person-by-canton term strengthens the efficiency and gender findings (18.9 points of gain forgone, women 23.0 points behind) while weakening the gap finding to 2.6 points without reversing it.

### 4.3 The Household-Size Effect

The mechanism is not merely algebraic. I extend the simulation to a stylised mix of one- to four-adult households (shares 35/30/20/15) and rank, for each household, all 26 cantons by its own prospects rather than by the compressed score it is matched on. Since the yardstick must not be the objective of a rule being compared, I use two: the household's mean employment probability, which ranks cantons identically to the expected number employed and so privileges neither the mean nor the sum, and the probability for its least employable member, which no rule maximises. The table gives the average rank of the canton received, 1 being best of 26 and a household-blind lottery 13.5; the first three rule columns use the per-person yardstick, the last three the worst-member one.

| household size | at-least-one | mean | sum | at-least-one | mean | sum |
|---|---|---|---|---|---|---|
| 1 | 1.7 | 2.5 | 4.0 | 1.7 | 2.5 | 4.0 |
| 2 | 3.8 | 2.6 | 2.6 | 7.1 | 3.5 | 3.5 |
| 3 | 5.6 | 3.2 | 2.1 | 9.9 | 5.1 | 4.0 |
| 4 | **7.8** | 3.6 | 1.9 | **11.7** | 6.4 | 4.6 |
| all sizes | 4.0 | 2.8 | 2.9 | 6.5 | 3.9 | 3.9 |

Take the negative result first. No rule leaves any size below the lottery: the deployed objective's worst cell is a four-adult household at rank 11.7 on the yardstick least favourable to it, with a 21% chance of its weakest member landing in that member's own worst quartile against the lottery's 27%. Optimisation, including this one, helps every size. What it does not do is help them equally, and the gradient is steep. At-least-one takes a single adult to rank 1.7 and a four-adult household to 7.8 on the first yardstick and 11.7 on the second: the benefit of being optimised for falls by a factor of five, then seven, as a household adds members. The mean rule runs from 2.5 to 3.6 and from 2.5 to 6.4 — graded too, since a large household is harder to place well, but at a fifth of that slope and two fifths respectively. That difference is the score compression of Section 3 in action; single adults gain from the same mechanism, which is why at-least-one places them better than the mean does. In probability units the gap between the best canton available to a four-adult household and the one it receives is 13.2 points under at-least-one against 5.6 under the mean.

The sum rule confirms that no aggregation is neutral: on the per-person yardstick it inverts the pattern, buying four-adult households rank 1.9 and pushing single adults to 4.0, and on the worst-member yardstick it is nearly flat. Each rule implies a different weighting of persons within a case, and the pooled row hides all of it — every rule beats the lottery there, within 1.2 ranks of the others. Only the size-disaggregated rows show the objective choosing who benefits.

One result here does depend on the calibration: in the no-synergy limit four-adult households fall to rank 17.5, below the lottery, because when every household ranks the cantons alike a deprioritised one receives the options nobody wanted. That crossing disappears at three tenths of the calibrated synergy, so it belongs to the degeneracy rather than to the objective; the gradient itself holds throughout.

## 5. Against the Programme's Own Benchmark

Sections 3 and 4 measured the objective against its own stated goal rather than the field's standard criteria, none of which fits: individual fairness (Dwork et al. 2012) is unattainable under hard capacity, since two identical individuals must receive different cantons once the last slot fills; equalised odds (Hardt et al. 2016) presupposes a binary decision; statistical parity would forbid the synergies the tool exists to exploit.

What does fit is the criterion Freund et al. (2023) formalise as the Random fairness rule: is each group's average outcome under optimisation at least as good as its expectation under random assignment? Its baseline is not one I have selected: Article 21 of the Asylum Ordinance 1 makes the characteristics-blind proportional key the legal default, and the Swiss pilot is a randomised trial against that key, so the question is the institution's own.

Section 4 ran that test on the two partitions available to me, and the deployed objective passes both: women do better under at-least-one than under the legal default, and so does every household size. But that is a floor, and a floor is silent about the distribution above it. The size gradient — fivefold between one- and four-adult households, against well under twofold for the mean rule — lies entirely inside the region the criterion licenses, as does the finding that the gain is spent on the member who needed it least. A criterion asking only whether anyone was left worse off than a lottery cannot distinguish an optimiser that shares its gains from one that concentrates them, which is why vertical equity (Black et al. 2022) has to be carried alongside it rather than reduced to it.

Neither objection points to an unavoidable cost of fairness. Switching to the mean answers both — narrowing the within-household gap rather than widening it, and flattening the size gradient to a fifth of its slope — while raising the employment gain from 39% to 47%, so the repair costs nothing. Whether the mean satisfies the criterion on partitions I have not tested is open, but the remaining bill is bounded: in Freund et al.'s study, binding group-fairness requirements cost at most 2% of total employment — in a dynamic setting rather than the static assignment simulated here, so an order of magnitude rather than a transferable bound, but between zero and a few per cent no cost-based defence survives. And the repair belongs where Kleinberg et al. (2018) argue an equity preference belongs, in the decision rule rather than the estimator.

None of this makes the mean correct; distribution-neutrality is a choice too, and the sum is not neutral either, only biased the other way. If the goal is household self-sufficiency, the honest implementation is an explicit threshold on the number of earners; if total employment, the sum, as Annie MOORE uses; if employment per person, the mean; if protecting the least employable member, the minimum. Each is defensible; none is defensible silently. Rawls's (1958) requirement that an inequality be justifiable to those it disadvantages does not demand that this one be abolished, but that someone state it and argue for it.

## 6. Conclusion

GeoMatch is a genuine advance over a lottery: every group I can define does better under it than under the legal default. The argument of this paper is narrower and survives that concession. The deployed objective is dominated on the very metric it claims to maximise; it is biased against equality within households in a formally demonstrable way; and it distributes the gain from optimisation in inverse proportion to household size, for reasons internal to its algebra rather than to anyone's need. The practical implication is small and specific: report the mapping function, justify it, and publish the distributional consequences of the alternatives, which the designers have already computed. Bennett and Keyes (2020) argue that fairness metrics can crowd out the prior question of what a system is for; here that question was not crowded out by metrics but disappeared into an implementation detail nobody read as normative. Its designers did check the aggregate cost of their default and disclose it candidly; what nobody checked is who absorbs that cost inside the household, and how it falls across sizes. The most consequential normative decision in a system that has shaped the lives of asylum seekers across three countries since 2020 sits in a four-line helper function: disclosed, but never argued for.

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
