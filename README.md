# Whose Efficiency? The Undefended Normative Core of Algorithmic Refugee Assignment

Term paper and replication code for *Algorithms and Fairness* (851-0746-00L), ETH Zurich, Spring 2026.

Julian Weide

## The argument

GeoMatch, the refugee assignment algorithm piloted by the Swiss State Secretariat for Migration, does not maximise employment. Its mapping stage aggregates individual predicted employment probabilities to the household level using

```
phi(p) = 1 - prod(1 - p_i)
```

the probability that *at least one* household member finds work. This choice is made in a four-line helper function, `compute_femp_prob`, in the authors' replication archive (Bansak et al. 2018, Harvard Dataverse [doi:10.7910/DVN/MS8XES](https://doi.org/10.7910/DVN/MS8XES), `Functions/func_M_to_Mstar.R`).

Two properties of that formula do the work, and each follows in a line: $\log(1-p)$ is concave, so among locations with the same average prospect the score is highest where prospects are most concentrated in one member; and $\partial\varphi/\partial p_i = \prod_{j\neq i}(1-p_j)$, so the weight on any member falls as the others' prospects rise. Hence:

1. **Within households**, the objective attends least to whoever is already least likely to find work — in Swiss refugee households, disproportionately the woman.
2. **Between households**, large households' scores become insensitive to location, so the matching stage serves them last and the gain from optimisation falls as a household adds members.

Both are quantified by simulation, calibrated so that the deployed rule reproduces the employment gain Bansak et al. report for the U.S. backtest (41%) and the mean variant reproduces the 47% of their fig. S8. At that calibration, choosing at-least-one over the mean gives up 17% of the achievable employment gain and 12.9 percentage points of women's employment probability, and it widens the within-household gap in 100% of replications.

Neither effect leaves any group absolutely worse off than the characteristics-blind proportional key of Art. 21 AsylV 1 — the legal default and the control arm of the Swiss trial. An earlier version of this analysis reported that the largest households did fall below that key; that result came from omitting idiosyncratic person-by-canton match quality, which makes all households rank the cantons alike, and it does not survive the calibration. `size_effect_idio` reports the grid that shows where it fails. What survives is the gradient: the deployed rule sends one-adult households to their 1.7th-best canton of 26 and four-adult households to their 7.8th, a spread the mean rule compresses to 2.5 against 3.6.

The comparison set includes the sum of individual probabilities, the metric Annie MOORE uses (Ahani et al. 2021), which maximises expected total employment. It is not neutral either: it displaces small households the way at-least-one displaces large ones.

## Contents

| Path | Description |
|---|---|
| `paper/Weide_FairnessAndAlgorithms_2026.md` | Paper, source format |
| `paper/Weide_FairnessAndAlgorithms_2026.pdf` | Paper, submitted format |
| `paper/Weide_FairnessAndAlgorithms_2026.docx` | Paper, Word format |
| `paper/build.ps1` | Rebuilds the `.docx` and `.pdf` from the markdown source |
| `analysis/case_level_objective.py` | Simulation: all analyses |
| `analysis/run_full.py` | Driver for the full-scale run |
| `analysis/full_run_output.txt` | Raw output of the run reported in the paper |

## Reproducing the results

Requires Python 3.12 with `numpy` and `scipy`.

```bash
pip install numpy scipy
cd analysis
python run_full.py
```

The pipeline mirrors the published replication code: each canton is replicated as many times as it has slots, cost is set to `1 - phi`, and a one-to-one optimal assignment is solved (`scipy.optimize.linear_sum_assignment`, standing in for `optmatch::pairmatch`).

`run_full.py` reproduces every number reported in the paper, in the order it appears. Individual results can also be run separately:

```python
import case_level_objective as m

m.minimal_example()                            # the two-canton illustration in Section 3
m.calibration()                                # pins sigma_idio to the published gain
m.monte_carlo(n_reps=400, n_cases=1000)        # Table 1: efficiency and the gender gap
m.sensitivity(n_reps=250, n_cases=800)         # robustness across interaction strengths
m.idio_sensitivity()                           # robustness across synergy levels
m.within_case_correlation(n_reps=150, n_cases=600)   # robustness to members resembling each other
m.sum_vs_mean_check()                          # sum == mean when all cases are the same size
m.size_effect(n_reps=800, n_cases=800)         # Table 2: the household-size effect
m.size_effect_idio()                           # which part of Table 2 needs synergies
m.gender_rank_effect()                         # the same rank test at the individual level
```

Seeds are fixed, so runs are reproducible. Passing the same `seed`, `n_reps` and `n_cases` to `monte_carlo` and `sensitivity` makes the `sigma_int=0.30` row of the latter reproduce the former exactly, so the two series can be quoted together.

`SIGMA_IDIO` is the standard deviation of person-by-canton idiosyncratic match quality on the logit scale, and the analysis turns on it. At `SIGMA_IDIO=0` everyone of a given gender ranks the cantons identically, the deployed rule's gain over the status quo collapses to 1.4% and the best of the four candidate rules to 2.3%, and the household-size result crosses the lottery for an artefactual reason; `calibration()` prints the grid that pins it to 1.0 instead, where the deployed rule gains 39.4% against the published 41% and the mean variant 47.3% against the published 47%. Gender intercepts are re-solved by quadrature for each value (`intercept_for`), so every specification holds the SEM base rates and differs in synergy alone.

## What the simulation does and does not use

No individual-level ZEMIS data is public, so the modeling stage cannot be replicated. Three inputs are real:

- cantonal capacity shares — Annex 3 AsylV 1, population-proportional key
- employment seven years after entry, 2018 cohort, age 16–55 at entry: men 64%, women 32% (SEM monitoring)
- the published employment gain of the U.S. backtest, 41% under the deployed rule and 47% under its mean variant, used to pin `SIGMA_IDIO`

Everything else — cantonal main effects, canton-by-gender interactions, the household-size mix, and the additive logit functional form — is stylised. The Monte Carlo over parameter draws exists so that no conclusion rests on a single such choice. Section 3 of the paper is a property of the formula and holds without any simulated data; the simulation only establishes that its consequences are large at parameter values consistent with the tool's own published performance.

Rank results are reported against two yardsticks, neither of which any compared rule maximises in the sense that would make the comparison circular: the household's own mean employment probability (which ranks cantons identically to the expected number of employed members, so it does not privilege the mean rule over the sum rule) and the employment probability of the household's least employable member.

The third-party GeoMatch replication archive is not included here; it is available from the Dataverse link above.

## Licence

Code is released under the MIT Licence (`LICENSE`). The paper text is © 2026 Julian Weide, all rights reserved.
