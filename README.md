# Whose Efficiency? The Undefended Normative Core of Algorithmic Refugee Assignment

Term paper and replication code for *Fairness and Algorithms* (851-0746-00L), ETH Zurich, Spring 2026.

Julian Weide

## The argument

GeoMatch, the refugee assignment algorithm piloted by the Swiss State Secretariat for Migration, does not maximise employment. Its mapping stage aggregates individual predicted employment probabilities to the household level using

```
phi(p) = 1 - prod(1 - p_i)
```

the probability that *at least one* household member finds work. This choice is made in a four-line helper function, `compute_femp_prob`, in the authors' replication archive (Bansak et al. 2018, Harvard Dataverse [doi:10.7910/DVN/MS8XES](https://doi.org/10.7910/DVN/MS8XES), `Functions/func_M_to_Mstar.R`).

The paper shows two consequences that follow from the shape of that formula rather than from the training data:

1. **Within households**, the objective prefers concentrating employment prospects in one member over spreading them evenly, and therefore attends least to whoever is already least likely to find work.
2. **Between households**, large households' scores become insensitive to location, so the matching stage displaces them to worse cantons.

Both are quantified by simulation, and both leave the affected group worse off than the characteristics-blind proportional key of Art. 21 AsylV 1 — the legal default and the control arm of the Swiss trial.

## Contents

| Path | Description |
|---|---|
| `paper/Weide_FairnessAndAlgorithms_2026.md` | Paper, source format |
| `paper/Weide_FairnessAndAlgorithms_2026.pdf` | Paper, submitted format |
| `paper/Weide_FairnessAndAlgorithms_2026.docx` | Paper, Word format |
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

Individual results can also be run separately:

```python
import case_level_objective as m

m.monte_carlo(n_reps=400, n_cases=1000)   # Table 1: efficiency and the gender gap
m.sensitivity(n_reps=150, n_cases=600)    # robustness across interaction strengths
m.size_effect(n_reps=1500, n_cases=800)   # Table 2: the household-size effect
m.minimal_example()                       # the two-canton illustration in Section 3
```

Table 2 in the paper uses `n_reps=1500, n_cases=800`; `run_full.py` ships a faster default. The seed is fixed (`np.random.default_rng(20260827)`), so runs are reproducible.

## What the simulation does and does not use

No individual-level ZEMIS data is public, so the modeling stage cannot be replicated. Two inputs are real:

- cantonal capacity shares — Annex 3 AsylV 1, population-proportional key
- employment seven years after entry, 2018 cohort, age 16–55 at entry: men 64%, women 32% (SEM monitoring)

Everything else — cantonal main effects, canton-by-gender interactions, the household-size mix, and the additive logit functional form — is stylised. The Monte Carlo over parameter draws exists so that no conclusion rests on a single such choice. Section 3 of the paper is a property of the formula and holds without any simulated data; the simulation only establishes that its consequences are large at plausible parameter values.

The third-party GeoMatch replication archive is not included here; it is available from the Dataverse link above.

## Licence

Code is released under the MIT Licence (`LICENSE`). The paper text is © 2026 Julian Weide, all rights reserved.
