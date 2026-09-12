"""Full-scale run: every number reported in the paper, in the order it appears."""

import case_level_objective as m

m.minimal_example()

# Section 4.1: what the person-by-canton term has to be for the simulated
# matching problem to be the one the tool actually solves.
m.calibration(n_reps=60, n_cases=600)

# Section 4.2, Table 1: the calibrated specification (sigma_idio = 1.0),
# then the two ends of the published gain range.
m.monte_carlo(n_reps=400, n_cases=1000)
m.monte_carlo(n_reps=150, n_cases=1000, sigma_idio=0.0)
m.monte_carlo(n_reps=150, n_cases=1000, sigma_idio=2.0)

# Robustness of Section 4.2: interaction strength, synergy level, and
# members of a household resembling each other.
m.sensitivity(n_reps=250, n_cases=800)
m.idio_sensitivity(sigmas=(0.0, 0.5, 1.0, 2.0), n_reps=150, n_cases=600)
m.within_case_correlation(n_reps=150, n_cases=600)
m.sum_vs_mean_check(n_cases=800, n_reps=30)

# Section 4.3, Table 2: the household-size effect, and which part of it
# depends on households ranking the cantons identically.
m.size_effect(n_reps=800, n_cases=800)
m.size_effect_idio(sigmas=(0.0, 0.3, 1.0, 2.0), n_reps=250, n_cases=800)

# Section 4.2, individual-level counterpart: are women sent to cantons that
# are worse by their own prospects?
m.gender_rank_effect(n_reps=150, n_cases=800)
