"""Full-scale run: 400 draws x 1000 households, plus a better-powered sensitivity."""

import case_level_objective as m

m.monte_carlo(n_reps=400, n_cases=1000)
m.sensitivity(n_reps=150, n_cases=600)
m.size_effect(n_reps=200, n_cases=800)

