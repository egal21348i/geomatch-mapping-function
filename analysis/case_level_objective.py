"""
The case-level objective in GeoMatch-style refugee assignment.

Compares three case-level mapping functions phi that aggregate individual
predicted employment probabilities to a household ("case") score:

    at-least-one : phi(p) = 1 - prod(1 - p_i)      [Bansak et al. 2018, default]
    mean         : phi(p) = mean(p_i)
    maxmin       : phi(p) = min(p_i)

The matching stage mirrors the published replication code: each canton is
replicated as many times as it has slots, cost = 1 - phi, then a 1:1 optimal
assignment is solved (func_Mstar_to_D + optmatch::pairmatch).

Calibration targets (published aggregates):
  - cantonal capacity shares: Anhang 3 AsylV 1 (population-proportional key)
  - employment rate 7 years after entry, 2018 entry cohort, age 16-55 at entry:
    men 64%, women 32% (SEM, Erwerbssituation von VA/FL)

Cantonal main effects and canton-by-gender interactions are stylized; the
Monte Carlo over their draws is what makes the conclusion parameter-free.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

RNG = np.random.default_rng(20260827)

# Anhang 3 AsylV 1: population-proportional distribution key (percent, sums to 100)
CANTON_KEY = {
    "AG": 8.1, "AR": 0.6, "AI": 0.2, "BL": 3.3, "BS": 2.2, "BE": 11.8,
    "FR": 3.8, "GE": 5.9, "GL": 0.5, "GR": 2.3, "JU": 0.8, "LU": 4.8,
    "NE": 2.0, "NW": 0.5, "OW": 0.4, "SH": 1.0, "SZ": 1.9, "SO": 3.2,
    "SG": 6.0, "TI": 4.0, "TG": 3.3, "UR": 0.4, "VD": 9.5, "VS": 4.1,
    "ZG": 1.5, "ZH": 17.9,
}

P_MALE, P_FEMALE = 0.64, 0.32          # SEM monitoring, 2018 cohort
SIGMA_INDIV = 0.5                       # individual heterogeneity (logit scale)
SIGMA_CANTON = 0.30                     # canton main effect (logit scale)
SIGMA_INTERACT = 0.30                   # canton-by-gender interaction (logit scale)


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def phi_at_least_one(p):
    """P(at least one member employed), assuming within-case independence."""
    return 1 - np.prod(1 - p, axis=-1)


def phi_mean(p):
    return p.mean(axis=-1)


def phi_maxmin(p):
    return p.min(axis=-1)


RULES = {
    "at-least-one": phi_at_least_one,
    "mean": phi_mean,
    "maxmin": phi_maxmin,
}


def draw_probabilities(n_cases, sigma_interact=SIGMA_INTERACT, rng=RNG):
    """Return p of shape (n_cases, n_cantons, 2); member 0 = man, 1 = woman."""
    n_cantons = len(CANTON_KEY)

    beta = rng.normal(0, SIGMA_CANTON, n_cantons)
    beta -= beta.mean()
    delta = rng.normal(0, sigma_interact, n_cantons)
    delta -= delta.mean()

    u = rng.normal(0, SIGMA_INDIV, (n_cases, 2))

    base = np.array([logit(P_MALE), logit(P_FEMALE)])
    sign = np.array([-1.0, 1.0])          # interaction shifts women up, men down

    lin = (base[None, None, :]
           + u[:, None, :]
           + beta[None, :, None]
           + sign[None, None, :] * delta[None, :, None])
    return expit(lin)


def capacity_slots(n_cases):
    """Integer slots per canton proportional to the statutory key, summing to n_cases."""
    shares = np.array(list(CANTON_KEY.values())) / 100.0
    raw = shares * n_cases
    slots = np.floor(raw).astype(int)
    remainder = n_cases - slots.sum()
    if remainder:
        order = np.argsort(-(raw - slots))
        slots[order[:remainder]] += 1
    return slots


def constrained_assignment(mstar, slots):
    """Replicate each canton by its slots, then solve the 1:1 optimal matching."""
    canton_of_slot = np.repeat(np.arange(len(slots)), slots)
    cost = 1 - mstar[:, canton_of_slot]
    rows, cols = linear_sum_assignment(cost)
    chosen = np.empty(mstar.shape[0], dtype=int)
    chosen[rows] = canton_of_slot[cols]
    return chosen


def evaluate(p, chosen):
    """Individual-level outcomes at the chosen cantons."""
    sel = p[np.arange(p.shape[0]), chosen, :]
    return {
        "employed_per_case": sel.sum(axis=1).mean(),
        "p_male": sel[:, 0].mean(),
        "p_female": sel[:, 1].mean(),
        "within_case_gap": np.abs(sel[:, 0] - sel[:, 1]).mean(),
    }


def random_baseline(p, slots):
    """Status quo: capacity-weighted random assignment, blind to characteristics."""
    weights = slots / slots.sum()
    per_canton = p.sum(axis=2)                       # (n_cases, n_cantons)
    return {
        "employed_per_case": (per_canton * weights).sum(axis=1).mean(),
        "p_male": (p[:, :, 0] * weights).sum(axis=1).mean(),
        "p_female": (p[:, :, 1] * weights).sum(axis=1).mean(),
        "within_case_gap": (np.abs(p[:, :, 0] - p[:, :, 1]) * weights).sum(axis=1).mean(),
    }


def one_replication(n_cases, sigma_interact, rng):
    p = draw_probabilities(n_cases, sigma_interact, rng)
    slots = capacity_slots(n_cases)

    out = {"baseline": random_baseline(p, slots)}
    choices = {}
    for name, phi in RULES.items():
        mstar = phi(p)
        chosen = constrained_assignment(mstar, slots)
        choices[name] = chosen
        out[name] = evaluate(p, chosen)

    out["disagreement"] = float(np.mean(choices["at-least-one"] != choices["mean"]))

    # Unconstrained preference reversal: ignoring capacity, how often does the
    # at-least-one rule pick a canton with a strictly lower within-case mean?
    argmax_alo = phi_at_least_one(p).argmax(axis=1)
    argmax_mean = phi_mean(p).argmax(axis=1)
    mean_at_alo = phi_mean(p)[np.arange(n_cases), argmax_alo]
    mean_at_mean = phi_mean(p)[np.arange(n_cases), argmax_mean]
    out["reversal_unconstrained"] = float(np.mean(mean_at_alo < mean_at_mean - 1e-12))
    out["mean_loss_unconstrained"] = float(np.mean(mean_at_mean - mean_at_alo))
    return out


def monte_carlo(n_reps=120, n_cases=400, sigma_interact=SIGMA_INTERACT, seed=20260827):
    rng = np.random.default_rng(seed)
    reps = [one_replication(n_cases, sigma_interact, rng) for _ in range(n_reps)]

    def collect(path):
        if isinstance(path, tuple):
            return np.array([r[path[0]][path[1]] for r in reps])
        return np.array([r[path] for r in reps])

    print(f"\n{'='*78}")
    print(f"Monte Carlo: {n_reps} draws of cantonal effects, {n_cases} two-adult "
          f"households each")
    print(f"sigma_canton={SIGMA_CANTON}, sigma_interaction={sigma_interact}, "
          f"sigma_individual={SIGMA_INDIV}")
    print(f"{'='*78}")

    base_emp = collect(("baseline", "employed_per_case"))
    print(f"\nStatus quo (population-proportional, characteristics-blind):")
    print(f"  expected employed adults per household : {base_emp.mean():.4f}")
    print(f"  P(employed), men                       : "
          f"{collect(('baseline','p_male')).mean():.4f}")
    print(f"  P(employed), women                     : "
          f"{collect(('baseline','p_female')).mean():.4f}")
    print(f"  mean within-household gap              : "
          f"{collect(('baseline','within_case_gap')).mean():.4f}")

    print(f"\n{'rule':<14}{'employed/hh':>13}{'gain vs SQ':>12}"
          f"{'P(f)':>9}{'P(m)':>9}{'gap':>9}")
    print("-" * 78)
    for name in RULES:
        emp = collect((name, "employed_per_case"))
        gain = 100 * (emp / base_emp - 1)
        print(f"{name:<14}{emp.mean():>13.4f}{gain.mean():>11.1f}%"
              f"{collect((name,'p_female')).mean():>9.4f}"
              f"{collect((name,'p_male')).mean():>9.4f}"
              f"{collect((name,'within_case_gap')).mean():>9.4f}")

    print("-" * 78)
    alo = collect(("at-least-one", "employed_per_case"))
    mean_ = collect(("mean", "employed_per_case"))
    gain_alo = 100 * (alo / base_emp - 1)
    gain_mean = 100 * (mean_ / base_emp - 1)
    gap = gain_mean - gain_alo
    forgone_share = 100 * gap / gain_mean

    print(f"\nEfficiency forgone by choosing at-least-one over mean:")
    print(f"  {gap.mean():.2f} pp of the gain over the status quo "
          f"(90% MC interval {np.quantile(gap,0.05):.2f} to {np.quantile(gap,0.95):.2f})")
    print(f"  = {forgone_share.mean():.1f}% of the achievable gain is given up "
          f"(90% MC interval {np.quantile(forgone_share,0.05):.1f} to "
          f"{np.quantile(forgone_share,0.95):.1f})")
    print(f"  share of replications where mean beats at-least-one: "
          f"{100*np.mean(gap > 0):.1f}%")

    fem_alo = collect(("at-least-one", "p_female"))
    fem_mean = collect(("mean", "p_female"))
    print(f"\nWomen's employment probability, mean rule minus at-least-one rule:")
    print(f"  {100*(fem_mean - fem_alo).mean():+.2f} percentage points "
          f"(90% MC interval {100*np.quantile(fem_mean-fem_alo,0.05):+.2f} to "
          f"{100*np.quantile(fem_mean-fem_alo,0.95):+.2f})")

    print(f"\nWithin-household gap relative to the characteristics-blind status quo:")
    base_gap = collect(("baseline", "within_case_gap"))
    for name in RULES:
        d = collect((name, "within_case_gap")) - base_gap
        worse = 100 * np.mean(d > 0)
        print(f"  {name:<14}{100*d.mean():+7.2f} pp   "
              f"(widens the gap in {worse:.0f}% of replications)")

    print(f"\nHouseholds assigned to a different canton by the two rules: "
          f"{100*collect('disagreement').mean():.1f}%")
    print(f"Unconstrained preference reversals (at-least-one picks a canton with "
          f"strictly\n  lower within-household mean): "
          f"{100*collect('reversal_unconstrained').mean():.1f}% of households")
    return reps


def sensitivity(sigmas=(0.10, 0.20, 0.30, 0.40, 0.50), n_reps=40, n_cases=300):
    print(f"\n{'='*78}")
    print("Sensitivity to the strength of the canton-by-gender interaction")
    print(f"{'='*78}")
    print(f"{'sigma_int':>10}{'reversals':>12}{'disagree':>11}"
          f"{'eff. forgone':>15}{'women pp':>11}")
    print("-" * 78)
    for s in sigmas:
        rng = np.random.default_rng(4711)
        reps = [one_replication(n_cases, s, rng) for _ in range(n_reps)]
        base = np.array([r["baseline"]["employed_per_case"] for r in reps])
        alo = np.array([r["at-least-one"]["employed_per_case"] for r in reps])
        mn = np.array([r["mean"]["employed_per_case"] for r in reps])
        forgone = (100 * (mn / base - 1) - 100 * (alo / base - 1)).mean()
        rev = 100 * np.mean([r["reversal_unconstrained"] for r in reps])
        dis = 100 * np.mean([r["disagreement"] for r in reps])
        fem = 100 * np.mean([r["mean"]["p_female"] - r["at-least-one"]["p_female"]
                             for r in reps])
        print(f"{s:>10.2f}{rev:>11.1f}%{dis:>10.1f}%{forgone:>14.2f}pp{fem:>+10.2f}pp")


def minimal_example():
    print(f"\n{'='*78}")
    print("Minimal example: one couple, two cantons")
    print(f"{'='*78}")
    for label, pm, pf in [("A", 0.80, 0.10), ("B", 0.50, 0.50)]:
        p = np.array([pm, pf])
        print(f"  canton {label}: man {pm:.2f}, woman {pf:.2f}  |  "
              f"at-least-one {1-np.prod(1-p):.4f}   mean {p.mean():.4f}   "
              f"min {p.min():.4f}")
    print("  at-least-one prefers A; mean and min prefer B.")
    print("  A yields 0.90 expected employed adults, B yields 1.00.")


SIZE_WEIGHTS = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.15}   # stylised household-size mix
MAX_SIZE = max(SIZE_WEIGHTS)


def draw_probabilities_varsize(n_cases, sigma_interact=SIGMA_INTERACT, rng=RNG):
    """Households of size 1-4 (stylised mix). Returns:
      p     : (n_cases, n_cantons, MAX_SIZE), zero-padded for absent members
      sizes : (n_cases,) integer household size
    """
    n_cantons = len(CANTON_KEY)
    sizes = rng.choice(list(SIZE_WEIGHTS.keys()), size=n_cases,
                        p=list(SIZE_WEIGHTS.values()))
    mask = (np.arange(MAX_SIZE)[None, :] < sizes[:, None]).astype(float)

    beta = rng.normal(0, SIGMA_CANTON, n_cantons)
    beta -= beta.mean()
    delta = rng.normal(0, sigma_interact, n_cantons)
    delta -= delta.mean()

    gender = rng.integers(0, 2, (n_cases, MAX_SIZE))          # 0=man, 1=woman
    u = rng.normal(0, SIGMA_INDIV, (n_cases, MAX_SIZE))
    base = np.where(gender == 0, logit(P_MALE), logit(P_FEMALE))
    sign = np.where(gender == 0, -1.0, 1.0)

    lin = (base[:, None, :] + u[:, None, :] + beta[None, :, None]
           + sign[:, None, :] * delta[None, :, None])
    p = expit(lin) * mask[:, None, :]     # padding members contribute (1-0)=1 to the product
    return p, sizes


def phi_mean_sized(p, sizes):
    return p.sum(axis=-1) / sizes[:, None]


def random_assignment(n_cases, slots, rng):
    """One realised draw of the characteristics-blind, capacity-respecting lottery."""
    canton_of_slot = np.repeat(np.arange(len(slots)), slots)
    rng.shuffle(canton_of_slot)
    return canton_of_slot[:n_cases]


def canton_rank(quality, chosen):
    """For each case, rank (1=best) of its assigned canton among all cantons,
    ranked by that case's own quality score."""
    order = np.argsort(-quality, axis=1)               # best-to-worst canton per case
    rank_of = np.empty_like(order)
    rows = np.arange(quality.shape[0])[:, None]
    rank_of[rows, order] = np.arange(1, quality.shape[1] + 1)[None, :]
    return rank_of[np.arange(quality.shape[0]), chosen]


def size_effect(n_reps=150, n_cases=600, seed=20260828):
    """Does the at-least-one rule send larger households to cantons that are
    worse for their own true (mean-probability) prospects, relative to the
    mean rule and the characteristics-blind lottery?"""
    rng = np.random.default_rng(seed)
    sizes_seen = sorted(SIZE_WEIGHTS)
    n_cantons = len(CANTON_KEY)

    sums = {s: {"alo_rank": 0.0, "mean_rank": 0.0, "base_rank": 0.0,
                "alo_q": 0.0, "mean_q": 0.0, "base_q": 0.0,
                "alo_bottomq": 0.0, "base_bottomq": 0.0,
                "ceiling": 0.0, "floor": 0.0, "spread": 0.0,
                "regret_alo": 0.0, "regret_mean": 0.0, "regret_base": 0.0, "n": 0}
            for s in sizes_seen}

    for _ in range(n_reps):
        p, sizes = draw_probabilities_varsize(n_cases, rng=rng)
        slots = capacity_slots(n_cases)
        quality = phi_mean_sized(p, sizes)          # "true" per-member employment prospect

        chosen_alo = constrained_assignment(phi_at_least_one(p), slots)
        chosen_mean = constrained_assignment(quality, slots)
        chosen_base = random_assignment(n_cases, slots, rng)

        rank_alo = canton_rank(quality, chosen_alo)
        rank_mean = canton_rank(quality, chosen_mean)
        rank_base = canton_rank(quality, chosen_base)

        q_alo = quality[np.arange(n_cases), chosen_alo]
        q_mean = quality[np.arange(n_cases), chosen_mean]
        q_base = quality[np.arange(n_cases), chosen_base]

        # cardinal ("how much is actually at stake") counterparts to the ordinal rank:
        ceiling = quality.max(axis=1)       # this household's best-case canton, in probability
        floor = quality.min(axis=1)         # this household's worst-case canton
        spread = ceiling - floor            # how much the canton choice could possibly matter

        bottom_cut = 0.75 * n_cantons
        for s in sizes_seen:
            m = sizes == s
            n = m.sum()
            sums[s]["alo_rank"] += rank_alo[m].sum()
            sums[s]["mean_rank"] += rank_mean[m].sum()
            sums[s]["base_rank"] += rank_base[m].sum()
            sums[s]["alo_q"] += q_alo[m].sum()
            sums[s]["mean_q"] += q_mean[m].sum()
            sums[s]["base_q"] += q_base[m].sum()
            sums[s]["alo_bottomq"] += (rank_alo[m] > bottom_cut).sum()
            sums[s]["base_bottomq"] += (rank_base[m] > bottom_cut).sum()
            sums[s]["ceiling"] += ceiling[m].sum()
            sums[s]["floor"] += floor[m].sum()
            sums[s]["spread"] += spread[m].sum()
            sums[s]["regret_alo"] += (ceiling[m] - q_alo[m]).sum()
            sums[s]["regret_mean"] += (ceiling[m] - q_mean[m]).sum()
            sums[s]["regret_base"] += (ceiling[m] - q_base[m]).sum()
            sums[s]["n"] += n

    print(f"\n{'='*90}")
    print(f"Household-size effect: {n_reps} draws x {n_cases} households "
          f"(size mix {SIZE_WEIGHTS}), {n_cantons} cantons")
    print("quality = each household's own mean employment probability (not the score "
          "it was matched on)")
    print(f"{'='*90}")
    print(f"{'size':>5}{'n':>9}{'  rank: at-least-one':>21}{'rank: mean':>13}"
          f"{'rank: lottery':>15}{'quality: alo':>14}{'quality: mean':>15}"
          f"{'bottom-Q%: alo':>16}{'bottom-Q%: lot.':>17}")
    print("-" * 90)
    for s in sizes_seen:
        d = sums[s]
        n = d["n"]
        print(f"{s:>5}{n:>9}{d['alo_rank']/n:>21.2f}{d['mean_rank']/n:>13.2f}"
              f"{d['base_rank']/n:>15.2f}{d['alo_q']/n:>14.4f}{d['mean_q']/n:>15.4f}"
              f"{100*d['alo_bottomq']/n:>15.1f}%{100*d['base_bottomq']/n:>16.1f}%")
    print("-" * 90)
    print("\nCardinal counterparts (probability units, not rank):")
    print(f"{'size':>5}{'n':>9}{'own spread (max-min)':>22}{'quality: lottery':>18}"
          f"{'regret: alo':>13}{'regret: mean':>14}{'regret: lottery':>17}")
    print("-" * 90)
    for s in sizes_seen:
        d = sums[s]
        n = d["n"]
        print(f"{s:>5}{n:>9}{d['spread']/n:>22.4f}{d['base_q']/n:>18.4f}"
              f"{d['regret_alo']/n:>13.4f}{d['regret_mean']/n:>14.4f}"
              f"{d['regret_base']/n:>17.4f}")
    print("-" * 90)
    print("own spread: this household's own best-canton quality minus its own worst-canton")
    print("  quality -- how much the canton choice could possibly be worth to it.")
    print("regret: this household's own best-canton quality minus what it actually got")
    print("  under each rule -- a feasibility-blind upper bound, not a claim that regret 0")
    print("  is attainable for everyone at once under capacity constraints.")
    # Pooled across sizes at their actual mix (SIZE_WEIGHTS) -- answers "does this
    # wash out in the population average if large households are a minority?"
    n_all = sum(sums[s]["n"] for s in sizes_seen)
    alo_all = sum(sums[s]["alo_rank"] for s in sizes_seen) / n_all
    mean_all = sum(sums[s]["mean_rank"] for s in sizes_seen) / n_all
    base_all = sum(sums[s]["base_rank"] for s in sizes_seen) / n_all
    bq_alo_all = 100 * sum(sums[s]["alo_bottomq"] for s in sizes_seen) / n_all
    bq_base_all = 100 * sum(sums[s]["base_bottomq"] for s in sizes_seen) / n_all
    print(f"{'ALL':>5}{n_all:>9}{alo_all:>21.2f}{mean_all:>13.2f}{base_all:>15.2f}"
          f"{'':>14}{'':>15}{bq_alo_all:>15.1f}%{bq_base_all:>16.1f}%")
    print("-" * 90)
    print("rank: 1 = the best of 26 cantons for that household's own prospects, "
          "26 = the worst.")
    print("bottom-Q%: share of households sent to a canton in their own worst quartile.")
    print(f"ALL = pooled across the stylised size mix {SIZE_WEIGHTS}: the population-wide "
          f"average, i.e. what you would see if you ignored household size altogether.")


def gender_rank_effect(n_reps=200, n_cases=800, sigma_interact=SIGMA_INTERACT, seed=20260828):
    """Individual analogue of size_effect: rank the assigned canton against each
    person's OWN individual employment probability (not the household score),
    separately for men and women, under each rule."""
    rng = np.random.default_rng(seed)
    n_cantons = len(CANTON_KEY)
    labels = {0: "men", 1: "women"}
    sums = {g: {"alo_rank": 0.0, "mean_rank": 0.0, "base_rank": 0.0,
                "alo_bottomq": 0.0, "base_bottomq": 0.0, "n": 0} for g in (0, 1)}

    for _ in range(n_reps):
        p = draw_probabilities(n_cases, sigma_interact, rng)     # (n_cases, n_cantons, 2)
        slots = capacity_slots(n_cases)
        chosen_alo = constrained_assignment(phi_at_least_one(p), slots)
        chosen_mean = constrained_assignment(phi_mean(p), slots)
        chosen_base = random_assignment(n_cases, slots, rng)

        bottom_cut = 0.75 * n_cantons
        for g in (0, 1):
            quality_g = p[:, :, g]                # this person's own true probability per canton
            r_alo = canton_rank(quality_g, chosen_alo)
            r_mean = canton_rank(quality_g, chosen_mean)
            r_base = canton_rank(quality_g, chosen_base)
            sums[g]["alo_rank"] += r_alo.sum()
            sums[g]["mean_rank"] += r_mean.sum()
            sums[g]["base_rank"] += r_base.sum()
            sums[g]["alo_bottomq"] += (r_alo > bottom_cut).sum()
            sums[g]["base_bottomq"] += (r_base > bottom_cut).sum()
            sums[g]["n"] += n_cases

    print(f"\n{'='*90}")
    print(f"Gender rank effect: {n_reps} draws x {n_cases} couples, {n_cantons} cantons")
    print("quality = each individual's own true employment probability (not the "
          "household score)")
    print(f"{'='*90}")
    print(f"{'':>7}{'n':>9}{'rank: at-least-one':>20}{'rank: mean':>13}"
          f"{'rank: lottery':>15}{'bottom-Q%: alo':>16}{'bottom-Q%: lot.':>17}")
    print("-" * 90)
    for g in (0, 1):
        d = sums[g]
        n = d["n"]
        print(f"{labels[g]:>7}{n:>9}{d['alo_rank']/n:>20.2f}{d['mean_rank']/n:>13.2f}"
              f"{d['base_rank']/n:>15.2f}{100*d['alo_bottomq']/n:>15.1f}%"
              f"{100*d['base_bottomq']/n:>16.1f}%")
    print("-" * 90)
    print("rank: 1 = the best of 26 cantons for that person's own prospects, 26 = the worst.")


def high_synergy():
    """Stronger canton effects, closer to the synergies a full GBM can exploit."""
    global SIGMA_CANTON
    saved = SIGMA_CANTON
    SIGMA_CANTON = 0.60
    print(f"\n{'#'*78}")
    print("HIGH-SYNERGY SCENARIO (sigma_canton=0.60)")
    print(f"{'#'*78}")
    monte_carlo(n_reps=80, n_cases=400, sigma_interact=0.50, seed=99)
    SIGMA_CANTON = saved


if __name__ == "__main__":
    minimal_example()
    monte_carlo()
    sensitivity()
    high_synergy()
    size_effect()
    gender_rank_effect()
