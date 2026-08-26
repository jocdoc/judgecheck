"""
Synthetic validation for the new by_consistency metric (average q across
EVERY tested event, not just flagged/second-look ones).

WHY THIS EXISTS: averaging q-values across many of a judge's events is a
NEW combined statistic, not a restatement of an existing one -- the same
shape of thing as Fisher's combined probability, which this project already
found fails with a near-100% false-alarm rate when the underlying tests
aren't independent (tests sharing a round aren't independent). This script
checks two things before the metric is trusted:

1. SIGNAL: does the metric actually rank a genuinely persistent, low-grade
   bias judge as "worse" (lower avg_q) than a judge with one severe outlier
   and an otherwise clean, well-tested record? (This is the exact gap the
   old non-clear-only average had -- a persistent-but-rarely-flagged judge
   could be invisible to it.)

2. FALSE ALARM: does a genuinely UNBIASED judge who happens to repeatedly
   judge the same team/panel (correlated tests) get a systematically lower
   average q than an unbiased judge whose tests are more independent --
   purely from correlation, with no real bias present? Run across many
   random seeds and compare distributions, not a single lucky/unlucky draw.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/judgecheck")
from api.analyze import competitor_watch, team_watch, _archive_wide_event_severity, _worst_judges_by_consistency, _worst_judges_by_rate

COMPETITORS = [f"Dancer_{i}" for i in range(40)]
TEAM_OF = {name: f"Team_{i % 5}" for i, name in enumerate(COMPETITORS)}
JUDGES_POOL = ["Clean_A", "Clean_B", "Clean_C", "Clean_D", "CorrelatedClean",
               "OneOutlier", "MildPersistent"]


def build_archive(rng, n_rounds=80, n_competitors_per_round=12, n_judges_per_round=5):
    rounds = []
    for r in range(n_rounds):
        competitors = list(rng.choice(COMPETITORS, size=n_competitors_per_round, replace=False))
        # CorrelatedClean and MildPersistent always judge together with a
        # fixed team focus, to stress-test non-independence; other judges
        # rotate through a random subset of the pool each round.
        judges = ["CorrelatedClean", "MildPersistent"] + list(
            rng.choice([j for j in JUDGES_POOL if j not in ("CorrelatedClean", "MildPersistent")],
                       size=n_judges_per_round - 2, replace=False))
        base = rng.normal(70, 6, size=n_competitors_per_round)
        marks = {j: base + rng.normal(0, 2, size=n_competitors_per_round) for j in judges}

        # OneOutlier: clean every round except ONE, where they heavily favor
        # a single competitor (severe single-event bias).
        if "OneOutlier" in judges and r == 40 and competitors:
            idx = 0
            marks["OneOutlier"][idx] += 25

        # MildPersistent: small but consistent bias favoring Team_0's
        # competitors in every round they judge (persistent, low-grade).
        if "MildPersistent" in judges:
            for i, c in enumerate(competitors):
                if TEAM_OF[c] == "Team_0":
                    marks["MildPersistent"][i] += 3.0

        # CorrelatedClean: genuinely no bias term added at all -- only
        # correlated in WHICH rounds/panels they share, not in behavior.

        df = pd.DataFrame({
            "name": competitors,
            "competitor_id": list(range(1, n_competitors_per_round + 1)),
            "team": [TEAM_OF[c] for c in competitors],
        })
        for j in judges:
            df[j] = marks[j]
        rounds.append((f"Round {r}", df, judges))
    return rounds


def run_once(seed):
    rng = np.random.default_rng(seed)
    rounds = build_archive(rng)
    comp_results, _, _ = competitor_watch(rounds)
    team_results, _ = team_watch(rounds)
    per_judge_events = _archive_wide_event_severity(rounds, comp_results, team_results)
    consistency = {row["judge"]: row for row in _worst_judges_by_consistency(comp_results, team_results, top_n=100)}
    rate = {row["judge"]: row for row in _worst_judges_by_rate(per_judge_events, top_n=100)}
    return consistency, rate


def main():
    print("=== Single-run signal check (seed=0) ===")
    consistency, rate = run_once(0)
    for j in JUDGES_POOL:
        c = consistency.get(j)
        r = rate.get(j)
        print(f"{j:16s}  avg_q={c['avg_q']:.4f} (n_tests={c['n_tests']})" if c else f"{j:16s}  not enough tests",
              f"| n_notable={r['n_notable']}" if r else "| no rate data")

    assert "MildPersistent" in consistency, "MildPersistent should have enough tested events to qualify"
    if "OneOutlier" in consistency:
        assert consistency["MildPersistent"]["avg_q"] < consistency["OneOutlier"]["avg_q"], (
            "FAIL: persistent low-grade bias should rank WORSE (lower avg_q) than a single severe "
            "outlier diluted by an otherwise clean history -- this is the exact gap the redesign "
            "was meant to fix.")
        print("PASS: MildPersistent ranks worse than OneOutlier on the consistency metric.")
    else:
        print("NOTE: OneOutlier had too few tested events to qualify this run -- inconclusive on this point.")

    print("\n=== False-alarm check: CorrelatedClean vs independent Clean judges, 40 seeds ===")
    correlated_avgs, independent_avgs = [], []
    for seed in range(1, 41):
        consistency, _ = run_once(seed)
        if "CorrelatedClean" in consistency:
            correlated_avgs.append(consistency["CorrelatedClean"]["avg_q"])
        for j in ("Clean_A", "Clean_B", "Clean_C", "Clean_D"):
            if j in consistency:
                independent_avgs.append(consistency[j]["avg_q"])

    correlated_avgs = np.array(correlated_avgs)
    independent_avgs = np.array(independent_avgs)
    print(f"CorrelatedClean: mean avg_q = {correlated_avgs.mean():.4f}  (n={len(correlated_avgs)} seeds)")
    print(f"Independent Clean judges: mean avg_q = {independent_avgs.mean():.4f}  (n={len(independent_avgs)} rows)")
    diff = correlated_avgs.mean() - independent_avgs.mean()
    print(f"Difference (correlated - independent): {diff:+.4f}")

    # A genuinely unbiased-but-correlated judge should NOT systematically
    # score lower (worse) than genuinely unbiased-and-independent judges.
    # Small sampling noise is expected; a large, consistent negative gap
    # would indicate the same non-independence problem that broke Fisher's
    # combined-probability approach elsewhere in this project.
    if diff < -0.05:
        print("FAIL-ish: CorrelatedClean is scoring meaningfully worse purely from shared-round "
              "correlation, with no real bias present. This metric would need a fix "
              "(e.g. one q per judge-round instead of per judge-competitor pair) before shipping.")
    else:
        print("PASS: no meaningful false-alarm drift from shared-round correlation alone.")


if __name__ == "__main__":
    main()
