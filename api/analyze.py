"""
/api/analyze -- the single serverless endpoint behind the site.

THREE REQUEST SHAPES, same endpoint:
1. {files: [{name, content}]}  where all files are .csv/.xlsx
   -> runs analysis immediately, returns {html, mode}  (unchanged from v1)
2. {files: [{name, content}]}  where a file is .pdf
   -> does NOT analyze yet. Parses the PDF and returns
      {preview: true, format, rounds: [{label, judges, rows}], warnings}
      for the browser to show as an editable table.
3. {confirmed_rounds: [{label, judges, rows}], title}
   -> the browser sends back the (possibly hand-edited) table from step 2
      and THIS runs the actual analysis, returning {html, mode}.

Nothing is stored between requests -- each call is stateless; the browser
carries the extracted/edited table in step 2->3, not the server.
"""

import base64
import os
import io
import json
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .pdf_parsers import parse_results_pdf, UnrecognizedFormatError

MAX_FILES = 6  # lowered from 12 after real timing test: 12 real PDFs took 67s to parse,
               # exceeding the 60s Vercel function limit. 6 files measured at ~30s locally,
               # leaving real margin for live-deployment overhead (cold starts, network I/O)
               # that this sandbox doesn't fully reproduce.
MAX_COMPETITORS = 500
MAX_JUDGES = 15
MIN_JUDGES = 3
MIN_COMPETITORS = 8
N_PERMUTATIONS = 1000
RESERVED_COLUMNS = {"competitor_id", "team", "name"}


class UserError(Exception):
    pass


# ===========================================================================
# CORE STATISTICS (unchanged from v1 -- vectorized, validated against the
# slow reference implementation; see project history)
# ===========================================================================
def normalize_scores(scores_df, judge_columns):
    normalized = scores_df.copy()
    for judge in judge_columns:
        std = scores_df[judge].std()
        normalized[judge] = (scores_df[judge] - scores_df[judge].mean()) / std
    return normalized


def _min_z_and_agreements(matrix):
    ranks = sstats.rankdata(matrix, axis=0)
    corr = np.corrcoef(ranks, rowvar=False)
    k = matrix.shape[1]
    avg_agreement = (corr.sum(axis=1) - 1) / (k - 1)
    std = avg_agreement.std(ddof=1)
    z = (avg_agreement - avg_agreement.mean()) / std if std > 0 else np.zeros(k)
    return z.min(), avg_agreement, z


def calibrated_judge_test(matrix, n_permutations=N_PERMUTATIONS, seed=0):
    rng = np.random.default_rng(seed)
    observed_min_z, avg_agreement, z_scores = _min_z_and_agreements(matrix)
    n, k = matrix.shape
    null_min = np.empty(n_permutations)
    for p in range(n_permutations):
        idx = np.argsort(rng.random((n, k)), axis=1)
        shuffled = np.take_along_axis(matrix, idx, axis=1)
        null_min[p], _, _ = _min_z_and_agreements(shuffled)
    p_value = float(np.mean(null_min <= observed_min_z))
    return observed_min_z, avg_agreement, z_scores, p_value, null_min


def team_bias_check(scores_df, normalized_df, judge_columns, judge_of_interest,
                     team_column="team", n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    other = [j for j in judge_columns if j != judge_of_interest]
    gap = (normalized_df[judge_of_interest] - normalized_df[other].mean(axis=1)).values
    teams = scores_df[team_column].values
    results = []
    for team in pd.unique(teams):
        mask = teams == team
        observed = gap[mask].mean()
        size = int(mask.sum())
        shuffled = np.empty(n_permutations)
        for i in range(n_permutations):
            pick = rng.choice(len(gap), size=size, replace=False)
            shuffled[i] = gap[pick].mean()
        p = float(np.mean(np.abs(shuffled) >= abs(observed)))
        results.append({"team": str(team), "n": size, "gap": observed,
                        "direction": "favored" if observed > 0 else "punished", "p": p})
    results.sort(key=lambda r: r["p"])
    return results


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    raw_q = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(adjusted, 0, 1)
    return q


def _competitor_identity(df, i):
    """Returns (key, display_label, method) for one row. Prefers matching by
    name across events (since competition numbers are assigned fresh per
    event and are NOT a reliable way to track the same person over time);
    falls back to competitor_id only when no usable name is present, and
    says so, since that fallback is a real reliability downgrade worth
    surfacing rather than hiding."""
    name = str(df["name"].iloc[i]).strip() if "name" in df.columns and pd.notna(df["name"].iloc[i]) else ""
    if name:
        return (("name", name), name, "name")
    cid = df["competitor_id"].iloc[i]
    return (("id", cid), f"Competitor #{cid}", "competitor_id")


def competitor_watch(events, min_shared_events=3, q_threshold=0.10):
    rows = []
    for label, df, judge_cols in events:
        normalized = normalize_scores(df, judge_cols)
        for judge in judge_cols:
            other = [j for j in judge_cols if j != judge]
            gap = normalized[judge] - normalized[other].mean(axis=1)
            for i in range(len(df)):
                key, display, method = _competitor_identity(df, i)
                rows.append({"key": key, "display": display, "method": method,
                            "team": str(df["team"].iloc[i]) if "team" in df.columns else "",
                            "judge": judge, "gap": float(gap.iloc[i])})
    history = pd.DataFrame(rows)
    if history.empty:
        return [], 0, {"name": 0, "competitor_id": 0}
    match_method_counts = history.drop_duplicates(subset=["key"])["method"].value_counts().to_dict()

    results = []
    for (key, judge), grp in history.groupby(["key", "judge"]):
        if len(grp) < min_shared_events:
            continue
        gaps = grp["gap"].values
        _, p = sstats.ttest_1samp(gaps, popmean=0.0)
        results.append({"display": grp["display"].iloc[0], "team": grp["team"].iloc[0], "judge": judge,
                        "match_method": grp["method"].iloc[0],
                        "events": len(grp), "avg_gap": float(gaps.mean()),
                        "direction": "favored" if gaps.mean() > 0 else "punished", "p": float(p)})
    if not results:
        return [], 0, {"name": match_method_counts.get("name", 0), "competitor_id": match_method_counts.get("competitor_id", 0)}
    q = benjamini_hochberg([r["p"] for r in results])
    for r, qv in zip(results, q):
        r["q"] = float(qv)
        r["flagged"] = bool(qv < q_threshold)
    results.sort(key=lambda r: r["q"])
    return results, len(results), {"name": match_method_counts.get("name", 0),
                                    "competitor_id": match_method_counts.get("competitor_id", 0)}


def team_watch(events, min_observations=5, q_threshold=0.10):
    """
    Cross-event version of the single-event team-bias check: does a specific
    judge consistently score a specific SCHOOL's competitors higher or lower
    than the rest of the panel, pooled across every uploaded event?

    WHY THIS EXISTS SEPARATELY FROM competitor_watch: that function needs the
    SAME PERSON to reappear across multiple SEPARATE events to build up
    enough history -- which is the right question for "does this judge
    target this dancer over time," but gives a useless answer (zero shared
    people) when someone uploads several DIFFERENT divisions from the SAME
    event, since a U16 dancer and a U17 dancer are different people by
    definition. A school, though, typically has multiple dancers spread
    across several divisions at one event, and the same judge often judges
    more than one of those divisions -- so a team has enough data to check
    even when no individual competitor does. `min_observations` counts
    individual competitor encounters, not separate events, since several
    team members in a single event already provide independent data points.
    """
    rows = []
    for label, df, judge_cols in events:
        if "team" not in df.columns:
            continue
        normalized = normalize_scores(df, judge_cols)
        for judge in judge_cols:
            other = [j for j in judge_cols if j != judge]
            gap = normalized[judge] - normalized[other].mean(axis=1)
            for i in range(len(df)):
                team = str(df["team"].iloc[i]) if pd.notna(df["team"].iloc[i]) else ""
                if not team:
                    continue
                rows.append({"team": team, "judge": judge, "gap": float(gap.iloc[i]), "event": label})
    history = pd.DataFrame(rows)
    if history.empty:
        return [], 0

    results = []
    for (team, judge), grp in history.groupby(["team", "judge"]):
        if len(grp) < min_observations:
            continue
        gaps = grp["gap"].values
        _, p = sstats.ttest_1samp(gaps, popmean=0.0)
        results.append({"team": team, "judge": judge, "n_competitors": len(grp),
                        "n_rounds": grp["event"].nunique(), "avg_gap": float(gaps.mean()),
                        "direction": "favored" if gaps.mean() > 0 else "punished", "p": float(p)})
    if not results:
        return [], 0
    q = benjamini_hochberg([r["p"] for r in results])
    for r, qv in zip(results, q):
        r["q"] = float(qv)
        r["flagged"] = bool(qv < q_threshold)
    results.sort(key=lambda r: r["q"])
    return results, len(results)


# ===========================================================================
# CSV/XLSX PARSING AND VALIDATION (unchanged from v1)
# ===========================================================================
def parse_uploaded_file(name, b64content):
    raw = base64.b64decode(b64content)
    buffer = io.BytesIO(raw)
    lower = name.lower()
    try:
        if lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(buffer)
        elif lower.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            raise UserError(f'"{name}": unsupported file type. Please upload .csv, .xlsx, or .pdf files.')
    except UserError:
        raise
    except Exception:
        raise UserError(f'"{name}" could not be read. Make sure it is a valid CSV or Excel file.')

    df.columns = [str(c).strip() for c in df.columns]
    if "competitor_id" not in df.columns:
        raise UserError(f'"{name}" is missing a "competitor_id" column. '
                        'Download the template on this page to see the expected format.')

    judge_cols = [c for c in df.columns if c not in RESERVED_COLUMNS]
    if len(judge_cols) < MIN_JUDGES:
        raise UserError(f'"{name}" has {len(judge_cols)} judge column(s); at least {MIN_JUDGES} judges '
                        'are needed for panel comparison to be meaningful.')
    if len(judge_cols) > MAX_JUDGES:
        raise UserError(f'"{name}" has {len(judge_cols)} judge columns; the maximum supported is {MAX_JUDGES}.')

    df = df.dropna(subset=["competitor_id"])
    if len(df) < MIN_COMPETITORS:
        raise UserError(f'"{name}" has {len(df)} competitors; at least {MIN_COMPETITORS} are needed '
                        'for the statistics to mean anything.')
    if len(df) > MAX_COMPETITORS:
        raise UserError(f'"{name}" has {len(df)} competitors; the maximum supported is {MAX_COMPETITORS}.')

    for c in judge_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            n_bad = int(df[c].isna().sum())
            raise UserError(f'"{name}": column "{c}" has {n_bad} missing or non-numeric score(s). '
                            'Every judge needs a numeric score for every competitor.')
        if df[c].std() == 0:
            raise UserError(f'"{name}": judge "{c}" gave every competitor the identical score, '
                            'which makes comparison impossible. Please check this column.')

    if df["competitor_id"].duplicated().any():
        raise UserError(f'"{name}" has duplicate competitor_id values. Each competitor needs a unique ID.')

    return df.reset_index(drop=True), judge_cols


# ===========================================================================
# HTML REPORT TEMPLATES (unchanged visual system from v1)
# ===========================================================================
REPORT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Instrument+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{--paper:#EFF1EC;--paper-raised:#F8F9F6;--ink:#1E2A3A;--ink-soft:#55606F;--brass:#B08D3E;--line:#D8D9D2;--clear:#3F7A5B;--clear-bg:#E4EFE8;--caution:#C1792E;--caution-bg:#F5E8D8;--flag:#A23B34;--flag-bg:#F3E1DF;--dot-off:#D3D5CC;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:'Instrument Sans',sans-serif;padding:28px 16px 64px}
.sheet{max-width:640px;margin:0 auto}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--brass);margin:0 0 4px}
h1{font-family:'Fraunces',serif;font-weight:600;font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.subtitle{color:var(--ink-soft);font-size:13px;margin:0 0 24px;border-bottom:1px solid var(--line);padding-bottom:20px}
.hero{display:flex;gap:20px;align-items:flex-start;background:var(--paper-raised);border:1px solid var(--line);border-radius:10px;padding:24px;margin-bottom:24px}
.stamp{flex:0 0 auto;width:104px;height:104px;border-radius:50%;border:3px solid currentColor;display:flex;align-items:center;justify-content:center;text-align:center;transform:rotate(-7deg);font-family:'Fraunces',serif;font-weight:700;font-size:15px;letter-spacing:.03em;text-transform:uppercase;line-height:1.15;padding:6px;position:relative}
.stamp::before{content:'';position:absolute;inset:6px;border:1px solid currentColor;border-radius:50%;opacity:.55}
.status-clear .stamp{color:var(--clear)}.status-caution .stamp{color:var(--caution)}.status-flag .stamp{color:var(--flag)}
.hero-text{flex:1}
.hero-text .headline{font-family:'Fraunces',serif;font-weight:600;font-size:18px;line-height:1.35;margin:0 0 10px}
.rarity-caption{font-size:12.5px;color:var(--ink-soft);margin:8px 0 0;line-height:1.5}
.rarity-caption b{color:var(--ink)}
.dot-grid{display:grid;grid-template-columns:repeat(40,1fr);gap:2px}
.dot-grid span{aspect-ratio:1;border-radius:1px;background:var(--dot-off)}
.status-clear .dot-grid span.lit{background:var(--clear)}.status-caution .dot-grid span.lit{background:var(--caution)}.status-flag .dot-grid span.lit{background:var(--flag)}
section.block{margin-bottom:24px}
h2{font-family:'IBM Plex Mono',monospace;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft);border-bottom:1px solid var(--line);padding-bottom:8px;margin:0 0 14px}
.judge-row{display:grid;grid-template-columns:140px 1fr 46px;align-items:center;gap:10px;margin-bottom:10px;font-size:13.5px}
.judge-name{font-weight:500}.judge-name.flagged{color:var(--flag);font-weight:600}
.bar-track{background:var(--line);border-radius:5px;height:10px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;background:var(--ink-soft)}.bar-fill.flagged{background:var(--flag)}
.judge-pct{text-align:right;color:var(--ink-soft);font-size:12.5px;font-family:'IBM Plex Mono',monospace}
.team-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--line);font-size:13.5px}
.team-row:last-child{border-bottom:none}
.team-name{font-weight:500}.team-note{color:var(--ink-soft);font-size:12px}
.pill{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em}
.pill.favored-flag{background:var(--clear-bg);color:var(--clear)}.pill.punished-flag{background:var(--flag-bg);color:var(--flag)}.pill.quiet{background:transparent;color:var(--ink-soft);font-weight:500}
details{background:var(--paper-raised);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
summary{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-soft);cursor:pointer}
.tech-table{margin-top:12px;font-family:'IBM Plex Mono',monospace;font-size:12px;width:100%;border-collapse:collapse}
.tech-table td{padding:4px 0;color:var(--ink-soft)}.tech-table td:last-child{text-align:right;color:var(--ink)}
footer{margin-top:28px;font-size:11.5px;color:var(--ink-soft);line-height:1.6}
.watch-card{background:var(--paper-raised);border:1px solid var(--line);border-left:4px solid var(--ink-soft);border-radius:8px;padding:16px 18px;margin-bottom:12px}
.watch-card.favored{border-left-color:var(--clear)}.watch-card.punished{border-left-color:var(--flag)}
.watch-card-top{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:6px}
.watch-who{font-family:'Fraunces',serif;font-weight:600;font-size:16px}
.watch-meta{font-size:12px;color:var(--ink-soft);margin-top:2px}
.watch-detail{font-size:13.5px;line-height:1.5;margin:8px 0 0}.watch-detail b{color:var(--ink)}
.clear-banner{background:var(--clear-bg);border:1px solid var(--clear);border-radius:8px;padding:18px 20px;color:var(--clear);font-family:'Fraunces',serif;font-weight:600;font-size:15px;margin-bottom:20px}
.clear-banner span{display:block;font-family:'Instrument Sans',sans-serif;font-weight:400;color:var(--ink);font-size:13px;margin-top:6px}
.round-divider{margin:36px 0 20px;padding-top:20px;border-top:2px solid var(--ink)}
.round-divider h1{font-size:20px;margin-bottom:2px}
.warn-box{background:var(--caution-bg);border:1px solid var(--caution);border-radius:8px;padding:12px 14px;font-size:12.5px;color:var(--ink);margin-bottom:20px;line-height:1.5}
.warn-box b{color:var(--caution)}
"""

DISCLAIMER = ("This is a statistical screening tool, not proof of misconduct. It measures how "
              "unusual a scoring pattern is compared to chance; it cannot know why a pattern "
              "exists. Results should prompt review by the appropriate governing body, never "
              "punishment on their own.")


def _analyze_one_round(df, judge_cols, round_label, is_multi_round):
    """Returns the HTML fragment for one judge panel (shared by single-round
    and multi-round-from-PDF reports)."""
    matrix = df[judge_cols].to_numpy(float)
    observed_min_z, avg_agreement, z_scores, p, null_min = calibrated_judge_test(matrix)
    order = np.argsort(z_scores)
    target_judge = judge_cols[order[0]]

    if p >= 0.10:
        status_class, status_word = "status-clear", "Clear"
        headline = (f"All {len(judge_cols)} judges' rankings line up about as well as you'd "
                    "expect from ordinary differences in opinion.")
    elif p >= 0.05:
        status_class, status_word = "status-caution", "Second Look"
        headline = (f"{target_judge} stood out from the rest of the panel more than most judges do. "
                    "Not proof of a problem, but worth watching.")
    else:
        status_class, status_word = "status-flag", "Flagged"
        headline = (f"{target_judge}'s scores broke from the panel by an amount that's rare for "
                    "honest differences of opinion.")

    extreme = int(np.sum(null_min <= observed_min_z))
    n_dots = len(null_min)
    lit = set(np.argsort(null_min)[:extreme].tolist())
    dots = "".join(f'<span class="{"lit" if i in lit else ""}"></span>' for i in range(n_dots))

    lo = avg_agreement.min() - 0.08
    span = max(avg_agreement.max() + 0.08 - lo, 0.01)
    judge_rows = []
    for i in order:
        pct = max(0, min(100, (avg_agreement[i] - lo) / span * 100))
        is_flagged = (judge_cols[i] == target_judge) and (p < 0.10)
        judge_rows.append(
            f'<div class="judge-row"><div class="judge-name{" flagged" if is_flagged else ""}">{judge_cols[i]}</div>'
            f'<div class="bar-track"><div class="bar-fill{" flagged" if is_flagged else ""}" style="width:{pct:.0f}%"></div></div>'
            f'<div class="judge-pct">{avg_agreement[i]:.2f}</div></div>')

    team_html = ""
    if "team" in df.columns and df["team"].notna().all() and df["team"].nunique() >= 2:
        normalized = normalize_scores(df, judge_cols)
        team_results = team_bias_check(df, normalized, judge_cols, target_judge)
        n_teams = len(team_results)
        team_rows = []
        for r in team_results:
            if r["p"] < 0.05:
                cls = "pill favored-flag" if r["direction"] == "favored" else "pill punished-flag"
                text = f'{r["direction"]} &middot; {r["p"]*100:.1f}% odds by chance'
            else:
                cls, text = "pill quiet", "no clear pattern"
            team_rows.append(f'<div class="team-row"><div><div class="team-name">{r["team"]}</div>'
                             f'<div class="team-note">{r["n"]} competitors</div></div>'
                             f'<div class="{cls}">{text}</div></div>')
        caution = ""
        if n_teams >= 4:
            caution = (f'<p class="team-note" style="margin-top:10px;">Note: {n_teams} teams were each '
                       'checked individually, so expect the occasional team to show a small-odds pattern '
                       'by luck alone. Treat a single team flag as a hint.</p>')
        team_html = (f'<section class="block"><h2>Team Patterns &mdash; {target_judge}</h2>'
                     f'{"".join(team_rows)}{caution}</section>')

    tech_rows = "".join(
        f'<tr><td>{judge_cols[i]}</td><td>agreement {avg_agreement[i]:.3f} &middot; z = {z_scores[i]:.2f}</td></tr>'
        for i in order)

    heading = f'<div class="round-divider"><h1>{round_label}</h1><p class="subtitle" style="border:none;padding:0;margin:0;">{len(judge_cols)} judges &middot; {len(df)} competitors</p></div>' if is_multi_round else ""

    return f"""{heading}
<div class="hero {status_class}"><div class="stamp">{status_word}</div><div class="hero-text">
<p class="headline">{headline}</p>
<div class="dot-grid">{dots}</div>
<p class="rarity-caption">Each square is one random reshuffle of who-scored-who.
<b>{extreme} out of {n_dots}</b> reshuffles looked at least this uneven &mdash; that's the honest
odds of seeing this by pure chance.</p></div></div>
<section class="block"><h2>How Closely Each Judge Matched the Panel</h2>{"".join(judge_rows)}</section>
{team_html}
<details><summary>For the statistically curious &mdash; {round_label}</summary><table class="tech-table">
<tr><td colspan="2" style="padding-top:0;">Permutation-calibrated significance ({N_PERMUTATIONS} shuffles)</td></tr>
<tr><td>most unusual judge</td><td>{target_judge}</td></tr>
<tr><td>p-value</td><td>{p:.3f}</td></tr>
<tr><td colspan="2" style="padding-top:10px;">Per-judge panel agreement (Spearman)</td></tr>
{tech_rows}</table></details>"""


def render_single_event_report(rounds, event_name, extra_warnings=None):
    """rounds: list of (label, df, judge_cols). One round = classic single
    report; multiple rounds (from a PDF with several judging stages) stack
    as sections in one page."""
    is_multi = len(rounds) > 1
    body = "".join(_analyze_one_round(df, jc, label, is_multi) for label, df, jc in rounds)
    warn_html = ""
    if extra_warnings:
        items = "".join(f"<li>{w}</li>" for w in extra_warnings)
        warn_html = f'<div class="warn-box"><b>From PDF extraction:</b><ul style="margin:6px 0 0;padding-left:18px;">{items}</ul></div>'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{event_name} &mdash; Judge Panel Review</title><style>{REPORT_CSS}</style></head>
<body><div class="sheet">
<p class="eyebrow">Judge Panel Review</p><h1>{event_name}</h1>
{warn_html}
{body}
<footer>{DISCLAIMER} With a small number of judges or competitors, results are a hint, not proof
&mdash; the same check repeated across multiple events is far more reliable than any single-event flag.</footer>
</div></body></html>"""


def _count_distinct_events(rounds):
    """Rounds are labeled '{source file/competition} \u2014 {round label}' when
    they come from a multi-file batch (or just the plain label for a
    single-file upload). Counts distinct COMPETITIONS, not distinct
    judging panels/rounds -- those are different numbers and the report
    text needs to say the right one."""
    sources = set()
    for label, _, _ in rounds:
        source = label.split(" \u2014 ", 1)[0] if " \u2014 " in label else label
        sources.add(source)
    return len(sources)


def render_watch_report(events, title):
    results, n_tested, match_counts = competitor_watch(events)
    n_events = _count_distinct_events(events)
    n_rounds = len(events)
    flagged = [r for r in results if r.get("flagged")]
    mode_note = f"Checked all {n_tested} competitor/judge pairs with at least 3 shared rounds"

    id_note = ""
    n_id_fallback = match_counts.get("competitor_id", 0)
    if n_id_fallback > 0:
        id_note = (f'<p class="team-note" style="margin-top:4px;">{n_id_fallback} competitor(s) had no name '
                   'in the data and were matched across events by competitor number instead, which is less '
                   'reliable -- competition numbers are usually reassigned at each event, so this can '
                   'occasionally merge two different people or miss a real match. Any flag built on that '
                   'fallback is marked below.</p>')

    if not flagged:
        competitor_body = (f'<div class="clear-banner">No consistent pattern found.'
                f'<span>{mode_note}, across {n_events} events. A single unusual event for one '
                'competitor isn\'t enough to flag on its own.</span></div>' + id_note)
    else:
        cards = []
        for r in flagged:
            word = "Favored" if r["direction"] == "favored" else "Punished"
            verb = ("scored this competitor higher" if r["direction"] == "favored"
                    else "scored this competitor lower")
            method_note = ("" if r["match_method"] == "name" else
                           ' <span style="color:var(--caution);">(matched by competition number only -- '
                           'no name on file, confirm this is really one person)</span>')
            cards.append(
                f'<div class="watch-card {r["direction"]}"><div class="watch-card-top"><div>'
                f'<div class="watch-who">{r["display"]} &middot; {r["judge"]}</div>'
                f'<div class="watch-meta">{("Team " + r["team"] + " &middot; ") if r["team"] else ""}{r["events"]} shared rounds</div></div>'
                f'<div class="pill {"favored-flag" if r["direction"]=="favored" else "punished-flag"}">{word}</div></div>'
                f'<p class="watch-detail">Across their shared rounds, <b>{r["judge"]}</b> consistently {verb} '
                f'than the rest of the panel did. Estimated chance this is a false alarm: <b>{r["q"]*100:.1f}%</b>.'
                f'{method_note}</p></div>')
        competitor_body = (f'<p class="subtitle" style="border-bottom:none;padding-bottom:0;">{mode_note}, '
                f'across {n_events} events.</p>{id_note}{"".join(cards)}')

    # --- Team-level check: pooled across every uploaded event, catches the case
    # where individual dancers don't repeat (different divisions, no shared people)
    # but the same judge sees the same SCHOOL's dancers across several divisions ---
    team_results, n_team_tested = team_watch(events)
    team_flagged = [r for r in team_results if r.get("flagged")]
    team_mode_note = f"Checked all {n_team_tested} school/judge pairs with at least 5 scored competitors"

    if n_team_tested == 0:
        team_body = ""  # no team data available (e.g. CSVs without a team column) -- section omitted
    elif not team_flagged:
        team_body = (f'<div class="clear-banner">No consistent pattern found.'
                f'<span>{team_mode_note}, across {n_events} events.</span></div>')
    else:
        cards = []
        for r in team_flagged:
            word = "Favored" if r["direction"] == "favored" else "Punished"
            verb = ("scored this school's competitors higher" if r["direction"] == "favored"
                    else "scored this school's competitors lower")
            cards.append(
                f'<div class="watch-card {r["direction"]}"><div class="watch-card-top"><div>'
                f'<div class="watch-who">{r["team"]} &middot; {r["judge"]}</div>'
                f'<div class="watch-meta">{r["n_competitors"]} competitors across {r["n_rounds"]} rounds</div></div>'
                f'<div class="pill {"favored-flag" if r["direction"]=="favored" else "punished-flag"}">{word}</div></div>'
                f'<p class="watch-detail">Across every competitor from this school that <b>{r["judge"]}</b> scored, '
                f'they consistently {verb} than the rest of the panel did. '
                f'Estimated chance this is a false alarm: <b>{r["q"]*100:.1f}%</b>.</p></div>')
        team_body = (f'<p class="subtitle" style="border-bottom:none;padding-bottom:0;">{team_mode_note}, '
                f'across {n_events} events.</p>{"".join(cards)}')

    team_section = ""
    if team_body:
        team_section = f"""
  <section class="block" style="margin-top:32px;">
    <h2>School Patterns Across Events</h2>
    {team_body}
  </section>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title><style>{REPORT_CSS}</style></head>
<body><div class="sheet">
<p class="eyebrow">Multi-Event Pattern Check</p><h1>{title}</h1>
<section class="block">
  <h2>Competitor Patterns Across Events</h2>
  {competitor_body}
</section>
{team_section}
<footer>{DISCLAIMER} This tracks the same person across separate competitions by name where possible
(competition numbers are reassigned each event, so they aren't used as the primary match). The school check
pools every competitor from that school a judge scored across all uploaded events, which works even when the
events are different divisions of one competition with no individual dancers in common. Results are corrected
for the number of pairs checked, so a flag here is meaningfully rarer than a single-event flag.</footer>
</div></body></html>"""


# ===========================================================================
# REQUEST HANDLING
# ===========================================================================
def _rounds_to_preview_json(rounds):
    """Converts (label, df, judge_cols) tuples into a JSON-friendly shape
    for the browser to render as an editable table."""
    out = []
    for label, df, judge_cols in rounds:
        rows = []
        for _, row in df.iterrows():
            marks = {j: (None if pd.isna(row[j]) else float(row[j])) for j in judge_cols}
            rows.append({"competitor_id": int(row["competitor_id"]),
                        "name": str(row["name"]) if "name" in df.columns and pd.notna(row["name"]) else "",
                        "team": str(row["team"]) if "team" in df.columns else "",
                        "marks": marks})
        out.append({"label": label, "judges": judge_cols, "rows": rows})
    return out


def _preview_json_to_rounds(confirmed_rounds):
    """Reverses the above: turns the browser's (possibly hand-edited) JSON
    back into (label, df, judge_cols) tuples for analysis."""
    rounds = []
    for r in confirmed_rounds:
        label = r.get("label", "Round")
        judges = r.get("judges", [])
        if len(judges) < MIN_JUDGES:
            raise UserError(f'"{label}" has fewer than {MIN_JUDGES} judges after editing; cannot analyze.')
        records = []
        for row in r.get("rows", []):
            rec = {"competitor_id": row.get("competitor_id"), "name": row.get("name", ""),
                  "team": row.get("team", "")}
            marks = row.get("marks", {})
            ok = True
            for j in judges:
                v = marks.get(j)
                if v is None or v == "":
                    ok = False
                    break
                try:
                    rec[j] = float(v)
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                records.append(rec)
        df = pd.DataFrame(records)
        if len(df) < MIN_COMPETITORS:
            raise UserError(f'"{label}" has only {len(df)} competitors with complete marks after '
                            f'editing; at least {MIN_COMPETITORS} are needed.')
        for j in judges:
            if df[j].std() == 0:
                raise UserError(f'"{label}": judge "{j}" has identical scores for everyone after '
                                'editing, which makes comparison impossible.')
        rounds.append((label, df, judges))
    if not rounds:
        raise UserError("No rounds to analyze.")
    return rounds


def _group_rounds_by_source(rounds):
    """Splits a confirmed_rounds list back into per-source-file groups for
    ingestion -- one database 'event' per uploaded PDF, not per round. Relies
    on the labeling convention used everywhere else in this file: a
    multi-file batch labels each round '{source filename} \u2014 {round label}',
    so the prefix before the dash identifies which file it came from. A
    single-PDF (non-batch) preview's rounds have no such prefix; those are
    treated as one group under the whole title."""
    groups = {}
    for label, df, judge_cols in rounds:
        if " \u2014 " in label:
            source, round_part = label.split(" \u2014 ", 1)
        else:
            source, round_part = label, label
        groups.setdefault(source, []).append((round_part, df, judge_cols))
    return groups


def _content_hash_for_rounds(rounds):
    """A stable fingerprint of one event's confirmed data, used to prevent
    ingesting the same competition twice. Hashing the actual score data
    (not the original PDF bytes, which aren't available at this stage --
    only the reviewed/possibly-edited rounds are) also has a nice side
    effect: two different PDF exports of the identical result set still
    hash the same, so a re-export under a different filename is still
    caught as a duplicate."""
    import hashlib
    parts = []
    for label, df, judge_cols in sorted(rounds, key=lambda r: r[0]):
        parts.append(label)
        for j in judge_cols:
            parts.append(j)
        for _, row in df.sort_values("competitor_id").iterrows():
            parts.append(str(row["competitor_id"]))
            for j in judge_cols:
                parts.append(f"{row[j]:.3f}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


_BENIGN_WARNING_MARKERS = (
    # These are informational notices about known, correctly-handled format
    # quirks (see quickfeis_parser.py / feisresults_parser.py) -- not signs
    # of an actual extraction problem. Counting them against a file would
    # route perfectly good data to manual review for no real reason, which
    # defeats the point of an automated gate.
    "excluded from the bias-detection statistics",
    "did not dance",
    "likely withdrew",
)


def bulk_sanity_checks(rounds, stated_count, parse_warnings):
    """
    The automated QA gate for bulk import, standing in for the human
    preview-and-confirm step that's impractical at hundreds of files.
    Returns a list of human-readable issue strings; an empty list means
    the file passes every check and is safe to auto-save without a human
    looking at it first. Any single issue routes the file to manual review
    instead -- deliberately conservative, since a bad row silently entering
    the permanent archive is a much worse failure than a clean file
    waiting an extra minute for a human glance.
    """
    issues = []
    real_warnings = [w for w in parse_warnings
                     if not any(marker in w for marker in _BENIGN_WARNING_MARKERS)]
    if real_warnings:
        issues.append(f"{len(real_warnings)} extraction warning(s) during parsing "
                      f"(e.g. \"{real_warnings[0]}\")")
    if not rounds:
        issues.append("No usable rounds were extracted from this file.")
        return issues  # nothing else to meaningfully check

    largest_label, largest_df, _ = max(rounds, key=lambda r: len(r[1]))
    largest_count = len(largest_df)

    if stated_count is not None and abs(largest_count - stated_count) > 2:
        issues.append(f"Competitor count mismatch: the PDF states {stated_count} competed, "
                      f"but {largest_count} were extracted (in \"{largest_label}\").")
    if largest_count < 3:
        issues.append(f"Only {largest_count} competitors were extracted -- too few to check reliably.")

    for label, df, judge_cols in rounds:
        for j in judge_cols:
            vals = df[j]
            if (vals < 0).any() or (vals > 100).any():
                issues.append(f"\"{label}\": judge \"{j}\" has a mark outside the expected 0-100 range.")
            if vals.std() == 0:
                issues.append(f"\"{label}\": judge \"{j}\" gave every competitor an identical score "
                              "(likely an extraction problem, not a real judging pattern).")

    return issues


def render_history_report(rounds, subject_kind, subject_name):
    """Renders a profile-style report for 'everything the archive knows
    about this judge/team/competitor' -- runs the same competitor_watch and
    team_watch used for session-based multi-event uploads, on whatever
    history the database query returned, then filters the results down to
    the one subject that was actually searched for (a judge query still
    pulls in the other judges who share those rounds, since that's needed
    to compute the panel comparison -- but only the target judge's own
    results are shown)."""
    n_events = _count_distinct_events(rounds)
    n_rounds = len(rounds)

    team_results, n_team_tested = team_watch(rounds)
    if subject_kind == "team":
        team_results = [r for r in team_results if r["team"] == subject_name]
    elif subject_kind == "judge":
        team_results = [r for r in team_results if r["judge"] == subject_name]
    else:
        team_results = []  # a single competitor's own team pattern isn't the question being asked here
    team_flagged = [r for r in team_results if r.get("flagged")]

    comp_results, n_comp_tested, match_counts = competitor_watch(rounds)
    if subject_kind == "judge":
        comp_results = [r for r in comp_results if r["judge"] == subject_name]
    elif subject_kind == "competitor":
        comp_results = [r for r in comp_results if r["display"] == subject_name]
    else:
        comp_results = []
    comp_flagged = [r for r in comp_results if r.get("flagged")]

    subject_label = {"judge": "Judge", "team": "School", "competitor": "Competitor"}[subject_kind]
    cards = []
    for r in comp_flagged:
        word = "Favored" if r["direction"] == "favored" else "Punished"
        verb = "scored higher" if r["direction"] == "favored" else "scored lower"
        cards.append(
            f'<div class="watch-card {r["direction"]}"><div class="watch-card-top"><div>'
            f'<div class="watch-who">{r["display"]} &middot; {r["judge"]}</div>'
            f'<div class="watch-meta">{r["events"]} shared rounds</div></div>'
            f'<div class="pill {"favored-flag" if r["direction"]=="favored" else "punished-flag"}">{word}</div></div>'
            f'<p class="watch-detail">Consistently {verb} than the rest of the panel. '
            f'Estimated chance this is a false alarm: <b>{r["q"]*100:.1f}%</b>.</p></div>')
    for r in team_flagged:
        word = "Favored" if r["direction"] == "favored" else "Punished"
        verb = "scored higher" if r["direction"] == "favored" else "scored lower"
        cards.append(
            f'<div class="watch-card {r["direction"]}"><div class="watch-card-top"><div>'
            f'<div class="watch-who">{r["team"]} &middot; {r["judge"]}</div>'
            f'<div class="watch-meta">{r["n_competitors"]} competitors across {r["n_rounds"]} rounds</div></div>'
            f'<div class="pill {"favored-flag" if r["direction"]=="favored" else "punished-flag"}">{word}</div></div>'
            f'<p class="watch-detail">Consistently {verb} than the rest of the panel for this school\'s competitors. '
            f'Estimated chance this is a false alarm: <b>{r["q"]*100:.1f}%</b>.</p></div>')

    if not cards:
        body = (f'<div class="clear-banner">No consistent pattern found.'
               f'<span>Checked {n_rounds} judging panels across {n_events} events in the archive '
               f'involving this {subject_kind}.</span></div>')
    else:
        body = (f'<p class="subtitle" style="border-bottom:none;padding-bottom:0;">Checked {n_rounds} judging '
               f'panels across {n_events} events in the archive.</p>{"".join(cards)}')

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{subject_label}: {subject_name}</title><style>{REPORT_CSS}</style></head>
<body><div class="sheet">
<p class="eyebrow">Archive History Lookup</p><h1>{subject_name}</h1>
<p class="subtitle">{subject_label} &middot; {n_events} events on record</p>
{body}
<footer>{DISCLAIMER} This draws on every event in the archive involving this {subject_kind}, not just
one session's uploads, which is why it can have more statistical power than a one-off check.</footer>
</div></body></html>"""


def analyze_payload(payload):
    action = payload.get("action")

    # --- Database actions: ingesting into or querying the permanent archive.
    # Imported lazily so the existing session-based analysis (everything
    # above) keeps working even before a database is provisioned. ---
    if action in ("ingest", "bulk_ingest", "check_duplicate", "query_judge", "query_team",
                 "query_competitor", "list_events", "search_names"):
        from . import db

        if action == "check_duplicate":
            existing = db.already_ingested(payload["content_hash"])
            return {"duplicate": existing is not None, "existing": existing}

        if action == "ingest":
            rounds = _preview_json_to_rounds(payload["confirmed_rounds"])
            groups = _group_rounds_by_source(rounds)
            saved = []
            for source_name, group_rounds in groups.items():
                content_hash = _content_hash_for_rounds(group_rounds)
                existing = db.already_ingested(content_hash)
                if existing:
                    saved.append({"source": source_name, "status": "duplicate", "event_id": existing["id"]})
                    continue
                event_id = db.ingest_confirmed_rounds(
                    group_rounds, source_name, payload.get("title", source_name),
                    payload.get("format", "unknown"), payload.get("warnings", []), content_hash,
                )
                saved.append({"source": source_name, "status": "saved", "event_id": event_id})
            return {"saved": saved}

        if action == "bulk_ingest":
            from .pdf_parsers import detect_format, extract_stated_competitor_count

            files = payload.get("files", [])
            if not files:
                raise UserError("No files were received.")
            if len(files) > MAX_FILES:
                raise UserError(f"Send at most {MAX_FILES} files per bulk-import request "
                                "(the frontend should be chunking automatically).")

            results = []
            for idx, f in enumerate(files):
                name = str(f.get("name", f"file_{idx}"))
                entry = {"filename": name}
                try:
                    raw = base64.b64decode(f["content"])
                    tmp_path = f"/tmp/bulk_{idx}.pdf"
                    with open(tmp_path, "wb") as out:
                        out.write(raw)

                    fmt_key = detect_format(tmp_path)
                    if fmt_key is None:
                        entry["status"] = "needs_review"
                        entry["reasons"] = ["Format not recognized (not QuickFeis or feisresults.com)."]
                        results.append(entry)
                        continue

                    rounds, fmt_label, parse_warnings = parse_results_pdf(tmp_path)
                    stated_count = extract_stated_competitor_count(tmp_path, fmt_key)
                    issues = bulk_sanity_checks(rounds, stated_count, parse_warnings)

                    if issues:
                        entry["status"] = "needs_review"
                        entry["reasons"] = issues
                        results.append(entry)
                        continue

                    title = name.rsplit(".", 1)[0]
                    content_hash = _content_hash_for_rounds(rounds)
                    existing = db.already_ingested(content_hash)
                    if existing:
                        entry["status"] = "duplicate"
                        entry["existing_event_id"] = existing["id"]
                    else:
                        event_id = db.ingest_confirmed_rounds(
                            rounds, title, title, fmt_label, parse_warnings, content_hash)
                        entry["status"] = "saved"
                        entry["event_id"] = event_id
                    results.append(entry)
                except UnrecognizedFormatError as e:
                    entry["status"] = "needs_review"
                    entry["reasons"] = [str(e)]
                    results.append(entry)
                except Exception as e:
                    entry["status"] = "error"
                    entry["reasons"] = [f"Unexpected error: {e}"]
                    results.append(entry)

            return {"results": results}

        if action == "list_events":
            return {"events": db.list_events()}

        if action == "search_names":
            return {"matches": db.search_names(payload.get("prefix", ""), payload.get("kind", "competitor_name"))}

        # query_judge / query_team / query_competitor
        kind_map = {
            "query_judge": ("judge", db.fetch_rounds_for_judge),
            "query_team": ("team", db.fetch_rounds_for_team),
            "query_competitor": ("competitor", db.fetch_rounds_for_competitor),
        }
        subject_kind, fetch_fn = kind_map[action]
        subject_name = payload.get("name", "").strip()
        if not subject_name:
            raise UserError("Please provide a name to search for.")
        rounds = fetch_fn(subject_name)
        if not rounds:
            raise UserError(f'No history found for "{subject_name}". Check the spelling, or it may not be in the archive yet.')
        html = render_history_report(rounds, subject_kind, subject_name)
        return {"html": html, "mode": "history_query"}

    # --- Path 3: confirmed data from the PDF preview step ---
    if "confirmed_rounds" in payload:
        rounds = _preview_json_to_rounds(payload["confirmed_rounds"])
        title = payload.get("title") or "Judge Panel Review"
        if payload.get("mode") == "watch_list":
            # Rounds may come from many different PDFs (many competitions) --
            # pool everything and look for judges who consistently favor or
            # punish the same person across separate events.
            html = render_watch_report(rounds, title)
            return {"html": html, "mode": "watch_list"}
        else:
            # One PDF's rounds (same competition, different panels per round) --
            # stack as sections in one page, no cross-round person-tracking.
            html = render_single_event_report(rounds, title)
            return {"html": html, "mode": "single_event"}

    files = payload.get("files", [])
    if not files:
        raise UserError("No files were received. Please choose at least one score sheet.")
    if len(files) > MAX_FILES:
        raise UserError(f"Please upload at most {MAX_FILES} files at a time.")

    pdf_files = [f for f in files if str(f.get("name", "")).lower().endswith(".pdf")]
    other_files = [f for f in files if not str(f.get("name", "")).lower().endswith(".pdf")]

    if pdf_files:
        if other_files:
            raise UserError("Please upload PDFs separately from CSV/Excel files.")

        all_rounds = []       # [(label, df, judge_cols), ...] across every PDF
        all_warnings = []
        formats_seen = set()
        for f in pdf_files:
            raw = base64.b64decode(f["content"])
            tmp_path = f"/tmp/upload_{len(all_rounds)}.pdf"
            with open(tmp_path, "wb") as out:
                out.write(raw)
            source_name = f["name"].rsplit(".", 1)[0]
            try:
                rounds, fmt, warnings = parse_results_pdf(tmp_path)
            except UnrecognizedFormatError as e:
                raise UserError(f'"{f["name"]}": {e}')
            if not rounds:
                raise UserError(f'"{f["name"]}": couldn\'t extract any usable judge panels. ' +
                                (" ".join(warnings) if warnings else ""))
            formats_seen.add(fmt)
            for label, df, judge_cols in rounds:
                # tag each round with its source file so the preview can group
                # them and the report can still say which competition is which
                all_rounds.append((f"{source_name} \u2014 {label}", df, judge_cols))
            all_warnings.extend(f'"{f["name"]}": {w}' for w in warnings)

        is_batch = len(pdf_files) > 1
        return {
            "preview": True,
            "format": " + ".join(sorted(formats_seen)),
            "title": (f"Competitor Watch List \u2014 {len(pdf_files)} competitions" if is_batch
                     else pdf_files[0]["name"].rsplit(".", 1)[0]),
            "mode": "watch_list" if is_batch else "single_event",
            "rounds": _rounds_to_preview_json(all_rounds),
            "warnings": all_warnings,
        }

    # --- Path 1: CSV/XLSX, unchanged from v1 ---
    parsed = []
    for f in other_files:
        name = str(f.get("name", "file"))
        df, judge_cols = parse_uploaded_file(name, f.get("content", ""))
        label = name.rsplit(".", 1)[0]
        parsed.append((label, df, judge_cols))

    if len(parsed) == 1:
        label, df, judge_cols = parsed[0]
        html = render_single_event_report([(label, df, judge_cols)], label)
        return {"html": html, "mode": "single_event"}
    else:
        title = payload.get("title") or "Competitor Watch List"
        html = render_watch_report(parsed, title)
        return {"html": html, "mode": "watch_list"}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # This function currently receives ALL traffic to the site, not just
        # /api/analyze -- so it needs to serve the homepage itself for GET
        # requests, not just handle the POST analysis calls.
        try:
            index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
            with open(index_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = analyze_payload(payload)
            self._send(200, result)
        except UserError as e:
            self._send(400, {"error": str(e)})
        except Exception:
            self._send(500, {"error": "Something went wrong analyzing this file. "
                                       "Please check it against the template and try again."})

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
