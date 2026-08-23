"""
Parser for the "Mid Atlantic Region Oireachtas" template -- a third
distinct layout from the feisresults.com platform (see
feisresults_parser.py's docstring for the other one; both share the
"feisresults.com" copyright footer but are structurally unrelated, so
detection has to key off something more specific than that branding).

STRUCTURE, confirmed against real files before relying on any of it:
  - One PDF per competition (like QuickFeis), not one PDF per whole event.
  - Page 0: a title/cover page listing which 5 judges sat on each of up to
    3 rounds, in plain upright text -- easy to read directly.
  - Page 1+: the results table. Each judge's name is printed as a column
    header using ROTATED (vertical) text. pdfplumber extracts rotated text
    as character-REVERSED words read bottom-to-top rather than top-to-
    bottom -- both quirks verified directly against real coordinates
    before being relied on here; get the reversal or the reading order
    wrong and every judge's name would be silently scrambled.
  - Each competitor's mark from each judge is a single "SCORE/IP" token
    (e.g. "94/100"). Only the SCORE half is used as that judge's raw mark,
    consistent with every other parser in this project preferring raw
    scores over derived points.
  - Not every competitor reaches every round. A competitor not recalled to
    a later round simply has NO data at all in that round's columns for
    their row (blank, not a marker). A competitor who was recalled but
    didn't complete a round they were expected to (e.g. "FTC" status)
    shows a literal "0/0" instead. Both mean "no real score" and are
    treated identically: missing, not a genuine zero.
"""

import re

import pdfplumber
import pandas as pd

from .quickfeis_parser import _dominant_x_cluster

_RANK_PATTERN = re.compile(r"^\d+$")
_SCORE_IP_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$")


def _reconstruct_rotated_word(word):
    return word["text"][::-1]


def _extract_rotated_columns(page):
    """Returns [(x0, reconstructed_label), ...] sorted left to right, for
    every rotated-text column header on the page."""
    words = page.extract_words()
    rotated = [w for w in words if not w.get("upright", True)]
    from collections import defaultdict
    cols = defaultdict(list)
    for w in rotated:
        cols[round(w["x0"], 1)].append(w)
    result = []
    for x0 in sorted(cols.keys()):
        ws = sorted(cols[x0], key=lambda w: -w["top"])  # bottom-to-top reading order
        label = " ".join(_reconstruct_rotated_word(w) for w in ws)
        result.append((x0, label))
    return result


_TOTAL_LABEL_MARKERS = ("round total", "tottal ip", "total ip")


def _classify_rounds(rotated_columns):
    """Splits the ordered column list into per-round judge groups, using
    the "N Round Total" / "TOTAL IP" label columns as separators between
    rounds (and discarding those label columns themselves, and the
    "Tottal IP for Rounds 1 & 2" running-total column, since none of them
    are judge columns). Returns [[(judge_name, x0), ...], ...] per round,
    in round order. A round with zero judges found is dropped."""
    rounds = []
    current = []
    for x0, label in rotated_columns:
        if any(marker in label.lower() for marker in _TOTAL_LABEL_MARKERS):
            if current:
                rounds.append(current)
                current = []
        else:
            current.append((label, x0))
    if current:
        rounds.append(current)
    return rounds


def _nearest_score(words, target_x0, tolerance=15):
    """Finds the SCORE/IP-shaped word nearest a judge's calibrated column
    position. A generous-but-bounded tolerance handles the small, consistent
    offset (~4-5 units) between a judge's narrow rotated name label and the
    wider score token beneath it -- confirmed against real data -- without
    risking a match against the next judge's column over."""
    candidates = [w for w in words if abs(w["x0"] - target_x0) < tolerance
                 and _SCORE_IP_PATTERN.match(w["text"])]
    if not candidates:
        return None
    return min(candidates, key=lambda w: abs(w["x0"] - target_x0))


def parse_maro_pdf(pdf_path):
    """
    Returns (rounds, warnings) -- same (label, DataFrame, judge_columns)
    shape as every other parser in this project.
    """
    warnings = []
    round_rows = {}  # round_idx -> list of row dicts

    with pdfplumber.open(pdf_path) as pdf:
        # Page 0: competition title, used as the round labels' prefix
        cover_text = pdf.pages[0].extract_text() or ""
        m = re.search(r"Results for Competition\s+(\d+)\s*:\s*(.+)", cover_text)
        competition_label = f"Competition {m.group(1)} - {m.group(2).strip()}" if m else "Competition"

        round_judges = None  # calibrated once, reused per page (same template throughout)

        for page_num, page in enumerate(pdf.pages[1:], start=2):
            rotated = _extract_rotated_columns(page)
            if rotated:
                classified = _classify_rounds(rotated)
                if classified:
                    round_judges = classified

            if round_judges is None:
                warnings.append(f"Page {page_num}: couldn't find the rotated judge-column headers; skipped.")
                continue

            words = page.extract_words()
            upright = [w for w in words if w.get("upright", True)]

            candidates = [w for w in upright if 20 <= w["x0"] <= 100 and w["top"] > 100
                         and _RANK_PATTERN.match(w["text"])]
            if not candidates:
                continue
            card_x0 = _dominant_x_cluster(candidates)
            card_words = sorted([w for w in candidates if abs(w["x0"] - card_x0) < 8],
                                key=lambda w: w["top"])

            for cw in card_words:
                anchor_top = cw["top"]
                try:
                    competitor_id = int(cw["text"])
                except ValueError:
                    continue

                main_line = [w for w in upright if abs(w["top"] - anchor_top) < 1.5]
                name_line = [w for w in upright if -6 <= w["top"] - anchor_top < -1.5]
                school_line = [w for w in upright if 1.5 < w["top"] - anchor_top <= 7]

                name_words = [w for w in name_line if 95 <= w["x0"] < 216 and w["text"] != "*"]
                name = " ".join(w["text"] for w in sorted(name_words, key=lambda w: w["x0"]))
                school_words = [w for w in school_line if 95 <= w["x0"] < 216]
                team = " ".join(w["text"] for w in sorted(school_words, key=lambda w: w["x0"]))

                if not name:
                    warnings.append(f"Page {page_num}: couldn't read a name for card {competitor_id}; skipped.")
                    continue

                for round_idx, judges in enumerate(round_judges):
                    marks = {}
                    n_present, n_total = 0, len(judges)
                    for judge_name, judge_x0 in judges:
                        w = _nearest_score(main_line, judge_x0)
                        if w is None:
                            continue  # not recalled to this round -- expected, not an error
                        score_text, ip_text = _SCORE_IP_PATTERN.match(w["text"]).groups()
                        score = float(score_text)
                        if score_text == "0" and ip_text == "0":
                            continue  # literal "0/0" -- did not complete this round, not a real zero
                        marks[judge_name] = score
                        n_present += 1

                    if n_present == 0:
                        continue  # not recalled to this round at all -- expected
                    if 0 < n_present < n_total:
                        warnings.append(f"Card {competitor_id}, round {round_idx+1}: only {n_present} of "
                                        f"{n_total} judges' marks were readable. Check this row manually.")
                        continue
                    round_rows.setdefault(round_idx, []).append(
                        {"competitor_id": competitor_id, "name": name, "team": team, **marks})

    rounds = []
    for round_idx in sorted(round_rows.keys()):
        rows = round_rows[round_idx]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        judge_cols = [c for c in df.columns if c not in ("competitor_id", "name", "team")]
        df = df.dropna(subset=judge_cols)
        if len(df) == 0:
            continue
        rounds.append((f"{competition_label} - Round {round_idx + 1}", df, judge_cols))

    if not rounds:
        warnings.append("No usable rounds were extracted from this file.")

    return rounds, warnings


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "maro_boys_u12.pdf"
    rounds, warnings = parse_maro_pdf(path)
    for label, df, judge_cols in rounds:
        print(f"\n{label}: {len(df)} competitors, judges = {judge_cols}")
        print(df.head(5).to_string())
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(" -", w)
