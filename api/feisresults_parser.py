"""
Parser for feisresults.com-format results PDFs (the other dominant
tabulation software, alongside QuickFeis). Built and tested against a
real World Championship sample.

STRUCTURE OF THIS FORMAT (found by testing against a real file):
- A competition here can have up to THREE analyzable panel stages:
    1. "RECALL" table: the full field (everyone who competed) scored on
       two dances (R1, R2) by a panel of judges. This determines who
       gets recalled to the final round.
    2. "FINAL MARKS" table: only the recalled competitors, scored on a
       third dance ("Set Dance") by the SAME panel. The "Pre-recall
       Score" column here is just the Recall stage's Total carried
       forward -- not a new judgment, so we don't treat it as a fresh
       score.
    3. A single "Solo Judge" ranks a "Round 1" / "Round 2" on a
       placement-only basis (no marks, just 1st/2nd/3rd...). Because
       it's ONE judge, not a panel, there's no other judge to compare
       them against -- our whole method requires a panel. We extract
       this for completeness but it is NOT fed into the bias-detection
       statistics, and the report says so explicitly rather than silently
       dropping it.
- Unlike the QuickFeis format, there's no clean grid the table-detector
  can find automatically -- rows are reconstructed from word positions
  (coordinates), anchored on the Rank/Card number column.
"""

import re
import pdfplumber
import pandas as pd


def _extract_judge_columns(page, name_top_range=(85, 100)):
    """
    Reads the judge-name header row to get judge names and their
    approximate x-centers, then calibrates each judge's 4 metric-column
    x-positions from the FIRST DATA ROW's actual numeric values, rather
    than matching sub-header label text.

    WHY: on one of the two table types in this format, the sub-header
    labels ("Pre-recall Score", "Set Dance") come out of the PDF as
    broken character fragments due to a font-kerning quirk -- matching
    against label text is fragile here. Numbers in the data rows extract
    cleanly regardless, so we use the first row as a ruler instead.
    """
    words = page.extract_words()
    header_words = [w for w in words if name_top_range[0] < w["top"] < name_top_range[1] and w["x0"] > 200]
    header_words.sort(key=lambda w: w["x0"])
    names, current = [], []
    for w in header_words:
        if current and w["x0"] - current[-1]["x1"] > 15:
            names.append((" ".join(x["text"] for x in current), current[0]["x0"], current[-1]["x1"]))
            current = []
        current.append(w)
    if current:
        names.append((" ".join(x["text"] for x in current), current[0]["x0"], current[-1]["x1"]))
    if not names:
        return None

    centers = [(n0 + n1) / 2 for _, n0, n1 in names]
    # band boundaries: midpoints between consecutive judge centers, page edges otherwise
    bounds = [200] + [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)] + [page.width]

    # first data row = topmost row with a small integer at x0 in [38,48] (the rank column)
    rank_words = sorted([w for w in words if 20 <= w["x0"] <= 100 and w["top"] > 115 and
                        w["text"].replace("(T)", "").isdigit()], key=lambda w: w["top"])
    if not rank_words:
        return None
    rank_x0 = rank_words[0]["x0"]
    rank_words = [w for w in rank_words if abs(w["x0"] - rank_x0) < 6]
    if not rank_words:
        return None
    anchor_top = rank_words[0]["top"]
    row_words = [w for w in words if abs(w["top"] - anchor_top) < 3 and w["x0"] > 200]

    judge_columns = []
    for i, (name, _, _) in enumerate(names):
        band_lo, band_hi = bounds[i], bounds[i + 1]
        band_vals = sorted([w for w in row_words if band_lo <= w["x0"] < band_hi], key=lambda w: w["x0"])
        if len(band_vals) < 4:
            return None  # calibration failed for this page -- caller treats as unparseable
        band_vals = band_vals[:4]  # extra trailing values (e.g. a grand-total column after the last judge) ignored
        judge_columns.append({
            "name": name,
            "metric_x": [w["x0"] for w in band_vals],  # [col0, col1, col2, col3] left to right
        })
    return judge_columns


def _dominant_x_cluster(candidates, tolerance=6):
    """Returns the x0 of the LARGEST cluster of candidate words, not simply
    the leftmost one -- robust to an outlier like a tied rank whose wider
    text shifts its bounding box slightly, which could otherwise hijack a
    naive minimum-x0 approach and silently misidentify a whole column."""
    xs = sorted(w["x0"] for w in candidates)
    best_cluster = []
    for x in xs:
        cluster = [y for y in xs if abs(y - x) < tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return sum(best_cluster) / len(best_cluster)


def _nearest(words, x_target, top_target, x_tol=20, top_tol=3):
    for w in words:
        if abs(w["x0"] - x_target) < x_tol and abs(w["top"] - top_target) < top_tol:
            return w["text"]
    return None


def _parse_panel_table(pdf_path, page_indices, table_name):
    """Shared logic for RECALL and FINAL MARKS -- both are 'rank, card,
    name/school, then N judges x (R1, R2, ...)' tables with the same
    row-reconstruction approach, just different column labels."""
    warnings = []
    rows = []
    judge_columns = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in page_indices:
            page = pdf.pages[page_idx]
            jc = _extract_judge_columns(page)
            if jc is None:
                warnings.append(f"{table_name} page {page_idx+1}: couldn't read the judge header row; skipped.")
                continue
            judge_columns = jc  # assume stable across pages (verified identical on sample)

            words = page.extract_words()

            # Rank and competitor-number columns sit at different fixed x-positions in the two
            # table types (an extra "Q" flag column on Final Marks shifts everything), so detect
            # both columns' x-positions dynamically per page rather than assuming a fixed value.
            candidates = [w for w in words if 20 <= w["x0"] <= 100 and w["top"] > 115 and
                         w["text"].replace("(T)", "").isdigit()]
            if not candidates:
                warnings.append(f"{table_name} page {page_idx+1}: no rank column detected; page skipped.")
                continue
            rank_x0 = _dominant_x_cluster(candidates)
            rank_words = sorted([w for w in candidates if abs(w["x0"] - rank_x0) < 6], key=lambda w: w["top"])
            card_candidates = [w for w in candidates if w["x0"] > rank_x0 + 6]
            card_x0 = min((w["x0"] for w in card_candidates), default=None)

            for rw in rank_words:
                anchor_top = rw["top"]
                card_word = None
                if card_x0 is not None:
                    card_word = _nearest(words, card_x0, anchor_top, x_tol=6, top_tol=4)
                if card_word is None or not card_word.isdigit():
                    warnings.append(f"{table_name} page {page_idx+1}: couldn't find a competitor number "
                                    f"near rank {rw['text']} (y={anchor_top:.0f}); row skipped.")
                    continue
                competitor_id = int(card_word)

                name_words = [w for w in words if 75 < w["x0"] < 225 and -6 < w["top"] - anchor_top < -1]
                school_words = [w for w in words if 75 < w["x0"] < 225 and 2 < w["top"] - anchor_top < 9]
                name = " ".join(w["text"] for w in sorted(name_words, key=lambda w: w["x0"]))
                team = " ".join(w["text"] for w in sorted(school_words, key=lambda w: w["x0"]))

                marks = {}
                incomplete = False
                for j in judge_columns:
                    for metric_idx in (0, 1):  # col0, col1 -- meaning depends on table (handled by caller)
                        x_pos = j["metric_x"][metric_idx]
                        text = _nearest(words, x_pos, anchor_top, x_tol=18, top_tol=3)
                        col_name = f"{j['name']} col{metric_idx}"
                        if text is None or text.strip() in ("", "-"):
                            marks[col_name] = None
                            incomplete = True
                        else:
                            try:
                                marks[col_name] = float(text)
                            except ValueError:
                                marks[col_name] = None
                                warnings.append(f"{table_name}: unreadable value '{text}' for competitor "
                                                f"{competitor_id}, {col_name}. Left blank.")

                row = {"competitor_id": competitor_id, "name": name, "team": team}
                row.update(marks)
                if incomplete:
                    # a dancer who didn't dance a round (e.g. withdrew) -- legitimate, not an error;
                    # keep the row but it will naturally drop out of any judge-column that needs it
                    pass
                rows.append(row)

    if not rows:
        warnings.append(f"No rows extracted from {table_name}.")
        return pd.DataFrame(), [], warnings

    df = pd.DataFrame(rows)
    if df["competitor_id"].duplicated().any():
        dupes = df.loc[df["competitor_id"].duplicated(), "competitor_id"].tolist()
        warnings.append(f"{table_name}: duplicate competitor number(s) {dupes}. Check the source PDF.")

    all_judge_cols = [c for c in df.columns if c not in ("competitor_id", "name", "team")]
    return df, all_judge_cols, warnings


def _drop_did_not_dance(df, metric_cols, label, warnings):
    """This format marks 'didn't dance this specific dance' as a literal 0.00
    from every judge, rather than leaving the cell blank (unlike QuickFeis's
    'NR' text). A uniform zero across the whole panel isn't a real scoring
    disagreement -- it's a no-show -- so we detect and drop it explicitly
    rather than silently let a fake data point sit in the analysis."""
    all_zero = (df[metric_cols] == 0).all(axis=1)
    if all_zero.any():
        n = int(all_zero.sum())
        warnings.append(f"{label}: {n} competitor(s) show 0.00 from every judge, which this format uses "
                        "to mean 'did not dance this round' rather than a real score. Excluded from this "
                        "round's analysis.")
        df = df[~all_zero]
    return df


def parse_feisresults_pdf(pdf_path):
    """
    Returns (rounds, warnings) where `rounds` is a list of
    (round_label, scores_df, judge_columns) tuples -- one entry per
    independently-judged dance (R1 and R2 of Recall, Set Dance of Final),
    each a clean competitor x judge matrix with no missing values, ready
    for the same statistics used on the QuickFeis format.

    A competitor who didn't complete a given dance (withdrew, etc.) is
    dropped from THAT dance's table only -- handled the same principled
    way as QuickFeis's "not recalled" case.
    """
    warnings = []
    rounds = []

    with pdfplumber.open(pdf_path) as pdf:
        page_types = {}
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if "RECALL" in text:
                page_types.setdefault("recall", []).append(i)
            elif "FINAL MARKS" in text:
                page_types.setdefault("final", []).append(i)
            elif "Round 1" in text or "Round 2" in text:
                page_types.setdefault("solo_judge", []).append(i)

    if "solo_judge" in page_types:
        warnings.append(f"{len(page_types['solo_judge'])} page(s) show a single 'Solo Judge' ranking "
                        "(placements only, no marks, no panel to compare against). These are extracted "
                        "for reference but excluded from the bias-detection statistics, which require a panel.")

    if "recall" in page_types:
        df, judge_cols, w = _parse_panel_table(pdf_path, page_types["recall"], "Recall")
        warnings.extend(w)
        if len(df):
            for metric in ("col0", "col1"):  # col0 = R1, col1 = R2 -- both are fresh, independent marks
                label = "Recall R1" if metric == "col0" else "Recall R2"
                cols = [c for c in judge_cols if c.endswith(f" {metric}")]
                round_df = df[["competitor_id", "name", "team"] + cols].dropna()
                rename = {c: c.replace(f" {metric}", "") for c in cols}
                round_df = round_df.rename(columns=rename)
                clean_cols = list(rename.values())
                n_dropped = len(df) - len(round_df)
                if n_dropped:
                    warnings.append(f"{label}: {n_dropped} competitor(s) missing a mark for this "
                                    "dance (likely withdrew) and were excluded from this dance's analysis only.")
                round_df = _drop_did_not_dance(round_df, clean_cols, label, warnings)
                rounds.append((label, round_df.reset_index(drop=True), clean_cols))

    if "final" in page_types:
        df, judge_cols, w = _parse_panel_table(pdf_path, page_types["final"], "Final Marks")
        warnings.extend(w)
        if len(df):
            # col0 = Pre-recall Score (a repeat of the Recall stage's Total -- not a fresh
            # judgment, so we deliberately skip it). col1 = Set Dance -- the new, independent mark.
            cols = [c for c in judge_cols if c.endswith(" col1")]
            if cols:
                round_df = df[["competitor_id", "name", "team"] + cols].dropna()
                rename = {c: c.replace(" col1", "") for c in cols}
                round_df = round_df.rename(columns=rename)
                clean_cols = list(rename.values())
                n_dropped = len(df) - len(round_df)
                if n_dropped:
                    warnings.append(f"Final Set Dance: {n_dropped} competitor(s) missing a mark and were excluded.")
                round_df = _drop_did_not_dance(round_df, clean_cols, "Final Set Dance", warnings)
                rounds.append(("Final Set Dance", round_df.reset_index(drop=True), clean_cols))

    if not rounds:
        warnings.append("No analyzable judge panels were extracted from this file.")

    return rounds, warnings


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "feisresults_sample.pdf"
    rounds, warnings = parse_feisresults_pdf(path)
    for label, df, judge_cols in rounds:
        print(f"\n{label}: {len(df)} competitors, judges = {judge_cols}")
        print(df.head(3).to_string())
    print(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        print(" -", w)
