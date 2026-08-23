"""
Parser for QuickFeis-family results PDFs. Handles variation found across
real files from different events/years:
  - Panel size varies (5 judges/round in one sample, 3 judges/round in
    another) -- detected dynamically from the header table's column count,
    never hardcoded.
  - "Not recalled to this round" is written two different ways across
    samples: a blank cell (older sample) or an explicit "0.000" mark paired
    with literal text like "NoRecall" or "No Show" in the Place column
    (newer sample). Both are detected as "this round wasn't judged for this
    competitor" rather than a real score, by checking whether the Place
    cell actually contains a valid rank -- not by matching specific marker
    text, so a third phrasing wouldn't silently slip through as a real 0.

WHY POSITION-BASED, NOT TABLE-DETECTION-BASED: an earlier version of this
parser used pdfplumber's automatic table detection for the per-round mark
rows. That worked cleanly on one real sample but fragmented into dozens of
tiny, inconsistent tables on another real sample with a different
underlying PDF structure (same software family, different render). Word
coordinates extract consistently across both, so this version reconstructs
rows from coordinates throughout, the same approach already validated for
the feisresults.com parser.
"""

import re
import pdfplumber
import pandas as pd

_RANK_PATTERN = re.compile(r"^\d+(-Tie)?$")


def _cluster_lines(words, tolerance=3):
    lines = []
    for w in sorted(words, key=lambda w: w["top"]):
        placed = False
        for line in lines:
            if abs(line[0]["top"] - w["top"]) <= tolerance:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    lines.sort(key=lambda line: line[0]["top"])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def _parse_dancer_block(lines):
    """Same 3-line dancer-info block shape as before: [number+name],
    [rank-box+school], [points+qualifying status]."""
    if len(lines) != 3:
        return None
    try:
        num_words = [w for w in lines[0] if w["x0"] < 60]
        text_words = [w for w in lines[0] if w["x0"] >= 70]
        competitor_id = int(num_words[0]["text"])
        name = " ".join(w["text"] for w in text_words)

        text_words = [w for w in lines[1] if w["x0"] >= 70]
        school = " ".join(w["text"] for w in text_words)

        return {"competitor_id": competitor_id, "name": name, "team": school}
    except (IndexError, ValueError):
        return None


def _find_x_clusters(candidates, tolerance=6, min_cluster_size=3):
    """Returns every qualifying cluster's center, sorted left to right (not
    just the chosen one) -- the caller needs to know whether a page shows
    ONE cluster or several, since only a page with multiple competing
    clusters can actually confirm which one is genuinely the rank column
    (see parse_quickfeis_pdf for why that distinction matters)."""
    centers = sorted((w["x0"] + w["x1"]) / 2 for w in candidates)
    found = []
    for c in centers:
        cluster = [y for y in centers if abs(y - c) < tolerance]
        if len(cluster) >= min_cluster_size:
            mean = sum(cluster) / len(cluster)
            if not any(abs(mean - f) < tolerance for f in found):
                found.append(mean)
    return sorted(found)


def _dominant_x_cluster(candidates, tolerance=6, min_cluster_size=3):
    """Returns the horizontal CENTER of the LEFTMOST cluster of candidate
    words with at least `min_cluster_size` members.

    WHY CENTER, NOT LEFT EDGE (x0): these are right-aligned-looking numeric
    columns, but they're actually center-aligned within a fixed box width.
    A single-digit rank ('1') and a wider one ('90-Tie') have very
    different left edges (their box shrinks/grows from a shared center),
    which previously caused a left-edge-based match to split what is
    really one column into multiple false clusters. The center coordinate
    stays essentially constant across 1-digit, 2-digit, and '-Tie'-suffixed
    values alike -- verified directly against real data before switching
    to it.

    WHY LEFTMOST-AMONG-QUALIFYING-CLUSTERS, NOT SIMPLY LARGEST: on a page
    where every competitor has a real final rank (no DNA rows), the rank
    column and the always-numeric competitor-ID column next to it can be
    the same size, so 'largest' doesn't reliably disambiguate them. The
    rank column is reliably the leftmost of the two by this layout's
    design, so once genuine one-off outliers are filtered by the minimum
    cluster size, leftmost is the correct tiebreaker.
    """
    clusters = _find_x_clusters(candidates, tolerance, min_cluster_size)
    if clusters:
        return clusters[0]
    centers = sorted((w["x0"] + w["x1"]) / 2 for w in candidates)
    best = []
    for c in centers:
        cluster = [y for y in centers if abs(y - c) < tolerance]
        if len(cluster) > len(best):
            best = cluster
    return sum(best) / len(best) if best else centers[0]


def _judge_columns_for_page(page):
    """Returns (judge_names_by_round, column_x) where judge_names_by_round
    is [[round1 names...], [round2 names...], [round3 names...]] and
    column_x is a list of (marks_x, place_x) per column position, both
    length n_judges (panel size, detected dynamically -- not assumed)."""
    tables = page.find_tables()
    header = None
    for t in tables:
        if len(t.rows) == 3 and t.rows[0].cells and all(t.rows[0].cells):
            header = t
            break
    if header is None:
        return None, None
    n_judges = len(header.rows[0].cells)

    names_by_round = []
    for row in header.rows:
        names = []
        for cell in row.cells:
            names.append(page.crop(cell).extract_text().strip() if cell else None)
        names_by_round.append(names)

    header_bottom = header.bbox[3]
    words = page.extract_words()
    sub = [w for w in words if header_bottom < w["top"] < header_bottom + 30]
    marks_x = sorted(w["x0"] for w in sub if w["text"] == "Marks")
    place_x = sorted(w["x0"] for w in sub if w["text"] == "Place")
    if len(marks_x) != n_judges or len(place_x) != n_judges:
        return None, None
    column_x = list(zip(marks_x, place_x))
    return names_by_round, column_x


def parse_quickfeis_pdf(pdf_path):
    """
    Returns (rounds, competitor_index, warnings) -- same shape as before.
    See module docstring for what changed and why.
    """
    warnings = []
    competitor_index = {}
    round_rows = {0: [], 1: [], 2: []}
    judge_names_by_round = None
    column_x = None
    known_rank_x = None  # persisted once confidently detected -- see below

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_words = page.extract_words()

            names, cols = _judge_columns_for_page(page)
            if names is not None:
                judge_names_by_round, column_x = names, cols

            if judge_names_by_round is None:
                warnings.append(f"Page {page_num}: couldn't find the judge-name header; skipping this page.")
                continue

            n_judges = len(column_x)

            candidates = [w for w in page_words if 20 <= w["x0"] <= 100 and w["top"] > 90
                         and _RANK_PATTERN.match(w["text"])]
            if not candidates:
                continue

            # The rank-box column's position is a fixed layout property of the
            # whole document, not something to re-guess from scratch on every
            # page. Crucially, a page is only trusted to CONFIRM which cluster
            # is the true rank column when it shows multiple competing
            # clusters -- a page with only ONE cluster present (e.g. a page of
            # nothing but "Did Not Attend" rows, where only the always-numeric
            # competitor-ID column has any digits at all) can't actually prove
            # anything, and blindly trusting it there previously caused the
            # wrong column to silently override an already-correct one.
            clusters = _find_x_clusters(candidates)
            if len(clusters) >= 2:
                known_rank_x = clusters[0]
            elif known_rank_x is None and clusters:
                known_rank_x = clusters[0]  # only choice available so far -- best effort
            if known_rank_x is None:
                continue
            rank_x = known_rank_x

            rank_words = sorted([w for w in candidates if abs((w["x0"] + w["x1"]) / 2 - rank_x) < 6],
                                key=lambda w: w["top"])


            for rw in rank_words:
                rank_top = rw["top"]
                name_words = [w for w in page_words if w["x0"] < 230 and -19 <= w["top"] - rank_top <= -8]
                mid_words = [w for w in page_words if w["x0"] < 230 and -4 <= w["top"] - rank_top <= 4]
                bottom_words = [w for w in page_words if w["x0"] < 230 and 8 <= w["top"] - rank_top <= 20]
                meta = _parse_dancer_block([_cluster_lines(name_words)[0] if name_words else [],
                                           _cluster_lines(mid_words)[0] if mid_words else [],
                                           _cluster_lines(bottom_words)[0] if bottom_words else []])
                if meta is None:
                    # a page (or row) where the competitor "Did Not Attend" has no
                    # real rank to anchor on -- checked for directly since different
                    # events phrase this differently ("(DNA)" + "No Show", or "NS" +
                    # "Did Not Attend") rather than an actual extraction failure
                    nearby = [w for w in page_words if abs(w["top"] - rank_top) < 20]
                    nearby_text = " ".join(w["text"] for w in nearby).upper()
                    compact = nearby_text.replace(" ", "")
                    if ("DNA" in nearby_text or "NOSHOW" in compact or "ATTEND" in nearby_text
                            or re.search(r"\bNS\b", nearby_text)):
                        continue  # did not attend -- correctly has no data to extract, not an error
                    warnings.append(f"Page {page_num}: couldn't read dancer info near y={rank_top:.0f}; skipped.")
                    continue
                cid = meta["competitor_id"]
                competitor_index[cid] = meta

                round_offsets = [(-19, -8), (-4, 4), (8, 20)]
                did_not_attend = False
                for round_idx, (lo, hi) in enumerate(round_offsets):
                    row_words = [w for w in page_words if w["x0"] > 230 and lo <= w["top"] - rank_top <= hi]
                    marks = {}
                    n_present, n_missing = 0, 0
                    for col_idx in range(n_judges):
                        judge_name = judge_names_by_round[round_idx][col_idx]
                        marks_x, place_x = column_x[col_idx]
                        marks_word = next((w for w in row_words if abs(w["x0"] - marks_x) < 15), None)
                        place_words = [w for w in row_words if abs(w["x0"] - place_x) < 25]
                        place_text = "".join(w["text"] for w in place_words)

                        valid_rank = bool(_RANK_PATTERN.match(place_text))
                        if not valid_rank:
                            n_missing += 1
                            if place_text.replace(" ", "").upper() in ("NOSHOW",):
                                did_not_attend = True
                            continue
                        if marks_word is None:
                            n_missing += 1
                            warnings.append(f"Page {page_num}: competitor {cid} round {round_idx+1} judge "
                                            f"'{judge_name}' has a rank but no readable mark; check manually.")
                            continue
                        try:
                            marks[judge_name] = float(marks_word["text"])
                            n_present += 1
                        except ValueError:
                            n_missing += 1
                            warnings.append(f"Page {page_num}: unreadable mark '{marks_word['text']}' for "
                                            f"competitor {cid}, round {round_idx+1}, judge '{judge_name}'.")

                    if did_not_attend:
                        continue
                    if 0 < n_present < n_judges:
                        warnings.append(f"Competitor {cid}, round {round_idx+1}: only {n_present} of {n_judges} "
                                        "judges' marks were readable. Check this competitor's row manually.")
                    if n_present == n_judges:
                        row = {"competitor_id": cid, "name": meta["name"], "team": meta["team"]}
                        row.update(marks)
                        round_rows[round_idx].append(row)

    rounds = []
    for round_idx in range(3):
        rows = round_rows[round_idx]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        judge_cols = [c for c in df.columns if c not in ("competitor_id", "name", "team")]
        rounds.append((f"Round {round_idx + 1}", df, judge_cols))

    if not rounds:
        warnings.append("No usable rounds were extracted from this file.")

    return rounds, competitor_index, warnings


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "feis_sample.pdf"
    rounds, competitor_index, warnings = parse_quickfeis_pdf(path)
    print(f"Extracted {len(competitor_index)} competitors across {len(rounds)} rounds.")
    for label, df, judge_cols in rounds:
        print(f"\n{label}: {len(df)} competitors, judges = {judge_cols}")
        print(df.head(3).to_string())
    print()
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(" -", w)
    else:
        print("\nNo warnings.")
