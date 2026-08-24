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
    length n_judges (panel size, detected dynamically -- not assumed).

    POSITION-BASED, NOT TABLE-DETECTION-BASED -- matches the rest of this
    module (see module docstring for why). This used to rely on
    page.find_tables() to locate the 3-row judge-name header, which worked
    on larger files but was found live to misdetect the header as 2 cells
    instead of 5 on a small (6-competitor) file: pdfplumber's table
    heuristics are sensitive to how much content surrounds a table, and a
    short results page apparently changed what it inferred as cell/row
    boundaries there. Rebuilt to use the same word-coordinate approach
    already proven reliable for the mark rows themselves: every "Marks"
    header word (exactly one per judge column, always present) fixes both
    the column count and x-positions; the three "Round" labels fix the
    three header-row y-positions; each judge-name word is then assigned to
    whichever Marks column it sits closest to.
    """
    words = page.extract_words()

    marks_words = sorted((w for w in words if w["text"] == "Marks" and w["top"] < 200 and w["x0"] > 150),
                         key=lambda w: w["x0"])
    place_words = sorted((w for w in words if w["text"] == "Place" and w["top"] < 200 and w["x0"] > 150),
                         key=lambda w: w["x0"])
    if not marks_words or len(marks_words) != len(place_words):
        return None, None
    n_judges = len(marks_words)
    marks_x = [w["x0"] for w in marks_words]
    column_x = list(zip(marks_x, (w["x0"] for w in place_words)))

    round_label_words = sorted((w for w in words if w["text"] == "Round" and w["top"] < 150), key=lambda w: w["top"])
    if len(round_label_words) < 3:
        return None, None
    round_tops = [w["top"] for w in round_label_words[:3]]

    # Judge names are grouped by GAP, not by forcing them into fixed bins tied
    # to the Marks columns. Verified against a real file: a long surname
    # ("Yzanne Cloonan Noone") can extend past the midpoint between two Marks
    # columns, which silently misassigned words to the wrong judge under a
    # midpoint-binning approach that was tried first -- it merged "Noone"
    # into the NEXT judge's name instead of keeping it with "Yzanne Cloonan".
    # Within one judge's name, the gap between consecutive words is ~2pt;
    # between two different judges' names, it's 40pt+ -- confirmed directly
    # against this file's actual word coordinates, a wide and reliable
    # margin. x0 > 230 excludes the "Round N" label itself, which otherwise
    # gets swept into the first judge's name (confirmed first judge names on
    # this file consistently start past x0=238; the Round label ends by
    # x0=221).
    NAME_GAP_THRESHOLD = 20
    names_by_round = []
    for rtop in round_tops:
        row_words = sorted([w for w in words if abs(w["top"] - rtop) < 4 and w["x0"] > 230],
                           key=lambda w: w["x0"])
        groups = []
        for w in row_words:
            if groups and (w["x0"] - groups[-1][-1]["x1"]) < NAME_GAP_THRESHOLD:
                groups[-1].append(w)
            else:
                groups.append([w])
        if len(groups) != n_judges:
            return None, None  # can't reliably tell judges apart on this row -- don't guess
        names_by_round.append([" ".join(gw["text"] for gw in g) for g in groups])

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
    known_rank_xs = set()  # persisted once confidently detected -- see below (a SET,
                            # not a single value -- see the "REAL BUG FOUND LIVE" note)

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

            # Real bug found live: the page footer's second line ("2023 Mid
            # America Oireachtas", or whatever the event name is) starts with
            # a bare year number -- pure digits, so it matches _RANK_PATTERN
            # just like a real F Place value. When that word's x-position
            # happens to land in the same column as the actual rank box
            # (varies per file depending on judge-panel width), it gets
            # mistaken for a dancer's rank, and the parser then fails to find
            # a real name/school around it and logs a false "couldn't read
            # dancer info" warning -- confusing on files where it doesn't
            # happen to align, since the same footer text is on every page
            # but only *some* panel layouts put it in the rank column's path.
            # Verified against this real file: every genuine dancer row's
            # rank-box sits above top=513, while the footer starts at
            # top=562 -- a ~49pt gap. Excluding the bottom 60pt of the page
            # keeps a comfortable margin on both sides without needing to
            # match the footer's actual wording (which varies by event).
            candidates = [w for w in page_words if 20 <= w["x0"] <= 100
                         and 90 < w["top"] < page.height - 60
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
            #
            # REAL BUG FOUND LIVE: a tied F Place ("68 (Tie)") renders its
            # digit measurably further LEFT than the same digit alone ("68")
            # would, because the digit and "(Tie)" are centered together as
            # one unit within a fixed-width box. On a page with several ties,
            # this creates a THIRD x-cluster -- tied F-Place values, untied
            # F-Place values, and the Dancer-number column -- and the old
            # logic ("pick the leftmost cluster," written back when only 2
            # clusters were ever possible) grabbed the tied-only subset,
            # silently dropping every UNTIED competitor on that page from the
            # archive with no warning at all. Confirmed against this real
            # file: page 9 has [tied F-Place=32.2, untied F-Place=46.5,
            # Dancer#=61.1] -- the old code kept only the 3 tied rows and
            # dropped the other 5 real competitors on that page.
            #
            # FIX: the Dancer-number column is reliably the RIGHTMOST cluster
            # (it's always further right than F Place in this template, with
            # or without ties present), so accept every OTHER cluster as a
            # valid F-Place position, however many sub-clusters ties happen
            # to split it into on a given page.
            clusters = _find_x_clusters(candidates)
            if len(clusters) >= 2:
                # Accumulate (never discard) -- a page with only ONE tied row
                # can't independently reach min_cluster_size on its own tie
                # sub-position, so it must be able to rely on a tie position
                # confirmed by an EARLIER page, not just what this page alone
                # proves. Confirmed against this real file: page 9 has 3 tied
                # rows (enough to self-confirm x=32.2), but page 10 has only
                # 1 tied row -- replacing instead of accumulating here caused
                # that one competitor to be silently dropped even after the
                # main fix above.
                known_rank_xs |= set(clusters[:-1])  # everything except Dancer# (rightmost)
            elif not known_rank_xs and clusters:
                known_rank_xs = set(clusters)  # only choice available so far -- best effort
            if not known_rank_xs:
                continue

            rank_words = sorted(
                [w for w in candidates if any(abs((w["x0"] + w["x1"]) / 2 - rx) < 6 for rx in known_rank_xs)],
                key=lambda w: w["top"])

            # SEMANTIC recovery, independent of position clustering: a digit
            # immediately followed by the literal text "(Tie)" on the same
            # line IS an F-Place value, full stop -- the Dancer-number
            # column never has "(Tie)" next to it. This catches tied rows
            # that never reach min_cluster_size above (e.g. a single-page
            # file with only 1-2 tied competitors total, nowhere to
            # accumulate a confirmed tie-position from across pages) --
            # confirmed live on a real 6-competitor file where 2 tied rows
            # were silently dropped with zero warning even after the
            # clustering fix above, because 2 instances never reaches the
            # min_cluster_size=3 threshold needed to self-confirm.
            already_have = {(round(w["top"], 1), round(w["x0"], 1)) for w in rank_words}
            for tw in [w for w in page_words if w["text"] == "(Tie)"]:
                same_line_digits = [w for w in candidates if abs(w["top"] - tw["top"]) < 3 and w["x0"] < tw["x0"]]
                if not same_line_digits:
                    continue
                digit_word = max(same_line_digits, key=lambda w: w["x0"])  # nearest one to the left of "(Tie)"
                key = (round(digit_word["top"], 1), round(digit_word["x0"], 1))
                if key not in already_have:
                    rank_words.append(digit_word)
                    already_have.add(key)
            rank_words.sort(key=lambda w: w["top"])


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
