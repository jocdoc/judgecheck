"""
Parser for QuickFeis-format results PDFs (the dominant software used to
tabulate Irish dance feis/oireachtas results). Built and tested against a
real sample file, not a guessed format.

STRUCTURE OF THIS PDF FORMAT (for reference):
- Each competition has 3 rounds, each round judged by a DIFFERENT panel of
  5 judges (15 distinct judges total). This is a real anti-bias mechanism
  already built into how Irish dance is judged -- panels rotate so no
  single judge sees every dancer's every round.
- For each dancer: 5 "position" columns (Left/Center-Left/Center/
  Center-Right/Right) x 3 rounds x 3 numbers (Marks, Place, Indicative
  Points). We only want "Marks" -- the judge's own raw 0-100 score --
  since Place and I.Points are already-processed derivatives of Marks,
  and our statistics do their own normalization from raw scores.
- Dancer identity (number, name, school, final placement, qualifying
  status) sits in a left-hand block, 3 short lines per dancer, that lines
  up vertically with that dancer's 3 rounds of marks.
- A dancer who didn't attend shows "NS" (not scored) instead of numbers.
"""

import re
import pdfplumber
import pandas as pd


def _cluster_lines(words, tolerance=3):
    """Groups words into text lines by vertical position (PDFs don't
    label lines explicitly -- this reconstructs them from coordinates)."""
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
    """Turns 3 reconstructed text lines into dancer metadata. Returns None
    for a block that doesn't match the expected shape (defensive -- so a
    malformed block gets flagged rather than silently mis-parsed)."""
    if len(lines) != 3:
        return None
    try:
        num_words = [w for w in lines[0] if w["x0"] < 60]
        text_words = [w for w in lines[0] if w["x0"] >= 70]
        competitor_id = int(num_words[0]["text"])
        name = " ".join(w["text"] for w in text_words)

        num_words = [w for w in lines[1] if w["x0"] < 60]
        text_words = [w for w in lines[1] if w["x0"] >= 70]
        f_place_raw = num_words[0]["text"] if num_words else None
        school = " ".join(w["text"] for w in text_words)

        num_words = [w for w in lines[2] if w["x0"] < 60]
        text_words = [w for w in lines[2] if w["x0"] >= 70]
        t_points_raw = num_words[0]["text"] if num_words else None
        qualifying_status = " ".join(w["text"] for w in text_words)

        return {
            "competitor_id": competitor_id,
            "name": name,
            "team": school,
            "f_place": f_place_raw,
            "t_points": t_points_raw,
            "qualifying_status": qualifying_status,
        }
    except (IndexError, ValueError):
        return None


def parse_quickfeis_pdf(pdf_path):
    """
    Returns (rounds, competitor_index, warnings).

    IMPORTANT STRUCTURAL POINT (found by testing against a real file):
    lower-ranked competitors get "recalled" to fewer rounds than the top
    competitors -- a real, standard part of how these competitions work,
    not a data error. That means one competition isn't really one judge
    panel scoring everyone; it's up to 3 SEPARATE panels (one per round),
    each scoring only the competitors who made it that far. So this
    function returns each round as its own clean, complete table --
    exactly the (event_label, dataframe, judge_columns) shape our
    single-event report already expects, and also exactly the shape our
    multi-event competitor-watch tool expects if you feed it all 3 rounds
    as if they were 3 "events" -- a natural way to check whether a
    specific round-3 judge treated a specific competitor differently than
    that same competitor's round-1/round-2 judges did.

    `rounds` is a list of (round_label, scores_df, judge_columns) tuples.
    `competitor_index` maps competitor_id -> {name, team, f_place, ...}
    for use in reports (looked up once, not repeated per round).
    `warnings` are plain-language strings about anything genuinely
    unusual -- NOT including the normal "not recalled to this round"
    case, which is expected and handled, not an error.
    """
    warnings = []
    competitor_index = {}
    # per-round accumulation: round_idx -> list of {competitor_id, team, judge: mark}
    round_rows = {0: [], 1: [], 2: []}
    judge_names_by_round = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_words = page.extract_words()
            tables = page.find_tables()
            if not tables:
                continue

            judge_header = None
            data_tables = []
            for t in tables:
                if len(t.rows) == 3 and t.rows[0].cells and len(t.rows[0].cells) == 5:
                    judge_header = t
                else:
                    data_tables.append(t)

            if judge_header is not None:
                names_by_round = []
                for row in judge_header.rows:
                    names = []
                    for cell in row.cells:
                        names.append(page.crop(cell).extract_text().strip() if cell else None)
                    names_by_round.append(names)
                judge_names_by_round = names_by_round

            if judge_names_by_round is None:
                warnings.append(f"Page {page_num}: couldn't find the judge-name header; skipping this page.")
                continue

            row_tables = [t for t in data_tables if len(t.rows) == 1 and len(t.rows[0].cells) == 15]
            if len(row_tables) % 3 != 0:
                warnings.append(f"Page {page_num}: found {len(row_tables)} score rows, not a multiple of 3. "
                                "Check this page manually.")

            for i in range(0, len(row_tables) - len(row_tables) % 3, 3):
                round_tables = row_tables[i:i + 3]
                block_top = round_tables[0].rows[0].cells[0][1]
                block_bottom = round_tables[2].rows[0].cells[0][3]

                left_words = [w for w in page_words
                              if w["x0"] < 222 and block_top - 2 <= w["top"] <= block_bottom + 2]
                meta = _parse_dancer_block(_cluster_lines(left_words))
                if meta is None:
                    warnings.append(f"Page {page_num}: couldn't read dancer info near y={block_top:.0f}; skipped.")
                    continue
                cid = meta["competitor_id"]
                competitor_index[cid] = meta

                did_not_attend = False
                for round_idx, t in enumerate(round_tables):
                    cells = t.rows[0].cells
                    marks = {}
                    n_present, n_missing = 0, 0
                    for pos_idx in range(5):
                        judge_name = judge_names_by_round[round_idx][pos_idx]
                        text = (page.crop(cells[pos_idx * 3]).extract_text() or "").strip()
                        if text == "":
                            n_missing += 1  # not recalled to this round -- expected, not an error
                            continue
                        if text.upper() == "NS":
                            did_not_attend = True
                            continue
                        try:
                            marks[judge_name] = float(text)
                            n_present += 1
                        except ValueError:
                            warnings.append(f"Page {page_num}: unreadable mark '{text}' for competitor {cid}, "
                                            f"round {round_idx+1}, judge '{judge_name}'. Left blank -- check this cell.")

                    if did_not_attend:
                        continue
                    if 0 < n_present < 5:
                        warnings.append(f"Competitor {cid}, round {round_idx+1}: only {n_present} of 5 judges' "
                                        "marks were readable (expected 0 or 5). Check this competitor's row manually.")
                    if n_present == 5:
                        row = {"competitor_id": cid, "team": meta["team"]}
                        row.update(marks)
                        round_rows[round_idx].append(row)

    rounds = []
    for round_idx in range(3):
        rows = round_rows[round_idx]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        judge_cols = [c for c in df.columns if c not in ("competitor_id", "team")]
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

