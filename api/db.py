"""
Database layer for the historical results archive.

KEY DESIGN CHOICE: every query function here returns data in EXACTLY the
same (round_label, DataFrame, judge_columns) shape that analyze.py's
competitor_watch() and team_watch() already consume from freshly-uploaded
files. That means the statistics code -- already validated with real PDFs,
false-positive checks, and injected-bias detection tests -- needs ZERO
changes to work against the database. Only the data source changes.

WHY A FULL ROUND GETS FETCHED, NOT JUST ONE JUDGE'S ROWS: computing "how
far did this judge's score stray from the panel" requires the OTHER
judges' scores for the same competitors in the same round -- that's the
whole basis of the method. So looking up "everything involving Judge X"
means: find every (event, round) where Judge X appears, then pull the
COMPLETE panel data for each of those rounds, not just Judge X's own
marks.
"""

import hashlib
import itertools
import json
import os

import pandas as pd
import psycopg2
import psycopg2.extras


def _connection():
    """Opens a new connection using whichever connection-string environment
    variable is actually present. Confirmed on a real deployment that
    Vercel's Neon Marketplace integration doesn't always name it
    DATABASE_URL -- it created a POSTGRES_* set (POSTGRES_URL,
    POSTGRES_HOST, etc.) instead on at least one real install. Checking
    several known names, in the order Neon/Vercel integrations are known to
    use them, means this doesn't silently break if the naming differs
    between integration versions or plans. POSTGRES_URL is preferred over
    POSTGRES_URL_NON_POOLING when both exist, since Neon's pooled connection
    string is what's meant for short-lived serverless functions like this
    one -- using the unpooled one instead works but defeats the point of
    pooling under real traffic.

    A fresh connection per request (not a pool on our side) matches how
    Neon expects short-lived serverless functions to connect."""
    url = (os.environ.get("DATABASE_URL")
           or os.environ.get("POSTGRES_URL")
           or os.environ.get("POSTGRES_PRISMA_URL")
           or os.environ.get("POSTGRES_URL_NON_POOLING"))
    if not url:
        raise RuntimeError(
            "No database connection string found in the environment (checked "
            "DATABASE_URL, POSTGRES_URL, POSTGRES_PRISMA_URL, POSTGRES_URL_NON_POOLING). "
            "Provision a Neon database from the Vercel dashboard (Storage tab -> "
            "Marketplace -> Neon) and one of these will be set automatically; see DEPLOY.md."
        )
    return psycopg2.connect(url, sslmode="require")


def ensure_schema():
    """Creates the tables/indexes if they don't already exist. Safe to call
    on every request -- CREATE TABLE IF NOT EXISTS is a no-op once the
    schema is in place, so this isn't a real migration system, just a
    zero-effort way to make sure a fresh database is ready."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    with open(schema_path) as f:
        schema_sql = f.read()
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    finally:
        conn.close()


def content_hash_of(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def already_ingested(content_hash):
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title FROM events WHERE content_hash = %s", (content_hash,))
            row = cur.fetchone()
            return {"id": row[0], "title": row[1]} if row else None
    finally:
        conn.close()


def ingest_confirmed_rounds(rounds, source_filename, title, fmt, warnings, content_hash):
    """
    rounds: list of (round_label, DataFrame, judge_columns) -- the exact
    shape produced by _preview_json_to_rounds() after a human has reviewed
    and confirmed the extracted data. Only ever call this on data that's
    already been through that review step; there is no separate QA gate
    inside this function.

    Returns the new event's id, or raises if content_hash already exists
    (callers should check already_ingested() first for a friendlier message).
    """
    conn = _connection()
    try:
        with conn.cursor() as cur:
            n_marks = sum(len(df) * len(jc) for _, df, jc in rounds)
            cur.execute(
                """INSERT INTO events (source_filename, title, format, content_hash, warnings, n_rounds, n_marks)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (source_filename, title, fmt, content_hash, json.dumps(warnings), len(rounds), n_marks),
            )
            event_id = cur.fetchone()[0]

            rows_to_insert = []
            for round_label, df, judge_cols in rounds:
                for _, row in df.iterrows():
                    cid = int(row["competitor_id"])
                    name = str(row["name"]) if "name" in df.columns and pd.notna(row["name"]) else ""
                    team = str(row["team"]) if "team" in df.columns and pd.notna(row["team"]) else ""
                    for judge in judge_cols:
                        rows_to_insert.append((event_id, round_label, cid, name, team, judge, float(row[judge])))

            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO marks (event_id, round_label, competitor_number, competitor_name, team, judge_name, mark)
                   VALUES %s""",
                rows_to_insert,
            )
        conn.commit()
        return event_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _pivot_long_to_wide(rows, event_title, round_label):
    """Pure transformation, kept separate from any database call so it can
    be tested directly: takes long-format rows (competitor_id, name, team,
    judge, mark) for one event+round and reconstructs the (label, wide_df,
    judge_columns) shape the statistics functions expect. Returns None if
    the round has no usable data."""
    if not rows:
        return None
    long_df = pd.DataFrame(rows, columns=["competitor_id", "name", "team", "judge", "mark"])
    # The mark column arrives as psycopg2's mapping of Postgres NUMERIC -- Python's
    # decimal.Decimal, not float. Confirmed live: every statistics function downstream
    # assumes plain floats and mixing the two raises "unsupported operand type(s) for
    # -: 'float' and 'decimal.Decimal'" the moment any arithmetic touches archive data
    # (which is every archive search, hence all three kinds failing identically).
    # Cast right here, once, at the single point every archive query reconstructs
    # through, rather than downstream in each statistics function.
    long_df["mark"] = long_df["mark"].astype(float)
    judge_cols = sorted(long_df["judge"].unique().tolist())
    wide = long_df.pivot_table(index=["competitor_id", "name", "team"], columns="judge",
                               values="mark", aggfunc="first").reset_index()
    wide.columns.name = None
    # only keep competitors with a mark from every judge in this round (should
    # always hold, since ingestion only stores complete-panel rows, but this
    # guards against a partial/corrupted round rather than silently misreading one)
    wide = wide.dropna(subset=judge_cols)
    if len(wide) == 0:
        return None
    label = f"{event_title} \u2014 {round_label}"
    return (label, wide, judge_cols)


def _rounds_where(cur, where_clause, params):
    """Shared logic: find every (event_id, round_label) matching a filter,
    then fetch the COMPLETE panel data for each -- see module docstring
    for why the full round is needed, not just the filtered rows.

    TWO SEPARATE QUERIES, DELIBERATELY: the first finds which
    (event_id, round_label) keys match the filter; the second fetches the
    FULL PANEL (every judge, every mark) for exactly those keys, with NO
    filter applied to the second query. This distinction is not optional --
    an EARLIER version applied where_clause to a single combined query
    directly, which happened to work for the unconditional full-archive
    fetch (fetch_all_rounds) but silently broke every SCOPED fetch
    (fetch_rounds_for_judge, fetch_rounds_for_team): filtering by
    "judge_name = X" doesn't just find X's rounds, it also discards every
    OTHER judge's marks in those same rounds, leaving nothing for the
    gap-vs-panel comparison to compare against. That produced a real
    production bug where a specific judge's single-judge lookup silently
    showed all-clear despite the archive-wide tally correctly flagging her
    16 times -- caught only by directly diffing the two code paths against
    real production data, not by synthetic testing, since the earlier
    synthetic db.py tests only checked the ROUND-GROUPING logic on data
    that was already correct, never a genuinely FILTERED query end-to-end.

    Still just 2 total queries regardless of archive size (not the N+1
    per-round pattern this was originally rewritten to avoid -- see below).
    The ORDER BY is required: itertools groupby only groups CONSECUTIVE
    matching rows, so the rows must already be sorted by the group key
    before we iterate.

    TEMPORARY TIMING INSTRUMENTATION: printed with flush=True so the
    numbers land in Vercel's log even if the function is later killed by
    the 60s timeout (unflushed output can be lost on an abrupt kill).
    Added to diagnose exactly which stage a 504 on "Recompute now" is
    coming from at current archive scale, before picking a fix -- remove
    once the bottleneck is identified and addressed."""
    import time
    t0 = time.monotonic()
    cur.execute(f"SELECT DISTINCT event_id, round_label FROM marks WHERE {where_clause}", params)
    keys = cur.fetchall()
    t1 = time.monotonic()
    print(f"[timing] _rounds_where: key query = {t1 - t0:.2f}s, {len(keys)} (event,round) keys", flush=True)
    if not keys:
        return []

    event_ids = [k[0] for k in keys]
    round_labels = [k[1] for k in keys]
    cur.execute(
        """SELECT m.event_id, m.round_label, e.title, m.competitor_number,
                  m.competitor_name, m.team, m.judge_name, m.mark
           FROM marks m JOIN events e ON e.id = m.event_id
           WHERE (m.event_id, m.round_label) IN (SELECT * FROM unnest(%s::int[], %s::text[]))
           ORDER BY m.event_id, m.round_label""",
        (event_ids, round_labels),
    )
    all_rows = cur.fetchall()
    t2 = time.monotonic()
    print(f"[timing] _rounds_where: full-panel query = {t2 - t1:.2f}s, {len(all_rows)} rows", flush=True)

    rounds = []
    for (event_id, round_label), group in itertools.groupby(all_rows, key=lambda r: (r[0], r[1])):
        group = list(group)
        event_title = group[0][2]
        long_rows = [(r[3], r[4], r[5], r[6], r[7]) for r in group]
        result = _pivot_long_to_wide(long_rows, event_title, round_label)
        if result:
            rounds.append(result)
    t3 = time.monotonic()
    print(f"[timing] _rounds_where: per-round reconstruction = {t3 - t2:.2f}s, {len(rounds)} rounds built", flush=True)
    return rounds


def count_all_marks():
    """Total number of individual results (rows in `marks`) across the whole
    archive. A single COUNT(*) -- no join, no round reconstruction -- so
    it's safe to call on every page load, unlike fetch_all_rounds() which
    rebuilds full panel data for the whole archive."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM marks")
            return cur.fetchone()[0]
    finally:
        conn.close()


def count_events_for_judge(judge_name):
    """Fast COUNT query -- deliberately cheap, unlike fetch_rounds_for_judge
    which reconstructs full panel data for every round the judge appears in.
    Used to show a lightweight 'N events on record' figure for every judge
    in a single-event report without paying the cost of a full history
    fetch-and-analyze for judges nobody's asking about."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT event_id) FROM marks WHERE judge_name = %s", (judge_name,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def judge_experience_percentile(judge_name):
    """Gives 'N events on record' real context by comparing it against
    every other judge in the archive -- the same idea already used in the
    single-event report, which shows each panel judge's own prior-event
    count side by side for comparison, just extended to the WHOLE judge
    pool instead of just the judges on one panel.

    One GROUP BY over the whole marks table, not a per-judge loop -- gets
    every judge's event count in a single query, then this judge's
    percentile is trivial arithmetic in Python. Uses idx_marks_judge.

    Returns (n_events, percentile, n_judges_in_archive). percentile is None
    if the archive has no judges on record at all yet (can't rank against
    an empty pool)."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT judge_name, COUNT(DISTINCT event_id) FROM marks GROUP BY judge_name")
            counts = dict(cur.fetchall())
    finally:
        conn.close()
    target = counts.get(judge_name, 0)
    if not counts:
        return target, None, 0
    n_judges = len(counts)
    n_at_or_below = sum(1 for n in counts.values() if n <= target)
    percentile = round(n_at_or_below / n_judges * 100)
    return target, percentile, n_judges


def fetch_rounds_for_judge(judge_name):
    conn = _connection()
    try:
        with conn.cursor() as cur:
            return _rounds_where(cur, "judge_name = %s", (judge_name,))
    finally:
        conn.close()


def fetch_rounds_for_team(team_name):
    conn = _connection()
    try:
        with conn.cursor() as cur:
            return _rounds_where(cur, "team = %s", (team_name,))
    finally:
        conn.close()


def fetch_rounds_for_competitor(name):
    conn = _connection()
    try:
        with conn.cursor() as cur:
            return _rounds_where(cur, "competitor_name = %s", (name,))
    finally:
        conn.close()


def fetch_all_rounds(limit_events=None):
    """Full-database pull -- every round, every event. Expensive on a large
    archive; see analyze.py for why this is treated differently (background
    job candidate) from a judge/team/competitor-scoped lookup."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            if limit_events:
                cur.execute("SELECT id FROM events ORDER BY id DESC LIMIT %s", (limit_events,))
                ids = [r[0] for r in cur.fetchall()]
                if not ids:
                    return []
                return _rounds_where(cur, "event_id = ANY(%s)", (ids,))
            return _rounds_where(cur, "TRUE", ())
    finally:
        conn.close()


def list_events(limit=100):
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, title, format, n_rounds, n_marks, ingested_at
                   FROM events ORDER BY ingested_at DESC LIMIT %s""",
                (limit,),
            )
            cols = ["id", "title", "format", "n_rounds", "n_marks", "ingested_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def latest_import_timestamp():
    """The ingested_at of the most recently saved event, or None if the
    archive is empty. Used to answer 'has anything new landed since the
    tally was last recomputed' without pulling any actual round data --
    a single-row MAX() lookup, not an archive scan."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(ingested_at) FROM events")
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def get_worst_judges_cache():
    """Returns the cached worst-judges tally dict (with a 'computed_at' ISO
    timestamp added), or None if it's never been computed yet -- e.g. right
    after ensure_schema() on a brand-new database, before any import has
    happened. Read-only, cheap: a single-row lookup, not an archive scan."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data, computed_at FROM worst_judges_cache WHERE id = 1")
            row = cur.fetchone()
            if not row:
                return None
            data, computed_at = row
            return {**data, "computed_at": computed_at.isoformat()}
    finally:
        conn.close()


def set_worst_judges_cache(data):
    """Recomputes and stores the tally. Called by analyze.py right after a
    successful import (an event actually saved, not a duplicate/no-op) --
    deliberately NOT called on a plain page load or archive search, per the
    explicit decision that this only needs to update when new data lands,
    not run continuously."""
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO worst_judges_cache (id, computed_at, data)
                   VALUES (1, now(), %s)
                   ON CONFLICT (id) DO UPDATE SET computed_at = now(), data = EXCLUDED.data""",
                (json.dumps(data),),
            )
        conn.commit()
    finally:
        conn.close()


def search_names(prefix, kind="competitor_name", limit=15):
    """Powers a type-ahead search box -- kind is 'competitor_name',
    'judge_name', or 'team'."""
    if kind not in ("competitor_name", "judge_name", "team"):
        raise ValueError("invalid search kind")
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT {kind} FROM marks WHERE {kind} ILIKE %s ORDER BY {kind} LIMIT %s",
                (f"%{prefix}%", limit),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def fuzzy_match_names(query, kind="judge_name", limit=5):
    """
    'Did you mean' suggestions for a search that found no exact match.
    Deliberately NOT built on SQL substring matching (ILIKE) -- a real
    case that came up confirmed why: searching "Aaron Crosby" should
    suggest "Aaron Crosbie", but "Aaron Crosby" isn't a substring of
    "Aaron Crosbie" (the "-y" vs "-ie" ending breaks it), so ILIKE '%Aaron
    Crosby%' finds nothing. Character-level similarity (Python's stdlib
    difflib) correctly catches a one-letter spelling difference like this
    where substring matching can't. The full distinct-name list for one
    column is fetched and compared in Python rather than attempting this
    in SQL -- for an archive of judges/schools spanning years of events,
    that list is realistically in the hundreds, not large enough to
    justify a fuzzy-search Postgres extension (pg_trgm) for this.
    """
    if kind not in ("competitor_name", "judge_name", "team"):
        raise ValueError("invalid search kind")
    from difflib import get_close_matches
    conn = _connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT {kind} FROM marks WHERE {kind} != '' LIMIT 5000")
            all_names = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    return get_close_matches(query, all_names, n=limit, cutoff=0.6)
