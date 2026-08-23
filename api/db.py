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
import json
import os

import pandas as pd
import psycopg2
import psycopg2.extras


def _connection():
    """Opens a new connection using the DATABASE_URL environment variable
    Vercel's Neon integration sets automatically. A fresh connection per
    request (not a pool) matches how Neon expects short-lived serverless
    functions to connect -- Neon's pooled connection string handles the
    actual pooling on their end."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Provision a Neon database from the Vercel "
            "dashboard (Storage tab -> Marketplace -> Neon) and it will be set "
            "automatically; see DEPLOY.md."
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
    for why the full round is needed, not just the filtered rows."""
    cur.execute(f"SELECT DISTINCT event_id, round_label FROM marks WHERE {where_clause}", params)
    keys = cur.fetchall()

    rounds = []
    for event_id, round_label in keys:
        cur.execute(
            """SELECT e.title, m.competitor_number, m.competitor_name, m.team, m.judge_name, m.mark
               FROM marks m JOIN events e ON e.id = m.event_id
               WHERE m.event_id = %s AND m.round_label = %s""",
            (event_id, round_label),
        )
        rows = cur.fetchall()
        if not rows:
            continue
        event_title = rows[0][0]
        long_rows = [(r[1], r[2], r[3], r[4], r[5]) for r in rows]  # drop the title column
        result = _pivot_long_to_wide(long_rows, event_title, round_label)
        if result:
            rounds.append(result)
    return rounds


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
