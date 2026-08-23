-- JudgeCheck historical database schema.
--
-- DESIGN PRINCIPLE: store raw marks only, never pre-computed statistics.
-- z-scores, gaps, and significance tests are cheap to compute at query
-- time and the methodology may improve later -- storing derived numbers
-- risks them silently drifting out of sync with the code that produced
-- them. The database holds exactly what was printed on each results PDF,
-- nothing more.
--
-- IDENTITY RESOLUTION IS DELIBERATELY NOT DONE HERE. There is no
-- "competitors" table with a permanently merged identity per person.
-- competitor_name and team are stored as plain text on each mark, and
-- matching "is this the same dancer across events" happens at QUERY TIME
-- via name comparison -- the same approach already used for in-session
-- analysis. A bad automatic name-merge in a live query produces a wrong
-- report you can inspect and dismiss; a bad merge baked into stored data
-- would be silent, permanent corruption of the historical record. Given
-- real spelling variation across a decade of PDFs, that trade favors
-- query-time resolution.

CREATE TABLE IF NOT EXISTS events (
    id              SERIAL PRIMARY KEY,
    source_filename TEXT NOT NULL,
    title           TEXT NOT NULL,
    format          TEXT NOT NULL,          -- 'QuickFeis' or 'feisresults.com'
    content_hash    TEXT NOT NULL UNIQUE,    -- sha256 of the uploaded PDF bytes; prevents re-ingesting the same file twice
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    warnings        JSONB NOT NULL DEFAULT '[]'::jsonb,  -- extraction warnings captured at ingestion time, for later review
    n_rounds        INTEGER NOT NULL,
    n_marks         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS marks (
    id               BIGSERIAL PRIMARY KEY,
    event_id         INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    round_label      TEXT NOT NULL,          -- e.g. "Round 1", "Recall R2", "Final Set Dance"
    competitor_number INTEGER NOT NULL,      -- the raw Dancer#/Card# as printed -- NOT a stable cross-event identity, see above
    competitor_name  TEXT NOT NULL,
    team             TEXT NOT NULL DEFAULT '',
    judge_name       TEXT NOT NULL,
    mark             NUMERIC NOT NULL
);

-- Every query pattern this app actually runs: look up a judge, look up a
-- competitor by name, look up a team, or look up everything for one event.
CREATE INDEX IF NOT EXISTS idx_marks_judge      ON marks (judge_name);
CREATE INDEX IF NOT EXISTS idx_marks_competitor ON marks (competitor_name);
CREATE INDEX IF NOT EXISTS idx_marks_team       ON marks (team);
CREATE INDEX IF NOT EXISTS idx_marks_event      ON marks (event_id);
-- The two-column indexes below are what a judge-vs-team or judge-vs-competitor
-- lookup actually filters on together, so a composite index serves those
-- queries directly instead of intersecting two single-column indexes.
CREATE INDEX IF NOT EXISTS idx_marks_judge_team       ON marks (judge_name, team);
CREATE INDEX IF NOT EXISTS idx_marks_judge_competitor ON marks (judge_name, competitor_name);
