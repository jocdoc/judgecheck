# Deploying JudgeCheck to Vercel

Same GitHub → Vercel workflow as secondhandlegends. Any Python file inside
`api/` automatically becomes a live endpoint — Vercel sees `api/analyze.py`
and turns it into `yoursite.com/api/analyze`. Analysis is stateless by
default (nothing saved between requests) unless you explicitly provision the
optional database and use "Save to archive" — see that section below.

## What's in this folder

| File | What it is |
|---|---|
| `index.html` | The whole website (upload, PDF preview/edit grid, report display) |
| `api/analyze.py` | The statistics engine + request routing (CSV/XLSX, PDF preview, confirmed analysis) |
| `api/pdf_parsers.py` | Detects which PDF format was uploaded and routes to the right parser |
| `api/quickfeis_parser.py` | Parser for QuickFeis-format results PDFs |
| `api/feisresults_parser.py` | Parser for feisresults.com-format results PDFs |
| `api/db.py` | Optional historical archive: database connection, ingestion, and query functions (see "The historical archive" section) |
| `api/schema.sql` | Database schema for the archive feature |
| `requirements.txt` | Tells Vercel which Python libraries to install (`pdfplumber`, `psycopg2-binary`) |
| `vercel.json` | Gives the function up to 60 seconds (usually needs 2-10s depending on PDF size) |

## PDF support

Two formats are supported today, both tested against real competition PDFs with
zero extraction errors on independent verification: **QuickFeis** and
**feisresults.com**. A PDF upload never analyzes immediately — it always shows an
editable preview table first, with any extraction warnings called out (e.g. "this
competitor didn't dance this round"), so a human confirms the numbers before the
statistics run. Uploading a PDF in an unrecognized format returns a clear message
pointing at the CSV/Excel template instead of guessing.

If you get PDFs from other tabulation software later, send Claude a real sample
(not a mocked-up one) — the extraction logic was only trusted after testing against
actual files, and that's the right bar for any new format too.

## Steps

1. **Create a new GitHub repository** in the Ohana Ventures org (e.g. `judgecheck`).

2. **Upload these files**, preserving the `api/` folder structure exactly:
   - `index.html`, `requirements.txt`, `pyproject.toml`, `vercel.json` at the repo root
   - everything from `api/` into an `api/` folder in the repo, **including `schema.sql`**
     (it's not a Python file, but it needs to sit alongside `db.py`, which reads it at
     startup)
   GitHub's drag-and-drop can flatten folders — if it does, create files manually
   (Add file → Create new file → type `api/analyze.py` as the name; the slash
   creates the folder).

3. **Import into Vercel.** vercel.com → Add New → Project → pick the `judgecheck`
   repo → Framework Preset: "Other" → Deploy. Vercel reads `requirements.txt` and
   installs the Python libraries automatically, including `pdfplumber`.

4. **Test it**, in this order:
   - Download the CSV template, fill a few rows, upload, confirm a report appears.
   - Upload a QuickFeis or feisresults.com PDF, confirm the preview table shows
     correct-looking numbers, edit one cell, click through, confirm the report
     reflects the edit.
   - Upload a file with a missing column to confirm you get a friendly error.
   - Upload a PDF in neither known format to confirm you get the "format not
     recognized, use the template" message rather than a crash or bad output.

5. **Custom domain (optional):** Vercel project → Settings → Domains, same as your
   existing site.

## Updating later

Edit a file in the GitHub repo → Vercel redeploys automatically within a minute
or two.

## When you're ready for the paywall

Stripe Checkout + a lightweight auth layer (e.g. Clerk) slot in front of the
Generate/Confirm buttons without touching the statistics or parsing engines at
all. Bring that back to Claude as its own project when usage justifies it.

## Two things to do before promoting it publicly

1. **Legal review of the wording.** The reports name judges next to words like
   "Flagged." The disclaimer language is built in, but a lawyer should bless it
   before strangers pay for reports about real people.
2. **Rate limiting.** No limits yet on who can call the API. Fine for a quiet
   launch; revisit alongside the paywall/auth work.

## Still worth doing before wide release

- **Batch upload timing:** 4 real PDFs together took ~27 seconds to parse in
  testing (well under the 60s function limit, but not instant). A season's
  worth of competitions (10+ PDFs) uploaded at once could approach that
  limit — worth testing directly if that becomes a real usage pattern.
- ~~Test both parsers against 3-4 more real PDFs each~~ — done: a second
  QuickFeis variant (3-judge panel, different not-recalled markup) turned up
  real bugs in column detection that the first sample never exposed, now
  fixed and verified against 626 independently-checked competitor-rounds
  across three QuickFeis files. Worth continuing this habit with any further
  new PDFs — different regions/organizers still likely to show new layout
  quirks.
- The mitigations discussed earlier for circumvention (audit logging + report
  IDs, the judge-pair collusion check) aren't built yet — worth prioritizing
  before this handles reports people might rely on for real decisions.

## Multi-event watch list now has two independent checks

Uploading several PDFs together runs two separate analyses, since they answer
different questions and one can find something when the other has nothing to
test:

- **Competitor Patterns** — the same named dancer, scored unusually by the
  same judge, across separate events. Needs that dancer to actually reappear,
  so it correctly shows zero results when the uploaded files are different
  age/skill divisions of one event (a U16 dancer and a U17 dancer are
  different people).
- **School Patterns** — pools every competitor from a given school that a
  given judge scored, across every uploaded event. This is the one that
  actually works for "several divisions from one competition," since a
  school typically has dancers in more than one division and the same judge
  often judges more than one division. Validated against a realistic
  synthetic multi-division scenario (well-calibrated false-positive rate,
  correctly detected an injected bias) and confirmed against two real MAO
  2019 division files that genuinely share judges.

## The historical archive (new)

Instead of re-uploading a decade of PDFs every time you want to check
something, results can now be saved permanently to a small Postgres database
and queried directly — a specific judge's entire history, a specific school's,
or a specific competitor's, pooled across everything ever saved.

### What's new in this folder

| File | What it is |
|---|---|
| `api/db.py` | Database connection, ingestion, and query functions |
| `api/schema.sql` | The two-table schema (`events`, `marks`) — read and executed by `db.py` |

**Design choice worth understanding:** the database stores raw marks only —
never pre-computed statistics. A query reconstructs the exact same
`(round_label, DataFrame, judge_columns)` shape the existing, already-validated
`competitor_watch()` and `team_watch()` functions consume from freshly-uploaded
files — so those functions needed zero changes. Verified locally (round-tripped
all 4 real sample PDFs through flatten-then-reconstruct with zero mismatches,
and confirmed the statistics produce identical output — to within floating-point
noise — whether fed the original or the reconstructed data), but the actual
SQL execution against a live Neon database has **not** been tested from this
sandbox (no network path to Neon here) — that verification happens on your
first real deploy, the same way the entrypoint and routing issues did earlier.

### Provisioning the database (one-time)

1. In the Vercel dashboard, open the `judgecheck` project → **Storage** tab.
2. **Browse Marketplace Database Storage** → choose **Neon** (Postgres) →
   Free tier → Create. This provisions a database and automatically sets a
   `DATABASE_URL` environment variable on your project — no separate account,
   no manual connection-string copying.
3. Redeploy (or it may redeploy automatically after the storage change).
4. **Create the tables**: the schema needs to be run once. Easiest path —
   in the Vercel Storage tab, open the Neon dashboard's SQL editor (linked
   from your database's page) and paste the contents of `api/schema.sql`,
   then run it. You should see `events` and `marks` appear with no errors.
5. Test it: upload a PDF on the live site, preview it, click **"Save to
   archive"** instead of (or in addition to) "Run analysis." You should see
   a confirmation line, not an error. Then try **"Search the archive"** for
   a judge name from that same PDF — you should get a result, not
   "No history found."

### If something goes wrong here

This is new, unverified-on-real-infra territory, so treat any error here
the same way we treated the earlier Vercel-specific surprises — read the
actual error message and bring it back rather than guessing:

- **"DATABASE_URL is not set"** — the Neon integration didn't finish wiring
  up, or the deploy that picked up the new env var hasn't happened yet.
  Check Settings → Environment Variables for `DATABASE_URL`, redeploy if
  it's there but the error persists.
- **A `psycopg2` import or install error in the build log** — `psycopg2-binary`
  is a compiled package; it's normally fine on Vercel's Linux build
  environment, but if the build log shows it failing to install, that's a
  real, different problem worth bringing back — the fix is likely switching
  to the newer `psycopg` (v3) package, which has better prebuilt-wheel
  support, but that's a deliberate call to make with real evidence in hand,
  not a preemptive change.
- **A SQL error mentioning "relation does not exist"** — step 4 above (running
  `schema.sql`) didn't complete. Re-run it in the Neon SQL editor.
- **Storage approaching the free 0.5GB limit** — estimated to comfortably fit
  100-500 competitions of raw marks (~200-400MB), but if you get close, Neon's
  overage is cheap (~$0.35/GB-month) and provisions automatically; no action
  needed beyond noticing the bill.

### What's NOT built yet in the archive feature

- **No UI to browse "what's in the archive"** beyond searching a specific
  name — `db.list_events()` exists as a function but isn't wired to any page
  yet. Worth adding once there's enough saved data to want an overview.
- **No bulk-ingest tool.** Saving your existing decade of PDFs means
  uploading them through the same one-at-a-time (or small-batch) preview →
  confirm → save flow as everything else — genuinely tedious for hundreds of
  files. A dedicated bulk-upload path (skip the individual preview, rely on
  automated sanity checks instead, flag only what looks wrong) is the natural
  next step once the basic save/search flow is confirmed working live.
- **No full-archive scan** ("which judge looks most unusual across
  everything") — only targeted lookups (a specific judge/school/competitor)
  are wired up. A full scan is a much heavier computation across hundreds of
  events and is a better fit for a scheduled background job (Vercel Cron)
  than a live request; not attempted yet.
