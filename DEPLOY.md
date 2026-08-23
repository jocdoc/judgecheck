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
| `api/feisresults_parser.py` | Parser for the original feisresults.com layout (multi-competition file, RECALL/FINAL MARKS tables) |
| `api/maro_parser.py` | Parser for the "Mid Atlantic Region Oireachtas" feisresults.com layout (one competition per file, rotated vertical judge headers) |
| `api/db.py` | Optional historical archive: database connection, ingestion, and query functions (see "The historical archive" section) |
| `api/schema.sql` | Database schema for the archive feature |
| `requirements.txt` | Tells Vercel which Python libraries to install (`pdfplumber`, `psycopg2-binary`) |
| `vercel.json` | Gives the function up to 60 seconds (usually needs 2-10s depending on PDF size) |

## PDF support

Three formats are supported today, all tested against real competition PDFs with
zero extraction errors on independent verification: **QuickFeis**, the original
**feisresults.com** layout, and a third layout used by "Mid Atlantic Region
Oireachtas" — also feisresults.com-branded, but structurally unrelated to the
other one (one PDF per competition rather than one per whole event, and judge
names printed as rotated vertical column headers that need coordinate-and-
rotation-aware extraction, not plain text reading). Worth knowing: "feisresults.com"
branding alone doesn't mean a PDF matches either feisresults.com parser — the
platform apparently supports more than one export template, and format detection
had to be built around a more specific text signal than the copyright footer.
This format also uses a "drop the high and low score, sum the middle three" IP
system (confirmed against real totals before trusting the extraction), though only
the raw Score half of each judge's mark is used for analysis, consistent with every
other parser here.

A PDF upload never analyzes immediately — it always shows an
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
- **No full-archive scan** ("which judge looks most unusual across
  everything") — only targeted lookups (a specific judge/school/competitor)
  are wired up. A full scan is a much heavier computation across hundreds of
  events and is a better fit for a scheduled background job (Vercel Cron)
  than a live request; not attempted yet.

## Bulk import (new)

A dedicated bulk-upload path exists now, separate from the one-file-at-a-time
review flow: the "Bulk import to archive" card on the homepage. Select any
number of PDFs at once (no practical limit at the file picker) and the
browser automatically sends them to the server in small batches of 6,
sequentially, so no single request ever risks the 60-second function
timeout no matter how many files are selected in total.

**There's no human preview step in this path** — that's impractical at
hundreds of files. In its place, every file goes through an automated QA
gate before being saved:

- The PDF's own stated competitor count (printed on the page — confirmed
  present on all sample files from both formats) must match what was
  actually extracted, within a small tolerance
- No mark outside the 0-100 range
- No judge scoring every competitor identically (a strong signal of an
  extraction problem, not real judging)
- No genuine extraction warnings (informational notices about known,
  correctly-handled format quirks — like "did not dance this round" — are
  explicitly excluded from this check, so normal files aren't flagged for
  no reason)

A file that passes every check saves automatically. A file that fails any
check is listed at the end with the specific reason, for you to run through
the normal single-file upload (which still shows the full editable preview)
instead. Duplicate detection works exactly as in the single-file flow, and
correctly persists across separate batch requests — confirmed by testing 9
files (forcing 2 separate server calls) where files repeating an
already-saved competition were caught as duplicates even though they arrived
in a different HTTP request than the original.

**A real bug found and fixed while building this:** the existing multi-PDF
batch-upload limit (`MAX_FILES`, used by the interactive review flow) was
set to 12, but a real timing test showed 12 real PDFs take ~67 seconds to
parse — over the 60-second function limit. It's been lowered to 6, based on
measured timing with real files, not a guess.

## Single-event reviews now cross-reference the archive (new)

Uploading and reviewing one competition used to be completely blind to the
archive — a judge flagged across ten prior saved competitions would show as
"Clear" on an eleventh, with no indication the archive holds contrary
evidence. Every single-event report now cross-references each judge against
their archive history, if the database is configured.

**Deliberately two-tier for performance**, since a QuickFeis PDF can involve
up to 15 distinct judges across its 3 rotating panels, and this project has
already found two real timeout bugs from naive per-item processing:
- Every judge gets a cheap "N prior events on record" count (one fast query
  each).
- Only the judge already flagged as most unusual in the fresh single-event
  statistics gets the full archive-pattern check (fetching their complete
  history and re-running the same competitor/school pattern detection used
  by "Search the archive"). This bounds the expensive computation to at
  most one judge per round, regardless of how many total judges are in the
  file.

If a judge's archive history shows a real pattern, it appears both as a
small badge next to their name in the judge-agreement bars, and as a
dedicated "Archive History" section with the same level of detail as a
direct archive search. If the database isn't configured, or a query fails
for any reason, the report renders exactly as before with no archive
section — this was tested directly (no `DATABASE_URL` set at all) to
confirm it never breaks the core report.

**Worth knowing about the numbers that can appear here:** the archive
cross-check uses the same FDR-corrected, multi-observation statistics as
"Search the archive" and the multi-event watch list — which is a
deliberately more conservative test than the single-event permutation
check used for the fresh upload itself. A team/judge pattern that looks
notable within one event (say, 1-2% odds by chance) can legitimately show
as "no clear pattern" when tested against a judge's full archive, corrected
for every school they've worked with. That's not a bug or an inconsistency
between the two numbers — it's the same "multiple comparisons demand
stronger evidence" principle this whole project is built on, just visible
in one place instead of two separate reports.

### Toward a judge rating/directory system

This is the natural foundation for what was discussed as a longer-term
goal: a searchable directory of every judge in the archive, sortable by
event count and flag history, for quickly vetting judges before inviting
them to an event. Not built yet — this session's work only wires archive
awareness into the single-event report. The next step would be a dedicated
page reusing `judge_archive_badge()` (or a bulk variant of it) across every
judge in the archive at once, rather than one PDF's worth of judges at a
time.

## Archive search now suggests close matches (fix)

Reported live: searching the archive for "Aaron Crosby" returned "No history
found" even though the archive had "Aaron Crosbie" (one-letter difference,
"-y" vs "-ie"). The search was always exact-match by design — but with no
fallback, any spelling variance across years of differently-transcribed PDFs
would silently dead-end.

Fixed with `db.fuzzy_match_names()`, which deliberately does NOT use SQL
substring matching (`ILIKE`) — tested and confirmed that approach wouldn't
have caught this exact case, since "Aaron Crosby" isn't a substring of
"Aaron Crosbie". It uses Python's stdlib `difflib` for character-level
similarity instead, fetching the full distinct-name list for the relevant
column (judges/schools/competitors across a multi-year archive is
realistically in the hundreds, not enough to need a database fuzzy-search
extension) and ranking by similarity in Python.

A search with no exact match now returns up to 5 close-match suggestions as
clickable chips ("Did you mean: Aaron Crosbie") instead of a dead end.
Confirmed both paths through the real browser: a close match found (chip
shown, clicking it re-runs the search with the corrected name) and a
genuine no-match case (plain message, no suggestions box).

## Archive search: real bug found and fixed (float vs Decimal)

The improved error handling from the previous fix immediately paid off — it
surfaced the actual cause instead of a dead end: `TypeError: unsupported
operand type(s) for -: 'float' and 'decimal.Decimal'`, identical across all
three search kinds (judge/school/competitor), confirming a single shared
root cause rather than three separate bugs.

**Root cause:** Postgres's `NUMERIC` column type (used for `mark` in the
schema) comes back from psycopg2 as Python's `decimal.Decimal`, not `float`.
Every statistics function in this project assumes plain floats, and mixing
the two in arithmetic raises exactly this error. This affected every
archive read path uniformly, since they all reconstruct rounds through the
same function, `_pivot_long_to_wide()` in `db.py`.

**Fix:** cast the mark column to float once, at that single shared
reconstruction point, rather than downstream in every statistics function
that touches archive data. Verified directly with real `decimal.Decimal`
values (not just floats) through the full pipeline: the pivot itself, full
`competitor_watch`/`team_watch` statistics, `render_history_report`, and
the single-event archive cross-reference feature (`judge_archive_badge`) --
confirmed that one wasn't reported broken but shared the same underlying
function, so it needed the same verification.

Worth noting for next time: this bug existed in every archive read function
from the point they were first built, but was never caught locally because
every local/sandbox test used mocked Python data (plain floats or numbers
built directly in Python), never real values round-tripped through an
actual Postgres connection -- which is the only way `decimal.Decimal` shows
up. There isn't a live database reachable from the development sandbox, so
this category of bug can only really be caught on a real deploy. Worth
remembering as a class of thing to watch for again if a new archive
function is added later.
