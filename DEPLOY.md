# Deploying JudgeCheck to Vercel

Same GitHub → Vercel workflow as secondhandlegends. Any Python file inside
`api/` automatically becomes a live endpoint — Vercel sees `api/analyze.py`
and turns it into `yoursite.com/api/analyze`. Nothing is stored between
requests; each call is stateless.

## What's in this folder

| File | What it is |
|---|---|
| `index.html` | The whole website (upload, PDF preview/edit grid, report display) |
| `api/analyze.py` | The statistics engine + request routing (CSV/XLSX, PDF preview, confirmed analysis) |
| `api/pdf_parsers.py` | Detects which PDF format was uploaded and routes to the right parser |
| `api/quickfeis_parser.py` | Parser for QuickFeis-format results PDFs |
| `api/feisresults_parser.py` | Parser for feisresults.com-format results PDFs |
| `requirements.txt` | Tells Vercel which Python libraries to install (now includes `pdfplumber`) |
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
   - `index.html`, `requirements.txt`, `vercel.json` at the repo root
   - everything from `api/` into an `api/` folder in the repo
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

- Test both parsers against 3-4 more real PDFs each, ideally from different
  regions/organizers, to catch layout drift the two samples so far didn't show.
- The mitigations discussed earlier for circumvention (audit logging + report
  IDs, the judge-pair collusion check) aren't built yet — worth prioritizing
  before this handles reports people might rely on for real decisions.
