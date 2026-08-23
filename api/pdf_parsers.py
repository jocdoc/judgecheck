"""
Detects which of the known PDF formats an upload is, and routes it
to the right parser. If nothing matches, says so plainly instead
of guessing -- a wrong silent guess is worse than an honest "I don't
recognize this."
"""

import re

import pdfplumber

from .quickfeis_parser import parse_quickfeis_pdf
from .feisresults_parser import parse_feisresults_pdf
from .maro_parser import parse_maro_pdf


class UnrecognizedFormatError(Exception):
    pass


def detect_format(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page_text = pdf.pages[0].extract_text() or ""
        # QuickFeis: branded footer + "dancers competed" appears on every variant seen so
        # far (title text itself varies -- "FINAL RESULTS for:" in one real sample,
        # "Solo Championship Final Report" in another -- so don't key off that alone)
        if "QuickFeis" in first_page_text and "dancers competed" in first_page_text:
            return "quickfeis"
        # "Mid Atlantic Region Oireachtas" template: also feisresults.com-branded, but a
        # completely different per-competition layout (see maro_parser.py) -- this specific
        # phrase pattern doesn't appear on either other known format, confirmed before relying
        # on it as the sole signal
        if "Results for Competition" in first_page_text and "Round 1" in first_page_text:
            return "maro"
        # feisresults.com (original/Worlds-style): an "Adjudicators" cover page, RECALL/FINAL
        # MARKS tables follow
        if "Adjudicators" in first_page_text:
            return "feisresults"
        # also check page 2 in case of a different cover-page arrangement
        if len(pdf.pages) > 1:
            second_page_text = pdf.pages[1].extract_text() or ""
            if "RECALL" in second_page_text and "IP = Irish Points" in second_page_text:
                return "feisresults"
    return None


def extract_stated_competitor_count(pdf_path, fmt):
    """
    Both QuickFeis and the original feisresults.com layout print their own
    count of how many competitors took part, right on the page -- confirmed
    against real samples before relying on this (48/108/108 for QuickFeis's
    two title variants, 108 for feisresults.com). This is the strongest
    automated sanity check available: if what was extracted doesn't match
    what the PDF itself claims, something went wrong in a way worth a
    human's attention, no matter how confident the parser otherwise looked.
    The Mid Atlantic Region Oireachtas layout doesn't print an equivalent
    total anywhere -- returns None for it (and for anything else the figure
    can't be found for), and the caller skips this specific check rather
    than fail on it.
    """
    with pdfplumber.open(pdf_path) as pdf:
        if fmt == "quickfeis":
            text = pdf.pages[0].extract_text() or ""
            m = re.search(r"(\d+)\s+dancers competed", text)
            return int(m.group(1)) if m else None
        elif fmt == "feisresults":
            for page in pdf.pages[:2]:
                text = page.extract_text() or ""
                m = re.search(r"Number danced\s*=\s*(\d+)", text)
                if m:
                    return int(m.group(1))
            return None
    return None


def parse_results_pdf(pdf_path):
    """
    Returns (rounds, format_name, warnings). Raises UnrecognizedFormatError
    if the PDF doesn't match any known format -- the caller should show
    the person a clear message and point them at the Excel template rather
    than attempt a guess.
    """
    fmt = detect_format(pdf_path)
    if fmt == "quickfeis":
        rounds, competitor_index, warnings = parse_quickfeis_pdf(pdf_path)
        return rounds, "QuickFeis", warnings
    elif fmt == "feisresults":
        rounds, warnings = parse_feisresults_pdf(pdf_path)
        return rounds, "feisresults.com", warnings
    elif fmt == "maro":
        rounds, warnings = parse_maro_pdf(pdf_path)
        return rounds, "feisresults.com (Mid Atlantic Region)", warnings
    else:
        raise UnrecognizedFormatError(
            "This PDF doesn't match a results format JudgeCheck currently recognizes "
            "(QuickFeis or feisresults.com). You can still run an analysis by filling in "
            "the downloadable Excel/CSV template with the same numbers."
        )
