"""
Detects which of the two known PDF formats an upload is, and routes it
to the right parser. If neither pattern matches, says so plainly instead
of guessing -- a wrong silent guess is worse than an honest "I don't
recognize this."
"""

import re

import pdfplumber

from .quickfeis_parser import parse_quickfeis_pdf
from .feisresults_parser import parse_feisresults_pdf


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
        # feisresults.com: an "Adjudicators" cover page, RECALL/FINAL MARKS tables follow
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
    Both known formats print their own count of how many competitors took
    part, right on the page -- confirmed against all 4 real samples before
    relying on this (48/108/108 for QuickFeis's two title variants,
    108 for feisresults.com). This is the strongest automated sanity check
    available: if what we extracted doesn't match what the PDF itself
    claims, something went wrong in a way worth a human's attention, no
    matter how confident the parser otherwise looked. Returns None if the
    figure can't be found (a genuinely different layout), in which case
    the caller should skip this specific check rather than fail on it.
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
    if the PDF doesn't match either known format -- the caller should show
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
    else:
        raise UnrecognizedFormatError(
            "This PDF doesn't match a results format JudgeCheck currently recognizes "
            "(QuickFeis or feisresults.com). You can still run an analysis by filling in "
            "the downloadable Excel/CSV template with the same numbers."
        )
