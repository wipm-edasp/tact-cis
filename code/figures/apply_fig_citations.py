# -*- coding: utf-8 -*-
"""Add missing figure/table citations to manuscript.docx (text-only; no images embedded)."""
import docx
from pathlib import Path

BASE = Path(r"E:\Users\Dell\Documents\OpenScience\药物i\projects\tact\manuscript_merged")
DOCX = BASE / "manuscript.docx"

EDITS = [
    ("We used 750 crystal complexes across 29 targets, each with",
     "We used 750 crystal complexes across 29 targets (Figure 1), each with"),
    ("types per ligand atom; at 80%",
     "types per ligand atom (Figure 3); at 80%"),
    ("and 0.787 strict (overall 0.868, size 0.672).",
     "and 0.787 strict (overall 0.868, size 0.672; Figure 4)."),
    ("The class-specific pattern is the result:",
     "The class-specific pattern is the result (Figure 6):"),
    ("reported for human Hsp90 [52,53].",
     "reported for human Hsp90 [52,53] (Figure 7)."),
    ("capability-gated typed-contact network.",
     "capability-gated typed-contact network (Figure 5)."),
    ("set size (0.709 to 0.711) are almost unchanged.",
     "set size (0.709 to 0.711) are almost unchanged (Table 1)."),
    ("Water shows the opposite direction:",
     "Water shows the opposite direction (Table 2):"),
]


def replace_once(p, old, new):
    full = p.text
    if old not in full:
        return 0
    i = full.index(old)
    j = i + len(old)
    spans, pos = [], 0
    for r in p.runs:
        t = r.text or ""
        spans.append((r, pos, pos + len(t)))
        pos += len(t)
    sr = next((s for s in spans if s[1] <= i < s[2]), None)
    er = next((s for s in spans if s[1] < j <= s[2]), None)
    if sr is None or er is None:
        return 0
    if sr is er:
        r0, s0, _ = sr
        r0.text = (r0.text or "")[: i - s0] + new + (r0.text or "")[j - s0:]
        return 1
    r0, s0, _ = sr
    re_, se0, _ = er
    r0.text = (r0.text or "")[: i - s0] + new
    re_.text = (re_.text or "")[j - se0:]
    i0, i1 = spans.index(sr), spans.index(er)
    for r, _, _ in spans[i0 + 1: i1]:
        r.text = ""
    return 1


d = docx.Document(DOCX)
for old, new in EDITS:
    cnt = sum(replace_once(p, old, new) for p in d.paragraphs)
    print("%d x %s" % (cnt, old[:50]))
d.save(DOCX)
