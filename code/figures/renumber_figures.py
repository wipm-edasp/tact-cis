# -*- coding: utf-8 -*-
"""Renumber figures so first-citation order is 1-7: architecture 5 -> 7, heatmap 6 -> 5, case 7 -> 6."""
import docx
from pathlib import Path

BASE = Path(r"E:\Users\Dell\Documents\OpenScience\药物i\projects\tact\manuscript_merged")
DOCX = BASE / "manuscript.docx"

TEXT_EDITS = [
    ("typed-contact network (Figure 5).", "typed-contact network (Figure 7)."),
    ("The class-specific pattern is the result (Figure 6):", "The class-specific pattern is the result (Figure 5):"),
    ("for human Hsp90 [52,53] (Figure 7).", "for human Hsp90 [52,53] (Figure 6)."),
    ("Figure 5. Base model architecture", "Figure 7. Base model architecture"),
    ("Figure 6. Per-type coverage heatmap", "Figure 5. Per-type coverage heatmap"),
    ("Figure 7. Chemical case", "Figure 6. Chemical case"),
]


def replace_once(p, old, new):
    full = p.text
    if old not in full:
        return 0
    i, j = full.index(old), full.index(old) + len(old)
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
for old, new in TEXT_EDITS:
    cnt = sum(replace_once(p, old, new) for p in d.paragraphs)
    print("%d x %s" % (cnt, old[:55]))

# move the architecture legend to the end of the legend block
paras = d.paragraphs
arch = next(p for p in paras if p.text.strip().startswith("Figure 7. Base model architecture"))
case = next(p for p in paras if p.text.strip().startswith("Figure 6. Chemical case"))
arch._p.addnext(case._p)   # architecture right after case -> order ... 5, 6, 7
d.save(DOCX)

# verify
d2 = docx.Document(DOCX)
legends = [p.text.strip()[:60] for p in d2.paragraphs if p.text.strip().startswith("Figure ") and "|" not in p.text]
print("\nlegend order:")
for l in legends:
    if l.startswith("Figure "):
        print("  ", l)
