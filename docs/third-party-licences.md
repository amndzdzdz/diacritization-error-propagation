# Third-party licences

Code and data this project vendors or redistributes, and the terms attached.
Add a row here *in the same commit* that vendors anything new.

---

## Vendored code

### `src/arabic_mdd/data/_halabi_phonetiser.py`

| | |
|---|---|
| Upstream | [`nawarhalabi/Arabic-Phonetiser`](https://github.com/nawarhalabi/Arabic-Phonetiser), `phonetise-Arabic.py` |
| Commit | `e75e06bdb8069153251adc08fb81f5bdd31ab953` (2016-03-11) |
| Author | Nawar Halabi |
| Paper | Halabi & Wald, "Phonetic Inventory for an Arabic Speech Corpus", LREC 2016 — <https://aclanthology.org/L16-1116/> |
| Licence | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| Modifications | Python 2 → 3 (`/` → `//`); stress-assignment pass removed; debug `print`s and `__main__` block removed. Enumerated in the file's header. |

**Why vendored rather than reimplemented.** This is the phonetiser the
Iqra'Eval organizers name in their own papers ("phonetization via the Halabi
MSA phonetizer", IQRA 2026) and the origin of the 68-token `sws_arabic.txt`
inventory the baseline's CTC head is trained against. The official
`Iqra-Eval/interspeech_IqraEval` repo ships no phonetizer, so there is no
first-party artifact to use instead. Same reasoning as using S3PRL verbatim
for the week 1–3 baseline gate: a reimplementation's disagreements with the
released references would be ambiguous between "our bug" and "their data",
which is exactly the distinction this project exists to make.

#### ⚠ Open: the NonCommercial term

CC BY-NC 4.0 permits academic use and redistribution **with attribution**, and
this file is the only NC-encumbered thing in the tree. Two consequences worth
settling before the paper's artifact release (plan §5, contribution 5;
`docs/msa-arm.md` §7):

1. The repository as a whole cannot be released under a permissive licence
   (MIT/Apache) while this file is in it. Either the release carries a
   NonCommercial restriction, or this file is moved to an optional,
   separately-licensed component that users fetch from upstream themselves.
2. NC is *not* OSI-open and is incompatible with GPL, so downstream reuse is
   narrower than "open source" implies. Do not describe the release as open
   source without qualifying this.

Neither blocks week 4. Both block the release promise.

Tracked with the existing open licence item for `QuranMB.v2` /
`Iqra_Extra_IS26` (open since week 2, `insights/week-04.md`).

---

## Redistributed data

None yet. `scripts/baseline_reproduction/vocab/sws_arabic.txt` is a 68-line
token list from the Iqra'Eval baseline — a bare vocabulary, not a
copyrightable work, but it is covered by whatever terms the
`QuranMB.v2` / `Iqra_Extra_IS26` question resolves to.
