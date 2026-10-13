"""The Tanzil Qur'an text — the authoritative diacritized source for the Qur'anic arm.

Lifted out of `scripts/recover_quran_verses.py` once a second consumer appeared
(the Qur'anic DER measurement, week-05 task 3). Both consumers must read the
*same* edition from the *same* file, which is the whole reason this is one
module with a pinned digest rather than two copies of an XML walk.

**The edition is load-bearing.** `simple` and `uthmani` are not
interchangeable: on the clip→verse recovery the same matcher scores 98/98 on
`simple` and 16/98 on `uthmani`, because Uthmani writes alef wasla (U+0671),
which survives `strip_diacritics` and never matches a bare alef. Anything
measured against the Qur'an has to say which edition it used.

**Not vendored.** Tanzil ships a copyright block asking that it not be removed,
so the text stays a local, gitignored file and is pinned by sha256 here.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

#: The `simple` edition, NOT `uthmani` — see the module docstring.
#: `imlaei-simple` is a 404 on this endpoint.
TANZIL_URL = "https://tanzil.net/pub/download/index.php?quranType=simple&outType=xml&agree=true"
TANZIL_EDITION = "tanzil-simple-xml"

#: sha256 of the file `run/quranmb_verse_map.json` was built from, recorded in
#: that artifact's `edition.sha256`. A consumer that joins against the recovered
#: spans is only reading one text if its own input hashes to this.
EDITION_SHA256 = "d9b5bad94e4725191aa8da9a0404bd8e6447696ef05a71f9220ab546d7a8f47f"

#: 114 surahs, 6,236 ayahs — Hafs/Kufan numbering, with the basmala of every
#: surah but al-Fatiha excluded from the verse text (see `load_verses`).
N_VERSES = 6236


@dataclass(frozen=True)
class Verse:
    surah: int
    ayah: int
    text: str
    """Fully diacritized ayah text, exactly as the edition writes it."""


def load_verses(path: Path) -> list[Verse]:
    """Read every ayah of a Tanzil XML edition, in scriptural order.

    `bismillah` is a separate attribute on the first `<aya>` of a surah and is
    **not** part of the verse text; including it would insert four words no
    prompt reads and no reciter in QuranMB utters mid-verse.
    """
    root = ET.parse(path).getroot()
    out: list[Verse] = []
    for sura in root.iter("sura"):
        s = int(sura.get("index"))  # type: ignore[arg-type]
        for aya in sura.iter("aya"):
            out.append(Verse(s, int(aya.get("index")), aya.get("text", "")))  # type: ignore[arg-type]
    if not out:
        raise ValueError(f"{path}: no <aya> elements — wrong file, or a 404 page saved as XML?")
    return out


def file_sha256(path: Path) -> str:
    """Digest the edition file, for recording or for checking `EDITION_SHA256`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(dest: Path) -> None:
    """Fetch the pinned edition. Kept here so the URL has exactly one home."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TANZIL_URL, timeout=120) as response:  # noqa: S310
        dest.write_bytes(response.read())
