from pathlib import Path

import pytest

from arabic_mdd.data.tanzil import EDITION_SHA256, file_sha256, load_verses

#: Shaped like the real edition: `bismillah` is an attribute on the first aya
#: of a surah, never part of `text`.
XML = """<?xml version="1.0" encoding="UTF-8"?>
<quran>
  <sura index="1" name="الفاتحة">
    <aya index="1" text="بِسْمِ ٱللَّهِ" />
    <aya index="2" text="ٱلْحَمْدُ لِلَّهِ" />
  </sura>
  <sura index="2" name="البقرة">
    <aya index="1" text="الم" bismillah="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ" />
  </sura>
</quran>
"""


@pytest.fixture
def edition(tmp_path: Path) -> Path:
    path = tmp_path / "quran.xml"
    path.write_text(XML, encoding="utf-8")
    return path


def test_load_verses_reads_every_aya_in_scriptural_order(edition: Path) -> None:
    verses = load_verses(edition)
    assert [(v.surah, v.ayah) for v in verses] == [(1, 1), (1, 2), (2, 1)]


def test_bismillah_is_not_part_of_the_verse_text(edition: Path) -> None:
    """Including it would insert four words no prompt reads and no reciter utters."""
    assert load_verses(edition)[2].text == "الم"


def test_a_file_with_no_ayas_is_an_error_not_an_empty_corpus(tmp_path: Path) -> None:
    """A saved 404 page parses as XML; silently measuring on zero verses would not."""
    path = tmp_path / "not-the-quran.xml"
    path.write_text("<html><body>404</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="no <aya> elements"):
        load_verses(path)


def test_edition_sha256_is_a_sha256_digest() -> None:
    """The pin is what stops two consumers reading two different Qur'ans."""
    assert len(EDITION_SHA256) == 64
    assert set(EDITION_SHA256) <= set("0123456789abcdef")


def test_file_sha256_digests_bytes(edition: Path) -> None:
    import hashlib

    assert file_sha256(edition) == hashlib.sha256(edition.read_bytes()).hexdigest()
