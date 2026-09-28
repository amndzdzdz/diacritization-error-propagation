from arabic_mdd.diacritizers.mishkal import MishkalDiacritizer


class FakeVocalizer:
    """Stands in for `mishkal.tashkeel.TashkeelClass`, reproducing its output quirks.

    Real Mishkal prepends a space and substitutes U+0001 for sentence-final
    punctuation *and the space that follows it*; both are replayed here from
    observed output.
    """

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []

    def tashkeel(self, text: str) -> str:
        self.calls.append(text)
        return self.outputs[text]


def test_leading_space_artifact_is_removed() -> None:
    mishkal = MishkalDiacritizer(vocalizer=FakeVocalizer({"عاد": " عَاد"}))

    assert mishkal.diacritize("عاد") == "عَاد"


def test_sentence_separator_becomes_a_space_so_word_count_survives() -> None:
    raw = " ذَهَبَ الْوَلَدُ\x01ثُمَّ عَادٌ"
    mishkal = MishkalDiacritizer(vocalizer=FakeVocalizer({"ذهب الولد. ثم عاد": raw}))

    output = mishkal.diacritize("ذهب الولد. ثم عاد")

    assert "\x01" not in output
    assert len(output.split()) == 4


def test_trailing_separator_does_not_leave_trailing_whitespace() -> None:
    mishkal = MishkalDiacritizer(vocalizer=FakeVocalizer({"عاد.": " عَادٍ\x01"}))

    assert mishkal.diacritize("عاد.") == "عَادٍ"


def test_diacritize_batch_preserves_order() -> None:
    vocalizer = FakeVocalizer({"ذهب": " ذَهَبَ", "عاد": " عَادٌ"})
    mishkal = MishkalDiacritizer(vocalizer=vocalizer)

    assert mishkal.diacritize_batch(["ذهب", "عاد"]) == ["ذَهَبَ", "عَادٌ"]
    assert vocalizer.calls == ["ذهب", "عاد"]


def test_construction_does_not_load_the_morphology_databases() -> None:
    assert MishkalDiacritizer()._vocalizer is None


def test_name_and_provenance() -> None:
    mishkal = MishkalDiacritizer(vocalizer=FakeVocalizer({}))

    assert mishkal.name == "mishkal"
    assert mishkal.provenance()["tool"] == "mishkal"
    assert mishkal.provenance()["version"]
