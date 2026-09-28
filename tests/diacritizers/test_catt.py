import pytest

from arabic_mdd.diacritizers.catt import CattDiacritizer


class FakeCattModel:
    """Stands in for `catt_tashkeel.CATTEncoderOnly`/`CATTEncoderDecoder`.

    Pads its output with whitespace so the wrapper's stripping is exercised.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def do_tashkeel_batch(
        self, texts: list[str], batch_size: int = 16, verbose: bool = True
    ) -> list[str]:
        self.calls.append({"texts": list(texts), "batch_size": batch_size, "verbose": verbose})
        return [f"  {text}\u064e  " for text in texts]


def test_diacritize_batch_preserves_order_and_strips_output() -> None:
    catt = CattDiacritizer(variant="eo", model=FakeCattModel())

    assert catt.diacritize_batch(["ذهب", "عاد"]) == ["ذهب\u064e", "عاد\u064e"]


def test_diacritize_delegates_to_the_batch_path() -> None:
    model = FakeCattModel()
    catt = CattDiacritizer(variant="eo", model=model)

    assert catt.diacritize("ذهب") == "ذهب\u064e"
    assert model.calls[0]["texts"] == ["ذهب"]


def test_verbose_is_suppressed_and_batch_size_is_forwarded() -> None:
    model = FakeCattModel()
    catt = CattDiacritizer(variant="ed", model=model, batch_size=4)

    catt.diacritize_batch(["ذهب"])

    assert model.calls[0]["verbose"] is False
    assert model.calls[0]["batch_size"] == 4


def test_empty_batch_does_not_touch_the_model() -> None:
    model = FakeCattModel()
    catt = CattDiacritizer(variant="eo", model=model)

    assert catt.diacritize_batch([]) == []
    assert model.calls == []


@pytest.mark.parametrize(("variant", "expected"), [("eo", "catt-eo"), ("ed", "catt-ed")])
def test_both_published_variants_are_wrapped_and_distinctly_named(
    variant: str, expected: str
) -> None:
    catt = CattDiacritizer(variant=variant, model=FakeCattModel())

    assert catt.name == expected
    assert catt.provenance()["variant"] == variant


def test_unknown_variant_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown CATT variant"):
        CattDiacritizer(variant="encoder-only")


def test_construction_does_not_load_the_model() -> None:
    """The model archives are 72-86 MB downloads, so construction must stay free."""
    catt = CattDiacritizer(variant="ed")

    assert catt._model is None


def test_provenance_records_the_installed_package_version() -> None:
    provenance = CattDiacritizer(variant="eo", model=FakeCattModel()).provenance()

    assert provenance["tool"] == "catt"
    assert provenance["package"] == "catt-tashkeel"
    assert provenance["version"]
