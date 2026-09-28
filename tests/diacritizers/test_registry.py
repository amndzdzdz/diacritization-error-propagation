import pytest

from arabic_mdd.diacritizers import DIACRITIZERS, Diacritizer, build


def test_the_registry_holds_exactly_the_tool_set_the_plan_fixes() -> None:
    assert sorted(DIACRITIZERS) == ["catt-ed", "catt-eo", "mishkal", "shakkala"]


@pytest.mark.parametrize("name", sorted(DIACRITIZERS))
def test_every_registered_name_builds_and_reports_itself_under_that_name(name: str) -> None:
    diacritizer = build(name)

    assert isinstance(diacritizer, Diacritizer)
    assert diacritizer.name == name


@pytest.mark.parametrize("name", sorted(DIACRITIZERS))
def test_every_tool_reports_provenance(name: str) -> None:
    provenance = build(name).provenance()

    assert provenance["tool"]
    assert provenance["version"]


def test_unknown_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown diacritizer 'farasa'"):
        build("farasa")
