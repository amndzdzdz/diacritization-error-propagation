from arabic_mdd.data.iqra_train import TrainingExample, parse_iqra_train_rows, parse_iqra_tts_rows


def test_parse_iqra_train_rows() -> None:
    rows = [
        {
            "id": "u1",
            "phoneme_ref": "b a t",
            "sentence": "بات",
            "tashkeel_sentence": "بَاتَ",
            "phoneme_aug": "b u t",
        },
        {
            "id": "u2",
            "phoneme_ref": "q u l",
            "sentence": "قل",
            "tashkeel_sentence": "قُلْ",
            "phoneme_aug": "q u l",
        },
    ]

    examples = parse_iqra_train_rows(rows)

    assert examples == [
        TrainingExample(
            id="u1", canonical=["b", "a", "t"], sentence="بات", tashkeel_sentence="بَاتَ"
        ),
        TrainingExample(id="u2", canonical=["q", "u", "l"], sentence="قل", tashkeel_sentence="قُلْ"),
    ]


def test_parse_iqra_tts_rows_leaves_tashkeel_empty() -> None:
    """`Iqra_TTS` ships no vowelized column, so it cannot serve the round-trip."""
    rows = [{"sentence_ref": "بات", "phoneme_ref": "b a t"}]

    assert parse_iqra_tts_rows(rows)[0].tashkeel_sentence == ""


def test_parse_iqra_tts_rows_synthesizes_ids_from_position() -> None:
    rows = [
        {
            "sentence_ref": "بات",
            "sentence_aug": "بات",
            "phoneme_ref": "b a t",
            "phoneme_mis": "b a t",
        },
        {
            "sentence_ref": "قل",
            "sentence_aug": "قل",
            "phoneme_ref": "q u l",
            "phoneme_mis": "q u l",
        },
    ]

    examples = parse_iqra_tts_rows(rows)

    assert [e.id for e in examples] == ["iqra_tts_0", "iqra_tts_1"]
    assert examples[0].canonical == ["b", "a", "t"]
    assert examples[0].sentence == "بات"
