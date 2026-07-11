import pytest

from src.rag.chunking import chunk_text


def test_chunk_text_splits_long_text_into_overlapping_windows():
    text = "x" * 1000

    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert [len(c) for c in chunks] == [300, 300, 300, 250]


def test_chunk_text_returns_single_chunk_when_text_shorter_than_chunk_size():
    text = "short text"

    chunks = chunk_text(text, chunk_size=900, overlap=200)

    assert chunks == ["short text"]


def test_chunk_text_returns_empty_list_for_empty_text():
    assert chunk_text("", chunk_size=900, overlap=200) == []


def test_chunk_text_raises_when_overlap_is_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_default_arguments_are_900_and_200():
    text = "y" * 1000

    chunks = chunk_text(text)

    assert len(chunks) == 2
    assert len(chunks[0]) == 900
    assert len(chunks[1]) == 300
