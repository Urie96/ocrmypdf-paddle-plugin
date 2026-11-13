"""Smoke tests that do not need models, network or a GPU.

Run with:  pytest
"""

from __future__ import annotations

import argparse

import pytest

from ocrmypdf_rapidocr import __version__
from ocrmypdf_rapidocr.plugin import (
    _MODEL_CACHE_SIZE,
    LANGUAGE_MAP,
    RapidOCREngine,
    _estimated_words,
    _hocr_document,
    _poly_to_bbox,
    _resolve_lang,
    add_options,
)


def make_parser_options(argv: list[str] | None = None):
    """Build an options namespace the same way OCRmyPDF does."""
    parser = argparse.ArgumentParser()
    add_options(parser)
    return parser.parse_args(argv or [])


class FakeResult:
    """Minimal stand-in for rapidocr's RapidOCROutput."""

    def __init__(self, txts, scores, boxes, word_results=()):
        self.txts = txts
        self.scores = scores
        self.boxes = boxes
        self.word_results = word_results


def test_version_is_a_string():
    assert isinstance(__version__, str) and __version__


def test_language_map_is_limited_to_chi_sim_and_eng():
    assert LANGUAGE_MAP == {"chi_sim": "ch", "eng": "en"}


def test_engine_declares_only_the_supported_languages():
    assert RapidOCREngine.languages(None) == {"chi_sim", "eng"}


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        (["chi_sim"], "ch"),
        (["eng"], "en"),
        (["ENG"], "en"),
        (["deu"], "ch"),  # unsupported -> fallback, with a warning
        ([], "ch"),
        (["deu", "chi_sim"], "ch"),
    ],
)
def test_resolve_lang(languages, expected):
    opts = argparse.Namespace(languages=languages)
    assert _resolve_lang(opts) == expected


def test_new_options_have_sane_defaults():
    opts = make_parser_options()
    assert opts.rapid_model_type == "small"
    assert opts.rapid_use_gpu is False
    assert opts.rapid_use_angle_cls is True
    assert opts.rapid_det_limit_type == "max"
    assert opts.rapid_det_limit_side_len == 960
    assert opts.rapid_show_log is False


def test_options_can_be_overridden():
    opts = make_parser_options(
        [
            "--rapid-model-type",
            "medium",
            "--rapid-use-gpu",
            "--rapid-no-angle-cls",
            "--rapid-det-limit-type",
            "min",
            "--rapid-det-limit-side-len",
            "1216",
        ]
    )
    assert opts.rapid_model_type == "medium"
    assert opts.rapid_use_gpu is True
    assert opts.rapid_use_angle_cls is False
    assert opts.rapid_det_limit_type == "min"
    assert opts.rapid_det_limit_side_len == 1216


def test_module_cache_is_small():
    # Each instance keeps its own weights alive.
    assert 1 <= _MODEL_CACHE_SIZE <= 2


@pytest.mark.parametrize(
    ("poly", "expected"),
    [
        ([[10, 20], [110, 20], [110, 50], [10, 50]], (10, 20, 110, 50)),
        # Rotated quad: vertical bounds come from averaging the edges, so the
        # box is tighter than min/max would be.
        ([[10, 22], [110, 18], [110, 52], [10, 48]], (10, 20, 110, 50)),
        ([[0, 5], [10, 0], [20, 30]], (0, 0, 20, 30)),
    ],
)
def test_poly_to_bbox(poly, expected):
    assert _poly_to_bbox(poly) == expected


def test_estimated_words_covers_the_line():
    words = _estimated_words("hello brave world", (0, 0, 100, 20), 90)
    assert [w for w, _, _ in words] == ["hello", "brave", "world"]
    assert words[0][1][0] == 0
    assert words[-1][1][2] == 100  # last word snaps to the line end
    assert all(conf == 90 for _, _, conf in words)
    # No overlap, all boxes non-degenerate and inside the line box.
    for first, second in zip(words, words[1:]):
        assert first[1][2] <= second[1][0]
    assert all(0 <= x0 < x1 <= 100 for _, (x0, _, x1, _), _ in words)


def test_estimated_words_empty_text():
    assert _estimated_words("   ", (0, 0, 10, 10), 90) == []


def line_boxes(result):
    """Return the ocrx_word bboxes of an hOCR document."""
    import re

    lines, _ = _hocr_document(result, 200, 100, "eng", "test")
    return re.findall(r'class="ocrx_word".*?bbox ([^;"]+); x_wconf (\d+)', "\n".join(lines))


def test_hocr_uses_native_word_boxes():
    result = FakeResult(
        txts=("Hello world",),
        scores=(0.99,),
        boxes=[[[10, 10], [110, 10], [110, 40], [10, 40]]],
        word_results=(
            (
                ("Hello", 0.99, [[10, 10], [60, 10], [60, 40], [10, 40]]),
                ("world", 0.98, [[65, 10], [110, 10], [110, 40], [65, 40]]),
            ),
        ),
    )
    assert line_boxes(result) == [("10 10 60 40", "99"), ("65 10 110 40", "98")]


def test_hocr_falls_back_to_estimated_word_boxes():
    result = FakeResult(
        txts=("Hello world",),
        scores=(0.5,),
        boxes=[[[10, 10], [110, 10], [110, 40], [10, 40]]],
    )
    boxes = line_boxes(result)
    assert [b.split()[0] for b, _ in boxes] == ["10", "64"]
    assert boxes[-1][0].endswith("110 40")
    assert all(conf == "50" for _, conf in boxes)


def test_hocr_line_bbox_and_metadata():
    result = FakeResult(
        txts=("Hello world",),
        scores=(0.75,),
        boxes=[[[10, 10], [110, 10], [110, 40], [10, 40]]],
    )
    lines, text_lines = _hocr_document(result, 200, 100, "eng", "RapidOCR 3.9.2")
    document = "\n".join(lines)
    assert text_lines == ["Hello world"]
    assert 'bbox 0 0 200 100' in document  # ocr_page
    assert 'bbox 10 10 110 40; baseline 0 0; x_wconf 75' in document  # ocr_line
    assert 'content="RapidOCR 3.9.2"' in document
    assert 'lang="eng"' in document


def test_hocr_skips_empty_lines_and_escapes_text():
    result = FakeResult(
        txts=("", "a < b & c"),
        scores=(0.9, 0.9),
        boxes=[
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[0, 20], [100, 20], [100, 40], [0, 40]],
        ],
    )
    lines, text_lines = _hocr_document(result, 200, 100, "chi_sim", "t")
    document = "\n".join(lines)
    assert text_lines == ["a < b & c"]
    for escaped in ("&lt;", "&amp;"):
        assert escaped in document
    assert "a < b & c" not in document


def test_hocr_handles_no_results():
    result = FakeResult(txts=None, scores=None, boxes=None)
    lines, text_lines = _hocr_document(result, 200, 100, "eng", "t")
    assert text_lines == []
    assert lines[-1] == "</html>"


class AmbiguousArray(list):
    """Mimics a numpy array: ``if array:`` raises instead of returning a bool."""

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous")


def test_hocr_accepts_numpy_like_arrays():
    result = FakeResult(
        txts=("Hello world",),
        scores=(0.9,),
        boxes=AmbiguousArray([[[0, 0], [100, 0], [100, 20], [0, 20]]]),
        word_results=(),
    )
    lines, text_lines = _hocr_document(result, 200, 100, "eng", "t")
    assert text_lines == ["Hello world"]
