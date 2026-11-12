"""Smoke tests that do not need models, network or a GPU.

Run with:  pytest
"""

from __future__ import annotations

import argparse

import pytest

from ocrmypdf_paddleocr import __version__
from ocrmypdf_paddleocr.plugin import (
    _MODEL_CACHE_SIZE,
    PaddleOCREngine,
    _resolve_inference_engine,
    add_options,
)


def make_parser_options(argv: list[str] | None = None):
    """Build an options namespace the same way OCRmyPDF does."""
    parser = argparse.ArgumentParser()
    add_options(parser)
    return parser.parse_args(argv or [])


def test_version_is_a_string():
    assert isinstance(__version__, str) and __version__


def test_language_map_covers_tesseract_codes():
    for code in ("eng", "chi_sim", "chi_tra", "jpn", "kor"):
        assert code in PaddleOCREngine.LANGUAGE_MAP


def test_engine_declares_languages_including_tesseract_codes():
    langs = PaddleOCREngine.languages(None)
    assert "chi_sim" in langs and "eng" in langs


def test_new_options_have_sane_defaults():
    opts = make_parser_options()
    assert opts.paddle_inference_engine == "auto"
    assert opts.paddle_det_limit_type == "max"
    assert opts.paddle_det_limit_side_len == 960
    assert opts.paddle_enable_mkldnn is True
    assert opts.paddle_engine == "classic"


def test_options_can_be_overridden():
    opts = make_parser_options(
        [
            "--paddle-inference-engine",
            "paddle",
            "--paddle-no-mkldnn",
            "--paddle-det-limit-type",
            "min",
            "--paddle-det-limit-side-len",
            "64",
        ]
    )
    assert opts.paddle_inference_engine == "paddle"
    assert opts.paddle_enable_mkldnn is False
    assert opts.paddle_det_limit_type == "min"
    assert opts.paddle_det_limit_side_len == 64


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("paddle", "paddle_static"),
        ("onnxruntime", "onnxruntime"),
    ],
)
def test_resolve_inference_engine_explicit(requested, expected):
    opts = argparse.Namespace(paddle_inference_engine=requested)
    assert _resolve_inference_engine(opts) == expected


def test_resolve_inference_engine_auto_picks_available_backend():
    opts = argparse.Namespace(paddle_inference_engine="auto")
    resolved = _resolve_inference_engine(opts)
    # onnxruntime if importable, otherwise paddle_static - never None for "auto"
    assert resolved in {"onnxruntime", "paddle_static"}


def test_cache_key_is_hashable_and_stable():
    opts = make_parser_options(["--paddle-inference-engine", "paddle"])
    first = PaddleOCREngine._paddle_cache_key(opts)
    second = PaddleOCREngine._paddle_cache_key(opts)
    assert first == second
    assert hash(first)  # must be usable as an lru_cache key


def test_cache_key_distinguishes_configurations():
    keys = {
        PaddleOCREngine._paddle_cache_key(make_parser_options()),
        PaddleOCREngine._paddle_cache_key(
            make_parser_options(["--paddle-det-limit-side-len", "640"])
        ),
        PaddleOCREngine._paddle_cache_key(make_parser_options(["--paddle-no-mkldnn"])),
        PaddleOCREngine._paddle_cache_key(
            make_parser_options(["--paddle-inference-engine", "paddle"])
        ),
        PaddleOCREngine._paddle_cache_key(make_parser_options(["--paddle-use-gpu"])),
    }
    assert len(keys) == 5


def test_model_cache_is_bounded_and_clearable():
    info = PaddleOCREngine._build_paddle_ocr.cache_info()
    assert info.maxsize == _MODEL_CACHE_SIZE
    PaddleOCREngine._build_paddle_ocr.cache_clear()
    assert PaddleOCREngine._build_paddle_ocr.cache_info().currsize == 0
    assert PaddleOCREngine._build_paddle_vl.cache_info().maxsize == _MODEL_CACHE_SIZE
