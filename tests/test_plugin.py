"""Smoke tests that do not need models, network or a GPU.

Run with:  pytest
"""

from __future__ import annotations

import argparse

import pytest

from ocrmypdf_paddleocr import __version__
from ocrmypdf_paddleocr.plugin import (
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
