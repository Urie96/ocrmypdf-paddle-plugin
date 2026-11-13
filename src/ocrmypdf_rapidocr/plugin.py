"""RapidOCR (PP-OCRv6) engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

try:
    from rapidocr import ModelType, RapidOCR
except ImportError:  # pragma: no cover - reported by check_options()
    RapidOCR = None
    ModelType = None

log = logging.getLogger(__name__)

# Number of distinct RapidOCR instances kept alive per process.  OCRmyPDF calls
# generate_hocr() once per page, so without the cache every page would pay the
# model load again (~0.15 s for PP-OCRv6 small, ~0.5 s for medium).  Each cached
# instance holds its own weights (31 MB for small, 100+ MB for medium), so keep
# the bound small.
_MODEL_CACHE_SIZE = 2

#: OCRmyPDF/Tesseract language code -> RapidOCR ``LangRec``/``LangDet`` value.
#:
#: PP-OCRv6 ships a single multilingual det+rec model, so both entries resolve
#: to the same weights; the language code only selects the character set that
#: RapidOCR uses for visualisation and RTL ordering.
LANGUAGE_MAP = {
    "chi_sim": "ch",
    "eng": "en",
}

#: Used when OCRmyPDF asks for a language this plugin does not handle.
_FALLBACK_LANG = "ch"

_MODEL_TYPES = {
    "tiny": "TINY",
    "small": "SMALL",
    "medium": "MEDIUM",
}

_HTML_ESCAPE = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def _escape(text: str) -> str:
    """Escape the three characters that matter inside hOCR text nodes."""
    return text.translate(_HTML_ESCAPE)


@hookimpl
def add_options(parser):
    """Add RapidOCR-specific options to the argument parser."""
    rapid = parser.add_argument_group(
        "RapidOCR",
        "Options for the RapidOCR (PP-OCRv6) engine"
    )
    rapid.add_argument(
        '--rapid-model-type',
        choices=['tiny', 'small', 'medium'],
        default='small',
        dest='rapid_model_type',
        help=(
            'PP-OCRv6 model size: "tiny" (fastest), "small" (default, the '
            'weights bundled with the rapidocr package) or "medium" (most '
            'accurate, needs the medium weights on disk - see '
            '--rapid-det-model-path/--rapid-rec-model-path)'
        ),
    )
    rapid.add_argument(
        '--rapid-use-gpu',
        action='store_true',
        help=(
            'Use the CUDA execution provider of onnxruntime (requires an '
            'onnxruntime build with CUDA support)'
        ),
    )
    rapid.add_argument(
        '--rapid-no-angle-cls',
        action='store_false',
        dest='rapid_use_angle_cls',
        default=True,
        help='Disable text-line orientation classification',
    )
    rapid.add_argument(
        '--rapid-det-limit-type',
        choices=['max', 'min'],
        default='max',
        dest='rapid_det_limit_type',
        help=(
            'How to limit the detection input size: "max" means "do not '
            'upscale, downscale large pages" (RapidOCR then picks 960/1500/2000 '
            'based on the page size), "min" guarantees the shortest side is at '
            'least --rapid-det-limit-side-len, i.e. it upsamples small pages. '
            '(default: max)'
        ),
    )
    rapid.add_argument(
        '--rapid-det-limit-side-len',
        type=int,
        default=960,
        metavar='N',
        dest='rapid_det_limit_side_len',
        help=(
            'Detection input side length in pixels, normally a multiple of 32. '
            'Only honoured with --rapid-det-limit-type min; raise it to 1216 for '
            'dense small text, lower it to 640/736 for speed. (default: 960)'
        ),
    )
    rapid.add_argument(
        '--rapid-det-model-path',
        metavar='FILE',
        dest='rapid_det_model_path',
        help='Path to a custom text detection .onnx model',
    )
    rapid.add_argument(
        '--rapid-rec-model-path',
        metavar='FILE',
        dest='rapid_rec_model_path',
        help='Path to a custom text recognition .onnx model',
    )
    rapid.add_argument(
        '--rapid-cls-model-path',
        metavar='FILE',
        dest='rapid_cls_model_path',
        help='Path to a custom text orientation classification .onnx model',
    )
    rapid.add_argument(
        '--rapid-show-log',
        action='store_true',
        help='Show RapidOCR internal logging',
    )


@hookimpl
def check_options(options):
    """Validate RapidOCR options."""
    if RapidOCR is None:
        from ocrmypdf.exceptions import MissingDependencyError
        raise MissingDependencyError(
            "RapidOCR is not installed. "
            "Install it with: pip install ocrmypdf-rapidocr[onnxruntime]"
        )
    for name, path in (
        ('--rapid-det-model-path', getattr(options, 'rapid_det_model_path', None)),
        ('--rapid-rec-model-path', getattr(options, 'rapid_rec_model_path', None)),
        ('--rapid-cls-model-path', getattr(options, 'rapid_cls_model_path', None)),
    ):
        if path and not Path(path).is_file():
            from ocrmypdf.exceptions import BadArgsError
            raise BadArgsError(f"{name} {path} is not a file")


def _requested_languages(options):
    """Return the languages OCRmyPDF was asked to OCR, lower-cased."""
    return [lang.lower() for lang in (getattr(options, 'languages', None) or [])]


def _resolve_lang(options):
    """Map the OCRmyPDF language selection onto a RapidOCR language."""
    requested = _requested_languages(options)
    for lang in requested:
        mapped = LANGUAGE_MAP.get(lang)
        if mapped:
            return mapped
    if requested:
        log.warning(
            "RapidOCR plugin: language %r is not supported (only %s), "
            "falling back to %s",
            requested[0], ", ".join(sorted(LANGUAGE_MAP)), _FALLBACK_LANG,
        )
    return _FALLBACK_LANG


def _cache_key(options):
    """Hashable identity of a RapidOCR configuration.

    argparse.Namespace is not hashable and almost every field is irrelevant to
    the pipeline, so only the settings that change what gets loaded are kept.
    """
    return (
        _resolve_lang(options),
        getattr(options, 'rapid_model_type', 'small'),
        bool(getattr(options, 'rapid_use_gpu', False)),
        bool(getattr(options, 'rapid_use_angle_cls', True)),
        getattr(options, 'rapid_det_limit_type', 'max'),
        int(getattr(options, 'rapid_det_limit_side_len', 960)),
        getattr(options, 'rapid_det_model_path', None),
        getattr(options, 'rapid_rec_model_path', None),
        getattr(options, 'rapid_cls_model_path', None),
        bool(getattr(options, 'rapid_show_log', False)),
    )


def _get_rapid_ocr(options):
    """Return the cached RapidOCR pipeline for this configuration."""
    # OCRmyPDF's Tesseract plugin sets OMP_THREAD_LIMIT to limit Tesseract
    # threading.  That leaks into every plugin in the process (paperless sets it
    # on every parse() call), so drop it before building the onnxruntime session:
    # with an OpenMP-enabled onnxruntime it would clamp inference to one thread.
    saved_omp_limit = os.environ.pop('OMP_THREAD_LIMIT', None)
    if saved_omp_limit:
        log.debug("Unset OMP_THREAD_LIMIT=%s set by the Tesseract plugin", saved_omp_limit)

    key = _cache_key(options)
    hits_before = _build_rapid_ocr.cache_info().hits
    engine = _build_rapid_ocr(key)
    log.debug(
        "%s RapidOCR instance for lang=%s model=%s det_limit=%s/%s",
        "Reusing cached" if _build_rapid_ocr.cache_info().hits > hits_before else "Created",
        key[0], key[1], key[4], key[5],
    )
    return engine


@lru_cache(maxsize=_MODEL_CACHE_SIZE)
def _build_rapid_ocr(key):
    """Actually construct RapidOCR.  Cached - see _get_rapid_ocr()."""
    (
        lang,
        model_type,
        use_gpu,
        use_angle_cls,
        det_limit_type,
        det_limit_side_len,
        det_model_path,
        rec_model_path,
        cls_model_path,
        show_log,
    ) = key

    # RapidOCR only accepts enum members for model_type (a plain string raises
    # "The value of Det.model_type must be Enum Type").
    enum_model_type = getattr(ModelType, _MODEL_TYPES[model_type])

    params = {
        'Global.log_level': 'info' if show_log else 'warning',
        'Global.use_cls': use_angle_cls,
        'Det.lang_type': lang,
        'Rec.lang_type': lang,
        'Det.model_type': enum_model_type,
        'Rec.model_type': enum_model_type,
        'Det.limit_type': det_limit_type,
        'Det.limit_side_len': det_limit_side_len,
        # CUDA execution provider; a CPU-only onnxruntime raises at session
        # creation, which is the error message the user needs to see.
        'EngineConfig.onnxruntime.use_cuda': use_gpu,
    }
    if det_model_path:
        params['Det.model_path'] = det_model_path
    if rec_model_path:
        params['Rec.model_path'] = rec_model_path
    if cls_model_path:
        params['Cls.model_path'] = cls_model_path

    log.debug("Creating RapidOCR with params: %s", params)
    try:
        return RapidOCR(params=params)
    except PermissionError as exc:
        raise RuntimeError(
            "RapidOCR could not write its model files. In an immutable "
            "installation (Nix, read-only container image) RapidOCR may only use "
            "models that are already on disk: keep --rapid-model-type small "
            "(the size bundled with the rapidocr package) or point "
            "--rapid-det-model-path / --rapid-rec-model-path at .onnx files. "
            f"Original error: {exc}"
        ) from exc


def _poly_to_bbox(poly):
    """Convert a 4-point polygon (TL,TR,BR,BL) to a tight (x0, y0, x1, y1)."""
    points = [(float(p[0]), float(p[1])) for p in poly]
    x0 = int(min(p[0] for p in points))
    x1 = int(max(p[0] for p in points))
    if len(points) == 4:
        # Average the top edge (TL+TR) and the bottom edge (BR+BL); on rotated
        # lines this is tighter than min/max and closer to what a text layer
        # would use.
        y0 = int((points[0][1] + points[1][1]) / 2)
        y1 = int((points[2][1] + points[3][1]) / 2)
    else:
        y0 = int(min(p[1] for p in points))
        y1 = int(max(p[1] for p in points))
    return x0, y0, x1, y1


def _estimated_words(text, bbox, conf):
    """Fallback word boxes: split the line box proportionally by char count.

    Used only when RapidOCR did not return word-level boxes for a line (e.g.
    return_word_box found nothing to split on).  Characters and inter-word
    spaces share the line width in proportion to their count, so the boxes tile
    the line without gaps or overlaps.
    """
    x0, y0, x1, y1 = bbox
    words = text.split()
    if not words:
        return []

    line_width = max(x1 - x0, 1)
    total_chars = sum(len(word) for word in words)
    num_spaces = len(words) - 1
    unit = line_width / max(total_chars + num_spaces, 1)

    result = []
    current_x = x0
    for index, word in enumerate(words):
        word_width = max(int(round(len(word) * unit)), 1)
        word_x1 = x1 if index == len(words) - 1 else min(current_x + word_width, x1)
        result.append((word, (current_x, y0, word_x1, y1), conf))
        current_x = word_x1 + int(round(unit))
    return result


def _attr(result, name):
    """Read an attribute without triggering numpy's ambiguous truth value."""
    value = getattr(result, name, None)
    return () if value is None else value


def _line_words(result, index, text, bbox, conf):
    """Word boxes for line ``index``: native ones if available, else estimated."""
    word_results = _attr(result, 'word_results')
    if index < len(word_results):
        native = []
        for entry in word_results[index] or ():
            try:
                word, score, poly = entry
            except (TypeError, ValueError):
                continue
            if not word or not word.strip() or poly is None:
                continue
            native.append((word, _poly_to_bbox(poly), int(float(score) * 100)))
        if native:
            return native
    return _estimated_words(text, bbox, conf)


def _hocr_document(result, width, height, hocr_lang, creator_tag):
    """Turn a RapidOCROutput into hOCR lines plus the plain text lines.

    Kept free of RapidOCR types (it only reads ``txts``/``scores``/``boxes``/
    ``word_results``) so it can be unit tested without models or network.
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
        '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
        '<head>',
        '<title></title>',
        '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
        f'<meta name="ocr-system" content="{_escape(creator_tag)}" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
        '</head>',
        '<body>',
        f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
    ]

    text_lines = []
    txts = _attr(result, 'txts')
    scores = _attr(result, 'scores')
    boxes = _attr(result, 'boxes')

    word_id = 1
    for index, text in enumerate(txts):
        if not text or not text.strip():
            continue

        box = boxes[index] if index < len(boxes) else ((0, 0), (width, 0), (width, height), (0, height))
        bbox = _poly_to_bbox(box)
        conf = int(float(scores[index]) * 100) if index < len(scores) else 0
        x0, y0, x1, y1 = bbox
        line_id = index + 1

        text_lines.append(text)
        lines.append(
            f'<div class="ocr_carea" id="carea_{line_id}" title="bbox {x0} {y0} {x1} {y1}">'
        )
        lines.append(
            f'<p class="ocr_par" id="par_{line_id}" lang="{hocr_lang}" '
            f'title="bbox {x0} {y0} {x1} {y1}">'
        )
        lines.append(
            f'<span class="ocr_line" id="line_{line_id}" '
            f'title="bbox {x0} {y0} {x1} {y1}; baseline 0 0; x_wconf {conf}">'
        )

        words = _line_words(result, index, text, bbox, conf)
        for position, (word, word_bbox, word_conf) in enumerate(words):
            wx0, wy0, wx1, wy1 = word_bbox
            lines.append(
                f'<span class="ocrx_word" id="word_{word_id}" '
                f'title="bbox {wx0} {wy0} {wx1} {wy1}; x_wconf {word_conf}">'
                f'{_escape(word)}</span>'
            )
            if position < len(words) - 1:
                lines.append(' ')
            word_id += 1

        lines.extend(['</span>', '</p>', '</div>'])

    lines.extend(['</div>', '</body>', '</html>'])
    return lines, text_lines


class RapidOCREngine(OcrEngine):
    """Implements OCR with RapidOCR (PP-OCRv6 + ONNX Runtime)."""

    LANGUAGE_MAP = LANGUAGE_MAP

    @staticmethod
    def version():
        """Return the RapidOCR version."""
        try:
            from importlib.metadata import version
            return version("rapidocr")
        except Exception:  # pragma: no cover - metadata always present in practice
            return "unknown"

    @staticmethod
    def creator_tag(options):
        """Return the creator tag to identify this software."""
        model_type = getattr(options, 'rapid_model_type', 'small')
        backend = "onnxruntime+cu" if getattr(options, 'rapid_use_gpu', False) else "onnxruntime"
        return f"RapidOCR {RapidOCREngine.version()} (PP-OCRv6 {model_type}, {backend})"

    def __str__(self):
        """Return name of OCR engine and version."""
        return f"RapidOCR {RapidOCREngine.version()} (PP-OCRv6)"

    @staticmethod
    def languages(options):
        """Return the set of OCRmyPDF language codes this plugin handles."""
        return set(LANGUAGE_MAP)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        """Get page orientation.

        Always neutral: OCRmyPDF rotates pages itself, and the per-textline
        classifier (--rapid-no-angle-cls to disable) only fixes 180° lines.
        """
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        """Get deskew angle - not provided by RapidOCR."""
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR output for an image."""
        log.debug(f"Running RapidOCR on {input_file}")

        engine = _get_rapid_ocr(options)
        with Image.open(input_file) as img:
            width, height = img.size
            dpi = img.info.get('dpi', (300, 300))
        log.debug(f"Input image: {width}x{height}, DPI: {dpi}")

        # return_word_box=True gives native word boxes; without it RapidOCR only
        # reports one box per detected text line.
        result = engine(str(input_file), return_word_box=True)

        lang = _resolve_lang(options)
        lang_map_reverse = {v: k for k, v in LANGUAGE_MAP.items()}
        hocr_lang = lang_map_reverse.get(lang, 'eng')

        lines, text_lines = _hocr_document(
            result, width, height, hocr_lang, RapidOCREngine.creator_tag(options)
        )
        output_hocr.write_text('\n'.join(lines), encoding='utf-8')
        output_text.write_text('\n'.join(text_lines), encoding='utf-8')
        log.debug(f"Generated hOCR with {len(text_lines)} text lines")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options):
        """Generate a text-only PDF from an image.

        RapidOCR has no PDF writer, so we go through hOCR and OCRmyPDF's
        HocrTransform.
        """
        log.debug(f"Generating PDF from {input_file}")

        output_hocr = output_pdf.with_suffix('.hocr')
        RapidOCREngine.generate_hocr(input_file, output_hocr, output_text, options)

        from ocrmypdf.hocrtransform import HocrTransform

        with Image.open(input_file) as img:
            dpi = img.info.get('dpi', (300, 300))[0]

        HocrTransform(hocr_filename=output_hocr, dpi=dpi).to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True,
        )


@hookimpl
def get_ocr_engine():
    """Register RapidOCR as an OCR engine."""
    return RapidOCREngine()
