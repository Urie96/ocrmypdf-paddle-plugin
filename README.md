# ocrmypdf-paddleocr

A [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) engine plugin for
[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF): turn scanned PDFs into
searchable PDF/A with **PP-OCRv6** (or **PaddleOCR-VL**, a 0.9B VLM) instead of
Tesseract.

```bash
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim scanned.pdf searchable.pdf
```

Drop-in replacement for the Tesseract engine. It speaks the OCRmyPDF `OcrEngine`
plugin API, so you keep everything else OCRmyPDF does for you: rasterisation,
deskew/clean, page rotation, image optimisation, PDF/A conversion.

> This is a **fork** of [clefru/ocrmypdf-paddleocr](https://github.com/clefru/ocrmypdf-paddleocr)
> (MPL-2.0). It adds inference-backend selection, detection size tuning and Nix
> packaging. See [NOTICE](NOTICE) for the complete list of changes.

## Why

* **Chinese / Japanese / Korean** quality far above Tesseract, including
  traditional Chinese and vertical-ish layouts.
* **PP-OCRv6** is 34.5M parameters and reads **50 languages with one model**
  (zh, zh-Hant, en, ja + 46 Latin-script languages).
* Accurate invisible text layer: native word-level boxes
  (`return_word_box=True`) plus polygon-edge averaging for tight line bounds.
  Measured median offset versus an electronic text layer: **dx 1.2 pt / dy 2.1 pt
  (≈0.7 mm)**, 90th percentile 4.4 pt / 2.8 pt.
* Optional **PaddleOCR-VL** engine (`--paddle-engine vl`) for tables, formulas and
  reading order.

## Install

### pip

```bash
pip install 'ocrmypdf-paddleocr[onnxruntime]'
```

`onnxruntime` is optional but strongly recommended — see
[Performance](#performance). Without it the plugin falls back to `paddle_static`.

### Nix

```bash
nix run   .#default -- --version      # from a checkout
nix-shell ./nix/shell.nix             # shell with ocrmypdf + plugin + gs/qpdf/fonts
```

Or as a flake input:

```nix
{
  inputs.ocrmypdf-paddleocr.url = "github:USERNAME/ocrmypdf-paddle-plugin";
  # then
  environment.systemPackages = [ inputs.ocrmypdf-paddleocr.packages.${system}.default ];
}
```

## Usage

```bash
# Chinese simplified, defaults (PP-OCRv6 medium, onnxruntime if available)
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim in.pdf out.pdf

# English
ocrmypdf --plugin ocrmypdf_paddleocr -l eng in.pdf out.pdf

# Force re-OCR of a PDF that already has a (bad) text layer
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim --force-ocr in.pdf out.pdf

# PDFs with a digital signature refuse to be modified
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim --invalidate-digital-signatures in.pdf out.pdf

# Faster CPU run: one worker, smaller detection input
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim -j 1 \
  --paddle-det-limit-side-len 736 in.pdf out.pdf
```

Python API:

```python
import ocrmypdf

ocrmypdf.ocr(
    "in.pdf", "out.pdf",
    plugins=["ocrmypdf_paddleocr"],
    language="chi_sim",
)
```

## Options

Everything under the `PaddleOCR` group of `ocrmypdf --help`.

| Option | Default | Description |
|---|---|---|
| `--paddle-engine {classic,vl}` | `classic` | `classic` = PP-OCRv6 CNN det+rec, `vl` = PaddleOCR-VL 0.9B VLM (needs `[vl]` extra, much slower) |
| `--paddle-inference-engine {auto,paddle,onnxruntime}` | `auto` | classic pipeline backend. `auto` = onnxruntime when installed, else `paddle_static` |
| `--paddle-use-gpu` | off | `device=gpu` (needs a CUDA build of paddlepaddle) |
| `--paddle-no-mkldnn` | off (i.e. oneDNN on) | disable oneDNN/MKL-DNN, see [troubleshooting](#troubleshooting) |
| `--paddle-det-limit-type {max,min}` | `max` | `max` caps the longest side at `--paddle-det-limit-side-len` |
| `--paddle-det-limit-side-len N` | `960` | detection input size, multiple of 32. 1216 for dense small text, 640/736 for speed |
| `--paddle-no-angle-cls` | off | disable text-line orientation classification |
| `--paddle-det-model-dir DIR` | – | custom detection model directory |
| `--paddle-rec-model-dir DIR` | – | custom recognition model directory |
| `--paddle-cls-model-dir DIR` | – | custom orientation model directory |
| `--paddle-show-log` | off | show PaddleOCR's own logging |

`-l` is mapped to PaddleOCR languages (`chi_sim`→`ch`, `eng`→`en`,
`chi_tra`→`chinese_cht`, `jpn`→`japan`, `kor`→`korean`, …). Languages covered by
PP-OCRv6 resolve to the unified **PP-OCRv6 medium** model automatically;
`korean`, `ar`, `th`, `ru`, … fall back to PP-OCRv5.

## Performance

Measured on an x86_64 CPU (Xeon 8336C, 8 threads), 400 DPI page raster of a
dense A4 contract page (42 text lines):

| Backend | Time per page |
|---|---|
| `paddle_static`, oneDNN on | crashes on some builds (see below) |
| `paddle_static`, oneDNN off | **40.4 s** |
| `onnxruntime` | **3.5 s** |

That ~11x difference is why `--paddle-inference-engine auto` prefers
onnxruntime. End-to-end through OCRmyPDF (34-page born-digital contract forced
through OCR, `-j 1`, PDF/A output): **3 min 01 s**.

Detection size matters as much as the backend, because PaddleX ships
`limit_type=min / limit_side_len=64` which means *do not downscale*:

| Settings | 2 pages |
|---|---|
| `min`/`64` (PaddleX default), onnxruntime | 3 min 51 s |
| `max`/`960`, onnxruntime | **21.9 s** |

### File size

OCRmyPDF does not re-rasterise pages that already contain images — it keeps them
and only adds the invisible text layer. A 2-page 150 DPI scan went from
**519 KB to 469 KB (-9.6%)** after OCR; the text layer plus the subsetted embed
fonts cost about **5.5 KB per page**.

But if you force OCR on a *born-digital* PDF (vector content), OCRmyPDF
rasterises it at `VECTOR_PAGE_DPI = 400` and re-encodes every page, which can
inflate the file 8-12x. In that case the original text layer is already better
than any OCR — don't force it.

## Troubleshooting

**`NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not
support [pir::ArrayAttribute<pir::DoubleAttribute>] (onednn_instruction.cc:116)`**

The `paddle_static` oneDNN path of some paddlepaddle builds is broken (seen with
the nixpkgs 3.3.0 CPU wheel). Fix with either of:

```bash
--paddle-no-mkldnn                    # works, but paddle_static gets much slower
--paddle-inference-engine onnxruntime # recommended
```

`FLAGS_use_mkldnn=0` and `FLAGS_enable_pir_api=0` do **not** help: paddlex turns
oneDNN back on inside its runner.

**`DependencyError: OCR requires additional dependencies`**

`paddlex` declares `imagesize`, `pypdfium2`, `python-bidi`, `opencv-contrib-python`
(and more) under its `ocr-core`/`ocr` extras and checks them via
`importlib.metadata` at pipeline creation. `pip install 'paddleocr[all]'` covers
it; for a hand-rolled environment install those four.

**`No installed font has glyphs for the detected 'chi_sim' text`**

The invisible text layer needs a CJK font. Install e.g. Noto Sans CJK
(`noto-fonts-cjk-sans` on NixOS, `fonts-noto-cjk` on Debian). Otherwise the text
stays searchable and copyable but highlighting shows nothing.

**`DigitalSignatureError: Input PDF has a digital signature`**

Add `--invalidate-digital-signatures` (and accept that the signature is gone).

**`redo-ocr` refuses: "This PDF has a user fillable form"**

Use `--force-ocr` instead, and expect re-rasterisation.

**It is slow**

1. `-j 1` — every worker loads its own copy of the model.
2. `--paddle-det-limit-side-len 640` (or `736`).
3. `--paddle-inference-engine onnxruntime`.
4. First run downloads weights to `~/.paddlex/official_models`; later runs reuse
   them. Set `PADDLE_PDX_MODEL_SOURCE=BOS` to download from Baidu instead of
   HuggingFace.

## Models

Weights are downloaded on first use into `~/.paddlex/official_models/`:

| Engine | Models |
|---|---|
| `classic` | `PP-OCRv6_medium_det` + `PP-OCRv6_medium_rec` (or `*_onnx` for the onnxruntime backend) |
| `vl` | PaddleOCR-VL pipeline (`paddleocr[doc-parser]`) |

Point `--paddle-det-model-dir` / `--paddle-rec-model-dir` at a directory to use
your own fine-tuned weights.

## Development

```bash
pytest                                  # no models / network needed
nix-shell ./nix/shell.nix               # full runtime, ocrmypdf on PATH
```

## License

MPL-2.0 — same as OCRmyPDF and the upstream plugin. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the attribution and the list of modifications.
