# ocrmypdf-rapidocr

A [RapidOCR](https://github.com/RapidAI/RapidOCR) engine plugin for
[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF): turn scanned PDFs into
searchable PDF/A with **PP-OCRv6** instead of Tesseract.

```bash
ocrmypdf -l chi_sim scanned.pdf searchable.pdf
```

It speaks the OCRmyPDF `OcrEngine` plugin API, so you keep everything else
OCRmyPDF does for you: rasterisation, deskew/clean, page rotation, image
optimisation, PDF/A conversion. The plugin registers itself through the
`ocrmypdf` entry point, so there is **no `--plugin` flag to remember** — having
it installed in the same Python environment as `ocrmypdf` is enough.

> Fork of [clefru/ocrmypdf-paddleocr](https://github.com/clefru/ocrmypdf-paddleocr)
> (MPL-2.0), originally a PaddleOCR plugin, ported to RapidOCR and reduced to
> PP-OCRv6. See [NOTICE](NOTICE) for the full list of changes.

## Why

* **No paddlepaddle.** The old engine needed `paddleocr` + `paddlex` +
  `paddlepaddle`; on nixpkgs that chain also pulls `modelscope` (insecure) and
  `aistudio-sdk` (unfree), so the package could only be built with
  `NIXPKGS_ALLOW_INSECURE` **and** `NIXPKGS_ALLOW_UNFREE`, and four Python
  packages had to be compiled: 417 store paths, ≥3.2 GiB unpacked. The RapidOCR
  route is `rapidocr` + `onnxruntime` only — nothing needs an allow-flag, every
  path comes from the binary cache. The engine closure is 1.2 GiB; with
  `ocrmypdf` + opencv + ghostscript on top, the plugin package is 2.5 GiB
  (ocrmypdf and opencv dominate, not the engine).
* **Models ship with the package.** On nixpkgs, `rapidocr` bundles the PP-OCRv6
  *small* det/rec weights plus the PP-OCRv4 text-line cls model (~31 MiB), so a
  Nix build OCRs offline with no model download.
* **Chinese / English quality far above Tesseract**, including traditional
  Chinese.
* **Accurate invisible text layer**: native word-level boxes
  (`return_word_box=True`) that RapidOCR already maps back to the original image
  coordinates — no scaling bookkeeping, no token re-merging.

## Install

### pip

```bash
pip install 'ocrmypdf-rapidocr[onnxruntime]'
```

`onnxruntime` is what actually runs the models; the extra keeps RapidOCR's other
backends (openvino / torch / tensorrt) out of the dependency graph.

### Nix

```bash
nix build .#default                  # the plugin package
nix-shell ./nix/shell.nix            # shell with ocrmypdf + plugin + gs/qpdf/fonts
```

Or as a flake input:

```nix
{
  inputs.ocrmypdf-rapidocr.url = "github:Urie96/ocrmypdf-paddle-plugin";
  # then
  environment.systemPackages = [ inputs.ocrmypdf-rapidocr.packages.${system}.default ];
}
```

## Usage

```bash
# Chinese simplified
ocrmypdf -l chi_sim in.pdf out.pdf

# English
ocrmypdf -l eng in.pdf out.pdf

# Force re-OCR of a PDF that already has a (bad) text layer
ocrmypdf -l chi_sim --force-ocr in.pdf out.pdf

# PDFs with a digital signature refuse to be modified
ocrmypdf -l chi_sim --invalidate-digital-signatures in.pdf out.pdf

# One worker (each worker loads its own model copy) + smaller detection input
ocrmypdf -l chi_sim -j 1 --rapid-det-limit-side-len 736 in.pdf out.pdf
```

Python API — the engine is picked up automatically once the package is
installed:

```python
import ocrmypdf

ocrmypdf.ocr("in.pdf", "out.pdf", language="chi_sim")
```

## Languages

Only the two languages this fork cares about are wired up:

| OCRmyPDF `-l` | RapidOCR language |
|---|---|
| `chi_sim` | `ch` |
| `eng` | `en` |

PP-OCRv6 det/rec is a single multilingual model, so both entries load the same
weights; the language code only selects the character set RapidOCR reports. Any
other `-l` value logs a warning and falls back to `ch`.

## Options

Everything under the `RapidOCR` group of `ocrmypdf --help`.

| Option | Default | Description |
|---|---|---|
| `--rapid-model-type {tiny,small,medium}` | `small` | PP-OCRv6 model size. `small` is the size bundled with the `rapidocr` package, so it works with no downloads |
| `--rapid-use-gpu` | off | use onnxruntime's CUDA execution provider (needs a CUDA-enabled onnxruntime) |
| `--rapid-no-angle-cls` | off (i.e. cls on) | disable text-line orientation classification |
| `--rapid-det-limit-type {max,min}` | `max` | `max` = never upscale, downscale big pages (RapidOCR then picks 960/1500/2000 from the page size); `min` = guarantee the shortest side is at least `--rapid-det-limit-side-len` |
| `--rapid-det-limit-side-len N` | `960` | detection input size, multiple of 32. Only honoured with `--rapid-det-limit-type min`; 1216 for dense small text, 640/736 for speed |
| `--rapid-det-model-path FILE` | – | custom detection `.onnx` |
| `--rapid-rec-model-path FILE` | – | custom recognition `.onnx` |
| `--rapid-cls-model-path FILE` | – | custom orientation `.onnx` |
| `--rapid-show-log` | off | show RapidOCR's own logging |

With `--rapid-det-limit-type max`, RapidOCR ignores the side length and picks
960/1500/2000 by page size — that is upstream behaviour, and it is what keeps a
300 DPI page raster from being fed to the detector at full resolution. If you
need an exact cap, pass explicit model paths and pre-scale, or use `min`.

## Performance

x86_64 CPU (Xeon 8336C), PP-OCRv6 **small**, onnxruntime, dense synthetic A4 page
raster at 300 DPI (2480×3508, 40 text lines), first-run model load included:

| Settings | Time per page |
|---|---|
| `min`/736 (RapidOCR default) | 2.4 s |
| `max`/960 → actually 1500 | 2.2 s |
| `min`/960 | 2.1 s |
| `min`/1920 | 4.3 s |

End-to-end through OCRmyPDF on a single-page 300 DPI scan: **2.7 s** wall clock
(`-l chi_sim`, one page, PDF output including ghostscript/qpdf post-processing).

### Model caching

Instances are cached per process (`functools.lru_cache`, keyed on language,
model type, backend, detection size, device and model paths; at most
`_MODEL_CACHE_SIZE = 2`). OCRmyPDF builds the pipeline once **per page**, so
without the cache every page would pay the model load again. With `-j 1` the
model is loaded once per document.

## Models

| Model type | Where it comes from |
|---|---|
| `small` | bundled in the `rapidocr` package (nixpkgs: `PP-OCRv6_det_small.onnx`, `PP-OCRv6_rec_small.onnx`, `ch_ppocr_mobile_v2.0_cls_mobile.onnx`) |
| `tiny`, `medium` | downloaded by RapidOCR on first use into its own `models/` directory |

RapidOCR writes downloads next to its installed package, which is read-only in
Nix and in most container images. For `tiny`/`medium` in such an environment,
fetch the weights at build time and pass them explicitly:

```nix
let models = pkgs.linkFarm "pp-ocrv6-medium" {
  det = pkgs.fetchurl {
    url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx";
    hash = "...";
  };
  rec = pkgs.fetchurl {
    url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx";
    hash = "...";
  };
};
in pkgs.writeShellScript "ocr-medium" ''
  exec ocrmypdf --rapid-model-type medium \
    --rapid-det-model-path ${models}/det \
    --rapid-rec-model-path ${models}/rec "$@"
''
```

## Troubleshooting

**`Removing OMP_THREAD_LIMIT=… set by the Tesseract plugin` (warning)**

OCRmyPDF's Tesseract plugin sets `OMP_THREAD_LIMIT` for the whole process, which
onnxruntime would inherit. The plugin unsets it once before building the session;
the warning is expected when both plugins are installed.

**`RapidOCR could not write its model files`**

You asked for `tiny`/`medium` (or a language/model the package does not bundle)
on a read-only installation. See [Models](#models).

**`No installed font has glyphs for the detected 'chi_sim' text`**

The invisible text layer needs a CJK font. Install e.g. Noto Sans CJK
(`noto-fonts-cjk-sans` on NixOS, `fonts-noto-cjk` on Debian). Otherwise the text
stays searchable and copyable but highlighting shows nothing.

**`DigitalSignatureError: Input PDF has a digital signature`**

Add `--invalidate-digital-signatures` (and accept that the signature is gone).

**`redo-ocr` refuses: "This PDF has a user fillable form"**

Use `--force-ocr` instead, and expect re-rasterisation.

**Do not pass `--plugin`**

The plugin is registered through the `ocrmypdf` entry point. `--plugin rapidocr`
would import the *rapidocr library* and pluggy rejects the name collision, so
leave `--plugin` off even though older versions of this fork suggested it.

**It is slow**

1. `-j 1` — every worker loads its own copy of the model.
2. `--rapid-det-limit-side-len 640` with `--rapid-det-limit-type min`.
3. `--rapid-model-type tiny` (needs the tiny weights on disk, see above).

## Development

```bash
pytest                                  # no models / network needed
nix-shell ./nix/shell.nix               # full runtime, ocrmypdf on PATH
nix build .#default                     # builds the package and runs pytest in the sandbox
```

## License

MPL-2.0 — same as OCRmyPDF and the upstream plugin. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for the attribution and the list of modifications.
