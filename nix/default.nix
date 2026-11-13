# Build the plugin package:
#   nix-build -E 'with import <nixpkgs> {}; callPackage ./nix/default.nix {}'
#   pkgs.callPackage ./nix/default.nix { }
#
# The engine is RapidOCR on ONNX Runtime, so nothing here needs paddlepaddle
# (and nothing needs modelscope/unfree packages, unlike paddleocr).
#
# Models: the nixpkgs `rapidocr` package bundles the PP-OCRv6 *small* det/rec
# weights plus the PP-OCRv4 textline cls model (~31 MiB) inside its own store
# path, and this plugin defaults to --rapid-model-type small, so OCR works
# offline out of the box.  Other model sizes (tiny/medium) are downloaded by
# RapidOCR on demand and therefore need a writable model dir; in a pinned build
# pass them explicitly with --rapid-det-model-path / --rapid-rec-model-path.
{
  lib,
  pkgs,
  pythonPkg ? pkgs.python3,
  withOnnxRuntime ? true,
}:

let
  ps = pythonPkg.pkgs;
in
ps.buildPythonPackage rec {
  pname = "ocrmypdf-rapidocr";
  version = "0.3.0";
  pyproject = true;
  src = lib.cleanSource ../.;

  build-system = [ ps.setuptools ];

  dependencies =
    with ps;
    [ ocrmypdf rapidocr pillow ]
    # rapidocr talks to onnxruntime through a plugin-style backend, so it is a
    # runtime extra rather than a hard dependency.
    ++ lib.optionals withOnnxRuntime [ onnxruntime ];

  nativeCheckInputs = [ ps.pytest ];
  doCheck = true;

  pythonImportsCheck = [ "ocrmypdf_rapidocr" ];

  meta = {
    description = "RapidOCR (PP-OCRv6) engine plugin for OCRmyPDF";
    homepage = "https://github.com/Urie96/ocrmypdf-paddle-plugin";
    license = lib.licenses.mpl20;
    maintainers = [ ];
    platforms = lib.platforms.unix;
  };
}
