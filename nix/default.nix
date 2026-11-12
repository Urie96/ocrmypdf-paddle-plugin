# Build the plugin package:
#   nix-build -E 'with import <nixpkgs> {}; callPackage ./nix/default.nix {}'
#   pkgs.callPackage ./nix/default.nix { }
#
# Pinned to python313 by default: paddlepaddle has no python314 wheels yet, and
# nixpkgs' default `python3Packages` moves ahead of it.
{
  lib,
  pkgs,
  pythonPkg ? pkgs.python313,
  withOnnxRuntime ? true,
}:

let
  ps = pythonPkg.pkgs;
in
ps.buildPythonPackage rec {
  pname = "ocrmypdf-paddleocr";
  version = "0.2.0";
  pyproject = true;
  src = lib.cleanSource ../.;

  build-system = [ ps.setuptools ];

  dependencies =
    with ps;
    [
      ocrmypdf
      paddleocr
      paddlepaddle
      pillow
    ]
    ++ lib.optionals withOnnxRuntime [ onnxruntime ]
    # paddlex declares these under its "ocr-core" extra but nixpkgs does not
    # propagate optional-dependencies, so PaddleOCR() would abort with
    #   DependencyError: `OCR` requires additional dependencies
    # because paddlex checks importlib.metadata for each of them.
    ++ [
      imagesize
      pypdfium2
      python-bidi
      opencv-contrib-python
    ];

  nativeCheckInputs = [ ps.pytest ];
  doCheck = true;

  pythonImportsCheck = [ "ocrmypdf_paddleocr" ];

  meta = {
    description = "PaddleOCR (PP-OCRv6 / PaddleOCR-VL) engine plugin for OCRmyPDF";
    homepage = "https://github.com/USERNAME/ocrmypdf-paddle-plugin";
    license = lib.licenses.mpl20;
    maintainers = [ ];
    platforms = lib.platforms.unix;
  };
}
