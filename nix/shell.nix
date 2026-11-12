# Ready-to-use shell:  nix-shell nix/shell.nix
#
#   ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim in.pdf out.pdf
#
# Bundles the plugin plus everything OCRmyPDF needs from the outside world
# (ghostscript / qpdf / pngquant) and a CJK font so the invisible text layer can
# pick a font that actually has the glyphs.
{
  pkgs,
  pythonPkg ? pkgs.python313,
}:
let
  plugin = pkgs.callPackage ./default.nix { pythonPkg = pythonPkg; };
  pythonEnv = pythonPkg.withPackages (_: [ plugin ]);
in
pkgs.mkShell {
  packages = [
    pythonEnv
    pkgs.ghostscript
    pkgs.qpdf
    pkgs.pngquant
    pkgs.tesseract # ocrmypdf still shells out to it for some checks
    pkgs.noto-fonts-cjk-sans # without this: "No installed font has glyphs for 'chi_sim'"
  ];

  shellHook = ''
    export PADDLE_PDX_MODEL_SOURCE="''${PADDLE_PDX_MODEL_SOURCE:-HuggingFace}"
    export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
    echo "ocrmypdf: $(command -v ocrmypdf) ($(ocrmypdf --version 2>/dev/null | head -1))"
    echo "models:   ''${HOME}/.paddlex/official_models"
  '';
}
