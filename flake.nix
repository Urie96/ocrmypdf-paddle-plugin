{
  description = "RapidOCR (PP-OCRv6) engine plugin for OCRmyPDF";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAll = nixpkgs.lib.genAttrs systems;
    in
    {
      packages = forAll (system: {
        default = nixpkgs.legacyPackages.${system}.callPackage ./nix/default.nix { };
      });

      devShells = forAll (system: {
        default = nixpkgs.legacyPackages.${system}.callPackage ./nix/shell.nix { };
      });

      overlays.default = final: prev: {
        ocrmypdf-rapidocr = final.callPackage ./nix/default.nix { };
      };
    };
}
