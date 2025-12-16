{
  description = "Exam flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
  };

  outputs = { self, nixpkgs }: {
    devShells."x86_64-linux".default = let
      pkgs = import nixpkgs { inherit system; };
      system = "x86_64-linux";
    in
      pkgs.mkShell {
        packages = with pkgs; [
          python313
          python313Packages.numpy
          gcc 
          stdenv.cc.cc.lib 
          python3Packages.pyqt5 
          qt5.qtwayland 
          qt5.qtbase
        ];

      shellHook = ''
        echo "Loaded nix shell"

        zsh
      '';
    };
  };
}
