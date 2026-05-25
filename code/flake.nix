{
  description = "Python development environment with Jupyter kernel support for Molten.nvim";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        python = pkgs.python312;
        
        pythonEnv = python.withPackages (ps: with ps; [
          # Jupyter kernel support (required for Molten.nvim)
          ipykernel
          jupyter-client
          jupyter  # Needed for Molten to manage kernels
          jupytext # For converting .ipynb to .py/.md
          pylatexenc # For LaTeX rendering (latex2text)
          
          # Common data science packages (customize as needed)
          numpy
          pandas
          matplotlib
          sympy
          scipy
          
          # Optional: additional visualization
          # plotly
          # seaborn
          
          # Optional: machine learning
          # scikit-learn
          # torch
          
          # Development tools
          pynvim  # For Neovim integration
        ]);
        
        # Unique kernel name based on project directory
        kernelName = "nix-jupyter";
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];

          shellHook = ''
            KERNEL_NAME="${kernelName}"
            KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
            CURRENT_PYTHON="${pythonEnv}/bin/python"
            
            # Always update kernel to ensure it points to current nix store path
            mkdir -p "$KERNEL_DIR"
            cat > "$KERNEL_DIR/kernel.json" << EOF
            {
              "argv": ["$CURRENT_PYTHON", "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
              "display_name": "Python (Nix Jupyter)",
              "language": "python",
              "metadata": {"debugger": true}
            }
            EOF
            
            echo ""
            echo "Python Jupyter environment ready!"
            echo "  - Python: $(python --version)"
            echo "  - Kernel: $KERNEL_NAME"
            echo ""
            echo "In Neovim, run :MoltenInit and select '$KERNEL_NAME'"
            echo ""
            
            # Return to your preferred shell (zsh) while keeping the nix environment
            exec zsh
          '';
        };
      }
    );
}
