{
  description = "Biofilms Odin port (Cellular Potts + voxel renderer)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }: let
    perSystem = flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };

        shellNativeBuildInputs = with pkgs; [
          odin
          ols
          raylib # biofilm-viewer link (needs raylib in link path)
          vulkan-headers # compute-shader placement headers (odin/shaders)
          glslang # optional: build.sh all compiles field_diffusion.comp to SPIR-V
          pkg-config
        ];

        odinRuntime = pkgs.stdenv.mkDerivation {
          pname = "biofilms-odin";
          version = "0.1.0";
          src = pkgs.lib.cleanSource ./odin;
          nativeBuildInputs = [pkgs.odin];
          buildInputs = [];
          hardeningDisable = ["all"];
          dontConfigure = true;
          buildPhase = ''
            runHook preBuild
            # Headless build: odin/build.sh runs `odin test tests/` then
            # `odin build . -out:build/biofilm`. -o:speed matches the
            # optimized release build; drop to -o:fast if the builder
            # runs out of memory.
            ${pkgs.bash}/bin/bash ./build.sh
            runHook postBuild
          '';
          checkPhase = ''
            runHook preCheck
            odin test tests
            runHook postCheck
          '';
          doCheck = true;
          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin"
            cp build/biofilm "$out/bin/"
            runHook postInstall
          '';
        };
      in {
        packages = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          default = odinRuntime;
          biofilms-odin = odinRuntime;
        };

        checks = pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          odin-runtime = odinRuntime;
        };

        devShells.default = pkgs.mkShell {
          name = "shell";
          hardeningDisable = ["all"];
          nativeBuildInputs = shellNativeBuildInputs;
          NIX_ENFORCE_NO_NATIVE = 0;
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath shellNativeBuildInputs}";
          shellHook = ''
            export LANG=C.UTF-8
            unset NIX_ENFORCE_NO_NATIVE
          '';
        };
      }
    );
  in
    perSystem;
}
