#!/usr/bin/env bash
# One-time install, run INSIDE the Ubuntu (WSL2) terminal from the repo root:
#     bash setup/install_wsl.sh
# Safe to re-run: every step skips what is already there.
set -eo pipefail   # no -u: conda's compiler activate scripts read unset variables
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/third_party"
mkdir -p "$THIRD"

echo "== GPU =="
# The NVIDIA driver lives on the WINDOWS side; WSL sees it through this.
nvidia-smi || { echo "No GPU visible in WSL. Update the Windows NVIDIA driver and run 'wsl --update'."; exit 1; }

echo "== system packages =="
sudo apt-get update -y
sudo apt-get install -y build-essential git wget curl xz-utils \
  libgl1 libglib2.0-0 libxi6 libxrender1 libxkbcommon0 libsm6 libxxf86vm1 libxfixes3

echo "== conda =="
if [ ! -d "$HOME/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q '^gen3d ' || conda create -y -n gen3d python=3.10
conda activate gen3d

echo "== CUDA toolkit + torch =="
# nvcc is needed to build Hunyuan's two texture-painting extensions. Kept inside
# the env, matched to the torch build, so nothing system-wide has to agree with it.
conda install -y -c "nvidia/label/cuda-12.4.1" cuda-toolkit
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3090

echo "== host compiler for CUDA 12.4 =="
# Always conda-forge's gcc 12, never the system one. Two separate reasons, and new
# Ubuntu hits both: nvcc 12.4 refuses gcc newer than 13, and glibc 2.41+ declares
# cospi/sinpi/rsqrt in <math.h>, which collide with CUDA 12.4's own declarations.
# The conda compiler carries its own older sysroot, so the system headers are never
# read at all.
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
unset NVCC_PREPEND_FLAGS   # torch already passes -ccbin "$CC"
echo "  using $("$CXX" --version | head -1)"

echo "== Hunyuan3D-2 =="
[ -d "$THIRD/Hunyuan3D-2" ] || git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 "$THIRD/Hunyuan3D-2"
cd "$THIRD/Hunyuan3D-2"
pip install -r requirements.txt
pip install -e .
# Hunyuan3D-2 was written against transformers 4.x. transformers 5 renamed the
# DINOv2 SwiGLU layers (mlp.weights_in/out), so the image encoder's checkpoint no
# longer loads -- a wall of "main_image_encoder... weights" keys at the shape step.
# transformers 4.x needs huggingface-hub < 1.0, and recent diffusers needs >= 1.x,
# so diffusers is held back with it.
pip install "transformers>=4.48,<4.50" "diffusers>=0.32,<0.33" "huggingface-hub>=0.26,<1.0" "tokenizers<0.22"
rm -rf hy3dgen/texgen/custom_rasterizer/build hy3dgen/texgen/differentiable_renderer/build
( cd hy3dgen/texgen/custom_rasterizer && python setup.py install )
( cd hy3dgen/texgen/differentiable_renderer && python setup.py install )
cd "$ROOT"
pip install -r requirements.txt

echo "== Blender 4.2 LTS (headless: remesh, UV, bake, FBX) =="
# Tries a few 4.2 LTS builds (download.blender.org only keeps some), then falls
# back to Ubuntu's own Blender. Loud on purpose: a silent failed download here
# used to end the installer with no message at all.
BL_EXE=""
for v in 4.2.9 4.2.3 4.2.0; do
  d="$THIRD/blender-$v-linux-x64"
  if [ -x "$d/blender" ]; then BL_EXE="$d/blender"; break; fi
  echo "  downloading Blender $v ..."
  if wget -q --show-progress -O /tmp/bl.tar.xz \
       "https://download.blender.org/release/Blender4.2/blender-$v-linux-x64.tar.xz"; then
    tar -xf /tmp/bl.tar.xz -C "$THIRD" && rm -f /tmp/bl.tar.xz
    [ -x "$d/blender" ] && { BL_EXE="$d/blender"; break; }
  fi
  echo "  Blender $v not available, trying the next one"
done
if [ -z "$BL_EXE" ]; then
  echo "  falling back to Ubuntu's blender package"
  sudo apt-get install -y blender
  BL_EXE="$(command -v blender)"
fi
[ -n "$BL_EXE" ] || { echo "Could not install Blender."; exit 1; }
"$BL_EXE" --version | head -1
BL="$(dirname "$BL_EXE")"

# Activation hook so `conda activate gen3d` always sets these.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/gen3d.sh" <<EOF
export BLENDER="$BL/blender"
export CUDA_HOME="\$CONDA_PREFIX"
export PYTHONPATH="$ROOT:\${PYTHONPATH:-}"
EOF

echo "== desktop shortcut =="
# A .bat on the Windows desktop that starts the app inside WSL and opens the browser.
DESK_WIN="$(powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Desktop')" 2>/dev/null | tr -d '\r')"
if [ -n "$DESK_WIN" ] && DESK="$(wslpath "$DESK_WIN" 2>/dev/null)" && [ -d "$DESK" ]; then
  printf '@echo off\r\ntitle 3dGen (close this window to stop it)\r\nwsl.exe -d %s -e bash -lc "%s/start.sh"\r\n' \
    "$WSL_DISTRO_NAME" "$ROOT" > "$DESK/3dGen.bat"
  echo "  put 3dGen.bat on your desktop -- double-click it to open the app"
else
  echo "  couldn't find your Windows desktop; start the app with: $ROOT/start.sh"
fi

echo
echo "Installed. Double-click 3dGen on your desktop, or open a new Ubuntu terminal and:"
echo "  $ROOT/start.sh          (then open http://localhost:7860)"
echo "(model weights, ~10 GB, download on the first run)"
