#!/usr/bin/env bash
# One-time install, run INSIDE the Ubuntu (WSL2) terminal from the repo root:
#     bash setup/install_wsl.sh
# Safe to re-run: every step skips what is already there.
set -euo pipefail
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

echo "== Hunyuan3D-2 =="
[ -d "$THIRD/Hunyuan3D-2" ] || git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 "$THIRD/Hunyuan3D-2"
cd "$THIRD/Hunyuan3D-2"
pip install -r requirements.txt
pip install -e .
( cd hy3dgen/texgen/custom_rasterizer && python setup.py install )
( cd hy3dgen/texgen/differentiable_renderer && python setup.py install )
cd "$ROOT"
pip install -r requirements.txt

echo "== Blender 4.2 LTS (headless: remesh, UV, bake, FBX) =="
BL="$THIRD/blender-4.2.3-linux-x64"
if [ ! -x "$BL/blender" ]; then
  wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz -O /tmp/bl.tar.xz
  tar -xf /tmp/bl.tar.xz -C "$THIRD"
fi

# Activation hook so `conda activate gen3d` always sets these.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/gen3d.sh" <<EOF
export BLENDER="$BL/blender"
export CUDA_HOME="\$CONDA_PREFIX"
export PYTHONPATH="$ROOT:\${PYTHONPATH:-}"
EOF

echo
echo "Installed. Open a new Ubuntu terminal, then:"
echo "  conda activate gen3d && cd $ROOT"
echo "  python -m gen3d make /mnt/c/Users/<you>/Pictures/thing.png --faces 5000"
echo "(model weights, ~10 GB, download on the first run)"
