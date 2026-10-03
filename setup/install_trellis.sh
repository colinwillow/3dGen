#!/usr/bin/env bash
# TRELLIS.2 (Microsoft, MIT licence): the finer-detail shape model.
# Called by install_wsl.sh; safe to run on its own and to re-run:
#     bash setup/install_trellis.sh
# It gets its own conda env (trellis2): its CUDA extensions are built against
# torch 2.6, Hunyuan's against 2.5, and the two cannot share one env.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/third_party"
T2="$THIRD/TRELLIS.2"
EXT="$THIRD/trellis_ext"
mkdir -p "$THIRD" "$EXT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q '^trellis2 ' || conda create -y -n trellis2 python=3.10
conda activate trellis2

[ -d "$T2" ] || git clone --recursive https://github.com/microsoft/TRELLIS.2 "$T2"

echo "== TRELLIS.2: CUDA 12.4 + gcc 12 + torch 2.6 =="
# Same two reasons as the main env: nvcc 12.4 refuses gcc > 13, and new glibc's
# <math.h> collides with CUDA 12.4's headers. conda's gcc 12 brings its own sysroot.
conda install -y -c "nvidia/label/cuda-12.4.1" cuda-toolkit
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
unset NVCC_PREPEND_FLAGS
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3090
export MAX_JOBS="${MAX_JOBS:-8}"    # the CUDA builds use a lot of RAM per job
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo "== TRELLIS.2: python packages =="
# transformers >= 4.56 has the DINOv3 encoder it needs; held under 5 like the main env.
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
  "transformers>=4.56,<5" tensorboard pandas lpips zstandard kornia timm pillow
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
# Attention: xformers has a prebuilt wheel for exactly this torch, where flash-attn
# 2.7.3 would compile from source for an hour or more. TRELLIS.2 supports both.
pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124

echo "== TRELLIS.2: CUDA extensions (several minutes each) =="
get() { [ -d "$EXT/$2" ] || git clone --recursive $3 "$1" "$EXT/$2"; }
get https://github.com/NVlabs/nvdiffrast.git nvdiffrast "-b v0.4.0"
get https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec "-b renderutils"
get https://github.com/JeffreyXiang/CuMesh.git CuMesh ""
get https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM ""
for d in nvdiffrast nvdiffrec CuMesh FlexGEMM; do
  echo "  building $d"
  pip install "$EXT/$d" --no-build-isolation
done
echo "  building o-voxel"
pip install "$T2/o-voxel" --no-build-isolation

rm -f "$T2/.gen3d_ok"
python -c "import torch, xformers, nvdiffrast.torch, cumesh, flex_gemm, o_voxel; print('  TRELLIS.2 ok, torch', torch.__version__)"
touch "$T2/.gen3d_ok"   # the app only offers TRELLIS.2 once this exists
echo
echo "TRELLIS.2 installed. Its image encoder (Meta's DINOv3) is a gated download, so once:"
echo "  1) sign in at huggingface.co and request access to facebook/dinov3-vitl16-pretrain-lvd1689m"
echo "  2) make a Read token (huggingface.co/settings/tokens), then here run:"
echo "       conda activate trellis2 && hf auth login"
