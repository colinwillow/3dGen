# 3dGen

Image → **textured high-poly model** (FBX + GLB), ready to retopo in Cinema 4D or Blender.
Open models only, running on your own GPU. An automatic quad retopo is there as an option.

```
image ──Hunyuan3D-2──▶ textured high-poly ──Blender──▶ voxel shell ─▶ QuadriFlow quads
                                                     ─▶ UV unwrap  ─▶ bake colour / normal / roughness
                                                     ─▶ model.fbx (quads, textures embedded)
```

**Why it bakes:** every open texturing model works on triangles, so painting the quad mesh
directly would triangulate it. Instead the high-poly is painted, then its colour and surface
detail are baked onto the quad mesh's own UVs. This is the standard high-to-low workflow.

## What it does and doesn't do

| | |
|---|---|
| Shape from 1 image, or front + left/back/right | Hunyuan3D-2 / 2mv (open weights) |
| Quads at a target face count | Blender's QuadriFlow: an even, all-quad grid |
| Texture | Hunyuan paint on the high-poly, baked to 2K maps |
| Output | `.fbx` with quads and embedded textures; `--glb` adds a triangulated GLB |
| **Not** Tripo's Smart Mesh | QuadriFlow gives a clean, even quad grid, but no artist edge loops around eyes, mouth or joints. Tripo's learned topology is still better there. |

## Setup (Windows + RTX 3090)

These models' CUDA extensions are much easier to build on Linux, so this runs in **WSL2**
(Ubuntu inside Windows, with full access to the GPU).

1. Update the NVIDIA driver on **Windows** (GeForce app or nvidia.com). Do not install a
   driver inside Ubuntu.
2. In PowerShell (as admin): `wsl --update`, then `wsl --install -d Ubuntu` (any Ubuntu from `wsl --list --online` works). Reboot if asked, then open **Ubuntu** from the Start menu and pick a username.
3. In the Ubuntu terminal, clone into your Linux home (clones under `/mnt/c` are much slower)
   and run the installer:
   ```bash
   cd ~ && git clone https://github.com/colinwillow/3dGen && cd 3dGen
   bash setup/install_wsl.sh
   ```
   It installs conda, CUDA 12.4, PyTorch, Hunyuan3D-2 and a headless Blender 4.2. Allow
   20–40 minutes. The model weights (~10 GB) download on the first run.

## Use: the web app

Double-click **3dGen** on your desktop (the installer puts it there). A small window opens,
and that window is the app running; close it to stop the app. Your browser then opens
**http://localhost:7860**:

1. Drop in a photo, plus optional left/back/right views for a better back side. Or switch to
   **Retopo a mesh** and drop in a high-poly GLB/FBX/OBJ you already have.
2. **Output: High-poly** (default) gives you the AI's textured model as-is. **Quad retopo**
   adds an automatic QuadriFlow pass, which shows face count, topology and symmetry settings.
3. Click **Generate** and watch the stages: Shape → Texture → Export (plus Quads → Bake for retopo).
4. Spin the result around in the viewer. **Wireframe** shows the real quad edges, not the
   triangulated preview. Then click **Download FBX**. High-poly jobs also offer the textured
   **GLB** and the **Full-res GLB**: the untextured shape at full density, which is finer than
   the textured mesh because painting needs a reduced one (`--paint-faces`, default 100k).

Every result stays in the History list (stored in `3dGen/jobs/`). Jobs run one at a time.
If you queue several, they wait their turn, and they resume if you close the app halfway.

## Use: the command line (same pipeline)

Open an Ubuntu terminal, then:

```bash
conda activate gen3d && cd ~/3dGen

# one image -> textured high-poly FBX/GLB (your Windows files are under /mnt/c)
python -m gen3d make /mnt/c/Users/<you>/Pictures/robot.png

# ... plus an automatic quad retopo at ~5000 faces
python -m gen3d make /mnt/c/Users/<you>/Pictures/robot.png --retopo --faces 5000

# multi-view: better backs and sides
python -m gen3d make front.png --left left.png --back back.png --faces 5000

# quad-retopo ANY mesh you already have (e.g. a high-poly from Tripo or ZBrush)
python -m gen3d retopo statue.glb --faces 5000 --symmetry
```

Output goes to `out/<name>/`. Open it from Windows Explorer at `\\wsl$\Ubuntu\home\<you>\3dGen\out`,
or write straight to Windows with `--out /mnt/c/Users/<you>/Desktop/3dgen`.

Useful knobs:

- `--faces 5000`: target face count (QuadriFlow lands within ~10%)
- `--symmetry`: mirrored quad flow, good for characters
- `--tris`: decimated triangles instead of quads
- `--voxel 0.006`: detail of the clean shell built before quadding. Lower it (0.003) if thin
  parts like fingers or antennae disappear; raise it if QuadriFlow fails.
- `--tex 2048`: baked map size
- `--turbo`: much faster shape, a little less detail
- `--no-texture`: shape only

### Side by side with Tripo

```bash
export TRIPO_API_KEY=tsk_...
python -m gen3d tripo robot.png --faces 5000
```

**Optional, and it costs money:** it uses your own Tripo API key and spends your Tripo credits, like generating on their site. Nothing goes to Tripo unless you run this command. It sends the image through Tripo's API with quads + smart low-poly + PBR, then
grounds and re-exports the result as FBX. The task fields follow Tripo's docs; if one is
refused, `--set field=value` passes any option through without a code change.

## Status

- **Blender stage (remesh, UV, bake, FBX): tested** headless on a 65k-tri textured mesh →
  4,551 all-quad faces, textures baked, and the quads survived an FBX re-import.
- **Web app: tested** in a headless browser: upload a mesh, progress, viewer (textured / clay /
  quad wireframe), FBX and texture downloads, history.
- **Hunyuan and Tripo stages: written against their documented APIs, not yet run on a
  GPU.** If the first run breaks, it will most likely be there. Paste the error back.
