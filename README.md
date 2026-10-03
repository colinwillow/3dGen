# 3dGen

Image → **textured high-poly model** (FBX + GLB), ready to retopo in Cinema 4D or Blender.
Open models only, running on your own GPU. An automatic quad retopo is there as an option.

```
image ──TRELLIS.2 or Hunyuan3D-2──▶ textured high-poly ──▶ QuadWild (or QuadriFlow) quads
                                                     ─▶ UV unwrap  ─▶ bake colour / normal / roughness
                                                     ─▶ model.fbx (quads, textures embedded)
```

**Why it bakes:** every open texturing model works on triangles, so painting the quad mesh
directly would triangulate it. Instead the high-poly is painted, then its colour and surface
detail are baked onto the quad mesh's own UVs. This is the standard high-to-low workflow.

## What it does and doesn't do

| | |
|---|---|
| Shape from 1 image | **TRELLIS.2** (Microsoft, MIT): sparse grid up to 1536³, much finer detail |
| ... or front + left/back/right | Hunyuan3D-2 / 2mv: softer shapes (384³ grid), takes extra views |
| Quads at a target face count | **QuadWild** (default): loops follow creases and features. QuadriFlow: an even grid, with symmetry |
| Texture | Hunyuan paint on the high-poly, baked to 2K maps |
| Output | `.fbx` with quads and embedded textures; `--glb` adds a triangulated GLB |
| **Not** Tripo's Smart Mesh | QuadWild traces creases and curvature, so it does well on hard surfaces, but it does not know what an eye or a mouth is. Tripo's learned topology is still better on faces. |

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
   It installs conda, CUDA 12.4, PyTorch, Hunyuan3D-2, TRELLIS.2, QuadWild and a headless
   Blender 4.2. Allow 40–60 minutes. The model weights (~25 GB) download on the first run.
4. **TRELLIS.2 needs one Hugging Face step**, because it uses Meta's DINOv3 image encoder,
   which is a gated download. Once:
   - sign in at huggingface.co and request access to
     [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
     (Meta reviews it by hand, so it can take a while);
   - make a **Read** token at huggingface.co/settings/tokens;
   - in Ubuntu: `conda activate trellis2 && hf auth login`, and paste the token.

   Until then, Hunyuan3D-2 still works.

**Updating:** `cd ~/3dGen && git pull`, then restart the app. When an update adds something new
to install (it will say so), also re-run `bash setup/install_wsl.sh`. It skips what's already there.

## Use: the web app

Double-click **3dGen** on your desktop (the installer puts it there). A small window opens,
and that window is the app running; close it to stop the app. Your browser then opens
**http://localhost:7860**:

1. Drop in a photo. Pick the **Shape model**: **TRELLIS.2** (default once installed) for the
   finest detail, or **Hunyuan3D-2**, which also takes left/back/right views. TRELLIS.2's
   **Detail** setting is max (1536) by default. If the GPU runs out of memory there, it drops
   to 1024 by itself. Or switch to **Retopo a mesh** and drop in a high-poly GLB/FBX/OBJ you
   already have.
2. **Output: High-poly** (default) gives you the AI's textured model. The **Polygons** slider
   sets its face count, from 10k to 2M (Tripo's high-poly ceiling):
   - up to 100k, the AI paints the mesh directly;
   - above that, the AI paints a 100k copy and Blender bakes that colour onto the denser mesh;
   - above 500k, it also runs a finer shape pass (octree 512) and bakes at 4K once you pass 1M.
   You can't get more faces than the AI actually made: ask for more and the log says how many
   you got. 2M takes several minutes, mostly the UV unwrap and the bake.
   **Quad retopo** adds an automatic quad pass. **QuadWild** (default) follows creases and
   features. Untick **Hard edges** for smooth organic shapes. **QuadriFlow** lays an even grid
   and has a symmetry option.
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

# ... at a chosen polygon count (10k-2M)
python -m gen3d make /mnt/c/Users/<you>/Pictures/robot.png --polys 1000000

# ... plus an automatic quad retopo at ~5000 faces
python -m gen3d make /mnt/c/Users/<you>/Pictures/robot.png --retopo --faces 5000

# TRELLIS.2: finer shape detail (one image)
python -m gen3d make robot.png --model trellis --polys 1000000

# multi-view with Hunyuan: better backs and sides
python -m gen3d make front.png --left left.png --back back.png --faces 5000

# quad-retopo ANY mesh you already have (e.g. a high-poly from Tripo or ZBrush)
python -m gen3d retopo statue.glb --faces 5000 --symmetry
```

Output goes to `out/<name>/`. Open it from Windows Explorer at `\\wsl$\Ubuntu\home\<you>\3dGen\out`,
or write straight to Windows with `--out /mnt/c/Users/<you>/Desktop/3dgen`.

Useful knobs:

- `--faces 5000`: target face count (QuadriFlow lands within ~10%)
- `--engine quadwild|quadriflow`: the quad remesher (QuadWild falls back to QuadriFlow if it fails)
- `--sharp 35`: QuadWild's crease angle kept as a hard edge; `-1` for organic shapes
- `--symmetry`: mirrored quad flow (QuadriFlow only)
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
- **Hunyuan: run on the 3090.** High-poly and the texture bake work.
- **QuadWild: tested** headless (all-quad, within ~10% of the face count).
- **TRELLIS.2: written against its published code, not yet run on a GPU.** The plumbing is
  tested with a stand-in model. If the install or first run breaks, it will most likely be
  there. Paste the error back.
