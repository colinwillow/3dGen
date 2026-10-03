"""TRELLIS.2 (Microsoft, MIT): image -> textured high-poly GLB.

Runs inside its own conda env (`trellis2`, made by setup/install_wsl.sh), because its
CUDA extensions want torch 2.6 and Hunyuan's want 2.5. gen3d calls it as a child
process, the same way it calls Blender:

    $GEN3D_TRELLIS gen3d/trellis_run.py photo.png out/dir --res 1536 --polys 1000000

Why it is here: Hunyuan3D-2.0 builds its surface on a 384^3 grid, which is what makes
its shapes soft and bulbous. TRELLIS.2 builds on a sparse grid up to 1536^3 -- four
times the resolution along each axis -- so creases, panel lines and small parts survive.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(*a):
    print("[trellis]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--res", choices=["512", "1024", "1536"], default="1536",
                    help="shape grid: 1536 is the most detail and the most VRAM")
    ap.add_argument("--polys", type=int, default=1000000, help="faces in the output mesh")
    ap.add_argument("--tex", type=int, default=2048, help="baked texture size")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from PIL import Image
    import trellis2.pipelines.rembg as rembg
    from trellis2.modules import image_feature_extractor as fx
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    # Background removal: the model's config asks for briaai/RMBG-2.0, which is a gated
    # download. ZhengPeng7/BiRefNet is the open model it was built from (MIT), and does
    # the same job, so it is used unless GEN3D_REMBG says otherwise.
    Base = rembg.BiRefNet

    class OpenBiRefNet(Base):
        def __init__(self, model_name=None, **kw):
            super().__init__(os.environ.get("GEN3D_REMBG", "ZhengPeng7/BiRefNet"))

    rembg.BiRefNet = OpenBiRefNet

    # The image encoder (DINOv3) is gated by Meta and has no open twin. GEN3D_DINOV3
    # can point at a local copy or a mirror if access is a problem.
    if os.environ.get("GEN3D_DINOV3"):
        init = fx.DinoV3FeatureExtractor.__init__
        fx.DinoV3FeatureExtractor.__init__ = \
            lambda self, model_name, **kw: init(self, os.environ["GEN3D_DINOV3"], **kw)

    t0 = time.time()
    log("loading microsoft/TRELLIS.2-4B")
    try:
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(
            os.environ.get("GEN3D_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"))
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            sys.exit("TRELLIS.2 needs Meta's DINOv3 image encoder, which is a gated download.\n"
                     "1) on huggingface.co, request access to facebook/dinov3-vitl16-pretrain-lvd1689m\n"
                     "2) in Ubuntu: conda activate trellis2 && hf auth login   (paste a read token)\n"
                     f"({e})")
        raise
    pipe.cuda()

    image = Image.open(a.image)
    kind = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[a.res]
    log(f"shape + texture at {a.res}^3, seed {a.seed}")
    try:
        mesh = pipe.run(image, seed=a.seed, pipeline_type=kind)[0]
    except torch.cuda.OutOfMemoryError:
        if a.res == "512":
            raise
        log(f"out of GPU memory at {a.res}; trying 1024")
        torch.cuda.empty_cache()
        mesh = pipe.run(image, seed=a.seed, pipeline_type="1024_cascade")[0]
    log(f"{mesh.faces.shape[0]} faces raw ({time.time() - t0:.0f}s)")

    mesh.simplify(16777216)  # nvdiffrast's limit, needed before baking
    target = min(a.polys, int(mesh.faces.shape[0]))
    if a.polys > target:
        log(f"asked for {a.polys} faces; the model made {target}, so you get all {target}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target, texture_size=a.tex,
        remesh=True, remesh_band=1, remesh_project=0, verbose=True)
    path = os.path.join(a.out, "textured.glb")
    glb.export(path)  # PNG textures, not webp: C4D's glTF importer can't read webp
    log(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
