"""Image -> textured high-poly mesh, with Tencent's Hunyuan3D-2 (open weights).

Single image, or front + left/back/right for the multi-view model. Shape runs
first and is freed before paint, so both fit in the 3090's 24 GB one at a time.

This file has NOT been run on a GPU by the person who wrote it. If something
breaks on the first run it is most likely the Hunyuan API itself; their README
(github.com/Tencent-Hunyuan/Hunyuan3D-2) is the reference.
"""
import gc
from pathlib import Path

import torch
from PIL import Image


def _rgba(path, rembg):
    img = Image.open(path).convert("RGBA")
    # A photo or render with a background: cut it out. Art that already has
    # transparency is left alone.
    if img.getextrema()[3][0] == 255:
        img = rembg(img)
    return img


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()



# Hunyuan's own cleanup (FloaterRemover, FaceReducer) goes through pymeshlab, whose
# file-format plugins need system OpenGL libraries a bare WSL install may lack -- it
# then fails with "Unknown format for load: ply" right after the shape is made.
# Try it first; if it can't run, do the same two jobs with trimesh.

def clean(mesh):
    try:
        from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover
        return DegenerateFaceRemover()(FloaterRemover()(mesh))
    except Exception as e:
        print(f"[shape] pymeshlab cleanup unavailable ({e}); using trimesh")
    import trimesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    big = max(len(p.faces) for p in parts)
    keep = [p for p in parts if len(p.faces) >= big * 0.01]  # drop floating specks
    print(f"[shape] kept {len(keep)} of {len(parts)} pieces")
    return trimesh.util.concatenate(keep)


def reduce(mesh, faces):
    if len(mesh.faces) <= faces:
        return mesh
    try:
        from hy3dgen.shapegen import FaceReducer
        return FaceReducer()(mesh, max_facenum=faces)
    except Exception as e:
        print(f"[shape] pymeshlab reducer unavailable ({e}); using fast-simplification")
    return mesh.simplify_quadric_decimation(face_count=faces)

def generate(front, out_dir, left=None, back=None, right=None, texture=True,
             steps=50, octree=None, seed=1234, paint_faces=100000, turbo=False, faces=None):
    """Returns (textured, dense). `dense` is the model at `faces` when that is more than
    the painting can take; the caller bakes the painted texture onto it. Otherwise None
    and `textured` already is the model at `faces`."""
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if octree is None:
        # Finer sampling only when the count asked for needs it: it costs time and VRAM.
        octree = 512 if (faces or 0) > 500_000 else 384
    rembg = BackgroundRemover()
    views = {k: v for k, v in dict(front=front, left=left, back=back, right=right).items() if v}
    multi = len(views) > 1

    if multi:
        repo, sub = "tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv" + ("-turbo" if turbo else "")
        image = {k: _rgba(v, rembg) for k, v in views.items()}
    else:
        repo, sub = "tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0" + ("-turbo" if turbo else "")
        image = _rgba(front, rembg)
    print(f"[shape] {repo}/{sub}, views: {', '.join(views)}, octree {octree}")

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo, subfolder=sub, use_safetensors=True)
    if turbo:
        pipe.enable_flashvdm()
    mesh = pipe(image=image, num_inference_steps=10 if turbo else steps,
                octree_resolution=octree, num_chunks=20000,
                generator=torch.manual_seed(seed), output_type="trimesh")[0]
    _free(pipe)

    mesh = clean(mesh)
    raw = len(mesh.faces)
    mesh.export(out / "shape.glb")
    print(f"[shape] {raw} faces -> {out / 'shape.glb'}")

    # The face count asked for. More than the AI made cannot be honoured -- subdividing
    # would add faces and no detail -- so it is capped at what is there, and said so.
    target = raw if not faces else min(faces, raw)
    if faces and faces > raw:
        print(f"[shape] asked for {faces} faces; the AI made {raw}, so you get all {raw}")
    model = reduce(mesh.copy(), target) if target < raw else mesh
    print(f"[shape] model at {len(model.faces)} faces")
    if not texture:
        model.export(out / "model.glb")
        return out / "model.glb", None

    # Painting is slow on dense meshes, so it runs on a reduced copy. Past that size the
    # paint is baked onto the dense model afterwards, in Blender.
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    pmesh = model if len(model.faces) <= paint_faces else reduce(model.copy(), paint_faces)
    paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    front_img = image["front"] if multi else image
    pmesh = paint(pmesh, image=front_img)
    _free(paint)
    pmesh.export(out / "textured.glb")
    print(f"[shape] textured {len(pmesh.faces)} faces -> {out / 'textured.glb'}")
    if len(model.faces) > len(pmesh.faces):
        model.export(out / "dense.glb")
        return out / "textured.glb", out / "dense.glb"
    return out / "textured.glb", None
