"""Blender stage: high-poly mesh in, quad (or tri) low-poly FBX out, with the
high-poly's colour and surface detail baked onto the low-poly's own UVs.

Runs two ways, with the same arguments:
    blender --background --python gen3d/retopo.py -- --high in.glb --out dir
    python gen3d/retopo.py --high in.glb --out dir        # with `pip install bpy`

Why bake instead of painting the low-poly directly: every texturing model goes
through trimesh/xatlas, which TRIANGULATES. Painting the high-poly and baking
onto the quad mesh is the only route that keeps the quads.

Modes:
    retopo   (default) remesh -> UV -> bake colour/normal/roughness -> export
    clean    no remesh, just ground/centre it and re-export (e.g. a Tripo FBX)
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", required=True, help="glb / gltf / fbx / obj")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--name", default="model")
    ap.add_argument("--mode", choices=["retopo", "clean"], default="retopo")
    ap.add_argument("--faces", type=int, default=5000, help="target face count")
    ap.add_argument("--tris", action="store_true", help="decimate to triangles instead of quads")
    ap.add_argument("--tex", type=int, default=2048, help="baked texture size")
    ap.add_argument("--voxel", type=float, default=0.006,
                    help="pre-remesh voxel size as a fraction of the model's size (0 = skip)")
    ap.add_argument("--symmetry", action="store_true", help="QuadriFlow X symmetry")
    ap.add_argument("--glb", action="store_true", help="also write a (triangulated) GLB")
    return ap.parse_args(argv)


def log(*a):
    print("[retopo]", *a, flush=True)


def import_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"can't import {ext}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in " + path)
    return meshes


def only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def join(meshes):
    # Unparent (keeping transforms) so the armature/empties from a glTF don't
    # carry a scale into the result, then join and bake the transform in.
    for o in meshes:
        only(o)
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o is not obj and o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    return obj


def ground(objs):
    """Feet on the floor, centred on the origin -- moves every object together."""
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    shift = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for o in objs:
        o.location -= shift
        only(o)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    return (hi - lo).length


def stats(obj):
    polys = obj.data.polygons
    quads = sum(1 for p in polys if len(p.vertices) == 4)
    tris = sum(1 for p in polys if len(p.vertices) == 3)
    return f"{len(polys)} faces ({quads} quads, {tris} tris, {len(polys) - quads - tris} ngons)"


def remesh(high, diag, a):
    low = high.copy()
    low.data = high.data.copy()
    low.name = a.name
    bpy.context.collection.objects.link(low)
    only(low)
    low.data.materials.clear()

    # Generated meshes are rarely manifold, and QuadriFlow refuses anything that
    # isn't. A fine voxel remesh first makes it a clean closed shell.
    if a.voxel > 0:
        m = low.modifiers.new("vox", "REMESH")
        m.mode = "VOXEL"
        m.voxel_size = diag * a.voxel
        bpy.ops.object.modifier_apply(modifier=m.name)
        log("voxel shell:", stats(low))
    # QuadriFlow also wants consistently wound normals, and says so only as a
    # warning while still reporting FINISHED -- hence the count check below.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=diag * 1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    if a.tris:
        m = low.modifiers.new("dec", "DECIMATE")
        m.ratio = min(1.0, a.faces / max(1, len(low.data.polygons)))
        bpy.ops.object.modifier_apply(modifier=m.name)
    else:
        before = len(low.data.polygons)
        bpy.ops.object.quadriflow_remesh(
            mode="FACES", target_faces=a.faces, use_mesh_symmetry=a.symmetry,
            use_preserve_sharp=False, use_preserve_boundary=False,
            smooth_normals=False, seed=0)
        got = len(low.data.polygons)
        if got == before or got > a.faces * 2:
            raise SystemExit(f"QuadriFlow did not remesh ({before} -> {got} faces). "
                             "Try a bigger --voxel, or --tris")
    bpy.ops.object.shade_smooth()
    log("low:", stats(low))
    return low


def unwrap(low):
    only(low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
    bpy.ops.uv.pack_islands(margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def cycles_device():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                if any(d.type == kind for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = "GPU"
                    log("baking on", kind)
                    return
            except TypeError:
                continue
    except Exception:
        pass
    scene.cycles.device = "CPU"
    log("baking on CPU")


def bake(high, low, diag, a, outdir):
    cycles_device()
    mat = bpy.data.materials.new(a.name)
    mat.use_nodes = True
    low.data.materials.append(mat)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    def image(kind, colour):
        img = bpy.data.images.new(f"{a.name}_{kind}", a.tex, a.tex, alpha=False)
        img.colorspace_settings.name = "sRGB" if colour else "Non-Color"
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        return node

    base, nrm, rough = image("basecolor", True), image("normal", False), image("roughness", False)
    bpy.ops.object.select_all(action="DESELECT")
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low
    common = dict(use_selected_to_active=True, cage_extrusion=diag * 0.01,
                  max_ray_distance=diag * 0.04, margin=8)
    for node, kind, extra in ((base, "DIFFUSE", dict(pass_filter={"COLOR"})),
                              (nrm, "NORMAL", dict(normal_space="TANGENT")),
                              (rough, "ROUGHNESS", {})):
        for n in nt.nodes:
            n.select = False
        node.select = True
        nt.nodes.active = node
        bpy.ops.object.bake(type=kind, **common, **extra)
        path = os.path.join(outdir, node.image.name + ".png")
        node.image.filepath_raw = path
        node.image.file_format = "PNG"
        node.image.save()
        log("baked", os.path.basename(path))

    nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Metallic"].default_value = 0.0


def export(objs, a, outdir):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
        for m in o.data.materials:
            if m:
                m.use_backface_culling = True  # glTF doubleSided = false
    bpy.context.view_layer.objects.active = objs[0]
    fbx = os.path.join(outdir, a.name + ".fbx")
    # path_mode COPY + embed: one file that opens in C4D/Blender with its textures.
    bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, path_mode="COPY",
                             embed_textures=True, apply_scale_options="FBX_SCALE_ALL",
                             mesh_smooth_type="FACE", use_triangles=False)
    log("wrote", fbx)
    if a.glb:
        glb = os.path.join(outdir, a.name + ".glb")
        bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True)
        log("wrote", glb, "(triangulated -- glTF has no quads)")
    preview(objs, outdir)


def preview(objs, outdir):
    """What the web viewer shows: a textured GLB (browsers can't draw quads, so
    it's triangulated) plus the REAL quad edges as line pairs, so the wireframe
    shows the topology you'll get in the FBX rather than the triangulation."""
    import json
    import struct
    bpy.ops.export_scene.gltf(filepath=os.path.join(outdir, "preview.glb"),
                              export_format="GLB", use_selection=True)
    data, faces, quads, tris = bytearray(), 0, 0, 0
    for o in objs:
        mw, me = o.matrix_world, o.data
        vs = [mw @ v.co for v in me.vertices]
        for e in me.edges:
            for i in e.vertices:
                v = vs[i]
                data += struct.pack("<3f", v.x, v.z, -v.y)  # Blender Z-up -> glTF Y-up
        faces += len(me.polygons)
        quads += sum(1 for p in me.polygons if len(p.vertices) == 4)
        tris += sum(1 for p in me.polygons if len(p.vertices) == 3)
    with open(os.path.join(outdir, "wire.bin"), "wb") as f:
        f.write(data)
    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump(dict(faces=faces, quads=quads, tris=tris, ngons=faces - quads - tris), f)


def main():
    a = parse()
    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meshes = import_any(os.path.abspath(a.high))

    if a.mode == "clean":
        diag = ground(meshes)
        log("clean:", ", ".join(stats(o) for o in meshes))
        export(meshes, a, outdir)
        return

    high = join(meshes)
    diag = ground([high])
    log("high:", stats(high))
    low = remesh(high, diag, a)
    unwrap(low)
    bake(high, low, diag, a, outdir)
    bpy.data.objects.remove(high, do_unlink=True)
    export([low], a, outdir)
    log("done:", stats(low))


if __name__ == "__main__":
    main()
    # Everything is written by now. Blender can segfault while tearing itself
    # down after glTF exports, which would turn a finished job into a failed
    # one -- so skip the teardown entirely.
    sys.stdout.flush()
    os._exit(0)
