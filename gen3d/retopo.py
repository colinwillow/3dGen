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
import shutil
import subprocess
import sys
import tempfile

import bpy
from mathutils import Vector


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", required=True, help="glb / gltf / fbx / obj")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--name", default="model")
    ap.add_argument("--mode", choices=["retopo", "clean", "transfer"], default="retopo")
    ap.add_argument("--dense", help="transfer mode: the mesh to bake --high's texture onto")
    ap.add_argument("--faces", type=int, default=5000, help="target face count")
    ap.add_argument("--tris", action="store_true", help="decimate to triangles instead of quads")
    ap.add_argument("--tex", type=int, default=2048, help="baked texture size")
    ap.add_argument("--voxel", type=float, default=0.006,
                    help="pre-remesh voxel size as a fraction of the model's size (0 = skip)")
    ap.add_argument("--symmetry", action="store_true", help="QuadriFlow X symmetry")
    ap.add_argument("--engine", choices=["quadwild", "quadriflow"], default="quadwild",
                    help="quad remesher (QuadWild falls back to QuadriFlow if it fails)")
    ap.add_argument("--sharp", type=float, default=35,
                    help="QuadWild: crease angle kept as a hard edge, in degrees (-1 = none)")
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


def fix_materials(objs):
    """glTF says a material that gives no metallic value is fully METALLIC -- and the
    trimesh exporter the AI's output goes through leaves it out. A metal has no diffuse
    colour, so it bakes black and shows up in C4D/Blender as dark chrome. The AI paints
    colour only, so anything without a real metallic map is set to 0."""
    for o in objs:
        for m in (o.data.materials if o.type == "MESH" else []):
            if not (m and m.use_nodes):
                continue
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED" and not n.inputs["Metallic"].is_linked \
                        and n.inputs["Metallic"].default_value > 0.5:
                    n.inputs["Metallic"].default_value = 0.0
                    log(f"material {m.name}: metallic 1 -> 0 (the file left it unset)")


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


def shell(high, diag, a, voxel):
    """A copy of the high-poly that QuadriFlow will accept: closed and manifold."""
    low = high.copy()
    low.data = high.data.copy()
    low.name = a.name
    bpy.context.collection.objects.link(low)
    only(low)
    low.data.materials.clear()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    if voxel > 0:
        bpy.ops.object.mode_set(mode="OBJECT")
        # Generated meshes are rarely manifold. A voxel remesh rebuilds the surface
        # as one closed shell; QuadriFlow can still refuse it, which is what the
        # coarser retries in remesh() are for.
        m = low.modifiers.new("vox", "REMESH")
        m.mode = "VOXEL"
        m.voxel_size = diag * voxel
        bpy.ops.object.modifier_apply(modifier=m.name)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        log(f"voxel shell ({voxel:g}):", stats(low))
    # Measured: on a clean voxel shell this weld is what lets QuadriFlow accept it
    # first time (without it the same blob needed 4 tries).
    bpy.ops.mesh.remove_doubles(threshold=diag * 1e-5)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return low


QW_DIR = os.environ.get("GEN3D_QUADWILD") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "quadwild")
QW_PREP = """do_remesh 1
sharp_feature_thr {sharp:g}
alpha {alpha:g}
scaleFact 1
"""


def faces_in(obj_path):
    n = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("f "):
                n += 1
    return n


def quadwild(high, diag, a):
    """QuadWild (Bi-MDF build): traces creases and curvature first, then lays the quads
    along them -- loops follow brows, sockets and panel lines where QuadriFlow lays an
    even grid over everything. Its face count is steered with scaleFact (quad size):
    faces ~ 1/scale^2, so one pass to measure and one or two to land on the target."""
    exe = os.path.join(QW_DIR, "quadwild")
    qfp = os.path.join(QW_DIR, "quad_from_patches")
    if not (os.path.exists(exe) and os.path.exists(qfp)):
        log(f"QuadWild not installed at {QW_DIR} -- using QuadriFlow")
        return None
    tmp = tempfile.mkdtemp(prefix="qw_")
    try:
        # Clean, welded, triangulated copy. QuadWild re-meshes its input anyway, so a
        # couple of hundred thousand triangles carry all the detail it will use.
        src = high.copy()
        src.data = high.data.copy()
        bpy.context.collection.objects.link(src)
        only(src)
        src.data.materials.clear()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=diag * 1e-5)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        if len(src.data.polygons) > 200000:
            m = src.modifiers.new("dec", "DECIMATE")
            m.ratio = 200000 / len(src.data.polygons)
            bpy.ops.object.modifier_apply(modifier=m.name)
        mesh = os.path.join(tmp, "mesh.obj")
        bpy.ops.wm.obj_export(filepath=mesh, export_selected_objects=True, export_materials=False,
                              export_uv=False, export_normals=False, apply_modifiers=True)
        bpy.data.objects.remove(src, do_unlink=True)

        prep = os.path.join(tmp, "prep.txt")
        with open(prep, "w") as f:
            f.write(QW_PREP.format(sharp=a.sharp, alpha=0.01 if a.sharp >= 0 else 0.02))
        log(f"QuadWild: tracing features (crease {a.sharp:g} deg)")
        r = subprocess.run([exe, mesh, "2", prep], cwd=QW_DIR, capture_output=True, text=True)
        patches = os.path.join(tmp, "mesh_rem_p0.obj")
        if not os.path.exists(patches):
            log("QuadWild could not prepare this mesh:", (r.stdout + r.stderr)[-600:])
            return None

        with open(os.path.join(QW_DIR, "config", "main_config", "flow.txt")) as f:
            flow = f.read()
        scale, got, best = 1.0, 0, None
        for i in range(4):
            cfg = os.path.join(tmp, f"flow{i}.txt")
            with open(cfg, "w") as f:
                f.write(flow.replace("\nscaleFact 1\n", f"\nscaleFact {scale:.4f}\n"))
            out = os.path.join(tmp, f"mesh_rem_p0_{i}_quadrangulation_smooth.obj")
            # It writes the result and then often segfaults on exit: the file decides.
            subprocess.run([qfp, patches, str(i), cfg], cwd=QW_DIR, capture_output=True)
            if not os.path.exists(out):
                break
            got = faces_in(out)
            log(f"QuadWild: quad size x{scale:.2f} -> {got} faces")
            if not best or abs(got - a.faces) < abs(best[1] - a.faces):
                best = (out, got)
            if got and abs(got - a.faces) <= a.faces * 0.12:
                break
            scale *= math.sqrt(max(got, 1) / a.faces)
        if not best:
            log("QuadWild could not quadrangulate this mesh")
            return None

        before = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=best[0])
        low = [o for o in bpy.context.scene.objects if o not in before][0]
        low.name = a.name
        low.data.materials.clear()
        only(low)
        if a.symmetry:
            log("note: QuadWild has no symmetry option; symmetry is ignored")
        return low
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def remesh(high, diag, a):
    if not a.tris and a.engine == "quadwild":
        low = quadwild(high, diag, a)
        if low:
            bpy.ops.object.shade_smooth()
            log("low:", stats(low))
            return low
        log("falling back to QuadriFlow")
    if a.tris:
        low = shell(high, diag, a, a.voxel)
        m = low.modifiers.new("dec", "DECIMATE")
        m.ratio = min(1.0, a.faces / max(1, len(low.data.polygons)))
        bpy.ops.object.modifier_apply(modifier=m.name)
    else:
        # QuadriFlow refuses some shells (reporting it only as a warning). A coarser
        # voxel is the reliable cure, so retry before giving up.
        base = a.voxel if a.voxel > 0 else 0.006
        tries = ([0] if a.voxel == 0 else []) + [base, base * 1.6, base * 2.5, base * 4]
        for v in tries:
            low = shell(high, diag, a, v)
            before = len(low.data.polygons)
            bpy.ops.object.quadriflow_remesh(
                mode="FACES", target_faces=a.faces, use_mesh_symmetry=a.symmetry,
                use_preserve_sharp=False, use_preserve_boundary=False,
                smooth_normals=False, seed=0)
            got = len(low.data.polygons)
            if got != before and got <= a.faces * 2:
                break
            log(f"QuadriFlow refused this shell ({before} -> {got} faces); retrying coarser")
            bpy.data.objects.remove(low, do_unlink=True)
        else:
            raise SystemExit("QuadriFlow could not remesh this model at any voxel size. "
                             "Try --tris, or a cleaner input mesh.")
    bpy.ops.object.shade_smooth()
    log("low:", stats(low))
    return low


def unwrap(low):
    only(low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    # A mesh stored as loose triangles (many exporters do this) unwraps into one island
    # per face, packed so small the bake has almost nowhere to land. Weld first.
    bpy.ops.mesh.remove_doubles(threshold=1e-6)
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.004)
    bpy.ops.uv.pack_islands(margin=0.004)
    bpy.ops.object.mode_set(mode="OBJECT")


def cycles_device():
    """CPU unless GEN3D_BAKE_GPU=1. A distro Blender ships no prebuilt GPU kernels
    and tries to compile them with the system compiler -- which on new Ubuntu fails
    and can leave the bake blank. Baking three maps on a modern CPU takes seconds."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.device = "CPU"
    if os.environ.get("GEN3D_BAKE_GPU") == "1":
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
    log("baking on CPU")


def op(fn, **kw):
    """Call an exporter with only the options this Blender's version of it has.
    Option names move between Blender releases (5.0 dropped the FBX exporter's
    use_selection), and an unknown keyword is a hard error."""
    have = set(fn.get_rna_type().properties.keys())
    use = {k: v for k, v in kw.items() if k in have}
    gone = sorted(set(kw) - set(use))
    if gone:
        log(f"note: {fn.idname_py()} in this Blender has no {', '.join(gone)}")
    return fn(**use)


def bake(high, low, diag, a, outdir, maps=("basecolor", "normal", "roughness")):
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
    jobs = {"basecolor": (base, "DIFFUSE", dict(pass_filter={"COLOR"})),
            "normal": (nrm, "NORMAL", dict(normal_space="TANGENT")),
            "roughness": (rough, "ROUGHNESS", {})}
    for node, kind, extra in (jobs[m] for m in maps):
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
    if "roughness" in maps:
        nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if "normal" in maps:
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
    # Some Blender versions can't export "selected only", so make the scene hold
    # nothing else -- then exporting everything is the same thing.
    for o in list(bpy.context.scene.objects):
        if o not in objs:
            bpy.data.objects.remove(o, do_unlink=True)
    op(bpy.ops.export_scene.fbx, filepath=fbx, use_selection=True, path_mode="COPY",
       embed_textures=True, apply_scale_options="FBX_SCALE_ALL",
       mesh_smooth_type="FACE", use_triangles=False)
    log("wrote", fbx)
    if a.glb:
        glb = os.path.join(outdir, a.name + ".glb")
        op(bpy.ops.export_scene.gltf, filepath=glb, export_format="GLB", use_selection=True)
        log("wrote", glb, "(triangulated -- glTF has no quads)")
    preview(objs, outdir)


def preview(objs, outdir):
    """What the web viewer shows: a textured GLB (browsers can't draw quads, so
    it's triangulated) plus the REAL quad edges as line pairs, so the wireframe
    shows the topology you'll get in the FBX rather than the triangulation."""
    import json
    import struct
    op(bpy.ops.export_scene.gltf, filepath=os.path.join(outdir, "preview.glb"),
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
    fix_materials(meshes)

    if a.mode == "clean":
        diag = ground(meshes)
        log("clean:", ", ".join(stats(o) for o in meshes))
        export(meshes, a, outdir)
        return

    if a.mode == "transfer":
        # --high is the painted (reduced) model, --dense the full-count one. Same shape,
        # so the painted colour bakes straight across; the dense mesh keeps its own
        # geometry, which is the whole point, so no normal map.
        src = join(meshes)
        before = set(bpy.context.scene.objects)
        new = import_any(os.path.abspath(a.dense))
        dense = join([o for o in new if o not in before])
        dense.name = a.name
        # Its GLB brings an empty material; left in slot 0, every face keeps using it and
        # the bake lands in a material nothing shows -- a black texture.
        dense.data.materials.clear()
        diag = ground([src, dense])
        log("transfer:", stats(src), "->", stats(dense))
        unwrap(dense)
        bake(src, dense, diag, a, outdir, maps=("basecolor", "roughness"))
        bpy.data.objects.remove(src, do_unlink=True)
        export([dense], a, outdir)
        log("done:", stats(dense))
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
    # Blender exits 0 even when the script it ran threw, so a crash here would look
    # like success to the caller. Report it with a real exit code.
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    # Everything is written by now. Blender can segfault while tearing itself
    # down after glTF exports, which would turn a finished job into a failed
    # one -- so skip the teardown entirely.
    sys.stdout.flush()
    os._exit(0)
