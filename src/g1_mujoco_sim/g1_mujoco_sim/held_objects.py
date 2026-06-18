#!/usr/bin/env python3
"""A small library of held-object (payload) presets for testing the prior-free
held-object self-filter against a *variety* of shapes.

The filter (g1_sim_node, held_filter_mode = connected | carry_volume | online)
discovers the carried object from the LiDAR/RGBD returns alone -- it is NOT told
the shape. To exercise that, we need to put different shapes in the robot's hand
without hand-editing the scene XML each run. Each preset below describes the
geom(s) of the `held_object` body; `build_model()` loads the scene, swaps in the
preset, and compiles a ready-to-use `MjModel`.

Pick a preset on the command line:

    ros2 launch g1_nav2_bringup bringup.launch.py payload:=pole

`payload:=scene` keeps whatever geom is already in the scene XML (no swap).

Editing is done through MuJoCo's own model editor (`mujoco.MjSpec`), so the
scene's relative `<include>`/mesh paths resolve exactly as in a normal load (and
MuJoCo's parser tolerates the scene's free-text comments, which stricter XML
parsers reject).

Each geom dict holds MuJoCo `<geom>` fields. Geoms are defined in the
held_object body frame, whose origin is the in-hand grasp point
(`held_object_pos` on the grasp link), so a geom at the origin sits in the hand
and the connectivity seed reaches it. `size` follows MuJoCo conventions:
box -> 3 half-extents, sphere -> radius, cylinder -> "radius half-length"
(local +z axis; rotate with `quat` to lay it down). Material fields
(rgba/contype/conaffinity/group) are applied to every geom so each payload is
non-colliding and in the LiDAR/camera geom group.
"""
import mujoco


# The held_object body origin is the grasp point on the body midline (held_object_pos
# brings it to y=0, ~0.17 m in front of the wrists); the two hands sit at y = +/-0.149,
# z = +0.04 in this frame. So a payload only looks GRIPPED if it reaches a hand:
# wide objects (box/sphere) span out to the hands on their own; thin/long ones are
# placed at the right hand (RH below) or laid across both hands so a hand closes on
# them instead of floating on the midline. Every preset still overlaps the grasp
# point so the connectivity seed (held_connect_seed ~ 0.15 m) catches it.
RH = (-0.17, -0.149, 0.04)   # right-hand position in the held_object body frame

# cylinder long axis (local +z) -> body +x, so a rod points forward out of the hand.
# quat is (w, x, y, z): a -90 deg rotation about Y.
_Z_TO_X = "0.7071068 0 0.7071068 0"


PRESETS = {
    # ---- basic primitives ---------------------------------------------------
    "box": {
        "desc": "compact box ~0.20 x 0.36 x 0.24 m (the default carried payload)",
        "geoms": [dict(type="box", size="0.10 0.18 0.12")],
    },
    "sphere": {
        "desc": "ball, r = 0.13 m (a rounded payload)",
        "geoms": [dict(type="sphere", size="0.13")],
    },
    "cylinder": {
        "desc": "upright canister, r = 0.06 m, length 0.36 m, gripped in the "
                "right hand (a bottle held one-handed)",
        # centred on the right hand and raised so the hand closes on its lower body
        "geoms": [dict(type="cylinder", size="0.06 0.18", pos="-0.12 -0.149 0.10")],
    },
    # ---- shapes that stress the prior-free filter ---------------------------
    "pole": {
        "desc": "long thin rod, r = 0.03 m, length ~1.0 m, gripped in the right "
                "hand and pointing forward (a broom/pole). Its far end reaches "
                "~0.9 m -- far beyond the carry_volume radius (0.32 m), so "
                "carry_volume leaves the tip as a phantom obstacle while connected "
                "removes it end-to-end.",
        # axis along body +x at the right-hand line; spans x in [-0.25, 0.75] so the
        # hand grips near the back of the handle and the rod sweeps the grasp point.
        "geoms": [dict(type="cylinder", size="0.03 0.50", pos="0.25 -0.149 0.04",
                       quat=_Z_TO_X)],
    },
    "board": {
        "desc": "flat tray ~0.32 x 0.36 x 0.04 m carried level in both hands: wide "
                "and thin, a poor fit for a single sphere or a fat carry volume",
        # flat in the x-y plane (thin in z), wide in y so both hands close on its
        # sides; held level at grasp height so it does not present a tall near-field
        # surface that the region-grow could bridge into the floor.
        "geoms": [dict(type="box", size="0.16 0.18 0.02", pos="0.02 0 0.03")],
    },
    "lshape": {
        "desc": "L-shaped two-box composite: a non-convex, multi-part payload no "
                "single primitive can describe (exercises multi-geom handling "
                "and connectivity around a corner)",
        "geoms": [
            # arm A: along +x at the right-hand line; the hand grips its back end
            dict(type="box", size="0.13 0.04 0.04", pos="0.00 -0.149 0.04"),
            # arm B: a short up-turn (+z) at the far end of arm A -> forms the
            # corner. Kept short so the L does not rise into a tall near-field
            # surface that the region-grow could bridge into the world.
            dict(type="box", size="0.04 0.04 0.10", pos="0.09 -0.149 0.10"),
        ],
    },
}

# values that mean "do not swap; use the geom already in the scene XML"
_SCENE_KEYWORDS = {"", "scene", "keep", "none", "default"}

_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}

_RGBA = [0.10, 0.60, 0.90, 1.0]


def is_scene_keyword(preset):
    """True if `preset` means 'keep the scene XML geom' (no swap)."""
    return preset is None or str(preset).strip().lower() in _SCENE_KEYWORDS


def preset_names():
    """Selectable preset names, in declaration order."""
    return list(PRESETS.keys())


def _vec(value, n, default):
    """Parse a "a b c" string / sequence into a length-n float list, padded with
    `default`'s tail."""
    if value is None:
        vals = []
    elif isinstance(value, (list, tuple)):
        vals = [float(x) for x in value]
    else:
        vals = [float(x) for x in str(value).split()]
    vals = vals + list(default)[len(vals):]
    return vals[:n]


def apply_preset(spec, preset):
    """Replace the `held_object` body's geom(s) in an `MjSpec` with `preset`."""
    if preset not in PRESETS:
        raise ValueError(
            f"unknown payload preset '{preset}'; choose one of "
            f"{preset_names()} or 'scene' to keep the XML geom")
    body = spec.body("held_object")
    if body is None:
        raise RuntimeError("scene has no <body name='held_object'> to swap into")

    for g in list(body.geoms):
        spec.delete(g)

    for i, gspec in enumerate(PRESETS[preset]["geoms"]):
        gtype = gspec["type"]
        if gtype not in _TYPES:
            raise ValueError(f"preset '{preset}' uses unsupported geom type '{gtype}'")
        g = body.add_geom()
        # the first geom keeps the canonical name the sim looks up by name
        # (shape mode / id resolution); extra geoms get suffixed names.
        g.name = "held_object_geom" if i == 0 else f"held_object_geom_{i + 1}"
        g.type = _TYPES[gtype]
        g.size = _vec(gspec.get("size"), 3, (0.0, 0.0, 0.0))
        g.pos = _vec(gspec.get("pos"), 3, (0.0, 0.0, 0.0))
        g.quat = _vec(gspec.get("quat"), 4, (1.0, 0.0, 0.0, 0.0))
        g.rgba = list(_RGBA)
        g.contype = 0
        g.conaffinity = 0
        g.group = 1


def build_model(scene_xml, preset):
    """Load `scene_xml`, swap the held_object body to `preset`, return a compiled
    `mujoco.MjModel`."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    apply_preset(spec, preset)
    return spec.compile()
