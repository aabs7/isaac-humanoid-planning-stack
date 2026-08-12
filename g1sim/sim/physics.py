"""Making apartment objects physically real enough to grasp.

The InteriorAgent apartment ships with colliders, but nothing in it is authored for
manipulation, and the gaps are not the ones you would guess. Verified against
``kujiale_0021.usda``:

* **Every pickable object is ``physics:kinematicEnabled = True``** -- infinite mass, immune
  to contact forces. A hand closing on one does not move it; the object pushes the hand.
  This alone makes a real grasp impossible, and it is invisible until you try.
* **No ``MassAPI`` anywhere on the stage.** PhysX would derive mass from convex-hull volume
  times a default density, which across this apartment ranges from 0.009 kg (a spoon, below
  reliable contact-solver resolution) to 2.8 kg (an "ornament" a hand should not lift).
* **No physics material** -- default friction only, marginal for lifting a smooth cup.
* The prim the semantic map records is a physics-free ``Xform``; the rigid body and collider
  live one level deeper, at ``<prim_path>/Meshes/<name>``. Everything here resolves that.

So a target object has to be re-authored before PhysX parses the stage. That timing is not
negotiable: :func:`g1sim.sim.scene.build_world` authors inside ``prepare_apartment()`` and
then calls ``sim.reset()``, which is where the parse happens. Flipping these attributes
afterwards is not reliably picked up.

Deliberately *not* handled here: the doubly-nested, often non-uniform scale above the rigid
body (78 of 97 objects), the 6 objects with mirrored (negative-determinant) scale, and the 57
whose several meshes fuse into a single convex hull -- so the collider a finger touches is
not always the shape you see (``cup_0000``'s hull includes its saucer). Those are asset bugs;
:func:`describe_physics` reports them so a bad target is diagnosed rather than debugged.
"""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

# Explicit masses, in kg, for categories we expect to pick up. Chosen to be physically
# sensible for a hand rather than derived from hull volume (which gets these badly wrong in
# both directions -- see the module docstring).
CATEGORY_MASS_KG = {
    "cup": 0.25,
    "bowl": 0.30,
    "plate": 0.20,
    "spoon": 0.05,
    "book": 0.40,
    "flower": 0.10,
    "ornament": 0.30,
    "vase": 0.50,
}
DEFAULT_MASS_KG = 0.25

# A grippy-but-plausible pair. The stage authors no material at all, so without this we get
# Isaac's default (mu ~= 0.5), which is marginal for a three-finger lift of a smooth cup.
GRASP_STATIC_FRICTION = 0.9
GRASP_DYNAMIC_FRICTION = 0.8

PHYSICS_MATERIAL_PATH = "/World/PhysicsMaterials/GraspableMaterial"


def physics_prim(stage, prim_path: str):
    """The descendant of ``prim_path`` that actually carries the rigid body.

    The semantic map records ``/World/Apartment/Meshes/kitchen_753/cup_0000``, a plain Xform
    with no physics on it at all; the ``RigidBodyAPI`` is on
    ``.../cup_0000/Meshes/cup_0000``. Every caller wants the latter, and no caller should
    have to know that. Returns ``None`` if there is no rigid body in the subtree.
    """
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return None


def ensure_physics_scene(stage) -> bool:
    """Make sure the stage has a ``UsdPhysics.Scene``; return True if one had to be created.

    The apartment USD authors none. Isaac's ``SimulationContext`` normally creates one, so
    this is usually a no-op -- but it is cheap insurance against a silently gravity-free
    world, which would look exactly like a grasp that works suspiciously well.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return False
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    print("[physics] no UsdPhysics.Scene on stage -- created /physicsScene")
    return True


def grasp_material(stage):
    """Get-or-create the shared high-friction physics material for graspable objects."""
    existing = stage.GetPrimAtPath(PHYSICS_MATERIAL_PATH)
    if existing and existing.IsValid():
        return UsdShade.Material(existing)

    material = UsdShade.Material.Define(stage, PHYSICS_MATERIAL_PATH)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(GRASP_STATIC_FRICTION)
    api.CreateDynamicFrictionAttr(GRASP_DYNAMIC_FRICTION)
    api.CreateRestitutionAttr(0.0)
    return material


def make_graspable(stage, prim_path: str, *, mass_kg: float = DEFAULT_MASS_KG,
                   verbose: bool = True) -> bool:
    """Turn one apartment object into a dynamic, grabbable rigid body.

    Undoes what :func:`g1sim.sim.scene.make_apartment_static` did to this one prim and fixes
    the two authoring gaps that make contact grasping impossible: the kinematic flag and the
    missing mass. Must be called **before** ``sim.reset()``.

    Returns False (with a reason logged) if the object has no rigid body to work with, which
    is true of a handful of prims -- doors, windows, and ``cabinet_0009``.
    """
    prim = physics_prim(stage, prim_path)
    if prim is None:
        print(f"[physics] {prim_path}: no RigidBodyAPI in subtree -- cannot make graspable")
        return False

    body = UsdPhysics.RigidBodyAPI(prim)
    body.CreateRigidBodyEnabledAttr(True)
    # The load-bearing line. Authored True for all 97 pickable objects; a kinematic body has
    # infinite mass and ignores contact, so fingers cannot move it.
    body.CreateKinematicEnabledAttr(False)

    mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) \
        else UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))

    UsdShade.MaterialBindingAPI(prim).Bind(
        grasp_material(stage), UsdShade.Tokens.weakerThanDescendants, "physics")

    if verbose:
        print(f"[physics] {prim.GetPath()}: dynamic, {mass_kg:.2f} kg, "
              f"mu={GRASP_STATIC_FRICTION}/{GRASP_DYNAMIC_FRICTION}")
    return True


def mass_for(category: str) -> float:
    """A sensible mass for a semantic category (see :data:`CATEGORY_MASS_KG`)."""
    return CATEGORY_MASS_KG.get(category, DEFAULT_MASS_KG)


def describe_physics(stage, prim_path: str) -> dict:
    """What PhysX will actually see for this object, plus the asset problems that would make
    a grasp behave strangely.

    Reported because each of these was found the hard way, and each produces a *plausible*
    but wrong grasp rather than an error: a non-uniform scale above the rigid body skews the
    collider, a mirrored scale is invalid mesh scale for PhysX, and fused sub-meshes mean the
    shape the fingers hit is not the shape on screen.
    """
    out = {"prim_path": prim_path, "found": False}
    prim = physics_prim(stage, prim_path)
    if prim is None:
        return out

    def attr(p, name):
        a = p.GetAttribute(name)
        return a.Get() if a and a.IsValid() else None

    scales = []
    walk = prim
    while walk and walk.IsValid() and str(walk.GetPath()) != "/":
        s = attr(walk, "xformOp:scale")
        if s is not None:
            scales.append((str(walk.GetPath()), tuple(round(v, 4) for v in s)))
        walk = walk.GetParent()

    meshes = [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    nonuniform = [(p, s) for p, s in scales if len(set(abs(v) for v in s)) > 1]
    mirrored = [(p, s) for p, s in scales if s[0] * s[1] * s[2] < 0]

    out.update(
        found=True,
        body_prim=str(prim.GetPath()),
        applied_schemas=list(prim.GetAppliedSchemas()),
        rigid_body_enabled=attr(prim, "physics:rigidBodyEnabled"),
        kinematic_enabled=attr(prim, "physics:kinematicEnabled"),
        collision_enabled=attr(prim, "physics:collisionEnabled"),
        approximation=attr(prim, "physics:approximation"),
        mass=attr(prim, "physics:mass"),
        collider_on_mesh=any(p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
                             for p in Usd.PrimRange(prim)),
        n_meshes=len(meshes),
        scales=scales,
        nonuniform_scale=nonuniform,
        mirrored_scale=mirrored,
    )
    return out


def physics_warnings(info: dict) -> list:
    """Human-readable problems from a :func:`describe_physics` result, worst first."""
    if not info.get("found"):
        return [f"{info['prim_path']}: no rigid body in subtree"]

    warnings = []
    if info["kinematic_enabled"]:
        warnings.append("kinematic: infinite mass, immune to contact -- cannot be grasped")
    if not info["rigid_body_enabled"]:
        warnings.append("rigid body disabled: static collider, will not move")
    if info["mass"] is None:
        warnings.append("no mass authored: PhysX will derive it from hull volume "
                        "(0.009-2.8 kg across this apartment)")
    if not info["collision_enabled"]:
        warnings.append("collision disabled: nothing to push against")
    if not info["collider_on_mesh"]:
        warnings.append("CollisionAPI is on an Xform, not a Mesh -- verify PhysX built a "
                        "collider at all")
    if info["n_meshes"] > 1:
        warnings.append(f"{info['n_meshes']} meshes fuse into one convex hull -- the "
                        f"collider is not the visible shape")
    for path, scale in info["mirrored_scale"]:
        warnings.append(f"mirrored scale {scale} at {path}: invalid mesh scale for PhysX")
    for path, scale in info["nonuniform_scale"]:
        warnings.append(f"non-uniform scale {scale} at {path}: skews the collider")
    return warnings
