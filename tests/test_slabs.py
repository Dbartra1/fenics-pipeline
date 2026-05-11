# tests/test_slabs.py
# Run from repo root: python3 tests/test_slabs.py
#
# Verifies Phase 5 two-region domain architecture with corrected bracket BC:
#   - Attachment slab bc=none  (slab is solid but does not own fixity)
#   - load_case.fixed = corner disks at wall bolt positions on x_min
#   - load_case.load  = full x_max face, gravity -Z
#
# This is the corrected single-load-case setup. Multi-load case and NEMA bolt
# pattern selector tests will be added in the next session.

import sys, json, importlib.util, types
import numpy as np

# ── Module loader ─────────────────────────────────────────────────────────────
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ps  = load_mod("param_schema",           "src/geometry/param_schema.py")
sys.modules["src"]                       = types.ModuleType("src")
sys.modules["src.geometry"]              = types.ModuleType("src.geometry")
sys.modules["src.geometry.param_schema"] = ps

rf  = load_mod("region_factory",         "src/geometry/region_factory.py")
sys.modules["src.geometry.region_factory"] = rf

vox = load_mod("voxelize",              "scripts/voxelize.py")

PASS = "✓"
FAIL = "✗"
errors = []

def check(label, cond, detail=""):
    tag = PASS if cond else FAIL
    print(f"  {tag} {label}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        errors.append(label)

# ── 1. params.json parse + validate ──────────────────────────────────────────
print("\n══ 1. params.json parse + validate ══════════════════════════════")
raw    = json.loads(open("scad/motor_mount_params.json").read())
params = ps.PipelineParams.from_dict(raw)
params.validate()
ar  = params.attachment_regions[0]
lc  = params.load_case_config
fix = lc.fixed
ld  = lc.load

check("attachment_regions present",        len(params.attachment_regions) == 1)
check("slab type = slab_x",               ar.type == "slab_x")
check("a_min=0, a_max=0.006",             ar.a_min == 0.0 and ar.a_max == 0.006)
check("slab bc = none (not fixed_full)",   ar.bc == "none")
check("4 bolt_voids",                     len(ar.bolt_voids) == 4)
check("bolt void radius > filter_radius", ar.bolt_voids[0].radius_m > 0.003)
check("nondesign_regions empty",          len(params.nondesign_regions) == 0)
check("load_case.fixed present",          fix is not None)
check("fixed face = x_min",              fix.face == "x_min")
check("fixed selector = corners",         fix.selector == "corners")
check("fixed inset_m = 0.010",           fix.inset_m == 0.010)
check("fixed disk_radius_m > void_r",    fix.disk_radius_m > ar.bolt_voids[0].radius_m,
      f"{fix.disk_radius_m} > {ar.bolt_voids[0].radius_m}")
check("load face = x_max",               ld.face == "x_max")
check("load selector = full",             ld.selector == "full")
check("load direction is -Z",             ld.direction == [0.0, 0.0, -1.0])
check("bolt_seats count = 1",            len(params.bolt_seats) == 1)

# ── 2. voxelize_domain ────────────────────────────────────────────────────────
print("\n══ 2. voxelize_domain — slab + bolt voids ═══════════════════════")
from src.geometry.region_factory import resolve_geometry_regions
resolved_voids, resolved_nondesign = resolve_geometry_regions(params)

h  = raw["simp"]["voxel_size_mm"] / 1000.0
nx = round(params.geometry.length / (h * 1000))
ny = round(params.geometry.width  / (h * 1000))
nz = round(params.geometry.height / (h * 1000))
grid = {"nx": nx, "ny": ny, "nz": nz, "voxel_size": h}
print(f"  grid: {nx}×{ny}×{nz}  voxel {h*1000:.1f}mm")

nd, vm = vox.voxelize_domain(
    geometry_params    = params.geometry,
    grid_config        = grid,
    nondesign_regions  = resolved_nondesign,
    void_regions       = resolved_voids,
    bolt_seats         = params.bolt_seats,
    attachment_regions = params.attachment_regions,
)
slab_slices = int(round(ar.a_max / h))
nd_in_slab  = nd[:, :, :slab_slices].sum()
vm_in_slab  = vm[:, :, :slab_slices].sum()
overlap     = ((nd == 1) & (vm == 1)).sum()

check("nondesign shape correct",         nd.shape == (nz, ny, nx))
check("slab has nondesign voxels",       nd_in_slab > 0,   f"{nd_in_slab}")
check("bolt voids punched in slab",      vm_in_slab > 0,   f"{vm_in_slab}")
check("nd/vm overlap = 0",               overlap == 0,      f"overlap={overlap}")
total = nx * ny * nz
free  = total - nd.sum() - vm.sum()
print(f"  voxel budget: {total} total, {nd.sum()} nondesign, "
      f"{vm.sum()} void, {free} free ({free/total*100:.1f}%)")
check("free voxels > 0",                free > 0)

# ── 3. build_load_case — corner disk fixity ───────────────────────────────────
print("\n══ 3. build_load_case — corner disk fixity ══════════════════════")
lc_result = vox.build_load_case(
    geometry_params    = params.geometry,
    load_hints         = params.load_hints,
    grid_config        = grid,
    load_case_config   = params.load_case_config,
    attachment_regions = params.attachment_regions,
)

# Corner disks are small patches — should be much less than full face
full_face_dofs = (ny + 1) * (nz + 1) * 3
fixed_count    = len(lc_result["fixed_dofs"])
load_count     = len(lc_result["load_dofs"])

check("fixed_dofs non-empty",            fixed_count > 0,
      f"{fixed_count} DOFs")
check("fixed_dofs << full face",         fixed_count < full_face_dofs * 0.5,
      f"{fixed_count} < {full_face_dofs//2} (half of full face)")
check("load_dofs = full x_max face",     load_count == (ny + 1) * (nz + 1),
      f"got {load_count}, expected {(ny+1)*(nz+1)} (1 DOF/node, -Z direction)")
check("load_vals sum ≈ -5000 N",         abs(abs(lc_result["load_vals"].sum()) - 5000.0) < 1.0,
      f"{lc_result['load_vals'].sum():.2f} N")
check("no DOF overlap fixed/load",       len(np.intersect1d(
                                             lc_result["fixed_dofs"],
                                             lc_result["load_dofs"])) == 0)

print(f"  fixed DOFs  : {fixed_count:,}  ({fixed_count/3} nodes)")
print(f"  load DOFs   : {load_count:,}")

# ── 4. Verify fixed nodes are inside the slab ─────────────────────────────────
print("\n══ 4. Fixed nodes inside slab region ════════════════════════════")
# Fixed nodes are on x_min face (ix=0). Check their y,z coords fall near
# the expected bolt positions.
h_val = grid["voxel_size"]
# Node coord: node index on x_min face = iy*(nx+1) + iz*(nx+1)*(ny+1)
# Fixed DOFs are multiples of 3 for DOF 0,1,2 of each node
fixed_nodes = sorted(set(int(d) // 3 for d in lc_result["fixed_dofs"]))
bolt_centers_yz = [
    (0.010, 0.010), (0.050, 0.010),
    (0.010, 0.070), (0.050, 0.070),
]
# Check that we have nodes near each bolt center
nodes_near_bolt = 0
for bc_y, bc_z in bolt_centers_yz:
    for n in fixed_nodes:
        # node layout on x_min: n = iy*(nx+1) + iz*(nx+1)*(ny+1), ix=0
        # so n % (nx+1) == 0 for all x_min nodes
        iy = (n % ((nx+1)*(ny+1))) // (nx+1)
        iz = n // ((nx+1)*(ny+1))
        y  = iy * h_val
        z  = iz * h_val
        if abs(y - bc_y) < fix.disk_radius_m + h_val and \
           abs(z - bc_z) < fix.disk_radius_m + h_val:
            nodes_near_bolt += 1
            break

check("fixed nodes near all 4 bolt centers",
      nodes_near_bolt == 4,
      f"{nodes_near_bolt}/4 bolt centers covered")

# ── 5. Guard — slab bc=none with no load_case.fixed raises ───────────────────
print("\n══ 5. guard — bc=none + no fixed raises ═════════════════════════")
# Build a params object with bc=none and no load_case.fixed
try:
    bad = ps.PipelineParams.from_dict({
        **{k: v for k, v in raw.items() if k != "load_case"},
        "load_case": {"load": raw["load_case"]["load"]},
        "attachment_regions": [
            {**{k: v for k, v in raw["attachment_regions"][0].items()
                if not k.startswith("_")},
             "bc": "none"}
        ]
    })
    bad.validate()
    check("missing-fixity guard fires", False)
except AssertionError:
    check("missing-fixity guard fires", True)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"══ {len(errors)} test(s) FAILED ══════════════════════════════════════════\n")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("══ All tests passed ══════════════════════════════════════════════\n")
