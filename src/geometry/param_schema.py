# src/geometry/param_schema.py
#
# Typed schema for pipeline parameters.
# Validates params.json at load time so bad values fail here,
# not mid-solve in 03_fea_fenicsx.ipynb.
#
# Phase 1: GeometryParams now accepts arbitrary extra fields and passes
#           them all through to OpenSCAD as -D defines.
# Phase 2: load_case_config drives face selection in voxelize.py,
#           nondesign_regions drives void/nondesign mask generation.
#           Both fall back to legacy behavior if absent.
# Phase 5: attachment_regions — forced-solid slabs that own both the
#           nondesign voxels AND the boundary condition for the solver.
#           LoadCaseConfig.fixed is now Optional; when None, the BC is
#           derived entirely from attachment_regions.
# Session 6: LoadFaceConfig gains bolt_pattern selector + bolt_centers_m field.
#             BoltSeatRegion gains optional load field (architectural stub;
#             wired into build_load_case in Phase 6).
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


class GeometryParams:
    """
    Geometry parameters for a part.

    Required: length, width, height  (used to compute voxel grid dimensions)
    Optional: any additional fields — stored as attributes and passed
              through to OpenSCAD as uppercase -D defines.

    This design allows arbitrary SCAD files without modifying this class.
    """
    def __init__(self, length: float, width: float, height: float, **kwargs):
        self.length = float(length)
        self.width  = float(width)
        self.height = float(height)
        # Store extra fields as attributes (e.g. wall_thickness, fillet_radius)
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Keep a full dict for OpenSCAD define generation
        self._all_fields: Dict[str, Any] = {
            "length": self.length,
            "width":  self.width,
            "height": self.height,
            **kwargs,
        }

    def get(self, key: str, default=None):
        """Safe attribute access with default — used by voxelize.py."""
        return getattr(self, key, default)

    def validate(self) -> None:
        assert self.length > 0, "length must be > 0"
        assert self.width  > 0, "width must be > 0"
        assert self.height > 0, "height must be > 0"

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self._all_fields.items())
        return f"GeometryParams({fields})"


@dataclass
class MeshHints:
    target_element_size:        float
    opt_domain_element_size_mm: float = 2.5
    refinement_regions:         List  = field(default_factory=list)

    def validate(self) -> None:
        assert self.target_element_size > 0, "target_element_size must be > 0"
        assert self.opt_domain_element_size_mm > 0, \
            "opt_domain_element_size_mm must be > 0"


@dataclass
class BoundaryConditions:
    """
    Describes how the part is fixed and loaded for FEA and topology optimization.
    Kept for backward compatibility with Stage 03 (FEniCSx path).
    For the Rust solver path, use load_case_config instead.
    """
    fixed_face:           str   = "corners"
    load_face:            str   = "top"
    load_direction:       List  = field(default_factory=lambda: [0.0, 0.0, -1.0])
    hole_inset_fraction:  float = 0.15
    shell_thickness_mm:   float = 2.0

    VALID_FACES = {"top", "bottom", "left", "right", "front", "back", "corners"}

    def validate(self) -> None:
        assert self.fixed_face in self.VALID_FACES, \
            f"fixed_face must be one of {self.VALID_FACES}"
        assert self.load_face in self.VALID_FACES - {"corners"}, \
            f"load_face must be one of {self.VALID_FACES - {'corners'}}"
        assert self.fixed_face != self.load_face, \
            "fixed_face and load_face cannot be the same"
        assert 0.0 < self.hole_inset_fraction < 0.5, \
            "hole_inset_fraction must be between 0 and 0.5"
        assert len(self.load_direction) == 3, \
            "load_direction must be a 3-element list [x, y, z]"
        assert self.shell_thickness_mm >= 0, \
            "shell_thickness_mm must be >= 0"


@dataclass
class FixedFaceConfig:
    """
    Declarative fixed BC specification for the Rust solver path.

    face:           which face to fix — "x_min","x_max","y_min","y_max","z_min","z_max"
    selector:       "full"      → fix all nodes on that face
                    "corners"   → fix disks at the 4 face-bbox corners
                                  (intended for rectangular parts)
                    "leg_holes" → fix N disks at leg-hole positions on a disk
                                  shape. N, radius, and first angle are read
                                  from the geometry dict:
                                      geometry.num_legs         (default 3)
                                      geometry.leg_hole_radius  (mm, required)
                                      geometry.first_leg_angle  (deg, default 90)
                                  The disk pattern is laid out about the part's
                                  XY centre (see region_factory.part_center_m).
    inset_m:        corner inset distance in metres (selector="corners" only)
    disk_radius_m:  radius of each fixed disk in metres (used by BOTH
                    "corners" and "leg_holes"; typically leg_hole_d/2 +
                    ~1–2 mm clearance so the fixed region contains the full
                    bolt-shank contact patch).
    """
    face:           str   = "z_min"
    selector:       str   = "corners"
    inset_m:        float = 0.010
    disk_radius_m:  float = 0.005

    VALID_FACES     = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    VALID_SELECTORS = {"full", "corners", "leg_holes", "center_disk"}

    def validate(self) -> None:
        assert self.face in self.VALID_FACES, \
            f"fixed face must be one of {self.VALID_FACES}, got '{self.face}'"
        assert self.selector in self.VALID_SELECTORS, \
            f"selector must be one of {self.VALID_SELECTORS}, got '{self.selector}'"
        assert self.inset_m > 0, "inset_m must be > 0"
        assert self.disk_radius_m > 0, "disk_radius_m must be > 0"


@dataclass
class LoadFaceConfig:
    """
    Declarative load specification for the Rust solver path.

    face:           which face receives the traction load
    selector:       "full"         → distribute load uniformly across the
                                     entire face (legacy / default behaviour)
                    "center_disk"  → concentrate load on a single disk centred
                                     on the face. Use when load runs through a
                                     central mounting feature (e.g. tripod bolt).
                    "bolt_pattern" → concentrate load on N disks, one per bolt
                                     position. Use for bolted connections where
                                     the motor/load transfers force through
                                     discrete fasteners, not face pressure.
                                     Requires bolt_centers_m (list of [a,b]
                                     face-local coordinate pairs in metres).
    direction:      [x, y, z] unit vector (normalised internally)
    magnitude_n:    total force in Newtons, distributed across selected DOFs
    disk_radius_m:  radius of the loaded disk/disks in metres.
                    For "center_disk": single central disk.
                    For "bolt_pattern": radius of each individual bolt disk
                    (typically matches bolt_seat wall_radius_m so load lands
                    on the collar region).
                    Ignored for "full".
    bolt_centers_m: list of [a, b] face-local center coordinates in metres.
                    Required when selector="bolt_pattern"; ignored otherwise.
                    Coordinate convention matches _face_node_coords:
                      x_min / x_max faces  →  a=Y,  b=Z
                      y_min / y_max faces  →  a=X,  b=Z
                      z_min / z_max faces  →  a=X,  b=Y
    """
    face:           str            = "z_max"
    selector:       str            = "full"
    direction:      List           = field(default_factory=lambda: [0.0, 0.0, -1.0])
    magnitude_n:    float          = 10000.0
    disk_radius_m:  float          = 0.010
    # bolt_pattern selector only: list of [a, b] in-plane center coords in metres.
    bolt_centers_m: Optional[List] = None
    # Multi-load (R1): scenario identity + relative importance in the aggregated
    # objective. `name` is used in diagnostics; `weight` feeds the weighted-sum
    # (and, later, the worst-case p-norm). Defaults make a single unnamed load
    # behave exactly as before.
    name:           str            = ""
    weight:         float          = 1.0

    VALID_FACES     = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    VALID_SELECTORS = {"full", "center_disk", "bolt_pattern"}

    def validate(self) -> None:
        assert self.face in self.VALID_FACES, \
            f"load face must be one of {self.VALID_FACES}, got '{self.face}'"
        assert self.selector in self.VALID_SELECTORS, \
            f"selector must be one of {self.VALID_SELECTORS}, got '{self.selector}'"
        assert len(self.direction) == 3, "direction must be [x, y, z]"
        assert self.magnitude_n > 0, "magnitude_n must be > 0"
        assert self.disk_radius_m > 0, "disk_radius_m must be > 0"
        assert self.weight >= 0, f"load weight must be >= 0, got {self.weight}"
        if self.selector == "bolt_pattern":
            assert self.bolt_centers_m is not None and len(self.bolt_centers_m) > 0, (
                "bolt_pattern selector requires bolt_centers_m "
                "(non-empty list of [a, b] pairs in metres)"
            )
            for c in self.bolt_centers_m:
                assert len(c) == 2, \
                    f"each bolt center must be [a, b], got {c}"


@dataclass
class LoadCaseConfig:
    """
    One or more load scenarios that SHARE a set of fixed supports.

    fixed  — shared Dirichlet supports (Optional). When None, fixity is derived
             from attachment_regions in voxelize.build_load_case() (the motor
             mount uses this — the attachment slab owns fixity).
    load   — back-compat single-scenario field. A params block with a single
             "load": {...} sets this; `loads` is then [load].
    loads  — full list of scenarios (R1 multi-load). A params block with
             "loads": [ {...}, {...} ] sets this; `load` mirrors loads[0] for
             back-compat accessors. Supports are shared across all of them.

    Per-case supports (a different `fixed` per scenario) are a future extension;
    today every scenario shares `fixed`.
    """
    fixed: Optional[FixedFaceConfig]      = None
    load:  LoadFaceConfig                 = field(default_factory=LoadFaceConfig)
    loads: Optional[List[LoadFaceConfig]] = None

    def __post_init__(self) -> None:
        # Normalise to a non-empty list. `load` (singular) and `loads` (list)
        # are kept in sync: whichever was supplied drives the other.
        if self.loads is None:
            self.loads = [self.load]
        else:
            assert len(self.loads) >= 1, "loads must be non-empty"
            self.load = self.loads[0]

    def validate(self) -> None:
        if self.fixed is not None:
            self.fixed.validate()
        seen = set()
        for i, ld in enumerate(self.loads):
            ld.validate()
            if self.fixed is not None:
                assert self.fixed.face != ld.face, \
                    f"fixed face and load face cannot be the same (load '{ld.name or i}')"
            nm = ld.name or f"lc{i}"
            assert nm not in seen, f"duplicate load case name '{nm}'"
            seen.add(nm)


@dataclass
class NondesignRegion:
    """
    A geometric region that is forced solid (nondesign) or void in the voxel grid.

    type:           "cylinder_z" | "cylinder_x" | "cylinder_y"
                    Cylinder axis determines which plane the center coords are in:
                    cylinder_z → centers in (x, y), cylinder_x → (y, z), cylinder_y → (x, z)
    centers_m:      list of [a, b] center coordinates in metres
    void_radius_m:  radius of the always-void core (the actual hole)
    wall_radius_m:  radius of the nondesign ring around the void (forced solid)
                    Set equal to void_radius_m to have no ring.
    """
    type:          str
    centers_m:     List
    void_radius_m: float
    wall_radius_m: float

    VALID_TYPES = {"cylinder_z", "cylinder_x", "cylinder_y"}

    def validate(self) -> None:
        assert self.type in self.VALID_TYPES, \
            f"type must be one of {self.VALID_TYPES}, got '{self.type}'"
        assert self.void_radius_m > 0, "void_radius_m must be > 0"
        assert self.wall_radius_m >= self.void_radius_m, \
            "wall_radius_m must be >= void_radius_m"
        assert len(self.centers_m) > 0, "centers_m must not be empty"
        for c in self.centers_m:
            assert len(c) == 2, "each center must be [a, b]"


@dataclass
class VoidRegion:
    """
    A region that is forced void in the voxel grid.

    Supported types:
      "box"                  — axis-aligned box; any of x/y/z_min/max may be
                               omitted (treated as unbounded). All bounds in metres.
      "cylinder_z_exterior"  — everything OUTSIDE a cylinder whose axis is parallel
                               to z, useful for masking the empty space around
                               circular parts (disks, rings) that sit in a
                               rectangular voxel grid.
                               Requires cx, cy, radius (metres).

    Examples:
        {"type": "box", "x_min": 0.020, "z_min": 0.020}
        {"type": "cylinder_z_exterior", "cx": 0.040, "cy": 0.040, "radius": 0.040}
    """
    type:  str
    # box fields
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    # cylinder_z_exterior fields
    cx:     Optional[float] = None
    cy:     Optional[float] = None
    radius: Optional[float] = None

    VALID_TYPES = {"box", "cylinder_z_exterior"}

    def validate(self) -> None:
        assert self.type in self.VALID_TYPES, \
            f"VoidRegion type must be one of {self.VALID_TYPES}, got '{self.type}'"

        if self.type == "box":
            if self.x_min is not None and self.x_max is not None:
                assert self.x_min <= self.x_max, "x_min must be <= x_max"
            if self.y_min is not None and self.y_max is not None:
                assert self.y_min <= self.y_max, "y_min must be <= y_max"
            if self.z_min is not None and self.z_max is not None:
                assert self.z_min <= self.z_max, "z_min must be <= z_max"

        elif self.type == "cylinder_z_exterior":
            assert self.cx is not None, "cylinder_z_exterior requires cx"
            assert self.cy is not None, "cylinder_z_exterior requires cy"
            assert self.radius is not None and self.radius > 0, \
                "cylinder_z_exterior requires radius > 0"


@dataclass
class BoltSeatRegion:
    """
    A bolt passing through the part with forced-solid collars only at the
    entry/exit faces, NOT along the full length.

    Physical model:
      - Through-hole (always void) spans the full axis
      - Solid collar (nondesign) exists only within ``seat_depth_m`` of each face
      - Middle of the bolt path: void core, surrounded by design space that
        the optimizer can shape freely

    This replaces the common misuse of ``cylinder_x/y/z`` NondesignRegion
    for bracket-style parts, where a full-length forced-solid sleeve
    consumes most of the material budget before optimization begins.

    Fields
    ------
    type:            "bolt_seat_x" | "bolt_seat_y" | "bolt_seat_z"
                     Axis the bolt passes along.
                        _x → centers in (y, z), bolt enters at x_min/x_max
                        _y → centers in (x, z), bolt enters at y_min/y_max
                        _z → centers in (x, y), bolt enters at z_min/z_max
    centers_m:       list of [a, b] pairs — in-plane centre of each bolt
    void_radius_m:   through-hole radius (always void, whole axis span)
    wall_radius_m:   collar radius (forced solid, only within seat_depth_m
                     of entry/exit). Must be >= void_radius_m.
    seat_depth_m:    how far the collar extends from each face (metres).
                     Typical: 3–8 mm (0.003–0.008). Must be > 0.
    entry_seat:      if True, emit a solid collar at the low-coord face
                     (x_min, y_min, or z_min depending on axis). Default True.
    exit_seat:       if True, emit a solid collar at the high-coord face.
                     Default True.  Setting either to False models a blind
                     bolt (one-sided anchor).
    load:            Optional load applied at the exit-seat face for this bolt
                     group. Format: {"direction": [x,y,z], "magnitude_n": float}
                     When set, build_load_case() can accumulate these loads
                     automatically, eliminating a separate load_case.load entry
                     and making it impossible for the load case to contradict
                     the geometry. Phase 6 wire-up: build_load_case() bolt_seats
                     parameter. Currently a schema stub — not yet accumulated.

    Example: 4 NEMA-17 motor bolts, 4mm through-hole, 7mm collar,
    8mm seat depth from both faces:

        {
          "type": "bolt_seat_x",
          "centers_m": [[0.0145, 0.0245], [0.0455, 0.0245],
                        [0.0145, 0.0555], [0.0455, 0.0555]],
          "void_radius_m": 0.004,
          "wall_radius_m": 0.007,
          "seat_depth_m":  0.008,
          "entry_seat":    true,
          "exit_seat":     true
        }
    """
    type:           str
    centers_m:      List
    void_radius_m:  float
    wall_radius_m:  float
    seat_depth_m:   float
    entry_seat:     bool           = True
    exit_seat:      bool           = True
    # Optional: load applied at the exit-seat face for this bolt group.
    # Phase 6 wire-up — currently stored in schema but not accumulated by
    # build_load_case(). Use load_case.load with selector="bolt_pattern"
    # for the current production path.
    load:           Optional[dict] = None
    through_ring_radius_m: Optional[float] = None

    VALID_TYPES = {"bolt_seat_x", "bolt_seat_y", "bolt_seat_z"}

    def validate(self) -> None:
        assert self.type in self.VALID_TYPES, \
            f"BoltSeatRegion type must be one of {self.VALID_TYPES}, got '{self.type}'"
        assert self.void_radius_m > 0, "void_radius_m must be > 0"
        assert self.wall_radius_m >= self.void_radius_m, \
            "wall_radius_m must be >= void_radius_m"
        assert self.seat_depth_m > 0, "seat_depth_m must be > 0"
        assert len(self.centers_m) > 0, "centers_m must not be empty"
        for c in self.centers_m:
            assert len(c) == 2, "each center must be [a, b]"
        assert self.entry_seat or self.exit_seat, \
            "at least one of entry_seat or exit_seat must be True " \
            "(otherwise the bolt has no anchor points)"
        if self.load is not None:
            assert "direction" in self.load and "magnitude_n" in self.load, (
                "BoltSeatRegion.load must have 'direction' and 'magnitude_n' keys"
            )
            assert len(self.load["direction"]) == 3, \
                "BoltSeatRegion.load direction must be [x, y, z]"
            assert float(self.load["magnitude_n"]) > 0, \
                "BoltSeatRegion.load magnitude_n must be > 0"


@dataclass
class AttachmentBoltVoid:
    """
    A cylindrical void punched through an attachment slab AFTER the slab
    is marked nondesign.  This is the key operation that eliminates
    BC-void singularities: the void is created in a region that is
    structurally independent of the BC assignment.

    Coordinate convention matches the slab axis:
      slab_x  →  center_a_m = Y coord,  center_b_m = Z coord
      slab_y  →  center_a_m = X coord,  center_b_m = Z coord
      slab_z  →  center_a_m = X coord,  center_b_m = Y coord

    All values in metres.
    """
    center_a_m: float
    center_b_m: float
    radius_m:   float

    def validate(self) -> None:
        assert self.radius_m > 0, \
            f"AttachmentBoltVoid radius_m must be > 0, got {self.radius_m}"


@dataclass
class AttachmentRegion:
    """
    A forced-solid slab that owns both nondesign voxels AND boundary
    conditions.  This is the two-region domain architecture entry point.

    The voxelizer processes attachment regions in two steps:
      1. Mark all voxels within [a_min, a_max] as nondesign = 1.
      2. Punch each bolt_void through the slab as void = 1, nondesign = 0.
    BC derivation (fixed DOFs) happens in build_load_case, not here.

    type:        "slab_x" | "slab_y" | "slab_z"
                   slab_x → slab bounded in X,   full Y/Z extent
                   slab_y → slab bounded in Y,   full X/Z extent
                   slab_z → slab bounded in Z,   full X/Y extent

    a_min:       Low bound of slab along the normal axis (metres).
    a_max:       High bound of slab along the normal axis (metres).
                 For a 6mm wall-mount slab at x=0: a_min=0.0, a_max=0.006.

    bc:          "fixed_full"  — fix every node on the outboard face
                                 (a_min face for wall-mount; a_max for
                                 a load-side pad — set bc_face to override).
                 "none"        — forced solid but no BC (load-bearing pad
                                 that attaches to a load region, not fixed).

    bc_face:     Which face of the slab carries the BC.  The sentinel
                 "a_min_face" auto-resolves to "x_min"/"y_min"/"z_min"
                 based on slab type — the correct default for a wall-mount
                 slab at the origin.  Set explicitly (e.g. "x_max") for
                 cases where the fixed face is the far side of the slab.

    bolt_voids:  Cylindrical voids punched through the slab post-marking.
                 No BC-void singularity is possible by construction because
                 the slab nondesign marking and void punching are sequential,
                 and the final priority rule (void > nondesign) applies.

    Example — motor mount wall plate (6mm slab, 4 bolt clearance holes):

        {
          "type": "slab_x",
          "a_min": 0.0,
          "a_max": 0.006,
          "bc": "fixed_full",
          "bolt_voids": [
            { "center_a_m": 0.010, "center_b_m": 0.010, "radius_m": 0.003 },
            { "center_a_m": 0.050, "center_b_m": 0.010, "radius_m": 0.003 },
            { "center_a_m": 0.010, "center_b_m": 0.070, "radius_m": 0.003 },
            { "center_a_m": 0.050, "center_b_m": 0.070, "radius_m": 0.003 }
          ]
        }
    """
    type:        str
    a_min:       float
    a_max:       float
    bc:          str  = "fixed_full"
    bc_face:     str  = "a_min_face"   # sentinel; resolved at voxelize time
    bolt_voids:  List = field(default_factory=list)

    VALID_TYPES    = {"slab_x", "slab_y", "slab_z"}
    VALID_BC_TYPES = {"fixed_full", "none"}

    def validate(self) -> None:
        assert self.type in self.VALID_TYPES, \
            f"AttachmentRegion type must be one of {self.VALID_TYPES}, got '{self.type}'"
        assert self.a_max > self.a_min, \
            (f"AttachmentRegion a_max ({self.a_max}) must be > "
             f"a_min ({self.a_min})")
        assert self.bc in self.VALID_BC_TYPES, \
            f"AttachmentRegion bc must be one of {self.VALID_BC_TYPES}, got '{self.bc}'"
        for v in self.bolt_voids:
            v.validate()

    def resolved_bc_face(self) -> str:
        """
        Return the concrete face name for this region's BC, resolving
        the "a_min_face" sentinel to the appropriate face string.
        """
        if self.bc_face != "a_min_face":
            return self.bc_face
        return {
            "slab_x": "x_min",
            "slab_y": "y_min",
            "slab_z": "z_min",
        }[self.type]


@dataclass
class LoadHints:
    """Kept for backward compatibility with Stage 03 FEniCSx path."""
    primary_face:     str
    load_magnitude_n: float

    VALID_FACES = {"top", "bottom", "left", "right", "front", "back"}

    def validate(self) -> None:
        assert self.primary_face in self.VALID_FACES, \
            f"primary_face must be one of {self.VALID_FACES}"
        assert self.load_magnitude_n > 0, "load_magnitude_n must be > 0"


@dataclass
class ExportParams:
    stl_output_dir: str
    stl_ascii:      bool


@dataclass
class PipelineParams:
    part_name:            str
    geometry:             GeometryParams
    mesh_hints:           MeshHints
    load_hints:           LoadHints
    export:               ExportParams
    boundary_conditions:  BoundaryConditions    = field(default_factory=BoundaryConditions)

    # Phase 2: declarative load case and nondesign regions (optional)
    load_case_config:     Optional[LoadCaseConfig]      = None
    nondesign_regions:    List[NondesignRegion]          = field(default_factory=list)

    # Phase 3: axis-aligned box regions forced void (empty space in non-rectangular parts)
    void_regions:         List[VoidRegion]               = field(default_factory=list)

    # Phase 4: bolt seats (through-hole + collar only at entry/exit faces).
    # Preferred over full-length cylinder_* NondesignRegion for bracket-
    # style parts where a thick sleeve would burn the material budget.
    bolt_seats:           List[BoltSeatRegion]           = field(default_factory=list)

    # Phase 5: attachment regions — forced-solid slabs that own both the
    # nondesign voxels and the fixed BC.  When present, LoadCaseConfig.fixed
    # may be omitted; fixity is derived from the slab face instead.
    attachment_regions:   List[AttachmentRegion]         = field(default_factory=list)

    def validate(self) -> None:
        """Run all sub-validators."""
        self.geometry.validate()
        self.mesh_hints.validate()
        self.load_hints.validate()
        self.boundary_conditions.validate()
        if self.load_case_config is not None:
            self.load_case_config.validate()
        for r in self.nondesign_regions:
            r.validate()
        for r in self.void_regions:
            r.validate()
        for r in self.bolt_seats:
            r.validate()
        for r in self.attachment_regions:
            r.validate()
        # Guard: if no attachment_regions provide fixity, load_case_config.fixed
        # must be present so the solver always receives a non-empty fixed_dofs array.
        has_attachment_bc = any(
            r.bc != "none" for r in self.attachment_regions
        )
        lc_has_fixed = (
            self.load_case_config is not None
            and self.load_case_config.fixed is not None
        )
        if self.load_case_config is not None and not has_attachment_bc and not lc_has_fixed:
            raise AssertionError(
                "load_case_config has no fixed spec and no attachment_regions "
                "provide bc != 'none'. The solver will receive an empty "
                "fixed_dofs array and the stiffness matrix will be singular."
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineParams":
        """Load and validate params from a JSON file."""
        raw = json.loads(Path(path).read_text())
        return cls._from_raw(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineParams":
        """Load params from a plain dict."""
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> "PipelineParams":
        """
        Deserialize with backward compatibility.
        New fields (load_case_config, nondesign_regions, attachment_regions)
        are all optional.
        """
        bc_raw = {k: v for k, v in raw.get("boundary_conditions", {}).items()
                  if not k.startswith("_")}

        # Parse load_case_config if present.
        # fixed is Optional — absent key means "derive fixity from attachment_regions".
        lc_raw = raw.get("load_case", None)
        load_case_config = None
        if lc_raw is not None:
            fixed_raw = lc_raw.get("fixed", None)
            fixed_cfg = FixedFaceConfig(**fixed_raw) if fixed_raw else None

            def _mk_load(d):
                return LoadFaceConfig(**{k: v for k, v in d.items()
                                         if not k.startswith("_")})

            loads_raw = lc_raw.get("loads", None)   # R1 multi-load (list)
            load_raw  = lc_raw.get("load",  None)   # legacy single
            if loads_raw is not None:
                load_case_config = LoadCaseConfig(
                    fixed=fixed_cfg,
                    loads=[_mk_load(d) for d in loads_raw],
                )
            elif load_raw is not None:
                load_case_config = LoadCaseConfig(fixed=fixed_cfg, load=_mk_load(load_raw))
            else:
                load_case_config = LoadCaseConfig(fixed=fixed_cfg)

        # Parse nondesign_regions if present
        nd_raw = raw.get("nondesign_regions", [])
        nondesign_regions = [
            NondesignRegion(**{k: v for k, v in r.items() if not k.startswith("_")})
            for r in nd_raw
        ]

        # Parse void_regions if present (Phase 3)
        vr_raw = raw.get("void_regions", [])
        void_regions = [
            VoidRegion(**{k: v for k, v in r.items() if not k.startswith("_")})
            for r in vr_raw
        ]

        # Parse bolt_seats if present (Phase 4)
        bs_raw = raw.get("bolt_seats", [])
        bolt_seats = [
            BoltSeatRegion(**{k: v for k, v in r.items() if not k.startswith("_")})
            for r in bs_raw
        ]

        # Parse attachment_regions if present (Phase 5)
        ar_raw = raw.get("attachment_regions", [])
        attachment_regions = []
        for r in ar_raw:
            bolt_voids_raw = r.get("bolt_voids", [])
            bolt_voids = [
                AttachmentBoltVoid(
                    **{k: v for k, v in bv.items() if not k.startswith("_")}
                )
                for bv in bolt_voids_raw
            ]
            region_data = {
                k: v for k, v in r.items()
                if not k.startswith("_") and k != "bolt_voids"
            }
            attachment_regions.append(
                AttachmentRegion(**region_data, bolt_voids=bolt_voids)
            )

        return cls(
            part_name=raw["part_name"],
            geometry=GeometryParams(**raw["geometry"]),
            mesh_hints=MeshHints(**raw["mesh_hints"]),
            load_hints=LoadHints(**raw["load_hints"]),
            export=ExportParams(**raw["export"]),
            boundary_conditions=BoundaryConditions(**bc_raw) if bc_raw
                                else BoundaryConditions(),
            load_case_config=load_case_config,
            nondesign_regions=nondesign_regions,
            void_regions=void_regions,
            bolt_seats=bolt_seats,
            attachment_regions=attachment_regions,
        )

    def to_openscad_defines(self) -> dict:
        """
        Flatten scalar-numeric geometry params into uppercase OpenSCAD -D defines.

        Only int / float / bool fields are emitted. Non-numeric fields
        (e.g. ``shape: "disk"``) are geometry-factory metadata consumed by
        ``region_factory`` / ``resolve_geometry_regions`` — NOT OpenSCAD
        variables. Passing them through as ``-D SHAPE="disk"`` pollutes
        the SCAD namespace and risks collisions with module / function
        identifiers in the .scad file.  Lists / dicts are filtered for
        the same reason (``-D`` expects scalars).

        Note: ``bool`` is a subclass of ``int`` in Python, so boolean
        fields fall through and are handled by openscad_runner's
        dedicated bool branch.
        """
        return {
            k.upper(): v
            for k, v in self.geometry._all_fields.items()
            if isinstance(v, (int, float))
        }