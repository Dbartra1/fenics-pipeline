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
    VALID_SELECTORS = {"full", "corners", "leg_holes"}

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

    face:          which face receives the traction load
    selector:      "full"        → distribute load uniformly across the
                                   entire face (legacy / default behaviour)
                   "center_disk" → concentrate load on a central disk patch.
                                   The XY centre of the disk is resolved per
                                   part shape via region_factory.part_center_m:
                                     shape="disk" → (diameter/2, diameter/2)
                                     shape="box" / absent → (length/2, width/2)
                                   Use this when the physical load path runs
                                   through a specific mounting feature (e.g.
                                   the central bolt of a tripod mount) rather
                                   than the entire face.
    direction:     [x, y, z] unit vector (will be normalised internally)
    magnitude_n:   total force in Newtons, distributed across the selected DOFs
    disk_radius_m: radius of the loaded disk in metres (selector="center_disk"
                   only). Size this to match the real contact patch — too
                   small approaches a point load and produces stress
                   singularities that dominate SIMP convergence.
    """
    face:          str   = "z_max"
    selector:      str   = "full"
    direction:     List  = field(default_factory=lambda: [0.0, 0.0, -1.0])
    magnitude_n:   float = 10000.0
    disk_radius_m: float = 0.010

    VALID_FACES     = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    VALID_SELECTORS = {"full", "center_disk"}

    def validate(self) -> None:
        assert self.face in self.VALID_FACES, \
            f"load face must be one of {self.VALID_FACES}, got '{self.face}'"
        assert self.selector in self.VALID_SELECTORS, \
            f"selector must be one of {self.VALID_SELECTORS}, got '{self.selector}'"
        assert len(self.direction) == 3, "direction must be [x, y, z]"
        assert self.magnitude_n > 0, "magnitude_n must be > 0"
        assert self.disk_radius_m > 0, "disk_radius_m must be > 0"

@dataclass
class LoadCaseConfig:
    """
    Full load case: fixed BCs + applied load.
    Replaces the hardcoded logic in build_load_case().
    """
    fixed: FixedFaceConfig = field(default_factory=FixedFaceConfig)
    load:  LoadFaceConfig  = field(default_factory=LoadFaceConfig)

    def validate(self) -> None:
        self.fixed.validate()
        self.load.validate()
        assert self.fixed.face != self.load.face, \
            "fixed face and load face cannot be the same"


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

    Example: 4 NEMA-17 motor bolts, 4mm through-hole, 3mm collar,
    5mm seat depth from the x_max face only (motor plate):

        {
          "type": "bolt_seat_x",
          "centers_m": [[0.0145, 0.0245], [0.0455, 0.0245],
                        [0.0145, 0.0555], [0.0455, 0.0555]],
          "void_radius_m": 0.002,
          "wall_radius_m": 0.003,
          "seat_depth_m":  0.005,
          "entry_seat":    false,
          "exit_seat":     true
        }
    """
    type:           str
    centers_m:      List
    void_radius_m:  float
    wall_radius_m:  float
    seat_depth_m:   float
    entry_seat:     bool = True
    exit_seat:      bool = True

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
    # Phase 3: axis-aligned box regions forced void (empty space in non-rectangular parts)
    void_regions:         List[VoidRegion]               = field(default_factory=list)

    # Phase 4: bolt seats (through-hole + collar only at entry/exit faces).
    # Preferred over full-length cylinder_* NondesignRegion for bracket-
    # style parts where a thick sleeve would burn the material budget.
    bolt_seats:           List[BoltSeatRegion]            = field(default_factory=list)

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
        New fields (load_case_config, nondesign_regions) are optional.
        """
        bc_raw = raw.get("boundary_conditions", {})

        # Parse load_case_config if present
        lc_raw = raw.get("load_case", None)
        load_case_config = None
        if lc_raw is not None:
            fixed_raw = lc_raw.get("fixed", {})
            load_raw  = lc_raw.get("load", {})
            load_case_config = LoadCaseConfig(
                fixed=FixedFaceConfig(**fixed_raw) if fixed_raw else FixedFaceConfig(),
                load=LoadFaceConfig(**load_raw)    if load_raw  else LoadFaceConfig(),
            )

        # Parse nondesign_regions if present
        nd_raw = raw.get("nondesign_regions", [])
        nondesign_regions = [
            NondesignRegion(**{k: v for k, v in r.items() if not k.startswith("_")})
            for r in nd_raw
        ]

        # Parse void_regions if present (Phase 3)
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
