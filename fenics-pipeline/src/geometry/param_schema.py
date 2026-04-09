# src/geometry/param_schema.py
#
# Typed schema for pipeline parameters.
# Validates params.json at load time so bad values fail here,
# not mid-solve in 03_fea_fenicsx.ipynb.
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class GeometryParams:
    length:                 float
    width:                  float
    height:                 float
    wall_thickness:         float
    fillet_radius:          float
    mounting_hole_diameter: float
    mounting_hole_inset:    float

    def validate(self) -> None:
        assert self.length > 0,                             "length must be > 0"
        assert self.width > 0,                              "width must be > 0"
        assert self.height > 0,                             "height must be > 0"
        assert self.wall_thickness < min(self.length, self.width) / 2, \
            "wall_thickness too large for given length/width"
        assert self.fillet_radius <= self.wall_thickness, \
            "fillet_radius must be <= wall_thickness"
        assert self.mounting_hole_diameter < self.width, \
            "mounting_hole_diameter must be < width"


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

    fixed_face:
        Which face is fully fixed (encastre), OR "corners" to fix only the
        corner mounting-hole regions on that face.
        Face names: "top" (+Z), "bottom" (-Z), "left" (-X), "right" (+X),
                    "front" (-Y), "back" (+Y), "corners" (corner regions on bottom)

    load_face:
        Which face the traction load is applied to.
        Same face names as above, excluding "corners".

    load_direction:
        [x, y, z] vector for load direction. Will be normalized internally.
        Default [0, 0, -1] = downward (-Z).

    hole_inset_fraction:
        When fixed_face="corners", the fraction of part length/width used as
        the corner region size. 0.15 = 15% inset matches a typical 4-hole bracket.

    shell_thickness_mm:
        Enforce a solid outer shell of this thickness (mm) in Stage 5 STL export.
        Prevents open surfaces at part boundaries. Set to 0 to disable.
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
class LoadHints:
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
    boundary_conditions:  BoundaryConditions = field(
        default_factory=BoundaryConditions
    )

    def validate(self) -> None:
        """Run all sub-validators. Call this immediately after loading."""
        self.geometry.validate()
        self.mesh_hints.validate()
        self.load_hints.validate()
        self.boundary_conditions.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineParams":
        """Load and validate params from a JSON file."""
        raw = json.loads(Path(path).read_text())
        return cls._from_raw(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineParams":
        """Load params from a plain dict — used by generate_test_cases notebook."""
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> "PipelineParams":
        """Shared deserialization with backward-compatible BC loading.
        If boundary_conditions is absent from params.json, sensible defaults apply.
        """
        bc_raw = raw.get("boundary_conditions", {})
        return cls(
            part_name=raw["part_name"],
            geometry=GeometryParams(**raw["geometry"]),
            mesh_hints=MeshHints(**raw["mesh_hints"]),
            load_hints=LoadHints(**raw["load_hints"]),
            export=ExportParams(**raw["export"]),
            boundary_conditions=BoundaryConditions(**bc_raw) if bc_raw
                                else BoundaryConditions(),
        )

    def to_openscad_defines(self) -> dict[str, float | str | bool]:
        """
        Flatten geometry params into a dict of OpenSCAD -D defines.
        Only geometry values go to OpenSCAD — mesh/load hints are Python-only.
        """
        g = self.geometry
        return {
            "LENGTH":                 g.length,
            "WIDTH":                  g.width,
            "HEIGHT":                 g.height,
            "WALL_THICKNESS":         g.wall_thickness,
            "FILLET_RADIUS":          g.fillet_radius,
            "MOUNTING_HOLE_DIAMETER": g.mounting_hole_diameter,
            "MOUNTING_HOLE_INSET":    g.mounting_hole_inset,
        }
