# src/geometry/param_schema.py
#
# Typed schema for pipeline parameters.
# Validates params.json at load time so bad values fail here,
# not mid-solve in 03_fea_fenicsx.ipynb.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
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
    target_element_size: float
    refinement_regions:  List = field(default_factory=list)

    def validate(self) -> None:
        assert self.target_element_size > 0, "target_element_size must be > 0"


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
    part_name:  str
    geometry:   GeometryParams
    mesh_hints: MeshHints
    load_hints: LoadHints
    export:     ExportParams

    def validate(self) -> None:
        """Run all sub-validators. Call this immediately after loading."""
        self.geometry.validate()
        self.mesh_hints.validate()
        self.load_hints.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineParams":
        """Load and validate params from a JSON file."""
        raw = json.loads(Path(path).read_text())
        return cls(
            part_name=raw["part_name"],
            geometry=GeometryParams(**raw["geometry"]),
            mesh_hints=MeshHints(**raw["mesh_hints"]),
            load_hints=LoadHints(**raw["load_hints"]),
            export=ExportParams(**raw["export"]),
        )

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineParams":
        """Load params from a plain dict — used by generate_test_cases notebook."""
        return cls(
            part_name=raw["part_name"],
            geometry=GeometryParams(**raw["geometry"]),
            mesh_hints=MeshHints(**raw["mesh_hints"]),
            load_hints=LoadHints(**raw["load_hints"]),
            export=ExportParams(**raw["export"]),
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