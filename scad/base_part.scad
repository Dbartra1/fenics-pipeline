// scad/base_part.scad
// Parametric mounting bracket.
// All caps variables are injected via -D flags from openscad_runner.py.
// Do not hardcode dimensions here — edit scad/base_part_params.json instead.

// ── Parameters (defaults match scad/base_part_params.json) ────────────────
LENGTH                 = 100.0;
WIDTH                  = 60.0;
HEIGHT                 = 20.0;
WALL_THICKNESS         = 4.0;
FILLET_RADIUS          = 2.0;
MOUNTING_HOLE_DIAMETER = 6.0;
MOUNTING_HOLE_INSET    = 10.0;

// ── Derived values ─────────────────────────────────────────────────────────
hole_r   = MOUNTING_HOLE_DIAMETER / 2;
fn_curve = 32;  // cylinder facets — increase for smoother export

// ── Main body ──────────────────────────────────────────────────────────────
module body() {
    minkowski() {
        cube([
            LENGTH - 2 * FILLET_RADIUS,
            WIDTH  - 2 * FILLET_RADIUS,
            HEIGHT - FILLET_RADIUS
        ]);
        cylinder(r=FILLET_RADIUS, h=FILLET_RADIUS, $fn=fn_curve);
    }
}

// ── Mounting holes (4 corners) ─────────────────────────────────────────────
module mounting_holes() {
    inset = MOUNTING_HOLE_INSET;
    positions = [
        [inset,          inset         ],
        [LENGTH - inset, inset         ],
        [inset,          WIDTH - inset ],
        [LENGTH - inset, WIDTH - inset ],
    ];
    for (pos = positions) {
        translate([pos[0], pos[1], -1])
            cylinder(r=hole_r, h=HEIGHT + 2, $fn=fn_curve);
    }
}

// ── Hollowed interior (shell thickness = WALL_THICKNESS) ───────────────────
module hollow() {
    translate([WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS])
        cube([
            LENGTH - 2 * WALL_THICKNESS,
            WIDTH  - 2 * WALL_THICKNESS,
            HEIGHT
        ]);
}

// ── Assembly ───────────────────────────────────────────────────────────────
difference() {
    body();
    hollow();
    mounting_holes();
}