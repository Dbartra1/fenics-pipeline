// scad/cantilever_arm.scad
// Parametric cantilever arm bracket (box envelope).
// Wall-mounted at x_min (4 bolts), tip-loaded at x_max (bottom-centre disk).
// Topology optimizer carves the envelope to produce a Michell truss.
//
// All caps variables are injected via -D flags from openscad_runner.py.
// Do not hardcode dimensions here — edit scad/cantilever_arm_params.json instead.

// ── Parameters ────────────────────────────────────────────────────────────
LENGTH                 = 100.0;  // X — wall → load tip
WIDTH                  = 50.0;   // Y — bracket width
HEIGHT                 = 75.0;   // Z — bracket height

WALL_HOLE_DIAMETER     = 6.0;    // M5 clearance for wall bolts
WALL_HOLE_INSET        = 12.0;   // inset from Y/Z edges of x_min face

LOAD_HOLE_DIAMETER     = 8.0;    // pin/eye-bolt clearance at load tip
LOAD_HOLE_OFFSET_Z     = 12.0;   // how far above bottom face the load point sits

// ── Derived ───────────────────────────────────────────────────────────────
wall_hole_r  = WALL_HOLE_DIAMETER  / 2;
load_hole_r  = LOAD_HOLE_DIAMETER  / 2;
fn_curve     = 32;

// ── Solid block ───────────────────────────────────────────────────────────
module block() {
    cube([LENGTH, WIDTH, HEIGHT]);
}

// ── Wall-mount holes (4 corners of x_min face, drilled through X) ─────────
module wall_holes() {
    positions = [
        [WALL_HOLE_INSET,          WALL_HOLE_INSET          ],
        [WIDTH  - WALL_HOLE_INSET, WALL_HOLE_INSET          ],
        [WALL_HOLE_INSET,          HEIGHT - WALL_HOLE_INSET ],
        [WIDTH  - WALL_HOLE_INSET, HEIGHT - WALL_HOLE_INSET ],
    ];
    for (pos = positions) {
        translate([-1, pos[0], pos[1]])
            rotate([0, 90, 0])
                cylinder(r=wall_hole_r, h=LENGTH + 2, $fn=fn_curve);
    }
}

// ── Load hole (single hole at x_max face, y-centered, near bottom) ────────
module load_hole() {
    translate([-1, WIDTH / 2, LOAD_HOLE_OFFSET_Z])
        rotate([0, 90, 0])
            cylinder(r=load_hole_r, h=LENGTH + 2, $fn=fn_curve);
}

// ── Assembly ──────────────────────────────────────────────────────────────
difference() {
    block();
    wall_holes();
    load_hole();
}