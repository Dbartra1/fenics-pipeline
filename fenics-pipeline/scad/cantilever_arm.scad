// scad/cantilever_arm.scad
// Parametric cantilever arm bracket.
// Fixed at the back wall plate (4 mounting holes on the vertical face),
// load applied downward on the tip of the horizontal arm.
//
// All caps variables are injected via -D flags from openscad_runner.py.

// ── Parameters ────────────────────────────────────────────────────────────
LENGTH                 = 100.0;  // total length of arm (X)
WIDTH                  = 50.0;   // width into page (Y)
HEIGHT                 = 75.0;   // total height (Z) — arm at bottom, wall above
ARM_THICKNESS          = 20.0;   // Z thickness of horizontal arm
WALL_THICKNESS         = 20.0;   // X depth of back wall
MOUNTING_HOLE_DIAMETER = 6.0;
MOUNTING_HOLE_INSET    = 12.0;

// ── Derived ───────────────────────────────────────────────────────────────
hole_r   = MOUNTING_HOLE_DIAMETER / 2;
fn_curve = 32;

// ── Simple L-body ─────────────────────────────────────────────────────────
module l_body() {
    union() {
        // Horizontal arm: full length, bottom portion
        cube([LENGTH, WIDTH, ARM_THICKNESS]);
        // Vertical wall: back portion, full height
        cube([WALL_THICKNESS, WIDTH, HEIGHT]);
    }
}

// ── Mounting holes through the back wall (drilled in X direction) ─────────
module mounting_holes() {
    inset = MOUNTING_HOLE_INSET;
    y_positions = [inset, WIDTH - inset];
    z_positions = [inset, HEIGHT - inset];
    for (y = y_positions) {
        for (z = z_positions) {
            translate([-1, y, z])
                rotate([0, 90, 0])
                    cylinder(r=hole_r, h=WALL_THICKNESS + 2, $fn=fn_curve);
        }
    }
}

// ── Assembly ──────────────────────────────────────────────────────────────
difference() {
    l_body();
    mounting_holes();
}