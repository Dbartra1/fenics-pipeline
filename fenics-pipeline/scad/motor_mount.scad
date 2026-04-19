// scad/motor_mount.scad
// Parametric motor mount bracket.
// Wall-mounted plate (fixed at x_min), motor-mount plate on opposite face
// (loaded at x_max).  Both plates are part of a single solid block;
// topology optimization carves away non-load-bearing material between them.
//
// All caps variables are injected via -D flags from openscad_runner.py.
// Do not hardcode dimensions here — edit scad/motor_mount_params.json instead.

// ── Parameters ────────────────────────────────────────────────────────────
LENGTH                 = 70.0;   // X — wall → motor direction
WIDTH                  = 60.0;   // Y — bracket width
HEIGHT                 = 80.0;   // Z — bracket height

WALL_HOLE_DIAMETER     = 6.0;    // M5 clearance for wall bolts
WALL_HOLE_INSET        = 10.0;   // inset from Y/Z edges on wall face

MOTOR_HOLE_DIAMETER    = 4.0;    // M3 clearance for motor bolts
MOTOR_HOLE_PITCH       = 31.0;   // NEMA-17 bolt circle side length (mm)
MOTOR_CENTER_Y         = 30.0;   // Y centre of motor bolt pattern on x_max face
MOTOR_CENTER_Z         = 40.0;   // Z centre of motor bolt pattern on x_max face

// ── Derived ───────────────────────────────────────────────────────────────
wall_hole_r  = WALL_HOLE_DIAMETER  / 2;
motor_hole_r = MOTOR_HOLE_DIAMETER / 2;
fn_curve     = 32;
motor_half   = MOTOR_HOLE_PITCH    / 2;

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

// ── Motor-mount holes (NEMA-17 square pattern on x_max face, drilled X) ───
module motor_holes() {
    positions = [
        [MOTOR_CENTER_Y - motor_half, MOTOR_CENTER_Z - motor_half],
        [MOTOR_CENTER_Y + motor_half, MOTOR_CENTER_Z - motor_half],
        [MOTOR_CENTER_Y - motor_half, MOTOR_CENTER_Z + motor_half],
        [MOTOR_CENTER_Y + motor_half, MOTOR_CENTER_Z + motor_half],
    ];
    for (pos = positions) {
        translate([-1, pos[0], pos[1]])
            rotate([0, 90, 0])
                cylinder(r=motor_hole_r, h=LENGTH + 2, $fn=fn_curve);
    }
}

// ── Assembly ──────────────────────────────────────────────────────────────
difference() {
    block();
    wall_holes();
    motor_holes();
}