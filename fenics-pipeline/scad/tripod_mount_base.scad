// scad/tripod_mount_base.scad
// Parametric tripod camera mount base.
//
// Circular platform with a central 1/4-20 bolt hole and three symmetric
// leg attachment points at 120° intervals. Optimized for vertical camera load
// transmitted through the central bolt to three legs.
//
// All caps variables are injected via -D flags from openscad_runner.py.

// ── Parameters ────────────────────────────────────────────────────────────
DIAMETER            = 80.0;   // bounding circle diameter — also sets bbox X and Y
HEIGHT              = 25.0;   // platform thickness (Z)
CENTER_HOLE_D       = 7.0;    // 1/4-20 clearance hole (6.35mm + clearance)
LEG_HOLE_D          = 5.0;    // M4 leg attachment holes
LEG_HOLE_RADIUS     = 28.0;   // radial distance from center to leg holes (mm)
FIRST_LEG_ANGLE     = 90.0;   // rotation of first leg (degrees)

// ── Derived ───────────────────────────────────────────────────────────────
fn_curve = 64;
radius   = DIAMETER / 2;
cx       = DIAMETER / 2;
cy       = DIAMETER / 2;

// ── Modules ───────────────────────────────────────────────────────────────
module platform() {
    translate([cx, cy, 0])
        cylinder(r=radius, h=HEIGHT, $fn=fn_curve);
}

module center_hole() {
    translate([cx, cy, -1])
        cylinder(r=CENTER_HOLE_D / 2, h=HEIGHT + 2, $fn=fn_curve);
}

module leg_holes() {
    for (i = [0 : 2]) {
        angle = FIRST_LEG_ANGLE + i * 120;
        lx = cx + LEG_HOLE_RADIUS * cos(angle);
        ly = cy + LEG_HOLE_RADIUS * sin(angle);
        translate([lx, ly, -1])
            cylinder(r=LEG_HOLE_D / 2, h=HEIGHT + 2, $fn=fn_curve);
    }
}

// ── Assembly ──────────────────────────────────────────────────────────────
difference() {
    platform();
    center_hole();
    leg_holes();
}