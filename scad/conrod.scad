// conrod.scad — connecting rod for topology optimization
// Flat plate envelope with two pin bores (cylinder_z through Z thickness).
// SIMP optimizes the web between the large (fixed) and small (load) bosses.
//
// Coordinate convention (matches params.json):
//   X — long axis (LENGTH), large boss at x_min, small boss at x_max
//   Y — width axis (WIDTH)
//   Z — thickness axis (HEIGHT)

// ── Papermill-injectable parameters ──────────────────────────────────────────
LENGTH             = 120.0;   // mm — total envelope length
WIDTH              =  40.0;   // mm — total envelope width
HEIGHT             =  20.0;   // mm — total envelope thickness (Z)
LARGE_BORE_D       =  20.0;   // mm — fixed-end pin bore diameter
SMALL_BORE_D       =  12.0;   // mm — load-end pin bore diameter
BORE_INSET         =  20.0;   // mm — bore centre distance from each end face
WALL_THICKNESS     =   3.0;   // mm — minimum wall around each bore
FILLET_RADIUS      =   2.0;   // mm — edge fillet on envelope

// ── Derived ───────────────────────────────────────────────────────────────────
large_cx = BORE_INSET;
small_cx = LENGTH - BORE_INSET;
bore_cy  = WIDTH / 2.0;

module boss(cx, d, wall) {
    // Solid cylinder at bore centre — nondesign region in optimizer
    translate([cx, bore_cy, 0])
        cylinder(h=HEIGHT, d=d + 2*wall, center=false, $fn=64);
}

module bore(cx, d) {
    translate([cx, bore_cy, -1])
        cylinder(h=HEIGHT + 2, d=d, center=false, $fn=64);
}

difference() {
    union() {
        // Main envelope block
        translate([FILLET_RADIUS, FILLET_RADIUS, 0])
            minkowski() {
                cube([LENGTH - 2*FILLET_RADIUS,
                      WIDTH  - 2*FILLET_RADIUS,
                      HEIGHT - FILLET_RADIUS]);
                cylinder(r=FILLET_RADIUS, h=FILLET_RADIUS, $fn=32);
            }
        // Boss pads — ensure wall material around bores
        boss(large_cx, LARGE_BORE_D, WALL_THICKNESS);
        boss(small_cx, SMALL_BORE_D, WALL_THICKNESS);
    }
    // Subtract pin bores through full Z
    bore(large_cx, LARGE_BORE_D);
    bore(small_cx, SMALL_BORE_D);
}