// scad/optimization_domain.scad
// Simple box — no holes, no fillets — for topology optimization mesh.
// Features are added back in post-processing.
LENGTH = 100.0;
WIDTH  = 60.0;
HEIGHT = 20.0;

cube([LENGTH, WIDTH, HEIGHT]);