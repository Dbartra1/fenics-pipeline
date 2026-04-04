# fenics-pipeline

Closed-loop Jupyter notebook pipeline for 3D part simulation and topology
optimization. Parametric geometry is defined in OpenSCAD, meshed with gmsh,
load-tested with FEniCSx, topology-optimized via SIMP, and exported as a
watertight STL — all orchestrated headlessly by Papermill inside Docker on WSL2.

---

## Stack

| Stage | Tool | Notebook | Module |
|---|---|---|---|
| Parametric geometry | OpenSCAD | `01_geometry_openscad.ipynb` | `src/geometry/` |
| Mesh generation | gmsh 4.12 | `02_mesh_gmsh.ipynb` | `src/meshing/` |
| FEA / load testing | FEniCSx 0.8 | `03_fea_fenicsx.ipynb` | `src/fea/` |
| Topology optimization | SIMP | `04_simp_optimization.ipynb` | `src/optimization/` |
| STL export | trimesh + scikit-image | `05_stl_export.ipynb` | — |
| Full pipeline | Papermill | `notebooks/pipeline_full.ipynb` | — |

---

## First-time setup

### Step 1 — Configure WSL2 memory (Windows, run once)

FEniCSx and gmsh are memory-hungry. The default WSL2 memory cap (50% of
physical RAM) is too low for meshes above ~100k elements. Run this from a
**Windows PowerShell terminal** — not WSL2:
```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts/setup_wsl.ps1
```

This writes `%USERPROFILE%\.wslconfig`, shuts down WSL2, and restarts it
with 24GB memory, 10 processors, and 8GB swap. Pass arguments to override:
```powershell
.\scripts\setup_wsl.ps1 -MemoryGB 32 -Processors 12 -SwapGB 16
```

You only need to do this once per machine. After WSL2 restarts, return to
your WSL2 terminal and continue with Step 2.

### Step 2 — Validate environment (WSL2)
```bash
make check
```

This runs `scripts/check_environment.sh` which validates WSL2 memory,
Docker daemon, `.env`, and `outputs/` directory structure.

### Step 3 — Build and start
```bash
make build   # builds the Docker image (~10 min first time)
make up      # starts JupyterLab at http://localhost:8888
```

---

## Running the pipeline

### Interactive (JupyterLab)

Open `http://localhost:8888` and run notebooks in order:
```
00_env_validation.ipynb       ← always run first on a new machine
01_geometry_openscad.ipynb
02_mesh_gmsh.ipynb
03_fea_fenicsx.ipynb
04_simp_optimization.ipynb
05_stl_export.ipynb
```

Each notebook writes a handoff JSON to `outputs/meshes/` that the next
stage reads automatically.

### Headless (Papermill)
```bash
make run
```

Executes `pipeline_full.ipynb` via Papermill. Executed notebooks with all
cell outputs are saved to `outputs/executed_nbs/` for inspection. Exit code
is non-zero if any stage fails.

### Parameter sweep

Edit the `SWEEP_PARAMS` list in `pipeline_full.ipynb` Cell 0 and set
`SWEEP_MODE = True`, then run:
```bash
make run
```

Each sweep configuration produces its own STL in `outputs/stl/` and its
own set of executed notebooks in `outputs/executed_nbs/`.

---

## Configuring the part

All geometry and simulation parameters live in `scad/params.json`:
```json
{
  "part_name": "base_part",
  "geometry": {
    "length": 100.0,
    "width":  60.0,
    "height": 20.0
  },
  "mesh_hints": {
    "target_element_size": 2.0
  },
  "load_hints": {
    "primary_face":      "top",
    "load_magnitude_n":  1000.0
  }
}
```

Change values here and re-run the pipeline. Papermill can also inject
overrides at runtime without editing the file — see `pipeline_full.ipynb`
Cell 0.

---

## Tuning the optimizer

Key parameters in `04_simp_optimization.ipynb` Cell 0:

| Parameter | Default | Effect |
|---|---|---|
| `VOLUME_FRACTION` | 0.4 | Fraction of material retained. Lower = lighter, weaker. |
| `FILTER_RADIUS` | 6.0 mm | Minimum feature size. Raise if STL has spikes. |
| `PENAL` | 3.0 | Push density to 0/1. Don't go below 2.5. |
| `MAX_ITERATIONS` | 100 | Increase if convergence plot hasn't flattened. |

---

## Outputs
```
outputs/
├── meshes/
│   ├── <name>.msh                   gmsh native mesh
│   ├── <name>.xdmf + .h5            FEniCSx volume mesh
│   ├── <name>_boundaries.xdmf       FEniCSx boundary tags
│   ├── <name>_displacement.xdmf     FEA displacement field
│   ├── <name>_stress.xdmf           Von Mises stress field
│   ├── <name>_density.xdmf          SIMP density field
│   └── <name>_stage0*.json          Stage handoff records
├── stl/
│   └── <name>_optimized.stl         Final export
├── reports/
│   ├── <name>_geometry.png          OpenSCAD render
│   ├── <name>_mesh.png              Mesh preview
│   ├── <name>_aspect_ratio.png      Mesh quality histogram
│   ├── <name>_stress.png            Von Mises stress map
│   ├── <name>_displacement.png      Displacement field
│   ├── <name>_density_iter*.png     SIMP progress snapshots
│   ├── <name>_density_final.png     Final density field
│   ├── <name>_convergence.png       Compliance history plot
│   └── <name>_before_after.png      Original vs optimized
└── executed_nbs/
    └── <name>_0*_<timestamp>.ipynb  Papermill output notebooks
```

---

## Tests
```bash
make test
```

Runs `tests/test_mesh_quality.py` and `tests/test_fea_smoke.py` inside the
container. Tests build their own minimal meshes programmatically — no
pipeline outputs required.

---

## Troubleshooting

**`ModuleNotFoundError: dolfinx` from Papermill kernel**
The kernel subprocess isn't inheriting the dolfinx venv. Verify:
```bash
docker exec fenics-pipeline python -c "import dolfinx; print(dolfinx.__version__)"
```
If this fails, rebuild the image: `make build`.

**PyVista segfault in Stage 3 or 5**
Override the JupyterLab command to use xvfb in `docker-compose.yml`:
```yaml
command: ["xvfb-run", "-a", "jupyter", "lab",
          "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

**Marching cubes produces no geometry (Stage 5)**
The density threshold is above the actual density range. Check
`density_grid.max()` in Stage 5 Cell 3 and lower `DENSITY_THRESHOLD`.

**Spiky or disconnected STL**
`FILTER_RADIUS` in Stage 4 is too small. Increase by 50% and re-run
from Stage 4 Cell 3. No need to re-mesh or re-run FEA.

**OOM during SIMP loop**
Reduce `target_element_size` in `scad/params.json` to coarsen the mesh,
or switch `USE_ITERATIVE_SOLVER = True` in Stage 3 and set
`ALGORITHM_3D = 10` (HXT parallel) in Stage 2.

**Mesh quality failures in Stage 2**
Reduce `target_element_size` in `scad/params.json`, or increase
`FILTER_RADIUS` in Stage 4 after fixing the mesh.

---

## Makefile reference
```
make setup         Print WSL2 setup instructions
make check         Validate environment (WSL2-side)
make build         Build Docker image
make up            Start JupyterLab on :8888
make down          Stop container
make run           Execute full pipeline headlessly via Papermill
make test          Run test suite inside container
make clean-outputs Remove all generated outputs
```