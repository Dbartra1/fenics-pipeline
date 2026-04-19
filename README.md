# fenics-pipeline — a 3D printing walkthrough

Take a 3D printable part, tell the computer where it's loaded and where it's
bolted down, and get back a lighter, stronger version of that part —
**topology optimized** — as an STL ready for your slicer.

This is the same class of tool that sits inside Altair Inspire, nTopology, and
Fusion 360 Generative Design, except you own all of it and it costs nothing.
It runs on your laptop. No cloud, no subscription, no license server.

This guide walks you through it end-to-end, from nothing installed on a Windows
machine to a printable STL in your `outputs/` folder. If you get stuck, there's
a troubleshooting section at the bottom.

---

## What you end up with

You point the pipeline at one of the included parametric parts — say, a
mounting bracket — describe the load on it (for example: *pushing down on the
top face with 10 kN*), and it produces an optimized mesh like this:

```
  Input bracket (solid)           Optimized bracket (~40% material)
  ┌─────────────────┐             ┌──╮     ╱╲     ╱╲    ╭──┐
  │                 │             │   ╲   ╱  ╲   ╱  ╲   │
  │                 │     →       │    ╲ ╱    ╲ ╱    ╲ ╱│
  │                 │             │    ╱ ╲    ╱ ╲    ╱ ╲│
  └─────────────────┘             └──╯     ╲╱     ╲╱    ╰──┘
```

The result is a watertight STL (exported by the pipeline to
`outputs/meshes/<part_name>_optimized.stl`) you can drop straight into your
slicer.

---

## What you need

### Hardware

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS, or Linux | Windows 11 or Ubuntu 22.04+ |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores (solver is parallelized) |
| Disk | 25 GB free | 50 GB free |
| GPU | **Not required** | NVIDIA with CUDA 12+ (optional, see Part 4) |

The solver is CPU-parallel and runs fine without a discrete GPU. If you have
an NVIDIA card with the CUDA toolkit set up, there's a path to turning on GPU
acceleration later — but it's a separate, optional build. Everything in this
tutorial works on CPU.

### Software

We're going to install:

1. **WSL2** — Linux inside Windows (skip if you're on macOS or Linux)
2. **Docker Desktop** — runs the containerized pipeline
3. **Git** — to clone the repo
4. **Rust toolchain** — one-time, to build the solver binary

None of these are heavy, and they're all free.

---

## Part 1 — Install WSL2 (Windows only)

If you're on macOS or Linux, skip to Part 2.

Open **PowerShell as Administrator** (right-click Start → "Windows PowerShell
(Admin)" or "Terminal (Admin)") and run:

```powershell
wsl --install -d Ubuntu-22.04
```

This installs the Windows Subsystem for Linux and an Ubuntu distribution.
**Reboot when it asks.**

After reboot, Ubuntu launches automatically and asks you to create a username
and password. Pick anything — this is just for the Linux VM, not tied to your
Windows account. Write the password down; you'll need it for `sudo`.

Verify it's working:

```powershell
wsl --status
```

You should see `Default Distribution: Ubuntu-22.04` and `Default Version: 2`.

From now on, any command block that says **WSL2 terminal** means: open the
Ubuntu app from your Start menu, or type `wsl` into PowerShell.

---

## Part 2 — Install Docker Desktop

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Run the installer. When it asks, leave "Use WSL 2 instead of Hyper-V"
   checked.
3. Start Docker Desktop. It'll sit in your system tray.
4. Open **Settings → Resources → WSL Integration**. Make sure the toggle for
   your Ubuntu distro is **on**. Click Apply & Restart.

Verify from your **WSL2 terminal**:

```bash
docker --version
docker compose version
```

You should see version numbers for both. If `docker compose version` fails,
you have an old version — update Docker Desktop.

---

## Part 3 — Clone the repo

From your **WSL2 terminal**:

```bash
cd ~
git clone https://github.com/Dbartra1/fenics-pipeline.git
cd fenics-pipeline/fenics-pipeline
```

> Yes, there's a `fenics-pipeline` folder inside a `fenics-pipeline` folder.
> The inner one is where you work.

Take a look around:

```bash
ls
```

You should see `Dockerfile`, `docker-compose.yml`, `Makefile`, `notebooks/`,
`scad/`, `solver/`, and so on.

---

## Part 4 — Do you have a GPU? (decide now)

The shipped `docker-compose.yml` assumes you have an NVIDIA GPU and the
`nvidia-container-toolkit` installed on your host. **If you don't, the default
`docker compose up` will fail** before Jupyter ever starts.

Two paths:

### Path A — No GPU (the common case)

Create a CPU-only override file in the repo root:

```bash
cat > docker-compose.cpu.yml << 'EOF'
# docker-compose.cpu.yml — override for CPU-only machines.
# Removes NVIDIA device reservation and the CUDA toolkit mount.
# Use with:  docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d

services:
  fenics-pipeline:
    volumes:
      # Shadow the CUDA mount so hosts without it don't fail
      - type: tmpfs
        target: /usr/local/cuda-12.6
    deploy:
      resources:
        reservations:
          # Empty devices list overrides the NVIDIA reservation
          devices: []
EOF
```

Any time the rest of this tutorial says `docker compose up`, **use this
instead**:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

If that's annoying, add an alias to your `~/.bashrc`:

```bash
echo "alias dcomp='docker compose -f docker-compose.yml -f docker-compose.cpu.yml'" >> ~/.bashrc
source ~/.bashrc
```

Then use `dcomp up -d`, `dcomp exec`, etc.

### Path B — You have an NVIDIA GPU

Install the NVIDIA Container Toolkit on WSL2:

```bash
# Add NVIDIA's repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker Desktop from the Windows tray.
```

Test:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If that prints your GPU, you're set — use `docker compose up` as-is throughout
the rest of this tutorial.

**Important:** GPU passthrough makes Docker start, but the Rust solver
currently doesn't use the GPU unless you build it with `--features gpu`. The
Python/FEA side does get GPU benefit. CPU solver is fine for all the included
example parts.

---

## Part 5 — Give WSL2 enough memory (Windows only)

WSL2 caps memory at 50% of your physical RAM by default. The mesher wants
more than that on anything but small parts.

From a **Windows PowerShell terminal** (not WSL2):

```powershell
cd $env:USERPROFILE
wsl -- bash -c "cd /home/$USER/fenics-pipeline/fenics-pipeline && pwd"
```

If that printed a path, your repo is where WSL2 can see it. Now run the setup
script from inside the repo:

```powershell
cd \\wsl$\Ubuntu-22.04\home\YOUR_WSL_USERNAME\fenics-pipeline\fenics-pipeline
powershell.exe -ExecutionPolicy Bypass -File scripts\setup_wsl.ps1
```

Replace `YOUR_WSL_USERNAME` with the username you picked in Part 1.

Default is 24 GB RAM, 10 CPUs, 8 GB swap. Override if you need to:

```powershell
.\scripts\setup_wsl.ps1 -MemoryGB 16 -Processors 6 -SwapGB 4
```

The script backs up your existing `.wslconfig` (if any), writes the new one,
and restarts WSL2. After it finishes, reopen your WSL2 terminal.

---

## Part 6 — Build the Rust solver (one-time)

The optimizer is a standalone binary written in Rust. The Docker container
doesn't have Rust installed (keeps the image small), so we build the binary
on WSL2 and drop it into `bin/` for the container to use.

**Install Rust** in your WSL2 terminal:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version
```

**Build the solver:**

```bash
cd ~/fenics-pipeline/fenics-pipeline/solver
cargo build --release
```

First build downloads dependencies and takes 3–5 minutes. Later builds are
seconds.

**Copy the binary into place:**

```bash
mkdir -p ../bin
cp target/release/simp_solver ../bin/simp_solver
chmod +x ../bin/simp_solver
cd ..
ls -lh bin/simp_solver
```

You should see something like:

```
-rwxr-xr-x 1 you you 1.3M Nov 15 14:23 bin/simp_solver
```

> **Heads up:** `bin/simp_solver` is in `.gitignore`. You build it once on
> your machine and it stays there. If you ever update the solver source
> (`solver/src/*.rs`), repeat `cargo build --release` and re-copy.

---

## Part 7 — Build the Docker container

Still in `fenics-pipeline/fenics-pipeline/`:

**Path A (no GPU):**
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
```

**Path B (GPU):**
```bash
docker compose build
```

This pulls the FEniCSx base image and installs OpenSCAD, gmsh, JupyterLab,
and the Python scientific stack. **First build takes 10–15 minutes** on a
decent internet connection. After that, rebuilds are incremental.

---

## Part 8 — Start JupyterLab

**Path A:**
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

**Path B:**
```bash
docker compose up -d
```

Wait about 20 seconds for it to fully start, then open a browser and go to:

```
http://localhost:8888
```

You should see JupyterLab with the repo files listed on the left.

**If the page doesn't load:**

```bash
docker compose logs fenics-pipeline
```

Look for a line starting with `[C ServerApp]` — that tells you what's wrong.
The most common issue is Docker didn't fully start; just wait another minute
and refresh.

---

## Part 9 — Run your first part

You have two easy ways to run the pipeline:

### Easy way: the orchestrator notebook

Inside JupyterLab, open **`getting_started.ipynb`** (in the repo root) and
step through it. It's designed to be the first thing you touch — it checks
your setup, explains what each stage does, and runs the whole pipeline for
you on the included `base_part`.

### Command-line way: Papermill

From your WSL2 terminal, outside the container:

```bash
make run
```

That's it. Papermill executes all five stage notebooks back-to-back for
`base_part`. It takes 5–10 minutes end-to-end on a modern CPU. Progress
prints to your terminal. A full run looks like:

```
[stage 1/5] 01_geometry_openscad.ipynb        ok  ( 12.3s)
[stage 2/5] 02_mesh_gmsh.ipynb                ok  ( 47.8s)
[stage 3/5] 03_fea_fenicsx.ipynb              ok  ( 89.1s)
[stage 4/5] 04_simp_optimization.ipynb        ok  (184.5s)
[stage 5/5] 05_stl_export.ipynb               ok  ( 23.7s)
Pipeline complete. Wrote outputs/meshes/base_part_optimized.stl
```

---

## Part 10 — Get your STL

```bash
ls outputs/meshes/*.stl
```

You should see `base_part_optimized.stl`. Copy it out of WSL2 onto Windows:

```bash
cp outputs/meshes/base_part_optimized.stl /mnt/c/Users/YOUR_WINDOWS_USER/Desktop/
```

Open it in your slicer. Slice. Print.

There's also a preview PNG at `outputs/reports/base_part_stl_wireframe.png`
if you want to eyeball the geometry before slicing.

---

## Running on your own part

Four example parts are included in `scad/`:

- **`base_part`** — flat mounting bracket (default)
- **`cantilever_arm`** — L-shaped cantilever
- **`motor_mount`** — motor mounting plate with hole pattern
- **`tripod_mount_base`** — camera mount

Each has a matching `<name>_params.json` that controls geometry dimensions,
mesh density, and load case. To run one:

1. Copy its params file to `scad/params.json`:
   ```bash
   cp scad/motor_mount_params.json scad/params.json
   ```
2. Edit `scad/params.json` if you want to change dimensions or load.
3. Run: `make run`.

### Parameters that matter for your first runs

In `scad/params.json`:

| Knob | What it does | Typical range |
|---|---|---|
| `geometry.*` | Dimensions of the input part (mm) | Part-specific |
| `mesh_hints.target_element_size` | Mesh resolution. Smaller = finer, slower. | 1.5–8.0 mm |
| `load_case.load.magnitude_n` | Load in newtons | 100–50000 |
| `load_case.load.direction` | Unit vector, e.g. `[0,0,-1]` = pushing down | — |
| `load_case.fixed.face` | Which face is bolted down | `z_min`, `z_max`, etc. |

In `notebooks/04_simp_optimization.ipynb` Cell 0:

| Knob | What it does | Typical range |
|---|---|---|
| `VOLUME_FRACTION` | How much material to keep | 0.3–0.5 |
| `FILTER_RADIUS` | Minimum feature size (mm) | 3–10 |
| `PENAL` | How aggressively to push to solid/void | Leave at 3.0 |
| `MAX_ITERATIONS` | Optimizer iteration cap | 50–200 |

Lower `VOLUME_FRACTION` = lighter part, takes more iterations to converge,
may produce stringy results. Higher `FILTER_RADIUS` = smoother geometry but
less material savings.

---

## Bringing your own STL

If you already have a part designed elsewhere (Fusion 360, SolidWorks,
FreeCAD, etc.), export it as STL and skip stage 1:

```bash
make import-stl STL=/mnt/c/Users/YOU/Desktop/my_part.stl PART=my_part SIZE=2.0 FACE=top LOAD=5000
```

Then open `notebooks/02_mesh_gmsh.ipynb` and run all cells. The rest of the
pipeline picks up from there.

---

## Stopping and restarting

Stop the container (keeps all your data):

```bash
# Path A
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down
# Path B
docker compose down
```

Bring it back up later with the same `up -d` command from Part 8.

Nuke all generated outputs (keeps the container):

```bash
make clean-outputs
```

Full tear-down (removes the container image; next `build` starts from scratch):

```bash
docker compose down --rmi all
```

---

## Troubleshooting

### `docker compose up` fails with `could not select device driver "" with capabilities: [[gpu]]`

You're on Path A but used Path B's command. Either:
- Use `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d`, or
- Set up the `dcomp` alias from Part 4.

### `ModuleNotFoundError: No module named 'dolfinx'` in a notebook

The Jupyter kernel selector at the top right of your notebook isn't on
**"FEniCSx Pipeline"**. Click it, switch to that kernel, and re-run.

### Stage 2 (meshing) runs out of memory

Either:
- Bump WSL2 memory (rerun `setup_wsl.ps1` with a higher `-MemoryGB`), or
- Increase `mesh_hints.target_element_size` in `scad/params.json` (fewer, larger elements).

### Stage 5 produces an empty or broken STL

The optimizer converged to a density field that the STL exporter can't
contour. In `04_simp_optimization.ipynb` Cell 0, increase `FILTER_RADIUS` by
~50% and rerun from stage 4. You don't need to re-mesh or re-run FEA.

### Stage 4 (the Rust solver) errors with `solver binary not found`

You skipped or fumbled Part 6. Rebuild:

```bash
cd solver && cargo build --release && cp target/release/simp_solver ../bin/simp_solver && cd ..
```

### The solver ran but the STL looks like random noise

Your load case probably doesn't make physical sense — for example, loading
a face that has no fixed counterpart, so the part just translates in space
instead of deforming. Check `load_case` in your params JSON: there should be
a `fixed` section pinning one face, and a `load` section applying force to
a different face.

### JupyterLab keeps disconnecting / kernel keeps dying

Docker Desktop ran out of memory. Open Docker Desktop → Settings → Resources
and give it more. 8 GB is a reasonable floor; 16 GB is better.

---

## What's actually happening (quick tour)

If you want to know what each pipeline stage does:

1. **`01_geometry_openscad.ipynb`** — renders `scad/<part>.scad` with your
   parameter values, producing an STL of the starting shape.
2. **`02_mesh_gmsh.ipynb`** — converts that STL into a 3D tetrahedral mesh
   (thousands of tiny tetrahedra filling the part).
3. **`03_fea_fenicsx.ipynb`** — runs finite-element analysis: given the load
   and fixed points you described, computes how every point in the part
   deforms and how stressed each tetrahedron is.
4. **`04_simp_optimization.ipynb`** — the optimizer. Treats the part as a
   grid of voxels, each with a "density" between 0 (void) and 1 (solid).
   Iteratively pushes material from low-stress regions to high-stress
   regions until it converges on a shape that handles the load with the
   minimum material. This is the Rust binary; it runs hot on all your cores.
5. **`05_stl_export.ipynb`** — runs marching cubes on the final density
   field to get a triangle mesh, cleans it up, and writes a watertight STL.

All intermediate artifacts (meshes, stress fields, density fields,
convergence plots) land in `outputs/`.

---

## License and contributing

MIT licensed. PRs welcome — bugs, new parts in `scad/`, material presets,
slicer-aware settings, cleaner docs, anything. The [issues
tracker](https://github.com/Dbartra1/fenics-pipeline/issues) is the right
place to start.

Happy printing.
