# syntax=docker/dockerfile:1.4
# Base: official dolfinx image — DO NOT swap without re-validating petsc4py/mpi4py ABI
# Tracks: https://github.com/FEniCS/dolfinx/pkgs/container/dolfinx
ARG DOLFINX_VERSION=0.8.0
FROM ghcr.io/fenics/dolfinx/dolfinx:${DOLFINX_VERSION}

LABEL maintainer="your@email.com"
LABEL description="FEniCSx + gmsh + Papermill + PyVista topology optimization pipeline"

# ── System deps ───────────────────────────────────────────────────────────────
# libgl1 + mesa: pyvista/VTK software rendering — no GPU in container
# openscad: required by src/geometry/openscad_runner.py at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    libxrender1 \
    libxext6 \
    xvfb \
    openscad \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
# ORDERING MATTERS:
#   1. gmsh first (bundles its own libs — must precede pyvista)
#   2. pyvista pinned to avoid VTK >9.3 conflict with gmsh's bundled libs
#   3. scikit-image pinned to stay on numpy <2.0 ABI (dolfinx hard constraint)
#   4. papermill + ipykernel last — pure Python, no ABI risk
#
# Used by:
#   gmsh==4.12.2        → src/meshing/gmsh_pipeline.py, src/meshing/mesh_quality.py
#   pyvista==0.43.10    → 03_fea_fenicsx.ipynb, 05_stl_export.ipynb (renders)
#   trimesh==4.4.0      → 05_stl_export.ipynb (STL post-processing)
#   scikit-image==0.22.0→ 05_stl_export.ipynb (marching cubes)
#   papermill==2.6.0    → pipeline_full.ipynb (orchestrator)

SHELL ["/bin/bash", "-c"]

RUN source /usr/local/bin/dolfinx-env.sh && \
    pip install --no-cache-dir \
        gmsh==4.12.2 \
        vtk==9.3.1 \
        pyvista==0.43.10 \
        trimesh==4.4.0 \
        networkx \
        scikit-image==0.22.0 \
        papermill==2.6.0 \
        ipykernel==6.29.5 \
        ipywidgets \
        jupyterlab==4.2.5 \
        notebook==7.2.2 \
        matplotlib \
        scipy \
        tqdm \
        h5py && \
        meshio \
    # Register kernel so pipeline_full.ipynb can find it via Papermill
    python -m ipykernel install --user --name "fenics-pipeline" --display-name "FEniCSx Pipeline"

# ── OpenSCAD sanity check ──────────────────────────────────────────────────
# Validates that openscad_runner.py will have its runtime dep available
RUN openscad --version 2>&1 | grep -q "OpenSCAD" || (echo "OpenSCAD install failed" && exit 1)

# ── MPI config for WSL2/Docker rootless ───────────────────────────────────
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# ── PyVista headless rendering ─────────────────────────────────────────────
# Required by any pyvista calls in 03_fea_fenicsx.ipynb and 05_stl_export.ipynb
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYVISTA_OFF_SCREEN=true
ENV MESA_GL_VERSION_OVERRIDE=3.3

# ── Workspace — maps to NOTEBOOK_DIR in .env ──────────────────────────────
WORKDIR /workspace

# ── Entrypoint: source dolfinx venv before any exec ───────────────────────
# Critical: Papermill spawns a subprocess kernel for pipeline_full.ipynb.
# Without venv activation here, that kernel won't find dolfinx.
RUN echo '#!/bin/bash\nsource /usr/local/bin/dolfinx-env.sh\nexec "$@"' \
    > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]

EXPOSE 8888