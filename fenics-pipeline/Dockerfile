# syntax=docker/dockerfile:1.4
ARG DOLFINX_VERSION=v0.9.0
FROM ghcr.io/fenics/dolfinx/dolfinx:${DOLFINX_VERSION}

LABEL maintainer="your@email.com"
LABEL description="FEniCSx + gmsh + Papermill + PyVista topology optimization pipeline"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dri \
    libegl1 \
    libgles2 \
    libxrender1 \
    libxext6 \
    xvfb \
    openscad \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

RUN pip install --no-cache-dir \
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
        h5py \
        meshio && \
    python -m ipykernel install --user --name "fenics-pipeline" --display-name "FEniCSx Pipeline"

RUN openscad --version 2>&1 | grep -q "OpenSCAD" || (echo "OpenSCAD install failed" && exit 1)

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYVISTA_OFF_SCREEN=true
ENV MESA_GL_VERSION_OVERRIDE=3.3

WORKDIR /workspace

RUN printf '#!/bin/bash\nexec "$@"\n' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]

EXPOSE 8888