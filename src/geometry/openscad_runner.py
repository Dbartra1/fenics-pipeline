# src/geometry/openscad_runner.py
#
# Shells out to the OpenSCAD binary with -D parameter overrides.
# Returns a structured RunResult rather than raising bare exceptions,
# so notebook cells can inspect partial failures without kernel death.

from __future__ import annotations
import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.geometry.param_schema import PipelineParams


@dataclass
class OpenSCADResult:
    success:      bool
    stl_path:     Optional[Path]
    stl_size_kb:  Optional[float]
    duration_s:   float
    stdout:       str
    stderr:       str
    command:      str          # full command string for debugging

    def raise_if_failed(self) -> None:
        if not self.success:
            raise RuntimeError(
                f"OpenSCAD failed.\nCommand: {self.command}\nStderr:\n{self.stderr}"
            )


def _build_command(
    scad_file: Path,
    output_stl: Path,
    defines: dict,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """
    Build the OpenSCAD CLI command.
    Each define becomes:  -D 'KEY=VALUE'
    String values are quoted for the shell.
    """
    cmd = ["openscad", str(scad_file), "-o", str(output_stl)]
    for key, value in defines.items():
        if isinstance(value, str):
            cmd += ["-D", f'{key}="{value}"']
        elif isinstance(value, bool):
            cmd += ["-D", f"{key}={'true' if value else 'false'}"]
        else:
            cmd += ["-D", f"{key}={value}"]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def run_openscad(
    params: PipelineParams,
    scad_file: str | Path = "scad/base_part.scad",
    timeout_s: int = 120,
) -> OpenSCADResult:
    """
    Compile base_part.scad with params injected as -D defines.
    Output STL is written to params.export.stl_output_dir/<part_name>.stl.

    Args:
        params:    Validated PipelineParams instance.
        scad_file: Path to the .scad source. Default: scad/base_part.scad.
        timeout_s: Kill OpenSCAD if it exceeds this. Complex geometry can hang.

    Returns:
        OpenSCADResult — always returns, never raises. Call .raise_if_failed()
        in the notebook if you want hard failure on error.
    """
    scad_path = Path(scad_file).resolve()
    if not scad_path.exists():
        return OpenSCADResult(
            success=False, stl_path=None, stl_size_kb=None,
            duration_s=0.0, stdout="", stderr=f"scad file not found: {scad_path}",
            command=str(scad_path),
        )

    if shutil.which("openscad") is None:
        return OpenSCADResult(
            success=False, stl_path=None, stl_size_kb=None,
            duration_s=0.0, stdout="", stderr="openscad binary not found in PATH",
            command="openscad",
        )

    # Resolve output path
    out_dir = Path(params.export.stl_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stl = out_dir / f"{params.part_name}.stl"

    defines = params.to_openscad_defines()
    cmd = _build_command(scad_path, out_stl, defines)
    cmd_str = " ".join(cmd)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return OpenSCADResult(
            success=False, stl_path=None, stl_size_kb=None,
            duration_s=time.perf_counter() - t0,
            stdout="", stderr=f"OpenSCAD timed out after {timeout_s}s",
            command=cmd_str,
        )
    duration = time.perf_counter() - t0

    # OpenSCAD exits 0 even on some geometry errors — check file too
    stl_ok = out_stl.exists() and out_stl.stat().st_size > 0
    success = proc.returncode == 0 and stl_ok

    return OpenSCADResult(
        success=success,
        stl_path=out_stl if stl_ok else None,
        stl_size_kb=round(out_stl.stat().st_size / 1024, 2) if stl_ok else None,
        duration_s=round(duration, 3),
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=cmd_str,
    )