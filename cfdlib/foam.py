"""Run OpenFOAM commands and read their ASCII output."""
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/lib/openfoam/openfoam2606/etc/bashrc")


def foam_env(var):
    """Value of an OpenFOAM environment variable, e.g. WM_PROJECT_VERSION or FOAM_RUN."""
    out = subprocess.run(["bash", "-c", f"source {FOAM_BASHRC} >/dev/null 2>&1; echo ${var}"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def run(cmd, case_dir, log_name=None, timeout=None, watch=None, watch_every=10.0):
    """Run an OpenFOAM command in case_dir, logging to log.<name>. Returns wall time in seconds.

    Raises RuntimeError on a non-zero exit, subprocess.TimeoutExpired after `timeout` seconds.
    watch: called with no arguments every `watch_every` seconds while the command runs (live checks,
      e.g. cfdlib.monitor.LiveStop, which ends the run cleanly once a physical check holds). An
      exception inside it is printed and the run continues: a broken check must not kill a good run.
    """
    case_dir = Path(case_dir)
    name = log_name or cmd.split()[0]
    log = case_dir / f"log.{name}"
    pidfile = case_dir / f"{name}.pid"
    t0 = time.perf_counter()
    with open(log, "w") as f:
        # Own process group, so a timeout kills the solver too, not just the bash wrapper.
        # The group id goes to <case>/<name>.pid so foam.stop() can kill an orphaned run later.
        p = subprocess.Popen(["bash", "-c", f"source {FOAM_BASHRC} && {cmd}"],
                             cwd=case_dir, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        pidfile.write_text(str(p.pid))
        ru = None
        t_watch = t0
        try:
            while True:                            # wait4: exit status AND resource use of this run only
                pid, status, ru = os.wait4(p.pid, os.WNOHANG)
                if pid:
                    p.returncode = os.waitstatus_to_exitcode(status)
                    break
                if timeout is not None and time.perf_counter() - t0 > timeout:
                    _killpg(p.pid)
                    p.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if watch is not None and time.perf_counter() - t_watch >= watch_every:
                    t_watch = time.perf_counter()
                    try:
                        watch()
                    except Exception as e:         # noqa: BLE001
                        print(f"watch failed ({type(e).__name__}: {e}); run continues", flush=True)
                time.sleep(0.1)
        except subprocess.TimeoutExpired:
            raise
        except BaseException:                      # Ctrl-C / SystemExit: do not leave an orphan solver
            _killpg(p.pid)
            raise
        finally:
            pidfile.unlink(missing_ok=True)
    wall = time.perf_counter() - t0
    # Performance record for cfdlib.perf (CPU time includes all processes of the run, e.g. MPI ranks).
    (case_dir / f"perf.{name}.json").write_text(json.dumps(dict(
        cmd=cmd, wall_s=wall, cpu_s=(ru.ru_utime + ru.ru_stime) if ru else None,
        peak_mem_mb=ru.ru_maxrss / 1024 if ru else None, exit=p.returncode)))
    if p.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"'{cmd}' failed in {case_dir} (exit {p.returncode}):\n{tail}")
    return wall


def map_fields(case_dir, source_dir, consistent=True, keep_new=False):
    """Initial fields of case_dir = the latest time of source_dir, interpolated by mapFields (grid
    sequencing: a fine mesh starts from a coarser mesh's converged answer, so it only has to remove
    the coarse mesh's error). consistent: same geometry and patch names (meshes of one family).

    mapFields maps every volume field in the source time, including post-processing output written
    there (C, Cx, Cy, Cz, V from writeCellCentres / writeCellVolumes). Field files the target did not
    have before are deleted afterwards unless keep_new (case 6: they became undocumented settings and
    the contract refused the report). mapFields also rewrites the target's field files in expanded form
    (constraint patches such as wedge or empty written out instead of an #includeEtc line), so a
    decision must cover those patch entries. Returns (wall time, list of deleted files).

    OpenFOAM file names cannot contain spaces: mapFields fails on a source path or working directory
    with one (for example /mnt/d/Projects/Agentic CFD/...). Keep runs under $FOAM_RUN."""
    for p in (case_dir, source_dir):
        if " " in str(Path(p).resolve()):
            raise ValueError(f"map_fields: OpenFOAM cannot use a path with a space: {p}")
    before = {f for d in time_dirs(case_dir) for f in d.iterdir() if f.is_file()}
    flag = " -consistent" if consistent else ""
    wall = run(f"mapFields {Path(source_dir).resolve()}{flag} -sourceTime latestTime", case_dir, "mapFields")
    removed = []
    if not keep_new:
        for d in time_dirs(case_dir):
            for f in d.iterdir():
                if f.is_file() and f not in before:
                    f.unlink()
                    removed.append(str(f.relative_to(case_dir)))
    return wall, removed


def ignored_settings(case_dir):
    """Settings the case's solver silently ignores, as warnings. Today: SIMPLE/PIMPLE `consistent yes`
    (SIMPLEC) for a solver whose source never reads it. Case 6: buoyantBoussinesqSimpleFoam has no
    SIMPLEC, so `consistent yes` ran plain SIMPLE with an unrelaxed pressure and diverged."""
    from . import foamdict
    case_dir = Path(case_dir)
    try:
        ctrl = foamdict.parse(case_dir / "system/controlDict")
        sol = foamdict.parse(case_dir / "system/fvSolution")
    except (OSError, ValueError, IndexError):
        return []
    app = str(ctrl.get("application", "")).strip()
    asked = [k for k in ("SIMPLE", "PIMPLE")
             if str((sol.get(k) or {}).get("consistent", "")).strip() in ("yes", "true", "on", "1")]
    if not app or not asked:
        return []
    root = Path(foam_env("FOAM_SOLVERS"))
    src = next((d for d in root.rglob(app) if d.is_dir() and (d / f"{app}.C").exists()), None)
    if src is None:
        return []                                  # solver source not installed: cannot tell
    text = "".join(f.read_text(errors="replace") for f in src.rglob("*") if f.suffix in (".C", ".H"))
    if "consistent()" in text:
        return []
    return [f"{app} does not read '{k}/consistent' (no SIMPLEC in its source): the setting is ignored and "
            f"the pressure-velocity coupling is plain {k}; relax the pressure (typical U 0.7, p 0.3)" for k in asked]


def _killpg(pgid, sig=signal.SIGKILL):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def stop(case_dir):
    """Kill every OpenFOAM run started by run() in case_dir (reads its *.pid files), then any OpenFOAM
    process still working in case_dir (an orphan whose Python wrapper died and took the PID file with
    it, case 6). Returns the killed ids."""
    killed = []
    for pidfile in Path(case_dir).glob("*.pid"):
        try:
            pgid = int(pidfile.read_text().strip())
        except ValueError:
            continue
        _killpg(pgid, signal.SIGTERM)
        time.sleep(1)
        _killpg(pgid)
        pidfile.unlink(missing_ok=True)
        killed.append(pgid)
    for pid, _, cwd, _ in _running_in(case_dir):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            time.sleep(0.5)
        killed.append(pid)
    return killed


def _running_in(case_dir):
    """OpenFOAM processes whose working directory is case_dir or inside it."""
    root = str(Path(case_dir).resolve())
    return [p for p in running() if p[2] == root or p[2].startswith(root + "/")]


def running():
    """OpenFOAM processes running now (any case): list of (pid, elapsed, working dir, command)."""
    out = subprocess.run(["ps", "-eo", "pid=,etime=,args="], capture_output=True, text=True).stdout
    procs = []
    for line in out.splitlines():
        pid, etime, args = line.split(None, 2)
        exe = args.split()[0].rsplit("/", 1)[-1]
        if exe.endswith("Foam") or exe in ("blockMesh", "snappyHexMesh", "postProcess", "mpirun"):
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
            except OSError:
                cwd = "?"
            procs.append((int(pid), etime, cwd, args))
    return procs


def render_case(template_dir, case_dir, params):
    """Copy a case template, filling {{ placeholders }} in every file with params."""
    template_dir, case_dir = Path(template_dir), Path(case_dir)
    if case_dir.exists():
        live = _running_in(case_dir)
        if live:          # deleting under a running solver lets it write into the new case (case 6)
            raise RuntimeError(f"render_case: a solver is still running in {case_dir} (pids "
                               f"{[p[0] for p in live]}); stop it first: python -m cfdlib.foam stop {case_dir}")
        shutil.rmtree(case_dir)
    env = Environment(loader=FileSystemLoader(str(template_dir)), undefined=StrictUndefined,
                      keep_trailing_newline=True)
    for src in template_dir.rglob("*"):
        if src.is_file():
            rel = src.relative_to(template_dir)
            dst = case_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(env.get_template(rel.as_posix()).render(**params))
    return case_dir


def time_dirs(case_dir):
    """Numeric time directories of a case, sorted by time."""
    dirs = [d for d in Path(case_dir).iterdir() if d.is_dir() and re.fullmatch(r"[0-9.eE+-]+", d.name)]
    return sorted(dirs, key=lambda d: float(d.name))


def read_internal_field(path):
    """internalField of an ASCII OpenFOAM field file as a numpy array: (n,) scalars or (n, 3) vectors."""
    text = Path(path).read_text()
    i = text.index("internalField")
    m = re.match(r"internalField\s+uniform\s+([^;]+);", text[i:])
    if m:
        raise ValueError(f"{path}: internalField is uniform ({m.group(1).strip()}), expected a list")
    m = re.match(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text[i:])
    if not m:
        raise ValueError(f"{path}: cannot parse internalField header")
    kind, n = m.group(1), int(m.group(2))
    start = i + m.end()
    end = text.index(";", start)
    body = text[start:end].rstrip()
    assert body.endswith(")"), f"{path}: unterminated list"
    body = body[:-1].replace("(", " ").replace(")", " ")
    vals = np.array(body.split(), dtype=float)
    width = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9}[kind]
    if vals.size != n * width:
        raise ValueError(f"{path}: expected {n}x{width} values, got {vals.size}")
    return vals if width == 1 else vals.reshape(n, width)


def read_boundary_field(path, patch):
    """Values of an ASCII OpenFOAM field file on one boundary patch, one per face in the patch's face
    order: (n,) scalars or (n, 3) vectors (for example wallShearStress or nut on a wall). A uniform value
    comes back as a single value, shape () or (3,). Raises ValueError if the patch has no `value` entry
    (zeroGradient and similar types do not store one: write the field with a function object, or use the
    cell values next to the patch)."""
    text = Path(path).read_text()
    b = text.index("boundaryField")
    m = re.search(r"\n\s*" + re.escape(patch) + r"\s*\{", text[b:])
    if not m:
        raise ValueError(f"{path}: no patch '{patch}' in boundaryField")
    start = b + m.end()
    depth, k = 1, start
    while depth and k < len(text):                # the patch's block, skipping the parentheses of lists
        ch = text[k]
        depth += (ch == "{") - (ch == "}")
        k += 1
    block = text[start:k - 1]
    m = re.search(r"(?<![A-Za-z0-9_])value\s+uniform\s+([^;]+);", block)
    if m:
        v = np.array(m.group(1).replace("(", " ").replace(")", " ").split(), dtype=float)
        return v[0] if v.size == 1 else v
    m = re.search(r"(?<![A-Za-z0-9_])value\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", block)
    if not m:
        raise ValueError(f"{path}: patch '{patch}' stores no value (type "
                         f"{(re.search(r'type\s+(\w+)', block) or [None, '?'])[1]})")
    kind, n = m.group(1), int(m.group(2))
    body = block[m.end():block.index(";", m.end())].rstrip()
    assert body.endswith(")"), f"{path}: unterminated list on patch {patch}"
    vals = np.array(body[:-1].replace("(", " ").replace(")", " ").split(), dtype=float)
    width = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9}[kind]
    if vals.size != n * width:
        raise ValueError(f"{path}: patch {patch}: expected {n}x{width} values, got {vals.size}")
    return vals if width == 1 else vals.reshape(n, width)


def read_dat(path):
    """Rows of a postProcessing .dat file, skipping '#' comments. Vectors '(a b c)' become tuples."""
    return parse_dat(Path(path).read_text())


def parse_dat(text):
    """Rows of the text of a postProcessing .dat file (see read_dat)."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = re.findall(r"\([^)]*\)|\S+", line)
        row = []
        for t in tokens:
            if t.startswith("("):
                row.append(tuple(float(v) for v in t[1:-1].split()))
            else:
                try:
                    row.append(float(t))
                except ValueError:
                    row.append(t)
        rows.append(row)
    return rows


def residual_history(log_path, fields=("Ux", "p")):
    """Initial residual of the first solve of each field per iteration, from a solver log."""
    hist = {f: [] for f in fields}
    seen = set()
    for line in Path(log_path).read_text().splitlines():
        if line.startswith("Time = "):
            seen.clear()
            continue
        m = re.search(r"Solving for (\w+), Initial residual = ([-0-9.eE+]+)", line)
        if m and m.group(1) in hist and m.group(1) not in seen:
            hist[m.group(1)].append(float(m.group(2)))
            seen.add(m.group(1))
    return hist


if __name__ == "__main__":
    # python -m cfdlib.foam ps            list running OpenFOAM processes and their case directories
    # python -m cfdlib.foam stop <case>   kill the runs started by foam.run() in that case, and any
    #                                     OpenFOAM process still working inside it (orphans)
    # python -m cfdlib.foam check <case>  settings the case's solver silently ignores
    import sys
    if sys.argv[1:2] == ["ps"]:
        procs = running()
        print("\n".join(f"{p[0]:>7}  {p[1]:>10}  {p[2]}  {p[3]}" for p in procs) or "no OpenFOAM processes")
    elif sys.argv[1:2] == ["stop"] and len(sys.argv) == 3:
        print("killed:", stop(sys.argv[2]) or "none")
    elif sys.argv[1:2] == ["check"] and len(sys.argv) == 3:
        print("\n".join(ignored_settings(sys.argv[2])) or "no ignored settings found")
    else:
        sys.exit("usage: python -m cfdlib.foam ps | stop <case_dir> | check <case_dir>")
