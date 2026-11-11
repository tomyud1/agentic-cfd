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


def run(cmd, case_dir, log_name=None, timeout=None):
    """Run an OpenFOAM command in case_dir, logging to log.<name>. Returns wall time in seconds.

    Raises RuntimeError on a non-zero exit, subprocess.TimeoutExpired after `timeout` seconds.
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


def _killpg(pgid, sig=signal.SIGKILL):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def stop(case_dir):
    """Kill every OpenFOAM run started by run() in case_dir (reads its *.pid files). Returns killed ids."""
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
    return killed


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


def read_dat(path):
    """Rows of a postProcessing .dat file, skipping '#' comments. Vectors '(a b c)' become tuples."""
    rows = []
    for line in Path(path).read_text().splitlines():
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
    # python -m cfdlib.foam stop <case>   kill the runs started by foam.run() in that case
    import sys
    if sys.argv[1:2] == ["ps"]:
        procs = running()
        print("\n".join(f"{p[0]:>7}  {p[1]:>10}  {p[2]}  {p[3]}" for p in procs) or "no OpenFOAM processes")
    elif sys.argv[1:2] == ["stop"] and len(sys.argv) == 3:
        print("killed process groups:", stop(sys.argv[2]) or "none")
    else:
        sys.exit("usage: python -m cfdlib.foam ps | stop <case_dir>")
