"""Turbulence resolved in time, without a model, in a flow between two walls (a plane channel).

gpucfd has no turbulence model. On a grid fine enough to carry the eddies it computes a turbulent flow
directly (direct numerical simulation); the outputs are then averages over time and over the directions
in which the flow repeats. Measured on the plane channel at friction Reynolds number 180 (numbers,
costs and limits: DESIGN.md, "Turbulence resolved without a model").

What is here, for a box with walls in y and periodic x (the flow direction) and z:

    from gpucfd import gcfd, grids, turbulence
    y = grids.tanh_wall(ny, 2 * delta, h_wall)        # wall cell of a chosen height
    s = gcfd.NavierStokes(n, (lx, 2 * delta, lz), nu, walls=(False, True, False), force=(f, 0, 0),
                          scheme="imex", grid=(None, y, None))
    turbulence.channel_start(s, u_tau)                # a start field that turns turbulent quickly
    st = turbulence.PlaneStats(s)                     # geometry, once
    o = st.sample()                                   # plane averages of the field s holds now

- channel_start(): a turbulent mean profile with streaks, vortices and noise on top. From rest, or
  from the laminar solution with small noise, the flow of this equation set can stay laminar for a very
  long time (the laminar solution is linearly stable up to high Reynolds numbers), and the laminar
  centre speed is many times the turbulent one, which forces a tiny time step.
- PlaneStats.sample(): mean velocity U(y), the fluctuation intensities, the turbulent shear stress
  and the wall shear, formed exactly as the solver forms its own momentum fluxes. So the balance of
  mean momentum holds to round-off for every single field (tests/test_gpu_tools.py), and after
  averaging in time what is left of it measures how well the average has settled, not the arithmetic
  of the post-processing.
- For the averages in time and their statistical error: cfdlib.signals.stationary(). The bulk velocity
  is the exception: it settles slowly (settling_time(), about 9 time units at friction Reynolds number
  180), so its error comes from bulk_average(), and channel_watch() is a stop rule built on it.
- For a finer grid started from a coarser one: gpucfd.steady.prolong(coarse, fine).
"""
import numpy as np


def reichardt(y_plus, kappa=0.41):
    """A smooth fit of the mean velocity of a turbulent wall layer in wall units (Reichardt 1951): the
    viscous layer u+ = y+ near the wall, the logarithmic law further out. A start profile, not a result."""
    y_plus = np.asarray(y_plus, float)
    return np.log(1.0 + kappa * y_plus) / kappa + 7.8 * (1.0 - np.exp(-y_plus / 11.0) - y_plus / 11.0 * np.exp(-y_plus / 3.0))


def _check_channel(s):
    if s.ncomp != 3 or s.kinds[0] or s.kinds[4] or not (s.kinds[2] and s.kinds[3]):
        raise ValueError("needs a 3D box with walls in y and periodic x and z")


def channel_start(s, u_tau, seed=0, streaks=3, amplitude=1.0):
    """Set s to a field that becomes turbulent within a few time units delta / u_tau, and project it.

    The mean is Reichardt's profile for the friction velocity u_tau (for a channel driven by a force f
    per unit mass, u_tau = sqrt(f delta), delta = half the distance between the walls). On top:
    `streaks` pairs of fast and slow streaks across z, wavy along x, with vortices along the flow, and
    random noise of 5 % of the centre speed. amplitude scales everything added to the mean (1: the
    values that started turbulence on 64 x 64 x 64 and 192 x 64 x 64 at friction Reynolds number 180).
    The start field is not turbulence: the first 10 to 30 time units of the run are a transient and
    do not belong in any average. Returns max |div u| after the projection."""
    _check_channel(s)
    rng = np.random.default_rng(seed)
    lx, ly, lz = s.length
    delta = 0.5 * ly
    kx, kz = 2.0 * np.pi / lx, 2.0 * np.pi / lz
    uc = u_tau * float(reichardt(delta * u_tau / s.nu))
    out = []
    for c in range(3):
        X, Y, Z = s.face_coords(c)
        env = 1.0 - (Y / delta - 1.0) ** 2
        if c == 0:
            mean = u_tau * reichardt(np.minimum(Y, ly - Y) * u_tau / s.nu)
            a = 0.25 * uc * env * np.cos(streaks * kz * Z) * (1.0 + 0.3 * np.sin(2 * kx * X))
        elif c == 1:
            mean = 0.0
            a = 0.10 * uc * env ** 2 * np.cos(streaks * kz * Z) * np.cos(kx * X)
        else:
            mean = 0.0
            a = 0.10 * uc * env * np.sin(streaks * kz * Z) * np.sin(2 * kx * X)
        out.append(mean + amplitude * (a + 0.05 * uc * env * rng.standard_normal(a.shape)))
    s.set(out)
    return s.project()


class PlaneStats:
    """Plane averages (over x and z) of a channel field, on the solver's own staggered positions.

    y: the ny cell centres (where u and w live); y_faces: the ny + 1 faces from wall to wall (where v
    lives). sample() returns, for the field s holds now (or for q = s.get()):
      U, W      mean streamwise and spanwise velocity at y (W is zero in the long run; its variance in
                time is what ww lacks to be the fluctuation about the full mean)
      uu, ww    plane averages of u'^2 and w'^2 at y (' = minus the plane mean)
      vv        plane average of v^2 at y_faces (zero on both walls)
      uv        turbulent shear stress, plane average of u v at y_faces, with u and v brought to the
                cell edge the way the solver's momentum flux does it (zero on both walls)
      dUdy      gradient of U at y_faces as the solver's diffusion term forms it; on the walls it uses
                the half cell, so nu dUdy[0] and - nu dUdy[-1] are the wall shear stresses per density
      bulk      mean velocity through the channel (U weighted with the cell heights)
      tau_low, tau_high   wall shear stress per unit density on the two walls
      umax, courant_rate  for cfdlib.gpu.run_transient's checks
    All are linear in the quantities averaged, so their time averages are the averages of the samples
    (cfdlib.signals.stationary takes the series of arrays as they are). From the time averages:
      rms of u = sqrt(time mean of uu). uu is taken about the plane mean of each field; about the
      time-mean profile it is larger by the variance in time of U, which the series of U gives.
      total shear stress per density at y_faces = nu dUdy - uv. In a settled average it is the straight
      line tau_low - f y_faces (f = driving force per unit mass): for a channel of half height delta,
      u_tau^2 (1 - y / delta). stress_balance() returns both and their difference."""

    def __init__(self, s):
        _check_channel(s)
        self.s = s
        self.dy = s.widths(1)
        self.y = s._axis(1, False)
        self.y_faces = np.concatenate((s._axis(1, True), [s.length[1]]))
        self.dyc = np.concatenate(([0.5 * self.dy[0]], np.diff(self.y), [0.5 * self.dy[-1]]))  # distances the gradients use

    def sample(self, q=None):
        s = self.s
        q = s.get() if q is None else q
        u, v, w = q
        U = u.mean(axis=(0, 2))
        up = u - U[None, :, None]
        W = w.mean(axis=(0, 2))
        zero = np.zeros(1)
        vx = 0.5 * (v + np.roll(v, 1, axis=2))                    # v at the x-face of u, on y-face j (j = 0: wall)
        uy = 0.5 * (u[:, 1:, :] + u[:, :-1, :])                   # u at y-faces 1 .. ny - 1
        uv = np.concatenate((zero, (vx[:, 1:, :] * uy).mean(axis=(0, 2)), zero))
        vv = np.concatenate(((v ** 2).mean(axis=(0, 2)), zero))
        dUdy = np.concatenate(([U[0]], np.diff(U), [-U[-1]])) / self.dyc
        return dict(U=U, W=W, uu=(up ** 2).mean(axis=(0, 2)), ww=((w - W[None, :, None]) ** 2).mean(axis=(0, 2)), vv=vv, uv=uv,
                    dUdy=dUdy, bulk=float(np.sum(U * self.dy) / s.length[1]), tau_low=float(s.nu * dUdy[0]),
                    tau_high=float(-s.nu * dUdy[-1]), umax=float(np.sqrt(u ** 2 + v ** 2 + w ** 2).max()),
                    courant_rate=s.courant_rate(q))

    def stress_balance(self, dUdy, uv, tau_wall=None):
        """Total shear stress per density at y_faces from (time-averaged) dUdy and uv, next to the straight
        line a settled average must give. tau_wall: the wall shear the line starts from (default: the
        driving force's value f delta, which the settled flow must reach).
        Returns dict(y_faces, viscous, turbulent, total, exact, residual = total - exact)."""
        s = self.s
        f = s.g[0]
        tw = 0.5 * f * s.length[1] if tau_wall is None else tau_wall
        visc, turb = s.nu * np.asarray(dUdy), -np.asarray(uv)
        exact = tw - f * self.y_faces
        return dict(y_faces=self.y_faces, viscous=visc, turbulent=turb, total=visc + turb, exact=exact,
                    residual=visc + turb - exact)


# ---- the bulk velocity of a channel driven by a constant force: its slow settling, its error, a stop rule ----
# The bulk velocity obeys dU_b/dt = force - wall shear / delta exactly, and the wall shear follows the
# bulk velocity, roughly as U_b^n with n about 1.75 (the Blasius-type friction law). So a disturbance of
# U_b decays with the settling time below: about 9 time units delta / u_tau at friction Reynolds number
# 180. Two consequences, both measured on case 17: a run of tens of time units holds only a few
# independent values of U_b (plain batch means gave an error three times too small), and a finer grid
# started from a coarser one's field needs about two settling times to reach its own level.

def settling_time(bulk, force, exponent=1.75):
    """Time in which the bulk velocity of a force-driven channel forgets a disturbance:
    delta U_b / (n u_tau^2) = U_b / (n force), with u_tau^2 = force x delta and the wall shear ~ U_b^n.
    n = 1.75 is an empirical friction law, good for planning and for the error estimate, not exact."""
    return bulk / (exponent * force)


def bulk_average(t, bulk, tau_wall, force, delta=1.0, t_start=None, exponent=1.75):
    """Mean bulk velocity of a force-driven channel after t_start, with a statistical error that knows
    the slow settling (cfdlib.signals.relaxing: the error comes from the exact balance of the box).
    bulk, tau_wall: the recorded series (tau_wall per unit density: the mean of PlaneStats' tau_low and
    tau_high); force per unit mass, delta the half height.
    Returns the dict of signals.relaxing (mean, stderr, stderr_batch, n_independent, reliable,
    drift_sigma, t0, t1, ...) plus T_s (the settling time) and imbalance (mean wall shear over
    force x delta, minus 1: how far the average is from settled)."""
    from cfdlib import signals
    t, bulk, tau = np.asarray(t, float), np.asarray(bulk, float), np.asarray(tau_wall, float)
    Ts = settling_time(signals.stationary(t, bulk, t_start=t_start)["mean"], force, exponent)
    out = signals.relaxing(t, bulk, force - tau / delta, Ts, t_start=t_start)
    tw = signals.stationary(t, tau, t_start=t_start)["mean"]
    return dict(out, T_s=Ts, imbalance=tw / (force * delta) - 1.0)


def channel_watch(force, delta=1.0, tol=2.5e-3, min_time=None, max_drift=2.0, balance=3e-3, left=2e-3,
                  exponent=1.75, log=None):
    """A stop rule for cfdlib.gpu.run_transient (its `watch`) on a force-driven channel whose observe
    returns PlaneStats.sample(): end the averaging when the mean bulk velocity is known well enough.

    Holds when, after the start-up (cfdlib.signals.settle_time with the settling time as memory, so the
    cut also waits for the slow approach to shrink to `left` of the bulk velocity): the error of the
    mean bulk velocity from bulk_average is below tol times it and reliable, the two halves agree
    within max_drift of their errors, at least min_time is averaged (default: four settling times),
    and the mean wall shear is within `balance` of the driving force. Fix tol before the run: at
    friction Reynolds number 180, 0.25 % took about 38 time units after the start-up, and halving it
    needs four times as long.
    log: a function that gets a line per check while the rule does not hold, with what is missing."""
    from cfdlib import signals

    def watch(t, d):
        tt, bulk = d["t"], d["bulk"]
        tau = 0.5 * (d["tau_low"] + d["tau_high"])
        try:
            Ts = settling_time(float(np.mean(bulk[len(bulk) // 2:])), force, exponent)
            cut = signals.settle_time(tt, bulk, memory=Ts, left=left)
            st = bulk_average(tt, bulk, tau, force, delta, t_start=cut["t_start"], exponent=exponent)
        except ValueError as e:
            if log:
                log(f"bulk velocity: record too short ({e})")
            return None
        need = 4.0 * st["T_s"] if min_time is None else min_time
        tests = {"error": st["stderr"] <= tol * abs(st["mean"]), "reliable": st["reliable"],
                 "drift": abs(st["drift_sigma"]) <= max_drift, "length": st["t1"] - st["t0"] >= need,
                 "balance": abs(st["imbalance"]) <= balance, "start-up found": cut["settled"]}
        text = (f"bulk {st['mean']:.4f} +- {st['stderr']:.4f} ({st['stderr'] / abs(st['mean']):.2%}; batch means alone "
                f"{st['stderr_batch']:.4f}) over t = {st['t0']:.1f} to {st['t1']:.1f}, {st['n_independent']:.0f} independent "
                f"samples, drift {st['drift_sigma']:+.1f} sigma, wall shear {st['imbalance']:+.2%} from the driving force")
        if all(tests.values()):
            return "stop rule holds: " + text
        if log:
            log(text + "; not yet: " + ", ".join(k for k, ok in tests.items() if not ok))
        return None
    return watch
