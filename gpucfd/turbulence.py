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
- For the averages in time and their statistical error: cfdlib.signals.stationary().
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
      U         mean streamwise velocity at y
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
        return dict(U=U, uu=(up ** 2).mean(axis=(0, 2)), ww=((w - W[None, :, None]) ** 2).mean(axis=(0, 2)), vv=vv, uv=uv,
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
