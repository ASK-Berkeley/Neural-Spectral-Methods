from .. import *
from ...dists import *
from ...basis import *

from .._domain import *
from .._params import *

from ...basis.fourier import *
from ...basis.chebyshev import *

def k(nx: int, ny: int) -> Tuple[X]:

    return np.tile(np.fft.fftfreq(nx)[:, None] * nx * 2*π, (1, ny)), \
           np.tile(np.fft.fftfreq(ny)[None, :] * ny * 2*π, (nx, 1))

def velocity(what: X = None, *, w: X = None) -> X:

    if what is None:
       what = np.fft.fft2(w)
    kx, ky = k(*what.shape)

    Δ = kx ** 2 + ky ** 2
    Δ = Δ.at[0, 0].set(1)

    vx = np.fft.irfft2(what * 1j*ky / Δ, what.shape)
    vy = np.fft.irfft2(what *-1j*kx / Δ, what.shape)

    return vx, vy

class Initial(Gaussian):

    grid = Fourier[2].grid(64, 64)

    def __str__(self): return f"{self.length}x{self.scaling}"

    def __init__(self, length: float, scaling: float = 1.0):
        super().__init__(Initial.grid, Gaussian.Per(length))

        self.length = length
        self.scaling = scaling

    def sample(self, prng, shape=()) -> X:

        x = super().sample(prng, shape)
        x-= np.mean(x, (-2, -1), keepdims=True)

        x = self.scaling * x[..., np.newaxis, :, :]
        return np.broadcast_to(x, shape + (2, *self.dim))

class NavierStokes(PDE):

    """
        wt + v ∇w = nu ∆w
        where
            ∇⨉v = w
            ∇·v = 0
    """

    T: int      # end time
    nu: float   # viscosity
    l: float    # length scale

    # forcing term?
    fn: Optional[Fx]

    def __str__(self): return f"Re={int(self.Re)}:T={self.T}:{self.F}"
    def __init__(self, ic: Initial, T: float, nu: float, fn: Fx = None):

        self.odim = 1
        self.ic = ic

        self.T = T
        self.nu = nu
        self.fn = fn

        self.l = (l:=ic.length)
        self.Re = l / nu * ic.scaling

        if fn is None: self.F = None
        else: self.F = fn.__name__

        self.domain = Rect(3)

        self.basis = series(Chebyshev, Fourier, Fourier)
        self.params = Interpolate(ic, self.basis)

        from ..mollifier import initial_condition
        self.mollifier = initial_condition

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        with jax.default_device(jax.devices("cpu")[0]):

            # solution data are typically large
            # transfer to RAM in the first place

            w = np.load(f"{dir}/w.{self.ic}.npy")
            u = np.load(f"{dir}/u.{self}.npy")

        return jax.vmap(self.basis)(w), u.shape[1:-1], u

    def equation(self, x: X, w0: X, w: X) -> X:
        w, w1, w2 = utils.fdm(w, n=2)

        wt = w1[..., 0, 0]
        wx = w1[..., 0, 1]
        wy = w1[..., 0, 2]
        Δw = Δ(w2[..., 0, 1:, 1:])

        vx, vy = jax.vmap(velocity)(w=w.squeeze(-1))
        Dwdt = wt / self.T + (vx * wx + vy * wy)

        if self.fn is None: f = np.zeros_like(Dwdt)
        else: f = self.fn(*w[0].squeeze(-1).shape)

        return Dwdt - self.nu * Δw - f

    def boundary(self, w: List[Tuple[X]]) -> List[X]:

        _, (wt, wb), (wl, wr) = w
        return [wt - wb, wl - wr]

    def spectral(self, w0: Basis, w: Basis) -> Basis:
        w1 = w.grad(); w2 = w1.grad()

        wt = self.basis(w1.coef[..., 0, 0])
        wx = self.basis(w1.coef[..., 0, 1])
        wy = self.basis(w1.coef[..., 0, 2])
        Δw = self.basis(Δ(w2.coef[..., 0, 1:, 1:]))

        vx, vy = jax.vmap(velocity)(w=w.inv().squeeze(-1))
        Dwdt = self.basis.add(wt.map(lambda coef: coef / self.T),
               self.basis.transform(vx * wx.inv() + vy * wy.inv()))

        if self.fn is None: f = self.basis(np.zeros_like(Dwdt.coef))
        else: f = self.basis.transform(np.broadcast_to(self.fn(*w.mode[1:]), w.mode))

        return self.basis.add(Dwdt, self.basis(-self.nu * Δw.coef), f.map(np.negative))

ic = Initial(0.8)

# ------------------------------- UNFORCED FLOW ------------------------------ #

re2 = NavierStokes(ic, T=3, nu=1e-2)
re3 = NavierStokes(ic, T=3, nu=1e-3)
re4 = NavierStokes(ic, T=3, nu=1e-4)

# ------------------------------ TRANSIENT FLOW ------------------------------ #

def transient(nx: int, ny: int) -> X:

    xy = utils.grid(nx, ny, mode="left").sum(-1)
    return 0.1*(np.sin(2*π*xy) + np.cos(2*π*xy))

tf = NavierStokes(ic, T=50, nu=2e-3, fn=transient)
