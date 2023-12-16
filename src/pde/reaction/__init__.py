from .. import *
from ...dists import *
from ...basis import *

from .._domain import *
from .._params import *

from ...basis.fourier import *
from ...basis.chebyshev import *

class ReactionDiffusion(PDE):

    """
        ut - \nu uxx = \rho u (1 - u)
    """

    rho: float  # reaction coefficient
    nu: float   # diffusion coefficient

    def __str__(self): return f"rho={self.rho}:nu={self.nu:.3f}"
    def __init__(self, res: int, rho: float, nu: float):

        self.odim = 1

        self.rho = rho
        self.nu = nu

        self.domain = Rect(2)
        self.basis = series(Chebyshev, Fourier)

        class Initial(Gaussian):

            def sample(self, prng, shape=()) -> X:
                x = super().sample(prng, shape)

                inf = np.min(x, axis=-1, keepdims=True)
                sup = np.max(x, axis=-1, keepdims=True)

                x, shape = (x - inf) / (sup - inf), shape + (2, res)
                return np.broadcast_to(x[..., np.newaxis, :], shape)

        initial = Initial(Fourier.grid(res), Gaussian.Per(0.2))
        self.params = Interpolate(initial, self.basis)

        from ..mollifier import initial_condition
        self.mollifier = initial_condition

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        h = np.load(f"{dir}/h.npy")
        u = np.load(f"{dir}/u.{self}.npy")

        return jax.vmap(self.basis)(h), u.shape[1:-1], u

    def equation(self, x: X, h: X, u: X) -> X:
        u, u1, u2 = utils.fdm(u, n=2)

        ut = u1[..., 0]
        uxx = u2[..., 1, 1]

        reaction = -self.rho * u * (1 - u)
        diffusion = -self.nu * uxx

        return ut + reaction + diffusion

    def boundary(self, u: List[Tuple[X]]) -> List[X]:

        _, (ul, ur) = u
        return [ul - ur]

    def spectral(self, h: Basis, u: Basis) -> Basis:

        u1 = u.map(lambda coef: coef.at[(0, 0)].add(-1))

        reaction = u.__class__.mul(u, u1).map(lambda uu1: uu1 * self.rho)
        diffusion = u.grad(2).map(lambda u2: -u2[..., 1] * self.nu)

        ut = u.grad().map(lambda u1: u1[..., 0])
        return u.__class__.add(ut, reaction, diffusion)

# --------------------------------- INSTANCE --------------------------------- #

nu005 = ReactionDiffusion(64, rho=5, nu=0.005)
nu01 = ReactionDiffusion(64, rho=5, nu=0.01)
nu05 = ReactionDiffusion(64, rho=5, nu=0.05)
nu1 = ReactionDiffusion(64, rho=5, nu=0.1)
