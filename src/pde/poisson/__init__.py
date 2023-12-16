from .. import *
from ...dists import *
from ...basis import *

from .._domain import *
from .._params import *

class Poisson(PDE):

    """
        -Δ u(x) = s(x)
    """

    def __init__(self):

        self.odim = 1

        self.domain = Rect(2)

    def spectral(self, s: Basis, u: Basis) -> Basis:

        return self.basis.add(s, u.grad().grad().map(Δ))

# ---------------------------------------------------------------------------- #
#                                   PERIODIC                                   #
# ---------------------------------------------------------------------------- #

from ...basis.fourier import *

class Periodic(Poisson):

    def __init__(self, res: int):
        super().__init__()

        class Source(Gaussian):

            def sample(self, prng, shape=()) -> X:

                x = super().sample(prng, shape)
                μ = np.mean(x, (-2, -1), keepdims=True)

                return x - μ

        source = Source(Fourier[2].grid(res, res), Gaussian.Per(0.2))
        self.params = Interpolate(source, Fourier[2])
        self.basis = Fourier[2]

        from ..mollifier import periodic
        self.mollifier = periodic

    def equation(self, x: X, s: X, u: X) -> X:

        s = s[:-1, :-1]     # periodic
        u = u[:-1, :-1]     # boundary

        # 5-point stencil for discrete laplacian

        Δ = np.roll(u, 1, 0) + np.roll(u, -1, 0) \
          + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u

        return s + Δ * len(u) ** 2

    def boundary(self, u: List[Tuple[X]]) -> List[X]:

        return [ul - ur for ul, ur in u]

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        s = np.load(f"{dir}/s.periodic.npy")
        u = np.load(f"{dir}/u.periodic.npy")

        return jax.vmap(self.basis)(s), u.shape[1:-1], u
    
# --------------------------------- INSTANCE --------------------------------- #

periodic = Periodic(16)

# ---------------------------------------------------------------------------- #
#                                   DIRICHLET                                  #
# ---------------------------------------------------------------------------- #

from ...basis.chebyshev import *

class Dirichlet(Poisson):

    def __init__(self, res: int):
        super().__init__()

        source = Gaussian(Chebyshev[2].grid(res, res), Gaussian.RBF(0.2))
        self.params = Interpolate(source, Chebyshev[2])
        self.basis = Chebyshev[2]
        
        from ..mollifier import dirichlet
        self.mollifier = dirichlet

    def equation(self, x: X, s: X, u: X) -> X:

        return s + Δ(utils.fdm(u, 2)[2])

    def boundary(self, u: List[Tuple[X]]) -> List[X]:

        return []

    @F.cached_property
    def solution(self):

        dir = os.path.dirname(__file__)

        s = np.load(f"{dir}/s.dirichlet.npy")
        u = np.load(f"{dir}/u.dirichlet.npy")

        return jax.vmap(self.basis)(s), u.shape[1:-1], u

# --------------------------------- INSTANCE --------------------------------- #

dirichlet = Dirichlet(16)
