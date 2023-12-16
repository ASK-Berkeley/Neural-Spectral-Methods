from . import *
from ..dists import *

class R(Ω):

    """
        Euclidean space
    """

    ndim: int
    boundary: List[Ω]

# ---------------------------------------------------------------------------- #
#                                     RECT                                     #
# ---------------------------------------------------------------------------- #

class Rect(Uniform, R):

    """
        N-d unit rectangle
    """

    def __init__(self, ndim: int):
        super().__init__(np.zeros(ndim),
                         np.ones(ndim))

        class Boundary(Uniform):

            def __init__(self, dim: int):
                super().__init__(np.zeros(ndim-1),
                                 np.ones(ndim-1))

                self.dim = dim

            def sample(self, prng, shape=()) -> X:

                x = super().sample(prng, shape)
                return np.insert(x, self.dim, np.zeros(shape), axis=-1), \
                       np.insert(x, self.dim, np.ones(shape), axis=-1)

        self.ndim = ndim
        self.boundary = list(map(Boundary, range(ndim)))
