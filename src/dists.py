from . import *

class Ω(ABC):

    """
        Distribution
    """

    @abstractmethod
    def sample(self, prng) -> X: pass

# ---------------------------------------------------------------------------- #
#                                    UNIFORM                                   #
# ---------------------------------------------------------------------------- #

class Uniform:

    """
        Uniform distribution
    """

    min: X
    max: X

    def __init__(self, min: X, max: X):

        self.min = np.array(min)
        self.max = np.array(max)

    def sample(self, prng, shape=()) -> X:
        
        scale = self.max - self.min
        x = random.uniform(prng, shape + scale.shape)

        return x * scale + self.min

# ---------------------------------------------------------------------------- #
#                                    NORMAL                                    #
# ---------------------------------------------------------------------------- #

class Normal:

    """
        Normal distribution
    """

    μ: X
    λ: X

    def __init__(self, μ: X, Σ: X):

        self.μ = μ

        U, Λ, _ = np.linalg.svd(Σ)
        self.λ = U * np.sqrt(Λ)

    def sample(self, prng, shape=()) -> X:

        var = random.normal(prng, shape + self.μ.shape)
        ε = np.einsum("...ij,...j->...i", self.λ, var)

        return self.μ + ε

# ---------------------------------------------------------------------------- #
#                                   GAUSSIAN                                   #
# ---------------------------------------------------------------------------- #

class Gaussian(Normal):

    """
        Gaussian Process
    """

    dim: Tuple[int]

    def __init__(self, grid: X, kernel: Fx):

        *dim, ndim = grid.shape
        assert len(dim) == ndim

        X = grid.reshape(-1, ndim)
        K = jax.vmap(kernel, (0, None))
        Σ = jax.vmap(lambda y: K(X, y))(X)

        super().__init__(np.zeros(len(Σ)), Σ)
        self.dim = tuple(dim)

    def sample(self, prng, shape=()) -> X:

        x = super().sample(prng, shape)
        return x.reshape(shape + self.dim)

# ---------------------------------- KERNEL ---------------------------------- #

    RBF = lambda ƛ: lambda x, y: np.exp(-np.sum((x-y)**2) / ƛ**2/2)
    Per = lambda ƛ: lambda x, y: np.exp(-np.sum((np.sin(π*(x-y))/2)**2) / ƛ**2*2)
