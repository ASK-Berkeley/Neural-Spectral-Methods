from . import *
from ..dists import *

class G(ABC):

    """
        Function space
    """

    idim: int
    odim: int

    @abstractmethod
    def sample(self, prng) -> Fx: pass

# ---------------------------------------------------------------------------- #
#                                  INTERPOLATE                                 #
# ---------------------------------------------------------------------------- #

from ..basis import *
class Interpolate(G):

    """
        Interpolated function
    """

    def __init__(self, dist: Ω, basis: Type[Basis]):

        self.dist = dist
        self.basis = basis

        self.idim = len(dist.dim)
        self.odim = 1

    def sample(self, prng, shape=()) -> Basis:

        x = self.dist.sample(prng, shape)[..., None]
        return utils.nmap(self.basis.transform, len(shape))(x)
