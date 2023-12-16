from .. import *

class PDE(ABC):

    from ._domain import R
    from ._params import G

    odim: int                           # output dimension

    domain: R                           # interior domain
    params: G                           # parameter function

    mollifier: Fx                       # transformation

    equation: Fx                        # PDE (equation)
    boundary: Fx                        # PDE (boundary)

    from ..basis import Basis
    basis: Basis                        # basis function
    spectral: Fx                        # PDE (spectral)

    solution: Any
