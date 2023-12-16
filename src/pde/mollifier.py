from . import *
from ..basis import *

SCALE = 1e-3

# ---------------------------------------------------------------------------- #
#                                   PERIODIC                                   #
# ---------------------------------------------------------------------------- #

def periodic(ϕ: X, u: X) -> X:

    if isinstance(ϕ, Basis):
        base = (0, ) * u.ndim()
        origin = np.array(base)

        return u.map(lambda coef: coef.at[base].add(-u(origin)) * SCALE)
    
    if isinstance(ϕ, Array):
        x, uofx = u

        return (uofx - uofx[(0,)*(uofx.ndim-1)]) * SCALE

# ---------------------------------------------------------------------------- #
#                                   DIRICHLET                                  #
# ---------------------------------------------------------------------------- #

def dirichlet(ϕ: X, u: X) -> X:

    if isinstance(ϕ, Basis):
        x = u.grid(*u.mode)

        mol = np.prod(np.sin(π*x), axis=-1, keepdims=True)
        return u.transform(u.inv() * mol * SCALE)

    if isinstance(ϕ, Array):
        x, uofx = u

        mol = np.prod(np.sin(π*x), axis=-1, keepdims=True)
        return uofx * mol * SCALE

# ---------------------------------------------------------------------------- #
#                               INITIAL-CONDITION                              #
# ---------------------------------------------------------------------------- #

def initial_condition(ϕ: X, u: X) -> X:

    """
        Initial condition problem. The first dimension is temporal and the rest
        of them have periodic boundaries.
    """

    if isinstance(ϕ, Basis):

        mol = u.grid(*u.mode)[..., [0]] * SCALE
        return u.__class__.add(u.transform(u.inv() * mol), ϕ)

    if isinstance(ϕ, Array):
        x, uofx = u

        mol = x[..., [0]] * SCALE
        return uofx * mol + ϕ
