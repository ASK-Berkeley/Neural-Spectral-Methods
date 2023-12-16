from .. import *
from .._base import *

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class SNO(Spectral):

    def __repr__(self): return "SNO"

    @nn.compact
    def forward(self, ϕ: Basis) -> Basis:

        u = ϕ.to(*self.cfg["mode"])

        bias = u.transform(u.grid(*u.mode)).coef
        u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = u.map(self.activate)

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = u.map(self.activate)

        for _ in range(self.cfg["depth"]):

            def Integral(coef: X) -> X:

                K = nn.DenseGeneral(u.mode, axis=range(-u.ndim(), 0))
                return np.moveaxis(K(np.moveaxis(coef, -1, 0)), 0, -1)

            u = u.map(Integral)

            u = u.map(nn.Dense(self.cfg["hdim"]))
            u = u.map(self.activate)

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = u.map(self.activate)

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = u.map(self.activate)

        u = u.map(nn.Dense(self.pde.odim))
        return self.pde.mollifier(ϕ, u)
