from .. import *
from .._base import *
from ...basis.fourier import *

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class FNO(Spectral):

    def __repr__(self): return "NSM"

    @nn.compact
    def forward(self, ϕ: Basis) -> Basis:

        if not self.cfg["fourier"]: T = self.pde.basis
        else: T = Fourier[self.pde.domain.ndim]

        u = ϕ.to(*self.cfg["mode"])

        bias = T.transform(u.grid(*u.mode)).coef
        u = u.map(lambda coef: np.concatenate([coef, bias], axis=-1))

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = T.transform(self.activate(u.inv()))

        for _ in range(self.cfg["depth"]):

            conv = SpectralConv(self.cfg["hdim"])(u)
            fc = u.map(nn.Dense(self.cfg["hdim"]))

            u = T.add(conv, fc)
            u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"]))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.cfg["hdim"] * 4))
        u = T.transform(self.activate(u.inv()))

        u = u.map(nn.Dense(self.pde.odim))
        return self.pde.mollifier(ϕ, u)
