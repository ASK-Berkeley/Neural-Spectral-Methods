from .. import *
from .._base import *
from ...basis.fourier import *

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class FNO(PINN):

    def __repr__(self): return f"FNOx{self.cfg['grid']}+PINN"

    @nn.compact
    def forward(self, ϕ: Basis, s: Tuple[int]) -> Tuple[X, X]:

        if self.cfg["cheb"]: T = self.pde.basis
        else: T = Fourier[self.pde.domain.ndim]

        from .. import utils
        x = utils.grid(*s)

        z = np.concatenate([x, y:=ϕ[s]], -1)

        z = nn.Dense(self.cfg["hdim"] * 4)(z)
        z = self.activate(z)

        z = nn.Dense(self.cfg["hdim"])(z)
        z = self.activate(z)

        for _ in range(self.cfg["depth"]):

            conv = SpectralConv(self.cfg["hdim"], self.cfg["mode"])(T.transform(z)).inv()
            fc = nn.Dense(self.cfg["hdim"])(z)

            z = conv + fc
            z = self.activate(z)

        z = nn.Dense(self.cfg["hdim"])(z)
        z = self.activate(z)

        z = nn.Dense(self.cfg["hdim"] * 4)(z)
        z = self.activate(z)

        z = nn.Dense(self.pde.odim)(z)
        return x, y, self.pde.mollifier(y, (x, z))
