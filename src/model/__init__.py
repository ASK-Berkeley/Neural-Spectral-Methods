from .. import *
from ..pde import *
from ..basis import *

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

class Solver(ABC, nn.Module):

    pde: PDE
    cfg: Dict

    @F.cached_property
    def activate(self) -> Fx: return \
        getattr(nn, self.cfg["activate"])

    @abstractmethod
    def u(self, ϕ: Fx, x: X) -> X: pass

    @abstractmethod
    def loss(self, ϕ: Fx) -> Dict[str, X]: pass

# ----------------------------------- PINN ----------------------------------- #

class PINN(Solver, ABC):

    @abstractmethod
    def forward(self, ϕ: Basis, s: Tuple[int]) -> Tuple[X, X]: pass

    def u(self, ϕ: Basis, x: Tuple[int]) -> X:

        assert isinstance(x, Tuple), "uniform"
        assert len(x) == self.pde.domain.ndim

        return self.forward(ϕ, x)[-1]

    def loss(self, ϕ: Basis) -> Dict[str, X]:

        x, y, u = self.forward(ϕ, (self.cfg["grid"], ) * (d:=self.pde.domain.ndim))
        edges = [(np.take(u, 0, axis=n), np.take(u, -1, axis=n)) for n in range(d)]

        R = self.pde.equation(x, y, u)
        B = self.pde.boundary(edges)

        return dict(
            residual=np.mean(R**2), **{
          f"boundary{n}": np.mean(Bn**2)
                    for n, Bn in enumerate(B)
        })

# --------------------------------- SPECTRAL --------------------------------- #

class Spectral(Solver, ABC):

    @abstractmethod
    def forward(self, ϕ: Basis) -> Basis: pass

    def u(self, ϕ: Basis, x: X) -> X:

        u = self.forward(ϕ)

        if isinstance(x, Tuple): return u[x]
        if isinstance(x, Array): return u(x)

    def loss(self, ϕ: Basis) -> Dict[str, X]:

        R = self.pde.spectral(ϕ, self.forward(ϕ))
        return dict(residual=np.sum(np.square(R.coef)))

# ---------------------------------------------------------------------------- #
#                                    TRAINER                                   #
# ---------------------------------------------------------------------------- #

class Trainer(ABC, nn.Module):

    mod: Solver
    pde: PDE
    cfg: Dict

    def setup(self):

# --------------------------------- SCHEDULER -------------------------------- #

        if self.cfg["schd"] is None:
            
            scheduler = self.cfg["lr"]

        if self.cfg["schd"] == "cos":

            scheduler = optax.cosine_decay_schedule(self.cfg["lr"], self.cfg["iter"])

        if self.cfg["schd"] == "exp":

            decay_rate = 1e-3 ** (1.0 / self.cfg["iter"])
            scheduler = optax.exponential_decay(self.cfg["lr"], 1, decay_rate)

# --------------------------------- OPTIMIZER -------------------------------- #

        self.optimizer = optax.adam(scheduler)

    @nn.compact
    def init(self):

        ϕ = self.pde.params.sample(prng:=self.make_rng("sample"))
        s = tuple([self.cfg["grid"]] * self.pde.domain.ndim)

        variable = self.mod.init(prng, ϕ, s, method="u")
        print(self.mod.tabulate(prng, ϕ, s, method="u"))

        self.variable("optim", "state", self.optimizer.init, variable["params"])

        return variable
