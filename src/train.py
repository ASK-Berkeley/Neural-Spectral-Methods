from . import *
from .pde import *
from .model import *
from .basis import *

# ---------------------------------------------------------------------------- #
#                                     STEP                                     #
# ---------------------------------------------------------------------------- #

def step(self: Trainer, variable: ϴ) -> Tuple[ϴ, Dict[str, X]]:

    params, prng = variable.get("params"), self.make_rng("sample")
    ϕ = self.pde.params.sample(prng, (self.cfg["bs"], ))

    @F.partial(jax.grad, has_aux=True)
    def loss(params: ϴ, ϕ: X) -> Tuple[X, Dict[str, X]]:

        loss = self.mod.apply(variable.copy(dict(params=params)), ϕ, method="loss")
        return jax.tree_util.tree_reduce(O.add, loss), loss

    grads, loss = jax.tree_map(F.partial(np.mean, axis=0),
                    utils.cmap(F.partial(loss, params), self.cfg["vmap"])(ϕ))

    if self.cfg["clip"] is not None:

        loss["norm"] = np.sqrt(sum(np.sum(np.square(grad))
              for grad in jax.tree_util.tree_leaves(grads)))

        grads = jax.tree_map(F.partial(jax.lax.cond, loss["norm"] < self.cfg["clip"],
              lambda grad: grad, lambda grad: grad / loss["norm"] * self.cfg["clip"]), grads)

    updates, state = self.optimizer.update(grads, self.get_variable("optim", "state"), params)

    self.put_variable("optim", "state", state)
    return variable.copy(dict(params=optax.apply_updates(params, updates))), loss

# ---------------------------------------------------------------------------- #
#                                     EVAL                                     #
# ---------------------------------------------------------------------------- #

def eval(self: Trainer, variable: ϴ) -> Tuple[Dict, X]:

    if isinstance(self.pde.solution, Tuple):
        ϕ, s, u = self.pde.solution

    v = utils.cmap(F.partial(self.mod.apply, variable, x=s, method="u"), self.cfg["vmap"])(ϕ)
    with jax.default_device(cpu:=jax.devices("cpu")[0]):

        u, v = jax.device_put((u, v), device=cpu)
        return jax.tree_map(np.mean, jax.vmap(metric(self.pde, s))(ϕ, u, v)), (u, v)

def metric(pde: PDE, s: Tuple[int]) -> Fx:

    def call(ϕ: Basis, u: X, v: X) -> Dict[str, X]:

        return dict(
            erra=np.mean(np.abs(np.ravel(u - v))),
            errr=np.linalg.norm(np.ravel(u - v)) / np.linalg.norm(np.ravel(u)),
            residual=np.mean(np.abs(pde.equation(utils.grid(*s), ϕ[s], v))),
        )

    return call
