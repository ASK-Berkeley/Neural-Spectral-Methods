from src.utils import *
from src.pde.reaction import *
from src.pde.reaction.generate import *

from tqdm import tqdm

def time(N: int = 12) -> X:

    h, s, u = nu005.solution
    h = h.map(lambda x: x[0])

    def check(n: int) -> float:

        return timeit(lambda:
            solution(lambda x: h(np.array([0, x]))[0],
                     1.0, 1.0, nx=2**n, nt=2**n+1))()

    return list(map(check, tqdm(range(1, N))))

def error(rd: ReactionDiffusion, N: int = 12) -> X:

    h, s, u = rd.solution

    def solve(f: Basis, n: int) -> X:

        h = lambda x: f(np.array([0, x])).squeeze()
        return solution(h, rd.nu, rd.rho, n, n+1)[1]

    U = jax.lax.map(F.partial(solve, n=(K:=2**N)), h)

    def call(h: Fx, u: X, n: int) -> X:

        un = solve(h, k:=2**n)
        uN = u[::K//k, ::K//k]

        return np.linalg.norm(np.ravel(uN - un)) \
             / np.linalg.norm(np.ravel(uN))

    return [jax.vmap(F.partial(call, n=n))(h, U)
                       for n in tqdm(range(1, N))]

# ---------------------------------------------------------------------------- #
#                                    SOLVER                                    #
# ---------------------------------------------------------------------------- #

np.save("log/rd/solver/errr.solver.npy", np.array(error(nu01)))
np.save("log/rd/solver/time.solver.npy", np.array([time() for _ in tqdm(range(16))]))

# ---------------------------------------------------------------------------- #
#                                     MODEL                                    #
# ---------------------------------------------------------------------------- #

pde, model = "don't run me this way"
prng = "run main with `--test` flag"

def solve(f: Basis, n: int) -> X:
    h = lambda x: f(np.array([0, x])).squeeze()
    return solution(h, pde.nu, pde.rho, n, n+1)[1]

U = jax.lax.map(F.partial(solve, n=4096), pde.solution[0])
# np.save("test.U.npy", U);;;;;; U=np.load("test.U.npy")

def acc(n: int, K=4096):
    uhat = jax.lax.map(F.partial(model.apply, model.variables, x=((k:=2**n)+1, k+1), method="u"), pde.solution[0])
    return np.linalg.norm((u:=U[:, ::K//k, ::K//k]) - uhat[:, :, :-1, 0]) / np.linalg.norm(u)

grid = model.cfg["grid"]
accuracy = np.stack([acc(n) for n in range(5, 10)])
np.save(f"log/rd/solver/errr.fnox{grid}.npy", accuracy)

def tim(n: int):
    v = model.init(prng, p:=pde.params.sample(prng), (s:=2**n+1, s), method="forward")
    timer = timeit(lambda: model.apply(v, p, (s:=2**n+1, s), method="forward"))
    return np.array([timer() for _ in range(16)])

time = np.stack([tim(n) for n in range(5, 10)])
np.save(f"log/rd/solver/time.fno.npy", time)

# ------------------------------------ NSM ----------------------------------- #

accuracy = np.stack([acc(n) for n in range(5, 10)])
np.save(f"log/rd/solver/errr.nsm.npy", accuracy)

v = model.init(prng, p:=pde.params.sample(prng), method="forward")
time = timeit(lambda: model.apply(v, p, method="forward"))

np.save("log/rd/solver/time.nsm.npy", np.array([time() for _ in range(16)]))
