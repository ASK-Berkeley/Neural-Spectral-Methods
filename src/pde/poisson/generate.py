from . import *

def solution(s: Basis, res: int = 256) -> X:

    freq = map(lambda n: np.square(np.fft.fftfreq(n) * 2 * π * n), s.mode)
    u = s.map(lambda coef: coef / sum(np.meshgrid(*freq, indexing="ij"))[..., np.newaxis])

    u = u.map(lambda coef: coef.at[(0, ) * u.ndim()].set(0))
    u = u.map(lambda coef: coef.at[(0, ) * u.ndim()].add(-u(np.zeros(u.ndim()))))

    return u[res, res]

def generate(pde: Periodic, N: int = 128, X: int = 256):

    params = pde.params.sample(random.PRNGKey(0), (N, ))
    u = jax.vmap(F.partial(solution, res=X))(params)

    dir = os.path.dirname(__file__)

    np.save(f"{dir}/s.periodic.npy", params.coef)
    np.save(f"{dir}/u.periodic.npy", u)

    return u

if __name__ == "__main__":

    generate(periodic)
