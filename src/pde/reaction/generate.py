# Modified from characterizing-pinns-failure-modes. Commit 4390d09c507c117a37e621ab1b785a43f0c32f57
# https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/systems_pbc.py

from . import *

def reaction(u: X, rho: float, dt: float) -> X:

    """
        du/dt = rho*u*(1-u)
    """

    factor_1 = u * np.exp(rho * dt)
    factor_2 = 1 - u

    return factor_1 \
        / (factor_1 + factor_2)

def diffusion(u: X, nu: float, dt: float, IKX2: X) -> X:

    """
        du/dt = nu*d2u/dx2
    """

    factor = np.exp(nu * IKX2 * dt)
    u_hat = np.fft.fft(u) * factor

    return np.fft.ifft(u_hat).real

def solution(h: Fx, nu: float, rho: float, nx=4096, nt=4097) -> X:

    """
        Computes the discrete solution of the reaction-diffusion PDE using pseudo
        spectral operator splitting.
    
    Args:
        h: initial condition
        nu: diffusion coefficient
        rho: reaction coefficient
        nx: number of points in the x grid
        nt: number of points in the t grid

    Returns:
        x: grids
        u: solution
    """

    L = 1
    T = 1
    dx = L/nx
    dt = T/(nt-1)
    x = np.arange(0, L, dx)     # not inclusive of the last point
    t = np.linspace(0, T, nt)       # inclusive of the end time 1
    u = np.zeros((nx, nt))

    IKX_pos = 2j * π * np.arange(0, nx/2+1, 1)
    IKX_neg = 2j * π * np.arange(-nx/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    u = [_u:=jax.vmap(h)(x)]

    for _ in range(nt-1):

        _u = reaction(_u, rho, dt)
        _u = diffusion(_u, nu, dt, IKX2)

        u.append(_u)

    return np.dstack(np.meshgrid(t, x, indexing="ij")), np.stack(u)

# ---------------------------------------------------------------------------- #
#                                   GENERATE                                   #
# ---------------------------------------------------------------------------- #

def generate(pde: ReactionDiffusion, N: int = 128):

    params = pde.params.sample(random.PRNGKey(0), (N, ))
    solve = F.partial(solution, nu=pde.nu, rho=pde.rho)

    u = jax.lax.map(lambda h: solve(lambda x: h(np.array([0, x])).squeeze()), params)[1]
    u = np.pad(u[:, ::8, ::8], [(0, 0), (0, 0), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)

    np.save(f"{dir}/h.npy", params.coef)
    np.save(f"{dir}/u.{pde}.npy", u)

    return u

if __name__ == "__main__":

    for nu in [0.005, 0.01, 0.05, 0.1]:

        generate(ReactionDiffusion(64, 5, nu))
