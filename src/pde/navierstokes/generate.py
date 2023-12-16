# Modified from neuraloperator. Commit ef3de3bb1140175c69a9fe3a8b45afd1335077d9
# https://github.com/neuraloperator/neuraloperator/blob/master/data_generation/navier_stokes/ns_2d.py

from . import *

def simulate(w0: X, nu: float, f: X) -> Fx:

    """
    Returns:
        u: callable what -> what' for next step
           advance vorticity in spectral domain
    """

    s, s = w0.shape
    kx, ky = k(s, s)

    diffuse = (Δ := kx ** 2 + ky ** 2) * nu

    dealias = (Δ < (2/3 * π * s) ** 2).astype(float)
    dealias = dealias.at[0, 0].set(0)   # zero-mean

    def Δhat(what: X) -> X:

        vx, vy = velocity(what)

        wx = np.fft.irfft2(what * 1j * kx, (s, s))
        wy = np.fft.irfft2(what * 1j * ky, (s, s))

        vxwx = np.fft.fft2(vx * wx, (s, s))
        vywy = np.fft.fft2(vy * wy, (s, s))

        return np.fft.fft2(f) - vxwx - vywy

    def call(what: X, dt: float) -> X:
        
        Δhat1 = Δhat(what)  # Heun's method

        what_tilde = what + dt * (Δhat1 - diffuse * what / 2)
        what_tilde/= 1 + dt * diffuse / 2

        Δhat2 = Δhat(what_tilde)  # Cranck-Nicholson + Heun

        what = what + dt * ((Δhat1 + Δhat2) - diffuse * what) / 2
        what/= 1 + dt * diffuse / 2

        return what * dealias

    return call

def solution(w0: X, T: float, nu: float, force: Fx,
             dt: float, nt: int) -> X:

    """
    Args:
        w0: initial condition

        T: total time
        nu: viscosity
        force: -ing term

        dt: advance step
        nt: record step

    Returns:
        u: solution recorded at each timestep
           inclusive of the end time
           shape = (nt, *w0.shape)
    """

    if not force: f = np.zeros_like(w0)
    else: f = force(*w0.shape)

    step = simulate(w0, nu, f)
    Δt = T / (N := nt - 1)

    def record(what: X, _) -> Tuple[X, X]:
        call = lambda _, what: step(what, dt)

        what = step(jax.lax.fori_loop(0., Δt // dt, call, what), Δt % dt)
        return what, np.fft.irfft2(what, s=w0.shape)

    _, w = jax.lax.scan(record, np.fft.fft2(w0), None, N)
    return np.concatenate([w0[np.newaxis, :], w], axis=0)

# ---------------------------------------------------------------------------- #
#                                   GENERATE                                   #
# ---------------------------------------------------------------------------- #

def generate(pde: NavierStokes, dt: float = 1e-3, T: int = 64, X: int = 256):

    params = pde.params.sample(random.PRNGKey(0), (128, ))
    solve = F.partial(solution, T=pde.T, nu=pde.nu, force=pde.fn, dt=dt, nt=T)

    w = jax.vmap(solve)(jax.vmap(lambda w: w.to(1, X, X).inv().squeeze())(params))
    w = np.pad(w, [(0, 0), (0, 0), (0, 1), (0, 1)], mode="wrap")[..., np.newaxis]

    dir = os.path.dirname(__file__)

    np.save(f"{dir}/w.{pde.ic}.npy", params.coef)
    np.save(f"{dir}/u.{pde}.npy", w)

    return w

if __name__ == "__main__":

    from sys import argv
    
    if argv[1] == "ns":

        generate(re2)
        generate(re3)
        generate(re4)

    if argv[1] == "tf":

        generate(tf, dt=5e-3)
