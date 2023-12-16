from . import *

def grid(*s: int, mode: str = None, flatten: bool = False) -> X:

    """
        Return grid on [0, 1)^n. If not flatten, shape=(*s, len(s));
        else shape=(∏s, len(s)).

        Mode:
            - `None`: uniformly spaced
            - "left": exclude endpoint
            - "cell": centers of rects
    """

    axes = F.partial(np.linspace, 0, 1, endpoint=mode is None)
    grid = np.stack(np.meshgrid(*map(axes, s), indexing="ij"), -1)

    if mode == "cell": grid += .5 / np.array(s)
    if flatten: return grid.reshape(-1, len(s))

    return grid

def nmap(f: Fx, n: int = 1, **kwargs) -> Fx:

    """
        Nested vmap. Keeps the same semantics as `jax.vmap` except that arbitrary
        `n` leading dimensions are vectorized. Returns the vmapped function.
    """

    if not n: return f

    if n > 1: f = nmap(f, n - 1)
    return jax.vmap(f, **kwargs)

def cmap(f: Fx, n: int = None, **kwargs) -> Fx:

    """
        Chunck vmap. Keeps the same semantics as `jax.vmap` but only vectorizing
        over `n` items, and uses loop-based map over the chunks of that size.
    """

    f = jax.vmap(f, **kwargs)
    def call(*args, **kwargs):

        return jax.tree_map(np.concatenate, jax.lax.map(f, *jax.tree_map(into:=lambda x:
                   x.reshape(-1, n, *x.shape[1:]), args), **jax.tree_map(into, kwargs)))

    if n is None: return f

    return call

def jit(f: Fx, **options) -> Fx:

    """
        JIT function with cost analysis on the first run. Keep in mind that loops
        are not taken in to account correctly (which means with cfg.vmap set, the
        results are not reliable).
    """

    f = jax.jit(f, **options)
    def call(*args, **kwargs) -> Any:

        nonlocal f

        if not isinstance(f, jax.stages.Compiled):

            print("=" * 116)

            print(f"compling function {f} ......")
            f = f.lower(*args, **kwargs).compile()

            cost, = f.cost_analysis()
            print("flop:", cost["flops"])
            print("memory:", cost["bytes accessed"])

            print("=" * 116)

        return f(*args, **kwargs)

    return call

def timeit(f: Fx, **options) -> Fx:
    fjit = jit(f, **options)

    def call(*args, **kwargs) -> float:
        fjit(*args, **kwargs) # compile

        import time
        iter = time.time()

        jax.tree_map(jax.block_until_ready,
                     fjit(*args, **kwargs))

        return time.time() - iter

    return call

# ---------------------------------------------------------------------------- #
#                                  DERIVATIVE                                  #
# ---------------------------------------------------------------------------- #

def grad(f: Fx, n: int, D: Fx = jax.jacfwd) -> Fx:

    """
        Calculate up to `n`th order derivatives. Return List[Array] of length `n`
        where `k`th element is the `k`th derivative of shape `(..., d^k)`.

        Differential scheme `D :: (X -> X) -> (X -> X)` determines how to obtain
        the Jacobian. Default to JAX's forward mode autograd.
    """

    u = [f]

    for _ in range(n): u.append(f:=D(f))
    return lambda x: [fk(x) for fk in u]

def fdm(x: X, n: int) -> List[X]:

    """
        Approximate the above derivative using finite difference. `x` is assumed
        to be evaluated on uniform grids on [0, 1]^d (include end-points), where
        dimension is taken as `x.ndim - 1`, i.e. `x` has a trailing channel dim.
    """

    u = [x]
    d = x.ndim - 1
    s = x.shape[:-1]

    for _ in range(n):

        grad = map(lambda i: np.gradient(x, axis=i), range(d))
        u.append(x:=np.stack(tuple(grad), axis=-1) * (np.array(s)-1))

    return u

# ---------------------------------------------------------------------------- #
#                                     DEBUG                                    #
# ---------------------------------------------------------------------------- #

def repl(local):

# ---------------------------------- IMPORT ---------------------------------- #

    import matplotlib.pyplot as plt
    import matplotlib.colors as clr

    import scienceplots
    plt.style.use(["science",
                   "no-latex"])

    from src.basis import Basis, series
    from src.basis.fourier import Fourier
    from src.basis.chebyshev import Chebyshev

# ---------------------------------- HELPER ---------------------------------- #

    def save(fig=plt): fig.savefig("test.jpg", dpi=300); fig.clf()
    def show(img, **kw): plt.colorbar(plt.imshow(img, **kw))
    def gif(*imgs, fps: int = 50, **kw):
        fig, ax = plt.subplots()

        vmin = kw.pop("vmin", min(map(np.min, imgs)))
        vmax = kw.pop("vmax", max(map(np.max, imgs)))
        im = ax.imshow(imgs[0], vmin=vmin, vmax=vmax, **kw)
        id = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha="right", va="bottom")

        plt.colorbar(im)
        def frame(index):

            i = len(imgs) * index//fps
            id.set_text(f"#{i:03}")
            im.set_array(imgs[i])
            return im, id

        from matplotlib import animation
        ani = animation.FuncAnimation(fig, frame, fps, blit=True)
        ani.save("test.gif", writer="pillow", fps=fps, dpi=300)

    import code; code.interact(local=dict(globals(), **dict(locals(), **local)))
