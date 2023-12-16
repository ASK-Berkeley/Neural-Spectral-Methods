from . import *
from ..basis import *

class SpectralConv(nn.Module):

    odim: int

    # do truncation or not
    mode: Tuple[int] = None

    # initialization modes
    init: Tuple[int] = None

    @nn.compact
    def __call__(self, u: Basis) -> Basis:

        def W(a: X) -> X:

            def init(prng, *shape: Tuple[int]) -> X:
                x = random.normal(prng, shape)

                *mode, idim, odim = shape                

                if self.init is None:
                    
                    scale = 1 / idim / odim

                else:

                    from math import prod
                    rate = prod(self.init) / prod(mode)

                    scale = np.sqrt(rate / idim)

                return x * scale

            W = self.param("W", init, *a.shape, self.odim)

            dims = (N:=u.ndim(), N), (B:=range(N), B)
            return jax.lax.dot_general(a, W, dims)

        mode = self.mode or u.mode
        return u.to(*mode).map(W).to(*u.mode)

class SpatialMixing(nn.Module):

    @nn.compact
    def __call__(self, u: Basis) -> Basis:

        def M(a: X, i: int) -> X:

            def init(prng, *shape: Tuple[int]) -> X:
                x = random.normal(prng, shape)

                return x / np.sqrt(shape[-1])

            M = self.param(f"M{i}", init, *a.shape, a.shape[i])

            batch = [*range(i), *range(i + 1, u.ndim()), u.ndim()]
            a = jax.lax.dot_general(a, M, ((i, i), (batch, batch)))

            return np.moveaxis(a, -1, i)

        return u.map(F.partial(F.reduce, M, range(u.ndim())))
