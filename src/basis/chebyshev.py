from . import *

@struct.dataclass
class Chebyshev(Series):

    """
        Chebyshev polynomial of T kind
            - Tn(x) = cos(n cos^-1(x))
            - Tn^*(x) = Tn(2 x - 1)
    """

    @staticmethod
    def repr() -> str: return "C"

    @staticmethod
    def grid(n: int) -> X: return np.cos(π * utils.grid(n))/2+0.5

    @staticmethod
    def ix(n: int) -> X: return np.arange(n)

    @staticmethod
    def fn(n: int, x: X) -> X: return np.cos(np.arange(n) * np.arccos(x*2-1))

    def __getitem__(self, s: int) -> X:
        if isinstance(s, Tuple): s, = s
        return self(utils.grid(s))

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod
    def transform(x: X):

        coef = np.fft.hfft(x, axis=0, norm="forward")[:len(x)]
        coef = coef.at[1:-1].multiply(2)

        assert len(x) > 1, "sharp bits!"
        return Chebyshev(coef)

    def inv(self) -> X:

        coef = self.coef.at[1:-1].divide(2)
        coef = np.concatenate([coef, coef[::-1][1:-1]])

        return np.fft.ihfft(coef, axis=0, norm="forward").real

# --------------------------------- OPERATOR --------------------------------- #

    def grad(self, k: int = 1):

        coef = np.linalg.matrix_power(np.pad(gradient(len(self)), [(0, 1), (0, 0)]), k)
        return Chebyshev(np.tensordot(coef, self.coef, (1, 0))[..., np.newaxis])

    def int(self, k: int = 1):

        coef = np.linalg.matrix_power(integrate(len(self))[:-1], k)
        return Chebyshev(np.tensordot(coef, self.coef, (1, 0))[..., np.newaxis])

# ---------------------------------------------------------------------------- #
#                                    MATRIX                                    #
# ---------------------------------------------------------------------------- #

"""
    Chebyshev gradient and integrate matrix

        - gradient ∈ R ^ n-1⨉n; integrate ∈ R ^ n+1⨉n
        - When aligned, they are pseudo-inverse of each other:
            `gradient(n+1) @ integrate(n) == identity(n)`
"""

def gradient(n: int) -> X:

    alternate = np.pad(np.eye(2), [(0, n-3), (0, n-3)], mode="reflect").at[0].divide(2)
    coef = np.concatenate([np.zeros(n - 1)[:, np.newaxis], np.triu(alternate)], axis=1)
    
    return coef * 4 * np.arange(n)

def integrate(n: int) -> X:
    
    shift = np.identity(n).at[0, 0].set(2) - np.eye(n, k=2)
    coef = np.concatenate([np.zeros(n)[np.newaxis], shift])
    
    return coef.at[1:].divide(4 * np.arange(1, n+1)[:, np.newaxis])
