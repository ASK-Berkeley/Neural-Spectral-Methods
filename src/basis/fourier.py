from . import *

@struct.dataclass
class Fourier(Series):

    """
        Trigonometric series

            fk(x) = e^{ik·x}
    """

    @staticmethod
    def repr() -> str: return "F"

    @staticmethod
    def grid(n: int) -> X: return utils.grid(n, mode="left")

    @staticmethod
    def ix(n: int) -> X: return np.r_[-n//2+1:n//2+1]

    @staticmethod
    def fn(n: int, x: X) -> X:

        return np.moveaxis(real(np.moveaxis(np.exp(x * -freq(n)), -1, 0), n), 0, -1)

    def __getitem__(self, s: int) -> X:
        if isinstance(s, Tuple): s, = s

        return np.concatenate([x:=self.to(s - 1).inv(), x[np.r_[0]]])

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod
    def transform(x: X):

        coef = np.fft.rfft(x, axis=0, norm="forward")
        coef = coef.at[1:-(len(x)//-2)].multiply(2)
 
        return Fourier(real(coef, len(x)))

    def inv(self) -> X:

        coef = comp(self.coef, n:=len(self))
        coef = coef.at[1:-(n//-2)].divide(2)

        return np.fft.irfft(coef, len(self), axis=0, norm="forward")

# --------------------------------- OPERATOR --------------------------------- #

    def grad(self, k: int = 1):

        coef = np.expand_dims(freq(len(self))**k, range(1, self.coef.ndim))
        return Fourier(real(comp(self.coef, n:=len(self)) * coef, n)[..., None])

    def int(self, k: int = 1):

        coef = np.expand_dims(self.freq(len(self)), range(1, self.coef.ndim))
        return Fourier((self.coef / coef ** k)[..., np.newaxis].at[0].set(0))

# ---------------------------------------------------------------------------- #
#                                    HELPER                                    #
# ---------------------------------------------------------------------------- #

def freq(n: int) -> X: return np.arange(n//2+1) * 2j * π

def real(coef: X, n: int) -> X:

    """Complex coef -> Real coef"""

    cos, sin = coef.real, coef.imag[1:-(n//-2)]
    return np.concatenate((cos, sin[::-1]), 0)

def comp(coef: X, n: int) -> X:

    """Real coef -> Complex coef"""

    cos, sin = np.split(coef, (m:=n//2+1, ))
    return (cos+0j).at[n-m:0:-1].add(sin*1j)
