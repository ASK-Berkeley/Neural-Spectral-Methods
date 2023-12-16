from .. import *
from .. import utils

@struct.dataclass
class Basis(ABC):

    coef: Array

    """
        Basis function on [0, 1]^ndim. The leading `ndim` axes are for:

            - (spectral) coefficients of basis functions
            - (physical) values evaluated on collocation points
    """

    @staticmethod

    @abstractmethod
    def repr(self) -> str: pass

    @staticmethod

    @abstractmethod
    def ndim() -> int: pass

    @property
    def mode(self): return self.coef.shape[:self.ndim()]

    def map(self, f: Fx): return self.__class__(f(self.coef))

    @staticmethod

    @abstractmethod
    def grid(*mode: int) -> X: pass

    @staticmethod

    @abstractmethod
    def ix(*mode: int) -> X: pass

    @staticmethod

    @abstractmethod
    def fn(*mode: int, x: X) -> X: pass
    def __call__(self, x: X) -> X:
        
        assert x.shape[-1] == self.ndim(), f"{x.shape[-1]=} =/= {self.ndim()=}"
        return np.tensordot(self.fn(*self.mode, x=x), self.coef, self.ndim())

    def to(self, *mode: int):

        if self.mode == mode: return self
        ax = self.ix(*map(min, mode, self.mode))

        zero = np.zeros(mode + self.coef.shape[self.ndim():])
        return self.__class__(zero.at[ax].set(self.coef[ax]))

    @classmethod
    def add(cls, *terms): return cls(sum(map(O.attrgetter("coef"), align(*terms, scheme=max))))

    @classmethod
    def mul(cls, *terms): return cls.transform(math.prod(map(cls.inv, align(*terms, scheme=sum))))

# --------------------------------- TRANSFORM -------------------------------- #

    @staticmethod

    @abstractmethod
    def transform(x: X): pass

    @abstractmethod
    def inv(self) -> X: pass

# --------------------------------- OPERATOR --------------------------------- #

    @abstractmethod
    def grad(self, k: int = 1): pass

    @abstractmethod
    def int(self, k: int = 1): pass

def align(*basis: Basis, scheme: Fx = max) -> Tuple[Basis]:

    # asserting uniform properties:

    _ = set(map(lambda cls: cls.repr(), basis))
    _ = set(map(lambda cls: cls.ndim(), basis))
    _ = set(map(lambda self: self.coef.ndim, basis))

    mode = tuple(map(scheme, zip(*map(O.attrgetter("mode"), basis))))
    return tuple(map(lambda self: self.to(*mode), basis))

# ---------------------------------------------------------------------------- #
#                                    SERIES                                    #
# ---------------------------------------------------------------------------- #

class SeriesMeta(ABCMeta, type):

    def __getitem__(cls, n: int):
        return series(*(cls,)*n)

@struct.dataclass
class Series(Basis, metaclass=SeriesMeta):

    """1-dimensional series on interval"""

    @staticmethod
    def ndim() -> int: return 1  # on [0, 1]
    def __len__(self): return len(self.coef)

    @abstractmethod
    def __getitem__(self, s: int) -> X: pass

def series(*types: Type[Series]) -> Type[Basis]:

    """
        Generate new basis using finite product of given series. Each argument
        type corresponds to certain kind of series used for each dimension.
    """

    @struct.dataclass
    class Class(Basis):

        @staticmethod
        def repr() -> str: return "".join(map(O.methodcaller("repr"), types))

        @staticmethod
        def ndim() -> int: return len(types)

        @staticmethod
        def grid(*mode: int) -> X:

            assert len(mode) == len(types)

            axes = mesh(lambda i, cls: cls.grid(mode[i]).squeeze(1))
            return np.stack(axes, axis=-1)

        def ix(self, *mode: int) -> X:

            return np.ix_(*map(lambda self, n: self.ix(n), types, mode))

        def fn(self, *mode: int, x: X) -> X:

            axes = mesh(lambda i, self: self.fn(mode[i], x=x[..., [i]]))
            return np.product(np.stack(axes, axis=-1), axis=-1)

        def __getitem__(self, s: Tuple[int]) -> X:

            return jax.vmap(F.partial(Super.__getitem__, s=s[1:]))(Super(Self(self.coef)[s[0]]))

# --------------------------------- TRANSFORM -------------------------------- #

        @staticmethod
        def transform(x: X):

            return Class(Self.transform(jax.vmap(Super.transform)(x).coef).coef)

        def inv(self) -> X:

            return jax.vmap(Super.inv)(Super(Self(self.coef).inv()))            

# --------------------------------- OPERATOR --------------------------------- #

        def grad(self, k: int = 1):

            coef = jax.vmap(F.partial(Super.grad, k=k))(Super(self.coef)).coef
            return Class(np.concatenate([Self(self.coef).grad(k).coef, coef], axis=-1))

        def int(self, k: int = 1):

            coef = jax.vmap(F.partial(Super.int, k=k))(Super(self.coef)).coef
            return Class(np.concatenate([Self(self.coef).int(k).coef, coef], axis=-1))

    def mesh(call: Fx) -> Tuple[X]:
        def cat(*x: X) -> Tuple[X]:

            n, = set(map(np.ndim, x))

            if n != 1: return jax.vmap(cat)(*x)
            return np.meshgrid(*x, indexing="ij")

        args = zip(*enumerate(types))
        return cat(*map(call, *args))

    try: cls, = types; return cls
    except:

        Self, *other = types
        Super = series(*other)

        return Class
