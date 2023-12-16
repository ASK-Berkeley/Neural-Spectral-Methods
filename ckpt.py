from src import *
from threading import *

class Checkpoint(Thread):

    def __init__(self, **kwargs):
        super().__init__(daemon=True)

        from queue import Queue
        self.metric = Queue()
        self.predict = Lock()

# ---------------------------------------------------------------------------- #
#                                  INITIALIZE                                  #
# ---------------------------------------------------------------------------- #

        import datetime, os
        time = datetime.datetime.now().strftime("%c")

        self.path = "log/" + (kwargs["note"] or time)
        self.title = kwargs["pde"] + ":" + kwargs["model"]
        if kwargs["spectral"]: self.title += "+spectral"

        os.makedirs(self.path, exist_ok=False)
        print(kwargs, file=open(f"{self.path}/cfg", "w"))

    @property
    def prediction(self):

        with self.predict:
            return self._predict

    @prediction.setter
    def prediction(self, answer):

        with self.predict:
            self._predict = answer

    def run(self):

        from collections import defaultdict
        log = defaultdict(lambda: list())

        while True:

# ---------------------------------------------------------------------------- #
#                                  CHECKPOINT                                  #
# ---------------------------------------------------------------------------- #

            try:
                for _ in range(max(1, self.metric.qsize())):
                    metric, it = self.metric.get()

                    for key, value in metric.items():
                        log[key].append(value)

            except: pass

            import matplotlib.pyplot as plt
            import matplotlib.colors as clr

            import scienceplots; plt.style.use(["science", "no-latex"])

# ---------------------------------------------------------------------------- #
#                                    METRIC                                    #
# ---------------------------------------------------------------------------- #

            for key, values in log.items():

                fig, ax = plt.subplots()
                xs = np.arange(len(values))+1

                if all(isinstance(value, Array) for value in values):

                    ax.plot(xs, ys:=np.array(values), label=key)
                    np.save(f"{self.path}/metric.{key}.npy", ys)

                if all(isinstance(value, Dict) for value in values):

                    for subkey in set.union(*map(set, map(dict.keys, values))):
                        ys = np.array(list(map(O.itemgetter(subkey), values)))

                        ax.plot(xs, ys, label=f"{key}:{subkey}")
                        np.save(f"{self.path}/metric.{key}:{subkey}.npy", ys)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(self.title)
                ax.legend()

                fig.savefig(f"{self.path}/metric.{key}.jpg")
                plt.close(fig)

# ---------------------------------------------------------------------------- #
#                                     IMAGE                                    #
# ---------------------------------------------------------------------------- #

            u, uhat = self.prediction
            np.save(f"{self.path}/uhat.npy", uhat)

            N = max(min(16, len(u)), 2)
            u, uhat = u[:N], uhat[:N]

            fig = plt.figure(figsize=(5 * N, 5 * 3))
            subfig = fig.subfigures(ncols=N)

            def create(subfig, u: X, uhat: X, i: int = None):
                axes = subfig.subplots(nrows=3)

                vmin, vmax = u.min().item(), u.max().item()
                if i is not None: u = u[i]; uhat = uhat[i]

                true = axes[0].imshow(u, vmin=vmin, vmax=vmax)
                pred = axes[1].imshow(uhat, vmin=vmin, vmax=vmax)

                vlim = max(abs(vmin), abs(vmax))
                diff = axes[2].imshow(uhat - u,
                        cmap="Spectral",
                        norm=clr.SymLogNorm(
                            linthresh=vlim / 100,
                            vmin=-vlim / 10,
                            vmax=+vlim / 10,
                        )
                    )

                axes[0].axis("off")
                axes[1].axis("off")
                axes[2].axis("off")

                subfig.colorbar(true, ax=axes[:2])
                subfig.colorbar(diff, ax=axes[2:])

                return true, pred, diff

            if u.ndim-2 == 2:

                _ = list(map(create, subfig, u, uhat))
                fig.savefig(f"{self.path}/image.jpg")

            plt.close(fig)

# ---------------------------------------------------------------------------- #
#                                     EXIT                                     #
# ---------------------------------------------------------------------------- #

            if it is None: break
