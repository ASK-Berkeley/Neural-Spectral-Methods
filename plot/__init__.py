import numpy as np

import scienceplots
import functools as F

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.ticker as tkr

plt.style.use(style=["science", "nature", "std-colors", "grid"])
CLR = plt.rcParams["axes.prop_cycle"].by_key()["color"]
CLR = [CLR[0], CLR[2], CLR[3], CLR[1]]

TIMES = "\\!\\times\\!"
PLUS = "\\!+\\!"

def load(pde: str, metric: str, dir: str, METHOD):

    def random(method: str):
        def call(seed: int):
            try: return np.load(f"{dir}/{pde}{method}.{seed}/{metric}.npy")
            except: return np.array([np.nan])
        return call

    return { method: list(map(random(method), range(4))) for method in METHOD }

def color(method: str) -> str:

    if method[0] == ":":

        if method[-1] == "M":

            return CLR[0]
        
        return CLR[-1]

    if method[0] == "x":

        if method[-1] == "C":

            return CLR[2]

        return CLR[1]

def lines(method: str) -> str:

    if method[0] == ":":

        return "-"

    if method[0] == "x":

        if method[-1] == "C":

            return "-"

        return {
            "64": ":",
            "128": "--",
            "256": "-",

            "96": "-.",
        }[method[1:]]

def reorder(ax, order):

    handles, labels = ax.get_legend_handles_labels()
    return [handles[i] for i in order], [labels[i] for i in order]
