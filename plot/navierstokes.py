from . import *

METHOD = ["x64", "x96", ":NSM"]
LABELS = [f"FNO${TIMES}64^3$+PINN", f"FNO${TIMES}96^3$+PINN", "NSM (ours)"]

res = ["re2", "re3", "re4"]

def load(pde: str, metric: str, dir: str = "log/gcp/ns.T3"):

    def random(method: str):
        def call(seed: int):
            try:
                if method == "x64": return np.load(f"{dir}/{pde}{method}.{seed}/{metric}.npy")[:int(3600*24 // 0.69)]
                if method == "x96": return np.load(f"{dir}/{pde}{method}.{seed}/{metric}.npy")[:int(3600*72 // 2.33)]
                return np.load(f"{dir}/{pde}{method}.{seed}/{metric}.npy")
            except: print("oh no not finished!"); return np.array([np.inf])
        return call

    return { method: list(map(random(method), range(4))) for method in METHOD }

# ---------------------------------------------------------------------------- #
#                                     TABLE                                    #
# ---------------------------------------------------------------------------- #

for re in res:

    print(f"{re = }", end=":\t")

    for method, errs in load(re, "metric.errr").items():

        errs = np.array([err[-1] for err in errs]) * 100
        print(f"{np.mean(errs):.2f}±{np.std(errs):.2f}", end="\t")

    print("")

# ---------------------------------------------------------------------------- #
#                                   VISUALIZE                                  #
# ---------------------------------------------------------------------------- #

import sys
if len(sys.argv) > 1:

    N = int(sys.argv[1])

    fig, [ok, nsm, fno] = plt.subplots(ncols=(T:=3)+1, nrows=3, figsize=(T*2, 5))

    u = np.load("log/gcp/ns.T3/u.npy")[N]
    unsm = np.load(f"log/{sys.argv[2]}/uhat.npy")[N]
    ufno = np.load(f"log/{sys.argv[3]}/uhat.npy")[N]

    # remove dark colors

    L, K = 15, 4
    vmin = u.min() * (L+2*K)/L
    vmax = u.max() * (L+2*K)/L

    def show(ax, img):

        cmap = plt.get_cmap("twilight_shifted")
        norm = clr.BoundaryNorm(np.linspace(vmin, vmax, L+2*K+1), cmap.N)
        ax.imshow(img, cmap=cmap, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(T+1):

        t = int(63 * i/T)

        show(ok[i], u[t])
        show(nsm[i], unsm[t])
        show(fno[i], ufno[t])

    ok[0].set_title("Initial vorticity", fontsize=10)
    for t in range(1, T + 1):
        ok[t].set_title(f"$T={t}$", fontsize=10)

    ok[0].set_ylabel("Numerical solver", rotation=90, labelpad=10, fontsize=10)
    nsm[0].set_ylabel("NSM (ours) prediction", rotation=90, labelpad=10, fontsize=10)
    fno[0].set_ylabel("FNO+PINN prediction", rotation=90, labelpad=10, fontsize=10)

    ok[0].text(-0.21, 1.03, "(c)", transform=ok[0].transAxes, fontsize=10)

    fig.tight_layout()
    fig.savefig(f"plot/navierstokes.{N}.jpg", dpi=300)

# ---------------------------------------------------------------------------- #
#                                     CURVE                                    #
# ---------------------------------------------------------------------------- #

for re in ["re4"]:

    figure = plt.figure(figsize=(3, 4.5))
    res, err = figure.subplots(nrows=2, sharex=True)

    pct = tkr.FuncFormatter(lambda y, _: f"{y:.1%}"[:3])

    rate = load(re, "metric.rate")

# --------------------------------- RESIDUAL --------------------------------- #

    residual = load(re, "metric.residual")

    for method, label in zip(METHOD, LABELS):

        xs = np.linspace(1, max(map(np.sum, rate[method])), N:=1000)

        ys = []
        for rat, rrr in zip(rate[method], residual[method]):

            a = np.concatenate([np.zeros(1), np.cumsum(rat)])[:-1]

            ys.append(np.interp(xs, a, rrr[:len(a)]))

        ys_mean = np.mean(np.stack(ys, axis=0), axis=0)
        ys_std = np.std(np.stack(ys, axis=0), axis=0)

        res.plot(xs, ys_mean, label=label, c=(c:=color(method)), ls=lines(method))
        res.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.2, color=c, edgecolor=None)

    res.set_xscale("log")
    res.set_yscale("log")

    res.xaxis.set_label_position("top")
    res.set_xlabel("Training time (seconds)", fontsize=10)
    res.set_ylabel(f"PDE residual on test set", fontsize=10)

    res.yaxis.grid(False, which='minor')
    res.set_yticks([0.1, 0.08, 0.06, 0.04, 0.02])
    res.yaxis.set_major_formatter(lambda x, _: str(x))
    res.yaxis.set_minor_formatter(lambda *_: "")

    err.set_xlim(10, 3e5)
    err.xaxis.tick_top()
    for tick in err.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")

# ----------------------------------- ERROR ---------------------------------- #

    # TRAIN

    errr = load(re, "metric.errr")

    for method, label in zip(METHOD, LABELS):

        xs = np.linspace(1, max(map(np.sum, rate[method])), N:=1000)

        ys = []
        for rat, rrr in zip(rate[method], errr[method]):

            a = np.concatenate([np.zeros(1), np.cumsum(rat)])[:-1]

            ys.append(np.interp(xs, a, rrr[:len(a)]))

        ys_mean = np.mean(np.stack(ys, axis=0), axis=0)
        ys_std = np.std(np.stack(ys, axis=0), axis=0)

        err.plot(xs, ys_mean, label=label, c=(c:=color(method)), ls=lines(method))
        err.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.2, color=c, edgecolor=None)

    err.set_xscale("log")
    err.set_yscale("log")

    err.set_ylabel(f"$L_2$ rel. error (\\%) on test set", fontsize=10)

    err.yaxis.set_major_formatter(pct)
    err.set_yticks([0.05, 0.1, 0.2, 0.3])

    res.legend(fontsize=7, loc="lower left", handlelength=1.7)
    err.legend(fontsize=7, loc="lower left", handlelength=1.7)

    res.xaxis.set_tick_params(labelsize=8)
    res.yaxis.set_tick_params(labelsize=8)
    err.yaxis.set_tick_params(labelsize=8)
    err.xaxis.set_tick_params(labelsize=8)

    res.yaxis.set_label_coords(-0.15, 0.5)
    err.yaxis.set_label_coords(-0.15, 0.5)

    res.text(-0.21, 1, "(a)", transform=res.transAxes, fontsize=10)
    err.text(-0.21, 1, "(b)", transform=err.transAxes, fontsize=10)

    figure.tight_layout(h_pad=0.1)
    figure.savefig(f"plot/ns.curve.{re}.jpg", dpi=300)
