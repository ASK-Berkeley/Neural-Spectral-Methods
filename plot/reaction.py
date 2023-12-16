from . import *

METHOD = [":SNO", "x64", "x128", "x256", "x256C", ":NSM"]

# FNOPINN = lambda n, m="": "FNO{\\scriptsize$\\!\\times\\!"+str(n)+"^2$"+m+"}+PINN"
FNOPINN = lambda n, m="": "FNO$\\!\\times\\!" + str(n) + "^2$" + m + "+PINN"
LABELS = [None, FNOPINN(64, "\\ \\ "), FNOPINN(128), FNOPINN(256), f"CNO+PINN(ours)", f"NSM (ours)"]

load = F.partial(load, dir="log/rd", METHOD=METHOD)

# ---------------------------------------------------------------------------- #
#                                     TABLE                                    #
# ---------------------------------------------------------------------------- #

for nu in ["005", "01", "05", "1"]:

    print(f"{nu = }", end=":\t")

    for method, errs in load(f"nu{nu}", "metric.errr").items():

        errs = np.array([err[-1] for err in errs]) * 100
        print(f"{np.mean(errs):.3f}±{np.std(errs):.3f}", end="\t")

    print("")


for nu in ["01"]:

    figure = plt.figure(figsize=(8, 4), constrained_layout=False)
    train, test = figure.subfigures(ncols=2, width_ratios=[1, 1.2])

    train.subplots_adjust(hspace=0.31)
    test.subplots_adjust(left=0.08)

    pct = tkr.FuncFormatter(lambda y, _: f"{y:.1%}"[:3])

# ---------------------------------------------------------------------------- #
#                                     TRAIN                                    #
# ---------------------------------------------------------------------------- #

    rate = load(f"nu{nu}", "metric.rate")
    res, err = train.subplots(nrows=2, sharex=True)

# --------------------------------- RESIDUAL --------------------------------- #

    residual = load(f"nu{nu}", "metric.residual")

    for method, label in zip(METHOD, LABELS):
        if label is None: continue

        xs = np.linspace(1, max(map(np.sum, rate[method])), N:=1000)
        ys = [np.interp(xs, np.concatenate([np.zeros(1), np.cumsum(rate)]), residual)
                     for rate, residual in zip(rate[method], residual[method])]

        ys_mean = np.mean(np.stack(ys, axis=0), axis=0)
        ys_std = np.std(np.stack(ys, axis=0), axis=0)

        res.plot(xs, ys_mean, label=label, c=(c:=color(method)), ls=lines(method))
        res.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.2, color=c, edgecolor=None)

    res.set_xscale("log")
    res.set_yscale("log")

    res.set_xlim(1, 1e4)
    res.set_xticks([1, 1e1, 1e2, 1e3, 1e4])

  # res.yaxis.set_major_formatter(tkr.ScalarFormatter())
    res.yaxis.set_label_coords(-0.13, None)

# ---------------------------------- ERROR % --------------------------------- #

    errr = load(f"nu{nu}", "metric.errr")

    for method, label in zip(METHOD, LABELS):
        if label is None: continue

        xs = np.linspace(1, max(map(np.sum, rate[method])), N:=1000)
        ys = [np.interp(xs, np.concatenate([np.zeros(1), np.cumsum(rate)]), errr)
                         for rate, errr in zip(rate[method], errr[method])]

        ys_mean = np.mean(np.stack(ys, axis=0), axis=0)
        ys_std = np.std(np.stack(ys, axis=0), axis=0)

        err.plot(xs, ys_mean, label=label, c=(c:=color(method)), ls=lines(method))
        err.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.2, color=c, edgecolor=None)

    err.set_xscale("log")
    err.set_yscale("log")

    err.set_xlim(1, 1e4)
    err.set_xticks([1, 1e1, 1e2, 1e3, 1e4])

    err.set_ylim(6e-4)

    err.yaxis.set_major_formatter(pct)
    err.yaxis.set_label_coords(-0.13, None)

# ---------------------------------------------------------------------------- #
#                                     TEST                                     #
# ---------------------------------------------------------------------------- #

    ax = test.subplots()

    mkr = dict(marker=".", markersize=7)

# ------------------------------------ FNO ----------------------------------- #

    time = np.load("log/rd/solver/time.fno.npy")

    for method, label in zip(METHOD, LABELS):

        if method[0] == "x" and method[-1] != "C":

            ax.plot(X:=time.mean(-1), Y:=np.load(f"log/rd/solver/errr.fnox{method[1:]}.npy"), **mkr, label=label+" (training)", c=color(method), ls=lines(method))

            def getl(n):

                style = dict(text=f"${2**n*32}{TIMES}{2**n*32}$", xy=(x:=X[n], y:=Y[n]), textcoords="offset pixels", fontsize=9)

                if n < 2: style.update(xytext=(x-20, y+40))
                if n > 1: style.update(xytext=(x-78, y+40))

                return style

            if method == "x256":
                for n in range(2): ax.annotate(**getl(n))
            if method == "x64":
                for n in range(2, 5): ax.annotate(**getl(n))

# ---------------------------------- SOLVER ---------------------------------- #

    time = np.load("log/rd/solver/time.solver.npy")
    errr = np.load(f"log/rd/solver/errr.solver.npy")

    ax.plot(X:=time.mean(-1), Y:=errr.mean(-1), c="black", **mkr, label="Numerical solver")
    for n in range(1, len(errr)+1): ax.annotate(f"${2**n}{TIMES}{2**n}$", (x:=X[n-1], y:=Y[n-1]), (x+16, y), textcoords="offset pixels", fontsize=9)

# ------------------------------------ NSM ----------------------------------- #

    time = np.load("log/rd/solver/time.nsm.npy")

    ax.plot(x:=time.mean(), y:=np.load("log/rd/solver/errr.nsm.npy").mean(), **mkr, c=CLR[0], label="NSM (ours)")
    ax.scatter(x, y, c=CLR[0], s=20)
    ax.annotate(f"$32{TIMES}32$", (x, y), (x-9, y+152), textcoords="offset pixels", fontsize=9)
    ax.annotate(f"$64{TIMES}64$", (x, y), (x-9, y+120), textcoords="offset pixels", fontsize=9)
    ax.annotate(f"$128{TIMES}128$", (x, y), (x-9, y+88), textcoords="offset pixels", fontsize=9)
    ax.annotate(f"$256{TIMES}256$", (x, y), (x-9, y+56), textcoords="offset pixels", fontsize=9)
    ax.annotate(f"$512{TIMES}512$", (x, y), (x-9, y+24), textcoords="offset pixels", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(6e-4, 1e-1)
    ax.set_ylim(6e-4, 1e-1)

    ax.yaxis.set_major_formatter(pct)
    ax.yaxis.set_label_coords(-0.09, None)

# ---------------------------------------------------------------------------- #
#                                    LAYOUT                                    #
# ---------------------------------------------------------------------------- #

    res.legend(*reorder(res, [0, 1, 2, 3, 4]), loc="lower center", handlelength=1.5, bbox_to_anchor=(0.5, -0.315), ncol=3, fontsize=7, labelspacing=0.5, columnspacing=1)
    ax.legend(loc="upper right", handlelength=1.8, ncols=2, fontsize=8, labelspacing=0.3, columnspacing=0.8)

    res.set_ylabel(f"PDE residual on\n $512{TIMES}512$ test res.", fontsize=10)
    err.set_xlabel(f"Training time (seconds)", fontsize=10)
    err.set_ylabel(f"$L_2$ rel. error (\\%) on\n $512{TIMES}512$ test res.", fontsize=10)

    ax.set_xlabel("Inference time (seconds)", fontsize=10)
    ax.set_ylabel("$L_2$ rel. error (\\%) on different test resolutions", fontsize=10)

    res.xaxis.set_tick_params(labelsize=8)
    res.yaxis.set_tick_params(labelsize=8)
    err.xaxis.set_tick_params(labelsize=8)
    err.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

    res.text(-0.23, 1, "(a)", transform=res.transAxes, fontsize=10)
    err.text(-0.23, 1, "(b)", transform=err.transAxes, fontsize=10)
    ax.text(-0.13, 1, "(c)", transform=ax.transAxes, fontsize=10)

    figure.savefig(f"plot/re.curve.jpg", dpi=300)

# ---------------------------------------------------------------------------- #
#                                      BOX                                     #
# ---------------------------------------------------------------------------- #

FNOPINN = lambda n, m="": "FNO$\\!\\times\\!" + str(n) + "^2$" + m
LABELS = [None, FNOPINN(64, "\\ \\ "), FNOPINN(128), FNOPINN(256), f"CNO (ours)", f"NSM (ours)"]

fig, axes = plt.subplots(figsize=(16, 3.4), ncols=4)
for ax, nu in zip(axes, ["005", "01", "05", "1"]):

    u = np.load(f"src/pde/reaction/u.rho=5:nu=0.{nu.ljust(3, '0')}.npy")
    uhat = load(f"nu{nu}", "uhat")

    ax.boxplot([np.concatenate([np.mean(np.abs(uhat-u), axis=(1, 2, 3))
        for uhat in uhat[method][:1]]) for method in METHOD[1:]], labels=LABELS[1:])

    ax.set_title(f"Absolute error distribution for $\\nu=0.{nu}$", fontsize=10)

    ax.set_ylim(0, 0.06)

axes[0].text(-0.08, 1.04, "(a)", transform=axes[0].transAxes, fontsize=10)
axes[1].text(-0.08, 1.04, "(b)", transform=axes[1].transAxes, fontsize=10)
axes[2].text(-0.08, 1.04, "(c)", transform=axes[2].transAxes, fontsize=10)
axes[3].text(-0.08, 1.04, "(d)", transform=axes[3].transAxes, fontsize=10)

fig.savefig(f"plot/re.box.jpg", dpi=300)
