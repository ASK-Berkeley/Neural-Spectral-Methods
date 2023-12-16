from . import *

from sys import argv; N = int(argv[1])

# ---------------------------------------------------------------------------- #
#                                    POISSON                                   #
# ---------------------------------------------------------------------------- #

u = np.load(f"src/pde/poisson/u.periodic.npy")[N]
s = np.load("test.s.npy")

fig, [ax_s, ax_u, err] = plt.subplots(ncols=3, figsize=(8.5, 2.2),
                                      width_ratios=[1, 1, 1.3],
                                      constrained_layout=True)

plt.colorbar(ax_s.imshow(s, origin='lower', extent=[0, 255/256, 0, 255/256]), fraction=0.046, pad=0.04                       ); ax_s.grid(False); ax_s.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax_s.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax_s.set_title("Source term $s$", fontsize=10)
plt.colorbar(ax_u.imshow(u, origin='lower', extent=[0, 255/256, 0, 255/256]), fraction=0.046, pad=0.04, ticks=[0, 0.01, 0.02]); ax_u.grid(False); ax_u.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); ax_u.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax_u.set_title("Solution $u$", fontsize=10)

ax_s.xaxis.set_tick_params(labelsize=8)
ax_s.yaxis.set_tick_params(labelsize=8)
ax_u.xaxis.set_tick_params(labelsize=8)
ax_u.yaxis.set_tick_params(labelsize=8)

def work(METHOD, LABELS, run: str):

    load_now = F.partial(load, dir=f"log/ps/{run}")
    rate = load_now("periodic", "metric.rate", METHOD=METHOD)
    errr = load_now("periodic", "metric.errr", METHOD=METHOD)

    for method, label in zip(METHOD, LABELS):

        xs = np.linspace(1, max(map(np.sum, rate[method])), 1000)
        ys = [np.interp(xs, np.concatenate([np.zeros(1), np.cumsum(rate)]), errr)
                            for rate, errr in zip(rate[method], errr[method])]

        ys_mean = np.mean(np.stack(ys, axis=0), axis=0)
        ys_std = np.std(np.stack(ys, axis=0), axis=0); ys_std = np.minimum(ys_std, ys_mean/2)

        err.plot(xs, ys_mean, label=label, c=(c:=color(method)), ls=lines(method))
        err.fill_between(xs, ys_mean-ys_std, ys_mean+ys_std, alpha=0.2, color=c, edgecolor=None)

    err.set_xscale("log")
    err.set_yscale("log")

    err.set_xlim(1, 1e5)
    err.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])

    err.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, _: f"{y:.1%}"[:3]))

work([":NSM", ":SNO"], ["NSM", "SNO"], "relu")
work(["x64", "x128"], [f"FNO${TIMES}64^2\\ \\ $+PINN", f"FNO${TIMES}128^2$+PINN"], "tanh")
work(["x256"], [f"FNO${TIMES}256^2$+PINN"], "long")

err.set_title(f"$L_2$ rel. error (\\%) on $256{TIMES}256$ test res.", fontsize=10)
err.legend(loc="lower right")#, handlelength=1.3, bbox_to_anchor=(1.5, 0.5), fontsize=8, labelspacing=0.3)
err.xaxis.set_tick_params(labelsize=8)
err.yaxis.set_tick_params(labelsize=8)

ax_s.text(-0.11, 1.05, "(a)", transform=ax_s.transAxes, fontsize=10)
ax_u.text(-0.11, 1.05, "(b)", transform=ax_u.transAxes, fontsize=10)
err.text(-0.08, 1.05, "(c)", transform=err.transAxes, fontsize=10)

fig.savefig(f"plot/poisson.{N}.jpg", dpi=300)

# ---------------------------------------------------------------------------- #
#                                   REACTION                                   #
# ---------------------------------------------------------------------------- #

nus = [0.005, 0.01, 0.05, 0.1]
us = [np.load(f"src/pde/reaction/u.rho=5:nu={nu:.3f}.npy")[N] for nu in nus]

fig, axes = plt.subplots(ncols=4, figsize=(7, 2), constrained_layout=True)
for u, ax, nu in zip(us:=np.stack(us), axes, nus):

    ax.grid(False)
    im = ax.imshow(u, cmap="Spectral", origin="lower",
                   vmin=us.min(), vmax=us.max(),
                   extent=[0, 511/512, 0, 1])

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.set_title(f"$\\nu={nu}$")

plt.colorbar(im,fraction=0.046, pad=0.1)

fig.savefig(f"plot/reaction.{N}.jpg", dpi=300)
