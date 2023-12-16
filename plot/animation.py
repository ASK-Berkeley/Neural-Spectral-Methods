from . import *
from sys import argv

N = list(map(int, argv[1:]))

u = np.load("log/aws/u.npy")[np.r_[N]]
unsm = np.load(f"log/aws/unsm.npy")[np.r_[N]]
ufno = np.load(f"log/aws/ux96.npy")[np.r_[N]]

fig, ax = plt.subplots(ncols=len(N), nrows=3, figsize=(2.1*len(N), 4),
                       constrained_layout=True, squeeze=False, dpi=72)

fig.set_constrained_layout_pads(hspace=0.1, wspace=0.1)

def create(sol, nsm, fno, i: int):

    N, K = 15, 4 # remove dark colors
    vmin = u[i].min() * (N + 2*K) / N
    vmax = u[i].max() * (N + 2*K) / N

    cmap = plt.get_cmap("twilight_shifted")
    norm = clr.BoundaryNorm(np.linspace(vmin, vmax, N+2*K), cmap.N)

    im_sol = sol.imshow(u[i, 0], cmap=cmap, norm=norm)
    im_nsm = nsm.imshow(unsm[i, 0], cmap=cmap, norm=norm)
    im_fno = fno.imshow(ufno[i, 0], cmap=cmap, norm=norm)

    sol.set_xticks([]); sol.set_yticks([])
    nsm.set_xticks([]); nsm.set_yticks([])
    fno.set_xticks([]); fno.set_yticks([])

    return im_sol, im_nsm, im_fno

im = []
for i in range(len(N)):
    im.append(create(*ax[:, i], i))

kw = dict(rotation=90, labelpad=10, fontsize=10)
ax[0, 0].set_ylabel("Numerical solver", **kw)
ax[1, 0].set_ylabel("NSM (ours) Prediction", **kw)
ax[2, 0].set_ylabel("FNO+PINN Prediction", **kw)

plt.colorbar(im[-1][0],fraction=0.046, pad=0.1, ticks=[-2, -1, 0, 1])
plt.colorbar(im[-1][1],fraction=0.046, pad=0.1, ticks=[-2, -1, 0, 1])
plt.colorbar(im[-1][2],fraction=0.046, pad=0.1, ticks=[-2, -1, 0, 1])

def frame(index, total: int = 64):
    i = int(total * (index/T/fps))

    for n, [sol, nsm, fno] in enumerate(im):

        sol.set_array(u[n, i])
        nsm.set_array(unsm[n, i])
        fno.set_array(ufno[n, i])

    return sum(im, ())

from matplotlib import animation
ani = animation.FuncAnimation(fig, frame, (T:=8)*(fps:=24), interval=T/fps * 1000, blit=True)
ani.save("plot/animation.gif", writer="pillow", fps=fps)
