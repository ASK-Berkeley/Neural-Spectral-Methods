from . import *

METHOD = [":SNO", "x64", "x128", "x256", ":NSM"]

load = F.partial(load, dir="log/ps.dirichlet", METHOD=METHOD)
bcs = ["periodic", "dirichlet"]

# ---------------------------------------------------------------------------- #
#                                     TABLE                                    #
# ---------------------------------------------------------------------------- #

print("="*116)
print("*relu*\t\t\t", "\t\t".join(METHOD))
print("-"*116)

for bc in bcs:

    print(f"{bc = }", end=":\t")

    for method, errs in load("relu/"+bc, "metric.errr").items():

        errs = np.array([err[-1] for err in errs]) * 100
        print(f"{np.mean(errs):.3f}±{np.std(errs):.3f}", end=" \t")

    print("")

print("="*116)
print("*tanh*\t\t\t", "\t\t".join(METHOD))
print("-"*116)

for bc in bcs:

    print(f"{bc = }", end=":\t")

    for method, errs in load("tanh/"+bc, "metric.errr").items():

        errs = np.array([err[-1] for err in errs]) * 100
        print(f"{np.mean(errs):.3f}±{np.std(errs):.3f}", end=" \t")

    print("")

print("="*116)
print("*long*\t\t\t", "\t\t".join(METHOD))
print("-"*116)

for bc in bcs:

    print(f"{bc = }", end=":\t")

    for method, errs in load("long/"+bc, "metric.errr").items():

        errs = np.array([err[-1] for err in errs]) * 100
        print(f"{np.mean(errs):.3f}±{np.std(errs):.3f}", end=" \t")

    print("")

print("="*116)
