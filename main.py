from src import *
from src.pde import *
from src.model import *
from src.basis import *

def main(cfg: Dict[str, Any]):

    # configure precision at the very beginning
    jax.config.update("jax_enable_x64", cfg["f64"])

    if cfg["smi"]:

        import jax_smi as smi
        smi.initialise_tracking()

    prng = random.PRNGKey(cfg["seed"])
    rngs = RNGS(prng, ["params", "sample"])

    from importlib import import_module

    if cfg["pde"] is not None:

        col, name = cfg["pde"].split(".", 2)
        mod = import_module(f"src.pde.{col}")

        pde: PDE = getattr(mod, name)
        pde.solution # load solution

        if cfg["model"] is not None:

            col, name = cfg["model"], cfg["model"].upper()

            if cfg["spectral"]: col += ".spectral"
            mod = import_module(f"src.model.{col}")

            Model: Solver = getattr(mod, name)
            model = Model(pde, cfg)

# ---------------------------------------------------------------------------- #
#                                     TRAIN                                    #
# ---------------------------------------------------------------------------- #

    if cfg["action"] == "train":

        from src.train import step, eval
        train = Trainer(model, pde, cfg)
        variable, state = train.init_with_output(next(rngs), method="init")

        step = utils.jit(F.partial(train.apply, method=step, mutable=True))
        step(state, variable, rngs=next(rngs))  # compile train iteration

        def evaluate():

            global metric, predictions
            metric, predictions = train.apply(state, variable,
                                  method=eval, rngs=next(rngs))

            if cfg["save"]:

                from flax.training.checkpoints import save_checkpoint
                save_checkpoint(f"{ckpt.path}/variable", variable, it)

        from ckpt import Checkpoint
        ckpt = Checkpoint(**cfg)
        ckpt.start()

        from tqdm import trange
        for it in (pbar:=trange(cfg["iter"])):

            if not it % cfg["ckpt"]: evaluate()

# ----------------------------------- STEP ----------------------------------- #

            import time
            rate = time.time()

            (variable, loss), state = jax.tree_map(jax.block_until_ready,
                                   step(state, variable, rngs=next(rngs)))

            rate = np.array(time.time() - rate)

# ----------------------------------- CKPT ----------------------------------- #

            metric.update(loss=loss, rate=rate)

            ckpt.metric.put((metric.copy(), it))
            ckpt.prediction = predictions

            pbar.set_postfix(jax.tree_map(lambda x: f"{x:.2e}", metric))

        evaluate()

        ckpt.metric.put((metric, None))
        ckpt.prediction = predictions
        ckpt.join()

        return pde, model.bind(variable, rngs=next(rngs))

# ---------------------------------------------------------------------------- #
#                                     LOAD                                     #
# ---------------------------------------------------------------------------- #

    else:

        if cfg["load"]:

            from flax.training.checkpoints import restore_checkpoint
            variable = restore_checkpoint(cfg["load"], target=None)
            model = model.bind(variable, rngs=next(rngs))

        exit(utils.repl(locals()))

# ---------------------------------------------------------------------------- #
#                                   ARGPARSE                                   #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":

    import argparse
    args = argparse.ArgumentParser()
    action = args.add_subparsers(dest="action")

    args.add_argument("--seed", type=int, default=19260817, help="random seed")
    args.add_argument("--f64", dest="f64", action="store_true", help="use double precision")
    args.add_argument("--smi", dest="smi", action="store_true", help="profile memory usage")

    args.add_argument("--pde", type=str, help="PDE name")
    args.add_argument("--model", type=str, help="model name", choices=["fno", "sno"]) # --cheb=cno
    args.add_argument("--spectral", dest="spectral", action="store_true", help="spectral training")

# ----------------------------------- MODEL ---------------------------------- #

    args.add_argument("--hdim", type=int, help="hidden dimension")
    args.add_argument("--depth", type=int, help="number of layers")
    args.add_argument("--activate", type=str, help="activation name")

    args.add_argument("--mode", type=int, nargs="+", help="number of modes per dim")
    args.add_argument("--grid", type=int, default=256, help="training grid size")

    ## ablation study

    args.add_argument("--fourier", dest="fourier", action="store_true", help="fourier basis only")
    args.add_argument("--cheb", dest="cheb", action="store_true", help="using chebyshev")

# ----------------------------------- TRAIN ---------------------------------- #

    args_train = action.add_parser("train", help="train model from scratch")

    args_train.add_argument("--bs", type=int, required=True, help="batch size")
    args_train.add_argument("--lr", type=float, required=True, help="learning rate")
    args_train.add_argument("--clip", type=float, required=False, help="gradient clipping")
    args_train.add_argument("--schd", type=str, required=True, help="scheduler name")
    args_train.add_argument("--iter", type=int, required=True, help="total iterations")
    args_train.add_argument("--ckpt", type=int, required=True, help="checkpoint every n iters")
    args_train.add_argument("--note", type=str, required=True, help="leave a note here")

    args_train.add_argument("--vmap", type=lambda x: int(x) if x else None, help="vectorization size")
    args_train.add_argument("--save", dest="save", action="store_true", help="save model checkpoints")

# ----------------------------------- TEST ----------------------------------- #

    args_test = action.add_parser("test", help="enter REPL after loading")

    args_test.add_argument("--load", type=str, help="saved model path")

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

    args = args.parse_args()
    cfg = vars(args); print(f"{cfg=}")

    pde, model = main(cfg)
    # utils.repl(locals())
