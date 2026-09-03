import argparse
import glob
import itertools
import os
import shutil

from train import run


def mirror_to(src_dir, dst_dir):
    # Training writes locally: a mounted Drive is a FUSE filesystem and dies
    # under the per-step writes. Copying once per finished run is fine for it.
    try:
        os.makedirs(dst_dir, exist_ok=True)
        for f in glob.glob(os.path.join(src_dir, "*.csv")):
            shutil.copy(f, dst_dir)
        for d in glob.glob(os.path.join(src_dir, "*", "")):
            name = os.path.basename(os.path.normpath(d))
            out = os.path.join(dst_dir, name)
            os.makedirs(out, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*.csv")) + glob.glob(os.path.join(d, "DONE")):
                shutil.copy(f, out)
    except Exception as e:
        print("could not mirror results, continuing anyway:", e)

DEPTHS = [3, 5, 9]        # n -> resnet depth 6n+2 = 20, 32, 56
SEEDS = [0, 1]
ALPHA_DEPTH = 9
ALPHAS = [0.25, 0.5, 0.75]  # the 0.0 and 1.0 ends are the plain/resnet runs at ALPHA_DEPTH,
ALPHA_SEED = SEEDS[0]       # which with option A shortcuts are the identical network


def depth_configs(seed):
    for n, variant in itertools.product(DEPTHS, ["resnet", "plain"]):
        alpha = 1.0 if variant == "resnet" else 0.0
        yield dict(n=n, variant=variant, alpha=alpha, seed=seed)


def alpha_configs():
    for alpha in ALPHAS:
        yield dict(n=ALPHA_DEPTH, variant="scaled", alpha=alpha, seed=ALPHA_SEED)


def full_configs():
    # ordered so that stopping early still leaves a complete study: one seed
    # across every depth, then the alpha ablation, and only then the repeat
    # seeds that turn the point estimates into error bars
    yield from depth_configs(SEEDS[0])
    yield from alpha_configs()
    for seed in SEEDS[1:]:
        yield from depth_configs(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "full"], default="full")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data_dir", default="./data_cache")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--mirror", default=None, help="copy results here after each run")
    args = p.parse_args()

    if args.preset == "smoke":
        epochs = args.epochs or 2
        out_dir = args.out_dir or "./results_smoke"
        configs = [
            dict(n=3, variant="resnet", alpha=1.0, seed=0),
            dict(n=3, variant="plain", alpha=0.0, seed=0),
            dict(n=3, variant="scaled", alpha=0.5, seed=0),
        ]
    else:
        epochs = args.epochs or 30
        out_dir = args.out_dir or "./results"
        configs = list(full_configs())

    print(f"{len(configs)} runs, {epochs} epochs each, preset={args.preset}, out_dir={out_dir}")
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg}")
        run(n=cfg["n"], variant=cfg["variant"], alpha=cfg["alpha"], seed=cfg["seed"],
            epochs=epochs, data_dir=args.data_dir, out_dir=out_dir)
        if args.mirror:
            mirror_to(out_dir, args.mirror)


if __name__ == "__main__":
    main()
