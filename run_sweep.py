import argparse
import itertools

from train import run

DEPTHS = [3, 5, 9]        # n -> resnet depth 6n+2 = 20, 32, 56
SEEDS = [0, 1]
ALPHA_DEPTH = 9
ALPHAS = [0.25, 0.5, 0.75]  # 0.0 and 1.0 already covered by plain/resnet at ALPHA_DEPTH
ALPHA_SEEDS = [0]


def baseline_configs():
    for n, variant, seed in itertools.product(DEPTHS, ["resnet", "plain"], SEEDS):
        alpha = 1.0 if variant == "resnet" else 0.0
        yield dict(n=n, variant=variant, alpha=alpha, seed=seed)


def alpha_configs():
    for alpha, seed in itertools.product(ALPHAS, ALPHA_SEEDS):
        yield dict(n=ALPHA_DEPTH, variant="scaled", alpha=alpha, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "full"], default="full")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--data_dir", default="./data_cache")
    p.add_argument("--out_dir", default="./results")
    args = p.parse_args()

    if args.preset == "smoke":
        epochs = args.epochs or 2
        configs = [
            dict(n=3, variant="resnet", alpha=1.0, seed=0),
            dict(n=3, variant="plain", alpha=0.0, seed=0),
            dict(n=3, variant="scaled", alpha=0.5, seed=0),
        ]
    else:
        epochs = args.epochs or 30
        configs = list(baseline_configs()) + list(alpha_configs())

    print(f"{len(configs)} runs, {epochs} epochs each, preset={args.preset}")
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg}")
        run(n=cfg["n"], variant=cfg["variant"], alpha=cfg["alpha"], seed=cfg["seed"],
            epochs=epochs, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
