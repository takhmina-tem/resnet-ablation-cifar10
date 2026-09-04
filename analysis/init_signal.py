"""Signal propagation at initialisation, before any weight has been updated.

The trained runs only measure gradients at 56 layers, so they show that the
plain network is badly conditioned but not how that scales with depth. Nothing
here needs training: build each network at its initial weights, push a few real
CIFAR-10 batches through, and look at what comes back.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import style
from style import SERIES, LABEL, MARKER, MUTED, FULL
from models.resnet_cifar import build_model
from data.dataset import get_dataloaders

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data_cache"
RESULTS_DIR = "results"
FIG_DIR = "figures"
DEPTHS = [3, 5, 9]
ARMS = ("resnet", "plain")
BATCHES = 8
SEED = 0


def measure(n, variant, batches):
    torch.manual_seed(SEED)
    model = build_model(n, variant)
    model.train()
    blocks = model.block_modules()

    acts = {i: [] for i in range(len(blocks))}
    handles = []
    for i, (_, block) in enumerate(blocks):
        def hook(_module, _inp, out, i=i):
            acts[i].append(out.detach().pow(2).mean().sqrt().item())
        handles.append(block.register_forward_hook(hook))

    grads = {i: [] for i in range(len(blocks))}
    criterion = nn.CrossEntropyLoss()
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        for i, (_, block) in enumerate(blocks):
            gsq = sum(p.grad.pow(2).sum().item() for p in block.parameters() if p.dim() == 4)
            grads[i].append(np.sqrt(gsq))
    for h in handles:
        h.remove()

    rows = []
    for i, (stage, block) in enumerate(blocks):
        wsq = sum(p.pow(2).sum().item() for p in block.parameters() if p.dim() == 4)
        rows.append(dict(depth=6 * n + 2, variant=variant, block=i, stage=stage,
                         position=i / (len(blocks) - 1),
                         act_rms=float(np.mean(acts[i])),
                         grad_norm=float(np.mean(grads[i])),
                         weight_norm=float(np.sqrt(wsq))))
    return pd.DataFrame(rows)


def shade(colour, k):
    # lighter for shallower networks, so the three depths read as one family
    rgb = np.array(plt.matplotlib.colors.to_rgb(colour))
    return tuple(rgb + (1 - rgb) * (0.55 * (1 - k)))


def main():
    style.use()
    os.makedirs(FIG_DIR, exist_ok=True)

    loader, _, _, _ = get_dataloaders(DATA_DIR, seed=SEED, batch_size=128, num_workers=0)
    torch.manual_seed(SEED)  # the training loader shuffles, so fix the batches too
    batches = []
    for i, batch in enumerate(loader):
        if i == BATCHES:
            break
        batches.append(batch)

    df = pd.concat([measure(n, v, batches) for n in DEPTHS for v in ARMS], ignore_index=True)
    df["relative"] = df.grad_norm / df.weight_norm
    df.to_csv(os.path.join(RESULTS_DIR, "init_signal.csv"), index=False)

    depths = sorted(df.depth.unique())
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.4))
    fig.subplots_adjust(wspace=0.58)

    for ax, col, ylabel in [(axes[0], "act_rms", "Block output RMS"),
                            (axes[1], "relative", "Grad norm / weight norm")]:
        for variant in ARMS:
            for k, d in enumerate(depths):
                s = df[(df.variant == variant) & (df.depth == d)].sort_values("block")
                ax.plot(s.position, s[col], color=shade(SERIES[variant], (k + 1) / len(depths)),
                        lw=1.3 + 0.25 * k)
        ax.set_yscale("log")
        ax.set_xlabel("Relative depth (input $\\rightarrow$ output)")
        ax.set_ylabel(ylabel)
        style.tidy(ax)
    axes[0].set_title("Forward signal", loc="left")
    axes[1].set_title("Backward signal", loc="left")
    for variant, y, va in [("resnet", 0.93, "top"), ("plain", 0.09, "bottom")]:
        axes[0].text(0.97, y, LABEL[variant], transform=axes[0].transAxes, ha="right",
                     va=va, color=SERIES[variant], fontsize=8)

    spread = df.groupby(["depth", "variant"]).relative.agg(lambda s: s.max() / s.min()).reset_index()
    ax = axes[2]
    for variant in ARMS:
        s = spread[spread.variant == variant].sort_values("depth")
        ax.plot(s.depth, s.relative, marker=MARKER[variant], color=SERIES[variant],
                label=LABEL[variant])
    ax.set_yscale("log")
    ax.set_xticks(depths)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("$\\max/\\min$ across blocks")
    ax.set_title("Imbalance vs depth", loc="left")
    ax.legend(loc="upper left")
    style.tidy(ax)

    # line weight and tint carry depth in the first two panels, so the key is
    # drawn in neutral grey rather than in either arm's colour
    handles = [plt.Line2D([], [], color=shade(MUTED, (k + 1) / len(depths)),
                          lw=1.3 + 0.25 * k, label=f"{d}L") for k, d in enumerate(depths)]
    axes[1].legend(handles=handles, loc="upper right", ncol=1, labelspacing=0.3,
                   handlelength=1.3, labelcolor=MUTED)

    fig.savefig(os.path.join(FIG_DIR, "fig9_init_signal.png"))
    plt.close(fig)

    spread.to_csv(os.path.join(RESULTS_DIR, "table4_init_spread.csv"), index=False)
    print(spread.to_string(index=False))
    print()
    ends = df[df.position.isin([0.0, 1.0])].pivot_table(
        index=["depth", "variant"], columns="position", values=["act_rms", "relative"])
    print(ends.to_string())


if __name__ == "__main__":
    main()
