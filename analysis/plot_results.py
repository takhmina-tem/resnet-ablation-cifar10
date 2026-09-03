import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style
from style import SERIES, LABEL, MARKER, INK, MUTED, FULL, HALF

RESULTS_DIR = "results"
FIG_DIR = "figures"
ARMS = ("resnet", "plain")
THRESHOLD = 0.80


def load_summary():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    df = df.drop_duplicates(subset="run_name", keep="last")
    dropped = df.diverged.astype(str).str.lower() == "true"
    if dropped.any():
        print("excluding diverged runs:", ", ".join(df[dropped].run_name))
    df = df[~dropped].copy()
    df["depth"] = 6 * df["depth_n"] + 2
    return df


def read_run(run_name, fname):
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    return None if d.empty else d


def arm_series(df, col):
    return df.groupby(["depth", "variant"])[col].agg(["mean", "std"]).reset_index()


def fig_fit_and_test(df):
    base = df[df.variant.isin(ARMS)]
    depths = sorted(base.depth.unique())
    # shared y: both panels are accuracy, and separate scales would exaggerate
    # the slope difference between them
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharex=True, sharey=True)

    for ax, col, title in [(axes[0], "fit_acc", "Training set (no augmentation)"),
                            (axes[1], "test_acc", "Test set")]:
        agg = arm_series(base, col)
        for variant in ARMS:
            s = agg[agg.variant == variant].sort_values("depth")
            ax.errorbar(s.depth, s["mean"] * 100, yerr=s["std"].fillna(0) * 100,
                        marker=MARKER[variant], color=SERIES[variant],
                        capsize=2.5, elinewidth=0.8, label=LABEL[variant])
            last = s.iloc[-1]
            ax.annotate(LABEL[variant], (last.depth, last["mean"] * 100),
                        textcoords="offset points", xytext=(6, -1),
                        color=INK, fontsize=7.5, va="center")
        ax.set_xlabel("Depth (layers)")
        ax.set_title(title, loc="left")
        ax.set_xticks(depths)
        ax.set_xlim(depths[0] - 3, depths[-1] + 11)
        style.tidy(ax)
    axes[0].set_ylabel("Accuracy (%)")
    fig.savefig(os.path.join(FIG_DIR, "fig1_fit_and_test_vs_depth.png"))
    plt.close(fig)

    out = base.groupby(["depth", "variant"])[["fit_acc", "test_acc"]].agg(["mean", "std"])
    out.to_csv(os.path.join(RESULTS_DIR, "table1_depth.csv"))
    return out


def epochs_to_threshold(df, thr=THRESHOLD):
    rows = []
    for _, r in df[df.variant.isin(ARMS)].iterrows():
        d = read_run(r.run_name, "epochs.csv")
        if d is None:
            continue
        hit = d[d.train_acc >= thr]
        rows.append(dict(depth=r.depth, variant=r.variant, seed=r.seed,
                          epochs=int(hit.epoch.iloc[0]) if not hit.empty else np.nan,
                          budget=int(d.epoch.max())))
    return pd.DataFrame(rows)


def fig_convergence(df):
    hit = epochs_to_threshold(df)
    if hit.empty:
        return
    agg = hit.groupby(["depth", "variant"]).agg(
        epochs=("epochs", "mean"),
        n_reached=("epochs", "count"),
        n_seeds=("epochs", "size"),
        budget=("budget", "max")).reset_index()
    # a seed that never reaches the threshold is right-censored, not missing:
    # averaging only the seeds that made it would understate the cost
    agg["censored"] = agg.n_reached < agg.n_seeds
    agg["plotted"] = np.where(agg.censored, agg.budget, agg.epochs)
    depths = sorted(agg.depth.unique())

    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    for variant in ARMS:
        s = agg[agg.variant == variant].sort_values("depth")
        ax.plot(s.depth, s.plotted, color=SERIES[variant], label=LABEL[variant], zorder=2)
        full = s[~s.censored]
        ax.plot(full.depth, full.plotted, marker=MARKER[variant], ls="none",
                color=SERIES[variant], zorder=3)
        for _, r in s[s.censored].iterrows():
            ax.plot([r.depth], [r.plotted], marker=MARKER[variant], ls="none",
                    mfc="white", mec=SERIES[variant], mew=1.2, zorder=3)
            ax.annotate(f"$\\geq${int(r.budget)}\n{int(r.n_seeds - r.n_reached)} of "
                        f"{int(r.n_seeds)} never reached",
                        (r.depth, r.plotted), textcoords="offset points",
                        xytext=(-4, -4), ha="right", va="top",
                        color=MUTED, fontsize=6.8)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel(f"Epochs to {int(THRESHOLD * 100)}% train accuracy")
    ax.set_xticks(depths)
    ax.set_ylim(0, agg.budget.max() * 1.12)
    ax.legend(loc="upper left")
    style.tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig7_convergence_speed.png"))
    plt.close(fig)
    agg.to_csv(os.path.join(RESULTS_DIR, "table3_convergence.csv"), index=False)
    return agg


def fig_gap(df):
    base = df[df.variant.isin(ARMS)].copy()
    base["gap"] = (base.fit_acc - base.final_val_acc) * 100
    agg = base.groupby(["depth", "variant"]).gap.agg(["mean", "std"]).reset_index()
    depths = sorted(base.depth.unique())

    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    for variant in ARMS:
        s = agg[agg.variant == variant].sort_values("depth")
        ax.errorbar(s.depth, s["mean"], yerr=s["std"].fillna(0),
                    marker=MARKER[variant], color=SERIES[variant],
                    capsize=2.5, elinewidth=0.8, label=LABEL[variant])
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Train $-$ val accuracy (pp)")
    ax.set_xticks(depths)
    ax.legend()
    style.tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig3_generalization_gap.png"))
    plt.close(fig)


def fig_loss_curves(runs, depth):
    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    for variant, run_name in runs.items():
        d = read_run(run_name, "epochs.csv")
        if d is None:
            continue
        ax.plot(d.epoch, d.train_loss, color=SERIES[variant], label=f"{LABEL[variant]} train")
        ax.plot(d.epoch, d.val_loss, color=SERIES[variant], ls=(0, (3, 2)),
                lw=1.1, alpha=0.85, label=f"{LABEL[variant]} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{depth} layers", loc="left")
    ax.legend(ncol=2, columnspacing=1.2, handlelength=1.6)
    style.tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig2_loss_curves.png"))
    plt.close(fig)


def _grad_at(run_name, epoch):
    d = read_run(run_name, "grad_norms.csv")
    if d is None:
        return None
    d = d[d.epoch == epoch].sort_values("block")
    return None if d.empty else d


def fig_grad_profile(runs, depth):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5))
    bounds = []
    for variant, run_name in runs.items():
        d = _grad_at(run_name, 1)
        if d is None:
            continue
        axes[0].plot(d.block, d.grad_norm, marker=MARKER[variant], ms=3,
                     color=SERIES[variant], label=LABEL[variant])
        axes[1].plot(d.block, d.grad_norm / d.weight_norm, marker=MARKER[variant], ms=3,
                     color=SERIES[variant], label=LABEL[variant])
        bounds = d.block[d.stage.ne(d.stage.shift())].tolist()[1:]

    for ax, ylabel in [(axes[0], "Gradient $L_2$ norm"),
                        (axes[1], "Gradient norm / weight norm")]:
        for b in bounds:
            ax.axvline(b - 0.5, color=style.GRID, lw=0.7, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Block index (input $\\rightarrow$ output)")
        ax.set_ylabel(ylabel)
        style.tidy(ax)
    axes[0].set_title("Absolute", loc="left")
    axes[1].set_title("Relative to weight scale", loc="left")
    axes[0].legend()
    fig.savefig(os.path.join(FIG_DIR, "fig4_grad_profile.png"))
    plt.close(fig)


def fig_grad_evolution(runs, depth):
    epochs = None
    for run_name in runs.values():
        d = read_run(run_name, "grad_norms.csv")
        if d is not None:
            epochs = sorted(d.epoch.unique())
            break
    if not epochs:
        return

    fig, axes = plt.subplots(1, len(epochs), figsize=(FULL, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ep in zip(axes, epochs):
        for variant, run_name in runs.items():
            d = _grad_at(run_name, ep)
            if d is None:
                continue
            ax.plot(d.block, d.grad_norm / d.weight_norm, color=SERIES[variant],
                    lw=1.4, label=LABEL[variant])
        ax.set_yscale("log")
        ax.set_xlabel("Block index")
        ax.set_title(f"Epoch {ep}", loc="left")
        style.tidy(ax)
    axes[0].set_ylabel("Grad norm / weight norm")
    axes[0].legend()
    fig.savefig(os.path.join(FIG_DIR, "fig8_grad_evolution.png"))
    plt.close(fig)


def fig_early_stability(runs, depth, window=25):
    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    for variant, run_name in runs.items():
        d = read_run(run_name, "steps.csv")
        if d is None:
            continue
        d = d.sort_values(["epoch", "step"]).reset_index(drop=True)
        x = np.arange(len(d))
        ax.plot(x, d.loss, color=SERIES[variant], lw=0.4, alpha=0.22)
        ax.plot(x, d.loss.rolling(window, min_periods=1).mean(),
                color=SERIES[variant], label=LABEL[variant])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Minibatch loss")
    ax.set_title(f"First epochs, {depth} layers", loc="left")
    ax.legend()
    style.tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig5_early_stability.png"))
    plt.close(fig)


def fig_alpha_sweep(df, depth_n):
    scaled = df[(df.variant == "scaled") & (df.depth_n == depth_n)]
    if scaled.empty:
        return
    seed = scaled.seed.min()
    at = df[(df.depth_n == depth_n) & (df.seed == seed)]
    pts = at[at.variant.isin(["scaled", "plain", "resnet"])]
    pts = pts.groupby("alpha")[["fit_acc", "test_acc"]].mean().reset_index().sort_values("alpha")

    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.5))
    ax.plot(pts.alpha, pts.fit_acc * 100, marker="D", color=style.AQUA, label="Train (clean)")
    ax.plot(pts.alpha, pts.test_acc * 100, marker="o", color=style.BLUE, label="Test")
    for a, lbl in [(0.0, "plain"), (1.0, "ResNet")]:
        row = pts[pts.alpha == a]
        if not row.empty:
            ax.annotate(lbl, (a, row.test_acc.iloc[0] * 100), textcoords="offset points",
                        xytext=(0, -12), ha="center", color=MUTED, fontsize=7.5)
    ax.set_xlabel(r"Residual scale $\alpha$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{6 * depth_n + 2} layers", loc="left")
    ax.legend(loc="lower right")
    style.tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig6_alpha_sweep.png"))
    plt.close(fig)
    pts.to_csv(os.path.join(RESULTS_DIR, "table2_alpha.csv"), index=False)


def latex_table(df):
    base = df[df.variant.isin(ARMS)]
    depths = sorted(base.depth.unique())
    lines = []
    for variant in ARMS:
        for metric, tag in [("fit_acc", "train"), ("test_acc", "test")]:
            cells = []
            for d in depths:
                s = base[(base.variant == variant) & (base.depth == d)][metric].dropna()
                if s.empty:
                    cells.append("--")
                elif len(s) > 1:
                    cells.append(f"{s.mean() * 100:.1f} $\\pm$ {s.std() * 100:.1f}")
                else:
                    cells.append(f"{s.mean() * 100:.1f}")
            lines.append(f"{LABEL[variant]} ({tag}) & " + " & ".join(cells) + r" \\")
    body = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "table1_body.tex"), "w") as f:
        f.write(body + "\n")
    print(body)


def main():
    style.use()
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_summary()

    deepest_n = df.depth_n.max()
    depth = 6 * deepest_n + 2
    runs = {}
    for variant in ARMS:
        s = df[(df.variant == variant) & (df.depth_n == deepest_n)].sort_values("seed")
        if not s.empty:
            runs[variant] = s.run_name.iloc[0]

    print(fig_fit_and_test(df), "\n")
    fig_gap(df)
    fig_loss_curves(runs, depth)
    fig_grad_profile(runs, depth)
    fig_grad_evolution(runs, depth)
    fig_early_stability(runs, depth)
    fig_alpha_sweep(df, deepest_n)
    conv = fig_convergence(df)
    if conv is not None:
        print(conv.to_string(index=False), "\n")
    latex_table(df)
    print("\nfigures written to", FIG_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    main()
