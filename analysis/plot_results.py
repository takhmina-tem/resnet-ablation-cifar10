import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#1a1a19"
MUTED = "#6b6a66"
GRID = "#d9d8d4"

SERIES = {"resnet": BLUE, "plain": ORANGE, "scaled": AQUA}
LABEL = {"resnet": "ResNet", "plain": "Plain", "scaled": r"$F(x)+\alpha x$"}
MARKER = {"resnet": "o", "plain": "s", "scaled": "D"}

FULL = 6.5
HALF = 3.25


def use_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.7,
        "lines.markersize": 4.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "axes.grid": False,
    })


def tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)


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


def missing(df, fname):
    absent = [r for r in df.run_name if read_run(r, fname) is None]
    if absent:
        print(f"no {fname} for:", ", ".join(absent))
    return absent


def arm_series(df, col):
    return df.groupby(["depth", "variant"])[col].agg(["mean", "std"]).reset_index()


def fig_fit_and_test(df):
    base = df[df.variant.isin(ARMS)]
    depths = sorted(base.depth.unique())
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
        tidy(ax)
    axes[0].set_ylabel("Accuracy (%)")
    fig.savefig(os.path.join(FIG_DIR, "fig1_fit_and_test_vs_depth.png"))
    plt.close(fig)

    out = base.groupby(["depth", "variant"])[["fit_acc", "test_acc"]].agg(["mean", "std"])
    out.to_csv(os.path.join(RESULTS_DIR, "table1_depth.csv"))
    return out


def epochs_to_threshold(df, thr=THRESHOLD, arms=ARMS):
    # running train acc, not the clean fit acc; read the same way for both arms
    rows = []
    for _, r in df[df.variant.isin(arms)].iterrows():
        d = read_run(r.run_name, "epochs.csv")
        if d is None:
            continue
        hit = d[d.train_acc >= thr]
        rows.append(dict(depth=r.depth, variant=r.variant, alpha=r.alpha, seed=r.seed,
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
    # censored, not missing
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
                        xytext=(-6, -9), ha="right", va="top",
                        color=MUTED, fontsize=6.8)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel(f"Epochs to {int(THRESHOLD * 100)}% train accuracy")
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 2, depths[-1] + 2)
    ax.set_ylim(0, agg.budget.max() * 1.25)
    ax.legend(loc="upper left")
    tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig7_convergence_speed.png"))
    plt.close(fig)
    agg.to_csv(os.path.join(RESULTS_DIR, "table3_convergence.csv"), index=False)
    return agg


def fig_gap(df):
    base = df[df.variant.isin(ARMS)].copy()
    base["gap"] = (base.fit_acc - base.sel_val_acc) * 100
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
    tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig3_generalization_gap.png"))
    plt.close(fig)


def fig_curves(df):
    depths = sorted(df[df.variant.isin(ARMS)].depth.unique())
    fig, axes = plt.subplots(2, len(depths), figsize=(FULL, 4.1), sharex=True)
    fig.subplots_adjust(hspace=0.22, wspace=0.28)

    for col, depth in enumerate(depths):
        for row, (metric, ylabel) in enumerate([("loss", "Cross-entropy loss"),
                                                 ("acc", "Accuracy (%)")]):
            ax = axes[row, col]
            for variant in ARMS:
                runs = df[(df.variant == variant) & (df.depth == depth)].run_name
                for run_name in runs:
                    d = read_run(run_name, "epochs.csv")
                    if d is None:
                        continue
                    scale = 100 if metric == "acc" else 1
                    ax.plot(d.epoch, d[f"train_{metric}"] * scale, color=SERIES[variant], lw=1.3)
                    ax.plot(d.epoch, d[f"val_{metric}"] * scale, color=SERIES[variant],
                            ls=(0, (3, 2)), lw=1.0, alpha=0.9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{depth} layers", loc="left")
            else:
                ax.set_xlabel("Epoch")
            tidy(ax)

    keys = [plt.Line2D([], [], color=SERIES[v], lw=1.3, label=LABEL[v]) for v in ARMS]
    keys += [plt.Line2D([], [], color=MUTED, lw=1.3, label="train"),
             plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="validation")]
    axes[0, -1].legend(handles=keys, loc="upper right", ncol=2, columnspacing=1.0,
                       handlelength=1.5, fontsize=7.5)
    fig.savefig(os.path.join(FIG_DIR, "fig2_curves.png"))
    plt.close(fig)


def _grad_at(run_name, epoch):
    d = read_run(run_name, "grad_norms.csv")
    if d is None:
        return None
    d = d[d.epoch == epoch].sort_values("block")
    return None if d.empty else d


def fig_grad_profile(runs):
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
            ax.axvline(b - 0.5, color=GRID, lw=0.7, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Block index (input $\\rightarrow$ output)")
        ax.set_ylabel(ylabel)
        tidy(ax)
    axes[0].set_title("Absolute", loc="left")
    axes[1].set_title("Relative to weight scale", loc="left")
    axes[0].legend()
    fig.savefig(os.path.join(FIG_DIR, "fig4_grad_profile.png"))
    plt.close(fig)


def fig_grad_evolution(runs):
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
        tidy(ax)
    axes[0].set_ylabel("Grad norm / weight norm")
    axes[0].legend()
    fig.savefig(os.path.join(FIG_DIR, "fig8_grad_evolution.png"))
    plt.close(fig)


def fig_stability(df, window=25):
    depths = sorted(df[df.variant.isin(ARMS)].depth.unique())
    fig, axes = plt.subplots(1, len(depths) + 1, figsize=(FULL, 2.3))
    fig.subplots_adjust(wspace=0.3)

    for ax, depth in zip(axes, depths):
        for variant in ARMS:
            s = df[(df.variant == variant) & (df.depth == depth)].sort_values("seed")
            if s.empty:
                continue
            d = read_run(s.run_name.iloc[0], "steps.csv")
            if d is None:
                continue
            d = d.sort_values(["epoch", "step"]).reset_index(drop=True)
            x = np.arange(len(d))
            ax.plot(x, d.loss, color=SERIES[variant], lw=0.4, alpha=0.22)
            ax.plot(x, d.loss.rolling(window, min_periods=1).mean(),
                    color=SERIES[variant], label=LABEL[variant])
        ax.set_yscale("log")
        ax.set_ylim(0.8, 20)
        ax.set_xlabel("Training step")
        ax.set_title(f"{depth} layers", loc="left")
        tidy(ax)
    axes[0].set_ylabel("Minibatch loss")
    axes[0].legend(loc="upper right")

    vol = val_volatility(df[df.variant.isin(ARMS)])
    agg = vol.groupby(["depth", "variant"]).volatility.agg(["mean", "min", "max"]).reset_index()
    ax = axes[-1]
    for variant in ARMS:
        v = agg[agg.variant == variant].sort_values("depth")
        ax.errorbar(v.depth, v["mean"], yerr=[v["mean"] - v["min"], v["max"] - v["mean"]],
                    marker=MARKER[variant], color=SERIES[variant], capsize=2.5,
                    elinewidth=0.8, label=LABEL[variant])
    ax.set_xticks(depths)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Mean $|\\Delta|$ val accuracy (pp)")
    ax.set_title("Epochs 5 to 30", loc="left")
    tidy(ax)

    fig.savefig(os.path.join(FIG_DIR, "fig5_stability.png"))
    plt.close(fig)
    vol.to_csv(os.path.join(RESULTS_DIR, "table7_volatility.csv"), index=False)
    return agg


def fig_alpha_sweep(df, depth_n):
    scaled = df[(df.variant == "scaled") & (df.depth_n == depth_n)]
    if scaled.empty:
        return
    seed = scaled.seed.min()
    at = df[(df.depth_n == depth_n) & (df.seed == seed)]
    pts = at.groupby("alpha")[["fit_acc", "test_acc"]].mean().reset_index().sort_values("alpha")

    hit = epochs_to_threshold(at, arms=("resnet", "plain", "scaled"))
    vol = val_volatility(at)
    pts = pts.merge(hit.groupby("alpha").epochs.mean().reset_index(), on="alpha", how="left")
    pts = pts.merge(vol.groupby("alpha").volatility.mean().reset_index(), on="alpha", how="left")

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.5))
    fig.subplots_adjust(wspace=0.38)

    ax = axes[0]
    ax.plot(pts.alpha, pts.fit_acc * 100, marker="D", color=AQUA, label="Train (clean)")
    ax.plot(pts.alpha, pts.test_acc * 100, marker="o", color=BLUE, label="Test")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final accuracy", loc="left")
    ax.legend(loc="lower right")

    axes[1].plot(pts.alpha, pts.epochs, marker="s", color=ORANGE)
    axes[1].set_ylabel(f"Epochs to {int(THRESHOLD * 100)}% train accuracy")
    axes[1].set_title("Optimization speed", loc="left")

    axes[2].plot(pts.alpha, pts.volatility, marker="o", color=AQUA)
    axes[2].set_ylabel("Mean $|\\Delta|$ val accuracy (pp)")
    axes[2].set_title("Late-training stability", loc="left")

    for ax in axes:
        ax.set_xlabel(r"Shortcut scale $\alpha$")
        ax.set_xticks(pts.alpha)
        tidy(ax)
    for a, lbl in [(0.0, "plain"), (1.0, "ResNet")]:
        row = pts[pts.alpha == a]
        if not row.empty:
            side = 7 if a == 0.0 else -7
            axes[0].annotate(lbl, (a, row.test_acc.iloc[0] * 100), textcoords="offset points",
                             xytext=(side, -2), ha="left" if a == 0.0 else "right",
                             va="center", color=MUTED, fontsize=7.5)
    fig.savefig(os.path.join(FIG_DIR, "fig6_alpha_sweep.png"))
    plt.close(fig)
    pts.to_csv(os.path.join(RESULTS_DIR, "table2_alpha.csv"), index=False)
    return pts


def fig_fit_vs_test(df):
    slope, intercept = np.polyfit(df.fit_acc * 100, df.test_acc * 100, 1)
    resid = df.test_acc * 100 - (slope * df.fit_acc * 100 + intercept)

    fig, ax = plt.subplots(figsize=(HALF + 0.4, 2.7))
    grid = np.linspace(df.fit_acc.min() * 100 - 1, df.fit_acc.max() * 100 + 1, 2)
    ax.plot(grid, slope * grid + intercept, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    for variant in ("resnet", "plain", "scaled"):
        s = df[df.variant == variant]
        ax.plot(s.fit_acc * 100, s.test_acc * 100, ls="none", marker=MARKER[variant],
                color=SERIES[variant], label=LABEL[variant], mew=0, zorder=3)

    pair = df[df.run_name.isin(["n3_resnet_a1.00_s0", "n9_scaled_a0.75_s0"])]
    if len(pair) == 2:
        ax.annotate("20L ResNet and 56L $\\alpha{=}0.75$\nland on the same point",
                    (pair.fit_acc.mean() * 100, pair.test_acc.mean() * 100),
                    textcoords="offset points", xytext=(-8, 16), ha="right",
                    fontsize=7, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                    shrinkA=0, shrinkB=4))
    ax.set_xlabel("Training accuracy, no augmentation (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.savefig(os.path.join(FIG_DIR, "fig10_fit_vs_test.png"))
    plt.close(fig)

    out = df[["run_name", "variant", "depth", "seed", "fit_acc", "test_acc"]].copy()
    out["residual_pp"] = resid
    out.to_csv(os.path.join(RESULTS_DIR, "table5_fit_vs_test.csv"), index=False)
    return slope, intercept, np.corrcoef(df.fit_acc, df.test_acc)[0, 1], resid


def imbalance_ratios(df):
    rows = []
    for _, r in df[df.variant.isin(ARMS)].iterrows():
        g = read_run(r.run_name, "grad_norms.csv")
        if g is None:
            continue
        g = g.assign(relative=g.grad_norm / g.weight_norm)
        for epoch, block in g.groupby("epoch"):
            block = block.sort_values("block")
            rows.append(dict(depth=r.depth, variant=r.variant, seed=r.seed, epoch=epoch,
                             ratio=block.relative.iloc[0] / block.relative.iloc[-1]))
    return pd.DataFrame(rows)


def fig_imbalance_vs_depth(df):
    ratios = imbalance_ratios(df)
    if ratios.empty:
        return None
    epochs = sorted(ratios.epoch.unique())[:2]
    depths = sorted(ratios.depth.unique())

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, epoch in zip(axes, epochs):
        at = ratios[ratios.epoch == epoch]
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
        for variant in ARMS:
            s = at[at.variant == variant].groupby("depth").ratio.mean().reset_index()
            ax.plot(s.depth, s.ratio, marker=MARKER[variant], color=SERIES[variant],
                    label=LABEL[variant], zorder=2)
            raw = at[at.variant == variant]
            ax.plot(raw.depth, raw.ratio, ls="none", marker=MARKER[variant], ms=3,
                    color=SERIES[variant], alpha=0.45, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(depths)
        ax.set_xlabel("Depth (layers)")
        ax.set_title(f"Epoch {int(epoch)}", loc="left")
        tidy(ax)
    axes[0].set_ylabel("Relative gradient,\nfirst block / last block")
    axes[0].legend(loc="upper left")
    axes[1].text(0.03, 0.06, "below 1: the early layers now\nreceive the smaller update",
                 transform=axes[1].transAxes, ha="left", va="bottom",
                 fontsize=7, color=MUTED)
    fig.savefig(os.path.join(FIG_DIR, "fig11_imbalance_vs_depth.png"))
    plt.close(fig)

    agg = ratios.groupby(["depth", "variant", "epoch"]).ratio.mean().reset_index()
    agg.to_csv(os.path.join(RESULTS_DIR, "table6_imbalance.csv"), index=False)
    return agg


def val_volatility(df, from_epoch=5):
    rows = []
    for _, r in df.iterrows():
        d = read_run(r.run_name, "epochs.csv")
        if d is None:
            continue
        late = d[d.epoch >= from_epoch]
        rows.append(dict(run_name=r.run_name, depth=r.depth, variant=r.variant,
                         alpha=r.alpha, seed=r.seed,
                         volatility=late.val_acc.diff().abs().mean() * 100))
    return pd.DataFrame(rows)


def latex_table(df):
    # whole tabular, not just the rows: \input of rows breaks \bottomrule
    base = df[df.variant.isin(ARMS)]
    depths = sorted(base.depth.unique())
    lines = [r"\begin{tabular}{l" + "c" * len(depths) + "}", r"\toprule",
             " & " + " & ".join(f"{d} layers" for d in depths) + r" \\", r"\midrule"]
    for variant in ARMS:
        for metric, tag in [("fit_acc", "train"), ("test_acc", "test")]:
            cells = []
            for d in depths:
                v = base[(base.variant == variant) & (base.depth == d)][metric].dropna()
                if v.empty:
                    cells.append("--")
                elif len(v) > 1:
                    cells.append(f"{v.mean() * 100:.1f} $\\pm$ {v.std() * 100:.1f}")
                else:
                    cells.append(f"{v.mean() * 100:.1f}")
            lines.append(f"{LABEL[variant]} ({tag}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "table1_depth.tex"), "w") as f:
        f.write(body + "\n")
    print(body)


def latex_runs_table(df):
    vol = val_volatility(df).set_index("run_name")
    lines = [r"\begin{tabular}{llcrcccrc}", r"\toprule",
             r"Depth & Variant & Seed & Params & Fit & Val & Test & Reach & Move \\",
             r"\midrule"]
    for _, r in df.sort_values(["depth", "variant", "alpha", "seed"]).iterrows():
        d = read_run(r.run_name, "epochs.csv")
        reached = "--"
        if d is not None:
            crossed = d[d.train_acc >= THRESHOLD]
            reached = str(int(crossed.epoch.iloc[0])) if not crossed.empty else f"$>${int(d.epoch.max())}"
        name = LABEL[r.variant] if r.variant != "scaled" else f"$\\alpha={r.alpha:g}$"
        params = f"{int(r.n_params):,}".replace(",", "{,}")
        lines.append(f"{int(r.depth)} & {name} & {int(r.seed)} & {params} & "
                     f"{r.fit_acc * 100:.2f} & {r.sel_val_acc * 100:.2f} & "
                     f"{r.test_acc * 100:.2f} & {reached} & "
                     f"{vol.volatility.get(r.run_name, float('nan')):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS_DIR, "table_runs.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    use_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_summary()
    for fname in ("epochs.csv", "grad_norms.csv", "steps.csv"):
        missing(df, fname)

    deepest_n = df.depth_n.max()
    depth = 6 * deepest_n + 2
    runs = {}
    for variant in ARMS:
        s = df[(df.variant == variant) & (df.depth_n == deepest_n)].sort_values("seed")
        if not s.empty:
            runs[variant] = s.run_name.iloc[0]

    print(fig_fit_and_test(df), "\n")
    fig_gap(df)
    fig_curves(df)
    fig_grad_profile(runs)
    fig_grad_evolution(runs)

    slope, intercept, r, resid = fig_fit_vs_test(df)
    print(f"test = {slope:.3f} * fit + {intercept:.2f}   r = {r:.4f}   "
          f"largest residual {np.abs(resid).max():.2f} pp")
    print(df.assign(resid=resid).groupby("variant").resid.mean().to_string(), "\n")

    stab = fig_stability(df)
    print(stab.to_string(index=False), "\n")
    print(fig_alpha_sweep(df, deepest_n).to_string(index=False), "\n")
    imb = fig_imbalance_vs_depth(df)
    if imb is not None:
        print(imb.pivot_table(index=["depth", "variant"], columns="epoch",
                              values="ratio").to_string(), "\n")
    conv = fig_convergence(df)
    if conv is not None:
        print(conv.to_string(index=False), "\n")
    latex_table(df)
    latex_runs_table(df)
    print("\nfigures written to", FIG_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
        # so a smoke sweep can't overwrite the report figures
        if os.path.basename(RESULTS_DIR.rstrip("/")) != "results":
            FIG_DIR = os.path.join(FIG_DIR, os.path.basename(RESULTS_DIR.rstrip("/")))
    main()
