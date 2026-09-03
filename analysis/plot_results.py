import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIG_DIR = "figures"

STYLE = {"resnet": ("o", "tab:blue", "ResNet"), "plain": ("s", "tab:orange", "Plain")}


def load_summary():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    # a re-run after a crash appends a second row for the same config
    df = df.drop_duplicates(subset="run_name", keep="last")
    dropped = df.diverged.astype(str).str.lower() == "true"
    if dropped.any():
        print("excluding diverged runs:", ", ".join(df[dropped].run_name))
    df = df[~dropped].copy()
    df["resnet_depth"] = 6 * df["depth_n"] + 2
    return df


def read_run(run_name, fname):
    path = os.path.join(RESULTS_DIR, run_name, fname)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    return None if d.empty else d


def fig_fit_and_test(df):
    base = df[df.variant.isin(["resnet", "plain"])]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)

    for ax, col, title in [(axes[0], "fit_acc", "Training set (no augmentation)"),
                            (axes[1], "test_acc", "Test set")]:
        agg = base.groupby(["resnet_depth", "variant"])[col].agg(["mean", "std"]).reset_index()
        for variant, (marker, color, label) in STYLE.items():
            sub = agg[agg.variant == variant].sort_values("resnet_depth")
            ax.errorbar(sub.resnet_depth, sub["mean"] * 100, yerr=sub["std"].fillna(0) * 100,
                        marker=marker, color=color, capsize=3, label=label)
        ax.set_xlabel("Depth (layers)")
        ax.set_title(title)
        ax.set_xticks(sorted(base.resnet_depth.unique()))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_fit_and_test_vs_depth.png"), dpi=150)
    plt.close(fig)

    out = base.groupby(["resnet_depth", "variant"])[["fit_acc", "test_acc"]].agg(["mean", "std"])
    out.to_csv(os.path.join(RESULTS_DIR, "table1_depth.csv"))
    return out


def fig_gap(df):
    base = df[df.variant.isin(["resnet", "plain"])].copy()
    base["gap"] = base.fit_acc - base.final_val_acc
    agg = base.groupby(["resnet_depth", "variant"]).gap.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for variant, (marker, color, label) in STYLE.items():
        sub = agg[agg.variant == variant].sort_values("resnet_depth")
        ax.errorbar(sub.resnet_depth, sub["mean"] * 100, yerr=sub["std"].fillna(0) * 100,
                    marker=marker, color=color, capsize=3, label=label)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Train - val accuracy (pp)")
    ax.set_title("Generalization gap vs. depth")
    ax.set_xticks(sorted(base.resnet_depth.unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_generalization_gap.png"), dpi=150)
    plt.close(fig)


def fig_loss_curves(runs, depth):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for run_name, (color, label) in runs.items():
        d = read_run(run_name, "epochs.csv")
        if d is None:
            continue
        ax.plot(d.epoch, d.train_loss, color=color, label=f"{label} train")
        ax.plot(d.epoch, d.val_loss, color=color, linestyle="--", alpha=0.7, label=f"{label} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Loss curves, {depth} layers")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_loss_curves.png"), dpi=150)
    plt.close(fig)


def fig_grad_profile(runs, depth, epoch=1):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    boundaries = []
    for run_name, (color, label) in runs.items():
        d = read_run(run_name, "grad_norms.csv")
        if d is None:
            continue
        d = d[d.epoch == epoch].sort_values("block")
        if d.empty:
            continue
        axes[0].plot(d.block, d.grad_norm, marker="o", ms=3, color=color, label=label)
        axes[1].plot(d.block, d.grad_norm / d.weight_norm, marker="o", ms=3, color=color, label=label)
        boundaries = d.block[d.stage.ne(d.stage.shift())].tolist()[1:]

    for ax, ylabel, title in [(axes[0], "Gradient L2 norm", "Absolute"),
                               (axes[1], "Grad norm / weight norm", "Relative to weight scale")]:
        for b in boundaries:
            ax.axvline(b - 0.5, color="grey", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Block index (input to output)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Gradient magnitude across blocks at epoch {epoch}, {depth} layers", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_grad_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_early_stability(runs, depth, window=20):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for run_name, (color, label) in runs.items():
        d = read_run(run_name, "steps.csv")
        if d is None:
            continue
        d = d.sort_values(["epoch", "step"]).reset_index(drop=True)
        x = np.arange(len(d))
        ax.plot(x, d.loss, color=color, alpha=0.25, lw=0.6)
        ax.plot(x, d.loss.rolling(window, min_periods=1).mean(), color=color, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Batch loss")
    ax.set_title(f"Early training, first epochs ({depth} layers)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_early_stability.png"), dpi=150)
    plt.close(fig)


def fig_alpha_sweep(df, depth_n):
    scaled = df[(df.variant == "scaled") & (df.depth_n == depth_n)]
    if scaled.empty:
        return
    # the endpoints have to come from the same seed the sweep was run at
    seed = scaled.seed.min()
    at_depth = df[(df.depth_n == depth_n) & (df.seed == seed)]
    pts = at_depth[at_depth.variant.isin(["scaled", "plain", "resnet"])]
    pts = pts[["alpha", "test_acc", "fit_acc"]].groupby("alpha").mean().reset_index().sort_values("alpha")
    if pts.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(pts.alpha, pts.test_acc * 100, marker="o", color="tab:blue", label="test")
    ax.plot(pts.alpha, pts.fit_acc * 100, marker="s", color="tab:green", label="train (clean)")
    ax.set_xlabel(r"Residual scale $\alpha$   (block output $= F(x) + \alpha x$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Effect of residual scale, {6 * depth_n + 2} layers")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_alpha_sweep.png"), dpi=150)
    plt.close(fig)
    pts.to_csv(os.path.join(RESULTS_DIR, "table2_alpha.csv"), index=False)


def latex_table(df):
    base = df[df.variant.isin(["resnet", "plain"])]
    depths = sorted(base.resnet_depth.unique())
    lines = []
    for variant, (_, _, label) in STYLE.items():
        for metric in ["fit_acc", "test_acc"]:
            cells = []
            for d in depths:
                sub = base[(base.variant == variant) & (base.resnet_depth == d)][metric].dropna()
                if sub.empty:
                    cells.append("--")
                elif len(sub) > 1:
                    cells.append(f"{sub.mean() * 100:.1f} $\\pm$ {sub.std() * 100:.1f}")
                else:
                    cells.append(f"{sub.mean() * 100:.1f}")
            name = f"{label} ({'train' if metric == 'fit_acc' else 'test'})"
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    path = os.path.join(RESULTS_DIR, "table1_body.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_summary()

    deepest = df.depth_n.max()
    depth_layers = 6 * deepest + 2

    def pick(variant):
        sub = df[(df.variant == variant) & (df.depth_n == deepest)].sort_values("seed")
        return sub.run_name.iloc[0] if not sub.empty else None

    runs = {}
    for variant, (_, color, label) in STYLE.items():
        name = pick(variant)
        if name:
            runs[name] = (color, label)

    print(fig_fit_and_test(df))
    fig_gap(df)
    fig_loss_curves(runs, depth_layers)
    fig_grad_profile(runs, depth_layers)
    fig_early_stability(runs, depth_layers)
    fig_alpha_sweep(df, deepest)
    latex_table(df)
    print("\nfigures written to", FIG_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    main()
