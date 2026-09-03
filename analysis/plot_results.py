import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load_summary():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))
    df["resnet_depth"] = 6 * df["depth_n"] + 2
    return df


def fig_accuracy_vs_depth(df):
    base = df[df.variant.isin(["resnet", "plain"])]
    agg = base.groupby(["resnet_depth", "variant"]).test_acc.agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    for variant, marker in [("resnet", "o"), ("plain", "s")]:
        sub = agg[agg.variant == variant].sort_values("resnet_depth")
        ax.errorbar(sub.resnet_depth, sub["mean"] * 100, yerr=sub["std"].fillna(0) * 100,
                    marker=marker, capsize=3, label=variant)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy vs. depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_accuracy_vs_depth.png"), dpi=150)
    plt.close(fig)
    agg.to_csv(os.path.join(RESULTS_DIR, "table1_accuracy_vs_depth.csv"), index=False)
    return agg


def fig_loss_curves(run_names, labels, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    for run_name, label in zip(run_names, labels):
        path = os.path.join(RESULTS_DIR, run_name, "epochs.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        ax.plot(d.epoch, d.train_loss, label=f"{label} train")
        ax.plot(d.epoch, d.val_loss, linestyle="--", label=f"{label} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close(fig)


def fig_generalization_gap(df):
    rows = []
    for _, row in df.iterrows():
        path = os.path.join(RESULTS_DIR, row.run_name, "epochs.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        last = d.iloc[-1]
        rows.append(dict(resnet_depth=row.resnet_depth, variant=row.variant,
                          gap=last.train_acc - last.val_acc))
    gap_df = pd.DataFrame(rows)
    agg = gap_df.groupby(["resnet_depth", "variant"]).gap.mean().reset_index()

    fig, ax = plt.subplots(figsize=(5, 4))
    for variant, marker in [("resnet", "o"), ("plain", "s")]:
        sub = agg[agg.variant == variant].sort_values("resnet_depth")
        ax.plot(sub.resnet_depth, sub.gap, marker=marker, label=variant)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("train acc - val acc")
    ax.set_title("Generalization gap vs. depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_generalization_gap.png"), dpi=150)
    plt.close(fig)


def fig_grad_norms(run_names, labels, epoch, title, filename):
    stages = ["stem", "stage1", "stage2", "stage3"]
    x = np.arange(len(stages))
    width = 0.8 / len(run_names)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (run_name, label) in enumerate(zip(run_names, labels)):
        path = os.path.join(RESULTS_DIR, run_name, "grad_norms.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        row = d[d.epoch == epoch]
        if row.empty:
            continue
        vals = [row.iloc[0][s] for s in stages]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * (len(run_names) - 1) / 2)
    ax.set_xticklabels(stages)
    ax.set_ylabel("mean grad L2 norm")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close(fig)


def fig_alpha_sweep(df):
    sub = df[df.variant == "scaled"]
    if sub.empty:
        return
    n = sub.depth_n.iloc[0]
    resnet_pt = df[(df.variant == "resnet") & (df.depth_n == n)]
    plain_pt = df[(df.variant == "plain") & (df.depth_n == n)]
    pts = pd.concat([sub[["alpha", "test_acc"]], plain_pt[["alpha", "test_acc"]],
                      resnet_pt[["alpha", "test_acc"]]])
    pts = pts.groupby("alpha").test_acc.mean().reset_index().sort_values("alpha")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pts.alpha, pts.test_acc * 100, marker="o")
    ax.set_xlabel(r"$\alpha$  (block output = F(x) + $\alpha$x)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Effect of residual scale (depth n={n})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_alpha_sweep.png"), dpi=150)
    plt.close(fig)


def main():
    df = load_summary()
    agg = fig_accuracy_vs_depth(df)
    print(agg)

    deepest_n = df.depth_n.max()
    resnet_run = df[(df.variant == "resnet") & (df.depth_n == deepest_n)].run_name.iloc[0]
    plain_run = df[(df.variant == "plain") & (df.depth_n == deepest_n)].run_name.iloc[0]

    fig_loss_curves([resnet_run, plain_run], ["ResNet", "Plain"],
                     f"Loss curves, depth n={deepest_n}", "fig2_loss_curves.png")
    fig_generalization_gap(df)
    fig_grad_norms([resnet_run, plain_run], ["ResNet", "Plain"], epoch=1,
                    title=f"Gradient norms by stage, epoch 1 (n={deepest_n})",
                    filename="fig4_grad_norms_epoch1.png")
    fig_alpha_sweep(df)

    print("done, figures in", FIG_DIR)


if __name__ == "__main__":
    main()
