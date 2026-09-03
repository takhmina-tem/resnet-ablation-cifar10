# ResNet ablation on CIFAR-10

Controlled comparison of a CIFAR-style ResNet against a matched plain CNN
(same depth/width, same training budget, skip connections removed), at
three depths (n=3,5,9 -> 20/32/56 layers), plus a sweep over a residual
scale alpha where the block output is `F(x) + alpha*x` instead of `F(x) + x`.

## Layout

- `models/resnet_cifar.py` - the block and network definition. `use_shortcut=False`
  gives the plain network; `alpha` controls the residual scale.
- `data/dataset.py` - CIFAR-10 loading, stratified 90/10 train/val split.
- `train.py` - trains one (depth, variant, alpha, seed) config, logs per-epoch
  train/val loss+acc to `results/<run>/epochs.csv`, logs per-stage gradient
  norms at a few epochs to `results/<run>/grad_norms.csv`, saves the best
  checkpoint by val accuracy, appends a row to `results/summary.csv`.
- `run_sweep.py` - runs the whole experiment matrix. Resumable: any run with
  a `DONE` marker in its results folder is skipped, so a Colab disconnect
  doesn't cost you the finished runs.
- `analysis/plot_results.py` - reads `results/summary.csv` and produces the
  figures in `figures/`.
- `report/` - NeurIPS-style report template with the sections drafted.

## Running

```
pip install -r requirements.txt

# sanity check first, ~2 min
python run_sweep.py --preset smoke

# the real thing
python run_sweep.py --preset full

python analysis/plot_results.py
```

`run_sweep.py --preset full` runs 15 configs at 30 epochs each:
3 depths x {resnet, plain} x 2 seeds, plus alpha in {0.25, 0.5, 0.75} at the
deepest depth (alpha=0 and alpha=1 there are already the plain/resnet runs).
On a T4 this is roughly 2-3 hours total; lower `run_sweep.py`'s `SEEDS` to
`[0]` or drop the n=9 depth if you're short on time.

Training uses SGD (lr=0.1, momentum=0.9, weight decay 5e-4, nesterov) with
cosine annealing over the run, batch size 128, standard crop+flip
augmentation - this is the recipe He et al. use for the CIFAR-10 ResNet
experiments, not the AdamW recipe from the CNN-vs-ViT project, since here
the training dynamics themselves (not just final accuracy) are part of what
is being measured.
