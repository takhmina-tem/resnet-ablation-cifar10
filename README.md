# Ablation of residual connections on CIFAR-10

Does the benefit of a residual connection come from easier optimization or
from better generalization? This compares CIFAR-style ResNets against
exactly parameter-matched plain networks (same depth, same width, same
budget, shortcut removed) at 20, 32 and 56 layers, and sweeps a shortcut
scale `alpha` in `F(x) + alpha*x` between the two.

Two measurements do most of the work. Fit to the training set is evaluated
separately, in eval mode with augmentation off, rather than being read off
the running training accuracy -- otherwise "optimization" and
"generalization" cannot be told apart. And test accuracy is then plotted
against that fit across every run, which asks whether the architecture
still predicts anything once fit is held fixed. It does not.

## Layout

- `models/resnet_cifar.py` - block and network. `use_shortcut=False` gives the
  plain network, `alpha` scales the shortcut. Shortcuts are He et al.'s
  option A (subsample + zero-pad, no parameters), so plain and residual
  networks have identical parameter counts and identical initialisation for
  a given seed.
- `data/dataset.py` - CIFAR-10, stratified 90/10 train/val split, plus an
  unaugmented loader over the training images for the fit measurement.
- `train.py` - one run. Writes `epochs.csv` (per-epoch losses/accuracies),
  `grad_norms.csv` (per-block gradient and weight norms at 3 epochs),
  `steps.csv` (per-minibatch loss for the first 3 epochs), the best
  checkpoint by validation accuracy, and a row in `results/summary.csv`.
- `run_sweep.py` - the experiment matrix. Resumable: a finished run is
  skipped, but only if it was finished at the same epoch count.
- `analysis/plot_results.py` - figures into `figures/`, plus the tables the
  report `\input`s.
- `analysis/init_signal.py` - forward and backward signal in freshly
  initialised networks at all three depths. No training, runs on CPU in
  about ninety seconds.
- `analysis/inspect_data.py` - what augmentation does to an image, and the
  class balance of the training set.
- `analysis/summarize.py` - prints every number the report quotes.
- `report/` - the report, in the course's NeurIPS template.

## Running

```
pip install -r requirements.txt

python run_sweep.py --preset smoke      # writes to results_smoke/
python run_sweep.py --preset full       # the real thing, writes to results/

python analysis/plot_results.py
python analysis/init_signal.py
python analysis/inspect_data.py
python analysis/summarize.py
```

The full preset is 15 runs of 30 epochs: 3 depths x {resnet, plain} x 2
seeds, plus alpha in {0.25, 0.5, 0.75} at 56 layers. The alpha=0 and
alpha=1 ends of that sweep are the plain and residual runs, which with
option A shortcuts are bit-for-bit the same networks, so they are not
retrained.

Runs are ordered so that stopping early still leaves something complete.
The first nine cover every depth and the whole alpha sweep at one seed,
which is the entire study as a set of point estimates; the last six repeat
the depth comparison at a second seed purely to put error bars on it. The
analysis script handles a partially finished sweep, so it is safe to stop
whenever and plot what exists.

Budget roughly 25-30 min per 56-layer run and 10-15 min per 20-layer run at
30 epochs on a T4, so the first nine runs are on the order of 4 hours and
the whole thing 6 or more. The smoke preset is 3 short runs, a couple of
minutes on a GPU and closer to fifteen on a laptop CPU. `--epochs 20` cuts
the full sweep by a third.

Passing a results directory to `plot_results.py` also redirects the figures
(`figures/results_smoke/`), so a smoke sweep cannot overwrite the ones the
report includes.

## Training setup

SGD, momentum 0.9, Nesterov, lr 0.1 cosine-annealed over 30 epochs, weight
decay 5e-4, batch size 128, random crop and horizontal flip. This keeps the
optimizer, learning rate and batch size of the original CIFAR ResNet
experiments but uses a much shorter schedule (they train ~164 epochs with
step decay) and a larger weight decay, to fit the compute budget. Both arms
are trained under identical settings; the shortcut is the only difference.

## Note on the committed results

The `lr` column in the committed `epochs.csv` files was written after the
scheduler had already stepped, so it records the rate the *next* epoch
used. Nothing in the analysis or the report reads that column. It is fixed
in `train.py`, so a fresh run logs the rate the epoch was actually trained
at; the committed CSVs are left as they were produced.
