# Ablation of residual connections on CIFAR-10

Does the benefit of a residual connection come from easier optimization or
from better generalization? This compares CIFAR-style ResNets against
exactly parameter-matched plain networks (same depth, same width, same
budget, shortcut removed) at 20, 32 and 56 layers, and sweeps a residual
scale `alpha` in `F(x) + alpha*x` between the two.

The key measurement is that fit to the training set is evaluated
separately, in eval mode with augmentation off, rather than being read off
the running training accuracy -- otherwise "optimization" and
"generalization" cannot be told apart.

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
- `analysis/plot_results.py` - figures into `figures/`, plus
  `results/table1_body.tex` with the table rows ready to paste.
- `report/` - the report, in the course's NeurIPS template.

## Running

```
pip install -r requirements.txt

python run_sweep.py --preset smoke      # ~2 min, writes to results_smoke/
python run_sweep.py --preset full       # the real thing, writes to results/
python analysis/plot_results.py
```

The full preset is 15 runs of 30 epochs: 3 depths x {resnet, plain} x 2
seeds, plus alpha in {0.25, 0.5, 0.75} at 56 layers. The alpha=0 and
alpha=1 ends of that sweep are the plain and residual runs, which with
option A shortcuts are bit-for-bit the same networks, so they are not
retrained. Roughly 2-3 hours on a T4. To cut it down, set `SEEDS = [0]` or
drop a depth from `DEPTHS` in `run_sweep.py`.

## Training setup

SGD, momentum 0.9, Nesterov, lr 0.1 cosine-annealed over 30 epochs, weight
decay 5e-4, batch size 128, random crop and horizontal flip. This keeps the
optimizer, learning rate and batch size of the original CIFAR ResNet
experiments but uses a much shorter schedule (they train ~164 epochs with
step decay) and a larger weight decay, to fit the compute budget. Both arms
are trained under identical settings; the shortcut is the only difference.
