# ResNet ablation on CIFAR-10

Project for 30562. ResNets vs the same networks with the shortcut removed, at
20, 32 and 56 layers, plus a sweep over alpha in `F(x) + alpha*x`.

Shortcuts are option A (subsample + zero-pad) so both arms have the same number
of parameters. Fit accuracy is measured separately with augmentation off.

## Running

```
pip install -r requirements.txt

python run_sweep.py --preset smoke     # quick check
python run_sweep.py --preset full      # 15 runs, ~6h on a T4

python analysis/plot_results.py
python analysis/init_signal.py
```

`train.py` runs one config and writes its CSVs into `results/`. `run_sweep.py`
loops over the matrix and skips anything already finished. `plot_results.py`
reads `results/` and writes the figures and tables. `init_signal.py` measures
forward and backward signal at initialisation, no training, runs on CPU.

Training: SGD, momentum 0.9, Nesterov, lr 0.1 cosine over 30 epochs, wd 5e-4,
batch 128, random crop + flip. Same for both arms.

Note: the `lr` column in the saved `epochs.csv` files is off by one epoch.
Nothing uses it.
