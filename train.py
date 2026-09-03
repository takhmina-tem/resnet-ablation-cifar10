import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data.dataset import get_dataloaders
from models.resnet_cifar import build_model, count_params

STEP_LOG_EPOCHS = 3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def block_grad_norms(model):
    rows = []
    for i, (stage, block) in enumerate(model.block_modules()):
        gsq, wsq = 0.0, 0.0
        for p in block.parameters():
            if p.dim() != 4:
                continue
            wsq += p.detach().pow(2).sum().item()
            if p.grad is not None:
                gsq += p.grad.detach().pow(2).sum().item()
        rows.append((i, stage, math.sqrt(gsq), math.sqrt(wsq)))
    return rows


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def already_done(run_dir, epochs):
    marker = os.path.join(run_dir, "DONE")
    if not os.path.exists(marker):
        return False
    with open(marker) as f:
        return f.read().strip() == str(epochs)


def run(n, variant, alpha, seed, epochs, lr=0.1, weight_decay=5e-4, batch_size=128,
        data_dir="./data_cache", out_dir="./results", augment=True,
        grad_log_epochs=None, num_workers=2, tag=None):

    set_seed(seed)
    device = get_device()

    run_name = tag or f"n{n}_{variant}_a{alpha:.2f}_s{seed}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if already_done(run_dir, epochs):
        print(f"skipping {run_name}, already done at {epochs} epochs")
        return

    ckpt_path = os.path.join(run_dir, "best.pth")

    train_loader, train_clean_loader, val_loader, test_loader = get_dataloaders(
        data_dir, seed=seed, batch_size=batch_size, augment=augment, num_workers=num_workers)

    model = build_model(n, variant, alpha=alpha).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                 weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if grad_log_epochs is None:
        grad_log_epochs = {1, max(1, epochs // 2), epochs}

    best_val_acc = -1.0
    diverged = False

    f_epoch = open(os.path.join(run_dir, "epochs.csv"), "w", newline="")
    f_grad = open(os.path.join(run_dir, "grad_norms.csv"), "w", newline="")
    f_step = open(os.path.join(run_dir, "steps.csv"), "w", newline="")
    try:
        epoch_writer = csv.writer(f_epoch)
        epoch_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])
        grad_writer = csv.writer(f_grad)
        grad_writer.writerow(["epoch", "block", "stage", "grad_norm", "weight_norm"])
        step_writer = csv.writer(f_step)
        step_writer.writerow(["epoch", "step", "loss"])

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            total_loss, correct, seen = 0.0, 0, 0
            logged_grad = None

            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                if epoch in grad_log_epochs and step == 0:
                    logged_grad = block_grad_norms(model)

                optimizer.step()

                batch_loss = loss.item()
                if epoch <= STEP_LOG_EPOCHS:
                    step_writer.writerow([epoch, step, batch_loss])

                total_loss += batch_loss * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                seen += x.size(0)

            scheduler.step()
            train_loss = total_loss / seen
            train_acc = correct / seen

            if not math.isfinite(train_loss):
                print(f"{run_name}: train loss went non-finite at epoch {epoch}, stopping early")
                diverged = True
                break

            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            dt = time.time() - t0

            epoch_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc,
                                    optimizer.param_groups[0]["lr"], dt])
            f_epoch.flush()
            f_step.flush()

            if logged_grad is not None:
                for block_i, stage, gnorm, wnorm in logged_grad:
                    grad_writer.writerow([epoch, block_i, stage, gnorm, wnorm])
                f_grad.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)

            print(f"{run_name} epoch {epoch}/{epochs} train_acc {train_acc:.4f} val_acc {val_acc:.4f} ({dt:.1f}s)")
    finally:
        f_epoch.close()
        f_grad.close()
        f_step.close()

    nan = float("nan")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        # training set without augmentation, in eval mode: this is the number
        # that says how well the model actually fits the data it trained on
        fit_loss, fit_acc = evaluate(model, train_clean_loader, device, criterion)
        final_val_loss, final_val_acc = evaluate(model, val_loader, device, criterion)
    else:
        test_acc = fit_loss = fit_acc = final_val_loss = final_val_acc = nan

    summary_path = os.path.join(out_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_name", "depth_n", "resnet_depth", "variant", "alpha", "seed",
                        "n_params", "epochs", "best_val_acc", "test_acc",
                        "fit_acc", "fit_loss", "final_val_acc", "final_val_loss", "diverged"])
        w.writerow([run_name, n, 6 * n + 2, variant, alpha, seed, n_params, epochs,
                    best_val_acc, test_acc, fit_acc, fit_loss, final_val_acc, final_val_loss, diverged])

    if not diverged:
        with open(os.path.join(run_dir, "DONE"), "w") as f:
            f.write(str(epochs))

    print(f"{run_name} done, test_acc {test_acc}, train fit {fit_acc}")
    return test_acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--variant", choices=["resnet", "plain", "scaled"], required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--data_dir", default="./data_cache")
    p.add_argument("--out_dir", default="./results")
    args = p.parse_args()
    run(args.n, args.variant, args.alpha, args.seed, args.epochs, lr=args.lr,
        batch_size=args.batch_size, data_dir=args.data_dir, out_dir=args.out_dir)
