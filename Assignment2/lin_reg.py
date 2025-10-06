"""
author:-aam35
"""
import os
os.environ["MPLBACKEND"] = "Agg"              # no GUI windows
os.environ["OMP_NUM_THREADS"] = "1"           # Apple Accelerate threads
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"    # TensorFlow intra-op
os.environ["TF_NUM_INTEROP_THREADS"] = "1"    # TensorFlow inter-op

import csv, time, argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")                         # reinforce headless mode
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def name_to_seed(name: str) -> int:
    return int(sum(ord(c) for c in name) * 2654435761 % 2_147_483_647)

def choose_device(prefer="auto"):
    if prefer == "auto":
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    return prefer


# -----------------------------
# Data generation
# -----------------------------
def make_data(n=10_000, noise_level=0.5, noise_type="gaussian", seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(n, 1)).astype(np.float32)
    true_y = 3.0 * x + 2.0
    if noise_type == "gaussian":
        eps = rng.normal(0.0, noise_level, size=(n, 1)).astype(np.float32)
    elif noise_type == "uniform":
        eps = rng.uniform(-noise_level, noise_level, size=(n, 1)).astype(np.float32)
    elif noise_type == "saltpepper":
        eps = rng.choice([0.0, noise_level, -noise_level],
                         p=[0.9, 0.05, 0.05], size=(n, 1)).astype(np.float32)
    else:
        eps = np.zeros_like(true_y, dtype=np.float32)
    return x, true_y + eps


# -----------------------------
# Loss functions
# -----------------------------
def l2_loss(y, yhat):  return tf.reduce_mean(tf.square(y - yhat))
def l1_loss(y, yhat):  return tf.reduce_mean(tf.abs(y - yhat))

def huber_loss(y, yhat, delta=1.0):
    e = tf.abs(y - yhat)
    quad = tf.minimum(e, delta)
    lin = e - quad
    return tf.reduce_mean(0.5 * tf.square(quad) + delta * lin)

def hybrid_loss(y, yhat, lam=0.5):
    return lam * l1_loss(y, yhat) + (1.0 - lam) * l2_loss(y, yhat)

def get_loss_fn(name, hybrid_lambda=0.5, huber_delta=1.0):
    name = name.lower()
    if name in ["mse", "l2"]:
        return lambda y, yh: l2_loss(y, yh)
    if name in ["mae", "l1"]:
        return lambda y, yh: l1_loss(y, yh)
    if name == "huber":
        return lambda y, yh: huber_loss(y, yh, delta=huber_delta)
    if name in ["hybrid", "l1+l2", "l1l2"]:
        return lambda y, yh: hybrid_loss(y, yh, lam=hybrid_lambda)
    raise ValueError(f"Unknown loss: {name}")


# -----------------------------
# Model helpers (no tf.function)
# -----------------------------
def forward(x, W, B):
    return tf.matmul(x, W) + B

def add_weight_noise(W, B, level=0.0, rng=None):
    if level <= 0.0: return W, B
    if rng is None:  rng = np.random.default_rng()
    nW = W + tf.constant(rng.normal(0, level, W.shape), dtype=W.dtype)
    nB = B + tf.constant(rng.normal(0, level, B.shape), dtype=B.dtype)
    return nW, nB

def noisy_lr(lr, level=0.0, rng=None):
    if level <= 0.0: return lr
    if rng is None:  rng = np.random.default_rng()
    jitter = 1.0 + rng.normal(0.0, level)
    return max(lr * jitter, 1e-8)


def one_epoch_train(x, y, W, B, lr, loss_fn,
                    weight_noise_level=0.0, lr_noise_level=0.0,
                    rng=None, device="/CPU:0"):
    t0 = time.time()
    with tf.device(device):
        Wn, Bn = add_weight_noise(W, B, level=weight_noise_level, rng=rng)
        with tf.GradientTape() as tape:
            yhat = forward(x, Wn, Bn)
            loss = loss_fn(y, yhat)
        dW, dB = tape.gradient(loss, [Wn, Bn])
        lr_eff = noisy_lr(lr, level=lr_noise_level, rng=rng)
        W.assign_sub(lr_eff * dW)
        B.assign_sub(lr_eff * dB)
    return float(loss.numpy()), (time.time() - t0)

def eval_loss(x, y, W, B, loss_fn, device="/CPU:0"):
    with tf.device(device):
        return float(loss_fn(y, forward(x, W, B)).numpy())


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", type=str, default="mse")
    ap.add_argument("--hybrid_lambda", type=float, default=0.5)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--init_w", type=float, default=0.0)
    ap.add_argument("--init_b", type=float, default=0.0)
    ap.add_argument("--data_size", type=int, default=10_000)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--noise_type", type=str, default="gaussian")
    ap.add_argument("--noise_level", type=float, default=0.5)
    ap.add_argument("--weight_noise_level", type=float, default=0.0)
    ap.add_argument("--lr_noise_level", type=float, default=0.0)
    ap.add_argument("--noise_every", type=int, default=1)
    ap.add_argument("--seed_name", type=str, default="Misan")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--outdir", type=str, default="linreg_outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed = name_to_seed(args.seed_name)
    tf.random.set_seed(seed); np.random.seed(seed)
    device = choose_device(args.device)

    x_all, y_all = make_data(args.data_size, args.noise_level,
                             args.noise_type, seed=seed)
    idx = np.arange(len(x_all)); np.random.shuffle(idx)
    x_all, y_all = x_all[idx], y_all[idx]
    n_val = int(len(x_all) * args.val_split)
    x_val, y_val = x_all[:n_val], y_all[:n_val]
    x_tr,  y_tr  = x_all[n_val:], y_all[n_val:]
    x_tr, y_tr = map(tf.constant, [x_tr, y_tr])
    x_val, y_val = map(tf.constant, [x_val, y_val])

    W = tf.Variable([[args.init_w]], dtype=tf.float32)
    B = tf.Variable([[args.init_b]], dtype=tf.float32)
    loss_fn = get_loss_fn(args.loss, args.hybrid_lambda, args.huber_delta)
    lr, best_val, wait = args.lr, float("inf"), 0

    log_csv = os.path.join(args.outdir, "linreg_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "W", "b", "time"])
        for ep in range(1, args.epochs + 1):
            rng = np.random.default_rng(seed + ep)
            w_noise = args.weight_noise_level if ep % args.noise_every == 0 else 0.0
            lr_noise = args.lr_noise_level if ep % args.noise_every == 0 else 0.0
            tr_loss, t_ep = one_epoch_train(x_tr, y_tr, W, B, lr, loss_fn,
                                            w_noise, lr_noise, rng, device)
            val_loss = eval_loss(x_val, y_val, W, B, loss_fn, device)
            if val_loss < best_val - 1e-12:
                best_val, wait = val_loss, 0
            else:
                wait += 1
                if wait >= args.patience:
                    lr = max(lr * 0.5, 1e-6); wait = 0
            writer.writerow([ep, tr_loss, val_loss, lr,
                             float(W.numpy()), float(B.numpy()), t_ep])
            if ep % 10 == 0 or ep == 1 or ep == args.epochs:
                print(f"[{args.loss.upper()}] ep {ep:04d} | tr {tr_loss:.6f} | "
                      f"val {val_loss:.6f} | lr {lr:.5f} | "
                      f"W {float(W.numpy()):+.4f} | b {float(B.numpy()):+.4f}")

    # ---- Plots ----
    data = np.genfromtxt(log_csv, delimiter=",", names=True)
    plt.figure()
    plt.plot(data["epoch"], data["train_loss"], label="train")
    plt.plot(data["epoch"], data["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Loss curves ({args.loss})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curves.png"), dpi=160)

    x_line = np.linspace(-5, 5, 200, dtype=np.float32).reshape(-1,1)
    y_true = 3.0 * x_line + 2.0
    y_pred = x_line @ W.numpy() + B.numpy()
    plt.figure()
    plt.scatter(x_all[:2000], y_all[:2000], s=6, alpha=0.35, label="data")
    plt.plot(x_line, y_true, "g--", label="true: 3x+2")
    plt.plot(x_line, y_pred, "r", label=f"fit ({args.loss})")
    plt.legend(); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fit.png"), dpi=160)

    print(f"\n✅ Saved results in {args.outdir}")
    print(f"Final W={float(W.numpy()):.4f}, b={float(B.numpy()):.4f}, best val loss={best_val:.6f}")


if __name__ == "__main__":
    main()
