""" 
author:-aam35
"""
# ---- i add to implement safety cause it kept on crushing) ----
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import time
import csv
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional baselines / clustering
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

NUM_CLASSES = 10
IMG_H, IMG_W = 28, 28
IMG_SIZE_FLAT = IMG_H * IMG_W

# --------------------------
# Data loadin and split
# --------------------------
def load_fashion_mnist():
    # Only used to fetch numpy arrays. No Keras models/layers involved.
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    return (x_tr.astype(np.float32), y_tr.astype(np.int32),
            x_te.astype(np.float32), y_te.astype(np.int32))


def preprocess(images):
    # scale to [0,1], flatten to 784
    images = images / 255.0
    return images.reshape(images.shape[0], -1).astype(np.float32)


def make_split(x, y, val_split=0.1, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_split)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def make_dataset(x, y, batch_size, shuffle=True, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds


# --------------------------
# Model / Loss / Metrics
# --------------------------
def logits_fn(X, W, b):
    # X: [B, 784], W: [784,10], b: [10]
    return tf.matmul(X, W) + b

def softmax_cross_entropy(y_true, logits):
    y_onehot = tf.one_hot(y_true, depth=NUM_CLASSES)
    # log-softmax for numerical stability
    log_probs = tf.nn.log_softmax(logits, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(y_onehot * log_probs, axis=1))

def accuracy_from_logits(y_true, logits):
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    return float(tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32)).numpy())

def get_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":
        return tf.optimizers.SGD(learning_rate=lr)
    if name in ["momentum", "sgdm"]:
        return tf.optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "adam":
        return tf.optimizers.Adam(learning_rate=lr)
    if name == "adagrad":
        return tf.optimizers.Adagrad(learning_rate=lr)
    raise ValueError(f"Unknown optimizer {name!r}")

# --------------------------
# Train / Eval loops
# --------------------------
def train_epoch(ds, W, b, opt, device="/CPU:0"):
    t0 = time.time()
    losses, accs = [], []
    with tf.device(device):
        for xb, yb in ds:
            xb = tf.convert_to_tensor(xb)  # [B,784]
            yb = tf.convert_to_tensor(yb)  # [B]
            with tf.GradientTape() as tape:
                logits = logits_fn(xb, W, b)
                loss = softmax_cross_entropy(yb, logits)
            dW, db = tape.gradient(loss, [W, b])
            opt.apply_gradients([(dW, W), (db, b)])
            losses.append(float(loss.numpy()))
            accs.append(accuracy_from_logits(yb, logits))
    return float(np.mean(losses)), float(np.mean(accs)), time.time() - t0


def eval_epoch(ds, W, b, device="/CPU:0"):
    losses, accs = [], []
    with tf.device(device):
        for xb, yb in ds:
            logits = logits_fn(xb, W, b)
            loss = softmax_cross_entropy(yb, logits)
            losses.append(float(loss.numpy()))
            accs.append(accuracy_from_logits(yb, logits))
    return float(np.mean(losses)), float(np.mean(accs))


# --------------------------
# Plots
# --------------------------
def plot_samples(x_orig, y_orig, outdir):
    plt.figure()
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(x_orig[i], cmap="gray")
        plt.axis("off")
        plt.title(int(y_orig[i]))
    plt.suptitle("Fashion-MNIST samples")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "samples.png"), dpi=160)

def plot_weight_templates(W, outdir):
    Wnp = W.numpy()  # (784,10)
    vmax = np.percentile(np.abs(Wnp), 99)
    plt.figure(figsize=(10,7))
    for c in range(NUM_CLASSES):
        plt.subplot(3,4,c+1)
        plt.imshow(Wnp[:,c].reshape(IMG_H,IMG_W), cmap="seismic", vmin=-vmax, vmax=vmax)
        plt.axis("off")
        plt.title(f"class {c}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weight_templates.png"), dpi=160)

def plot_curves(log_csv, outdir, title_tag=""):
    data = np.genfromtxt(log_csv, delimiter=",", names=True)
    plt.figure()
    plt.plot(data["epoch"], data["train_acc"], label="train_acc")
    plt.plot(data["epoch"], data["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(f"Accuracy curves {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curves.png"), dpi=160)

    plt.figure()
    plt.plot(data["epoch"], data["train_loss"], label="train_loss")
    plt.plot(data["epoch"], data["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Loss curves {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=160)

# --------------------------
# Baselines & Clustering 
# --------------------------
def run_baselines(x_tr, y_tr, x_te, y_te, outdir):
    if not HAVE_SK:
        print("[Baselines] scikit-learn not installed; skipping.")
        return
    # Train on 20k for speed
    xs, ys = x_tr[:20000], y_tr[:20000]
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    rf.fit(xs, ys)
    rf_acc = rf.score(x_te, y_te)
    svm = LinearSVC(dual=False, C=1.0, max_iter=3000, random_state=0)
    svm.fit(xs, ys)
    svm_acc = svm.score(x_te, y_te)
    with open(os.path.join(outdir, "classical_baselines.txt"), "w") as f:
        f.write(f"RandomForest test acc: {rf_acc:.4f}\n")
        f.write(f"LinearSVC   test acc: {svm_acc:.4f}\n")
    print(f"[Baselines] RF: {rf_acc:.4f} | LinearSVC: {svm_acc:.4f}")

def cluster_weights(W, outdir):
    if not HAVE_SK:
        print("[Clustering] scikit-learn not installed; skipping.")
        return
    Wc = W.numpy().T  # (10,784)
    emb = TSNE(n_components=2, perplexity=5, random_state=0).fit_transform(Wc)
    labels = KMeans(n_clusters=3, n_init="auto", random_state=0).fit(Wc).labels_
    plt.figure()
    for i in range(Wc.shape[0]):
        plt.scatter(emb[i,0], emb[i,1], s=80)
        plt.text(emb[i,0]+0.5, emb[i,1], str(i))
    plt.title("t-SNE of class weight vectors (0..9)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_weights.png"), dpi=160)
    with open(os.path.join(outdir, "kmeans_weights.txt"), "w") as f:
        f.write("Class -> ClusterID\n")
        for i, lab in enumerate(labels):
            f.write(f"{i} -> {lab}\n")


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", type=str, default="adam",
                    help="adam|sgd|momentum|adagrad")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto", help="auto|/CPU:0|/GPU:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--outdir", type=str, default="logreg_outputs")
    ap.add_argument("--do_baselines", action="store_true")
    ap.add_argument("--do_clusters", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tf.random.set_seed(args.seed); np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    else:
        device = args.device

    # Load
    x_tr_all, y_tr_all, x_te_raw, y_te = load_fashion_mnist()
    # Preview samples (raw)
    plot_samples(x_tr_all, y_tr_all, args.outdir)

    # Split
    x_tr_raw, y_tr, x_val_raw, y_val = make_split(x_tr_all, y_tr_all, args.val_split, seed=args.seed)

    # Preprocess
    x_tr = preprocess(x_tr_raw)
    x_val = preprocess(x_val_raw)
    x_te = preprocess(x_te_raw)

    # Datasets
    ds_tr  = make_dataset(x_tr,  y_tr,  args.batch_size, shuffle=True,  seed=args.seed)
    ds_val = make_dataset(x_val, y_val, args.batch_size, shuffle=False, seed=args.seed)
    ds_te  = make_dataset(x_te,  y_te,  args.batch_size, shuffle=False, seed=args.seed)

    # Params: logits = X W + b
    W = tf.Variable(tf.random.normal([IMG_SIZE_FLAT, NUM_CLASSES], stddev=0.01, seed=args.seed), name="W")
    b = tf.Variable(tf.zeros([NUM_CLASSES]), name="b")

    opt = get_optimizer(args.optimizer, args.lr)

    # Training loop
    log_csv = os.path.join(args.outdir, "logreg_log.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","epoch_time_sec","device","optimizer","lr","batch_size"])

        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc, ep_time = train_epoch(ds_tr, W, b, opt, device=device)
            va_loss, va_acc = eval_epoch(ds_val, W, b, device=device)
            writer.writerow([ep, tr_loss, tr_acc, va_loss, va_acc, ep_time, device, args.optimizer, args.lr, args.batch_size])
            print(f"Epoch {ep:03d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | val_loss {va_loss:.4f} val_acc {va_acc:.4f} | {ep_time:.2f}s on {device}")

    # Final test metrics
    te_loss, te_acc = eval_epoch(ds_te, W, b, device=device)
    print(f"\nTest  | loss {te_loss:.4f} acc {te_acc:.4f}")

    # Confusion matrix
    # Collect predictions on test set
    preds_all, labels_all = [], []
    for xb, yb in ds_te:
        logits = logits_fn(xb, W, b)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
        preds_all.append(preds); labels_all.append(yb.numpy())
    yhat = np.concatenate(preds_all); ytrue = np.concatenate(labels_all)

    try:
        cm = confusion_matrix(ytrue, yhat) if HAVE_SK else None
        if cm is not None:
            np.savetxt(os.path.join(args.outdir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
            print("Saved confusion_matrix.csv")
    except Exception:
        pass

    # Plots
    plot_curves(log_csv, args.outdir, title_tag=f"({args.optimizer}, bs={args.batch_size})")
    plot_weight_templates(W, args.outdir)

    # Optional extras
    if args.do_baselines:
        run_baselines(x_tr, y_tr, x_te, y_te, args.outdir)
    if args.do_clusters:
        cluster_weights(W, args.outdir)

    print(f"\nSaved all outputs in: {args.outdir}")
    print(f"Final params: ||W||={np.linalg.norm(W.numpy()):.3f}  |  Test acc={te_acc:.4f}")


if __name__ == "__main__":
    main()
