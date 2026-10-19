"""Generate report figures from saved metrics/checkpoints: training curves, confusion matrices, Grad-CAM."""
import json
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from . import data as D
from .evaluate import load_model
from .train import ROOT

FIG = ROOT / "reports" / "figures"; MET = ROOT / "reports" / "metrics"
FINAL = "fix_size224_pairw2"


def curves():
    names = ["baseline", "frozen", "finetune", FINAL]
    fig, ax = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7), sharey="row")
    for j, n in enumerate(names):
        h = json.load(open(MET / f"{n}_train.json"))["history"]
        e = [r["epoch"] for r in h]
        ax[0, j].plot(e, [r["train_loss"] for r in h], label="train"); ax[0, j].plot(e, [r["val_loss"] for r in h], label="val")
        ax[0, j].set_title(n + " - loss"); ax[0, j].legend()
        ax[1, j].plot(e, [r["train_acc"] for r in h], label="train"); ax[1, j].plot(e, [r["val_acc"] for r in h], label="val")
        ax[1, j].set_title(n + " - accuracy"); ax[1, j].set_xlabel("epoch"); ax[1, j].set_ylim(0.6, 1.0)
    plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=130); plt.close()


def confusions():
    for n, title in [("finetune", "fine-tuned (before fix)"), (FINAL, "fine-tuned + fix (after)")]:
        r = json.load(open(MET / f"{n}_test.json"))
        cm = np.array(r["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True)
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=r["classes"], yticklabels=r["classes"], cbar_kws={"label": "row-normalised"})
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"Test confusion matrix - {title}\nacc={r['test_acc']:.4f}")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(FIG / f"confusion_matrix_{'before' if n == 'finetune' else 'after'}.png", dpi=130); plt.close()


def gradcam(model, x, target=None):
    """Grad-CAM on resnet18 layer4. x: 1x3xHxW normalized. Returns (cam HxW in [0,1], predicted class)."""
    feats, grads = [], []
    h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    out = model(x)
    c = int(out.argmax(1)) if target is None else target
    model.zero_grad(); out[0, c].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = F.relu((w * feats[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam.detach().cpu().numpy(), c


def gradcam_figure():
    model, size, cls = load_model(FINAL)
    Xte, yte, _ = D.load_arrays()["test"]
    probs = np.load(MET / f"{FINAL}_test_probs.npy"); pred = probs.argmax(1)
    a, b = cls.index("AnnualCrop"), cls.index("PermanentCrop")
    rng = np.random.RandomState(0)
    pick = []  # 2 correct + 2 wrong for each of the two classes of the confused pair
    for c in (a, b):
        ok = np.where((yte == c) & (pred == c))[0]; bad = np.where((yte == c) & (pred != c))[0]
        pick += [(i, "correct") for i in rng.choice(ok, 2, replace=False)] + [(i, "WRONG") for i in bad[:2]]
    pick += [(rng.choice(np.where((yte == c) & (pred == c))[0]), "correct") for c in range(10) if c not in (a, b)][:4]
    fig, ax = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 5.2))
    for k, (i, tag) in enumerate(pick):
        x = D.to_tensor(Xte[i:i + 1], "cuda"); x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        cam, c = gradcam(model, x.requires_grad_(True))
        img = Xte[i]
        ax[0, k].imshow(img); ax[0, k].set_title(f"true: {cls[yte[i]][:11]}\npred: {cls[c][:11]}", fontsize=8,
                                                   color="red" if tag == "WRONG" else "black")
        ax[1, k].imshow(img); ax[1, k].imshow(F.interpolate(torch.tensor(cam)[None, None], size=img.shape[:2], mode="bilinear")[0, 0], cmap="jet", alpha=0.45)
        ax[0, k].axis("off"); ax[1, k].axis("off")
    plt.suptitle("Grad-CAM (resnet18 layer4, final model): image / heat-map; red titles = misclassified", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "gradcam.png", dpi=130); plt.close()


def model_comparison():
    names = ["baseline", "frozen", "finetune", FINAL]
    acc = [json.load(open(MET / f"{n}_test.json"))["test_acc"] for n in names]
    f1 = [json.load(open(MET / f"{n}_test.json"))["test_macro_f1"] for n in names]
    x = np.arange(4); plt.figure(figsize=(7, 4))
    plt.bar(x - .2, acc, .4, label="accuracy"); plt.bar(x + .2, f1, .4, label="macro-F1")
    plt.xticks(x, ["baseline CNN", "frozen R18", "fine-tuned R18", "fine-tuned+fix"]); plt.ylim(0.9, 1.0); plt.legend()
    plt.title("Test-set results (each model evaluated once)"); plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=130); plt.close()


if __name__ == "__main__":
    curves(); confusions(); gradcam_figure(); model_comparison(); print("figures ok")
