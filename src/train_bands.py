"""Stretch experiment: does NDVI / multispectral input help? Fine-tunes ResNet-18 (same recipe as 'finetune',
128 px) on channel sets built from the 13-band GeoTIFFs. Validation only; test via src.evaluate_bands.

Variants: rgb_tif (B04,B03,B02 control), rgb_ndvi, all13, all13_ndvi.
Usage: python -m src.train_bands --variant rgb_ndvi [--seed 42]
"""
import argparse, json, time, copy, os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, mlflow
from sklearn.metrics import f1_score
import torchvision
from . import bands as B, data as D
from .train import hardware, ROOT

VARIANTS = {"rgb_tif": 3, "rgb_ndvi": 4, "all13": 13, "all13_ndvi": 14}


def channels(X, variant):
    """X: uint16 [N,13,H,W] -> float32 [N,C,H,W] (unstandardised)."""
    X = X.astype(np.float32)
    ndvi = (X[:, 7:8] - X[:, 3:4]) / (X[:, 7:8] + X[:, 3:4] + 1e-6)
    if variant == "rgb_tif":
        return X[:, [3, 2, 1]]
    if variant == "rgb_ndvi":
        return np.concatenate([X[:, [3, 2, 1]], ndvi * 1000], 1)  # scaled to reflectance-like magnitude
    if variant == "all13":
        return X
    return np.concatenate([X, ndvi * 1000], 1)


def get_data(variant, dev="cuda"):
    d = B.load_bands(); out = {}; stats = None
    for s in ["train", "val", "test"]:
        X, idx, y = d[s]
        x = torch.tensor(channels(np.asarray(X[idx]), variant))
        if s == "train":
            stats = (x.mean((0, 2, 3), keepdim=True), x.std((0, 2, 3), keepdim=True))
        out[s] = (((x - stats[0]) / stats[1]).half().to(dev), torch.tensor(y).to(dev))
    return out, stats


def build_model(c, pretrained=True):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if c != 3:
        w = m.conv1.weight.data  # [64,3,7,7]; map R,G,B weights to first 3 channels; others = mean filter
        new = nn.Conv2d(c, 64, 7, 2, 3, bias=False)
        extra = w.mean(1, keepdim=True).repeat(1, c - 3, 1, 1) if pretrained else None
        if pretrained:
            new.weight.data = torch.cat([w, extra], 1) * (3.0 / c)
        m.conv1 = new
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))
    return m


@torch.no_grad()
def predict(model, x, size=128, bs=512):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        xb = F.interpolate(x[i:i + bs].float(), size=size, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(xb).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def aug(x):
    n = x.shape[0]
    x = torch.where((torch.rand(n, device=x.device) < .5).view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where((torch.rand(n, device=x.device) < .5).view(-1, 1, 1, 1), x.flip(2), x)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS)); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args(); name = f"bands_{a.variant}_s{a.seed}"
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"; torch.backends.cudnn.benchmark = True
    data, stats = get_data(a.variant); (xtr, ytr), (xva, yva) = data["train"], data["val"]
    c = VARIANTS[a.variant]; model = build_model(c).to(dev).to(memory_format=torch.channels_last)
    bs, lr = 128, 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * (len(xtr) // bs), pct_start=.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri()); mlflow.set_experiment("eurosat-bands")
    yv = yva.cpu().numpy(); hist = []; best = (-1, None, 0)
    log = open(ROOT / "logs" / f"{name}.log", "w")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(vars(a)); t0 = time.time()
        for e in range(a.epochs):
            model.train(); perm = torch.randperm(len(xtr), device=dev); tl = n = 0
            for i in range(0, len(perm) - bs + 1, bs):
                idx = perm[i:i + bs]; x = aug(xtr[idx].float()); y = ytr[idx]
                x = F.interpolate(x, size=a.size, mode="bilinear", align_corners=False).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                loss = crit(out.float(), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tl += loss.item() * len(y); n += len(y)
            pv = predict(model, xva, a.size); acc = float((pv.argmax(1) == yv).mean())
            f1 = float(f1_score(yv, pv.argmax(1), average="macro"))
            h = dict(epoch=e + 1, train_loss=tl / n, val_acc=acc, val_macro_f1=f1, elapsed_s=time.time() - t0)
            hist.append(h); mlflow.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=e + 1)
            log.write(json.dumps(h) + "\n"); log.flush(); print(json.dumps(h), flush=True)
            if acc > best[0]:
                best = (acc, copy.deepcopy(model.state_dict()), e + 1)
        t = time.time() - t0; model.load_state_dict(best[1]); pv = predict(model, xva, a.size)
        res = dict(name=name, variant=a.variant, seed=a.seed, channels=c, best_epoch=best[2], val_acc=float((pv.argmax(1) == yv).mean()),
                   val_macro_f1=float(f1_score(yv, pv.argmax(1), average="macro")), train_time_s=t, history=hist, hardware=hardware())
        torch.save({"state_dict": best[1], "variant": a.variant, "size": a.size, "mean": stats[0], "std": stats[1]}, ROOT / "models" / f"{name}.pt")
        json.dump(res, open(ROOT / "reports" / "metrics" / f"{name}_train.json", "w"), indent=1)
        mlflow.log_metrics({"best_val_acc": res["val_acc"], "train_time_s": t})
        print(f"DONE {name}: val_acc={res['val_acc']:.4f} f1={res['val_macro_f1']:.4f} t={t:.0f}s")


if __name__ == "__main__":
    main()
