"""Train one stage. Model selection uses the VALIDATION set only; test is never touched here.

Usage: python -m src.train --stage baseline|frozen|finetune [--name NAME] [--size 128]
                           [--pair A B --pair-weight 2]
"""
import argparse, json, time, platform, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow, psutil
from sklearn.metrics import f1_score
from . import data as D, models as M

ROOT = D.ROOT
torch.backends.cudnn.benchmark = True


def hardware():
    return {"gpu": torch.cuda.get_device_name(0),
            "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "cpu": platform.processor(), "cpu_count": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(), "os": platform.platform()}


@torch.no_grad()
def predict(model, X, size, dev, bs=512, tta=False):
    """Softmax probabilities for a uint8 NHWC array."""
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = D.to_tensor(X[i:i + bs], dev)
        if size != x.shape[-1]:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(3)).float().softmax(1) + model(x.flip(2)).float().softmax(1)
                     + model(torch.rot90(x, 1, (2, 3))).float().softmax(1)) / 4
        outs.append(p.cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["baseline", "frozen", "finetune"])
    ap.add_argument("--name"); ap.add_argument("--epochs", type=int)
    ap.add_argument("--size", type=int); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair", nargs=2, help="class names whose loss weight is boosted")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--smoothing", type=float, default=0.1)
    a = ap.parse_args()
    name = a.name or a.stage
    defaults = {"baseline": (30, 64, 2e-3), "frozen": (15, 128, 1e-3), "finetune": (15, 128, 3e-4)}
    ep, size, lr = defaults[a.stage]
    ep = a.epochs or ep; size = a.size or size; lr = a.lr or lr
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    arr = D.load_arrays()
    Xtr, ytr, _ = arr["train"]; Xva, yva, _ = arr["val"]
    cls = D.classes()
    w = torch.ones(10)
    if a.pair:
        for c in a.pair:
            w[cls.index(c)] = a.pair_weight
    model = M.build(a.stage).to(dev).to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4 if a.stage != "baseline" else 5e-4)
    steps = ep * (len(Xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=a.smoothing)
    Xtr_t = torch.as_tensor(Xtr); ytr_t = torch.as_tensor(ytr)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri()); mlflow.set_experiment("eurosat")
    hist = []; best = (-1, None, -1)
    log = open(ROOT / "logs" / f"{name}.log", "w")

    def P(*s):
        m = " ".join(str(x) for x in s); print(m, flush=True); log.write(m + "\n"); log.flush()

    with mlflow.start_run(run_name=name):
        mlflow.log_params({**vars(a), "epochs": ep, "size": size, "lr": lr,
                           "n_params_trainable": sum(p.numel() for p in params)})
        t0 = time.time()
        for e in range(ep):
            M.set_train_mode(model, a.stage)
            perm = torch.randperm(len(Xtr_t)); tl = 0; tc = 0; n = 0
            for i in range(0, len(perm) - a.bs + 1, a.bs):
                idx = perm[i:i + a.bs]
                x = D.augment(D.to_tensor(Xtr_t[idx], dev)); y = ytr_t[idx].to(dev)
                if size != x.shape[-1]:
                    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                loss = crit(out.float(), y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
            pv = predict(model, Xva, size, dev)
            va_acc = float((pv.argmax(1) == yva).mean())
            va_f1 = float(f1_score(yva, pv.argmax(1), average="macro"))
            va_loss = float(F.nll_loss(torch.log(torch.tensor(pv) + 1e-9), torch.tensor(yva)))
            h = dict(epoch=e + 1, train_loss=tl / n, train_acc=tc / n, val_loss=va_loss, val_acc=va_acc,
                     val_macro_f1=va_f1, elapsed_s=time.time() - t0)
            hist.append(h); mlflow.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=e + 1)
            P(json.dumps(h))
            if va_acc > best[0]:
                best = (va_acc, copy.deepcopy(model.state_dict()), e + 1)
        train_s = time.time() - t0
        model.load_state_dict(best[1])
        pv = predict(model, Xva, size, dev)
        res = dict(name=name, stage=a.stage, args=vars(a), epochs=ep, size=size, lr=lr, best_epoch=best[2],
                   val_acc=float((pv.argmax(1) == yva).mean()),
                   val_macro_f1=float(f1_score(yva, pv.argmax(1), average="macro")),
                   train_time_s=train_s, history=hist, hardware=hardware(),
                   trainable_params=sum(p.numel() for p in params),
                   total_params=sum(p.numel() for p in model.parameters()))
        (ROOT / "models").mkdir(exist_ok=True)
        torch.save({"state_dict": best[1], "stage": a.stage, "size": size, "classes": cls},
                   ROOT / "models" / f"{name}.pt")
        np.save(ROOT / "reports" / "metrics" / f"{name}_val_probs.npy", pv)
        json.dump(res, open(ROOT / "reports" / "metrics" / f"{name}_train.json", "w"), indent=1)
        mlflow.log_metrics({"best_val_acc": res["val_acc"], "best_val_macro_f1": res["val_macro_f1"],
                            "train_time_s": train_s})
        mlflow.log_artifact(str(ROOT / "reports" / "metrics" / f"{name}_train.json"))
        P(f"DONE {name}: best epoch {best[2]} val_acc={res['val_acc']:.4f} f1={res['val_macro_f1']:.4f} "
          f"train_time={train_s:.0f}s")


if __name__ == "__main__":
    main()
