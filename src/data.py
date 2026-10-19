"""EuroSAT data loading: stratified 70/15/15 split, in-memory uint8 cache, GPU augmentation."""
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "EuroSAT"
SPLIT_CSV = ROOT / "data" / "splits.csv"
CACHE = ROOT / "data" / "cache_uint8.npz"
SEED = 42
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def classes(img_dir=IMG_DIR):
    return sorted(p.name for p in Path(img_dir).iterdir() if p.is_dir())


def make_splits(img_dir=IMG_DIR, seed=SEED, out=SPLIT_CSV):
    """Stratified 70/15/15 split with a fixed seed; saved to CSV for reproducibility."""
    cls = classes(img_dir)
    rows = [(str(p.relative_to(img_dir)).replace("\\", "/"), c, i)
            for i, c in enumerate(cls) for p in sorted((Path(img_dir) / c).glob("*.jpg"))]
    df = pd.DataFrame(rows, columns=["path", "class", "label"])
    tr, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    df["split"] = ""
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    df.to_csv(out, index=False)
    return df


def load_arrays(img_dir=IMG_DIR, split_csv=SPLIT_CSV):
    """Returns dict split -> (uint8 NHWC array, labels, paths)."""
    df = pd.read_csv(split_csv) if Path(split_csv).exists() else make_splits(img_dir)
    if CACHE.exists():
        z = np.load(CACHE)
        X = z["X"]
    else:
        X = np.stack([np.asarray(Image.open(Path(img_dir) / p).convert("RGB")) for p in df["path"]])
        np.savez(CACHE, X=X)
    out = {}
    for s in ["train", "val", "test"]:
        m = (df["split"] == s).values
        out[s] = (X[m], df["label"].values[m], df["path"].values[m])
    return out


def to_tensor(xb_uint8, device):
    """uint8 NHWC (numpy/tensor) -> normalized float NCHW on device."""
    x = torch.as_tensor(xb_uint8).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x):
    """Batch GPU augmentation: random flips + random 90° rotations (aerial imagery is orientation-free)
    plus mild brightness/contrast jitter."""
    n = x.shape[0]
    f = torch.rand(n, device=x.device) < 0.5
    x = torch.where(f.view(-1, 1, 1, 1), x.flip(3), x)
    f = torch.rand(n, device=x.device) < 0.5
    x = torch.where(f.view(-1, 1, 1, 1), x.flip(2), x)
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, (2, 3))
    b = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    return x * c + b


if __name__ == "__main__":
    df = make_splits()
    print(df.groupby(["split", "class"]).size().unstack(0))
    print(df["split"].value_counts())
