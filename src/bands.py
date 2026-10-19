"""13-band EuroSAT (Sentinel-2 L1C) loading aligned with the RGB split; NDVI = (B08-B04)/(B08+B04)."""
import zipfile, io
import numpy as np, pandas as pd, tifffile
from . import data as D

ZIP = r"E:\Projects\EuroSat.zip"
CACHE = D.ROOT / "data" / "cache_13band_uint16.npy"
B04, B08 = 3, 7  # indices in the 13-band order B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12


def build_cache():
    df = pd.read_csv(D.SPLIT_CSV)
    z = zipfile.ZipFile(ZIP)
    X = np.empty((len(df), 13, 64, 64), dtype=np.uint16)
    for i, p in enumerate(df["path"]):
        a = tifffile.imread(io.BytesIO(z.read("EuroSATallBands/" + p.replace(".jpg", ".tif"))))
        X[i] = a.transpose(2, 0, 1) if a.shape[-1] == 13 else a
    np.save(CACHE, X)
    return X


def load_bands():
    """dict split -> (float32 [N,13,64,64] reflectance-ish / 10000, labels)."""
    df = pd.read_csv(D.SPLIT_CSV)
    X = np.load(CACHE, mmap_mode="r") if CACHE.exists() else build_cache()
    return {s: (X, np.where(df["split"].values == s)[0], df["label"].values[df["split"].values == s]) for s in ["train", "val", "test"]}


if __name__ == "__main__":
    X = build_cache(); print(X.shape, X.dtype, X.reshape(len(X), 13, -1).mean((0, 2)).round(0))
