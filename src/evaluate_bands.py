"""Score the band-experiment models on test once each (shares the once-only ledger). Usage: python -m src.evaluate_bands NAME..."""
import json, sys
import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .train_bands import get_data, build_model, predict, VARIANTS
from .evaluate import LEDGER
from .train import ROOT

names = sys.argv[1:]; ledger = json.load(open(LEDGER))
cache = {}
for n in names:
    if n in ledger:
        print("REFUSED", n); continue
    ck = torch.load(ROOT / "models" / f"{n}.pt", map_location="cuda"); v = ck["variant"]
    if v not in cache:
        cache[v] = get_data(v)[0]["test"]
    x, y = cache[v]; m = build_model(VARIANTS[v], pretrained=False).cuda(); m.load_state_dict(ck["state_dict"])
    p = predict(m, x, ck["size"]).argmax(1); y = y.cpu().numpy()
    r = dict(name=n, test_acc=float(accuracy_score(y, p)), test_macro_f1=float(f1_score(y, p, average="macro")),
             confusion_matrix=confusion_matrix(y, p).tolist())
    json.dump(r, open(ROOT / "reports" / "metrics" / f"{n}_test.json", "w"), indent=1)
    ledger[n] = {"test_acc": r["test_acc"], "test_macro_f1": r["test_macro_f1"]}; json.dump(ledger, open(LEDGER, "w"), indent=1)
    print(n, round(r["test_acc"], 4), round(r["test_macro_f1"], 4))
