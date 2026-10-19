"""Evaluate FINAL models on the held-out test set exactly once each.

A ledger (reports/metrics/test_ledger.json) records which models were already scored on test; re-running is
refused so that no choice can be tuned against the test set.
Usage: python -m src.evaluate NAME [NAME ...]
"""
import json, sys
import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from . import data as D, models as M
from .train import predict, ROOT

LEDGER = ROOT / "reports" / "metrics" / "test_ledger.json"


def load_model(name, dev="cuda"):
    ck = torch.load(ROOT / "models" / f"{name}.pt", map_location=dev)
    m = M.build(ck["stage"], pretrained=False).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["size"], ck["classes"]


def main(names):
    ledger = json.load(open(LEDGER)) if LEDGER.exists() else {}
    Xte, yte, _ = D.load_arrays()["test"]
    for n in names:
        if n in ledger:
            print(f"REFUSED: {n} already evaluated on test (once-only rule)")
            continue
        model, size, cls = load_model(n)
        p = predict(model, Xte, size, "cuda")
        pred = p.argmax(1)
        res = dict(name=n, test_acc=float(accuracy_score(yte, pred)),
                   test_macro_f1=float(f1_score(yte, pred, average="macro")),
                   confusion_matrix=confusion_matrix(yte, pred).tolist(), classes=cls,
                   report=classification_report(yte, pred, target_names=cls, output_dict=True), n_test=len(yte))
        json.dump(res, open(ROOT / "reports" / "metrics" / f"{n}_test.json", "w"), indent=1)
        np.save(ROOT / "reports" / "metrics" / f"{n}_test_probs.npy", p)
        ledger[n] = {"test_acc": res["test_acc"], "test_macro_f1": res["test_macro_f1"]}
        json.dump(ledger, open(LEDGER, "w"), indent=1)
        print(n, f"test_acc={res['test_acc']:.4f} macro_f1={res['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
