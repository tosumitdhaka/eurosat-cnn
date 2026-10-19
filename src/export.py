"""Export the final model to ONNX (self-contained single file) and verify against PyTorch."""
import json
import numpy as np, torch, onnx, onnxruntime as ort
from .evaluate import load_model
from .figures import FINAL
from .train import ROOT

OUT = ROOT / "models" / "eurosat_resnet18.onnx"


def main():
    model, size, cls = load_model(FINAL, "cpu")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(OUT), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(OUT)))
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    ref = model(x).detach().numpy(); got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    json.dump({"classes": cls, "size": size, "source_run": FINAL}, open(ROOT / "models" / "meta.json", "w"))
    print(f"exported {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB), max |torch-onnx| logit diff = {diff:.2e}")


if __name__ == "__main__":
    main()
