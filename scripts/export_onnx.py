"""Export the pinned RVM TorchScript model to a dynamic ONNX inference graph."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, r1, r2, r3, r4):
        output = self.model(src, r1, r2, r3, r4, 0.375, False)
        return output[0], output[1], output[2], output[3], output[4], output[5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    model = torch.jit.load(args.input, map_location="cpu").eval()
    src = torch.zeros(1, 3, args.height, args.width)
    with torch.inference_mode():
        initial = model(src, None, None, None, None, 0.375, False)
    recurrent = tuple(torch.zeros_like(item) for item in initial[2:6])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (src, *recurrent),
        args.output,
        input_names=["src", "r1i", "r2i", "r3i", "r4i"],
        output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "src": {2: "height", 3: "width"}, "fgr": {2: "height", 3: "width"}, "pha": {2: "height", 3: "width"},
            "r1i": {2: "r1h", 3: "r1w"}, "r1o": {2: "r1h", 3: "r1w"},
            "r2i": {2: "r2h", 3: "r2w"}, "r2o": {2: "r2h", 3: "r2w"},
            "r3i": {2: "r3h", 3: "r3w"}, "r3o": {2: "r3h", 3: "r3w"},
            "r4i": {2: "r4h", 3: "r4w"}, "r4o": {2: "r4h", 3: "r4w"},
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
