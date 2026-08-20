"""TensorRT engine build — P3, not in the minimum slice.

This module exists so the README and ``eval_rmse --backend trt`` have a
single place to point. Building fp32 then fp16 engines from the ONNX graphs
in ``engines/`` is the next phase, and it needs a GPU that the login node
we cut this slice on does not have.

Do not invent a TensorRT latency number from the PyTorch fp16 figure below.
They are different compilers.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    print(
        "P3 (TensorRT) is not in this slice. Export ONNX first "
        "(`python -m spur_depth.export.to_onnx`), then on a GPU node:\n"
        "  trtexec --onnx=engines/encoder.onnx --saveEngine=engines/encoder_fp32.plan\n"
        "See SHIPPING.md P3.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
