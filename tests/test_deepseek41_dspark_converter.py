#!/usr/bin/env python3
"""Validate real draft headers and reject malformed layouts without loading weights.

Run with the downloaded V4.1 checkpoint directory (three draft shards only).
"""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-tools"))
from deepseek41_dspark import DraftSource, build_plan
from glm53_quantize import QTYPE_MXFP4


def main(directory):
    with open(Path(directory) / "inference/config.json") as fp:
        config = json.load(fp)
    db = DraftSource(directory)
    try:
        plan = build_plan(db, config)
        assert len(plan) == 81
        assert all(a.offset + a.nbytes <= b.offset for a, b in zip(plan, plan[1:]))
        native = build_plan(db, config, "mxfp4")
        for old, new in zip(plan, native):
            assert old.name == new.name and old.shape == new.shape
            assert new.qtype == (QTYPE_MXFP4 if old.is_expert else old.qtype)
        assert all(a.offset + a.nbytes <= b.offset for a, b in zip(native, native[1:]))
        original = copy.deepcopy(db.tensors)
        cases = [
            ("wrong block size", lambda c, t: c.update(dspark_block_size=6)),
            ("wrong feature layers", lambda c, t: c.update(dspark_target_layer_ids=[36, 37, 38])),
            ("missing tensor", lambda c, t: t.pop("mtp.0.main_proj.weight")),
            ("wrong projection shape", lambda c, t: t["mtp.0.main_proj.weight"].update(shape=[1, 1])),
            ("wrong packed expert dtype", lambda c, t: t["mtp.0.ffn.experts.0.w1.weight"].update(dtype="BF16")),
            ("unexpected tensor", lambda c, t: t.update({"mtp.3.unexpected": {"shape": [1], "dtype": "F32"}})),
        ]
        for label, corrupt in cases:
            cfg = copy.deepcopy(config)
            db.tensors = copy.deepcopy(original)
            corrupt(cfg, db.tensors)
            try:
                build_plan(db, cfg)
            except (ValueError, KeyError):
                print(label + ": rejected PASS")
            else:
                raise AssertionError(label + " accepted")
        print("81-tensor draft layout and malformed-source rejection PASS")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_deepseek41_dspark_converter.py HF_DIRECTORY")
    main(sys.argv[1])
