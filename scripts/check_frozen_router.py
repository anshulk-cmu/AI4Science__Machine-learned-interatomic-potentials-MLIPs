"""Check a frozen-router run: the router parameters of the final inference checkpoint equal those of step 0.

  python scripts/check_frozen_router.py RUN/train/*/checkpoints/step_0/inference_ckpt.pt RUN/train/*/checkpoints/final/inference_ckpt.pt

Compares the routing MLP and the composition embedding in the model and the EMA state dicts (exact equality), checks
that the other weights did train, and that the stored model config still names the frozen backbone and its constant.
Prints one PASS or FAIL line per check; exits 1 if any check fails.
"""
import re
import sys

import torch

ROUTER = re.compile(r"(^|\.)(routing_mlp|composition_embedding)\.")


def checks(first, last):
    a = torch.load(first, map_location="cpu", weights_only=False)
    b = torch.load(last, map_location="cpu", weights_only=False)
    out = []
    for kind in ("model_state_dict", "ema_state_dict"):
        sa, sb = getattr(a, kind), getattr(b, kind)
        router = [k for k in sa if ROUTER.search(k)]
        other = [k for k in sa if not ROUTER.search(k) and torch.is_tensor(sa[k]) and sa[k].is_floating_point()]
        diff = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in router)
        moved = sum(not torch.equal(sa[k], sb[k]) for k in other)
        out.append((f"{kind}: {len(router)} router tensors, largest change {diff:.3g}", len(router) == 7 and diff == 0.0))
        out.append((f"{kind}: {moved} of {len(other)} other float tensors changed", moved > 0))
    backbone = b.model_config["backbone"]
    out.append((f"model config: {backbone['model']}, fixed_alpha {backbone.get('fixed_alpha')}",
                backbone["model"].endswith("eSCNMDFrozenMoeBackbone") and backbone.get("fixed_alpha") is not None))
    return out


def main():
    results = checks(*sys.argv[1:3])
    for text, ok in results:
        print(f"{'PASS' if ok else 'FAIL'} frozen router: {text}", flush=True)
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
