"""
Quick sanity check for ops.py.
Run with:  python -m nas_rl.search_space.test_ops
or:        python nas_rl/search_space/test_ops.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from nas_rl.search_space.ops import OPS, IDX_TO_OP, OP_TO_IDX, N_OPS

BATCH, C, H, W = 2, 16, 8, 8

def check(op_name, stride):
    op = OPS[op_name](C, stride)
    x = torch.randn(BATCH, C, H, W)
    with torch.no_grad():
        y = op(x)

    expected_h = H // stride
    expected_w = W // stride
    ok = (y.shape == torch.Size([BATCH, C, expected_h, expected_w]))
    status = "PASS" if ok else f"FAIL — got {tuple(y.shape)}"
    print(f"  {op_name:10s}  stride={stride}  "
          f"in={tuple(x.shape)}  out={tuple(y.shape)}  {status}")
    return ok


print("=" * 60)
print("Op shape tests")
print("=" * 60)
all_pass = True
for op_name in IDX_TO_OP:
    for stride in (1, 2):
        passed = check(op_name, stride)
        all_pass = all_pass and passed

print()
print("=" * 60)
print("Index mapping")
print("=" * 60)
for i, name in enumerate(IDX_TO_OP):
    assert OP_TO_IDX[name] == i, f"OP_TO_IDX[{name}] should be {i}"
    print(f"  {i}: {name}")

print()
print(f"N_OPS = {N_OPS}")
print()
if all_pass:
    print("All shape tests PASSED")
else:
    print("Some tests FAILED — see above")
    sys.exit(1)