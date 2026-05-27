import numpy as np
import os
import json

WEIGHTS_DIR = r"C:\Users\raeyu\Desktop\Final project\Final\weights"
CKPT_DIR    = r"C:\Users\raeyu\Desktop\Final project\Final\ckpt_2res"
DOWN_DIR    = r"C:\Users\raeyu\Downloads"
def load_mem(name, shape):
    path = os.path.join(WEIGHTS_DIR, f"{name}.mem")
    with open(path) as f:
        bits = [int(l.strip(), 2) for l in f if l.strip()]
    return np.array(bits, dtype=np.uint8).view(np.int8).reshape(shape)

def load_scales():
    for p in [os.path.join(DOWN_DIR, "scales1.json"),
              os.path.join(WEIGHTS_DIR, "..", "scales.json")]:
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            if isinstance(list(s.values())[0], dict):
                return {k: v['scale'] for k, v in s.items()}
            return s
    return {}

def calc_shift(w_scale, in_scale, out_scale=1/127):
    combined = w_scale * in_scale
    ratio = combined / out_scale
    return round(-np.log2(ratio))

def int8_sat(x):
    return np.clip(x, -128, 127).astype(np.int8)

def conv2d_int8(inp, weight, bias, shift, relu=True):
    oc, ic, kH, kW = weight.shape
    H, W = inp.shape[1], inp.shape[2]
    pad = kH // 2
    inp_p = np.pad(inp.astype(np.int32), ((0,0),(pad,pad),(pad,pad)))
    out = np.zeros((oc, H, W), dtype=np.int32)
    for o in range(oc):
        out[o] = int(bias[o])
        for i in range(ic):
            for kr in range(kH):
                for kc in range(kW):
                    out[o] += inp_p[i, kr:kr+H, kc:kc+W].astype(np.int32) * int(weight[o,i,kr,kc])
    out = out >> shift
    if relu:
        out = np.maximum(out, 0)
    return int8_sat(out)

def fc_int8(inp_flat, weight, bias, shift, relu=False):
    out = np.zeros(weight.shape[0], dtype=np.int32)
    for o in range(weight.shape[0]):
        out[o] = int(bias[o]) + np.sum(inp_flat.astype(np.int32) * weight[o].astype(np.int32))
    out = out >> shift
    if relu:
        out = np.maximum(out, 0)
    return int8_sat(out)

print("載入權重...")
w = {
    'entry_w':    load_mem('entry_0_weight',      (64, 2, 3, 3)),
    'entry_b':    load_mem('entry_0_bias',         (64,)),
    't0c1_w':     load_mem('tower_0_net_0_weight', (64,64,3,3)),
    't0c1_b':     load_mem('tower_0_net_0_bias',   (64,)),
    't0c2_w':     load_mem('tower_0_net_3_weight', (64,64,3,3)),
    't0c2_b':     load_mem('tower_0_net_3_bias',   (64,)),
    'pc_w':       load_mem('policy_head_0_weight', (2, 64,1,1)),
    'pc_b':       np.zeros(2, dtype=np.int8),
    'pfc_w':      load_mem('policy_head_4_weight', (81,162)),
    'pfc_b':      load_mem('policy_head_4_bias',   (81,)),
    'vc_w':       load_mem('value_head_0_weight',  (1, 64,1,1)),
    'vc_b':       np.zeros(1, dtype=np.int8),
    'vfc1_w':     load_mem('value_head_4_weight',  (64, 81)),
    'vfc1_b':     load_mem('value_head_4_bias',    (64,)),
    'vfc2_w':     load_mem('value_head_6_weight',  (1, 64)),
    'vfc2_b':     np.zeros(1, dtype=np.int8),
}

scales = load_scales()
in_s = 1/127

shifts = {}
for name, wk, default in [
    ('entry',   'entry_0_weight',       13),
    ('t0c1',    'tower_0_net_0_weight', 13),
    ('t0c2',    'tower_0_net_3_weight', 13),
    ('pc',      'policy_head_0_weight', 13),
    ('pfc',     'policy_head_4_weight', 13),
    ('vc',      'value_head_0_weight',  13),
    ('vfc1',    'value_head_4_weight',  13),
    ('vfc2',    'value_head_6_weight',  13),
]:
    if wk in scales:
        shifts[name] = calc_shift(scales[wk], in_s)
    else:
        shifts[name] = default

print("\n計算出的 shift 值：")
for k, v in shifts.items():
    print(f"  {k:8s}: {v}")

inp = np.zeros((2, 9, 9), dtype=np.int8)
inp[0, 4, 4] = 127

print("\n開始推理...")

x = conv2d_int8(inp, w['entry_w'], w['entry_b'], shifts['entry'], relu=True)
print(f"Entry:      min={x.min():4d} max={x.max():4d} nonzero={np.count_nonzero(x):4d}/{x.size}")

x = conv2d_int8(x, w['t0c1_w'], w['t0c1_b'], shifts['t0c1'], relu=True)
print(f"T0_Conv1:   min={x.min():4d} max={x.max():4d} nonzero={np.count_nonzero(x):4d}/{x.size}")

x = conv2d_int8(x, w['t0c2_w'], w['t0c2_b'], shifts['t0c2'], relu=True)
print(f"T0_Conv2:   min={x.min():4d} max={x.max():4d} nonzero={np.count_nonzero(x):4d}/{x.size}")

xp = conv2d_int8(x, w['pc_w'], w['pc_b'], shifts['pc'], relu=True)
print(f"Policy_C:   min={xp.min():4d} max={xp.max():4d} nonzero={np.count_nonzero(xp):4d}/{xp.size}")

xp_flat = xp.flatten()
print(f"Policy_FC input: shape={xp_flat.shape} min={xp_flat.min()} max={xp_flat.max()}")

policy = fc_int8(xp_flat, w['pfc_w'], w['pfc_b'], shifts['pfc'], relu=False)
print(f"Policy_FC:  min={policy.min():4d} max={policy.max():4d} nonzero={np.count_nonzero(policy):4d}/81")

print("\nPolicy 分布：")
policy_board = policy.reshape(9, 9)
print("    " + " ".join(f"{c:4d}" for c in range(9)))
for r in range(9):
    row = policy_board[r]
    print(f"{r}: " + " ".join(f"{int(v):4d}" for v in row))

print(f"\nPolicy ch=63~80（對應 row 7~8）：")
print(f"  {policy[63:81]}")

xv = conv2d_int8(x, w['vc_w'], w['vc_b'], shifts['vc'], relu=True)
print(f"\nValue_C:    min={xv.min():4d} max={xv.max():4d}")

xv_flat = xv.flatten()
vfc1 = fc_int8(xv_flat, w['vfc1_w'], w['vfc1_b'], shifts['vfc1'], relu=True)
print(f"Value_FC1:  min={vfc1.min():4d} max={vfc1.max():4d}")

value = fc_int8(vfc1, w['vfc2_w'], w['vfc2_b'], shifts['vfc2'], relu=False)
print(f"Value_FC2:  {value[0]}")

print("\n和 FPGA 對比：")
print(f"  模擬 Policy top5: {np.argsort(policy)[::-1][:5].tolist()}")
print(f"  模擬 Value:       {value[0]}")
print(f"\n  FPGA Policy top5: [38, 50, 37, 28, 1]")
print(f"  FPGA Value:       -13")