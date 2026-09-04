#!/usr/bin/env python3
# ec1_lps_campaign.py — EC1 LPS(3D Laplace) campaign: SW(fagesim_lps_replay) 与
# HW(NVBitFI via fault_injector.so + LPS app) 跑同一组确定性故障点,汇总 3-class
# 混淆矩阵、Cohen's κ 与值级一致率。
#
# 用法(仓库根目录执行):
#   python3 ispass2009-benchmarks/port/ec1_lps_campaign.py sw       # 纯沙箱可跑
#   python3 ispass2009-benchmarks/port/ec1_lps_campaign.py hw       # 需 GPU(沙箱外)
#   python3 ispass2009-benchmarks/port/ec1_lps_campaign.py analyze
#
# 故障点选择原则(详见 docs/ispass2009-testing-plan.md §4.7):
#   - 32³ grid(repeat=4):SW 单点 ~30s,HW 秒级;故障语义与 128³ 一致(trigger_count=1
#     首次到达,4 次迭代传播放大 interior 翻转 → SDC;boundary 每轮覆写 1.0 → MASKED)。
#   - A 组:stencil 累加器 R5 @ STG 0x900,interior thread 33(tid_x=1,tid_y=1)→ SDC;
#   - B 组:STG 地址寄存器 R8 @ 0x900,低位翻转 → misaligned address trap → CRASH;
#   - C 组:boundary thread 0(i=0,j=0),R5 翻转被 4 轮覆写吸收 → MASKED;
#   - D 组:compute 路径(FMUL/FADD/LDG 地址),类标签由 SW 预测、HW 核对;
#   - E 组:no-fire(thread 0 经 @!P0 BRA 跳过 0x5a0 内部 LDS)→ 注入不触发 → MASKED。
import json
import os
import random
import re
import subprocess
import sys
import time

PORT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PORT_DIR, "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "build", "bin", "fagesim_lps_replay")
CUBIN = os.path.join(REPO_ROOT, "ispass2009-benchmarks", "port", "cubin", "LPS.cubin")
INJECTOR = os.path.join(REPO_ROOT, "tools", "nvbit_fault_injector", "fault_injector.so")
LPS_BIN = os.path.join(PORT_DIR, "bin", "LPS")
KERNEL = "_Z13GPU_laplace3diiiiPfS_"
SW_RESULTS = os.path.join(PORT_DIR, "ec1_lps_sw_results.json")
HW_RESULTS = os.path.join(PORT_DIR, "ec1_lps_hw_results.json")
CAMPAIGN_JSON = os.path.join(PORT_DIR, "fi_lps_campaign.json")
KAPPA = os.path.join(PORT_DIR, "ec1_lps_kappa.json")
GOLDEN_RMS = 0.0
LPS_TOL = 1e-2
NX, NY, NZ, REPEAT = 32, 32, 32, 4

# pc=None 表示入口("any",指令索引 0)。pred_cls 为 SW 侧预期类,pred_val 预期 rms(仅供审计)。
C = [
    # --- A 组:R5@0x900 STG stencil 结果,interior thread 33(tid_x=1,tid_y=1,block0) ---
    # 4 次迭代传播放大单 cell 翻转 → rms 远超 tol → SDC
    ("a_r5_900_t33_b0",  0x900, "R5", 33,  0, "SDC", None),
    ("a_r5_900_t33_b1",  0x900, "R5", 33,  1, "SDC", None),
    ("a_r5_900_t33_b20", 0x900, "R5", 33, 20, "SDC", None),
    ("a_r5_900_t33_b30", 0x900, "R5", 33, 30, "SDC", None),
    ("a_r5_900_t33_b31", 0x900, "R5", 33, 31, "SDC", None),
    # interior thread 1(tid_x=1,tid_y=0 → j=0 boundary? tid_y=0 → j=0 → boundary)
    # 改用 thread 34(tid_x=2,tid_y=1)→ interior(i=2,j=1)
    ("a_r5_900_t34_b0",  0x900, "R5", 34,  0, "SDC", None),
    ("a_r5_900_t34_b31", 0x900, "R5", 34, 31, "SDC", None),
    # interior thread 65(tid_x=1,tid_y=2)→ (i=1,j=2)
    ("a_r5_900_t65_b0",  0x900, "R5", 65,  0, "SDC", None),
    # --- B 组:R8@0x900 STG 地址,低位翻转 → misaligned address trap → CRASH ---
    ("b_r8_900_t33_b0",  0x900, "R8", 33,  0, "CRASH", None),
    ("b_r8_900_t33_b1",  0x900, "R8", 33,  1, "CRASH", None),
    ("b_r8_900_t34_b0",  0x900, "R8", 34,  0, "CRASH", None),
    # R2 @ STS 0x420 共享内存地址(比例因子 X4),低位翻转 → shared misaligned → CRASH
    ("b_r2_420_t33_b0",  0x420, "R2", 33,  0, "CRASH", None),
    ("b_r2_420_t33_b1",  0x420, "R2", 33,  1, "CRASH", None),
    # --- C 组:boundary thread 0(i=0,j=0),R5 每轮覆写 1.0 → MASKED ---
    ("c_r5_900_t0_b31",  0x900, "R5",  0, 31, "MASKED", GOLDEN_RMS),
    ("c_r5_900_t0_b0",   0x900, "R5",  0,  0, "MASKED", GOLDEN_RMS),
    # boundary thread 32(tid_x=0,tid_y=1 → i=0)→ MASKED
    ("c_r5_900_t32_b31", 0x900, "R5", 32, 31, "MASKED", GOLDEN_RMS),
    # --- D 组:compute 路径(类标签由 SW 预测、HW 核对) ---
    # 0x8a0 FMUL R5,R5,1/6:stencil 最终值;翻转 R5 before FMUL → 改变结果 → SDC
    ("d_r5_8a0_t33_b0",  0x8a0, "R5", 33,  0, None, None),
    ("d_r5_8a0_t33_b30", 0x8a0, "R5", 33, 30, None, None),
    # 0x850 FADD R5,R5,R8:累加第一邻居;翻转 R8(邻居值)→ SDC
    ("d_r8_850_t33_b0",  0x850, "R8", 33,  0, None, None),
    # 0x8f0 IMAD.WIDE R8,R6,R9,c[0x170]:STG 地址计算;翻转 R6(global index)→ 地址错 → CRASH
    ("d_r6_8f0_t33_b0",  0x8f0, "R6", 33,  0, None, None),
    ("d_r6_8f0_t33_b1",  0x8f0, "R6", 33,  1, None, None),
    # --- E 组:no-fire(thread 0 经 @!P0 BRA 0x670 跳过 0x5a0 LDS)→ 注入不触发 ---
    ("e_nofire_t0_5a0",  0x5a0, "R9",  0,  0, "MASKED", GOLDEN_RMS),
    ("e_nofire_t0_5e0",  0x5e0, "R11", 0,  0, "MASKED", GOLDEN_RMS),
]
# baseline(无故障)单独追加
BASELINE = ("z_baseline", None, None, None, None, "MASKED", GOLDEN_RMS)


def spec_iter():
    yield {"name": BASELINE[0], "pc": None, "reg": None, "thread": None,
           "bit": None, "pred_cls": BASELINE[5], "pred_val": BASELINE[6],
           "baseline": True}
    for name, pc, reg, tid, bit, pcls, pval in C:
        yield {"name": name, "pc": pc, "reg": reg, "thread": tid, "bit": bit,
               "pred_cls": pcls, "pred_val": pval, "baseline": False}


def map3(verdict):
    """原始判级 → 3-class:MASKED(含 SUCCESS)/ SDC / CRASH(非零退出,含 CUDA_ERROR)"""
    v = verdict.upper()
    if v in ("SUCCESS", "MASKED"):
        return "MASKED"
    if v == "SDC":
        return "SDC"
    return "CRASH"


def run_sw_one(s):
    cmd = [REPLAY, "--cubin", CUBIN,
           "--nx", str(NX), "--ny", str(NY), "--nz", str(NZ),
           "--repeat", str(REPEAT)]
    if not s["baseline"]:
        pc = 0x0 if s["pc"] is None else s["pc"]
        cmd += ["--fault-bit", str(s["bit"]),
                "--fault-pc", f"0x{pc:x}",
                "--fault-reg", s["reg"],
                "--fault-thread", str(s["thread"])]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=300)
    out = proc.stdout
    mv = re.search(r"\[verdict\]\s+(\w+)", out)
    mr = re.search(r"\[sim\] SW result: rms=([0-9.eE+-]+)", out)
    mtrap = re.search(r"KERNEL TRAP: (.+)", out)
    return {
        "name": s["name"], "rc": proc.returncode,
        "verdict": mv.group(1) if mv else "UNKNOWN",
        "cls": map3(mv.group(1)) if mv else "CRASH",
        "value": float(mr.group(1)) if mr else None,
        "trap": mtrap.group(1).strip() if mtrap else None,
        "wall": round(time.time() - t0, 1),
    }


def cmd_sw():
    results = []
    for s in spec_iter():
        r = run_sw_one(s)
        results.append(r)
        print(f"[sw] {s['name']:<22} {r['cls']:<7} rms={r['value']} "
              f"({r['wall']}s)")
    with open(SW_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sw] -> {SW_RESULTS}")


def write_hw_json():
    exps = []
    for s in spec_iter():
        if s["baseline"]:
            continue
        exps.append({
            "name": s["name"],
            "cubin_path": "ispass2009-benchmarks/port/cubin/LPS.cubin",
            "kernel_name": KERNEL,
            "fault": {
                "type": "register_bit_flip",
                "target_thread": s["thread"],
                "target_pc": "any" if s["pc"] is None else f"0x{s['pc']:x}",
                "target_register": s["reg"],
                "target_bit": s["bit"],
                "trigger_count": 1,
            },
        })
    with open(CAMPAIGN_JSON, "w") as f:
        json.dump({"experiments": exps}, f, indent=2)
    print(f"[hw] campaign json -> {CAMPAIGN_JSON} ({len(exps)} experiments)")


def run_hw_one(s, config_json=CAMPAIGN_JSON):
    """HW:直接跑 LPS app(32³)+ ldpreload 注入器,解析 rms。"""
    t0 = time.time()
    cmd = [LPS_BIN, f"--nx={NX}", f"--ny={NY}", f"--nz={NZ}",
           f"--repeat={REPEAT}"]
    env = dict(os.environ)
    if not s["baseline"]:
        env["LD_PRELOAD"] = INJECTOR
        env["FAULT_CONFIG_PATH"] = config_json
        env["FAULT_EXPERIMENT_NAME"] = s["name"]
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(LPS_BIN),
                              env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=300)
        out = proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"name": s["name"], "rc": -1, "verdict": "HANG", "cls": "CRASH",
                "value": None, "injected": None, "inj_error": False,
                "detail": "timeout", "wall": round(time.time() - t0, 1)}
    # 注入器自检:配置失败会打印 [Gemu FI] ERROR 并静默回退默认参数
    inj_error = "[Gemu FI] ERROR" in out
    # LPS 输出 "rms error = X"
    mr = re.search(r"rms error\s*=\s*([0-9.eE+-]+)", out)
    inj = re.search(r"Faults injected \(actual\):\s*(\d+)", out)
    rms = float(mr.group(1)) if mr else None
    if rc != 0:
        verdict = "CUDA_ERROR" if "cuda error" in out.lower() else "CRASH"
        cls = "INJERROR" if inj_error else "CRASH"
    elif rms is None:
        verdict, cls = "SDC", "SDC"
    elif rms <= LPS_TOL:
        verdict, cls = "SUCCESS", "MASKED"
    else:
        verdict, cls = "SDC", "SDC"
    if inj_error:
        cls = "INJERROR"
    return {
        "name": s["name"], "rc": rc,
        "verdict": verdict, "cls": cls,
        "value": rms,
        "injected": int(inj.group(1)) if inj else None,
        "inj_error": inj_error,
        "detail": f"rms={rms}" if rms is not None else "no rms line",
        "wall": round(time.time() - t0, 1),
    }


def cmd_hw():
    write_hw_json()
    results = []
    for s in spec_iter():
        r = run_hw_one(s)
        results.append(r)
        print(f"[hw] {s['name']:<22} {r['cls']:<7} rms={r['value']} "
              f"inj={r['injected']} ({r['wall']}s)")
        with open(HW_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
    print(f"[hw] -> {HW_RESULTS}")


def cohen_kappa(pairs):
    """pairs: [(hw_cls, sw_cls), ...]; 返回 (kappa, Po, Pe, confusion)"""
    labels = ["MASKED", "SDC", "CRASH"]
    conf = {h: {s: 0 for s in labels} for h in labels}
    for h, s in pairs:
        conf[h][s] += 1
    n = len(pairs)
    po = sum(conf[l][l] for l in labels) / n
    pe = sum((sum(conf[h][s] for s in labels) / n) *
             (sum(conf[s][l] for s in labels) / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return k, po, pe, conf


def val_match(hv, sv):
    """rms 值级一致:双方均非 None 且 |hv-sv| <= 1e-4 * max(|hv|,1e-9)(相对容差)"""
    if hv is None or sv is None:
        return None
    return abs(hv - sv) <= 1e-4 * max(abs(hv), 1e-9)


def cmd_analyze():
    sw = {r["name"]: r for r in json.load(open(SW_RESULTS))}
    hw = {r["name"]: r for r in json.load(open(HW_RESULTS))}
    names = [s["name"] for s in spec_iter()]
    pairs, rows = [], []
    for nm in names:
        if nm not in sw or nm not in hw:
            print(f"[analyze] MISSING {nm}: sw={nm in sw} hw={nm in hw}")
            continue
        s, h = sw[nm], hw[nm]
        pairs.append((h["cls"], s["cls"]))
        vm = val_match(h.get("value"), s.get("value"))
        rows.append((nm, h["cls"], s["cls"], h.get("value"), s.get("value"),
                     "✓" if h["cls"] == s["cls"] else "✗",
                     "✓" if vm else ("—" if vm is None else "✗"),
                     h.get("injected")))

    print(f"{'site':<22} {'HW':<7} {'SW':<7} {'HW_rms':>12} {'SW_rms':>12}  L1  L2  inj")
    for nm, hc, sc, hv, sv, l1, l2, inj in rows:
        print(f"{nm:<22} {hc:<7} {sc:<7} {str(hv):>12} {str(sv):>12}  {l1}   {l2}  {inj}")

    k, po, pe, conf = cohen_kappa(pairs)
    n = len(pairs)
    agree = sum(1 for h, s in pairs if h == s)
    print(f"\n混淆矩阵 (行=HW, 列=SW), n={n}:")
    print(f"{'':<10}" + "".join(f"{s:>9}" for s in ("MASKED", "SDC", "CRASH")))
    for h in ("MASKED", "SDC", "CRASH"):
        print(f"{h:<10}" + "".join(f"{conf[h][s]:>9}" for s in ("MASKED", "SDC", "CRASH")))
    print(f"\nL1 分类一致: {agree}/{n} = {agree/n:.1%}")
    print(f"Po={po:.4f}  Pe={pe:.4f}  Cohen's κ = {k:.4f}")

    val_pairs = [(r[3], r[4]) for r in rows if r[3] is not None and r[4] is not None]
    val_agree = sum(1 for hv, sv in val_pairs if val_match(hv, sv))
    print(f"值级一致(非 CRASH 子集,相对容差 1e-4): {val_agree}/{len(val_pairs)}")

    rng = random.Random(20260904)
    ks = []
    for _ in range(10000):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        kk, _, pe_s, _ = cohen_kappa(sample)
        if pe_s < 1.0 - 1e-9:
            ks.append(kk)
    if ks:
        ks.sort()
        print(f"κ bootstrap 95% CI: [{ks[250]:.4f}, {ks[9750]:.4f}] "
              f"(n_boot={len(ks)}/10000)")
        ci = [ks[250], ks[9750]]
    else:
        print("κ bootstrap: 所有重采样均为单类边缘,无法估计 CI")
        ci = [None, None]

    out = {"n": n, "l1_agree": agree, "po": po, "pe": pe, "kappa": k,
           "ci95": ci,
           "value_agree": f"{val_agree}/{len(val_pairs)}",
           "confusion": conf,
           "rows": [{"name": r[0], "hw": r[1], "sw": r[2], "hw_val": r[3],
                     "sw_val": r[4], "l1": r[5], "l2": r[6]} for r in rows]}
    with open(KAPPA, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[analyze] -> {KAPPA}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"sw": cmd_sw, "hw": cmd_hw, "analyze": cmd_analyze}[mode]()
