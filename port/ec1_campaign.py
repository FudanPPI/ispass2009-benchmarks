#!/usr/bin/env python3
# ec1_campaign.py — EC1 全量扩样驱动:SW(fagesim_nqu_replay) 与 HW(NVBitFI via
# verify.py) 跑同一组确定性故障点,汇总 3-class 混淆矩阵、Cohen's κ 与 bootstrap CI。
#
# 用法(仓库根目录执行):
#   python3 ispass2009-benchmarks/port/ec1_campaign.py sw       # 纯沙箱可跑
#   python3 ispass2009-benchmarks/port/ec1_campaign.py hw       # 需 GPU(沙箱外)
#   python3 ispass2009-benchmarks/port/ec1_campaign.py analyze
#
# 故障点选择原则(全部确定性,详见 docs/ispass2009-testing-plan.md §4.5):
#   - 目标寄存器在目标 PC 为存活操作数(IPOINT_BEFORE 翻转后被该指令或后续消费);
#   - 仅数据/本 block 归约通路,不产生跨 block 全局写竞争(排除 R2 bit2..bit19 类点);
#   - 含 MASKED / SDC / CRASH 三类预期标签;含 1 个注入不触发点(t55 idle)审计触发门控。
import json
import os
import random
import re
import subprocess
import sys
import time

PORT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PORT_DIR, "..", ".."))
REPLAY = os.path.join(REPO_ROOT, "build", "bin", "fagesim_nqu_replay")
CUBIN = os.path.join(REPO_ROOT, "ispass2009-benchmarks", "port", "cubin", "NQU.cubin")
INJECTOR = os.path.join(REPO_ROOT, "tools", "nvbit_fault_injector", "fault_injector.so")
VERIFY = os.path.join(PORT_DIR, "verify.py")
KERNEL = "_Z24solve_nqueen_cuda_kerneliiPjS_S_S_i"
SW_RESULTS = os.path.join(PORT_DIR, "ec1_sw_results.json")
HW_RESULTS = os.path.join(PORT_DIR, "ec1_hw_results.json")
CAMPAIGN_JSON = os.path.join(PORT_DIR, "fi_nqu_campaign.json")
GOLDEN = 14200

# pc=None 表示入口("any",指令索引 0)。pred 为 SW 侧预期(值或类),仅供审计。
C = [
    # --- A 组:R5@0x760 归约最终值(t0 R5=7100=0x1BBC,置位 {2,3,4,5,7,8,9,11,12}) ---
    ("a_r5_760_t0_b1",  0x760, "R5", 0,   1,  "SDC", 14204),
    ("a_r5_760_t0_b2",  0x760, "R5", 0,   2,  "SDC", 14192),
    ("a_r5_760_t0_b6",  0x760, "R5", 0,   6,  "SDC", 14328),
    ("a_r5_760_t0_b8",  0x760, "R5", 0,   8,  "SDC", 13688),
    ("a_r5_760_t0_b10", 0x760, "R5", 0,  10,  "SDC", 16248),
    ("a_r5_760_t0_b11", 0x760, "R5", 0,  11,  "SDC", 10104),
    ("a_r5_760_t0_b12", 0x760, "R5", 0,  12,  "SDC", 6008),
    ("a_r5_760_t0_b16", 0x760, "R5", 0,  16,  "SDC", 145272),
    ("a_r5_760_t96_b1", 0x760, "R5", 96,  1,  "SDC", 14204),   # block1 tid0 R5=0
    ("a_r5_760_t96_b15",0x760, "R5", 96, 15,  "SDC", 79736),
    ("a_r5_760_t192_b7",0x760, "R5", 192, 7,  "SDC", 14456),   # block2 tid0 R5=0
    # --- B 组:R2@0x770 STG 地址 ---
    ("b_r2_770_t0_b0",  0x770, "R2", 0,   0,  "CRASH", None),  # 非对齐
    ("b_r2_770_t0_b1",  0x770, "R2", 0,   1,  "CRASH", None),  # 非对齐(addr&3=2)
    ("b_r2_770_t96_b0", 0x770, "R2", 96,  0,  "CRASH", None),
    ("b_r2_770_t96_b1", 0x770, "R2", 96,  1,  "CRASH", None),
    ("b_r2_770_t0_b20", 0x770, "R2", 0,  20,  "SDC", 0),       # 对齐越界,静默
    ("b_r2_770_t96_b20",0x770, "R2", 96, 20,  "MASKED", GOLDEN),  # block1 贡献恒 0,丢写无影响
    # --- C 组:入口死寄存器(首条指令前,R5 等在 0x90-0x160 才被定义) ---
    ("c_entry_r5_t0_b0", None, "R5", 0,  0,  "MASKED", GOLDEN),
    ("c_entry_r5_t0_b7", None, "R5", 0,  7,  "MASKED", GOLDEN),
    ("c_entry_r5_t96_b0", None, "R5", 96, 0,  "MASKED", GOLDEN),
    # --- D 组:求解器数据通路(类标签由 SW 预测、HW 核对) ---
    # 0x190 LOP3 R2,R4,R9,R6:R4=初始列掩码,低 bit 改变占位集,高 bit(>=n=12)被全 1 掩码屏蔽
    ("d_r4_190_t0_b3",  0x190, "R4", 0,   3,  None, None),
    ("d_r4_190_t0_b20", 0x190, "R4", 0,  20,  None, None),
    ("d_r4_190_t1_b11", 0x190, "R4", 1,  11,  None, None),
    ("d_r4_190_t54_b3", 0x190, "R4", 54,  3,  None, None),
    # 0x230 IADD3 R2,R7,1:R7=当前行占位掩码(回溯内循环)
    ("d_r7_230_t0_b3",  0x230, "R7", 0,   3,  None, None),
    ("d_r7_230_t0_b20", 0x230, "R7", 0,  20,  None, None),
    # 0x3c0 IADD3 R12,R12,1:R12=本线程解计数器(找到解时执行)
    ("d_r12_3c0_t0_b0", 0x3c0, "R12", 0,  0,  None, None),
    ("d_r12_3c0_t0_b7", 0x3c0, "R12", 0,  7,  None, None),
    # 0x460 STS [tid.X4+0x3c00],R12:本线程最终计数写归约槽
    ("d_r12_460_t0_b0", 0x460, "R12", 0,  0,  None, None),
    ("d_r12_460_t1_b0", 0x460, "R12", 1,  0,  None, None),
    # 0x700 IMAD.IADD R2,R2,1,R5(tid0 归约树最后一级:R5=tid1 子树和)
    ("d_r5_700_t0_b0",  0x700, "R5", 0,   0,  None, None),
    ("d_r5_700_t0_b7",  0x700, "R5", 0,   7,  None, None),
    ("d_r2_700_t0_b4",  0x700, "R2", 0,   4,  None, None),
    # --- E 组:idle 线程(tid55 >= conds=55,0x80 分支跳过求解器)→ 注入不触发 ---
    ("e_nofire_t55",    0x190, "R4", 55,  3,  "MASKED", GOLDEN),
]
# baseline(无故障)单独追加
BASELINE = ("z_baseline", None, None, None, None, "MASKED", GOLDEN)


def spec_iter():
    yield {"name": BASELINE[0], "pc": None, "reg": None, "thread": None,
           "bit": None, "pred_cls": BASELINE[5], "pred_val": BASELINE[6],
           "baseline": True}
    for name, pc, reg, tid, bit, pcls, pval in C:
        yield {"name": name, "pc": pc, "reg": reg, "thread": tid, "bit": bit,
               "pred_cls": pcls, "pred_val": pval, "baseline": False}


def map3(verdict):
    """原始判级 → 3-class:MASKED(值不变,含 SUCCESS)/ SDC / CRASH(非零退出,含 CUDA_ERROR/HANG)"""
    v = verdict.upper()
    if v in ("SUCCESS", "MASKED"):
        return "MASKED"
    if v == "SDC":
        return "SDC"
    return "CRASH"  # CRASH / CUDA_ERROR / HANG


def run_sw_one(s):
    cmd = [REPLAY, "--cubin", CUBIN, "--n", "12"]
    if not s["baseline"]:
        # HW 侧 "any" = 指令索引 0(PC=0x0000);SW replay 只接受数字 PC,语义等价传 0x0
        pc = 0x0 if s["pc"] is None else s["pc"]
        cmd += ["--fault-bit", str(s["bit"]),
                "--fault-pc", f"0x{pc:x}",
                "--fault-reg", s["reg"],
                "--fault-thread", str(s["thread"])]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=600)
    out = proc.stdout
    mv = re.search(r"\[verdict\]\s+(\w+)", out)
    mt = re.search(r"\[sim\] SW result: n=12 total=(-?\d+)", out)
    mtrap = re.search(r"KERNEL TRAP: (.+)", out)
    return {
        "name": s["name"], "rc": proc.returncode,
        "verdict": mv.group(1) if mv else "UNKNOWN",
        "cls": map3(mv.group(1)) if mv else "CRASH",
        "value": int(mt.group(1)) if mt else None,
        "trap": mtrap.group(1).strip() if mtrap else None,
        "wall": round(time.time() - t0, 1),
    }


def cmd_sw():
    results = []
    for s in spec_iter():
        r = run_sw_one(s)
        results.append(r)
        print(f"[sw] {s['name']:<20} {r['cls']:<7} value={r['value']} "
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
            "cubin_path": "ispass2009-benchmarks/port/cubin/NQU.cubin",
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
    t0 = time.time()
    cmd = ["python3", VERIFY, "--app", "NQU", "--mode", "check", "--timeout", "300"]
    env = dict(os.environ)
    if not s["baseline"]:
        cmd += ["--ldpreload", INJECTOR,
                "--env", f"FAULT_CONFIG_PATH={config_json}",
                "--env", f"FAULT_EXPERIMENT_NAME={s['name']}"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=400)
    out = proc.stdout
    # 注入器自检:配置文件缺失/实验名未命中时工具会打印 [Gemu FI] ERROR(且会
    # 静默回退到默认参数注入——见 fault_injector.cu loadConfigFromJson 失败路径),
    # 此类 run 必须判为 INJERROR,绝不能当作 MASKED 计入一致性。
    inj_error = "[Gemu FI] ERROR" in out
    m = re.search(r"\[check\] NQU: (\w+)(?:\s*\(([^)]*)\))?", out)
    verdict = m.group(1) if m else "CRASH"
    detail = m.group(2) or ""
    mq = re.search(r"queens=(-?\d+)", detail)
    inj = re.search(r"Faults injected \(actual\):\s*(\d+)", out)
    cls = "INJERROR" if inj_error else map3(verdict)
    return {
        "name": s["name"], "rc": proc.returncode,
        "verdict": verdict, "cls": cls,
        "value": int(mq.group(1)) if mq else None,
        "injected": int(inj.group(1)) if inj else None,
        "inj_error": inj_error,
        "detail": detail[:120],
        "wall": round(time.time() - t0, 1),
    }


def cmd_hw():
    write_hw_json()
    results = []
    for s in spec_iter():
        r = run_hw_one(s)
        results.append(r)
        print(f"[hw] {s['name']:<20} {r['cls']:<7} value={r['value']} "
              f"inj={r['injected']} ({r['wall']}s)")
        with open(HW_RESULTS, "w") as f:  # 增量落盘,长任务可断点查看
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
        val_match = (h["value"] == s["value"]) if (h["value"] is not None and
                                                  s["value"] is not None) else None
        rows.append((nm, h["cls"], s["cls"], h["value"], s["value"],
                     "✓" if h["cls"] == s["cls"] else "✗",
                     "✓" if val_match else ("—" if val_match is None else "✗"),
                     h.get("injected")))

    print(f"{'site':<20} {'HW':<7} {'SW':<7} {'HW_val':>8} {'SW_val':>8}  L1  L2  inj")
    for nm, hc, sc, hv, sv, l1, l2, inj in rows:
        print(f"{nm:<20} {hc:<7} {sc:<7} {str(hv):>8} {str(sv):>8}  {l1}   {l2}  {inj}")

    k, po, pe, conf = cohen_kappa(pairs)
    n = len(pairs)
    agree = sum(1 for h, s in pairs if h == s)
    print(f"\n混淆矩阵 (行=HW, 列=SW), n={n}:")
    print(f"{'':<10}" + "".join(f"{s:>9}" for s in ("MASKED", "SDC", "CRASH")))
    for h in ("MASKED", "SDC", "CRASH"):
        print(f"{h:<10}" + "".join(f"{conf[h][s]:>9}" for s in ("MASKED", "SDC", "CRASH")))
    print(f"\nL1 分类一致: {agree}/{n} = {agree/n:.1%}")
    print(f"Po={po:.4f}  Pe={pe:.4f}  Cohen's κ = {k:.4f}")

    # 值级一致(双方均非 CRASH 且有值)
    val_pairs = [(r[3], r[4]) for r in rows if r[3] is not None and r[4] is not None]
    val_agree = sum(1 for hv, sv in val_pairs if hv == sv)
    print(f"值级精确一致(非 CRASH 子集): {val_agree}/{len(val_pairs)}")

    # bootstrap 95% CI(百分位法,固定种子可复现)
    rng = random.Random(20260904)
    ks = []
    for _ in range(10000):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        ks.append(cohen_kappa(sample)[0])
    ks.sort()
    print(f"κ bootstrap 95% CI: [{ks[250]:.4f}, {ks[9750]:.4f}] "
          f"(10000 次重采样)")

    out = {"n": n, "l1_agree": agree, "po": po, "pe": pe, "kappa": k,
           "ci95": [ks[250], ks[9750]],
           "value_agree": f"{val_agree}/{len(val_pairs)}",
           "confusion": conf,
           "rows": [{"name": r[0], "hw": r[1], "sw": r[2], "hw_val": r[3],
                     "sw_val": r[4], "l1": r[5], "l2": r[6]} for r in rows]}
    with open(os.path.join(PORT_DIR, "ec1_kappa.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[analyze] -> {os.path.join(PORT_DIR, 'ec1_kappa.json')}")




# ====================== EC1 随机 campaign(§4.6) ======================
#
# 站点生成模拟 NVBitFI 随机注入:静态指令均匀采样,寄存器从该指令的 Rn 操作数
# 集合均匀采样(含源/目的),线程在 100x96 网格内均匀采样,位 0..31 均匀。
# trigger_count=1:目标线程首次到达该 PC(且谓词开启)时翻转。
# 目标线程/PC 不可达或谓词关闭时双方均不触发(injected=0, MASKED),
# 这类"死点"计入总体 κ,并单列 fired 子集 κ 作为敏感性分析。
RAND_SEED = 20260905
RAND_N = 120
RAND_K = 3                       # HW 每点重复次数(多数投票)
RAND_SPECS = os.path.join(PORT_DIR, "ec1rand_specs.json")
RAND_SW_RESULTS = os.path.join(PORT_DIR, "ec1rand_sw_results.json")
RAND_HW_RESULTS = os.path.join(PORT_DIR, "ec1rand_hw_results.json")
RAND_CAMPAIGN_JSON = os.path.join(PORT_DIR, "fi_nqu_campaign_rand.json")
RAND_KAPPA = os.path.join(PORT_DIR, "ec1rand_kappa.json")
RAND_BLOCKS = 100
RAND_LANES = 96


def source_reg_pool(opcode, operand_str):
    """返回该指令在 IPOINT_BEFORE 时刻"存活"的源寄存器集合。
    - 存储类(ST*/STL):地址与数据寄存器均为源,保留全部;
    - 其余指令:目的寄存器(首个 Rn)会被覆写,BEFORE 翻转它必然被屏蔽
      (v1 全 MASKED 的主因);若目的寄存器同时作为源出现
      (如 LDG R4,[R4.64]),地址用途仍存活,保留。"""
    ordered = [int(x) for x in
               re.findall(r"(?<![A-Za-z0-9_])R(\d+)", operand_str)]
    if not ordered:
        return []
    op_u = opcode.upper()
    if op_u.startswith(("STG", "STS", "STL", "ST")):
        return sorted(set(ordered))
    dest = ordered[0]
    # ordered[1:] 全是源位置操作数。注意"读写目的"必须保留:如
    # IMAD.IADD R2,R2,0x1,R5 中 R2 既是目的又是源,BEFORE 翻转 R2 会
    # 改变本条指令结果(探针 p3:0x4d0 R2 bit6 -> HW SDC 14328 证实);
    # 同理 LDG R4,[R4.64] 的 R4 地址用途存活。
    pool = ordered[1:]
    # 无源寄存器(MOV RZ/常量、S2R 等):目的寄存器 BEFORE 翻转必被覆写,跳过
    return sorted(set(pool))


def parse_sass_sites():
    """cuobjdump 反汇编 -> [(pc, opcode, [存活源 reg_idx,...])]。"""
    env = dict(os.environ)
    env["PATH"] = "/usr/local/cuda-12/bin:" + env.get("PATH", "")
    out = subprocess.run(["cuobjdump", "--dump-sass", CUBIN], capture_output=True,
                         text=True, env=env, cwd=REPO_ROOT).stdout
    sites = []
    for line in out.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]{4})\*/\s+(?:@!?P\d+\s+)?([A-Za-z][\w.]*)\s+(.*?)\s*;",
                     line)
        if not m:
            continue
        pc = int(m.group(1), 16)
        opcode = m.group(2)
        regs = source_reg_pool(opcode, m.group(3))
        if regs:
            sites.append((pc, opcode, regs))
    return sites


def gen_rand_specs(n, seed):
    rng = random.Random(seed)
    sass = parse_sass_sites()
    if not sass:
        raise RuntimeError("cuobjdump 解析失败:无可用指令站点")
    specs = []
    for i in range(n):
        pc, opcode, regs = rng.choice(sass)
        reg = rng.choice(regs)
        gtid = rng.randrange(RAND_BLOCKS * RAND_LANES)
        bit = rng.randrange(32)
        specs.append({
            "name": f"r{i:03d}_{pc:03x}_R{reg}_t{gtid}_b{bit}",
            "pc": pc, "reg": f"R{reg}", "thread": gtid, "bit": bit,
            "opcode": opcode, "baseline": False,
        })
    return specs


def load_or_gen_rand_specs(n, seed):
    if os.path.exists(RAND_SPECS):
        specs = json.load(open(RAND_SPECS))
        if len(specs) >= n and specs[0].get("seed") == seed:
            return specs[:n]
    specs = gen_rand_specs(n, seed)
    json.dump([dict(sp, seed=seed) for sp in specs],
              open(RAND_SPECS, "w"), indent=2)
    print(f"[rand] specs -> {RAND_SPECS} ({n} sites, seed={seed})")
    return specs


def cmd_swrand():
    import concurrent.futures
    n = int(sys.argv[2]) if len(sys.argv) > 2 else RAND_N
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else RAND_SEED
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    specs = load_or_gen_rand_specs(n, seed)
    print(f"[swrand] {n} sites, {workers} workers (每点 ~21s)")
    results = [None] * len(specs)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_sw_one, sp): i for i, sp in enumerate(specs)}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            r = fut.result()
            results[i] = r
            done += 1
            print(f"[swrand] {done}/{n} {r['name']:<28} {r['cls']:<7} "
                  f"value={r['value']} ({r['wall']}s)")
    json.dump(results, open(RAND_SW_RESULTS, "w"), indent=2)
    print(f"[swrand] -> {RAND_SW_RESULTS} (total {time.time()-t0:.0f}s)")


def write_rand_hw_json(specs):
    exps = [{
        "name": sp["name"],
        "cubin_path": "ispass2009-benchmarks/port/cubin/NQU.cubin",
        "kernel_name": KERNEL,
        "fault": {
            "type": "register_bit_flip",
            "target_thread": sp["thread"],
            "target_pc": f"0x{sp['pc']:x}",
            "target_register": sp["reg"],
            "target_bit": sp["bit"],
            "trigger_count": 1,
        },
    } for sp in specs]
    json.dump({"experiments": exps}, open(RAND_CAMPAIGN_JSON, "w"), indent=2)
    print(f"[hwrand] campaign json -> {RAND_CAMPAIGN_JSON} ({len(exps)} experiments)")


def cmd_hwrand():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else RAND_N
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else RAND_SEED
    k = int(sys.argv[4]) if len(sys.argv) > 4 else RAND_K
    specs = load_or_gen_rand_specs(n, seed)
    write_rand_hw_json(specs)
    results = {}
    if os.path.exists(RAND_HW_RESULTS):
        results = {r["name"]: r for r in json.load(open(RAND_HW_RESULTS))}
    t0 = time.time()
    total = n * k
    done = 0
    for sp in specs:
        r = results.get(sp["name"], {"name": sp["name"], "runs": []})
        while len(r["runs"]) < k:
            spec_arg = dict(sp, baseline=False)
            run = run_hw_one(spec_arg, config_json=RAND_CAMPAIGN_JSON)
            r["runs"].append({kk: run[kk] for kk in
                              ("cls", "verdict", "value", "injected",
                               "rc", "inj_error")})
            done += 1
            print(f"[hwrand] {done}/{total} {sp['name']:<28} "
                  f"run{len(r['runs'])}: {run['cls']:<7} inj={run['injected']} "
                  f"value={run['value']} ({run['wall']}s)")
            results[sp["name"]] = r
            json.dump(list(results.values()), open(RAND_HW_RESULTS, "w"), indent=2)
    print(f"[hwrand] -> {RAND_HW_RESULTS} (total {time.time()-t0:.0f}s)")


def majority(votes):
    """多数投票,平票时取标签序(MASKED<SDC<CRASH)最先者(保守,可复现)。"""
    order = {"MASKED": 0, "SDC": 1, "CRASH": 2}
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], -order[kv[0]]))
    return best[0]


def kappa_report(tag, pairs):
    k, po, pe, conf = cohen_kappa(pairs)
    n = len(pairs)
    agree = sum(1 for h, s in pairs if h == s)
    kappa_defined = pe < 1.0 - 1e-9
    print(f"\n[{tag}] n={n}  L1 一致 {agree}/{n} = {agree/n:.1%}  "
          f"Po={po:.4f} Pe={pe:.4f} "
          + (f"κ={k:.4f}" if kappa_defined else
             "κ=undefined (Pe=1: 单方边缘全单类,κ 无定义,只报原始一致率)"))
    print(f"{'':<10}" + "".join(f"{s:>9}" for s in ("MASKED", "SDC", "CRASH")))
    for h in ("MASKED", "SDC", "CRASH"):
        print(f"{h:<10}" + "".join(f"{conf[h][s]:>9}" for s in ("MASKED", "SDC", "CRASH")))
    if kappa_defined:
        rng = random.Random(20260906)
        ks = []
        for _ in range(10000):
            sample = [pairs[rng.randrange(n)] for _ in range(n)]
            kk, _, pe_s, _ = cohen_kappa(sample)
            if pe_s < 1.0 - 1e-9:
                ks.append(kk)
        if ks:
            ks.sort()
            lo, hi = int(0.025 * len(ks)), int(0.975 * len(ks))
            print(f"[{tag}] κ bootstrap 95% CI: [{ks[lo]:.4f}, {ks[hi]:.4f}] "
                  f"(n_boot={len(ks)}/10000, 单类样本已剔除)")
            ci = [ks[lo], ks[hi]]
        else:
            print(f"[{tag}] κ bootstrap: 所有重采样均为单类边缘,无法估计 CI")
            ci = [None, None]
    else:
        ci = [None, None]
    return (k if kappa_defined else None), po, pe, conf, ci, agree


def cmd_analyzerand():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else RAND_N
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else RAND_SEED
    specs = load_or_gen_rand_specs(n, seed)
    sw = {r["name"]: r for r in json.load(open(RAND_SW_RESULTS))}
    hw = {r["name"]: r for r in json.load(open(RAND_HW_RESULTS))}

    pairs_all, pairs_fired = [], []
    val_pairs, nondet, missing, invalid = [], [], [], []
    rows = []
    for sp in specs:
        nm = sp["name"]
        if nm not in sw or nm not in hw:
            missing.append(nm)
            continue
        s, h = sw[nm], hw[nm]
        # 仅"注入器配置错误"为无效 run;inj=0 是合法结果(目标 tid 未执行该
        # PC / 谓词守卫关闭 -> 未触发,等价 MASKED);CRASH run 摘要常缺失
        # (injected=None)亦合法。
        bad_runs = [run for run in h["runs"]
                    if run.get("inj_error") or run.get("cls") == "INJERROR"]
        if bad_runs:
            invalid.append(nm)
            continue
        votes = [run["cls"] for run in h["runs"]]
        hw_cls = majority(votes)
        inj_runs = [run.get("injected") for run in h["runs"]]
        fired = all(x == 1 for x in inj_runs if x is not None) and                 any(x == 1 for x in inj_runs)
        is_nondet = len(set(votes)) > 1 or             (hw_cls == "SDC" and len(set(run.get("value") for run in h["runs"])) > 1)
        pairs_all.append((hw_cls, s["cls"]))
        if fired:
            pairs_fired.append((hw_cls, s["cls"]))
        if is_nondet:
            nondet.append({"name": nm, "votes": votes,
                           "values": [run.get("value") for run in h["runs"]],
                           "sw": s["cls"], "sw_value": s["value"]})
        if hw_cls != "CRASH" and s["cls"] != "CRASH":
            hv = next((run.get("value") for run in h["runs"]
                       if run.get("value") is not None), None)
            if hv is not None and s["value"] is not None:
                val_pairs.append((hv, s["value"]))
        rows.append((nm, hw_cls, votes, s["cls"], s["value"],
                     "✓" if hw_cls == s["cls"] else "✗",
                     sum(1 for x in inj_runs if x == 1), "ND" if is_nondet else ""))

    print(f"{'site':<28} {'HWmaj':<7} {'votes':<22} {'SW':<7} {'SWval':>8}  L1 inj flag")
    for nm, hc, votes, sc, sv, l1, inj, flag in rows:
        print(f"{nm:<28} {hc:<7} {','.join(votes):<22} {sc:<7} {str(sv):>8}  {l1}  {inj}  {flag}")
    if missing:
        print(f"[analyzerand] MISSING {len(missing)}: {missing[:5]} ...")
    if invalid:
        print(f"[analyzerand] INVALID(inj_error/injected!=1) {len(invalid)}: {invalid[:5]} ...")

    k_all, po, pe, conf, ci, agree = kappa_report("all", pairs_all)
    fired_stats = None
    if pairs_fired:
        k_f, pof, pef, conff, cif, agreef = kappa_report("fired", pairs_fired)
        fired_stats = {"n": len(pairs_fired), "l1_agree": agreef,
                       "po": pof, "pe": pef, "kappa": k_f, "ci95": cif}
    val_agree = sum(1 for hv, sv in val_pairs if hv == sv)
    print(f"\n值级精确一致(非 CRASH 子集): {val_agree}/{len(val_pairs)}")
    print(f"HW 非确定站点: {len(nondet)}")
    for nd in nondet:
        print(f"  {nd['name']}: votes={nd['votes']} values={nd['values']} "
              f"sw={nd['sw']}/{nd['sw_value']}")

    out = {"seed": seed, "n": len(rows), "k_runs": RAND_K if len(sys.argv) <= 4
           else int(sys.argv[4]),
           "l1_agree": agree, "po": po, "pe": pe, "kappa": k_all, "ci95": ci,
           "confusion": conf,
           "fired_subset": fired_stats,
           "value_agree": f"{val_agree}/{len(val_pairs)}",
           "nondeterministic": nondet,
           "missing": missing, "invalid": invalid}
    json.dump(out, open(RAND_KAPPA, "w"), indent=2, ensure_ascii=False)
    print(f"[analyzerand] -> {RAND_KAPPA}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"sw": cmd_sw, "hw": cmd_hw, "analyze": cmd_analyze,
     "swrand": cmd_swrand, "hwrand": cmd_hwrand,
     "analyzerand": cmd_analyzerand}[mode]()
