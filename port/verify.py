#!/usr/bin/env python3
# verify.py — golden capture + fault-injection verdict classifier for the
# ISPASS-2009 Phase-0 apps (LPS / LIB / NQU) on FaGeSim.
#
# Modes:
#   baseline : run fault-free N times, capture golden values -> port/golden.json
#   check    : run once (optionally with FI env), classify the outcome
#
# Verdicts: SUCCESS | SDC | CRASH | CUDA_ERROR | HANG
#
# App verdict rules:
#   NQU : exact match of solution count against analytic table
#   LPS : rms error (GPU vs CPU gold) below tolerance
#   LIB : v / Lb within relative tolerance of golden (Monte Carlo, non-det)
#
# Examples:
#   python3 verify.py --app all --mode baseline
#   python3 verify.py --app NQU --mode check
#   python3 verify.py --app NQU --mode check --ldpreload tools/nvbit_fault_injector/fault_injector.so \
#       --env FAULT_KERNEL=... --env FAULT_TYPE=reg_bitflip ...
import argparse
import json
import os
import re
import subprocess
import sys
import time

PORT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(PORT_DIR, "bin")
GOLDEN_PATH = os.path.join(PORT_DIR, "golden.json")
CSV_PATH = os.path.join(PORT_DIR, "verdicts.csv")

# analytic n-queens solution counts
NQU_TABLE = {8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596}

# canonical run commands (also serve as golden-run params)
APP_CMDS = {
    "NQU": ["./NQU", "-g", "12", "9600"],
    "LPS": ["./LPS", "-nx=128", "-ny=128", "-nz=128", "-repeat=4"],
    "LIB": ["./LIB"],
}


def run_app(app, timeout, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        proc = subprocess.run(
            APP_CMDS[app], cwd=BIN_DIR, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rc, out, wall = proc.returncode, proc.stdout, time.time() - t0
    except subprocess.TimeoutExpired:
        return "HANG", "", time.time() - t0
    return rc, out, wall


def parse_lps(out):
    m = re.findall(r"rms error = ([0-9.eE+-]+)", out)
    return ("rms", float(m[-1])) if m else ("rms", None)


def parse_nqu(out):
    m = re.findall(r"GPU: (\d+) queen = (-?\d+)", out)
    return ("queens", int(m[-1][1])) if m else ("queens", None)


def parse_lib(out):
    vs = re.findall(r"v\s+=\s+([0-9.eE+-]+)", out)
    lbs = re.findall(r"Lb =\s+([0-9.eE+-]+)", out)
    return ("v_Lb",
            (float(vs[-1]), float(lbs[-1])) if vs and lbs else (None, None))


def classify(app, rc, out, golden, args):
    if rc != 0:
        # 注入器/cutil 输出为 "Cuda error in function ..."（首字母大写），
        # 统一按小写匹配，避免漏标 CUDA_ERROR。
        low = out.lower()
        if "cuda error" in low or "cutil-shim" in out:
            return "CUDA_ERROR", out.strip().splitlines()[-1][:120]
        return "CRASH", f"exit={rc}"
    if app == "NQU":
        _, val = parse_nqu(out)
        if val is None:
            return "SDC", "no GPU result line"
        exp = NQU_TABLE.get(12)
        return ("SUCCESS", f"queens={val}") if val == exp else \
               ("SDC", f"queens={val} expected={exp}")
    if app == "LPS":
        _, rms = parse_lps(out)
        if rms is None:
            return "SDC", "no rms line"
        return ("SUCCESS", f"rms={rms:.3e}") if rms <= args.lps_tol else \
               ("SDC", f"rms={rms:.3e} > tol={args.lps_tol}")
    if app == "LIB":
        _, (v, lb) = parse_lib(out)
        if v is None or golden is None:
            return "SDC", "no v/Lb line" if v is None else "no golden"
        gv, glb = golden["v"], golden["Lb"]
        ok_v = abs(v - gv) <= args.lib_rel_tol * max(abs(gv), 1e-9)
        ok_l = abs(lb - glb) <= args.lib_rel_tol * max(abs(glb), 1e-9)
        return ("SUCCESS", f"v={v:.5f} Lb={lb:.5f}") if ok_v and ok_l else \
               ("SDC", f"v={v:.5f}/{gv:.5f} Lb={lb:.5f}/{glb:.5f} reltol={args.lib_rel_tol}")
    return "CRASH", "unknown app"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="all",
                   help="LPS | LIB | NQU | all")
    p.add_argument("--mode", default="check", choices=["baseline", "check"])
    p.add_argument("--runs", type=int, default=3,
                   help="baseline mode: number of fault-free runs")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--lps-tol", type=float, default=1e-2)
    p.add_argument("--lib-rel-tol", type=float, default=2e-2)
    p.add_argument("--ldpreload", default=None,
                   help="path to fault_injector.so for FI runs")
    p.add_argument("--env", action="append", default=[],
                   help="extra KEY=VAL passed to the app (FI config)")
    p.add_argument("--show-output", action="store_true",
                   help="print full app+injector stdout for check runs")
    p.add_argument("--csv", default=None, help="append verdict rows here")
    args = p.parse_args()

    apps = list(APP_CMDS) if args.app == "all" else [args.app.upper()]
    golden = {}
    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH) as f:
            golden = json.load(f)

    csv_path = args.csv or CSV_PATH
    new_csv = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a")

    for app in apps:
        env_extra = {}
        if args.env:
            env_extra.update(dict(kv.split("=", 1) for kv in args.env))
        if args.ldpreload:
            env_extra["LD_PRELOAD"] = os.path.abspath(args.ldpreload)

        if args.mode == "baseline":
            samples = []
            for _ in range(args.runs):
                rc, out, wall = run_app(app, args.timeout)
                if rc != 0:
                    print(f"[baseline] {app} rc={rc}\n{out}")
                    sys.exit(f"baseline run failed for {app}")
                if app == "LPS":
                    samples.append(parse_lps(out)[1])
                elif app == "NQU":
                    samples.append(parse_nqu(out)[1])
                else:
                    samples.append(parse_lib(out)[1])
                print(f"[baseline] {app}: {samples[-1]} ({wall:.1f}s)")
            if app == "LIB":
                vs, lbs = zip(*samples)
                vals = {"v": sum(vs) / len(vs), "Lb": sum(lbs) / len(lbs),
                        "runs": args.runs}
            else:
                vals = {"value": samples[0], "all": samples}
            golden[app] = vals
            with open(GOLDEN_PATH, "w") as f:
                json.dump(golden, f, indent=2)
            print(f"[baseline] {app} golden -> {GOLDEN_PATH}: {vals}")
            continue

        rc, out, wall = run_app(app, args.timeout, env_extra)
        verdict, detail = classify(app, rc, out,
                                   golden.get(app) if app == "LIB" else None, args)
        print(f"[check] {app}: {verdict} ({detail}) wall={wall:.1f}s")
        if args.show_output:
            print(out)
        else:
            # surface injector lifecycle lines (they are easy to miss otherwise)
            keys = ("[Gemu FI]", "Gemu Fault Injector", "Instrumenting",
                    "Inserting", "Fault Injection Summary", "Faults injected")
            for line in out.splitlines():
                if any(k in line for k in keys):
                    print(f"    | {line}")
        csv_f.write(f"{time.strftime('%Y%m%d-%H%M%S')},{app},"
                    f"{','.join(k for k, v in (env_extra or {}).items())},"
                    f"{verdict},{detail}\n")

    csv_f.close()


if __name__ == "__main__":
    main()
