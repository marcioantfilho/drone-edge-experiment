from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

SYS = Path("/sys")
CLOCK_TOLERANCE = 0.97


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def thermal_zones() -> dict[str, dict]:
    zones = {}
    for z in sorted((SYS / "class/thermal").glob("thermal_zone*")):
        name = f"{_read(z / 'type') or z.name}"
        passive = [int(_read(t) or 0) / 1000 for t in z.glob("trip_point_*_temp")
                   if _read(t.with_name(t.name.replace("_temp", "_type"))) == "passive"]
        zones[name] = {"path": z / "temp", "trip_c": min(passive) if passive else None}
    return zones


def accelerators(pattern: str) -> dict[str, Path]:
    return {d.name: d for d in sorted((SYS / "class/devfreq").glob("*")) if pattern in d.name}


def cpu_policies() -> dict[str, Path]:
    return {p.name: p for p in sorted((SYS / "devices/system/cpu/cpufreq").glob("policy*"))}


def power_channels() -> dict[str, tuple[Path, Path]]:
    chans = {}
    for label in (SYS / "class/hwmon").glob("hwmon*/in*_label"):
        n = label.name[2:-6]
        volt, curr = label.with_name(f"in{n}_input"), label.with_name(f"curr{n}_input")
        if volt.exists() and curr.exists():
            chans[_read(label) or label.name] = (volt, curr)
    return chans


def sample(zones, accels, cpus, power) -> dict:
    row = {"t": time.time()}
    for name, z in zones.items():
        v = _read(z["path"])
        row[f"temp_{name}"] = int(v) / 1000 if v and v.lstrip("-").isdigit() else None
    for name, d in accels.items():
        row[f"freq_{name}"] = int(_read(d / "cur_freq") or 0)
    for name, p in cpus.items():
        row[f"cpu_{name}"] = int(_read(p / "scaling_cur_freq") or 0)
    for name, (volt, curr) in power.items():
        row[f"W_{name}"] = int(_read(volt) or 0) * int(_read(curr) or 0) / 1e6
    return row


def evaluate(rows: list[dict], zones: dict, accel_ref: dict[str, int], temp_limit: float | None) -> dict:
    """Critério pré-registrado: qualquer amostra com clock < 97% da referência ou zona >= trip."""
    clock_drops, hot = 0, 0
    for r in rows:
        if any(r.get(f"freq_{a}", ref) < CLOCK_TOLERANCE * ref for a, ref in accel_ref.items()):
            clock_drops += 1
        for name, z in zones.items():
            limit = z["trip_c"] if z["trip_c"] is not None else temp_limit
            t = r.get(f"temp_{name}")
            if limit is not None and t is not None and t >= limit:
                hot += 1
                break
    reasons = []
    if clock_drops:
        reasons.append(f"{clock_drops} amostras com acelerador < {CLOCK_TOLERANCE:.0%} do clock de referência")
    if hot:
        reasons.append(f"{hot} amostras com zona no trip point passivo")
    return {"valid": not reasons, "reasons": reasons, "clock_drop_samples": clock_drops, "hot_samples": hot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--accel", required=True, help="trecho do nome em /sys/class/devfreq (gpu no Jetson, npu no RK3588)")
    ap.add_argument("--cool-to", type=float, default=50.0)
    ap.add_argument("--cool-timeout", type=float, default=1200.0)
    ap.add_argument("--temp-limit", type=float, default=None, help="usado só em zonas sem trip point passivo")
    ap.add_argument("--ambient-c", type=float, default=None)
    ap.add_argument("--n-inferences", type=int, default=None, help="para energia por inferência")
    ap.add_argument("--out-dir", type=Path, default=Path("results/thermal"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("informe o comando depois de --")

    zones, accels, cpus, power = thermal_zones(), accelerators(args.accel), cpu_policies(), power_channels()
    if not accels:
        raise SystemExit(f"nenhum devfreq com '{args.accel}' em /sys/class/devfreq")
    accel_ref = {a: int(_read(d / "max_freq") or 0) for a, d in accels.items()}
    locked = {a: int(_read(d / "cur_freq") or 0) >= CLOCK_TOLERANCE * accel_ref[a] for a, d in accels.items()}
    governors = {a: _read(d / "governor") for a, d in accels.items()}
    governors.update({c: _read(p / "scaling_governor") for c, p in cpus.items()})
    power_mode = None
    try:
        power_mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    t0 = time.time()
    while max((sample(zones, {}, {}, {}).get(f"temp_{z}") or 0) for z in zones) > args.cool_to:
        if time.time() - t0 > args.cool_timeout:
            break
        time.sleep(2)
    cooled_s = round(time.time() - t0, 1)
    start_temp = max((sample(zones, {}, {}, {}).get(f"temp_{z}") or 0) for z in zones)
    print(f"[monitor] {args.label}: resfriado em {cooled_s}s, máx {start_temp:.1f} °C; "
          f"clock travado={locked}", flush=True)

    rows: list[dict] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            rows.append(sample(zones, accels, cpus, power))
            stop.wait(1.0)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    started = time.time()
    rc = subprocess.call(cmd)
    duration = time.time() - started
    stop.set()
    th.join()

    verdict = evaluate(rows, zones, accel_ref, args.temp_limit)
    temps = {z: max((r.get(f"temp_{z}") or float("-inf")) for r in rows) for z in zones} if rows else {}
    energy = {}
    for ch in power:
        pts = [(r["t"], r[f"W_{ch}"]) for r in rows]
        joules = sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(pts, pts[1:]))
        energy[ch] = {"mean_W": round(sum(w for _, w in pts) / len(pts), 3) if pts else None,
                      "energy_J": round(joules, 2),
                      "J_per_inference": round(joules / args.n_inferences, 5) if args.n_inferences else None}
    if not all(locked.values()):
        verdict["valid"] = False
        verdict["reasons"].append("clock do acelerador não estava travado no início")
    if rc != 0:
        verdict["valid"] = False
        verdict["reasons"].append(f"comando terminou com código {rc}")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_dir / f"{args.label}_{stamp}"
    if rows:
        with open(f"{base}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary = {"label": args.label, "cmd": cmd, "return_code": rc, "duration_s": round(duration, 1),
               "samples": len(rows), "cooldown_s": cooled_s, "start_max_temp_c": start_temp,
               "cool_to_c": args.cool_to, "ambient_c": args.ambient_c, "power_mode": power_mode,
               "governors": governors, "accel_ref_hz": accel_ref, "clock_locked_at_start": locked,
               "trip_points_c": {z: v["trip_c"] for z, v in zones.items()}, "max_temp_c": temps,
               "power": energy, "criterion": f"clock >= {CLOCK_TOLERANCE:.0%} da referência e zonas < trip passivo",
               **verdict}
    Path(f"{base}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    status = "VÁLIDA" if summary["valid"] else "INVÁLIDA: " + "; ".join(summary["reasons"])
    print(f"[monitor] {args.label}: {status}  (máx {max(temps.values(), default=float('nan')):.1f} °C) -> {base}.json")
    return 0 if summary["valid"] else 3


if __name__ == "__main__":
    sys.exit(main())
