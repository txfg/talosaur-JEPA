"""Summarise a dive from its logs: what the search found at which depth, how long each animal was
filmed and why it was left, and what the lamp did. These are the numbers that replace the
simulator's assumptions (docs/SEARCH.md §7).

  python -m talosaur.onboard.dive_report logs/guidance.jsonl logs/encounters.jsonl --out dive.md

Reads the onboard app's logs, replay telemetry (``scripts/replay.py``) or simulator telemetry
(``python -m talosaur.sim.run --telemetry DIR``). Runs anywhere; numpy only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from talosaur.utils.io import load_yaml

SEARCHING = ("SEARCH", "ACQUIRE")
BEFORE_START = ("SAFE", "ARMED")  # arm switch off, or armed and waiting to start (onboard/arming.py)


def load(paths: list[str | Path]) -> tuple[list[dict], list[dict]]:
    """-> (per-frame guidance telemetry sorted by time, encounter summaries without duplicates)."""
    frames, encounters, seen = [], [], set()

    def add_encounter(e: dict) -> None:
        key = (e.get("id"), e.get("t_start"), e.get("t_end"))
        if key not in seen:
            seen.add(key)
            encounters.append(e)

    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            kind = d.get("kind")
            if kind == "encounter":
                add_encounter(d)
            elif kind in (None, "guidance") and "state" in d and "t" in d:
                frames.append(d)
                if "encounter_summary" in d:
                    add_encounter(d["encounter_summary"])
    frames.sort(key=lambda d: d["t"])
    encounters.sort(key=lambda e: e.get("t_start", 0.0))
    return frames, encounters


def _pct(v, qs=(5, 50, 95)) -> str:
    return " / ".join(f"{np.percentile(v, q):.3g}" for q in qs) if len(v) else "-"


def summarise(
    frames: list[dict], encounters: list[dict], band_m: float = 10.0, dark_luma: float | None = None
) -> str:
    if not frames:
        return "No guidance telemetry found."
    t = np.array([f["t"] for f in frames], float)
    dt = np.clip(np.diff(t, append=t[-1]), 0.0, 1.0)  # a gap in the log counts at most 1 s
    total_h = max(float(t[-1] - t[0]), 1e-9) / 3600.0
    before = float(sum(d for f, d in zip(frames, dt) if f["state"] in BEFORE_START))
    hours = max(total_h - before / 3600.0, 1e-9)  # rates are per hour of mission
    out = [
        "# Dive report",
        "",
        f"- {len(frames)} frames over {total_h * 60:.1f} min ({len(frames) / max(total_h * 3600, 1e-9):.1f} fps)",
    ]
    if before > 0:
        starts = sum("mission_start" in (f.get("events") or []) for f in frames)
        out.append(
            f"- mission time {hours * 60:.1f} min in {starts} mission(s); {before / 60:.1f} min outside a mission "
            "(arm switch off, or armed and waiting to be in the water)"
        )

    state_s: Counter = Counter()
    for f, d in zip(frames, dt):
        state_s[f["state"]] += d
    out.append(
        "- time per state: "
        + ", ".join(f"{s} {100 * v / max(dt.sum(), 1e-9):.0f}%" for s, v in state_s.most_common())
    )

    # --- where the search found animals: search time and finds per depth band. A find is what the
    # planner counts: SEARCH -> ACQUIRE (the detection held for acquire_n of acquire_m frames)
    band_s: dict[float, float] = defaultdict(float)
    band_n: dict[float, int] = defaultdict(int)
    have_depth = False
    modes: Counter = Counter()
    for f, d in zip(frames, dt):
        found = "state:ACQUIRE" in (f.get("events") or [])
        if f["state"] not in SEARCHING:
            continue
        if f.get("search"):
            modes[f["search"].get("mode")] += d
        depth = (f.get("nav") or {}).get("depth_m")
        if depth is None:
            continue
        have_depth = True
        b = float(np.floor(depth / band_m) * band_m)
        band_s[b] += d
        band_n[b] += int(found)
    out += ["", "## Search", ""]
    if modes:
        total = sum(modes.values())
        out.append(
            "Search modes: " + ", ".join(f"{m} {100 * v / total:.0f}%" for m, v in modes.most_common()) + "."
        )
        out.append("")
    if have_depth:
        out += [
            "Finds per minute of search, by depth band (the planner's evidence):",
            "",
            "| depth band | search time (min) | finds | per minute |",
            "|---|---|---|---|",
        ]
        for b in sorted(band_s):
            m = band_s[b] / 60.0
            rate = f"{band_n[b] / m:.2f}" if m > 0 else "-"
            out.append(f"| {b:.0f}–{b + band_m:.0f} m | {m:.1f} | {band_n[b]} | {rate} |")
    else:
        out.append("No depth in the telemetry (no navigation input): detections cannot be placed by depth.")

    # --- water temperature by depth (from the depth sensor, if the bridge sends temp_c): the
    # thermocline often explains where the animals' layer sits (docs/TWILIGHT_ZONE.md 7.7)
    temps: dict[float, list[float]] = defaultdict(list)
    for f in frames:
        nv = f.get("nav") or {}
        if nv.get("depth_m") is not None and nv.get("temp_c") is not None:
            temps[float(np.floor(nv["depth_m"] / band_m) * band_m)].append(float(nv["temp_c"]))
    if temps:
        bands = sorted(temps)
        med = [float(np.median(temps[b])) for b in bands]
        out += ["", "Water temperature by depth (all states, descent included):", ""]
        out += ["| depth band | median temperature (°C) | samples |", "|---|---|---|"]
        out += [f"| {b:.0f}–{b + band_m:.0f} m | {m:.2f} | {len(temps[b])} |" for b, m in zip(bands, med)]
        steps = [(abs(med[i + 1] - med[i]), i) for i in range(len(bands) - 1)]
        if steps and max(steps)[0] >= 0.5:
            dT, i = max(steps)
            out += [
                "",
                f"Steepest change (thermocline): {dT:.1f} °C between {bands[i]:.0f}–{bands[i] + band_m:.0f} m "
                f"and {bands[i + 1]:.0f}–{bands[i + 1] + band_m:.0f} m.",
            ]

    # --- curiosity: looks at flashes and weak detections, and how many turned into an animal
    looks: dict[str, list[bool]] = defaultdict(list)  # cue -> did the look end in ACQUIRE?
    cur_cue, prev_looking = None, False
    for f in frames:
        cs = f.get("curiosity") or {}
        looking = bool(cs.get("looking"))
        if looking and not prev_looking:
            cur_cue = cs.get("cue") or "?"
            looks[cur_cue].append(False)
        if prev_looking and "state:ACQUIRE" in (f.get("events") or []):
            looks[cur_cue][-1] = True
        prev_looking = looking
    if looks:
        parts = [f"{len(v)} at {k} (of which {sum(v)} became a find)" for k, v in sorted(looks.items())]
        out += ["", "Curiosity looks: " + "; ".join(parts) + "."]

    # --- how long each animal was filmed and why it was left
    out += ["", "## Encounters", ""]
    if encounters:
        by_reason: dict[str, list[dict]] = defaultdict(list)
        for e in encounters:
            by_reason[e.get("reason", "?")].append(e)
        value = sum(float(e.get("value") or 0.0) for e in encounters)
        out += [
            f"{len(encounters)} encounters, footage value {value:.2f} ({value / hours:.2f} per hour; "
            "each animal is worth up to its novelty weight).",
            "",
            "| ended because | encounters | median time on animal (s) | median well framed (s) |",
            "|---|---|---|---|",
        ]
        for r, es in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            eng = [float(e.get("engaged_s") or 0.0) for e in es]
            good = [float(e.get("good_s") or 0.0) for e in es]
            out.append(f"| {r} | {len(es)} | {np.median(eng):.1f} | {np.median(good):.1f} |")
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for e in encounters:
            by_kind[e.get("behaviour") or "?"].append(e)
        out += [
            "",
            "How the animals behaved (it decides where the search goes next):",
            "",
            "| behaviour | encounters | median time on animal (s) | next |",
            "|---|---|---|---|",
        ]
        nxt = {"swarm": "loops around the spot", "mobile": "move on", "drifter": "hold its depth", "?": "-"}
        for k, es in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            eng = [float(e.get("engaged_s") or 0.0) for e in es]
            out.append(f"| {k} | {len(es)} | {np.median(eng):.1f} | {nxt.get(k, '-')} |")
        weights = [float(e["novelty_weight"]) for e in encounters if e.get("novelty_weight") is not None]
        if weights:
            out += ["", f"Novelty weights (5th / 50th / 95th percentile): {_pct(weights)}."]
    else:
        out.append("No encounters.")

    # --- appearance similarity and lamp decisions: calibration inputs
    sims = [f["reid"]["sim"] for f in frames if (f.get("reid") or {}).get("sim") is not None]
    skipped = sum(int((f.get("reid") or {}).get("skipped") or 0) > 0 for f in frames)
    out += ["", "## Calibration inputs", ""]
    out.append(
        f"- Appearance similarity of targets to animals already filmed (5th / 50th / 95th): {_pct(sims)}; "
        f"frames where a filmed animal was skipped: {skipped}. Compare with `encounter.same_sim`."
    )
    light = np.array([(f.get("cmd") or {}).get("light") or 0.0 for f in frames], float)
    dark = np.array([bool((f.get("lights") or {}).get("dark")) for f in frames])
    amb = [f["lights"]["ambient"] for f in frames if (f.get("lights") or {}).get("ambient") is not None]
    out.append(
        f"- Lamp: mean level {float((light * dt).sum() / max(dt.sum(), 1e-9)):.2f}; "
        f"judged too dark to search unlit {100 * float((dark * dt).sum() / max(dt.sum(), 1e-9)):.0f}% of the time."
    )
    if amb:
        note = f" (`lights.dark_luma` is {dark_luma})" if dark_luma is not None else ""
        out.append(
            f"- Ambient brightness with the lamp off (5th / 50th / 95th): {_pct(amb)}{note}. Set `dark_luma` "
            "below the values where the video still shows animals, above those where it is black."
        )
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "logs", nargs="+", help="guidance.jsonl / encounters.jsonl / replay or simulator telemetry"
    )
    ap.add_argument("--config", default=None, help="onboard config, for the thresholds used on the dive")
    ap.add_argument("--out", default=None, help="write the report here (markdown) as well as printing it")
    a = ap.parse_args(argv)
    g = ((load_yaml(a.config) or {}).get("guidance") or {}) if a.config else {}
    search, lights = g.get("search") or {}, g.get("lights") or {}
    frames, encounters = load(a.logs)
    text = summarise(
        frames, encounters, band_m=float(search.get("band_m", 10.0)), dark_luma=lights.get("dark_luma")
    )
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
