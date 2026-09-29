# Pre-dive checklist

Print this and tick it off every dive. The steps are grouped by when you do them. The arm switch,
the sensors and the software settings are described in docs/PI5.md §4, docs/SENSORS.md and
`configs/onboard/pi5.yaml`. The field plan behind the timing and the 200 m items is
docs/TWILIGHT_ZONE.md. Tags as there: **[E]** marks an engineering estimate to check on your own
vehicle.

To watch the vehicle's state from a laptop or over SSH while you work through the list:

```bash
tail -f logs/guidance.jsonl | python3 -c 'import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if d.get("kind") == "guidance":
        print(d["state"], d.get("arming"), d["cmd"], d.get("nav"), flush=True)'
```

## 1. The day before, on the bench

**Software and settings**
- [ ] The Pi runs the version you tested. Note `git log -1 --oneline` in the dive log.
- [ ] `configs/onboard/pi5.yaml` matches the site and date:
  - `search.sunrise_h` and `search.sunset_h`: sunrise and sunset at the site on the dive date, in
    the Pi's clock time (mind the time zone). The dusk and dawn behaviour depends on them.
  - `search.min_depth_m` and `max_depth_m`: inside the housing's rating and the water depth.
  - `search.min_altitude_m`: the closest the vehicle may come to the bottom (needs the sounder).
  - `arming`: `kind: gpio`, and `start_depth_m` from the float test (§3).
  - `nav: kind: udp` once the autopilot bridge sends depth and heading.
- [ ] A dry run on a recorded clip finishes without errors:
  `python -m talosaur.onboard.app --config configs/onboard/pi5.yaml --source video --video <clip>`.

**Clock**
- [ ] The Pi's clock is right (`date`) and the real-time-clock battery is fitted. With no network
  at the site the Pi keeps whatever time it has, and a wrong clock shifts the dusk and dawn
  windows. Set it before you leave, or from a phone hotspot.

**Storage and power**
- [ ] Free space for the whole dive plus the reserve: at 6 Mbit/s the recording takes 2.7 GB per
  hour, so a 10 h night needs about 27 GB plus `recording.min_free_mb` (2 GB). Check with `df -h`.
- [ ] The last dive's `recordings/` and `logs/` are copied off.
- [ ] Vehicle and Pi batteries are charged; the spare set too.

**Arm switch** (thrusters connected, propellers clear of hands and cables)
- [ ] Magnet off: `state` SAFE, `arming.switch` false, every `cmd` value zero, thrusters still.
- [ ] Magnet on: ARMED, and `arming.countdown_s` counts down. In air it then waits
  (`arming.wait: "in the water"`) and the thrusters stay still.
- [ ] Magnet off again: SAFE at once.
- [ ] Unplug the switch's lead: it must read as off (SAFE). A broken wire must never arm.

**Lamp**
- [ ] Test it in a bucket of water, not in air: many underwater LEDs rely on the water to cool
  them. Check yours.
- [ ] Each level lights, increases ramp up over `lights.ramp_s`, and the lamp is off in SAFE.
- [ ] The camera sees the far-red light (NoIR vs standard camera: docs/TWILIGHT_ZONE.md §7.6).

**Camera**
- [ ] Port clean inside and out. Focus (`source.controls.LensPosition`) set for the port.
- [ ] A short test recording plays to the end.

**Sensors** (docs/SENSORS.md)
- [ ] Depth reads about 0 in air. The water density matches the water you will dive in: fresh
  ~1000 kg/m³, sea ~1025. A wrong setting is 2.5 % off, 5 m at 200 m.
- [ ] Compass: calibrated inside the finished vehicle, then checked at four headings 90° apart
  against a hand compass, away from metal, with the thrusters off and then running. A few
  degrees of error is fine; tens of degrees means calibrate again.
- [ ] Sounder: reads the bottom of a tank or pool with good confidence.
- [ ] The telemetry's `nav` shows fresh depth, heading, altitude and temperature.

## 2. At the site, before closing the hull

- [ ] O-rings and sealing faces clean, with no hair, sand or scratches, and a thin film of
  silicone grease.
- [ ] A fresh desiccant pack inside. Close the hull in dry air: humid air fogs the port in cold
  deep water [E].
- [ ] Vacuum-test every enclosure: pull the vacuum its maker specifies and leave it at least
  15 min [E: common practice]. If the gauge moves at all, find the leak before diving.
- [ ] Penetrators and cable glands tight; connectors fully seated.

## 3. Trim (ballast)

- [ ] **Trim in the water you dive in.** Sea water is about 2.5 % denser than fresh water, so in
  the sea the vehicle gains about 0.025 kg of lift for every litre it displaces. A vehicle
  displacing 20 L needs about 0.5 kg more ballast in the Gulf than in a lake. MeCO's water-ready
  guide likewise adds a weight for salt water.
- [ ] **Slightly positive.** With the thrusters off it should rise slowly. Then anything that stops
  the thrusters (magnet off, a crash, a flat battery) brings it up. Keep the margin small: the
  thrusters must hold the extra lift down while hovering, and their noise and wash are what
  animals feel (docs/TWILIGHT_ZONE.md §5.2).
- [ ] Level in pitch and roll with the thrusters off, so the camera is level.
- [ ] **Float test for the start gate.** Let the vehicle float at rest and read `nav.depth_m`.
  Set `arming.start_depth_m` a little below that reading, so floating counts as "in the water",
  and well above the reading in air.
- [ ] After each dive, look at the vertical thrust the autopilot needed to hold depth while
  hovering. Deep water is colder and denser and the hull compresses a little, so the trim at
  depth is not the trim at the surface [E]. Adjust the ballast for the next dive.

## 4. Launch

- [ ] Power on. Within a minute the telemetry shows frames, `recording` true, the usual fps,
  `state` SAFE and a fresh depth near 0.
- [ ] Log the GPS position and time at the launch point (docs/TWILIGHT_ZONE.md §7.8).
- [ ] Surface beacon and strobe on: recovery at dusk and dawn happens in the dark.
- [ ] **Magnet on** (ARMED). Lower the vehicle into the water, let go, keep clear of the thrusters.
- [ ] It starts when the countdown is over and it has been in the water for `start_hold_s`, then
  descends. If it does not, take the magnet off, lift it out and check `arming.wait`. `"no depth"`
  means the depth input is missing.

## 5. Recovery

- [ ] **Magnet off before handling:** SAFE, the thrusters stop. Recording carries on.
- [ ] Log the GPS position and time at recovery.
- [ ] Stop the service before switching off: `sudo systemctl stop talosaur` closes the last
  video segment. The `.ts` segments survive a power cut anyway.

## 6. After the dive

- [ ] After salt water, rinse with fresh water: housing, thrusters, sounder face, magnet holder,
  O-ring grooves. Then dry.
- [ ] Open the hull and check for water, condensation and O-ring damage.
- [ ] Copy `recordings/` and `logs/`, then run the dive report:

  ```bash
  python -m talosaur.onboard.dive_report logs/guidance.jsonl logs/encounters.jsonl \
      --config configs/onboard/pi5.yaml --out reports/dives/<date>.md
  ```
- [ ] Note in the dive log:
  - mission time;
  - finds per depth band, and the thermocline;
  - lamp use and the ambient brightness (for `lights.dark_luma`);
  - appearance similarities (for `encounter.same_sim`);
  - trim.
- [ ] Label each encounter's animal group from the video (docs/TWILIGHT_ZONE.md §2).
- [ ] Charge the batteries.

## 7. Extra for 200 m in the Gulf

- [ ] Every housing, window, penetrator, connector, lamp, sounder and depth sensor is rated and
  pressure-tested to at least 1.5× the working depth, i.e. 300 m (docs/TWILIGHT_ZONE.md §7.8).
- [ ] The depth sensor is a 30 bar type, not a 2 bar one (docs/SENSORS.md §3).
- [ ] An independent way up that needs no software or electronics: a drop weight on a timed
  galvanic release, or a burn wire with its own timer [E].
- [ ] Surface beacon (GPS with radio or satellite) and a strobe.
- [ ] Recovery planned downstream of the current. A boat CTD cast before the dive if possible.
- [ ] Trim redone for sea water (§3).
