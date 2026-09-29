# Sensors: what the vehicle needs, and the plan for each

The vision software reads the vehicle's own state through one input: JSON datagrams on UDP port
14601 (docs/PI5.md §5). The fields are depth, heading, turn rate, altitude above the bottom and
water temperature. This page covers each sensor behind those fields:
- what the software does with it;
- what to buy (requirements first, then candidates to check);
- how to mount and calibrate it;
- how its readings reach the Pi.

The autopilot is still undecided (docs/PLAN.md §11, item 3), and it decides where most sensors
plug in (§6).

**Evidence tags** as in docs/TWILIGHT_ZONE.md:
- **[V]** verified in this project;
- **[L]** literature or product facts cited from knowledge and not re-read here, because the
  publishers' and manufacturers' sites were blocked. Check them before relying on them.
- **[E]** an estimate for this vehicle, to test.

**Check every product rating on its datasheet before buying.**

## 1. Summary

| sensor | the software uses it for | needed | status |
|---|---|---|---|
| depth (pressure) | depth bands, the depth profile, the start gate, the dive report | yes | decided |
| compass (magnetometer) | leg headings, the turn away from an animal, the map of water already searched | yes | decided |
| IMU (gyro, accelerometer) | turn rate for holding a heading; tilt for the compass and the sounder | yes | decided |
| **echosounder (altimeter)** | keeping `min_altitude_m` off the bottom | lakes: yes. Gulf: only where the bottom is within range | **to add** |
| water temperature | the thermocline in the dive report | recommended | free with most depth sensors |
| leak sensor | surfacing on a leak (the controller, §5) | yes | to add with the autopilot |
| battery voltage and current | surfacing with a reserve (the controller, §5) | yes | to add with the autopilot |
| arm switch (magnetic reed switch) | SAFE / ARMED / RUN (docs/PI5.md §4) | yes | built |

## 2. Echosounder (altimeter): the addition

**Why.**
- The search keeps `search.min_altitude_m` (5 m) off the bottom, but only when it knows the
  altitude.
- Without a sounder, the only way it finds the bottom is the depth profile stalling: it asks to go
  deeper and the depth stops changing. That means it is sitting on the bottom. Stirred-up silt
  ruins the video, and the vehicle can snag.
- In a lake the bottom is always within reach, so the sounder is needed there.
- In the Gulf over deep water the bottom is far out of range. The sounder then only matters near
  the shelf edge or over banks.

MeCO (University of Minnesota) carries an echosounder, which its documentation says "detects
obstacles and maps the seafloor" [V: MeCO's public documentation, read for the comparison in this
project].

**What it must do** [E]:
- **Depth rating** at least 1.5× the working depth: 300 m for 200 m work (docs/TWILIGHT_ZONE.md §7.8).
- **Range** at least 30 m. At a descent of 0.5 m/s that gives almost a minute's warning before
  the 5 m limit.
- **Minimum range** 0.5 m or less.
- **Beam**: a single narrow beam, about 20–30°, pointing straight down.
- **Rate**: 2–10 readings per second. Sound needs only 40 ms for a 30 m round trip, so the device
  sets the rate, not physics.
- **Output**: distance plus a confidence or signal strength, over serial or USB.

**Candidates** (check every figure on the datasheet):
- **Blue Robotics Ping2** is the usual low-cost single-beam echosounder on small vehicles, and
  ArduSub can use it as a rangefinder [L]. Check its depth rating, range, beam width and protocol.
- **Industrial altimeters**, from makers such as Tritech, Imagenex and Valeport, are the
  deeper-rated and more expensive option [L].

**Mounting.**
- Point it straight down, clear of the hull, thruster wash and bubbles.
- When the vehicle pitches or rolls, the beam meets a flat bottom at a slant. The altitude is then
  about the measured range × cos(tilt). The bridge should correct for this with the IMU's roll and
  pitch [E].

**How it reaches the software (built).**
- The bridge, or a small reader for the sounder, sends `{"altitude_m": 12.3}` to UDP 14601.
- It can be a separate process from the one sending depth and heading. The input merges fields
  from several senders, and each field goes stale on its own after 1.5 s.
- Send the altitude only when the sounder's confidence is good; otherwise send `null` [E]. A wrong
  altitude either keeps the vehicle needlessly shallow or, worse, tells it the bottom is far away.
- The planner never asks to go deeper than `depth + altitude − min_altitude_m`.
- If the sounder goes silent, the last limit is kept for `search.floor_hold_s` (300 s). A sounder
  can go silent when it is too close to read, after a bad ping, or over a steep drop.
- The telemetry shows the limit as `search.floor_m`.
- A downward sounder does not see obstacles ahead, such as walls, cliffs or rig legs. That needs a
  forward-looking sonar, which is not in this plan.

**Sound and the animals.**
- Small echosounders ping at ultrasonic frequencies; the Ping2 at about 115 kHz [L].
- Most fishes hear only up to a few kHz and will not hear it [L: Popper & Fay 2011].
- **The exceptions are some herrings.** Shads, alewife and menhaden detect ultrasound [L: Mann et
  al. 1997, 2001]. Alewives avoid high-frequency sound strongly enough that it has been used to
  keep them out of power-plant water intakes [L: Dunning et al. 1992].
- Toothed whales and dolphins hear it too [L].
- Whether any twilight-zone fish or squid reacts to 100+ kHz pings was not checked.

What this means in practice [E]:
- Ping no faster than needed.
- In midwater, with the bottom out of range, the bridge can slow the pinging down or stop it.
- In lakes with alewife or shad, compare how close fish let the vehicle come with the sounder on
  and off. That is one of the first-dive measurements.

## 3. Depth sensor

- **Range.** A 30 bar sensor covers about 290 m of sea water: 29 bar above atmospheric pressure, at
  about 0.1 bar per metre. A 2 bar sensor stops at about 10 m, which is pool-only. Blue Robotics'
  Bar30 (MS5837-30BA) is the common 30 bar choice [L]. It must be rated and tested to 1.5× the
  working depth, like everything else.
- **Water density.** Depth is gauge pressure ÷ (density × g).
  - Fresh water is about 1000 kg/m³, sea water about 1025.
  - A wrong setting is 2.5 % off: 5 m at 200 m, half a depth band.
  - Set it in the bridge or autopilot for the water you dive in.
- **Zero** at the surface before the dive, in air or floating. A change in the weather of a few hPa
  moves the zero by a few cm (1 hPa ≈ 1 cm of water).
- **Temperature.**
  - Most pressure sensors also report their temperature. Send it as `temp_c`.
  - The dive report shows the water temperature by depth band and the steepest change (built).
    The thermocline often explains where the animals' layer sits (docs/TWILIGHT_ZONE.md §7.7).
  - The reading lags a few seconds behind the water [E]. At 0.2 m/s that blurs the profile by
    about a metre, which is fine for a thermocline.
- **Start gate.** The mission starts only once the depth reads at least `arming.start_depth_m`
  for `start_hold_s`. Set it from a float test (docs/PREDIVE.md §3).
- **Rate.** 5–20 readings per second is plenty [E].

## 4. IMU and compass

**What heading is used for:**
- transect legs;
- the turn away from an animal after filming it;
- the map of water already searched, which is dead-reckoned from heading and commanded speed.

The turn rate from the gyro helps hold a heading.

**What to use.**
- A tilt-compensated heading (accelerometer plus magnetometer), fused with the gyro, i.e. an
  AHRS. MeCO uses one, the MicroStrain 3DM-CV7 [V: MeCO documentation].
- A separate IMU and magnetometer work too, if calibrated well.

**Mount and calibrate.**
- Mount the magnetometer as far as possible from thrusters, motor cables, batteries and steel.
- Calibrate it inside the finished vehicle, with batteries in and hull closed (hard and soft iron).
- Then check it with the thrusters running: motor currents bend the field, so the error changes
  with thrust (docs/TWILIGHT_ZONE.md §7.7).
- **Check:** four headings 90° apart against a hand compass, away from metal.
  - A few degrees of error is fine.
  - Tens of degrees means calibrate again.
  - A heading error bends the legs and the search map; it does not break the search [E].
- Magnetic declination does not matter: the software only needs a heading that stays consistent
  during the dive.

## 5. Leak, battery, and the controller's failsafes

These sensors do not feed the vision software. They belong to the controller that drives the
thrusters (the autopilot, or the bridge's microcontroller), which must act on them even if the Pi
has crashed:
- a leak sensor in each enclosure → surface;
- battery voltage and current → surface with a reserve;
- a maximum depth → abort upward;
- commands from the Pi older than about 0.5 s → zero thrust (docs/PI5.md §5); a longer silence →
  surface [E];
- a mission timer → surface.

The controller specification follows the autopilot choice. For 200 m, add a way up that needs no
electronics at all (docs/PREDIVE.md §7).

## 6. Where each sensor connects

The autopilot choice decides this.

- **Through an autopilot, for example ArduSub on a flight controller.**
  - The depth sensor, the IMU and compass, and the sounder plug into the autopilot.
  - A bridge forwards depth, heading, turn rate, altitude and temperature to UDP 14601.
  - The autopilot also does the depth and heading hold and the failsafes of §5.
  - Fewer drivers to write.
- **Straight into the Pi.**
  - Depth sensor and IMU over I2C (GPIO 2/3), sounder over USB serial or the UART (GPIO 14/15).
  - A small reader per sensor sends its fields to UDP 14601; fields merge.
  - More of our own code, and a microcontroller must still own the thrusters and the failsafes:
    the Pi should never be the only safety layer.

**Recommendation [E]: go through an autopilot.** The arm switch uses GPIO 17 (header pin 11,
ground on pin 9), clear of I2C and the UART.

## 7. Decisions and first-dive checks

**Decisions:**
1. **The autopilot route.** It decides where each sensor connects (§6).
2. **The sounder model.** For the Gulf, the depth rating must be at least 300 m.
3. **The depth sensor.** 30 bar.
4. **Compass hardware.** An AHRS, or a separate IMU and compass.

**First-dive checks:**
- how close fish let the vehicle come with the sounder on and off, in a lake with alewife or shad;
- the compass error with the thrusters running;
- the temperature lag, from a descent and ascent through the thermocline.

## 8. References

**Literature cited from knowledge [L]** (not re-read here; check before relying on them):
- Dunning D.J. et al. 1992. Alewives avoid high-frequency sound. *North American Journal of
  Fisheries Management* 12:407–416.
- Mann D.A., Lu Z., Popper A.N. 1997. A clupeid fish can detect ultrasound. *Nature* 389:341.
- Mann D.A. et al. 2001. Ultrasound detection by clupeiform fishes. *Journal of the Acoustical
  Society of America* 109:3048–3054.
- Popper A.N., Fay R.R. 2011. Rethinking sound detection by fishes. *Hearing Research* 273:25–36.

**Read in this project [V]:** the MeCO-AUV documentation (github.com/MeCO-AUV/MeCO-Documentation
and its wiki): sensor suite, architecture, water-ready procedure.
