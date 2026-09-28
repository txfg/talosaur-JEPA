# Searching, filming and moving on: behaviour design

How the vehicle decides **where to look**, **how long to film each animal**, **when to use
light**, and what it records. The design is grounded in the literature on twilight-zone animals,
AUV search and foraging theory (§1, cited), and compared in a closed-loop simulator (§7). The
simulator is a caricature of the ocean; your own dives replace its numbers.

<!-- EVIDENCE -->

## 2. What the planner does

Everything is built from what stays reliable underwater: **depth** (pressure sensor), **heading**
(compass / gyro) and **time**. Nothing uses geographic x/y, which drifts quickly without a DVL.

### Vertical: find the layer, then work it

1. **Profile.** One sweep through the allowed range (`search.min_depth_m` to `max_depth_m`)
   measures how often animals are detected in each 10 m band. This is the "find the peak on the
   first crossing" pattern MBARI's layer-tracking AUVs use.
   - **Shallower water is fine.** If the vehicle stops getting closer to the bottom of the range
     for `stall_s` (60 s), because of the bottom or the autopilot's own limit, the planner takes
     that depth as the bottom of the range. In a 40 m lake it then works the bands down to 40 m.
   - **With an altimeter** (`altitude_m` in the nav input) it never asks to go closer to the
     bottom than `min_altitude_m` (5 m).
2. **Choose bands by Thompson sampling.** Each band's detection rate has a Gamma–Poisson
   posterior. The planner samples from it, goes to the best sample, and stays `dwell_s` (3 min)
   before choosing again.
   - Well-performing bands get most of the time, but others are still checked now and then.
   - Old evidence fades (`halflife_s`, 20 min), because layers move.
   - A find is a detection the state machine accepts (`fsm.acquire_n` of `acquire_m` frames), so
     single-frame specks of marine snow do not count. Finds closer together than `onset_gap_s`
     count once.
   - Only search time counts. Time spent filming is not, so a band where animals are filmed for
     long is not mistaken for a band where they are hard to find.
3. **Time-of-day prior** (`diel_prior`):
   - While the migrators cross the corridor (100–200 m by default), its bands start with 3× the
     prior rate: from 1 h before to 1.5 h after sunset, and from 2.5 h before to 0.5 h after
     sunrise (`dusk_h`, `dawn_h`). In the Gulf the ascent starts about an hour before sunset and
     the descent about two hours before sunrise (§1).
   - At night, bands near `night_depth_m` are favoured.
   - By day, no prior.
   - Set `sunrise_h` / `sunset_h` to the times of sunrise and sunset at the dive site on the
     dive date, as the Pi's clock shows them.
   - **Check the Pi's clock before relying on the prior.** Out of network range, the Pi 5 keeps the
     time across a power-off only with a battery on its real-time-clock connector (not checked
     here). Run `date` after a cold start without network. If it is wrong, fit the battery, or
     set the clock before each dive; otherwise the time-of-day prior points at the wrong depths.
     The telemetry's `t` is the app clock, which does not depend on this.

### Horizontal: composite search

- **Extensive mode.** Long straight relocation legs (`leg_s`, 90 s, randomised ±30%) at a slow
  speed (`surge`), with large turns (`turn_deg`).
- **Intensive mode.** Any detection, and the end of every encounter, switch to a tight local
  search: short legs (`ars_leg_s`), sharper turns, slower. Animals come in patches, so where there
  is one there are likely more. If nothing turns up within `giveup_s` (2 min), it switches back to
  long legs.
- **Avoid water already searched.** The planner keeps a coarse map of cells it has passed through,
  dead-reckoned in the *water* frame from heading and commanded speed. Animals drift with the same
  water, so currents do not spoil it; only compass bias and speed error do. New legs prefer
  headings through cells not visited recently. The map fades (`coverage_halflife_s`) because both
  the animals and the position error move on.
- **Without navigation input:** scan (turn slowly for `scan_s`) and hop (go straight for `hop_s`),
  with no depth control.

The planner outputs heading and depth setpoints, for an autopilot with heading and depth hold, and
also the yaw and heave rates that steer toward them. Either kind of bridge works.

### Other behaviour

- **Leaving an animal.** RELEASE turns `release_turn_deg` (120°) away from the side the animal
  was on when a heading is available; otherwise it uses a timed turn. Then it swims on and starts
  an intensive search there for the rest of the patch. Animals already filmed are recognised and
  skipped (docs/PI5.md §12).
- **Approach speed** is capped (`controller.max_surge`), because fast approaches scatter animals.

## 3. How long to film each animal

`guidance.encounter.rule: mvt` applies the **marginal value theorem** of foraging theory to footage.

1. **Diminishing returns.** Footage of one animal is worth `w · (1 − exp(−g / tau_s))`, where
   `g` is the seconds it was *well framed*: centred, big enough, confidently detected.
   - `w = 1 + novelty_bonus · (1 − similarity to animals already filmed)`, so an animal unlike
     anything filmed so far on this run is worth up to 2×. Similarity is the model's appearance
     descriptor (docs/PI5.md §12), compared with every animal filmed this run, not only the
     ones remembered for the cooldown.
   - With `tau_s: 30`, about 95% of an animal's value is collected after 90 s of good footage.
2. **Expected gain of staying.** The *expected* marginal value rate is the recent fraction of
   time the animal was well framed × `w · exp(−g / tau_s) / tau_s`.
3. **What the mission earns on average.** R is the long-run rate from searching plus filming,
   estimated online as total value / total time. Until there is data, a prior of one animal per
   5 min, weighted as 30 min, stands in.
4. **Leave when staying is worth less than moving on**, i.e. when the expected rate falls below
   R, after `min_s`. The animal gets the time its footage is worth, given how easy other animals
   are to find:

   | one well-filmed animal every | framed seconds per animal | novel animal (w = 2) |
   |---|---|---|
   | 1 min | 21 s | 42 s |
   | 5 min | 69 s | 90 s |
   | 10 min | 90 s | 111 s |
   | 60 min | 144 s | 164 s |

5. **Other exits:**
   - **`fled`**: the apparent size shrinks fast (`flee_rate`) for `flee_s` while the vehicle is
     not backing off. The animal is leaving, and chasing it only disturbs it.
   - **`no_shot`**: no good shot within `giveup_s`, or the rule above gives up on an animal that
     was never well framed (it leaves after about 23 s with the default prior).
   - **`budget`**: the hard cap `max_s` is reached.
   - **`enough`**: the MVT rule above.

   `rule: fixed` keeps just `max_s`.

The telemetry shows `encounter.marginal_rate` and `long_run_rate` every frame. The encounter log
records `good_s`, `value`, `novelty_weight` and the reason each encounter ended, so the rule can
be checked against footage.

**Choosing `tau_s`** is a judgement about what the footage is for: 30 s suits a survey of what
lives where, and 120 s or more suits behaviour studies. The MVT then trades it off against how
easy other animals are to find.

## 4. Lights

The command carries `light` (0–1) for the vehicle's lamp driver. The rule (`guidance/lights.py`):
use as little light as possible, but enough to see.

- **Searching, when the camera can see by ambient light** (a lake by day, the upper twilight zone
  by day): lamp off (`lights.search: 0.0`).
- **Searching, when it cannot** (night, or deep enough that the picture is black): a dim level,
  `lights.search_dark: 0.3`.
  - "Cannot see" means the frame's mean brightness with the lamp off is below `dark_luma` (0.08).
    With auto-exposure, a scene the camera can expose sits near the exposure target, so a mean
    far below it means exposure and gain are at their limits.
  - It is only measured with the lamp off and settled for `settle_s` (1 s), so the lamp's own
    light never counts.
  - While the lamp is on for darkness, it is switched off every `check_s` (2 min) for
    `check_len_s` (3 s) to see whether ambient light is back (dawn, shallower water). Switching a
    lamp *off* does not disturb animals.
- **Close to an animal** (apparent size ≥ `near_size`): `lights.track: 0.3`.
- **Use far-red** if you can.
- **`dark_luma` is a first guess.** Check it on the vehicle: the telemetry's `lights.ambient` shows
  the measured brightness, and `lights.dark` the decision.
- **Without a brightness input** (the simulator), `search` applies throughout. The simulator
  (§7) shows the trade-off at fixed levels: visibility grows with light, and so does avoidance.

## 5. Recording

Video is recorded continuously for the whole run and is never switched off (docs/PI5.md §6), so
nothing depends on the vehicle predicting the right moment. The encounter log indexes each animal
into the segments by file and time offset.

## 6. JEPA and world models: what is used, what is realistic

- **Used now.**
  - The pooled I-JEPA embedding gives a running "something new in view" novelty score. It is a
    Mahalanobis distance to recent embeddings, the same idea as the elliptical episodic bonus used
    for exploration in reinforcement learning.
  - The patch tokens give each animal an appearance descriptor, used to recognise animals already
    filmed and to weight novel ones.
- **Realistic next.**
  - Ego-motion-compensated "surprise", with the gyro as the efference copy: prediction error of
    the next embedding, discounting the vehicle's own turning.
  - A tiny action-conditioned latent predictor trained on your own dive logs (embeddings plus
    commands). It could score a few candidate headings one step ahead.
- **Not realistic on a Pi 5.** Full latent planning in the style of V-JEPA 2-AC or DINO-WM, which
  runs a cross-entropy-method search over a large predictor, takes seconds per action on a desktop
  GPU (see §1).

<!-- SIMULATOR -->
