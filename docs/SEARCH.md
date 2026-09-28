# Searching, filming and moving on: behaviour design

How the vehicle decides **where to look**, **how long to film each animal**, **when to use
light**, and what it records. The design is grounded in the literature on twilight-zone animals,
AUV search and foraging theory (§1, cited), and compared in a closed-loop simulator (§7). The
simulator is a caricature of the ocean; your own dives replace its numbers.

## 1. What the evidence says

**How much to trust each finding.** Most journal sites are blocked from the environment this was
built in. Each finding therefore says how it was checked:

- **abstract**: the paper's abstract, read through search-engine extracts;
- **abstract, 3/3**: also confirmed against its quote by three independent reviewers;
- **search**: from search results only, which may paraphrase. Treat these as leads to read
  before relying on them;
- **source code** or **docs**: read in full.

Where a finding comes from another place (Monterey Bay, Arctic fjords), applying it to the Gulf
is an extrapolation, and says so.

### 1.1 Where the animals are, and when

- **By day the main layer is far below 200 m.** A ship-mounted 38 kHz ADCP in the northern Gulf
  (summers 2002–2003) found the main deep scattering layer at 450–550 m by day. The exception was
  near a front with Mississippi River plume water, where it rose to 200–300 m. *(abstract;
  [Kaltenberg et al. 2007](https://aquila.usm.edu/goms/vol25/iss2/1/))*
- **At dusk the migrators cross 100–200 m and end up above it.** At a northern Gulf site, part of
  the mesopelagic backscatter moved into the upper epipelagic, shallower than 100 m, while
  staying below the thermocline. At night the 100–200 m band held a more variable mix of species,
  consistent with animals also redistributing within the upper water column. This was a single
  30-hour survey in June 2011. *(abstract, 3/3;
  [D'Elia et al. 2016](https://www.sciencedirect.com/science/article/abs/pii/S0967063715301989))*
- **Timing.** A 10-year moored ADCP record in the western Gulf (2008–2018) gives the timing:
  - the shallowest migrating group starts rising about 1 h before sunset and starts descending
    about 2 h before sunrise;
  - backscatter is highest from the surface to about 100 m and at 400–600 m;
  - the effect of the moon is clearest at 1000–1200 m, not in the upper water column.

  *(abstract; [Ursella et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0079661121000495))*

**What the planner does with this.** Treat 100–200 m as a corridor:
- profile the whole range first;
- favour the corridor while the migrators cross it, starting earlier at dawn than at dusk;
- favour shallow bands at night;
- use no prior by day, and expect few animals at 100–200 m then.

### 1.2 How animals react to the vehicle

- **Light, red included, drives animals away.** Pelagic fish and zooplankton strongly avoided
  white, blue and red (575–700 nm) light on lowered instruments. Density fell by up to 99%, and
  avoidance reached 23–94 m from the light, depending on colour, brightness and community. This
  was measured with echosounders, so the lights did not bias the measurement. The sites were
  Arctic fjords in the polar night and coastal Newfoundland; the Gulf is an extrapolation.
  *(abstract; [Geoffroy et al. 2021](https://www.nature.com/articles/s41598-021-94355-6))*
- **Far-red is least disturbing, and species differ.** With no vehicle present (a fixed camera in
  Monterey Canyon), sablefish stayed in view longer under far-red (695 nm) than under red
  (685 nm) or white light. They often fled when white light came on, and sometimes when red did.
  Pacific grenadier did not flee in the same way. *(abstract;
  [Raymond & Widder 2007](https://www.int-res.com/abstracts/meps/v350/p291-298/))*
- **Speed matters.** A review of 48 demersal taxa found almost all react to underwater vehicles.
  Running an ROV at 0.5 instead of 0.25 m/s cut density estimates by 21–55% for some species.
  *(search; [Stoner et al. 2008](https://cdnsciencepub.com/doi/10.1139/F08-032))*
- **Fish flee vehicles.** Fish change their behaviour around AUVs, and a vehicle can predict and
  stay outside their flight-initiation distance. The work covers reef fish. *(search;
  [Cai et al. 2025](https://arxiv.org/abs/2506.11335), preprint)*

**What the planner does with this.**
- The lamp stays off whenever the camera can see, and is dim when it cannot (§4). Use far-red if
  possible.
- Approaches are slow (`controller.max_surge`).
- An animal that flees is let go rather than chased.

### 1.3 How to search

- **Find the layer on the first pass, then work it.** On a yo-yo through a layer, MBARI's AUVs find
  the peak on the first crossing and act at that level on the next. A drifting LRAUV followed a
  layer for four days. *(search;
  [Zhang et al. 2019](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2019.00415/full))*
- **Composite search.** For targets that stay where they are found, searching intensively after a
  find and giving up after a set time beats a Lévy walk. *(search;
  [Plank & James 2008](https://royalsocietypublishing.org/doi/10.1098/rsif.2008.0006))*
- **Plan in the water frame.** An AUV ran its survey pattern in the drifting frame of a
  drifter-marked water patch. This is the precedent for water-relative search plans.
  *(search; [Das et al. 2012](https://journals.sagepub.com/doi/10.1177/0278364912440736))*
- **Position error breaks fixed patterns.** Coverage has to account for pose uncertainty, because a
  submerged AUV's position drifts with no global fix. *(search;
  [Paull et al. 2014](https://ieeexplore.ieee.org/document/6907832/))*
- **What MBARI does.** Its midwater video transects run at about 0.5 m/s, for 10 min, at 100 m depth
  steps, and an AUV (i2MAP) now repeats them. *(search;
  [i2MAP, OCEANS 2016](https://ieeexplore.ieee.org/document/7761499/))*

**What the planner does with this.**
- Depth bands chosen from detections.
- Long legs with an intensive search after each find.
- A coverage map in the water frame that fades over time.
- Nothing depends on geographic x/y.

### 1.4 How long to film

- **Foraging theory has been applied to vehicles.** The patch model (the marginal value theorem)
  decides when to leave a patch; it has been proposed as the design rule for autonomous vehicles
  choosing tasks. *(search;
  [Andrews, Passino & Waite 2007](https://link.springer.com/article/10.1007/s10846-007-9138-9))*
- **The same rule has been used for attention across video streams.** Switch when the expected
  information rate falls below the average available elsewhere. *(search;
  [Napoletano et al. 2015](https://arxiv.org/abs/1410.5605))*
- **Gelatinous animals tolerate very long follows, so the vehicle has to decide to leave.**
  - MBARI's ML tracker followed a siphonophore for 5.27 h
    *(search; [Katija et al. 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Katija_Visual_Tracking_of_Deepwater_Animals_Using_Machine_Learning-Controlled_Robotic_Underwater_WACV_2021_paper.pdf))*.
  - Mesobot followed a giant larvacean autonomously for about 30–40 min. A pilot found its
    targets: search was not automated
    *(search; [Yoerger et al. 2021](https://www.science.org/doi/10.1126/scirobotics.abe1901))*.
  - Mesobot still damaged the larvacean's fragile house, so long close follows have a cost.

**What the planner does with this.** The leave rule of §3, with a hard cap, and early exits
for animals that flee or never give a good shot.

### 1.5 JEPA and world models

- **Full latent planning is far too slow for a Pi.** Planning with V-JEPA 2-AC by the
  cross-entropy method takes about 16 s per action on an RTX 4090. It uses a 300M-parameter
  predictor on a 1B-parameter encoder, is a goal-image reacher rather than an explorer, and was
  post-trained on under 62 h of robot video. *(abstract;
  [Assran et al. 2025](https://arxiv.org/abs/2506.09985))* DINO-WM's cross-entropy plans take
  tens of seconds on a desktop GPU. *(search; [Zhou et al. 2025](https://arxiv.org/abs/2411.04983))*
- **The nearest underwater precedent is a preprint.** DINO-Explorer uses a small action-conditioned
  predictor of frozen DINOv3 features. Its prediction error acts as "semantic surprise", with
  optical flow discounting the vehicle's own motion, which removed 45.5% of false positives. It
  was evaluated only for event triage and telemetry, not for closed-loop search, and not onboard.
  *(abstract; [Jin et al. 2026](https://arxiv.org/abs/2604.12933), preprint)*
- **A cheap novelty bonus.** How far an embedding lies outside the ellipse of embeddings seen so
  far is an exploration bonus that fits embedded compute. *(search;
  [Henaff et al. 2022, E3B](https://arxiv.org/abs/2210.05805))*
- **Small JEPA world models exist.** One of about 15M parameters plans in about 1 s, on a GPU.
  *(search; [Maes et al. 2026](https://arxiv.org/abs/2603.19312), preprint)*
- **Offloading the encoder.** Raspberry Pi's AI HATs could run the encoder and free the CPU for a
  predictor. A custom ViT must first be compiled with Hailo's tools. *(search;
  [Raspberry Pi AI HAT docs](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html))*

**What the planner does with this.** §6: embeddings for novelty and appearance now; a small
predictor trained on your own dive logs later; no full latent planning on the Pi.

### 1.6 Recording

- **picamera2's `SplittableOutput` switches files without losing frames.** Every encoded frame goes
  to the old or the new file, and by default the switch waits for a keyframe.
  - `split_output()` blocks until the switch, so it must be called from a thread other than the
    one delivering frames.
  - With no output attached, frames are silently dropped.

  The recorder starts with a file attached and switches in a background thread.
  *(source code;
  [splittableoutput.py](https://github.com/raspberrypi/picamera2/blob/main/picamera2/outputs/splittableoutput.py))*
- **The Pi 5 has no hardware H.264 encoder.** picamera2 uses software libx264 with the
  `ultrafast` preset and `zerolatency` tuning. *(source code;
  [libav_h264_encoder.py](https://github.com/raspberrypi/picamera2/blob/main/picamera2/encoders/libav_h264_encoder.py);
  docs: [rpicam-vid](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/camera/rpicam_vid.adoc))*
- **An unfinished MP4 is unreadable.** A normal MP4 that was never finalized cannot be decoded,
  which is why segments are MPEG-TS. *(docs;
  [FFmpeg muxers](https://github.com/FFmpeg/FFmpeg/blob/master/doc/muxers.texi))*
- **Other Pi cameras leave gaps.** FishCam, a Raspberry Pi underwater camera, restarts the camera
  for every file, leaving gaps of about 2 s. For continuous recording it recommends
  high-endurance microSD cards. *(docs; [FishCam](https://github.com/xaviermouy/FishCam))*

### 1.7 What could not be established

- **Densities.** How many animals per cubic metre a small camera meets at 100–200 m in the Gulf.
  The simulator's densities are guesses; your dive logs replace them (§7).
- **The camera.** Whether the Camera Module 3 sees anything at 100–200 m at night without light.
  Probably not, which is why the lamp rule measures instead of assuming (§4).
- **Gulf species and light.** How Gulf twilight-zone species respond to red and far-red light
  specifically.
- **Timing at your site.** Migration timing by season and at your dive site. The prior only
  nudges the planner; the detections decide.


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
     the descent about two hours before sunrise (§1.1).
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
- **Intensive mode.** Any find, and the end of every encounter, switch to a tight local
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
    for exploration in reinforcement learning (E3B, §1.5).
  - The patch tokens give each animal an appearance descriptor, used to recognise animals already
    filmed and to weight novel ones.
- **Realistic next.**
  - Ego-motion-compensated "surprise", with the gyro as the efference copy: prediction error of
    the next embedding, discounting the vehicle's own turning.
  - A tiny action-conditioned latent predictor trained on your own dive logs (embeddings plus
    commands). It could score a few candidate headings one step ahead.
- **Not realistic on a Pi 5.** Full latent planning in the style of V-JEPA 2-AC or DINO-WM, which
  runs a cross-entropy-method search over a large predictor, takes seconds per action on a desktop
  GPU (§1.5).

<!-- SIMULATOR -->
