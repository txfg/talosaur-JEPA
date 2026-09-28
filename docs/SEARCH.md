# Searching, filming and moving on: behaviour design

How the vehicle decides **where to look**, **how long to film each animal**, **when to use
light**, and what it records. The design is grounded in the literature on twilight-zone animals,
AUV search and foraging theory (§1, cited), and compared in a closed-loop simulator (§7). The
simulator is a caricature of the ocean; your own dives replace its numbers.

## 1. What the evidence says

**How much to trust each finding.** Most journal sites are blocked from the environment this was
built in. Each finding therefore says how it was checked:

- **full text** or **abstract**: how much of the paper could be read; abstracts came through
  search-engine extracts;
- **3/3** or **confirmed**: checked against the source by three independent reviewers, all or a
  majority agreeing;
- **search**: from search results only, which may paraphrase. Treat these as leads to read
  before relying on them;
- **source code** or **docs**: read in full.

Claims the reviewers rejected are left out, or named as rejected where the design had used them.

Where a finding comes from another place (Monterey Bay, Arctic fjords), applying it to the Gulf
is an extrapolation, and says so.

### 1.1 Where the animals are, and when

- **By day the main layer is far below 200 m.** A ship-mounted 38 kHz ADCP in the northern Gulf
  (summers 2002–2003) found the main deep scattering layer at 450–550 m by day. The exception was
  near a front with Mississippi River plume water, where it rose to 200–300 m. *(abstract, 3/3;
  [Kaltenberg et al. 2007](https://aquila.usm.edu/goms/vol25/iss2/1/))*
- **At dusk the migrators cross 100–200 m and end up above it.** At a northern Gulf site, part of
  the mesopelagic backscatter moved into the upper epipelagic, shallower than 100 m, while
  staying below the thermocline. This was a single 30-hour survey at one site in June 2011, so
  it shows the pattern, not what a given night holds. *(abstract, 3/3;
  [D'Elia et al. 2016](https://www.sciencedirect.com/science/article/abs/pii/S0967063715301989))*
- **Timing.** A 10-year moored ADCP record in the western Gulf (2008–2018, 120–1300 m) found that
  the shallowest migrating group starts rising about 1 h before sunset and starts descending
  about 2 h before sunrise. Migration also varies with the season and the moon. *(abstract, 3/3
  for the timing; [Ursella et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0079661121000495))*

**What the planner does with this.** Treat 100–200 m as a corridor:
- profile the whole range first;
- favour the corridor while the migrators cross it, starting earlier at dawn than at dusk;
- favour shallow bands at night;
- use no prior by day.

Do not assume the band is empty by day or at night: trawls in the Gulf have caught lanternfish
down to about 200 m by day. A claim that the band holds fewer animals outside the migrations was
rejected in verification. The detections decide.

### 1.2 How animals react to the vehicle

- **Light, red included, drives animals away.** Pelagic fish and zooplankton strongly avoided
  white, blue and red (575–700 nm) light on lowered instruments.
  - Density fell by up to 99%, and by more than 90% within 20 m of the light for every colour.
  - Avoidance reached 23–94 m, depending on colour, brightness and community.
  - The unlit instrument alone was avoided at 11–24 m, so the light more than doubled the distance.
  - Dimming the red light shortened it, from 54–66 m to 37–43 m.
  - Echosounders did the measuring, so the lights did not bias it.

  The sites were Arctic fjords in the polar night and coastal Newfoundland, so applying this to
  the Gulf is an extrapolation.
  *(full text, 3/3; [Geoffroy et al. 2021](https://www.nature.com/articles/s41598-021-94355-6))*
  The reviewers added that animals kept about 45 m from a lit ROV in Monterey Bay (Benoit-Bird et
  al. 2023). Responses vary by taxon: a 2022 review cites mesopelagic fish that avoided white,
  blue and green light but not red.
- **Far-red is least disturbing, and switching a light on is itself a trigger.** With no vehicle
  present (a fixed camera in Monterey Canyon), sablefish stayed in view longer under far-red
  (695 nm) than under red (685 nm) or white light. They often fled at the *onset* of white light,
  and sometimes at the onset of red. Pacific grenadier did not. *(abstract, confirmed;
  [Raymond & Widder 2007](https://www.int-res.com/abstracts/meps/v350/p291-298/))* A camera with
  an infrared-cut filter may see little at 695 nm, so measure yours first.
- **Speed matters.** A review of 48 demersal taxa found almost all react to underwater vehicles.
  Running an ROV at 0.5 instead of 0.25 m/s cut density estimates by 21–55% for some species.
  *(search; [Stoner et al. 2008](https://cdnsciencepub.com/doi/10.1139/F08-032))*
- **Fish flee vehicles.** Fish change their behaviour around AUVs, and a vehicle can predict and
  stay outside their flight-initiation distance. The work covers reef fish. *(search;
  [Cai et al. 2025](https://arxiv.org/abs/2506.11335), preprint)*

**What the planner does with this.**
- The lamp stays off whenever the camera can see, and is dim and steady when it cannot. It is
  never switched on suddenly, and not turned up near an animal (§4). Use far-red if possible.
- Approaches are slow (`controller.max_surge`).
- An animal that flees is never chased.

### 1.3 How to search

None of the sources in this section and the next could be opened or verified from here. They are
leads: the search and filming rules rest on them, on foraging theory, and on the simulator (§7).

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

- **Full latent planning is far too slow for a Pi, and it is not an explorer.**
  - V-JEPA 2-AC plans by the cross-entropy method and takes about 16 s per action on an RTX 4090.
  - It is model-predictive control toward a goal image, has no curiosity or information-gain
    objective, and was shown for predictions up to about 16 s.
  - It uses a 300M-parameter predictor on a 1B-parameter encoder, post-trained on under 62 h of
    robot video.

  *(paper and code, 3/3; [Assran et al. 2025](https://arxiv.org/abs/2506.09985))* A later version
  reports about 3 s per action on an A100, still far beyond a Pi. DINO-WM's cross-entropy plans
  take tens of seconds on a desktop GPU. *(search;
  [Zhou et al. 2025](https://arxiv.org/abs/2411.04983))*
- **Latent "surprise" for AUVs is unproven.** A preprint (DINO-Explorer) proposes the prediction
  error of an action-conditioned latent predictor as an AUV attention signal, with the vehicle's
  own motion discounted. Its claims were rejected in verification (0–3), so nothing here depends
  on it. *([arXiv 2604.12933](https://arxiv.org/abs/2604.12933), preprint)*
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
- **The Pi 5 has no hardware H.264 encoder.** picamera2 uses software libx264, and Raspberry Pi
  rate 1080p30 at about 30–40% CPU, on the cores that also run the model. The encoder's thread
  count can be capped; left alone, x264 uses 6 threads.
  *(source code and docs, 3/3;
  [libav_h264_encoder.py](https://github.com/raspberrypi/picamera2/blob/main/picamera2/encoders/libav_h264_encoder.py),
  [rpicam-vid](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/camera/rpicam_vid.adoc))*
- **A write error silently ends a picamera2 recording.** If writing a frame fails (a full card),
  `PyavOutput` closes its file and drops every later frame unless an error callback is set. The
  recorder sets one and switches to a new segment. *(source code, 3/3)*
- **An unfinished MP4 is unreadable, and closed files need a sync.** A normal MP4 that was never
  finalized cannot be decoded, while a fragmented MP4 plays up to its last fragment. Segments here
  are MPEG-TS, which has no index to finalize; the reviewers did not test its crash behaviour.
  Nothing in FFmpeg calls `fsync`, and Linux can hold about 30 s of written data in memory, so the
  recorder syncs each closed segment itself. *(docs and a reproduction, 3/3;
  [FFmpeg muxers](https://github.com/FFmpeg/FFmpeg/blob/master/doc/muxers.texi))*
- **What other Pi cameras get wrong.** FishCam, a Raspberry Pi underwater camera:
  - rebuilds the camera for every file, losing at least 2 s each time;
  - stops for good after 5 errors in a row, because nothing restarts it.

  Worth copying are its filenames that carry the settings, and its per-frame metadata. For
  continuous recording it recommends high-endurance microSD cards.
  *(source code, 3/3; [FishCam](https://github.com/xaviermouy/FishCam))*

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

5. **Never chase.** If the apparent size shrinks fast (`flee_rate`) for `flee_s` while the
   vehicle is not backing off, the animal is swimming away. The vehicle stops approaching and
   keeps it in view by turning only; the rule above keeps deciding from the footage it gets.
   - Many animals settle after a short burst and can be filmed on.
   - One that keeps going is soon out of sight. That encounter is logged as **`fled`**, and the
     animal is left alone for the cooldown, like one the vehicle chose to leave.
6. **Other exits:**
   - **`no_shot`**: no good shot within `giveup_s`, or the rule above gives up on an animal that
     was never well framed (it leaves after about 23 s with the default prior).
   - **`budget`**: the hard cap `max_s` is reached.
   - **`enough`**: the MVT rule above.

   `rule: fixed` keeps just `max_s`; the vehicle still never chases.

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
- **Searching, when it cannot** (night, or deep enough that the picture is black): a dim, steady
  level, `lights.search_dark: 0.3`.
  - "Cannot see" means the frame's mean brightness with the lamp off is below `dark_luma` (0.08).
    With auto-exposure, a scene the camera can expose sits near the exposure target, so a mean
    far below it means exposure and gain are at their limits.
  - It is only measured with the lamp off and settled for `settle_s` (1 s), so the lamp's own
    light never counts.
  - While the lamp is on for darkness, it is switched off every `check_s` (5 min) for
    `check_len_s` (3 s) to see whether ambient light is back (dawn, shallower water). This happens
    only while searching, never while filming.
- **Close to an animal: no change** (`lights.track: null`). Light coming on is itself what makes
  fish flee (§1.2), so the lamp is not switched on or up near an animal. Set `track` to a level
  to raise it anyway, for example for colour in footage.
- **Never a sudden switch-on.** Every increase is ramped over `ramp_s` (5 s, from off to full).
- **Use far-red** if you can.
- **`dark_luma` is a first guess.** Check it on the vehicle: the telemetry's `lights.ambient` shows
  the measured brightness, and `lights.dark` the decision.
- **Without a brightness input** (the simulator), `search` applies throughout. The simulator
  (§7) shows the trade-off at fixed levels: visibility grows with light, and so does avoidance.

## 5. Recording

Video is recorded continuously for the whole run and is never switched off (docs/PI5.md §6), so
nothing depends on the vehicle predicting the right moment. The encounter log indexes each animal
into the segments by file and time offset. What §1.6 found is built in:
- one encoder runs for the whole dive, and its output is switched between files at keyframes;
- a write error moves recording to a fresh file within seconds instead of silently ending it;
- closed files are synced to storage;
- the encoder's threads are capped, so it leaves CPU for the model;
- systemd restarts the app if it ever stops.

## 6. JEPA and world models: what is used, what is realistic

- **Used now.**
  - The pooled I-JEPA embedding gives a running "something new in view" novelty score. It is a
    Mahalanobis distance to recent embeddings, the same idea as the elliptical episodic bonus used
    for exploration in reinforcement learning (E3B, §1.5).
  - The patch tokens give each animal an appearance descriptor, used to recognise animals already
    filmed and to weight novel ones.
- **Possible, but unproven.**
  - "Surprise" as the prediction error of the next embedding, discounting the vehicle's own
    turning with the gyro. It has been proposed for AUVs, but those claims did not survive
    verification (§1.5). Test it offline on your dive logs before letting it steer.
  - A tiny action-conditioned latent predictor trained on your own dive logs (embeddings plus
    commands). It could score a few candidate headings one step ahead.
- **Not realistic on a Pi 5.** Full latent planning in the style of V-JEPA 2-AC or DINO-WM, which
  runs a cross-entropy-method search over a large predictor, takes seconds per action on a desktop
  GPU (§1.5).

## 7. Simulator results

`talosaur.sim` runs the real guidance code in a closed loop: the same `Guidance.step` as the Pi,
driving a simulated vehicle and camera. The world is a caricature of the twilight zone:

- **Water.** A 1 km square of water that wraps around, followed for 1.5 h at 5 frames per second.
- **The layer.** A migrating layer centred at 500 m by day and 70 m at night (standard deviation
  25 m). It moves over 1.5 h around sunrise and sunset.
- **Density.** 0.003–0.03 animals per m³ in the layer. Half the animals are in schools or swarms of
  about 40 within 8 m, which move together.
- **Species.**
  - Lanternfish, shrimp and squid migrate. They flee from the vehicle at 1.5–3 m, and from up to
    15–20 m further away with the lamp at full, each time in one 5 s burst.
  - Medusae and siphonophores stay at their depths and do not react.
- **Camera.** It sees animals out to 2 m with the lamp off and 5 m at full, fewer if small or far.
  It shows a false speck of "marine snow" every 50 frames.
- **Vehicle.** 0.6 m/s top speed, 0.3 m/s vertical, holding heading and depth setpoints.
- **Score.** Footage value per hour: the sum over animals of 1 − exp(−good / 20 s), where `good` is
  the time an animal was within 3 m, centred and detected. The first seconds of a new animal count
  most.

Every number is an assumption. Take the *ranking* of settings from it, not the values.

The results below are the mean ± standard error over simulated worlds; every setting ran in the
same worlds. The code is commit da35258. Each run is in `reports/sim/compare.jsonl`;
`python scripts/sim_compare.py` repeats the comparison (about 45 min on 4 cores).

**Where to search** (0.01 animals per m³, the leave rule of §3):

| search strategy | lamp while searching | night, from 21:00 | dusk, from 18:18 |
|---|---|---|---|
| turn in place (the previous behaviour) | off | 0.2 ± 0.2 | 0.0 ± 0.0 |
| turn in place | 0.5 | 3.4 ± 0.8 | 0.8 ± 0.5 |
| horizontal legs only | 0.5 | 6.6 ± 0.8 | 3.9 ± 0.5 |
| depth bands only | 0.5 | 7.2 ± 1.0 | 3.1 ± 0.7 |
| **adaptive (default)** | off | 2.0 ± 0.6 | 0.7 ± 0.3 |
| **adaptive (default)** | **0.5** | **6.6 ± 0.6** | **5.3 ± 0.6** |
| adaptive | 1.0 | 10.7 ± 0.8 | 4.8 ± 0.2 |

4 worlds each. Distinct animals well filmed per hour follow the same order. With a lamp of 0.5,
the adaptive planner filmed 7.5 per hour at night and 6.0 at dusk; turning in place filmed 3.7
and 1.0.

**How long to film each animal** (adaptive search, lamp 0.5, night):

| rule | scarce (0.003/m³, 8 worlds) | plentiful (0.03/m³, 4 worlds) | good footage per animal (scarce / plentiful) |
|---|---|---|---|
| fixed 60 s | 2.8 ± 0.3 | 14.7 ± 1.0 | 52 s / 49 s |
| fixed 180 s | 2.3 ± 0.3 | 11.7 ± 1.2 | 157 s / 132 s |
| **leave rule, `tau_s` 30 s (default)** | **3.0 ± 0.4** | **14.8 ± 0.1** | 90 s / 58 s |
| leave rule, `tau_s` 20 s | 3.1 ± 0.3 | 12.8 ± 1.2 | 66 s / 49 s |

**Navigation input** (adaptive, lamp 0.5, night, 0.01/m³): with depth and heading 6.6 ± 0.6;
without (scan and hop) 4.1 ± 0.9.

**What this says:**

1. **Move rather than turn in place.** The previous behaviour found almost nothing at dusk (0.8
   against 5.3). Paired over the same worlds, the adaptive planner beat turning in place by
   3.1 ± 0.8 at night and 4.5 ± 0.3 at dusk.
2. **Depth and horizontal search together matter most when the layer moves.** At dusk the
   adaptive planner scored 5.3, against 3.9 for legs alone and 3.1 for depth bands alone. At
   night, with the layer near the starting depth, the three are within noise of each other
   (6.6–7.2).
3. **The leave rule matched the best fixed budget without being tuned to the density.**
   - It was within noise of fixed 60 s where animals were scarce (+0.1 ± 0.4) and where they
     were plentiful (+0.1 ± 1.0).
   - It beat fixed 180 s by 0.7 ± 0.3 and 3.1 ± 1.2.
   - It did this by adapting: 90 s of good footage on each scarce animal, 58 s on each
     plentiful one. A fixed budget has to be tuned to a density you do not know before the dive.
   - An earlier run, with the lamp dimmed near animals, put the rule about 1 point ahead of
     fixed 60 s. Treat differences of that size as noise.
4. **Give the Pi depth and heading.** They raised the score by 2.5 ± 0.5 (about 1.6×).
5. **Lamp: some light beats none, but the simulator cannot say how much.**
   - A lamp of 0.5 beat none by 4.6 ± 1.1 at night and 4.6 ± 0.8 at dusk.
   - Full brightness scored higher still at night, and it kept doing so with the light-avoidance
     distance tripled to 60 m: 12.6 at full, 6.5 at 0.5 and 4.6 at 0.3.
   - The reason is how simulated animals flee: once, in a short burst, after which they tolerate
     the light. In the field trials of §1.2, density near a light stayed down by more than 90%
     for whole 10-minute trials, and a light coming on was itself a trigger to flee.
   - So the lamp defaults (off when the camera can see, dim and steady when it cannot) follow
     the field evidence, not the simulator. Tune them from the dive report's lamp log.

**The simulator found three bugs, each fixed and covered by a test:**

- **Marine snow counted as finds.** The planner took single-frame specks for detections. They
  drowned its per-band statistics and kept triggering the local search. Now a find is a detection
  the state machine accepts.
- **Filming looked like the bottom.** The depth profile's "can't get deeper" timer kept running
  while the vehicle filmed, so it mistook a long encounter for the bottom.
- **Animals were abandoned the moment they fled.** A first version of the leave rule left an
  animal as soon as it swam away. In the simulator that abandoned animals which settled after a
  short burst, and that version lost to fixed 60 s: 1.5 ± 0.3 against 2.7 ± 0.5 value per hour
  where animals were scarce. It was replaced by "never chase" (§3).

**Caveats:**

- **Lamp-off visibility.** The camera sees 2 m with the lamp off, which assumes ambient light or
  bioluminescence is enough. At 100–200 m at night a real camera probably sees nothing. Lamp-off
  search would then find nothing at all, which is why the lamp rule (§4) measures the picture
  rather than assuming.
- **Flight and light avoidance.** Both are single bursts in the simulator; real avoidance is
  sustained and starts further away (§1.2).
- **Densities, patches and species** are guesses.
- **Timing.** Migration in the simulator is symmetric around sunrise and sunset, unlike the planner's
  evidence-based windows (§2). The dusk runs start inside both.
- **Noise.** With 4–8 worlds per setting, differences under about 1 value per hour are within noise.

**Replacing the assumptions with dives.** Run `python -m talosaur.onboard.dive_report` on each
dive's logs (docs/PI5.md §15). The finds per minute in each depth band, the encounter durations
and reasons, and the ambient brightness it reports are the numbers to put back into the planner,
the leave rule and the lamp rule.
