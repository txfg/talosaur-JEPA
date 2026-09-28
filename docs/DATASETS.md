# Dataset sources: verification log

**Checked:** 2026-09-28.

**How verified.** This sandbox's egress policy blocks almost every primary data host (fathomnet.org, ncei.noaa.gov, oceanexplorer.noaa.gov, data.qld.edu.au, storage.pawsey.org.au, kaggle.com, zenodo.org, lila.science, huggingface.co, oceannetworks.ca, *.inf.ed.ac.uk). Facts were therefore checked in three ways:

- **READ:** I read the primary file myself. Usually a GitHub README or LICENSE, the project page source, or the PyPI package source.
- **SEARCH:** seen only in web-search results for the official page.
- **UNVERIFIED:** could not be confirmed.

No URL below was guessed. Every fetcher in `talosaur.data.sources` will print the source's license and require `--accept-license <id>` before downloading. It also records URL, checksum, license, and retrieval date in a manifest. **Before a first full download, re-read the license page yourself**; licenses change.

---

## Summary

| Source | Role in this project | Conditions | Labels | License | Verified |
|---|---|---|---|---|---|
| **FathomNet** (MBARI) | pretraining; patch probe (boxes) | deep and midwater, ROV-lit, dark falloff, particles | boxes plus WoRMS taxonomy | per upload: images CC0 / BY / BY-NC / BY-NC-ND; **ToS allows ML training**; no redistribution | client/API READ; ToS SEARCH |
| **NOAA Ocean Exploration**, Okeanos Explorer (EX) video | pretraining; V-JEPA clips | deep ROV dives including Gulf of Mexico; lit scenes, marine snow | none | **public domain** (EX cruises only) | SEARCH |
| **DeepFish** (JCU) | **frame probe** (fish/no-fish); patch probe (masks) | tropical coastal, daylight, mostly clear | 39,766 frame labels, 3,200 points, 620 masks | **CC BY 4.0** | READ |
| **Kakadu freshwater fish** | patch probe; pretraining; **freshwater** | freshwater billabongs | 82,904 COCO boxes, 23 species | **CC BY 4.0** | READ (GitHub README) |
| **MIT Sea Grant River Herring** (LILA) | **frame probe**; **dark slice** | freshwater river, **~35% night** | ~262k frames, ~162k reviewed empty | **CDLA-Permissive-1.0** | SEARCH |
| **Brackish** (AAU) | **murky-slice** evaluation (boxes) | turbid brackish water, LED-lit, fixed camera | ~35.6k boxes, 6 classes | **conflicting: CC BY-SA 4.0 / CC BY 4.0 / CC BY-NC-SA 4.0** | secondary only |
| **OzFish** (AIMS) | pretraining frames only | coastal BRUV video | weak, non-exhaustive boxes | CC BY (3.0 AU vs 4.0 inconsistent) | READ |
| **Ocean Networks Canada** | pretraining (dark, lights on/off) | deep fixed cameras, 393–985 m | timestamped observations | ONC-owned: CC BY 4.0; **partner devices vary** | SEARCH |
| **Your pool/lake dives** | **murky freshwater test set** | the real target domain | you label | yours | — |

**Deferred** (licence or relevance problems): SEAMAPD21 (Gulf of Mexico, no licence stated), Fish4Knowledge, UIEB, LSUI, TrashCan, SUIM, RUOD, Schmidt Ocean Institute, Roboflow Aquarium, FishTrack23, Labeled Fishes in the Wild. See §10.

---

## License restrictions to be aware of

1. **FathomNet: training is allowed, redistribution is not.**
   - Images carry CC0, CC-BY, CC-BY-NC or CC-BY-NC-ND, chosen by the contributor.
   - The Terms of Use add: *"Notwithstanding any contrary provisions of any license you select, you acknowledge and agree that all Visual Content submitted to the FathomNet Database may be used for the training and development of machine learning algorithms."* (SEARCH)
   - The competition terms add: *"all of the images may be used for training and development of machine learning algorithms for commercial, academic, and government purposes."* (READ)
   - Because of the NC/ND licences, **we never publish FathomNet images, crops, or augmented frames**. The repo stores only IDs and URLs.
   - Attribution: FathomNet/MBARI, doi:10.1038/s41598-022-19939-2.
2. **Brackish is probably share-alike (SA).** Any frames or labels we redistribute would have to carry the same licence. We won't redistribute them. **Please read the licence shown on the Kaggle page and tell me what it says.**
3. **Model weights trained on NC, ND, or SA data.** Whether weights count as "adapted material" is legally unsettled. The index stores `license` and `commercial_ok` for every image, so a commercial-clean training set can be rebuilt with one config flag. **Get legal review before any commercial release of weights.**
4. **NOAA portal content is not all NOAA's.** The portal also carries E/V Nautilus (Ocean Exploration Trust) footage under separate terms, and items captioned "copyright". The NOAA fetcher only accepts `EX…` cruise files.
5. **Ocean Networks Canada licences vary by device.** Partner-owned devices may be NC or stricter, so the fetcher reads the licence per device and skips anything that isn't CC BY.
6. **Not used by default because of restrictive terms.** UIEB (non-commercial, academic only, *"re-distribution… forbidden"*), LSUI (academic only), TrashCan (academic; commercial use needs JAMSTEC permission), Schmidt Ocean Institute (CC BY-NC-SA, by request).

---

## 1. FathomNet (MBARI): deep-sea and midwater ROV imagery with boxes

| | |
|---|---|
| Relevance | Mostly MBARI ROV imagery (Ventana, Doc Ricketts, Tiburon, MiniROV; 0–1300 m) plus NOAA Okeanos frames. Dark water with the ROV's own lights, falloff, and particles: the closest public match to the 100–200 m twilight zone. *"more than 400k images and 1M expert-curated bounding boxes, as of Feb 2026, across thousands of marine taxonomic, geologic, and equipment classes"* (READ, fgvc-comp-2026 README). |
| Access | `pip install fathomnet` (fathomnet-py 1.10.1, MIT; READ from PyPI). Read-only REST API at `https://database.fathomnet.org/api`, no login needed. Paging: `images.count_all()`, `images.find_all(Pageable(size, page))`, `images.find(GeoImageConstraints(limit, offset, minDepth, ownerInstitutionCodes))`, `images.find_by_concept(concept)`. Each record includes `url`, `boundingBoxes`, `depthMeters`, lat/lon. Images are fetched **one URL at a time**; there is no bulk archive. |
| Rate | Not documented. FathomNet's own 2026 downloader uses 1–5 workers with backoff on timeouts (READ). Our fetcher defaults to 4 workers with exponential backoff and resume. |
| Animal / not animal | A concept is an animal if `"Animalia" in worms.get_ancestors_names(concept)` (READ). Non-animal concepts exist, such as `'2G Robotics structured light laser'` and `'55-gallon drum'`, and are used as "not animal" labels. |
| Per-image license | Stored on the upload batch: `imagesetuploads.find_by_image_uuid(uuid)[0].darwinCore.license` (plus `rightsHolder`, `ownerInstitutionCode`) (READ). We cache it per upload. **Don't trust** the COCO exporter's single generic "FathomNet" license field (READ). Newer `imageLicense`/`annotationLicense` fields are planned upstream. The client silently drops unknown fields, so we read the raw JSON. |
| Size / format | PNG, about 3.8 MB per 1080p frame, so **about 1.5 TB if downloaded raw**. The fetcher resizes on the fly (long side ≤ 512 px for eval images, ≤ 320 px for pretraining) and never keeps originals. That's about 25–40 GB. |
| Gotchas | Many images are frame grabs from dives, so near-duplicates are common (dedup and grouped splits handle this). Subjects are often centred. Resolutions mix SD and HD. **Annotations are not exhaustive**, so a region without a box is not a guaranteed negative. |
| Competition subsets | FGVC 2023 (290 categories), 2025 (79 categories), and 2026 (positive-unlabelled detection) all inherit FathomNet's terms and ship as COCO JSON plus per-URL downloads (READ). |
| Open question | Read https://www.fathomnet.org/terms in full (blocked here), especially on distributing model weights. |

## 2. NOAA Ocean Exploration: Okeanos Explorer ROV video

| | |
|---|---|
| Relevance | Hours of deep ROV video including **Gulf of Mexico** expeditions (e.g. EX1711; SEARCH). Lit subjects, dark water column, marine snow. Unlabelled, so used for pretraining and V-JEPA clips. |
| Access | NCEI Ocean Exploration Video Portal: https://www.ncei.noaa.gov/access/ocean-exploration/video/ (SEARCH). **Low-res H.264 files download directly.** Full-res ProRes is ordered and arrives as emailed links that expire after about 96 hours. **No API, bulk download, or cloud mirror found.** Workflow: you export a list of segment URLs from the portal into `data_sources/noaa_oer_manifest.csv`, and the fetcher processes it. Contact: oer.video@noaa.gov. |
| License | *"All video on the video portal is in the public domain"*, credited to "NOAA Ocean Exploration" (SEARCH). Excluded: items captioned "copyright" and non-NOAA (e.g. E/V Nautilus) footage. **Only `EX…` cruises are used.** |
| Size / format | ROV HD camera: 1080i ProRes 422 at 147 Mbps in 5-minute ~5 GB segments, about **66 GB per hour per camera**. Interlaced, so the pipeline deinterlaces before sampling. The **low-res H.264 files are enough** for ≤ 320 px training frames and are the default. |
| Overlays | Archived EX1708 ROV frames have **no burned-in text** (READ, viewed). Other products are unverified, so the pipeline auto-detects and masks static overlays. |

## 3. DeepFish (James Cook University): core fish/no-fish evaluation set

| | |
|---|---|
| Relevance | 20 habitats in tropical Australian coastal waters (reef, seagrass, mangrove). 1920×1080 frames filmed *"during daylight hours and in relatively low turbidity periods"*. Mostly the "clear/dim daylight" reference, and the best source of **frame-level fish / no-fish labels plus masks**. |
| Access | `DeepFish.tar` (7.1 GB), SHA-256 `8acabb8a314fa8eb45fa9cc6829c84bdc127b1f8f605f2e152304e98b6dbb46c`. Original: `http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/`. Mirror: `hf download Alzayats/DeepFish DeepFish.tar DeepFish.tar.sha256 --repo-type dataset --local-dir .` Record: doi:10.25903/5f617fb6d6e0e. (READ, GitHub README and project page.) |
| License | **CC BY 4.0** for the data, MIT for the code: *"Dataset: CC BY 4.0 · Code: MIT · James Cook University"* (READ, project page footer). Some third parties mislabel the data "MIT". |
| Labels | Classification: 39,766 frames (17,409 with fish). Localization: 3,200 frames with points. Segmentation: 620 frames with masks. Official `train/val/test.csv` per subset. The habitat is the first path component of the ID, which lets **whole habitats be held out**. |

## 4. Kakadu freshwater fish: freshwater boxes

| | |
|---|---|
| Relevance | Freshwater billabongs in northern Australia. The best public **freshwater** proxy for lakes; its turbidity mix will be measured by the pipeline. |
| Access | Zenodo record 7250921 (URL from the KakaduFishAI GitHub README; READ). |
| License | *"The training dataset and model are licensed with… CC BY 4.0"* (READ). |
| Labels | 44,412 images, 82,904 COCO boxes, 23 species. |

## 5. MIT Sea Grant River Herring (LILA): fish/no-fish and night frames

| | |
|---|---|
| Relevance | River fish-passage cameras. About 262k frames, about 162k reviewed as empty, **about 35% at night**. Excellent for search-mode **fish/no-fish** and the **dark slice**, and it teaches what "empty" looks like. |
| Access | https://lila.science/datasets/mit-sea-grant-river-herring/ (SEARCH; LILA hosts files on public cloud buckets). The exact file URLs come from that page. |
| License | **CDLA-Permissive-1.0** (SEARCH plus secondary source). |

## 6. Brackish dataset (Aalborg University): turbid water, LED-lit, fixed camera

| | |
|---|---|
| Relevance | Camera about 9 m deep on a bridge pillar in the Limfjord, Denmark, in turbid brackish water with its own LED lights. The closest public match to **murky lake plus own lights**. 89 videos. |
| Access | Kaggle `aalborguniversity/brackish-dataset` (account needed; doi:10.34740/kaggle/ds/2695511). Project page vap.aau.dk/the-brackish-dataset. Roboflow mirror. (SEARCH) |
| License | **Conflicting. Please confirm on Kaggle.** Kaggle original: CC BY-SA 4.0. Roboflow mirror: CC BY 4.0. BrackishMOT: CC BY-NC-SA 4.0 (all secondary). We assume **CC BY-SA 4.0**: used locally, never redistributed. |
| Labels | About 14.7k frames with about 35.6k boxes (fish, small_fish, crab, shrimp, jellyfish, starfish); splits 11,739 / 1,467 / 1,468. The 2,230 unlabelled frames are **not** guaranteed to be empty. It is a single viewpoint, so it is split by video and capped. |

## 7. OzFish (Australian Institute of Marine Science): pretraining frames only

| | |
|---|---|
| Access | `https://storage.pawsey.org.au/public/m/FDFML/` (subfolders videos/, frames/, crops/, metadata/, labelled/…), doi:10.25845/5e28f062c5097 (READ, GitHub README). Not on the AWS Open Data Registry. |
| License | Attribution only. The README text says CC BY 3.0 AU while its badge says CC BY 4.0. It is described as *"completely open and free to use for advancing machine learning"*. |
| Labels | Boxes estimated from length measurements (about 45k), crops, and about 1.8k fish/no-fish frames. **Not exhaustive**; another project excluded it for annotation quality. **Pretraining only.** |
| Unverified | Total size and whether the links are still live (a GitHub issue reports download problems). |

## 8. Ocean Networks Canada (Oceans 3.0 / SeaTube)

| | |
|---|---|
| Relevance | Deep fixed cameras with scheduled lights (e.g. Barkley Canyon at 393 m and 985 m). Dark, backscatter, and lights-on/off frames, matching the twilight-zone condition. |
| Access | `pip install onc` (Apache-2.0 client; READ from PyPI) plus a free token from https://data.oceannetworks.ca/Registration. Getting one frame may mean downloading a whole video file. |
| License | ONC-owned data is **CC BY 4.0**. *"Data owned or co-owned by a data partner may be subject to alternative licensing such as… CC-BY-NC… or more restrictive"* (SEARCH, https://www.oceannetworks.ca/data/data-policy/). **The licence is checked per device.** |
| Labels | Timestamped observations, not boxes. Pretraining only. |

## 9. Fish4Knowledge: low priority

- **Recognition ground truth:** 27,370 small fish crops from shallow Taiwan reef cameras. Permissive notice ("copy, use, modify, or distribute… provided this copyright notice is retained"), but the data were "acquired for research purposes only" (SEARCH). The crops are too small to help patch-level pretraining.
- **Detection/tracking set:** 17 ten-minute videos at 320×240 and 5 fps, including *"high water turbidity, very low contrast"* (READ). Manual SharePoint download. **No licence stated**, so ask the authors first.
- **Decision:** skip for v1.

## 10. Deferred candidates

| Dataset | Why deferred | Action |
|---|---|---|
| **SEAMAPD21** (NOAA SEFSC) | Gulf of Mexico baited-camera reef-fish video: 90k annotations, 130 species, 26 GB in 263 tar.gz parts from `https://grunt.sefsc.noaa.gov/parr/SEAMAPD21.tar.gz.aa`… (READ, GitHub README). **No licence stated.** | Very relevant to the Gulf of Mexico. Ask SEFSC for the licence. |
| Labeled Fishes in the Wild (NOAA SWFSC) | 3,167 ROV stills of rockfish; only a credit request, no licence | Likely public domain; confirm before use |
| NOAA Puget Sound Nearshore Fish (LILA) | 77,739 images, CDLA-Permissive (SEARCH) | Can add later |
| FishTrack23 | CC BY 4.0 per one source; exact collection link unverified | Verify link first |
| UIEB / LSUI | academic, non-commercial, no redistribution | Not used |
| TrashCan 1.0 | academic only; commercial needs JAMSTEC permission | Not used |
| SUIM / RUOD | dataset licence unclear | Contact the authors |
| Schmidt Ocean Institute | CC BY-NC-SA 4.0, footage by request | Not used for now |
| Roboflow Aquarium | CC BY 4.0 but public-aquarium imagery | Low relevance |

---

## Planned v1 mix (to be tuned from the dataset report)

- **Pretraining pool:**
  - FathomNet (deduplicated, capped at ~150k);
  - NOAA EX dives (~20–40 dives, favouring the Gulf of Mexico, 1 fps, deduplicated);
  - Kakadu, DeepFish, River Herring and Brackish **train** splits (capped per source);
  - OzFish frames;
  - ONC (optional).
  - Target: 300–500k images after dedup and empty-water downsampling.
- **Evaluation (frozen, grouped, near-duplicates removed from pretraining):**
  - DeepFish test (held-out habitats);
  - Kakadu test;
  - River Herring test (day/night);
  - Brackish test videos;
  - FathomNet test (held-out dives and uploads);
  - then your own dives.
