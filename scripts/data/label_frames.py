#!/usr/bin/env python
"""Hand-label light / clarity on a few hundred frames, then calibrate the bucket thresholds.

1) Make a self-contained labelling page (open it in any browser, no server needed):

     python scripts/data/label_frames.py page --index data/index/underwater_v1.parquet --root data \
         --n 300 --out labels.html

   Keys: 1/2/3 = dark/dim/bright, q/w/e = murky/moderate/clear, arrows = previous/next.
   Progress is kept in the browser; click "Download labels" when done (labels.json).

2) Fit thresholds that best agree with your labels and write them for the curation config:

     python scripts/data/label_frames.py calibrate --labels labels.json \
         --index data/index/underwater_v1.parquet --out configs/curate/thresholds.yaml

   Then set `thresholds_file: configs/curate/thresholds.yaml` in configs/curate/v1.yaml and rebuild.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

from talosaur.data.index import read_table
from talosaur.data.quality import fit_thresholds_from_labels
from talosaur.utils.io import save_yaml

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Talosaur condition labels</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
#img{max-width:100%;max-height:70vh;display:block;margin:8px auto;border:1px solid #444}
.row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin:6px}
button{font-size:16px;padding:8px 14px;border-radius:6px;border:1px solid #666;background:#222;color:#eee;cursor:pointer}
button.on{background:#2b6cb0;border-color:#90cdf4}
#meta{text-align:center;color:#aaa;font-size:13px}
</style></head><body>
<div id="meta"></div><img id="img">
<div class="row" id="light"></div><div class="row" id="clarity"></div>
<div class="row"><button onclick="go(-1)">&larr; prev</button><button onclick="go(1)">next &rarr;</button>
<button onclick="dl()">Download labels</button></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "talosaur-labels-__TAG__";
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const L = ["dark","dim","bright"], C = ["murky","moderate","clear"];
function btns(id, names, field, keys){
  const el = document.getElementById(id); el.innerHTML = "";
  names.forEach((n,k)=>{const b=document.createElement("button"); b.textContent=`${keys[k]}: ${n}`;
    const it = ITEMS[i]; if((labels[it.id]||{})[field]===n) b.className="on";
    b.onclick=()=>set(field,n); el.appendChild(b);});
}
function show(){
  const it = ITEMS[i]; document.getElementById("img").src = it.src;
  const done = Object.keys(labels).length;
  document.getElementById("meta").textContent = `${i+1}/${ITEMS.length}  (${done} labelled)  ${it.id}`;
  btns("light", L, "light", ["1","2","3"]); btns("clarity", C, "clarity", ["q","w","e"]);
}
function set(field, v){ const id = ITEMS[i].id; labels[id] = Object.assign(labels[id]||{}, {[field]: v});
  localStorage.setItem(KEY, JSON.stringify(labels)); show();
  if(labels[id].light && labels[id].clarity) setTimeout(()=>go(1), 120); }
function go(d){ i = Math.min(ITEMS.length-1, Math.max(0, i+d)); show(); }
function dl(){ const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(labels,null,1)],{type:"application/json"}));
  a.download="labels.json"; a.click(); }
document.addEventListener("keydown", e=>{ const k=e.key;
  if(k==="1"||k==="2"||k==="3") set("light", L[+k-1]);
  else if(k==="q") set("clarity","murky"); else if(k==="w") set("clarity","moderate");
  else if(k==="e") set("clarity","clear"); else if(k==="ArrowRight") go(1); else if(k==="ArrowLeft") go(-1); });
show();
</script></body></html>"""


def _thumb_b64(path: Path, max_side: int = 320) -> str:
    from PIL import Image

    from talosaur.data.imageio import load_rgb

    im = Image.fromarray(load_rgb(path, max_side=max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_page(a) -> None:
    df = read_table(a.index)
    rng = np.random.default_rng(a.seed)
    # stratify over the current (uncalibrated) buckets and sources so labels span the range
    groups = list(df.groupby(["source", "light", "clarity_bucket"]).groups.values())
    per = max(1, a.n // max(1, len(groups)))
    picks: list[int] = []
    for g in groups:
        g = np.asarray(g)
        picks += rng.choice(g, min(per, len(g)), replace=False).tolist()
    rest = np.setdiff1d(np.arange(len(df)), picks)
    if len(picks) < a.n and len(rest):
        picks += rng.choice(rest, min(a.n - len(picks), len(rest)), replace=False).tolist()
    picks = rng.permutation(picks)[: a.n]
    items = [{"id": df.iloc[i]["image_id"], "src": _thumb_b64(Path(a.root) / df.iloc[i]["path"])} for i in picks]
    html = HTML.replace("__ITEMS__", json.dumps(items)).replace("__TAG__", Path(a.index).stem)
    Path(a.out).write_text(html)
    print(f"wrote {a.out} with {len(items)} images; open it in a browser")


def calibrate(a) -> None:
    df = read_table(a.index).set_index("image_id")
    labels = json.loads(Path(a.labels).read_text())
    ids = [i for i in labels if i in df.index]
    if not ids:
        raise SystemExit("no labelled image ids found in the index")
    sub = df.loc[ids]
    t, agree = fit_thresholds_from_labels(
        {"lum_mean": sub["lum_mean"].to_numpy(), "clarity": sub["clarity"].to_numpy()},
        [labels[i].get("light") for i in ids],
        [labels[i].get("clarity") for i in ids],
    )
    save_yaml({k: round(v, 5) for k, v in t.to_dict().items()}, a.out)
    print(f"{len(ids)} labelled images; agreement: {agree}")
    print(f"thresholds -> {a.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("page")
    p.add_argument("--index", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default="labels.html")
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate")
    c.add_argument("--labels", required=True)
    c.add_argument("--index", required=True)
    c.add_argument("--out", default="configs/curate/thresholds.yaml")
    a = ap.parse_args(argv)
    make_page(a) if a.cmd == "page" else calibrate(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
