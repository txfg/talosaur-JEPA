"""License registry. Every image in the index carries a normalised license id plus a
``train_commercial_ok`` flag, so a commercially usable model can be rebuilt with one filter.

These flags encode *our reading* of the terms (see docs/DATASETS.md) and are not legal advice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LicenseInfo:
    id: str
    name: str
    url: str | None
    commercial_ok: bool  # commercial use of the material itself
    derivatives_ok: bool  # adapted material may be shared
    share_alike: bool
    attribution: bool
    notes: str = ""


LICENSES: dict[str, LicenseInfo] = {
    lic.id: lic
    for lic in [
        LicenseInfo("CC0-1.0", "CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", True, True, False, False),
        LicenseInfo("CC-BY-4.0", "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", True, True, False, True),
        LicenseInfo("CC-BY-3.0", "CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/", True, True, False, True),
        LicenseInfo("CC-BY-3.0-AU", "CC BY 3.0 AU", "https://creativecommons.org/licenses/by/3.0/au/", True, True, False, True),
        LicenseInfo("CC-BY-SA-4.0", "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", True, True, True, True),
        LicenseInfo("CC-BY-ND-4.0", "CC BY-ND 4.0", "https://creativecommons.org/licenses/by-nd/4.0/", True, False, False, True),
        LicenseInfo("CC-BY-NC-4.0", "CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/", False, True, False, True),
        LicenseInfo("CC-BY-NC-SA-4.0", "CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/", False, True, True, True),
        LicenseInfo("CC-BY-NC-ND-4.0", "CC BY-NC-ND 4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/", False, False, False, True),
        LicenseInfo("CDLA-Permissive-1.0", "CDLA Permissive 1.0", "https://cdla.dev/permissive-1-0/", True, True, False, True),
        LicenseInfo("public-domain-us-gov", "US Government work (public domain)", None, True, True, False, False,
                    "Attribution requested by the agency (e.g. 'NOAA Ocean Exploration')."),
        LicenseInfo("own", "Own footage", None, True, True, False, False),
        LicenseInfo("synthetic", "Procedurally generated", None, True, True, False, False),
        LicenseInfo("unknown", "Unknown / unverified", None, False, False, False, True,
                    "Treated as non-commercial and non-redistributable until verified."),
    ]
}  # fmt: skip

_CC_VARIANTS = {
    "by": "CC-BY",
    "by-sa": "CC-BY-SA",
    "by-nd": "CC-BY-ND",
    "by-nc": "CC-BY-NC",
    "by-nc-sa": "CC-BY-NC-SA",
    "by-nc-nd": "CC-BY-NC-ND",
}


def normalize_license(raw: str | None) -> str:
    """Map free-form license strings/URLs to an id in :data:`LICENSES` (``unknown`` if unsure).

    Handles e.g. ``https://creativecommons.org/licenses/by-nc-nd/4.0/``, ``CC BY-NC-ND 4.0``,
    ``CC-BY``, ``CC0``, ``Attribution-NonCommercial 4.0 International``, ``CDLA-Permissive-1.0``.
    """
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    if s in LICENSES:
        return s
    low = s.lower()
    if "publicdomain/zero" in low or re.fullmatch(r"cc[\s_-]*0(\s*1\.0)?", low) or "cc0" in low:
        return "CC0-1.0"
    if "public domain" in low or "public-domain" in low:
        return "public-domain-us-gov"
    if "cdla" in low and "permissive" in low:
        return "CDLA-Permissive-1.0"

    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)(/([a-z]{2}))?", low)
    if m:
        variant, version, juris = m.group(1), m.group(2), m.group(4)
        base = _CC_VARIANTS.get(variant)
        if base:
            cand = f"{base}-{version}" + (f"-{juris.upper()}" if juris else "")
            if cand in LICENSES:
                return cand
            cand = f"{base}-{version}"
            return cand if cand in LICENSES else f"{base}-4.0" if f"{base}-4.0" in LICENSES else "unknown"

    words = low.replace("_", " ").replace("-", " ")
    if "attribution" in words or re.search(r"\bcc\b", words):
        nc = "noncommercial" in words.replace(" ", "") or re.search(r"\bnc\b", words) is not None
        nd = "noderivatives" in words.replace(" ", "") or "noderivs" in words.replace(" ", "") or re.search(r"\bnd\b", words) is not None
        sa = "sharealike" in words.replace(" ", "") or re.search(r"\bsa\b", words) is not None
        version = re.search(r"(\d\.\d)", words)
        ver = version.group(1) if version else "4.0"
        base = "CC-BY" + ("-NC" if nc else "") + ("-SA" if sa else "") + ("-ND" if nd else "")
        au = " au" in words or "australia" in words
        for cand in (f"{base}-{ver}-AU" if au else None, f"{base}-{ver}", f"{base}-4.0"):
            if cand and cand in LICENSES:
                return cand
    return "unknown"


def license_info(license_id: str) -> LicenseInfo:
    return LICENSES.get(license_id, LICENSES["unknown"])


def train_commercial_ok(license_id: str, source_ml_training_clause: bool = False) -> bool:
    """May a model trained on this image be used commercially (as far as we know)?

    ``source_ml_training_clause`` is set for sources whose terms explicitly allow ML training
    for commercial purposes irrespective of the per-image license (FathomNet's Terms of Use).
    """
    return bool(source_ml_training_clause or license_info(license_id).commercial_ok)
