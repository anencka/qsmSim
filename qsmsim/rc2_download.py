"""Download the QSM Reconstruction Challenge 2.0 (RC2) data, evaluation code and submissions.

    qsmsim-fetch-rc2 --p3                  # only the six Sim1 files phantom P3 needs (~0.1 GB), checksum-verified
    qsmsim-fetch-rc2                       # inputs + ground truth + evaluation code (~0.55 GB)
    qsmsim-fetch-rc2 --submissions         # + all Stage 1/2 submissions and leaderboard CSVs (~3.8 GB)

Destination: --out, else QSMSIM_RC2_DIR, else data/rc2 in the source checkout.
The data are CC-BY 4.0 (Bilgic et al., MRM 2021; Zenodo 10.5281/zenodo.4559541): cite the challenge when you use them.

Sources (Bilgic et al., MRM 2021, data availability statement):
  SURFdrive public share uTvrxJ5NELbnGa5 (data, ground truth, evaluation code) via WebDAV
  Zenodo 10.5281/zenodo.3687198 (metrics CSV), 3687342 (Stage 1), 3688703 (Stage 2 submissions)
Files that already exist with the right size are skipped, so the script can be re-run.
"""
import argparse
import base64
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

SHARE = "uTvrxJ5NELbnGa5"
DAV = "https://surfdrive.surf.nl/public.php/webdav/"
ZENODO = {"metrics": "3687198", "stage1": "3687342", "stage2": "3688703"}
DAV_DIRS = ["Sim1/", "Sim2/", "Sim1GT/", "Sim2GT/", "functions4challenge/"]


def _request(url, auth=None, method="GET", headers=None):
    h = dict(headers or {})
    if auth:
        h["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    return urllib.request.Request(url, method=method, headers=h)


def dav_list(path):
    req = _request(DAV + path, auth=f"{SHARE}:", method="PROPFIND", headers={"Depth": "1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = r.read().decode()
    items = []
    for resp in re.findall(r"<d:response>(.*?)</d:response>", out, re.S):
        href = re.search(r"<d:href>([^<]+)</d:href>", resp).group(1).replace("/public.php/webdav/", "")
        size = re.search(r"<d:getcontentlength>(\d+)", resp)
        if href != path:
            items.append((href, int(size.group(1)) if size else None))
    return items


def fetch(url, dest, size=None, auth=None, tries=3):
    """Download url to dest (via a .part file, so an interrupted download is never mistaken for a finished one)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return False
    part = dest.with_name(dest.name + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(_request(url, auth), timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dest)
            return True
        except OSError:
            part.unlink(missing_ok=True)
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def dav_mirror(path, root):
    for href, size in dav_list(path):
        if href.endswith("/"):
            dav_mirror(href, root)
        elif fetch(DAV + href, root / href, size, auth=f"{SHARE}:"):
            print(f"  {href} ({(size or 0) / 1e6:.1f} MB)", flush=True)


def zenodo(record, dest, workers=6):
    """Zenodo throttles single streams (~0.5 MB/s), so files are fetched in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with urllib.request.urlopen(urllib.request.Request(f"https://zenodo.org/api/records/{record}",
                                                       headers={"Accept": "application/json"})) as r:
        files = json.load(r)["files"]

    def one(f):
        if fetch(f["links"]["self"], dest / f["key"], f["size"]):
            print(f"  {dest.name}/{f['key']} ({f['size'] / 1e6:.1f} MB)", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))


def fetch_p3(out):
    """The Sim1 files phantom P3 needs (rc2_3t.FILES), checked against the recorded SHA-256 sums."""
    from .exp1.inputs import check_rc2
    from .rc2_3t import FILES
    for f in FILES:
        if fetch(DAV + f, out / f, auth=f"{SHARE}:"):
            print(f"  {f}", flush=True)
    if not check_rc2(out, verbose=True):
        raise SystemExit("checksum mismatch: delete the files above and fetch again")


def main(argv=None):
    from .rc2 import ROOT
    p = argparse.ArgumentParser(prog="qsmsim-fetch-rc2", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT)
    p.add_argument("--p3", action="store_true", help="only the files phantom P3 needs")
    p.add_argument("--submissions", action="store_true")
    a = p.parse_args(argv)
    if a.p3:
        print(f"SURFdrive: the Sim1 files for P3 -> {a.out}")
        fetch_p3(a.out)
        print("done")
        return
    print("SURFdrive: inputs, ground truth, evaluation code")
    for d in DAV_DIRS:
        dav_mirror(d, a.out)
    fetch(DAV + "EvaluationScript_4ChallengeOrganizers_clean.m", a.out / "EvaluationScript_4ChallengeOrganizers_clean.m",
          auth=f"{SHARE}:")
    print("Zenodo: leaderboard metrics")
    zenodo(ZENODO["metrics"], a.out / "metrics")
    if a.submissions:
        for k in ("stage1", "stage2"):
            print(f"Zenodo: {k} submissions")
            zenodo(ZENODO[k], a.out / "submissions" / k)
    print("done")


if __name__ == "__main__":
    main()
