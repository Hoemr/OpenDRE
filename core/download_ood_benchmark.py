#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download OpenOOD-style benchmark data (benchmark_imglist + images_classic) for imglist OOD.
Uses the same Google Drive IDs as OpenOOD. Run from repo root:

  pip install gdown
  python core/download_ood_benchmark.py --save_dir ../data

Then: --data_dir ../data --ood_use_imglist True --ood_in_dist cifar10
"""
import argparse
import os
import zipfile


def main():
    try:
        import gdown
    except ImportError:
        raise SystemExit("Install gdown: pip install gdown")

    # OpenOOD v1.5 download IDs (from OpenOOD/scripts/download/download.py)
    benchmark_imglist_id = "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1"
    ids = {
        "benchmark_imglist": benchmark_imglist_id,
        "cifar10": "1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1",
        "cifar100": "1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_",
        "tin": "1PZ-ixyx52U989IKsMA2OT-24fToTrelC",
        "mnist": "1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb",
        "svhn": "1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI",
        "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
        "places365": "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay",
    }

    ap = argparse.ArgumentParser(description="Download OpenOOD benchmark data for imglist OOD")
    ap.add_argument(
        "--save_dir",
        type=str,
        default="../data",
        help="Directory that will contain benchmark_imglist and images_classic",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["benchmark_imglist", "cifar10", "cifar100", "tin", "mnist", "svhn"],
        help="Which to download: benchmark_imglist (txt lists) and/or image sets",
    )
    args = ap.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # benchmark_imglist -> extract at save_dir (top level)
    if "benchmark_imglist" in args.datasets:
        out = os.path.join(save_dir, "benchmark_imglist.zip")
        if not os.path.isdir(os.path.join(save_dir, "benchmark_imglist")):
            print("Downloading benchmark_imglist ...")
            gdown.download(id=benchmark_imglist_id, output=out, quiet=False)
            with zipfile.ZipFile(out, "r") as z:
                z.extractall(save_dir)
            os.remove(out)
            print("Extracted benchmark_imglist/")
        else:
            print("benchmark_imglist/ already exists, skip")

    # Image datasets -> images_classic/<name>/
    classic = ["cifar10", "cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    for name in args.datasets:
        if name == "benchmark_imglist":
            continue
        if name not in ids:
            print("Unknown dataset: %s, skip" % name)
            continue
        store = os.path.join(save_dir, "images_classic", name)
        if os.path.isdir(store) and len(os.listdir(store)) > 0:
            print("%s already exists, skip" % store)
            continue
        os.makedirs(store, exist_ok=True)
        out = os.path.join(store, name + ".zip")
        print("Downloading %s ..." % name)
        gdown.download(id=ids[name], output=out, quiet=False)
        with zipfile.ZipFile(out, "r") as z:
            z.extractall(store)
        os.remove(out)
        print("Extracted %s" % store)

    print("Done. Use: --data_dir %s --ood_use_imglist True --ood_in_dist cifar10" % save_dir)


if __name__ == "__main__":
    main()

