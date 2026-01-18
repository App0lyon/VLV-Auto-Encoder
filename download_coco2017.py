import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
from tqdm import tqdm


COCO_2017_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    # Reprendre un téléchargement interrompu (si le serveur le supporte)
    resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_size = int(total) + resume_pos if total is not None else None

        mode = "ab" if resume_pos > 0 else "wb"
        with open(tmp_path, mode) as f, tqdm(
            total=total_size,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {out_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    tmp_path.rename(out_path)


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def filename_from_url(url: str) -> str:
    return os.path.basename(urlsplit(url).path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/coco2017", help="Destination root folder")
    parser.add_argument("--train", action="store_true", help="Download train2017 images")
    parser.add_argument("--val", action="store_true", help="Download val2017 images")
    parser.add_argument("--ann", action="store_true", help="Download annotations")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--keep_zips", action="store_true", help="Keep zip files after extraction")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    zips_dir = out_root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    if not (args.train or args.val or args.ann or args.all):
        print("Nothing selected. Use --all or any of --train/--val/--ann", file=sys.stderr)
        sys.exit(1)

    to_get = []
    if args.all or args.train:
        to_get.append(("train_images", COCO_2017_URLS["train_images"]))
    if args.all or args.val:
        to_get.append(("val_images", COCO_2017_URLS["val_images"]))
    if args.all or args.ann:
        to_get.append(("annotations", COCO_2017_URLS["annotations"]))

    for key, url in to_get:
        zip_name = filename_from_url(url)
        zip_path = zips_dir / zip_name

        if zip_path.exists():
            print(f"Already downloaded: {zip_path}")
        else:
            print(f"Downloading {key} from {url}")
            download_file(url, zip_path)

        # Extract
        unzip(zip_path, out_root)

        if not args.keep_zips:
            try:
                zip_path.unlink()
            except OSError:
                pass

    print("\nDone. COCO 2017 should be in:", out_root)
    print("Expected folders:")
    print(" - train2017/")
    print(" - val2017/")
    print(" - annotations/ (instances_train2017.json, instances_val2017.json, etc.)")


if __name__ == "__main__":
    main()
