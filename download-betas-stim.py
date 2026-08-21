################################################################
# Code adapted from nsdget
# https://github.com/xnought/nsdget
#
# Fetches all stimuli shown in the NSD experiment.
# Assumes executing in a directory with nsd_stim_info_merged.csv
# already downloaded from the NSD AWS bucket.
#################################################################

import ast

import os
import pandas as pd
from PIL import Image
from cloudpathlib import S3Client
from urllib.request import urlretrieve


def coco_image_links(coco_ids: list[int], splits: list[str]):
    assert len(coco_ids) == len(splits)
    for id, split in zip(coco_ids, splits):
        filename = f"{str(id).zfill(12)}.jpg"
        yield f"http://images.cocodataset.org/{split}/{filename}", filename


def percent_crop_image(im: Image.Image, percent_crop: list[float]) -> Image.Image:
    # percent crop is (top, bottom, left, right)
    [percent_top, percent_bottom, percent_left, percent_right] = percent_crop

    # but PIL.Image().crop takes in (left, top, right, bottom)
    width, height = im.size
    left = int(width * percent_left)
    top = int(height * percent_top)
    right = int(width * (1 - percent_right))
    bottom = int(height * (1 - percent_bottom))

    return im.crop([left, top, right, bottom])


def crop_stimuli_image(im: Image.Image, crop: list[float]):
    # resize based on https://cvnlab.slite.page/p/NKalgWd__F/Experiments
    # sometimes after crop the image is (426, 426) or (427, 427), so further resize to (425, 425)
    im = percent_crop_image(im, crop).resize((425, 425), Image.Resampling.LANCZOS)
    return im


def wget_if_not_already_downloaded(url: str, out: str, crop: list[float], skip_if_exists: bool):
    if not skip_if_exists or not os.path.exists(out):
        urlretrieve(url, out)
        crop_stimuli_image(Image.open(out), crop).save(out)  # override with cropped version


def stimuli_links_paths(
    coco_ids: list[int],
    splits: list[str],
) -> list[str]:

    # sub directories (ie val2017, train2017) to save to
    for split in splits:
        os.makedirs(os.path.join(split), exist_ok=True)

    # links to download
    links = []
    paths = []
    for (link, filename), split in zip(coco_image_links(coco_ids, splits), splits):
        links.append(link)
        paths.append(os.path.join(split, filename))

    return links, paths



if __name__ == "__main__":
    # download beta maps from S3
    subj_ids = [
        "subj01",
        "subj02",
        "subj03",
        "subj04",
        "subj05",
        "subj06",
        "subj07",
        "subj08"
    ]

    c = S3Client(no_sign_request=True)
    root_dir = c.CloudPath("s3://natural-scenes-dataset")

    for subj_id in subj_ids:
        s3_paths = root_dir.glob(
            f"nsddata_betas/ppdata/{subj_id}/MNI/betas_fithrf/**"
        )
        for path in s3_paths:
            path.download_to(f"./{subj_id}")

    # download images corresponding to each beta image from COCO.
    # read in NSD provided stimulus info ; assumed in CWD.
    df = pd.read_csv("nsd_stim_info_merged.csv")
    links, paths = stimuli_links_paths(df["cocoId"], df["cocoSplit"])
    crops=df["cropBox"].apply(ast.literal_eval)

    for url, out, crop in zip(links, paths, crops):
        wget_if_not_already_downloaded(url, out, crop, skip_if_exists=True)
