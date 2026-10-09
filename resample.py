from pathlib import Path

import click
import numpy as np
import nibabel as nib
from nilearn import image

def _resample_resp_to_3mm(stim, subj_dir):
    """Resample the brain responses to 3mm isotropic resolution."""
    print(f"Processing {stim}...")
    img = nib.load(subj_dir / stim)
    img.set_data_dtype(np.float32)
    res_img = image.resample_img(
        img,
        target_affine=np.diag((3, 3, 3)),
        interpolation="linear",
    )
    print(f"Resampled image shape: {res_img.shape}")
    res_img.to_filename(
        Path(subj_dir) / f"res-3mm_{stim}"
    )
    return


@click.command()
@click.option("--subj_name", default="subj01", help="Subject name.")
@click.option(
    "--data_dir",
    default="/scratch/emdupre/NSD",
    help="Data directory.",
)
def main(subj_name, data_dir):
    subj_dir = Path(data_dir, "stimuli.betas", subj_name)
    files = subj_dir.glob("stimulus-*.nii.gz")
    for f in files:
        _resample_resp_to_3mm(f.name, subj_dir)
    return


if __name__ == "__main__":
    main()