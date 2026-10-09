from copy import deepcopy
from pathlib import Path

import click
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from nibabel import Nifti1Image
import pandas as pd


def convert_to_nii(ref_affine, ref_header, out_path, stim_file):
    arr = np.load(stim_file)
    # Beta values should be divided by 300 for appropriate
    # scaling ; see https://cvnlab.slite.page/p/6CusMRYfk0#7dfe1d13
    arr = (arr / 300)

    # downcast to float32 to save disk space and memory
    img = Nifti1Image(
        arr, 
        ref_affine,
        ref_header,
        dtype=np.float32,
    )
    img.to_filename(
        Path(out_path, f'{stim_file.stem}.nii.gz')
    )
    print(f"Finished with {stim_file.name}...")
    return


@click.command()
@click.option("--subj_name", default="subj01", help="Subject name.")
@click.option(
    "--data_dir",
    default="/scratch/emdupre/NSD",
    help="Data directory.",
)
def main(subj_name, data_dir):
    """
    """
    out_path = Path(
        data_dir,
        "stimuli.betas",
        subj_name
    )

    ref_niimg = nib.load(Path(data_dir, subj_name, 'betas_session01.nii.gz'))
    ref_affine = deepcopy(ref_niimg.affine)
    ref_header = deepcopy(ref_niimg.header)

    stim_files = list(out_path.rglob(
                f'stimulus-*_session*.npy'))

    Parallel(n_jobs=25)(
        delayed(convert_to_nii)(
            ref_affine, ref_header, out_path, stim_file
            ) for stim_file in stim_files
    )


if __name__ == "__main__":
    main()
