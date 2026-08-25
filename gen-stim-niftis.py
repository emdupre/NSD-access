from pathlib import Path

import click
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from nibabel import Nifti1Image
import pandas as pd


def convert_to_nii(subj_name, ref_niimg, stim_num):
    arr = np.load(Path('stimuli.betas', subj_name, f'{stim_num}.npy'))

    # Beta values should be divided by 300 for appropriate
    # scaling ; see https://cvnlab.slite.page/p/6CusMRYfk0#7dfe1d13
    arr = (arr / 300)

    # downcast to float32 to save disk space and memory
    img = Nifti1Image(
        arr, 
        ref_niimg.affine, 
        ref_niimg.header, 
        dtype=np.float32,
    )
    img.to_filename(Path('stimuli.betas', subj_name, f'{stim_num}.nii.gz'))
    print(f"Finished with {stim_num}...")
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

    ref_niimg = nib.load(Path(subj_name, 'betas_session01.nii.gz'))
    stim_df = pd.read_csv(
        Path(data_dir, "nsd_stim_info_long_format.csv"),
        index_col=0
    )
    # load and parse subject-specific information
    subj_idx = int(subj_name[-1])
    subj_df = stim_df.loc[stim_df["subjectId"] == subj_idx]

    # list all stimulus identifiers and then iterate
    stim_names = subj_df["cocoId"].unique()
    Parallel(n_jobs=100)(
        delayed(convert_to_nii)(
            subj_name, ref_niimg, stim_name
            ) for stim_name in stim_names
    )


if __name__ == "__main__":
    main()
