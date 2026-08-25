from pathlib import Path

import click
import nibabel as nib
from nilearn import image
import numpy as np
import pandas as pd


@click.command()
@click.option("--subj_name", default="subj01", help="Subject name.")
@click.option(
    "--data_dir",
    default="/scratch/emdupre/NSD",
    help="Data directory.",
)
def main(subj_name, data_dir):
    """
    Create paired nifti, clip-embedding arrays
    of Shared1000 stimulus set.

    Parameters
    ----------
    subj_name : str
    data_dir : str
    """
    out_path = Path(
        data_dir,
        "encoding.inputs",
        subj_name
    )

    stim_df = pd.read_csv(
        Path(data_dir, "nsd_stim_info_long_format.csv"),
        index_col=0
    )

    clip_df = pd.read_csv(
        Path(data_dir, "stimuli.clip-features", "file_names.txt"),
        sep='/',
        header=None,
        names=["dir", "split", "filename"]
    )
    clip_arr = np.load(
        Path(data_dir, "stimuli.clip-features", "features.npy"),
    )

    # load and parse subject-specific information
    subj_idx = int(subj_name[-1])
    subj_df = stim_df.loc[stim_df["subjectId"] == subj_idx]

    # restrict to shared stimuli and grab stim identifiers
    shared_stim = subj_df[subj_df["shared1000"]]
    stim_names = shared_stim["cocoId"].unique()

    niimgs = []
    clip_feats = []

    for stim_name in stim_names:

        # grab stimulus beta image
        niimgs.append(
            nib.load(Path(
                data_dir,
                'stimuli.betas',
                subj_name,
                f'{stim_name}.nii.gz'
                )
            )
        )
        
        # grab stimulus clip embeddings
        stim_idx = clip_df.index[
            clip_df["filename"] == f"{str(stim_name).zfill(12)}.jpg"
        ].to_list()
        clip_feats.append(clip_arr[stim_idx])

    # collapse all betas into one nii image
    out_niimg = image.concat_imgs(niimgs)
    out_niimg.to_filename(
        Path(out_path, f"{subj_name}_shared1000_betas.nii.gz")
    )

    # stack all embeddings into one npy file
    np.save(
        Path(out_path, f"{subj_name}_shared1000_clip-features.npy"),
        np.asarray(clip_feats)
    )

    # save out all stimuli identifiers
    np.savetxt(
        Path(out_path, f"{subj_name}_shared1000_stimuli.txt"),
        stim_names,
        fmt="str"
    )


if __name__ == "__main__":
    main()
