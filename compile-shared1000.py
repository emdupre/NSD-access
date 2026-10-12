from pathlib import Path

import click
import nibabel as nib
from nilearn import image
import numpy as np
import pandas as pd


def _isolate_session(ses_file):
    """
    Isolate session identifier from a given file path.

    Parameters
    ----------
    ses_file : Path
        Path to the session file.

    Returns
    -------
    str
        Session identifier extracted from the file name.
    """
    return ses_file.stem.split("_")[2].split("-")[1]


@click.command()
@click.option(
    "--subj_name",
    default="subj01",
    type=click.Choice(["subj01", "subj02", "subj05", "subj07"]),
    help="Subject name. Must be one of: subj01, subj02, subj05, subj07."
)
@click.option(
    "--data_dir",
    default="/scratch/emdupre/NSD",
    help="Data directory.",
)
def main(subj_name, data_dir):
    """
    Create paired nifti, clip-embedding arrays of Shared1000 stimulus set.
    Note that only subj01, subj02, subj05, subj07 completed the 
    full Shared1000 stimulus set, so only these subjects are supported.
    All other subjects can be considered in the Shared515 stimulus subset.

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

    print(f"Working with {len(stim_names)} stimuli, from dataFrame of len {len(shared_stim)}")

    niimgs = []
    clip_feats = []
    out_names = []
    session_names = []

    for stim_name in stim_names:

        # grab stimulus beta image
        beta_files = list(
            Path(data_dir, 'stimuli.betas', subj_name).rglob(
                f'res-3mm_stimulus-{stim_name}_session-*.nii.gz'
                )
            )
        
        # check len
        if len(beta_files) != 3:
            print(
                len(beta_files),
                f"not exactly 3 betas found for stimulus-{stim_name}"
            )

        # grab stimulus clip embeddings
        stim_idx = clip_df.index[
            clip_df["filename"] == f"{str(stim_name).zfill(12)}.jpg"
        ].to_list()


        for b in beta_files:
            niimgs.append(nib.load(b))  # the betas,
            session_names.append(_isolate_session(b))  # session identifiers
            clip_feats.append(clip_arr[stim_idx])  # embeddings, 
            out_names.append(stim_name)  # and stimulus identifiers

    print(f"Found {len(niimgs)} betas, {len(clip_feats)} embeddings, "
          f"{len(out_names)} stimuli, and {len(session_names)} sessions.")

    # collapse all betas into one nii image
    out_niimg = image.concat_imgs(niimgs)
    out_niimg.to_filename(
        Path(out_path, f"{subj_name}_res-3mm_shared1000_trialBetas.nii.gz")
    )

    # stack all embeddings into one npy file
    np.save(
        Path(out_path, f"{subj_name}_shared1000_clip-trialFeatures.npy"),
        np.vstack(clip_feats)
    )

    # save out all stimuli identifiers
    np.savetxt(
        Path(out_path, f"{subj_name}_shared1000_trialStimuli.txt"),
        out_names,
        fmt="%s"
    )

    # save out all session identifiers
    np.savetxt(
        Path(out_path, f"{subj_name}_shared1000_trialSessions.txt"),
        session_names,
        fmt="%s"
    )


if __name__ == "__main__":
    main()
