from copy import deepcopy
from pathlib import Path

import click
from joblib import Parallel, delayed
import nibabel as nib
from nilearn import masking
import numpy as np
import pandas as pd


def create_group_mask(data_dir):
    """
    Generate mask for group-level analysis
    indexing voxels for which all subjects have
    signal.

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    group_mask : Niimg-like
    """
    masks = Path(data_dir).rglob("valid*.nii.gz")
    group_mask = masking.intersect_masks(
        list(masks),
        threshold=1,
    )
    nib.save(group_mask, "NSD_MNI-mask.nii.gz")

    return group_mask


def extract_average_betas(subj_name, out_path, subj_df, data_dir, stim):
    """
    Generate and save to disk a numpy array
    with the average beta values for a given stimulus
    image.

    Parameters
    ----------
    stim : str
    """
    print(f"Starting {stim}...")
    stim_df = subj_df.loc[subj_df["cocoId"] == stim]
    stim_betas = []

    # grab session indices where this stimulus appeared
    session_idc = stim_df["session_id"].unique()

    for ss_idc in session_idc:

        # grab session-specific betas
        beta_fname = list(
            Path(data_dir, subj_name).rglob(
                f"betas_session{ss_idc:02}.nii.gz"
            )
        )[0]

        # index betas on identified session_trial_id for
        # this stimulus and this session
        sess_df = subj_df.loc[
            (subj_df["session_id"] == ss_idc) &
            (subj_df["cocoId"] == stim)
        ]
        session_trials = sess_df["session_trial_id"].unique()
        for ss_trial in session_trials:  # these are 1- rather than 0-indexed...
            stim_betas.append(
                nib.load(beta_fname).slicer[..., ss_trial - 1]
            )

    avg_stim_arr = np.mean(
            np.stack([sbeta.get_fdata() for sbeta in stim_betas], axis=-1),
            axis=-1
        )
    np.save(Path(out_path, f"{stim}.npy"), avg_stim_arr)
    print(f"Finished with {stim}...")
    return


def extract_individual_betas(subj_df, ref_header, ref_affine, out_path, stim):
    """
    Generate and save to disk a numpy array
    with the individual beta values for a given
    stimulus image.

    Parameters
    ----------
    stim : str
    """
    print(f"Starting {stim}...")
    stim_df = subj_df.loc[subj_df["cocoId"] == stim]
    subj_idx = int(subj_df["subjectId"].unique()[0])

    # grab session indices where this stimulus appeared
    for row in stim_df.itertuples():
    
        # grab session-specific beta file
        beta_fname = list(
            Path(out_path.parent.parent, f"subj{subj_idx:02}").rglob(
                f"betas_session{row.session_id:02}.nii.gz"
            )
        )[0]

        # index betas on identified session_trial_id for
        # this stimulus and this session_trial_id
        # NOTE: these are 1- rather than 0-indexed.
        sbeta = nib.load(beta_fname).slicer[..., row.session_trial_id - 1]
        sbeta.to_filename(
            Path(
                out_path,
                f"stimulus-{stim}_session-{row.session_id:02}_trial-{row.session_trial_id:03}.nii.gz"
            )
        )
    print(f"Finished with {stim}...")
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
    if not out_path.is_dir():
        out_path.parent.mkdir(exist_ok=True, parents=True)

    ref_niimg = nib.load(Path(data_dir, subj_name, 'betas_session01.nii.gz'))
    ref_affine = deepcopy(ref_niimg.affine)
    ref_header = deepcopy(ref_niimg.header)

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
        delayed(extract_individual_betas)(
            subj_df, ref_header, ref_affine, out_path, stim
            ) for stim in stim_names
    )


if __name__ == "__main__":
    main()
