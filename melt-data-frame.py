import re

import pandas as pd

####################################
# Adapted from NSD-Flat
# https://github.com/clane9/NSD-Flat
####################################

NUM_TRIALS = 30000
MAX_SESSIONS = 40
TRIALS_PER_SESSION = NUM_TRIALS // MAX_SESSIONS
NUM_SESSIONS = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}


def unroll_stimuli_trials(stim_info: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stimulus data information from wide format to long format.

    Parameters
    ----------
    stim_info : pd.DataFrame
    """
    long_stim_info = []

    for _, row in stim_info.iterrows():

        for subject_idx in range(1, 9):  # 8 subjects, 1-indexing
            for rep_id in range(3):  # 3 repetitions per image
                trial_id = row[f"subject{subject_idx}_rep{rep_id}"]

                if trial_id > 0:  # if an image was shown
                    long_row = {"subjectId": subject_idx, "trialId": trial_id, **row}
                    long_stim_info.append(long_row)

    # create new long-format df
    long_stim_info = pd.DataFrame.from_records(
        long_stim_info,
        index=["subjectId", "trialId"],
    )
    long_stim_info = long_stim_info.sort_index()

    # drop now unnecessary columns
    for col in long_stim_info.columns:
        if re.match(r"subject*",col):
            long_stim_info.drop([col], axis=1, inplace=True)

    return long_stim_info


def mask_trials(df: pd.DataFrame):
    """
    Used to mask trials that were not actually run in the experiment,
    Following the maximum number of sessions as documented in:
    https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information

    Parameters
    ----------
    df : pd.MultiIndex
    """
    df["mask"] = False

    for subject_idx in range(1, 9):  # 8 subjects, 1-indexing
        for session_idx in range(NUM_SESSIONS[f"subj0{subject_idx}"]):

            for trial_idx in range(1, TRIALS_PER_SESSION + 1):
                global_trial_idx = session_idx * TRIALS_PER_SESSION + trial_idx

                df.loc[(subject_idx, global_trial_idx), "mask"] = True

    # apply the generated mask
    df = df[df["mask"]]
    df.drop(["mask"], axis=1, inplace=True)

    return df


def add_session_and_local_trial_info(df: pd.DataFrame):
    """
    Parameters
    ----------
    df : pd.MultiIndex
    """
    df["session_id"] = None
    df["session_trial_id"] = None

    for subject_idx in range(1, 9):  # 8 subjects, 1-indexing
        for session_idx in range(NUM_SESSIONS[f"subj0{subject_idx}"]):

            for trial_idx in range(1, TRIALS_PER_SESSION + 1):
                global_trial_idx = session_idx * TRIALS_PER_SESSION + trial_idx

                df.loc[(subject_idx, global_trial_idx), "session_id"] = session_idx + 1
                df.loc[(subject_idx, global_trial_idx), "session_trial_id"] = trial_idx

    cols = ['session_id', 'session_trial_id', 'cocoId', 'cocoSplit', 'cropBox',
            'loss', 'nsdId', 'flagged', 'BOLD5000', 'shared1000']
    df = df[cols]

    return df


if __name__ == "__main__":

    # read in NSD provided stimulus info ; assumed in CWD
    df = pd.read_csv("nsd_stim_info_merged.csv", index_col=0)

    # convert from wide to long format
    long_df = unroll_stimuli_trials(df)

    # mask all trials artificially added in unrolling
    long_df = mask_trials(long_df)

    # add information on session indices, trial indices
    # within sessions, for matching against betas
    long_df = add_session_and_local_trial_info(long_df)
    long_df.reset_index(inplace=True)

    # save to a new CSV
    long_df.to_csv("nsd_stim_info_long_format.csv")
    # read in with 
    # pd.read_csv("nsd_stim_info_long_format.csv", index_col=0)