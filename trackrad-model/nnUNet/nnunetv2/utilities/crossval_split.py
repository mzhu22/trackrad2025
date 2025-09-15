from typing import List

import numpy as np
from sklearn.model_selection import KFold


def generate_crossval_split(
    train_identifiers: List[str], seed=12345, n_splits=5
) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]["train"] = list(train_keys)
        splits[-1]["val"] = list(test_keys)
    return splits


def generate_video_split(
    train_identifiers: List[str], seed=12345, n_splits=5
) -> List[dict[str, List[str]]]:
    # Example ID: A_003-0000
    video_identifiers = sorted(set(t.split("-")[0] for t in train_identifiers))
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for i, (train_idx, test_idx) in enumerate(kfold.split(video_identifiers)):
        train_keys = np.array(video_identifiers)[train_idx]
        test_keys = np.array(video_identifiers)[test_idx]
        splits.append(
            {
                "train": [
                    t for t in train_identifiers if t.split("-")[0] in train_keys
                ],
                "val": [t for t in train_identifiers if t.split("-")[0] in test_keys],
            }
        )
    return splits
