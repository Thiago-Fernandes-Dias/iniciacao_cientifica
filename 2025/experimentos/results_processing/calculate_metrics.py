import sys
from pathlib import Path

import numpy as np
import pandas as pd


def map_range(value, src_min, src_max, dst_min, dst_max):
    return (value - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


def main():
    csv = Path(sys.argv[1]).resolve()
    if not csv.is_file() or not csv.name.endswith(".csv"):
        sys.exit(1)

    comps = pd.read_csv(csv)

    score_min, score_max = comps["score"].min(), comps["score"].max()
    comps["score"] = comps["score"].apply(lambda x: map_range(x, score_min, score_max, 0.0, 1.0))

    thresholds = comps["score"].unique()
    min_t, max_t = thresholds.max(), thresholds.min()
    sample_thresholds = np.linspace(min_t, max_t, 1000)

    results_series: list[pd.Series] = [0] * (sample_thresholds.size)

    for i, t in enumerate(sample_thresholds):
        users = comps["user_1"].unique()
        frr_per_user: list[float] = [0] * (users.size)
        far_per_user: list[float] = [0] * (users.size)
        for j, user in enumerate(users):
            genuine_attemps = comps[(comps["user_1"] == user) & (comps["user_2"] == user)]
            impostor_attemps = comps[(comps["user_1"] == user) & (comps["user_2"] != user)]
            true_positives = genuine_attemps[genuine_attemps["score"] >= t].count()
            false_rejections = genuine_attemps[genuine_attemps["score"] < t].count()
            true_negatives = impostor_attemps[impostor_attemps["score"] < t].count()
            false_acceptances = impostor_attemps[impostor_attemps["score"] >= t].count()
            frr_per_user[j] = false_rejections / (false_rejections + true_positives)
            far_per_user[j] = false_acceptances / (false_acceptances + true_negatives)
        result_serie = pd.Series()
        result_serie["frr"] = np.mean(frr_per_user)
        result_serie["far"] = np.mean(far_per_user)
        result_serie["threshold"] = t
        results_series[i] = result_serie

    result = pd.DataFrame(results_series)
    result.to_csv(Path(sys.argv[2]))


if __name__ == "__main__":
    main()
