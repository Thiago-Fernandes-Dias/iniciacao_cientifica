import sys
from typing import Any
from pathlib import Path
import pandas as pd
import numpy as np
from collections import deque
def main():
    results_series: deque[pd.Series] = deque[pd.Series]()

    csv = Path(sys.argv[1]).resolve()
    if not csv.is_file() or not csv.name.endswith(".csv"):
        sys.exit(1)

    comps = pd.read_csv(csv)
    thresholds = sorted(comps["score"].unique())

    for t in thresholds:
        print(f"Threshold {t}")
        users = comps["user_1"].unique()
        frr_per_user: list[float] = []
        far_per_user: list[float] = []
        for user in users:
            genuine_attemps = comps[(comps["user_1"] == user) & (comps["user_2"] == user)]
            impostor_attemps = comps[(comps["user_1"] == user) & (comps["user_2"] != user)]
            true_positives = genuine_attemps[genuine_attemps["score"] >= t].count()
            false_rejections = genuine_attemps[genuine_attemps["score"] < t].count()
            true_negatives = impostor_attemps[impostor_attemps["score"] < t].count()
            false_acceptances = impostor_attemps[impostor_attemps["score"] >= t].count()
            frr_per_user.append((false_rejections / (false_rejections + true_positives)) * 100)
            far_per_user.append((false_acceptances / (false_acceptances + true_negatives)) * 100)
        result_serie = pd.Series()
        result_serie["frr"] = np.mean(frr_per_user)
        result_serie["far"] = np.mean(far_per_user)
        result_serie["threshold"] = t
        results_series.append(result_serie)
    
    result = pd.DataFrame(results_series)
    result.to_csv(Path(sys.argv[2]))

    row = result.iloc[(result["frr"] - result["far"]).abs().idxmin()]
    print(f"threshold: {row['threshold']} | frr: {row['frr']:.4f} | far: {row['far']:.4f}")

if __name__ == "__main__":
    main()
    
