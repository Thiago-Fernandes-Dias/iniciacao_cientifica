from pathlib import Path

import numpy as np
import pandas as pd


FOLDERS = [
    ("../results/DeepPrint/fvc_2000_db1_a.csv", "../results/DeepPrint/fvc_2000_db1_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db1_b.csv", "../results/DeepPrint/fvc_2000_db1_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db2_a.csv", "../results/DeepPrint/fvc_2000_db2_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db2_b.csv", "../results/DeepPrint/fvc_2000_db2_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db3_a.csv", "../results/DeepPrint/fvc_2000_db3_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db3_b.csv", "../results/DeepPrint/fvc_2000_db3_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db4_a.csv", "../results/DeepPrint/fvc_2000_db4_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2000_db4_b.csv", "../results/DeepPrint/fvc_2000_db4_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db1_a.csv", "../results/DeepPrint/fvc_2002_db1_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db1_b.csv", "../results/DeepPrint/fvc_2002_db1_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db2_a.csv", "../results/DeepPrint/fvc_2002_db2_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db2_b.csv", "../results/DeepPrint/fvc_2002_db2_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db3_a.csv", "../results/DeepPrint/fvc_2002_db3_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db3_b.csv", "../results/DeepPrint/fvc_2002_db3_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db4_a.csv", "../results/DeepPrint/fvc_2002_db4_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2002_db4_b.csv", "../results/DeepPrint/fvc_2002_db4_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db1_a.csv", "../results/DeepPrint/fvc_2004_db1_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db1_b.csv", "../results/DeepPrint/fvc_2004_db1_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db2_a.csv", "../results/DeepPrint/fvc_2004_db2_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db2_b.csv", "../results/DeepPrint/fvc_2004_db2_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db3_a.csv", "../results/DeepPrint/fvc_2004_db3_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db3_b.csv", "../results/DeepPrint/fvc_2004_db3_b_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db4_a.csv", "../results/DeepPrint/fvc_2004_db4_a_metrics.csv"),
    ("../results/DeepPrint/fvc_2004_db4_b.csv", "../results/DeepPrint/fvc_2004_db4_b_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD_metrics.csv"),
    ("../results/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv", "../results/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD_metrics.csv"),
]


def map_range(value, src_min, src_max, dst_min, dst_max):
    return (value - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


def calculate_metrics(input_csv, output_csv):
    csv = Path(input_csv).resolve()
    if not csv.is_file() or not csv.name.endswith(".csv"):
        print(f"Skipping invalid input csv: {input_csv}")
        return

    comps = pd.read_csv(csv)

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
    result.to_csv(Path(output_csv), index=False)


def main():
    for input_csv, output_csv in FOLDERS:
        print(f"Processing: {input_csv} (saving results to: {output_csv})")
        calculate_metrics(input_csv, output_csv)


if __name__ == "__main__":
    main()
