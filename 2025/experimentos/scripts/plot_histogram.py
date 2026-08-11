from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

BIN_WIDTH = 0.05

FILES = [
    "../results/DeepPrint/fvc_2000_db1_a.csv",
    "../results/DeepPrint/fvc_2000_db1_b.csv",
    "../results/DeepPrint/fvc_2000_db2_a.csv",
    "../results/DeepPrint/fvc_2000_db2_b.csv",
    "../results/DeepPrint/fvc_2000_db3_a.csv",
    "../results/DeepPrint/fvc_2000_db3_b.csv",
    "../results/DeepPrint/fvc_2000_db4_a.csv",
    "../results/DeepPrint/fvc_2000_db4_b.csv",
    "../results/DeepPrint/fvc_2002_db1_a.csv",
    "../results/DeepPrint/fvc_2002_db1_b.csv",
    "../results/DeepPrint/fvc_2002_db2_a.csv",
    "../results/DeepPrint/fvc_2002_db2_b.csv",
    "../results/DeepPrint/fvc_2002_db3_a.csv",
    "../results/DeepPrint/fvc_2002_db3_b.csv",
    "../results/DeepPrint/fvc_2002_db4_a.csv",
    "../results/DeepPrint/fvc_2002_db4_b.csv",
    "../results/DeepPrint/fvc_2004_db1_a.csv",
    "../results/DeepPrint/fvc_2004_db1_b.csv",
    "../results/DeepPrint/fvc_2004_db2_a.csv",
    "../results/DeepPrint/fvc_2004_db2_b.csv",
    "../results/DeepPrint/fvc_2004_db3_a.csv",
    "../results/DeepPrint/fvc_2004_db3_b.csv",
    "../results/DeepPrint/fvc_2004_db4_a.csv",
    "../results/DeepPrint/fvc_2004_db4_b.csv",
    "../results/FLARE/FVC_2000_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2000_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2002_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB1_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB2_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB3_B/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_A/FDD_feat_VotingPose/score_FDD.csv",
    "../results/FLARE/FVC_2004_DB4_B/FDD_feat_VotingPose/score_FDD.csv",
]


def plot_histogram(input_csv: str):
    csv_path = (SCRIPT_DIR / input_csv).resolve()
    if not csv_path.is_file():
        print(f"File not found: {input_csv}")
        return

    df = pd.read_csv(csv_path)

    genuine = df[df["user_1"] == df["user_2"]]["score"]
    impostor = df[df["user_1"] != df["user_2"]]["score"]

    all_scores = pd.concat([genuine, impostor])
    bin_edges = np.arange(
        np.floor(all_scores.min() / BIN_WIDTH) * BIN_WIDTH,
        np.ceil(all_scores.max() / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH,
        BIN_WIDTH,
    )

    plt.figure(figsize=(10, 6))
    plt.hist(genuine, bins=bin_edges, alpha=0.5, label="Mesmo usuário", color="blue", edgecolor="black")
    plt.hist(impostor, bins=bin_edges, alpha=0.5, label="Usuários diferentes", color="red", edgecolor="black")
    plt.xlabel("Score")
    plt.ylabel("Quantidade")
    plt.title(csv_path.stem)
    plt.legend()
    plt.tight_layout()

    output_png = csv_path.with_suffix(".png")
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Histogram saved to: {output_png}")


def main():
    for input_csv in FILES:
        print(f"Processing: {input_csv}")
        plot_histogram(input_csv)


if __name__ == "__main__":
    main()
