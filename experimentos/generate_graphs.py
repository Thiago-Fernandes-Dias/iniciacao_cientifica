import json
import os
import re

from matplotlib import pyplot as plt

from lib.utils import float_range

directory = os.fsencode("results")


def generate_graph(title: str, label: str, file: str, index: str):
    with open(f"results/{file}") as f:
        data = json.load(f)
        metrics_dict = data[index]
        x_pos = float_range(0, 1, 0.05)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.barh(
            y=metrics_dict.keys(), width=metrics_dict.values(), align="edge", height=0.5
        )
        ax.set_xticks(x_pos)
        ax.set_xlabel(label)
        ax.set_title(title)
        plt.savefig(f"results/{file.replace('.py', '')}_{label}.png")


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if re.compile(r"(one|two)_class").search(filename) and filename.endswith(".json"):
        generate_graph(
            "False Acceptance Rate per user model", "FAR", filename, "user_model_far"
        )
        generate_graph(
            "False Rejection Rate per user model", "FRR", filename, "user_model_frr"
        )
