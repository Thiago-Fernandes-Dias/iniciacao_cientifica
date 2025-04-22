import json
import os
import re

from pymongo import MongoClient
from matplotlib import pyplot as plt

from lib.constants import MONGO_CONN_STRING
from lib.repositories.results_repository import results_repository_factory
from lib.utils import create_dir_if_not_exists, float_range

directory = os.fsencode("./")

def generate_graph(title: str, metric: str, exp_name: str, index: str):
    repo = results_repository_factory()
    data = repo.get_one_class_results(exp_name).first()
    if data is None:
        return
    metrics_dict = data[index]
    x_pos = float_range(0, 1, 0.05)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.barh(
        y=metrics_dict.keys(), width=metrics_dict.values(), align="edge", height=0.5
    )
    ax.set_xticks(x_pos)
    ax.set_xlabel(metric)
    ax.set_title(title)
    create_dir_if_not_exists("results")
    plt.savefig(f"results/{exp_name}_{metric}.png")

def main():
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if re.compile(r"(one|two)_class").search(filename) and filename.endswith(".py"):
            exp_name = filename.replace(".py", "")
            generate_graph("False Match Rate per user model", "FAR", exp_name, "user_model_far")
            generate_graph("False Non-Match Rate per user model", "FRR", exp_name, "user_model_frr")

if __name__ == "__main__":
    main()