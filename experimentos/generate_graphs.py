import json

import numpy as np
from matplotlib import pyplot as plt

from lib.utils import float_range

with open("results/one_class_lw.py_2025-02-15_09-25-26.json") as f:
    data = json.load(f)
    users = data["user_model_far"].keys()
    x_pos = float_range(0, 1, 0.1)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.barh(y=data["user_model_far"].keys(), width=data["user_model_far"].values(), align="edge", height=0.5)
    ax.set_xticks(x_pos)
    ax.set_xlabel('FAR')
    ax.set_title('False Acceptance Rate per user model')
    plt.savefig("results/one_class_lw.png", )
