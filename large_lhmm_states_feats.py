import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

#sns.set_theme(style="darkgrid")
#sns.set_context("paper", font_scale=2)
#sns.set_context(font_scale=10)
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)

data = np.array([
    #16k  8k   4k   2k   1k   
    (159, 160, 165, 172, 182,), # softmax
    (157, 162, 165, 168, 179,), # 2:1
    (156, 162, 163, 177, 180,), # 4:1
    (163, 164, 177, 182, 192,), # 8:1
])

df = pd.DataFrame(
    data.T,
    #columns=[4096, 2048, 1024, 512],
    #index=data[:,0],
    columns=["softmax", "2:1", "4:1", "8:1"],
    index=[16384, 8192, 4096, 2048, 1024],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "PPL")
g.legend.set_title("State:Features")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-states-features.png")


data = np.array([
    #16k  8k   4k 
    (146, 149, 155), # softmax
    (  1, 142, 154), # 4:1
    (  1, 146, 156), # 8:1
])

df = pd.DataFrame(
    data.T,
    columns=["softmax", "4:1", "8:1"],
    index=[16384, 8192, 4096],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "PPL")
g.legend.set_title("State:Features")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-states-features-dropout.png")

data = np.array([
    #16k   8k    4k 
    (5518, 1571, 550), # softmax
    (3992, 1139, 398), # 2:1
    (2197,  692, 304), # 4:1
    (1284,  462, 259), # 8:1
])

df = pd.DataFrame(
    data.T,
    columns=["softmax", "2:1", "4:1", "8:1"],
    index=[16384, 8192, 4096],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "Secs / Epoch")
g.legend.set_title("State:Features")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-states-features-speed.png")
