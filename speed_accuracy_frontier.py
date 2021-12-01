import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

#sns.set_theme(style="darkgrid")
#sns.set_context("paper", font_scale=2)
#sns.set_context(font_scale=10)
#sns.set(font_scale=1.5)
sns.set_theme(style="white", font_scale=1.5)

pal = sns.color_palette("flare", 1) + sns.color_palette("crest", 3)

# HMM
acc_data = np.array([
    #16k  8k   4k   2k 1k 
    (146, 149, 155, 163, 175,), # softmax
    (144, 139, 146, 159, 168,), # 2:1
    (144, 142, 154, 169, 178,), # 4:1
    (147, 146, 156, 180, 189,), # 8:1
])

# 16k has .1 feature dropout...rerun all with feature dropout?
# RERUNNING 8k with feature dropout.
# use max perf for softmax hmm 16k?
acc_data = np.array([
    #16k  8k   4k   2k 1k 
    (144, 152, 155, 163, 175,), # softmax
    (141, 149, 154, 159, 168,), # 2:1
    (139, 153, 156, 169, 178,), # 4:1
    (141, 161, 163, 180, 189,), # 8:1
])

speed_data = np.array([
    #16k   8k    4k   2k 1k
    (3415, 1027, 377, 254, 250,), # softmax
    (3992, 1139, 398, 290, 216,), # 2:1
    (2197,  692, 304, 234, 282,), # 4:1 # rerun 1k
    (1284,  462, 259, 226, 276,), # 8:1 # rerun 1k
]) / 6519

state_data = np.array([
    (2**14, 2**13, 2**12, 2**11, 2**10),
    (2**14, 2**13, 2**12, 2**11, 2**10),
    (2**14, 2**13, 2**12, 2**11, 2**10),
    (2**14, 2**13, 2**12, 2**11, 2**10),
])

ratio_data = np.array([
    (1,1,1,1,1),
    (2,2,2,2,2),
    (4,4,4,4,4),
    (8,8,8,8,8),
])

size_data = np.array([
    (16384,8192,4096,2048,1024),
    (16384,8192,4096,2048,1024),
    (16384,8192,4096,2048,1024),
    (16384,8192,4096,2048,1024),
])


softmax_data = np.vstack((
    speed_data[0],
    acc_data[0],
    np.zeros(5),
    state_data[0],
    ratio_data[0],
    size_data[0],
))
lhmm_data = np.vstack((
    speed_data[1:].reshape(-1),
    acc_data[1:].reshape(-1),
    np.ones(15),
    #["low-rank"] * 15,
    state_data[1:].reshape(-1),
    ratio_data[1:].reshape(-1),
    size_data[1:].reshape(-1),
))
data = np.hstack((softmax_data, lhmm_data))

hmm_df = pd.DataFrame(
    data.T,
    columns = ["speed", "accuracy", "model", "states", "ratio", "size"],
)

g = sns.relplot(
    data=hmm_df, x="speed", y="accuracy", hue="model", kind="scatter",
    size="size",
    sizes = (100, 400),
    palette = (pal[0], pal[-1]),
)
g.set_axis_labels("Sec / Batch", "Valid PPL")
#g.legend.set_title("Model")
# replace labels
new_labels = [
    "Parameterization",
    'softmax',
    'low-rank',
    "Num states",
    "1024",
    "2048",
    "4096",
    "8192",
    "16384",
]
sns.move_legend(g,
    "right",
    #"lower center",
    bbox_to_anchor=(1.1, 0.65), ncol=1, title=None, frameon=False,
)
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#ax.set_yscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-speed-accuracy.png")

g = sns.relplot(
    data=hmm_df, x="speed", y="accuracy", hue="model", kind="scatter",
    size="size",
    palette = (pal[0], pal[-1]),
    sizes = (100, 400),
)
g.set_axis_labels("Sec / Batch", "Valid PPL")
g.legend.set_title("Model")
# replace labels
new_labels = ['softmax', 'low-rank']
sns.move_legend(g,
    "upper right",
    #bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
)
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#ax.set_yscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-speed-accuracy-nolegend.png")


# PCFG
softmax_data = np.array([
    (1/4.37, 252.60, 0, 90,  90),
    (1/2.99, 234.01, 0, 180, 180),
    (1/.98,  191.08, 0, 300, 300),
])
lhmm_data = np.array((
    (1/3.75, 247.02, 1, 90,  8),
    (1/3.74, 250.59, 1, 90,  16),
    (1/3.55, 217.24, 1, 180, 16), 
    (1/3.35, 213.81, 1, 180, 32),
    (1/1.56, 203.47, 1, 300, 32),
    (1/1.24, 194.25, 1, 300, 64),
))
data = np.vstack((softmax_data, lhmm_data))

pcfg_df = pd.DataFrame(
    data,
    columns = ["speed", "accuracy", "model", "states", "rank"],
)

g = sns.relplot(
    data=pcfg_df, x="speed", y="accuracy", hue="model", kind="scatter",
    size="states",
    sizes = (100, 400),
    palette = (pal[0], pal[-1]),
)
#g.set_axis_labels("log(Batch / Sec)", "Valid PPL")
g.set_axis_labels("Sec / Batch", "Valid PPL")
# replace labels
new_labels = [
    "Parameterization",
    'softmax',
    'low-rank',
    "Num states",
    "90",
    "180",
    "300",
]
sns.move_legend(g,
    loc="right",
    bbox_to_anchor=(1.1, 0.7), ncol=1, title=None, frameon=False,
)
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#ax.set_yscale("log", base=2)
g.tight_layout()
g.savefig("pcfg-speed-accuracy.png")

# HSMM
softmax_data = np.array([
    (1/1.28, 1.428, 0, 2**6, 0),
    (1/.45, 1.427, 0, 2**7, 0),
    (1/.13,  1.426, 0, 2**8, 0),
])
lhmm_data = np.array((
    (1/.24, 1.427, 1, 2**7, 2**7),
    (1/.20, 1.426, 1, 2**8, 2**6),
    (1/.18, 1.424, 1, 2**9, 2**5),
    (1/.10, 1.423, 1, 2**10, 2**4),
))
data = np.vstack((softmax_data, lhmm_data))

df = pd.DataFrame(
    data,
    columns = ["speed", "accuracy", "model", "states", "rank"],
)

g = sns.relplot(
    data=df, x="speed", y="accuracy", hue="model", kind="scatter",
)
#g.set_axis_labels("log(Batch / Sec)", "Valid PPL")
g.set_axis_labels("Sec / Batch", "Valid NLL (e5)")
g.legend.set_title("Model")
# replace labels
new_labels = ['softmax', 'low-rank']
sns.move_legend(g,
    "upper right",
    #"lower center",
    #bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
)
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#ax.set_yscale("log", base=2)
g.tight_layout()
g.savefig("hsmm-speed-accuracy.png")

g = sns.relplot(
    data=df, x="states", y="accuracy", hue="model", kind="line",
    linewidth=3,
)
g.set_axis_labels("Num states", "Valid NLL (e5)")
g.legend.set_title("Model")
# replace labels
new_labels = ['softmax', 'low-rank']
sns.move_legend(g,
    "upper right",
    #"lower center",
    #bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
)
for t, l in zip(g.legend.texts, new_labels):
    t.set_text(l)
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#ax.set_yscale("log", base=2)
g.tight_layout()
g.savefig("hsmm-accuracy.png")


# JOINT HMM + PCFG

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
sns.scatterplot(
    data=hmm_df, x="speed", y="accuracy", hue="model", ax=ax1,
    legend=False,
)
sns.scatterplot(
    data=pcfg_df, x="speed", y="accuracy", hue="model", ax=ax2,
)

ax1.set_title("HMM Speed vs Accuracy")
ax2.set_title("PCFG Speed vs Accuracy")

ax1.set_xlabel("Sec / Batch")
ax1.set_ylabel("Valid PPL")
ax2.set_xlabel("Sec / Batch")
ax2.set_ylabel("Valid PPL")

ax1.set_xscale("log", base=2)
ax2.set_xscale("log", base=2)

ax1.legend().set_visible(False)
#ax2.legend().set_visible(False)

ax2.legend().set_title("Model")
# replace labels
new_labels = ['softmax', 'low-rank']
for t, l in zip(ax2.legend().texts, new_labels):
    t.set_text(l)
#ax.set_yscale("log", base=2)
fig.tight_layout()
fig.savefig("hmm-pcfg-speed-accuracy.png")

