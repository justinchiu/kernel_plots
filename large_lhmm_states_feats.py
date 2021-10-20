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
g.set_axis_labels("Number of states", "Valid PPL")
g.legend.set_title("State:Rank")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.set(ylim=(135,195))
g.tight_layout()
g.savefig("lhmm-states-features.png")


data = np.array([
    #16k  8k   4k   2k 1k 
    (146, 149, 155, 163, 175,), # softmax
    (144, 139, 146, 159, 168,), # 2:1
    (144, 142, 154, 169, 178,), # 4:1
    (147, 146, 156, 180, 189,), # 8:1
])
data = np.array([
    #16k  8k   4k   2k 1k 
    (144, 151, 155, 163, 175,), # softmax
    (141, 142, 154, 159, 168,), # 2:1
    (139, 145, 156, 169, 178,), # 4:1
    (141, 143, 163, 180, 189,), # 8:1
])

acc_df = pd.DataFrame(
    data.T,
    columns=["softmax", "2:1", "4:1", "8:1"],
    index=[16384, 8192, 4096, 2048, 1024],
)

g = sns.relplot(data=acc_df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "Valid PPL")
g.legend.set_title("State:Rank")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.set(ylim=(135,195))
g.tight_layout()
g.savefig("lhmm-states-features-dropout.png")

data = np.array([
    #16k   8k    4k   2k 1k
    (3415, 1027, 377, 254, 250,), # softmax # rerun bsz
    (3992, 1139, 398, 290, 216,), # 2:1
    (2197,  692, 304, 234, 282,), # 4:1 # rerun 1k
    (1284,  462, 259, 226, 276,), # 8:1 # rerun 1k
]) / 6519

speed_df = pd.DataFrame(
    data.T,
    columns=["softmax", "2:1", "4:1", "8:1"],
    index=[16384, 8192, 4096, 2048, 1024],
)

g = sns.relplot(data=speed_df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "Secs / Batch")
g.legend.set_title("State:Rank")
ax = g.axes[0][0]
ax.set_yscale("log", base=2)
ax.set_xscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-states-features-speed-log.png")

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "Secs / Batch")
g.legend.set_title("State:Rank")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
g.tight_layout()
g.savefig("lhmm-states-features-speed.png")

# joint plot
pal = sns.color_palette("flare", 1) + sns.color_palette("crest", 3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
sns.lineplot(
    data=acc_df, linewidth=3, ax=ax1, legend=False, dashes=False,
    palette=pal,
)
ax1.set_xlabel("Number of states")
ax1.set_ylabel("Valid PPL")
ax1.set_xscale("log", base=2)
ax1.set(ylim=(135,195))
sns.lineplot(
    data=speed_df, linewidth=3, ax=ax2, dashes=False,
    palette=pal,
)
ax2.set_xlabel("Number of states")
ax2.set_ylabel("Secs / Batch")
ax2.set_yscale("log", base=2)
ax2.set_xscale("log", base=2)

#ax1.legend().set_visible(False)
ax2.legend().set_visible(False)

handles, labels = ax2.get_legend_handles_labels()
l4 = fig.legend(
    bbox_to_anchor=(.5,1.02),
    loc="upper center",
    borderaxespad=0,
    ncol=4,
    title="State:Rank",
)

fig.savefig("lhmm-speed-acc-joint.png", bbox_inches="tight")
plt.close(fig)


data = np.array([
    #2k    1k    512   256   128
    (5.78, 5.91, 6.19, 6.54, 7.10,), # softmax
    (5.77, 5.78, 6.27, 6.71, 7.22,), # 2:1
    (5.83, 5.93, 6.38, 6.77, 7.26,), # 4:1
    (5.93, 6.00, 6.45, 6.88, 7.35,), # 8:1
])

df = pd.DataFrame(
    data.T,
    columns=["softmax", "2:1", "4:1", "8:1"],
    index=[2048, 1024, 512, 256, 128],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of states", "Valid NLL")
g.legend.set_title("State:Rank")
ax = g.axes[0][0]
ax.set_xscale("log", base=2)
#g.set(ylim=(135,195))
g.fig.get_axes()[0].legend(loc="upper right", title="State:Rank", frameon=False)
g.tight_layout()
g.savefig("music-states-features-dropout.png")
