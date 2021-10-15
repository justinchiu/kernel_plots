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

acc_data = np.array([
    #16k  8k   4k   2k 1k 
    (146, 149, 155, 163, 175,), # softmax
    (144, 139, 146, 159, 168,), # 2:1
    (144, 142, 154, 169, 178,), # 4:1
    (147, 146, 156, 180, 189,), # 8:1
])

speed_data = np.array([
    #16k   8k    4k   2k 1k
    (3415, 1027, 377, 254, 250,), # softmax
    (3992, 1139, 398, 290, 216,), # 2:1
    (2197,  692, 304, 234, 282,), # 4:1 # rerun 1k
    (1284,  462, 259, 226, 276,), # 8:1 # rerun 1k
])

softmax_data = np.vstack((
    speed_data[0],
    acc_data[0],
    np.zeros(5),
    #["softmax"] * 5,
))
lhmm_data = np.vstack((
    speed_data[1:].reshape(-1),
    acc_data[1:].reshape(-1),
    np.ones(15),
    #["low-rank"] * 15,
))
data = np.hstack((softmax_data, lhmm_data))

df = pd.DataFrame(
    data.T,
    columns = ["speed", "accuracy", "model"],
)

g = sns.relplot(
    data=df, x="speed", y="accuracy", hue="model", kind="scatter",
)
g.set_axis_labels("log(Sec / Epoch)", "Valid PPL")
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
g.savefig("lhmm-speed-accuracy.png")

