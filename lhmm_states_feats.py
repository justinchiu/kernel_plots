import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="darkgrid")
#sns.set_context("paper", font_scale=2)
#sns.set_context(font_scale=10)
sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)

data = np.array([
    (64, 206, 204, 206, 219),
    (128, 188, 193, 192, 201,),
    (256, 179, 185, 184, 192,),
    (512, 174, 173, 178, 191,),
])

df = pd.DataFrame(
    data[:,1:],
    columns=[4096, 2048, 1024, 512],
    index=data[:,0],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of features", "PPL")
g.legend.set_title("Num classes")
g.tight_layout()
g.savefig("hmm-states-features.png")

data = np.array([
    (64, 206, 204, 206, 219),
    (128, 188, 193, 192, 201,),
    (256, 179, 185, 184, 192,),
    (512, 174, 173, 178, 191,),
])

df = pd.DataFrame(
    data[:,1:].T,
    #columns=[4096, 2048, 1024, 512],
    #index=data[:,0],
    columns=data[:,0],
    index=[4096, 2048, 1024, 512],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of classes", "PPL")
g.legend.set_title("Num features")
g.tight_layout()
g.savefig("hmm-states-features-transpose.png")
