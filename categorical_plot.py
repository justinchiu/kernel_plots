import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="darkgrid")

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)

# old data from jax
data = np.array([
    (128 , .5982 , .28 , .1453 , .1067, 0.0038),  
    (256 , .7391 , .445 , .1944 , .177, 0.0054),  
    (512 , .8676 , .5288 , .3266 , .1266, 0.0071),  
    (1024  , 1.0 , .63 , .43 , .1706, 0.0085),  
    (2048  , .99 , .62 , .35 , .20, 0.011),  
])

# new data from pytorch
data = np.array([
    (128,  .5910, .3016, .2199, .0625, 0.0326),  
    (256,  .8771, .6665, .4304, .1476, 0.0393),  
    (512,  1.136, .7466, .3937, .2764, 0.0513),  
    (1024, 1.306, .8573, .7347, .2838, 0.0561),  
    (2048, 1.569, 1.008, .7689, .3866, 0.0537),  
])

df = pd.DataFrame(
    data[:,1:],
    columns=["64", "128", "256", "512", "softmax"],
    index=data[:,0],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3, legend=False)
g.set_axis_labels("Number of Keys", "KL")
#g.legend.set_title("Num features")
g.tight_layout()
g.savefig("cat-keys-kl.png")

"""
\begin{table}[!t]
\centering
\begin{tabular}{lrrrr}
\toprule
& \multicolumn{4}{c}{Number of features}\\
\cmidrule{2-5}
$|\mcX|$ & $64$ & $128$ & $256$ & $512$ \\
\midrule
128 & .61 & .28 & .18 & .12\\
256 & .76 & .45 & .21 & .18\\
512 & .90 & .54 & .34 & .15\\
1024  & 1.0 & .63 & .43 & .19\\
2048  & .99 & .63 & .37 & .23\\
\bottomrule
\end{tabular}
\caption{\label{fig:cat-kl}
The KL between the true synthetic discrete distribution and learned linearized model.
The number of queries is held constant at 128, while the number of keys $|\mcX|$
and features $n$ is varied.
}
\end{table}
"""
