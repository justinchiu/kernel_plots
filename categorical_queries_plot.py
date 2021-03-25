import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="darkgrid")

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)

data = np.array([
    (32,   .395,  .248, .2199, .0937, 0.0112),  
    (64 ,  .876,  .537, .3819, .2035, 0.0242),  
    (128, 1.308,  .857, .7347, .2838, 0.0561),  
    (256, 1.716, 1.274, .9449, .6793, 0.0973),  
])

df = pd.DataFrame(
    data[:,1:],
    columns=["64", "128", "256", "512", "softmax"],
    index=data[:,0],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3, legend=False)
g.set_axis_labels("Number of Queries", "KL")
#g.legend.set_title("Num features")
g.tight_layout()
g.savefig("cat-queries-kl.png")

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
