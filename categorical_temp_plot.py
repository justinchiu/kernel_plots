import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="darkgrid")

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)

data = np.array([
[0.9998, 0.9068, 0.8456, 0.693,  0.5751],
[0.6307, 0.4988, 0.3382, 0.2698, 0.1801], 
[0.4245, 0.3178, 0.2255, 0.132,  0.0702],
[0.1706, 0.136,  0.0839, 0.0628, 0.0463],
[0.0085, 0.0249, 0.0407, 0.0502, 0.0535],
])


df = pd.DataFrame(
    data.T,
    columns=["64", "128", "256", "512", "softmax"],
    index=np.arange(1,6),
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("True distribution temperature", "KL")
#g.set_title("PTB Perplexity")
g.legend.set_title("Num features")
#g._legend._loc = 1
g.tight_layout()
g.savefig("cat-temp-kl.png")

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
