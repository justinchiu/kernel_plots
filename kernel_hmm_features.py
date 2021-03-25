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
    (32, 190, 235),
    (64, 174, 219),
    (128, 162, 201),
    (256, 149, 192),
    (512, 146, 192),
])

df = pd.DataFrame(
    data[:,1:],
    columns=["Train", "Val"],
    index=data[:,0],
)

g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of features", "PPL")
g.tight_layout()
g.savefig("hmm-ppl-features.png")

"""
\begin{comment}
\begin{table}[!t]
\centering
\begin{tabular}{lrrr}
\toprule
$n$ & Train & Val \\
\midrule
32  & 190 & 235\\
64  & 174 & 219\\
128 & 162 & 201 \\
256 & 149 & 192\\
512 & 146 & 192\\
\bottomrule
\end{tabular}
\caption{\label{fig:hmm-ppl-features}
Perplexities on \textsc{PTB} for a LHMM with a varying number of features $n$
and a fixed number of states $|\mcZ| = 512$.}
\end{table}
\end{comment}
"""
