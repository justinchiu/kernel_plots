import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_theme(style="darkgrid")

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.5)


# temperature

data = np.array([
[9e-6, 3e-5, 2.8e-5],
[1.056, .948, .884],
[.0022, .0030, .0105],
])

df = pd.DataFrame(
    data.T,
    index=np.arange(1, 4),
    columns=["sm", "k", "klt"],
)
g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("True distribution temperature", "KL")
#g.set_title("PTB Perplexity")
#g._legend._loc = 1
g.tight_layout()
g.savefig("cat-exp-plots/cat-temp-kl.png")

# queries
 
data = np.array([
[3.2e-6, 3.5e-6, 5.5e-6, 6.2e-6, 0],
[.455, .5529, .6044, .8682, .7906],
[.0005, .0014, .0021, .0051, .0132],
])

df = pd.DataFrame(
    data.T,
    index=[64, 128, 256, 512, 1024],
    columns=["sm", "k", "klt"],
)
g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of queries", "KL")
#g.set_title("PTB Perplexity")
#g._legend._loc = 1
g.tight_layout()
g.savefig("cat-exp-plots/cat-queries-kl.png")

# keys
 
data = np.array([
[3e-6, 3.5e-6, 5e-6, 5.6e-6, 5.6e-6],
[.36, .55, .72, .89, .97],
[.0017, .0014, .0019, .0027, .0062],
])

df = pd.DataFrame(
    data.T,
    index=[64, 128, 256, 512, 1024],
    columns=["sm", "k", "klt"],
)
g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Number of keys", "KL")
#g.set_title("PTB Perplexity")
#g._legend._loc = 1
g.tight_layout()
g.savefig("cat-exp-plots/cat-keys-kl.png")

# rank
data = np.array([
[7.6e-6, 1.3e-5, 9.4e-6, 3.9e-6, 1.8e-6], # sm
[.75, .95, 1.05, 1.03, 1.09], # k
[.0017, .0022, .0022, .0043, .0128], # klt
])

df = pd.DataFrame(
    data.T,
    index=[32, 64, 128, 256, 512],
    columns=["sm", "k", "klt"],
)
g = sns.relplot(data=df, kind="line", linewidth=3, aspect=1.3)
g.set_axis_labels("Teacher emb dim", "KL")
#g.set_title("PTB Perplexity")
#g._legend._loc = 1
g.tight_layout()
g.savefig("cat-exp-plots/cat-emb-kl.png")
