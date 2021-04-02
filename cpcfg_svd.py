import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

basedir = Path("cpcfg/cpcfg_30_60_16_results/")
files = [
    basedir / "30_60_softmax_nounitnorm_features_16_1e3/transitions/rules15.pt",
    basedir / "30_60_rff_nounitnorm_features_16_1e3/transitions/rules15.pt",
]

fig, axes = plt.subplots(ncols=2, sharey=True)
for fil, ax in zip(files, axes):
    x = torch.load(fil, map_location=torch.device("cpu"))
    u,s,v = x[:,:30,:30].exp().view(30,-1).svd()
    sns.scatterplot(x=np.arange(len(s)), y=s.detach().numpy(), ax=ax)
    ax.set_title(fil.parts[2].split("_")[2])
fig.savefig("cpcfg_30_60_16-svd.png")
plt.close(fig)
