# %%
from dataclasses import asdict, dataclass

import matplotlib
# matplotlib.use("kitcat")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import (LeaveOneOut, cross_validate,
                                     train_test_split)

import mod

plt.style.use("seaborn-v0_8-colorblind")
color_pal = sns.color_palette()

# %%
ME03 = mod.get_data("ME03")
ME04 = mod.get_data("ME04")
ME06 = mod.get_data("ME06")
ME07 = mod.get_data("ME07")
ME10 = mod.get_data("ME10")
ME14 = mod.get_data("ME14")
ME15 = mod.get_data("ME15")
ME_All = mod.get_data("ME_All")

full_part = [ME04, ME07, ME14, ME15]
all_part = [ME03, ME04, ME06, ME07, ME10, ME14, ME15]

# %%

# for i in range(full_part):

#     brake_means, brake_rmses = mod.run_brake(full_part[i])
#     prop_means, prop_rmses = mod.run_prop(full_part[i])

#     mod.plot_metrics(brake_means, brake_rmses, prop_means, prop_rmses)

# %% Run metrics

for participant in full_part:

    mod.plot_metrics(participant)

mod.plot_metrics(ME_All)
# %%

for participant in full_part:

    mod.run_apgrf(participant, 18)

mod.run_apgrf(ME_All, 18)

# %% APGRF single part feature selection and training

participant = ME_All
X, y = mod.get_APGRF(participant)
model1 = mod.train_apgrf(X, y, 18)

X_test = X[-1, :, :]
# X_test = np.mean(X, axis=0)
y_pred = model1.predict(X_test)
y_test = y[:, -1]


fig, ax = plt.subplots(figsize=(10, 5), sharex=True, sharey=True)

ax.plot(range(len(y_test)), y_test, color=color_pal[0], label="Measured")
ax.plot(range(len(y_pred)), y_pred, color=color_pal[1], label="Estimated")
ax.legend(loc="upper center", ncols=2)
ax.set_title(f"{participant.name} APGRF")
ax.set_ylabel("%BW")
ax.set_xlabel("% Stance")

# %%
ax[0].scatter(X, y, label="Train data points")
ax[0].plot(
    X_train,
    y_pred,
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0].set(xlabel="Feature", ylabel="Target", title="Train set")
ax[0].legend()

ax[1].scatter(X_test, y_test, label="Test data points")
ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[1].legend()

fig.suptitle("Linear Regression")

plt.show()

# %% APGRF Full_part feature selection and training
rank = np.empty(18)
for n in range(len(full_part)):

    X, y = mod.get_APGRF(full_part[n])
    modelx = mod.train_apgrf(X, y, 5)

# %% APGRF All part feature selection and training
rank = np.empty(18)
for n in range(len(all_part)):

    X, y = mod.get_APGRF(all_part[n])
    modelx = mod.train_apgrf(X, y, 5)
