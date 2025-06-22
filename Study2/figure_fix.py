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
for participant in full_part:

    mod.run_apgrf(participant, 18)

mod.run_apgrf(ME_All, 18)

# %%

# def run_apgrf(participant, features):
participant = ME04
name = "Participant 1"
features = 18

X, y = mod.get_APGRF(participant)
metrics = mod.features_apgrf(X, y, features)

X_train, X_test, y_train, y_test = mod.split_apgrf(X, y, 0.7)

model = mod.train_apgrf(X_train, y_train, metrics)

y_pred = model.predict(X_test)

# save_apgrf(participant, y_test, y_pred)

fig, ax = mod.plot_apgrf(y_test, y_pred)
ax.set_title(f"{name} APGRF")
filename = f"figures/{name}_apgrf.pdf"
plt.savefig(filename, bbox_inches="tight")
plt.close(fig)

#  %%
participant = ME_All
name = "All Participants Combined"
brake_mag, brake_imp = mod.get_brake(participant)[1:]
prop_mag, prop_imp = mod.get_prop(participant)[1:]

brake_mag_mean = np.mean(brake_mag)
brake_mag_std = np.std(brake_mag)
brake_imp_mean = np.mean(brake_imp)
brake_imp_std = np.std(brake_imp)
prop_mag_mean = np.mean(prop_mag)
prop_mag_std = np.std(prop_mag)
prop_imp_mean = np.mean(prop_imp)
prop_imp_std = np.std(prop_imp)
brake_est_means, brake_est_stds = mod.run_brake(participant)
prop_est_means, prop_est_stds = mod.run_prop(participant)

metrics = (
    "Peak Braking\nMagnitude",
    "Braking\nImpulse",
    "Peak Propulsion\nMagnitude",
    "Propulsion\nImpulse",
)
heights = {
    "Measured": np.round(
        (-brake_mag_mean, -brake_imp_mean, prop_mag_mean, prop_imp_mean), 2
    ),
    "Estimated": np.round(
        (
            -brake_est_means[0],
            -brake_est_means[1],
            prop_est_means[0],
            prop_est_means[1],
        ),
        2,
    ),
}

std_bars = [
    brake_mag_std / 2,
    brake_est_stds[0],
    brake_imp_std / 2,
    brake_est_stds[1],
    prop_mag_std / 2,
    prop_est_stds[0],
    prop_imp_std / 2,
    prop_est_stds[1],
]

x = np.arange(len(metrics))
width = 0.25
multiplier = 0
i = 0

fig, ax = plt.subplots(layout="constrained")

for measure, result in heights.items():

    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        result,
        width,
        label=measure,
        yerr=std_bars[i],  # / 2,
        error_kw={"elinewidth": 1, "capsize": 3, "capthick": 1},
    )
    ax.bar_label(rects, padding=3)
    multiplier += 1
    i += 1

ax.set_ylim(0, np.max(np.abs(brake_mag_mean)) + np.max(std_bars) + 2)
ax.set_title(f"{name} Metrics")
ax.set_xticks(x + (width / 2), metrics)
ax.legend(loc="upper center", ncols=2)

# plt.show()

filename = f"figures/{name}_metrics.pdf"
plt.savefig(filename, bbox_inches="tight")

# return fig
