import pickle
import warnings
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lofo import Dataset, LOFOImportance, plot_importance
from sklearn import linear_model
from sklearn.feature_selection import f_regression
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_validate, train_test_split)

# Suppress PerformanceWarning
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*PerformanceWarning.*"
)


plt.style.use("fivethirtyeight")
color_pal = sns.color_palette()


@dataclass(frozen=True)
class Participant:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))

    name: str
    apgrf: float

    brake_mag: float
    brake_imp: float

    brake_thigh_accX: float
    brake_thigh_accY: float
    brake_thigh_accZ: float
    brake_thigh_gyroX: float
    brake_thigh_gyroY: float
    brake_thigh_gyroZ: float

    brake_shank_accX: float
    brake_shank_accY: float
    brake_shank_accZ: float
    brake_shank_gyroX: float
    brake_shank_gyroY: float
    brake_shank_gyroZ: float

    brake_foot_accX: float
    brake_foot_accY: float
    brake_foot_accZ: float
    brake_foot_gyroX: float
    brake_foot_gyroY: float
    brake_foot_gyroZ: float

    prop_mag: float
    prop_imp: float

    prop_thigh_accX: float
    prop_thigh_accY: float
    prop_thigh_accZ: float
    prop_thigh_gyroX: float
    prop_thigh_gyroY: float
    prop_thigh_gyroZ: float

    prop_shank_accX: float
    prop_shank_accY: float
    prop_shank_accZ: float
    prop_shank_gyroX: float
    prop_shank_gyroY: float
    prop_shank_gyroZ: float

    prop_foot_accX: float
    prop_foot_accY: float
    prop_foot_accZ: float
    prop_foot_gyroX: float
    prop_foot_gyroY: float
    prop_foot_gyroZ: float

    thigh_accX: float
    thigh_accY: float
    thigh_accZ: float
    thigh_gyroX: float
    thigh_gyroY: float
    thigh_gyroZ: float

    shank_accX: float
    shank_accY: float
    shank_accZ: float
    shank_gyroX: float
    shank_gyroY: float
    shank_gyroZ: float

    foot_accX: float
    foot_accY: float
    foot_accZ: float
    foot_gyroX: float
    foot_gyroY: float
    foot_gyroZ: float


@dataclass(frozen=True)
class Results:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))

    brake_mag: object
    brake_imp: object

    prop_mag: object
    prop_imp: object

    metrics_plot: object

    apgrf: object


def get_data(participant_name):
    file_name = f"data/{participant_name}.xls"
    df = pd.ExcelFile(file_name)

    apgrf = pd.read_excel(df, "Sheet1", index_col=None, header=None)

    brake = pd.read_excel(df, "Sheet2", index_col=None, header=None)
    prop = pd.read_excel(df, "Sheet3", index_col=None, header=None)

    brake_mag = brake.iloc[:, 0]
    brake_imp = brake.iloc[:, 1]
    prop_mag = prop.iloc[:, 0]
    prop_imp = prop.iloc[:, 1]

    brake_thigh_accX = brake.iloc[:, 2]
    brake_thigh_accY = brake.iloc[:, 3]
    brake_thigh_accZ = brake.iloc[:, 4]
    brake_thigh_gyroX = brake.iloc[:, 5]
    brake_thigh_gyroY = brake.iloc[:, 6]
    brake_thigh_gyroZ = brake.iloc[:, 7]

    brake_shank_accX = brake.iloc[:, 8]
    brake_shank_accY = brake.iloc[:, 9]
    brake_shank_accZ = brake.iloc[:, 10]
    brake_shank_gyroX = brake.iloc[:, 11]
    brake_shank_gyroY = brake.iloc[:, 12]
    brake_shank_gyroZ = brake.iloc[:, 13]

    brake_foot_accX = brake.iloc[:, 14]
    brake_foot_accY = brake.iloc[:, 15]
    brake_foot_accZ = brake.iloc[:, 16]
    brake_foot_gyroX = brake.iloc[:, 17]
    brake_foot_gyroY = brake.iloc[:, 18]
    brake_foot_gyroZ = brake.iloc[:, 19]

    prop_thigh_accX = prop.iloc[:, 2]
    prop_thigh_accY = prop.iloc[:, 3]
    prop_thigh_accZ = prop.iloc[:, 4]
    prop_thigh_gyroX = prop.iloc[:, 5]
    prop_thigh_gyroY = prop.iloc[:, 6]
    prop_thigh_gyroZ = prop.iloc[:, 7]

    prop_shank_accX = prop.iloc[:, 8]
    prop_shank_accY = prop.iloc[:, 9]
    prop_shank_accZ = prop.iloc[:, 10]
    prop_shank_gyroX = prop.iloc[:, 11]
    prop_shank_gyroY = prop.iloc[:, 12]
    prop_shank_gyroZ = prop.iloc[:, 13]

    prop_foot_accX = prop.iloc[:, 14]
    prop_foot_accY = prop.iloc[:, 15]
    prop_foot_accZ = prop.iloc[:, 16]
    prop_foot_gyroX = prop.iloc[:, 17]
    prop_foot_gyroY = prop.iloc[:, 18]
    prop_foot_gyroZ = prop.iloc[:, 19]

    thigh_accX = pd.read_excel(df, "Sheet4", index_col=None, header=None)
    thigh_accY = pd.read_excel(df, "Sheet5", index_col=None, header=None)
    thigh_accZ = pd.read_excel(df, "Sheet6", index_col=None, header=None)
    thigh_gryoX = pd.read_excel(df, "Sheet7", index_col=None, header=None)
    thigh_gyroY = pd.read_excel(df, "Sheet8", index_col=None, header=None)
    thigh_gyroZ = pd.read_excel(df, "Sheet9", index_col=None, header=None)

    shank_accX = pd.read_excel(df, "Sheet10", index_col=None, header=None)
    shank_accY = pd.read_excel(df, "Sheet11", index_col=None, header=None)
    shank_accZ = pd.read_excel(df, "Sheet12", index_col=None, header=None)
    shank_gryoX = pd.read_excel(df, "Sheet13", index_col=None, header=None)
    shank_gyroY = pd.read_excel(df, "Sheet14", index_col=None, header=None)
    shank_gyroZ = pd.read_excel(df, "Sheet15", index_col=None, header=None)

    foot_accX = pd.read_excel(df, "Sheet16", index_col=None, header=None)
    foot_accY = pd.read_excel(df, "Sheet17", index_col=None, header=None)
    foot_accZ = pd.read_excel(df, "Sheet18", index_col=None, header=None)
    foot_gryoX = pd.read_excel(df, "Sheet19", index_col=None, header=None)
    foot_gyroY = pd.read_excel(df, "Sheet20", index_col=None, header=None)
    foot_gyroZ = pd.read_excel(df, "Sheet21", index_col=None, header=None)

    return Participant(
        participant_name,
        apgrf,
        brake_mag,
        brake_imp,
        brake_thigh_accX,
        brake_thigh_accY,
        brake_thigh_accZ,
        brake_thigh_gyroX,
        brake_thigh_gyroY,
        brake_thigh_gyroZ,
        brake_shank_accX,
        brake_shank_accY,
        brake_shank_accZ,
        brake_shank_gyroX,
        brake_shank_gyroY,
        brake_shank_gyroZ,
        brake_foot_accX,
        brake_foot_accY,
        brake_foot_accZ,
        brake_foot_gyroX,
        brake_foot_gyroY,
        brake_foot_gyroZ,
        prop_mag,
        prop_imp,
        prop_thigh_accX,
        prop_thigh_accY,
        prop_thigh_accZ,
        prop_thigh_gyroX,
        prop_thigh_gyroY,
        prop_thigh_gyroZ,
        prop_shank_accX,
        prop_shank_accY,
        prop_shank_accZ,
        prop_shank_gyroX,
        prop_shank_gyroY,
        prop_shank_gyroZ,
        prop_foot_accX,
        prop_foot_accY,
        prop_foot_accZ,
        prop_foot_gyroX,
        prop_foot_gyroY,
        prop_foot_gyroZ,
        thigh_accX,
        thigh_accY,
        thigh_accZ,
        thigh_gryoX,
        thigh_gyroY,
        thigh_gyroZ,
        shank_accX,
        shank_accY,
        shank_accZ,
        shank_gryoX,
        shank_gyroY,
        shank_gyroZ,
        foot_accX,
        foot_accY,
        foot_accZ,
        foot_gryoX,
        foot_gyroY,
        foot_gyroZ,
    )


def get_brake(participant):

    brake_mag = participant.brake_mag
    brake_imp = participant.brake_imp

    X = pd.DataFrame(
        [
            participant.brake_thigh_accX,
            participant.brake_thigh_accY,
            participant.brake_thigh_accZ,
            participant.brake_thigh_gyroX,
            participant.brake_thigh_gyroY,
            participant.brake_thigh_gyroZ,
            participant.brake_shank_accX,
            participant.brake_shank_accY,
            participant.brake_shank_accZ,
            participant.brake_shank_gyroX,
            participant.brake_shank_gyroY,
            participant.brake_shank_gyroZ,
            participant.brake_foot_accX,
            participant.brake_foot_accY,
            participant.brake_foot_accZ,
            participant.brake_foot_gyroX,
            participant.brake_foot_gyroY,
            participant.brake_foot_gyroZ,
        ]
    ).T

    return X, brake_mag, brake_imp


def get_prop(participant):

    prop_mag = participant.prop_mag
    prop_imp = participant.prop_imp

    X = pd.DataFrame(
        [
            participant.prop_thigh_accX,
            participant.prop_thigh_accY,
            participant.prop_thigh_accZ,
            participant.prop_thigh_gyroX,
            participant.prop_thigh_gyroY,
            participant.prop_thigh_gyroZ,
            participant.prop_shank_accX,
            participant.prop_shank_accY,
            participant.prop_shank_accZ,
            participant.prop_shank_gyroX,
            participant.prop_shank_gyroY,
            participant.prop_shank_gyroZ,
            participant.prop_foot_accX,
            participant.prop_foot_accY,
            participant.prop_foot_accZ,
            participant.prop_foot_gyroX,
            participant.prop_foot_gyroY,
            participant.prop_foot_gyroZ,
        ]
    ).T

    return X, prop_mag, prop_imp


def LOFO(X, y):
    labels = [
        "y",
        "Thigh Acc X",
        "Thigh Acc Y",
        "Thigh Acc Z",
        "Thigh Gyro X",
        "Thigh Gyro Y",
        "Thigh Gyro Z",
        "Shank Acc X",
        "Shank Acc Y",
        "Shank Acc Z",
        "Shank Gyro X",
        "Shank Gyro Y",
        "Shank Gyro Z",
        "Foot Acc X",
        "Foot Acc Y",
        "Foot Acc Z",
        "Foot Gyro X",
        "Foot Gyro Y",
        "Foot Gyro Z",
    ]

    df = pd.concat([y, X], axis=1)
    df.columns = labels

    model = linear_model.LinearRegression()
    cv = LeaveOneOut()

    dataset = Dataset(df=df, target="y", features=[col for col in df if col != "y"])
    lofo_imp = LOFOImportance(
        dataset, model=model, cv=cv, scoring="neg_root_mean_squared_error"
    )
    importance_df = lofo_imp.get_importance()

    # hold = importance_df[importance_df['importance_mean'] > 0]['feature']
    hold = importance_df["feature"][0:3]
    selected = [labels.index(hold) if hold in labels else -1 for hold in hold]
    print(f"Features selected: {list(hold)}")

    return selected, hold


def LOOF(X, y, data):

    if data == "metrics":

        strides = np.size(y)
        metrics = np.shape(X)[1]
        final_ranks = np.empty(metrics) * 0
        for i in range(metrics):

            X_train = np.delete(X, i, axis=1)
            f_stat, p_values = f_regression(X_train, y)
            order = f_stat.argsort()
            ranks = order.argsort()

            final_ranks = final_ranks + np.insert(ranks, i, 0)

        hold = np.argpartition(final_ranks, 3)

    if data == "apgrf":

        strides = np.shape(y)[1]
        metrics = np.shape(X)[2]
        final_ranks = np.empty(metrics) * 0
        for i in range(metrics):

            X_hold = np.delete(X, i, axis=2)

            for k in range(strides):

                f_stat, p_values = f_regression(X_hold[k, :, :], y[:, k])
                order = f_stat.argsort()
                ranks = order.argsort()

                final_ranks = final_ranks + np.insert(ranks, i, 0)

        hold = np.argpartition(final_ranks, 3)

    return hold


def train_metrics(X, y):

    model = linear_model.LinearRegression()
    cv = LeaveOneOut()
    scoring = [
        "neg_mean_absolute_percentage_error",
        "neg_root_mean_squared_error",
    ]  # , 'r2']
    scores = cross_validate(
        model, X, y, scoring=scoring, cv=cv, return_train_score=True
    )
    y_pred = cross_val_predict(model, X, y, cv=cv)

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_pred_mean = np.mean(y_pred)
    y_pred_std = np.std(y_pred)
    MAPE_mean = np.mean(np.abs(scores["test_neg_mean_absolute_percentage_error"]))
    MAPE_std = np.std(np.abs(scores["test_neg_mean_absolute_percentage_error"]))
    RMSE_mean = np.mean(scores["test_neg_root_mean_squared_error"])
    RMSE_std = np.std(scores["test_neg_root_mean_squared_error"])

    print(f"Y mean: {y_mean:.2f}, StD: {y_std:.2f}")
    print(f"Y Est mean: {y_pred_mean:.2f}, StD: {y_pred_std:.2f}")
    print(f"MAPE mean: {MAPE_mean:.2f}, StD: {MAPE_std:.2f}")
    print(f"RMSE mean: {RMSE_mean:.2f}, StD: {RMSE_std:.2f}")

    return model, y_pred_mean, y_pred_std, MAPE_mean, MAPE_std, RMSE_mean, RMSE_std


def run_brake(participant):

    X, mag, imp = get_brake(participant)

    print("Metric: Peak Brake Magnitude")
    print("-----------------")
    # selected =  LOOF(X, mag, 'metrics')
    selected, feature_list = LOFO(X, mag)
    X_selected = X.iloc[:, selected]
    mag_model, mag_mean, mag_std, mape_mean, mape_std, rmse_mean, rmse_std = (
        train_metrics(X_selected, mag)
    )
    save_metrics(
        "Peak Braking Magnitude",
        participant.name,
        feature_list,
        mag,
        mag_mean,
        mag_std,
        mape_mean,
        mape_std,
        rmse_mean,
        rmse_std,
    )
    print("\n")

    print("Metric: Brake Impulse")
    print("-----------------")
    # selected =  LOOF(X, imp, 'metrics')
    selected, feature_list = LOFO(X, imp)
    X_selected = X.iloc[:, selected]
    imp_model, imp_mean, imp_std, mape_mean, mape_std, rmse_mean, rmse_std = (
        train_metrics(X_selected, imp)
    )
    save_metrics(
        "Braking Impulse",
        participant.name,
        feature_list,
        imp,
        imp_mean,
        imp_std,
        mape_mean,
        mape_std,
        rmse_mean,
        rmse_std,
    )
    print("\n")

    means = [mag_mean, imp_mean]
    stds = [mag_std, imp_std]

    mag_fn = f"models/{participant.name}_brakemag"
    pickle.dump(mag_model, open(mag_fn, "wb"))
    # participant.model_brakemag = mag_model

    imp_fn = f"models/{participant.name}_brakeimp"
    pickle.dump(imp_model, open(imp_fn, "wb"))
    # participant.model_brakeimp = imp_model

    return means, stds


def run_prop(participant):

    X, mag, imp = get_prop(participant)

    print("Metric: Peak Propulsion Magnitude")
    print("-----------------")
    # selected =  LOOF(X, mag, 'metrics')
    selected, feature_list = LOFO(X, mag)
    X_selected = X.iloc[:, selected]
    mag_model, mag_mean, mag_std, mape_mean, mape_std, rmse_mean, rmse_std = (
        train_metrics(X_selected, mag)
    )
    save_metrics(
        "Peak Propulsion Magnitude",
        participant.name,
        feature_list,
        mag,
        mag_mean,
        mag_std,
        mape_mean,
        mape_std,
        rmse_mean,
        rmse_std,
    )
    print("\n")

    print("Metric: Propulsion Impulse")
    print("-----------------")
    # selected =  LOOF(X, imp, 'metrics')
    selected, feature_list = LOFO(X, imp)
    X_selected = X.iloc[:, selected]
    imp_model, imp_mean, imp_std, mape_mean, mape_std, rmse_mean, rmse_std = (
        train_metrics(X_selected, imp)
    )
    save_metrics(
        "Propulsion Impulse",
        participant.name,
        feature_list,
        imp,
        imp_mean,
        imp_std,
        mape_mean,
        mape_std,
        rmse_mean,
        rmse_std,
    )
    print("\n")

    means = [mag_mean, imp_mean]
    stds = [mag_std, imp_std]

    mag_fn = f"models/{participant.name}_propmag"
    pickle.dump(mag_model, open(mag_fn, "wb"))
    # participant.model_propmag = mag_model

    imp_fn = f"models/{participant.name}_propimp"
    pickle.dump(imp_model, open(imp_fn, "wb"))
    # participant.model_propimp = imp_model

    return means, stds


def save_metrics(
    metric_name,
    participant_name,
    feature_list,
    y,
    y_pred_mean,
    y_pred_std,
    mape_mean,
    mape_std,
    rmse_mean,
    rmse_std,
):
    y_mean = np.mean(y)
    y_std = np.std(y)

    metrics = pd.DataFrame([y_mean, y_std, y_pred_mean, y_pred_std]).T
    errors = pd.DataFrame([mape_mean, mape_std, rmse_mean, rmse_std]).T
    df = pd.concat([metrics, errors])

    file_path = f"{participant_name}_metrics.txt"
    with open(file_path, "a") as f:
        f.write(f"{metric_name}\n")
        f.write("---------------------\n")
        f.write(f"{list(feature_list)}\n")
        f.write(f"Measured: {y_mean:.2f}, StD: {y_std:.2f}\n")
        f.write(f"Estimated: {y_pred_mean:.2f}, StD: {y_pred_std:.2f}\n")
        f.write(f"MAPE mean: {mape_mean:.2f}, StD: {mape_std:.2f}\n")
        f.write(f"RMSE mean: {rmse_mean:.2f}, StD: {rmse_std:.2f}\n")
        f.write("\n")
        # df.to_csv(file_path, header=None, index=None, sep=",", mode="a")


# def save_metrics(participant):
#     X, mag, imp = get_brake(participant)
#     model = linear_model.LinearRegression()
#     cv = LeaveOneOut()
#     y_avg = np.mean(mag)
#     y_std = np.std(mag)
#     y_pred = cross_val_predict(model, X, mag, cv=cv)
#     y_pred_avg = np.mean(y_pred)
#     y_pred_std = np.std(y_pred)
#     df = pd.DataFrame([y_avg, y_std, y_pred_avg, y_pred_std]).T
#     df.to_csv(
#         f"{participant.name}_metrics.txt", header=None, index=None, sep=",", mode="w"
#     )
#
#     y_avg = np.mean(imp)
#     y_std = np.std(imp)
#     y_pred = cross_val_predict(model, X, imp, cv=cv)
#     y_pred_avg = np.mean(y_pred)
#     y_pred_std = np.std(y_pred)
#     df = pd.DataFrame([y_avg, y_std, y_pred_avg, y_pred_std]).T
#     df.to_csv(
#         f"{participant.name}_metrics.txt", header=None, index=None, sep=",", mode="a"
#     )
#
#     X, mag, imp = get_prop(participant)
#     model = linear_model.LinearRegression()
#     cv = LeaveOneOut()
#     y_avg = np.mean(mag)
#     y_std = np.std(mag)
#     y_pred = cross_val_predict(model, X, mag, cv=cv)
#     y_pred_avg = np.mean(y_pred)
#     y_pred_std = np.std(y_pred)
#     df = pd.DataFrame([y_avg, y_std, y_pred_avg, y_pred_std]).T
#     df.to_csv(
#         f"{participant.name}_metrics.txt", header=None, index=None, sep=",", mode="a"
#     )
#
#     y_avg = np.mean(imp)
#     y_std = np.std(imp)
#     y_pred = cross_val_predict(model, X, imp, cv=cv)
#     y_pred_avg = np.mean(y_pred)
#     y_pred_std = np.std(y_pred)
#     df = pd.DataFrame([y_avg, y_std, y_pred_avg, y_pred_std]).T
#     df.to_csv(
#         f"{participant.name}_metrics.txt", header=None, index=None, sep=",", mode="a"
#     )


def plot_metrics(participant):
    print(f"{participant.name}")
    brake_mag, brake_imp = get_brake(participant)[1:]
    prop_mag, prop_imp = get_prop(participant)[1:]
    # brake_mag = np.mean(brake_mag)
    # brake_imp = np.mean(brake_imp)
    # prop_mag = np.mean(prop_mag)
    # prop_imp = np.mean(prop_imp)
    # brake_means, brake_stds = run_brake(participant)
    # prop_means, prop_stds = run_prop(participant)

    brake_mag_mean = np.mean(brake_mag)
    brake_mag_std = np.std(brake_mag)
    brake_imp_mean = np.mean(brake_imp)
    brake_imp_std = np.std(brake_imp)
    prop_mag_mean = np.mean(prop_mag)
    prop_mag_std = np.std(prop_mag)
    prop_imp_mean = np.mean(prop_imp)
    prop_imp_std = np.std(prop_imp)
    brake_est_means, brake_est_stds = run_brake(participant)
    prop_est_means, prop_est_stds = run_prop(participant)

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
    ax.set_title(f"{participant.name} Metrics")
    ax.set_xticks(x + (width / 2), metrics)
    ax.legend(loc="upper center", ncols=2)

    # plt.show()

    filename = f"figures/{participant.name}_metrics.pdf"
    plt.savefig(filename, bbox_inches="tight")

    return fig


# def run_metrics(participant):

#     brake_mag, brake_imp = get_brake(participant)[1:]
#     prop_mag, prop_imp = get_prop(participant)[1:]

#     brake_means, brake_stds = run_brake(participant)
#     prop_means, prop_stds = run_prop(participant)


def get_APGRF(participant):

    y = np.array(participant.apgrf)
    X = np.array(
        [
            participant.thigh_accX,
            participant.thigh_accY,
            participant.thigh_accZ,
            participant.thigh_gyroX,
            participant.thigh_gyroY,
            participant.thigh_gyroZ,
            participant.shank_accX,
            participant.shank_accY,
            participant.shank_accZ,
            participant.shank_gyroX,
            participant.shank_gyroY,
            participant.shank_gyroZ,
            participant.foot_accX,
            participant.foot_accY,
            participant.foot_accZ,
            participant.foot_gyroX,
            participant.foot_gyroY,
            participant.foot_gyroZ,
        ]
    )

    strides = np.shape(y)[1]
    metrics = np.shape(X)[0]
    X_hold = np.empty([strides, 100, metrics]) * 0
    for i in range(strides):

        X_hold[i, :, :] = X[:, :, i].T

    return X_hold, y


def features_apgrf(X, y, feature_count):
    labels = [
        "Thigh Acc X",
        "Thigh Acc Y",
        "Thigh Acc Z",
        "Thigh Gyro X",
        "Thigh Gyro Y",
        "Thigh Gyro Z",
        "Shank Acc X",
        "Shank Acc Y",
        "Shank Acc Z",
        "Shank Gyro X",
        "Shank Gyro Y",
        "Shank Gyro Z",
        "Foot Acc X",
        "Foot Acc Y",
        "Foot Acc Z",
        "Foot Gyro X",
        "Foot Gyro Y",
        "Foot Gyro Z",
    ]

    rank = LOOF(X, y, data="apgrf")
    print("-----------")
    print("Features selected: ")

    for n in range(feature_count):
        print(labels[rank[n]])

    return rank[0:feature_count]


def split_apgrf(X, y, train_size):

    # train_size = 0.75
    cut = np.int32(np.round(np.shape(y)[1] * train_size, decimals=0))
    cut / np.shape(y)[1]
    train = cut / np.shape(y)[1]

    X_hold = np.reshape(X, (-1, X.shape[2]), order="F")
    y_hold = y.flatten(order="F")

    X_train, X_test, y_train, y_test = train_test_split(
        X_hold, y_hold, train_size=train, random_state=305
    )

    X_train = np.reshape(X_train, (-1, 100, 18))
    # X_test = np.reshape(X_test, (-1, 100, 18))
    y_train = np.reshape(y_train, (100, -1))
    # y_test = np.reshape(y_test, (100, -1))

    # cut = np.int32(np.round(np.shape(y)[1] * train_size, decimals=0))
    # X_train = X[0:cut,:,:]
    # X_test = np.reshape(X[cut:,:,:], (-1, X.shape[2]), order='F')
    # y_train = y[:,0:cut]
    # y_test = y[:,cut:].flatten(order='F')

    return X_train, X_test, y_train, y_test


def train_apgrf(X, y, metrics):

    strides = y.shape[1]

    reg = linear_model.SGDRegressor()

    for i in range(strides - 1):

        X_hold = X[i, :, metrics].T
        y_hold = y[:, i]
        reg.partial_fit(X_hold, y_hold)

    y_pred = reg.predict(X[i + 1, :, metrics].T)
    y_test = y[:, i + 1]
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    return reg


def plot_apgrf(y_test, y_pred):

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(y_test)), y_test, color=color_pal[0], label="Measured")
    ax.plot(range(len(y_pred)), y_pred, color=color_pal[1], label="Estimated")
    ax.legend(loc="upper center", ncols=2)
    ax.set_ylabel("%BW")
    ax.set_xlabel("% Stance")

    return fig, ax


def save_apgrf(participant, y_test, y_pred):
    y_test_hold = np.reshape(y_test, (100, -1))
    y_test_avg = np.mean(y_test_hold, axis=1)
    y_pred_hold = np.reshape(y_pred, (100, -1))
    y_pred_avg = np.mean(y_pred_hold, axis=1)
    df = pd.DataFrame([y_test_avg, y_pred_avg]).T
    df.to_csv(f"{participant.name}.txt", header=None, index=None, sep=",", mode="w")


def run_apgrf(participant, features):

    X, y = get_APGRF(participant)
    metrics = features_apgrf(X, y, features)

    X_train, X_test, y_train, y_test = split_apgrf(X, y, 0.7)

    model = train_apgrf(X_train, y_train, metrics)

    y_pred = model.predict(X_test)

    save_apgrf(participant, y_test, y_pred)

    fig, ax = plot_apgrf(y_test, y_pred)
    ax.set_title(f"{participant.name} APGRF")
    filename = f"figures/{participant.name}_apgrf.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
