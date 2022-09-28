import logging
import os
import pickle
from functools import cache

import numpy as np
import pandas as pd
import treefiles as tf
from BasePackage.Pipeline import ModelParams, ElecModel
from HeartDB.analyse.sobol.test_sensivity import SensitivityAnalysis
from matplotlib import pyplot as plt
from matplotlib.pyplot import setp
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from anat_db.varying_params import extract_df
from sklearn.preprocessing import FunctionTransformer


def lambda_x(x):
    return x


NoScaler = FunctionTransformer(lambda_x)


def main():
    pa, fe = get_pa_fe()

    for s in Manager.SCALERS:
        g = Manager(pa, fe, s)
        # g.plot_sa()
        # g.plot_values()

        # g.train_models()  # /!\ overriding and time consuming
        # g.print_results_ml()
        g.plot_ml_test_resutls()

        # g.test_elec_model()  # /!\ overriding and time consuming


def get_pa_fe():
    os.environ["OUT_PATH"] = tf.env("SIMUS") / "AnatDB_final"
    pa, fe = extract_df()
    pa.drop(["coef_overlap", "coef_sept"], axis=1, inplace=True)
    # df = pd.concat([pa, fe], axis=1)
    # print(df.shape)
    # print(df.head())

    # print(fe.index[np.where(fe['vol_lv_blood']<)])
    # breakpoint()

    return pa, fe


class Manager:
    NOSCALER = "NoScaler"
    MINMAXSCALER = "MinMaxScaler"
    STANDARDSCALER = "StandardScaler"
    SCALERS = {
        NOSCALER: NoScaler,
        MINMAXSCALER: MinMaxScaler(),
        STANDARDSCALER: StandardScaler(),
    }

    def __init__(self, pa, fe, scaler: str):
        assert scaler in self.SCALERS
        self.scaler = self.SCALERS[scaler]
        self.out = tf.f(__file__, "out_pca").dump()
        tf.logd(self.out)
        q = self.scaler.__class__.__name__
        q = q.replace("FunctionTransformer", "NoScaler")
        self.out.file(models=f"models_{q}.pkl", results=f"results_{q}.pkl")
        self.pa = pa
        self.fe = fe

    def train_models(self):
        methods = {
            "ridge": self.ridge(),
            "kernel_ridge": self.kernel_ridge(),
            "knn": self.knn(),
            "nn": self.nn(),
        }
        pickle.dump(methods, open(self.out.models, "wb"))

    @cache
    def split(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.fe, self.pa, test_size=0.3, random_state=42
        )

        scaler = self.scaler.fit(x_train)
        x_train = pd.DataFrame(scaler.transform(x_train), columns=self.fe.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=self.fe.columns)

        pca = None
        if self.scaler.__class__.__name__ != "FunctionTransformer":
            pca = PCA(n_components=3)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)

        return x_train, x_test, y_train, y_test, scaler, pca

    def ridge(self):
        x_train, x_test, y_train, y_test, scaler, pca = self.split()

        reg = RidgeCV(alphas=np.logspace(-4, 4, 200))
        cv = RepeatedKFold(n_splits=3, n_repeats=1)
        reg.fit(x_train, y_train)
        scores = cross_val_score(
            reg, x_train, y_train, scoring="neg_mean_squared_error", cv=cv
        )
        scores = np.abs(scores)
        pred = pd.DataFrame(reg.predict(x_test), columns=self.pa.columns)
        test_err = mean_squared_error(y_test, pred, multioutput="raw_values")

        return {
            "scaler": scaler,
            "estimator": reg,
            "alpha": reg.alpha_,
            "mse_train": np.mean(scores),
            "mse_train_std": np.std(scores),
            "mse_test": np.mean(test_err),
            "mse_test_std": np.std(test_err),
            "test_samples": (x_test, y_test),
            "r2": r2_score(y_test, pred),
            "pca": pca,
        }

    def kernel_ridge(self):
        x_train, x_test, y_train, y_test, scaler, pca = self.split()

        cv = RepeatedKFold(n_splits=3, n_repeats=1)
        reg = KernelRidge(kernel="rbf", gamma=2.84)  # , cv=cv)
        # reg = GridSearchCV(
        #     reg_,
        #     {"alpha": np.logspace(-4, 4, 200), "kernel": ["linear", "polynomial", "rbf"]},
        #     scoring="neg_mean_squared_error",
        #     cv=cv,
        #     n_jobs=-1,
        # )
        reg.fit(x_train, y_train)
        scores = cross_val_score(
            reg, x_train, y_train, scoring="neg_mean_squared_error", cv=cv
        )
        scores = np.abs(scores)
        pred = pd.DataFrame(reg.predict(x_test), columns=self.pa.columns)
        test_err = mean_squared_error(y_test, pred, multioutput="raw_values")

        return {
            "scaler": scaler,
            "estimator": reg,
            # "alpha": reg.best_estimator_.alpha,
            "mse_train": np.mean(scores),
            "mse_train_std": np.std(scores),
            "mse_test": np.mean(test_err),
            "mse_test_std": np.std(test_err),
            "test_samples": (x_test, y_test),
            "r2": r2_score(y_test, pred),
            "pca": pca,
        }

    def knn(self):
        x_train, x_test, y_train, y_test, scaler, pca = self.split()

        cv = RepeatedKFold(n_splits=3, n_repeats=1)
        reg = KNeighborsRegressor(
            weights="distance", algorithm="ball_tree", n_neighbors=6
        )
        # reg = GridSearchCV(
        #     reg_,
        #     {"n_neighbors": np.linspace(1, 30, 20, dtype=int)},
        #     scoring="neg_mean_squared_error",
        #     cv=cv,
        #     n_jobs=-1,
        # )
        reg.fit(x_train, y_train)
        scores = cross_val_score(
            reg, x_train, y_train, scoring="neg_mean_squared_error", cv=cv
        )
        scores = np.abs(scores)
        pred = pd.DataFrame(reg.predict(x_test), columns=self.pa.columns)
        test_err = mean_squared_error(y_test, pred, multioutput="raw_values")

        return {
            "scaler": scaler,
            "estimator": reg,
            # "n_neighbors": reg.best_estimator_.n_neighbors,
            "mse_train": np.mean(scores),
            "mse_train_std": np.std(scores),
            "mse_test": np.mean(test_err),
            "mse_test_std": np.std(test_err),
            "test_samples": (x_test, y_test),
            "r2": r2_score(y_test, pred),
            "pca": pca,
        }

    def nn(self):
        x_train, x_test, y_train, y_test, scaler, pca = self.split()

        cv = RepeatedKFold(n_splits=3, n_repeats=1)
        reg = MLPRegressor(
            max_iter=1000,
            batch_size=100,
            learning_rate="adaptive",
            hidden_layer_sizes=(50, 100),
            learning_rate_init=0.01,
            activation="tanh",
            # cv=cv,
        )
        # reg = GridSearchCV(
        #     reg_,
        #     {"hidden_layer_sizes": [(100,), (50, 50), (25, 25, 25)]},
        #     scoring="neg_mean_squared_error",
        #     cv=cv,
        #     n_jobs=-1,
        # )
        reg.fit(x_train, y_train)
        scores = cross_val_score(
            reg, x_train, y_train, scoring="neg_mean_squared_error", cv=cv
        )
        scores = np.abs(scores)
        pred = pd.DataFrame(reg.predict(x_test), columns=self.pa.columns)
        test_err = mean_squared_error(y_test, pred, multioutput="raw_values")

        return {
            "scaler": scaler,
            "estimator": reg,
            # "hidden_layer_sizes": reg.best_estimator_.hidden_layer_sizes,
            "mse_train": np.mean(scores),
            "mse_train_std": np.std(scores),
            "mse_test": np.mean(test_err),
            "mse_test_std": np.std(test_err),
            "test_samples": (x_test, y_test),
            "r2": r2_score(y_test, pred),
            "pca": pca,
        }

    def pred_test(self, estimator, x, y):
        pred = pd.DataFrame(estimator.predict([x]), columns=self.pa.columns)
        r = pd.concat((y, pred.T), axis=1)
        r.columns = ["target", "pred"]
        r["E"] = np.square(r["target"] - r["pred"])
        return r

    def print_results_ml(self):
        methods = tf.munchify(pickle.load(open(self.out.models, "rb")))
        _, x, _, y, _, _ = self.split()
        idx = np.random.choice(x.shape[0])

        for m in ("ridge", "kernel_ridge", "knn", "nn"):
            r = self.pred_test(methods[m].estimator, x.iloc[idx], y.iloc[idx])
            print(f"{m:-^31}\n{r}\nMSE={r['E'].mean()}")

    def test_elec_model(self):
        target = pd.Series(
            {
                "vol_lv_blood": 130,
                "vol_rv_blood": 80,
                "dist_apico_basal": 85,
                "dist_diameter": 50,
                "dist_septum_thickness": 12,
            },
            name="target",
        )
        methods = tf.munchify(pickle.load(open(self.out.models, "rb")))

        results = {}
        for k, v in methods.items():
            x = v.scaler.transform([target])
            if v.scaler.__class__.__name__ != "FunctionTransformer":
                x = v.pca.transform(x)
            pred = pd.DataFrame(v.estimator.predict(x), columns=self.pa.columns)
            try:
                ft_pred = pd.Series(predict_one(pred.iloc[0].to_dict()), name="pred")
            except SimulationCrashed:
                print("SimulationCrashed")
            else:
                r = pd.concat((target, ft_pred), axis=1)
                r["E"] = np.square(r["target"] - r["pred"])
                print(r)
                results[k] = r
        pickle.dump(results, open(self.out.results, "wb"))

    def load_result(self):
        return pickle.load(open(self.out.results, "rb"))

    def plot_sa(self):
        sa = SensitivityAnalysis(self.pa, self.fe)
        print(sa.table)
        with tf.SPlot(self.out / "prcc_anat.png"):
            fig = sa.plot_table()
            fig.tight_layout()

    def plot_sa_fe(self):
        def calculate_pvalues(df):
            df = df.dropna()._get_numeric_data()
            dfcols = pd.DataFrame(columns=df.columns)
            pvalues = dfcols.transpose().join(dfcols, how="outer")
            for r in df.columns:
                for c in df.columns:
                    pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
            return pvalues

        pvalues = calculate_pvalues(self.fe)
        print(pvalues)
        print(self.fe.corr())
        # sa = SensitivityAnalysis(self.fe, self.fe)
        # print(sa.table)
        # with tf.SPlot(self.out / "prcc_anat_fe.png"):
        #     fig = sa.plot_table()
        #     fig.tight_layout()

    def plot_values(self):
        with tf.SPlot(fname=self.out / "plots_anat.png"):
            fig, axs = plt.subplots(
                figsize=(6, 8), nrows=5, ncols=3, sharex="col", sharey="row"
            )
            tf.despine(fig)
            for i, pai in enumerate(self.pa.columns):
                for j, fej in enumerate(self.fe.columns):
                    axs[j, i].scatter(self.pa[pai], self.fe[fej])
                    axs[j, 0].set_ylabel(fej)
            for j, pai in enumerate(self.pa.columns):
                axs[-1, j].set_xlabel(pai)
            fig.tight_layout()

    def plot_ml_test_resutls(self):
        methods = tf.munchify(pickle.load(open(self.out.models, "rb")))
        _, x, _, y, _, _ = self.split()
        plot_data = {}

        for m in ("ridge", "kernel_ridge", "knn", "nn"):
            print(f"{m:-^31}")

            print(
                f"MSE train: {methods[m].mse_train:.4f} ({methods[m].mse_train_std:.4f})"
            )
            print(
                f"MSE test: {methods[m].mse_test:.4f} ({methods[m].mse_test_std:.4f})"
            )
            print("r2:", methods[m].r2)

            x, y = methods[m].test_samples
            mean, std = self.pred_test_multiple(methods[m].estimator, x, y)
            print(f"recomputed MSE test: {mean:.4f} ({std:.4f})")

            plot_data[m] = self.pred_test_multiple_(methods[m].estimator, x, y)

        # with tf.SPlot(fname=self.out / "plots_anat_test_errors.png"):
        #     fig, ax = plt.subplots()
        #     tf.despine(fig)
        #     for k, v in plot_data.items():
        #         ax.boxplot(v)
        #         print(k)
        #     fig.tight_layout()
        return plot_data

    def pred_test_multiple(self, estimator, x, y):
        test_err = self.pred_test_multiple_(estimator, x, y)
        # test_err=np.mean(test_err, axis=0)
        return np.mean(test_err), np.std(test_err)

    def pred_test_multiple_(self, estimator, x, y):
        pred = pd.DataFrame(estimator.predict(x), columns=self.pa.columns)
        # print(np.mean(, axis=0))
        # print(mean_squared_error(y, pred, multioutput="raw_values"))
        # breakpoint()
        # test_err = mean_squared_error(y, pred, multioutput="raw_values")
        return np.square(y.values - pred.values)


def predict_one(deform_params_predict: dict):
    with tf.TmpDir() as o:
        params = ModelParams()
        params.out_dir(o)
        for k, v in deform_params_predict.items():
            params[k](v)
        params.anat_db_path(tf.env("SIMUS") / "db_anat_db")
        model = ElecModel["deform"](params)
        model.start()
        fts_ = tf.load_json(model.data.deform_infos)
        if "features" not in fts_:
            print(f"Raising SimulationCrashed because 'features' not in {fts_}")
            raise SimulationCrashed
        fts = {
            # "params": {k: v["value"] for k, v in fts_["params"].items()},
            "features": {k: v["value"] for k, v in fts_["features"].items()},
        }
        return fts["features"]


class SimulationCrashed(Exception):
    pass


def setBoxColors(bp):
    setp(bp["boxes"], color="blue")


def boxplot(data, cols):
    with tf.SPlot():
        fig, axs = plt.subplots(ncols=3)
        axs = axs.ravel()
        tf.despine(fig)

        # all_data = [
        #     np.vstack([np.random.normal(0, std, size=100) for _ in range(3)]).T
        #     for std in range(1, 4)
        # ]
        all_data = np.array(list(data.values()))
        # print(all_data.shape)
        labels = data.keys()

        bps = []
        for k, ax in enumerate(axs):
            bplot2 = ax.boxplot(
                all_data[..., k].T,
                notch=True,
                vert=True,
                patch_artist=True,
                # labels=["", cols[0], ""],  # labels,
                labels=labels,
                # positions=np.linspace(0.8, 1.2, 3),
            )
            bps.append(bplot2)
            # ax.set_xlabel(cols[0])
            ax.set_title(cols[k])

        # fill with colors
        colors = ["pink", "lightblue", "lightgreen"]
        for bplot in bps:
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)

        # adding horizontal grid lines
        for ax in axs:
            ax.yaxis.grid(True)
            # ax.set_xlabel("Three separate samples")
            # ax.set_ylabel("Observed values")

        fig.tight_layout()

    #     fig.suptitle(name)
    #     print(data)
    #     for i, (k, v) in enumerate(data.items()):
    #         for j in range(v.shape[1]):
    #             bp = ax.boxplot(v[:, j], positions=[j + 0.2 * i], widths=0.1)
    #             setBoxColors(bp)
    #         ax.legend(list(cols))
    #         ax.text(i, -0.03, k)
    #     ax.set_ylim(0, 0.25)
    #     fig.tight_layout()


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    main()

    # for scaler in Manager.SCALERS.keys():
    #     print(f"{scaler:-^30}")
    #     g = Manager(*get_pa_fe(), scaler)
    #     g.print_results_ml()
    #     plot_data = g.plot_ml_test_resutls()
    #     print(plot_data)
    #     break

    # pa, fe = get_pa_fe()
    # g = Manager(pa, fe, Manager.MINMAXSCALER)
    # plot_data = g.plot_ml_test_resutls()
    # boxplot(plot_data, Manager.NOSCALER, pa.columns)

    # for scaler in Manager.SCALERS.keys():
    #     print(f"{scaler:-^30}")
    #     g = Manager(pa, fe, scaler)
    #     results = g.load_result()
    #     print(results)
