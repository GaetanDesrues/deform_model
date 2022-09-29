import logging
import os
import pickle

import numpy as np
import treefiles as tf
from BasePackage.Pipeline import ModelParams, ElecModel
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error

from wrap_dask.dask_moo import MyCallback, DaskRunner
from wrap_dask.dask_wrap import Options, get_client, dask_task
import matplotlib.pyplot as plt

target = {
    "vol_lv_blood": (130, 5),
    "vol_rv_blood": (80, 1),
    "dist_apico_basal": (85, 1),
    "dist_diameter": (50, 1),
    "dist_septum_thickness": (12, 1),
}


class MecaProblem(ElementwiseProblem):
    def __init__(
        self,
        opt_params: tf.Params,
        default_model_params: ModelParams,
        **kwargs,
    ):
        self.params = opt_params
        xl = np.array([x.bounds[0] for x in opt_params.values()])
        xu = np.array([x.bounds[1] for x in opt_params.values()])
        super().__init__(n_var=len(self.params), n_obj=1, xl=xl, xu=xu, **kwargs)

        self.default_model_params = default_model_params

    def _evaluate(self, x, out, *args, **kwargs):
        log.info(f"evaluate: {x}")

        try:
            params: ModelParams = self.default_model_params.copy()
            for k, v in zip(self.params, x):
                params[k](v)
            out["F"] = start_one_task(params)
        except Exception as e:
            print(f"In '_evaluate': got error {e}")
            out["F"] = np.nan


@dask_task
def start_one_task(opt_params: ModelParams) -> float:
    os.environ["CARDIAC_CLUSTER"] = "1"

    with tf.TmpDir() as o:
        opt_params.out_dir(o / tf.get_string())
        opt_params.anat_db_path(tf.env("SIMUS") / "db_anat_db")
        model = ElecModel["deform"](opt_params)
        model.start()
        fts_pred = tf.load_json(model.data.deform_infos)["features"]
        fts_pred = {x["name"]: x["value"] for x in fts_pred.values()}

    fn = "/user/gdesrues/home/Documents/dev/JaGaMeca/anat_db/out_pca/models_MinMaxScaler.pkl"
    scaler = pickle.load(open(fn, "rb"))["ridge"]["scaler"]

    x = np.array([v[0] for v in target.values()])
    x = scaler.transform([x])[0]
    y = np.array([fts_pred[k] for k in target.keys()])
    y = scaler.transform([y])[0]

    err = mean_squared_error(x, y, sample_weight=[v[1] for v in target.values()])
    return err


def warm_start():
    fn = "/user/gdesrues/home/Documents/dev/JaGaMeca/anat_db/out_pca/models_MinMaxScaler.pkl"
    algo = tf.munchify(pickle.load(open(fn, "rb"))["nn"])

    x = algo.scaler.transform([[v[0] for v in target.values()]])
    if algo.scaler.__class__.__name__ != "FunctionTransformer":
        x = algo.pca.transform(x)
    dd = algo.estimator.predict(x)[0]
    return {
        "coef_dilate": dd[0],
        "coef_scale_r": dd[1],
        "coef_scale_z": dd[2],
    }


def main():
    opt = Options(
        popsize=30,
        maxiter=50,
        local_n_workers=15,
        local_threads_per_worker=1,
    )
    client = get_client(opt)

    s_ = lambda name, val, bs: tf.Param(name, bounds=bs, initial_value=val)
    opt_params = tf.Params(
        s_("coef_dilate", 0, (-10, 10)),
        s_("coef_scale_r", 1, (0.8, 1.2)),
        s_("coef_scale_z", 1, (0.8, 1.2)),
        s_("coef_overlap", 2, (0.5, 4)),
        s_("coef_sept", 0.5, (0.05, 2)),
    )

    default_model_params = ModelParams()

    # Start pymoo
    problem = MecaProblem(
        opt_params=opt_params,
        default_model_params=default_model_params,
        elementwise_runner=DaskRunner(client),
    )

    warm_start_pred = warm_start()
    # {'coef_dilate': -1.5697714374609908, 'coef_scale_r': 1.01619212813613, 'coef_scale_z': 1.0623496811084094}
    # warm_start_pred = {}  # NO warm start
    x0 = [
        warm_start_pred[x.name] if x.name in warm_start_pred else x.initial_value
        for x in opt_params.values()
    ]
    print(f"{x0=}")
    algo = CMAES(
        x0=np.array(x0),
        sigma=0.2,
        popsize=opt.popsize,
        maxiter=opt.maxiter,
        CMA_elitist=True,
        parallelize=True,
    )
    cb = MyCallback(tf.fenv(__file__, "SAVED_DATA") / "anat_cmaes_no_WS")

    res = minimize(
        problem,
        algo,
        termination=("n_gen", opt.maxiter),
        verbose=True,
        save_history=True,
        callback=cb,
        seed=42,
    )

    log.info(f"res.exec_time: {res.exec_time}")
    log.info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    client.close()


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = tf.get_logger()

    # main()
    # start_one_task(ModelParams())#debug

    # # plot_cmaes
    # out = tf.fenv(__file__, "SAVED_DATA") / "anat_cmaes_no_WS"
    # res = [tf.load_json(x, munch=True) for x in out.glob("cmaes_data_iter_*.json")]
    # n_eval = [r.n_eval for r in res]
    #
    # out_f = tf.dump(out / "plots")
    # ll = ["coef_dilate", "coef_scale_r", "coef_scale_z", "coef_overlap", "coef_sept"]
    #
    # with tf.SPlot(out_f / "error.png"):
    #     fig, ax = plt.subplots()
    #     tf.despine(fig)
    #
    #     s = np.array([r.F for r in res[1:]])
    #     ws = np.repeat(res[0].F, s.shape[1])
    #     s = np.vstack((ws, s))
    #     ax.fill_between(
    #         n_eval,
    #         np.quantile(s, 0.25, axis=1),
    #         np.quantile(s, 0.75, axis=1),
    #         label="25% - 75%",
    #         alpha=0.4,
    #         color=tf.EDSTIC,
    #     )
    #     ax.plot(
    #         n_eval, np.array([np.mean(r.F) for r in res]), label="mean", color="red"
    #     )
    #     ax.plot(
    #         n_eval, np.array([np.min(r.F) for r in res]), label="min", color="k", lw=2
    #     )
    #     ax.legend()
    #     ax.set_xlabel("function evaluations")
    #     ax.set_ylabel("WMSE")
    #     fig.tight_layout()
    #
    # for jj, kk in enumerate(ll):
    #     with tf.SPlot(fname=out_f / f"img_{kk}.png"):
    #         fig, ax = plt.subplots()
    #         tf.despine(fig)
    #         xx_ = []
    #         x0 = np.array(res[0].X)
    #         for r in res:
    #             xx = np.array(r.X)
    #             if xx.ndim == 1:
    #                 xx = xx[np.newaxis, :]
    #             xx_.append(xx[:, jj])
    #         ax.plot(
    #             n_eval, np.array([np.mean(r) for r in xx_]), label="mean", color="red"
    #         )
    #         ax.plot(
    #             n_eval, np.array([np.min(r) for r in xx_]), label="min", color=tf.EDSTIC
    #         )
    #         ax.plot(
    #             n_eval, np.array([np.max(r) for r in xx_]), label="max", color=tf.EDSTIC
    #         )
    #
    #         s = np.array([np.array(r.X)[:, jj] for r in res[1:]])
    #         ws = np.repeat(res[0].X[jj], s.shape[1])
    #         # print(ws.shape)
    #         # breakpoint()
    #         s = np.vstack((ws, s))
    #         ax.fill_between(
    #             n_eval,
    #             np.quantile(s, 0.25, axis=1),
    #             np.quantile(s, 0.75, axis=1),
    #             label="25% - 75%",
    #             alpha=0.4,
    #             color=tf.EDSTIC,
    #         )
    #         ax.set_title(kk)
    #         ax.axhline(x0[jj], ls="--", color="k")
    #         ax.legend()
    #         fig.tight_layout()
    #
    # s = np.array([r.F for r in res[1:]])
    # ws = np.repeat(res[0].F, s.shape[1])
    # s = np.vstack((ws, s))
    # # print(s.shape)  # nb_iter x pop_size
    #
    # miniter_i = np.argmin(np.min(s, axis=1))
    # minpop_i = np.argmin(s[miniter_i])
    #
    # all_min = s[miniter_i, minpop_i]
    # print(f"min F: {all_min}, iter: {miniter_i}, sample: {minpop_i}")
    # x = np.array(res[miniter_i].X[minpop_i])
    # pred = {k: v for k, v in zip(ll, x)}
    # pred = tf.Params.from_dict(pred)
    # print(pred.table())
    # # ML warm start
    # # min F: 0.001913892048487686, iter: 27, sample: 24
    # #          Name       Value
    # #  ------------   ---------
    # #   coef_dilate   -0.693101
    # #  coef_scale_r    1.032291
    # #  coef_scale_z    1.022878
    # #  coef_overlap    0.558974
    # #     coef_sept    0.079202
    #
    # # No warm start:
    # # min F: 0.00191595258217751, iter: 27, sample: 5
    # #          Name       Value
    # #  ------------   ---------
    # #   coef_dilate   -0.919133
    # #  coef_scale_r    1.038209
    # #  coef_scale_z    1.025821
    # #  coef_overlap    0.500066
    # #     coef_sept    0.056184
    #
    # opt_params = ModelParams()
    # for x in pred.values():
    #     opt_params[x.name](x.value)
    # with tf.TmpDir() as o:
    #     opt_params.out_dir(o / tf.get_string())
    #     opt_params.anat_db_path(tf.env("SIMUS") / "db_anat_db")
    #     model = ElecModel["deform"](opt_params)
    #     model.start()
    #     fts_pred = tf.load_json(model.data.deform_infos)["features"]
    #     fts_pred = {x["name"]: x["value"] for x in fts_pred.values()}
    #
    # print(fts_pred)
    # print(target)
    # # with warm start:
    # # target = {
    # #     "vol_lv_blood": (130, 126.91705479325385),
    # #     "vol_rv_blood": (80, 75.44692973506247),
    # #     "dist_apico_basal": (85, 86.1475830078125),
    # #     "dist_diameter": (50, 55.42589092254639),
    # #     "dist_septum_thickness": (12, 11.945785823512892),
    # # }
    # # NO warm start:
    # # target = {
    # #     "vol_lv_blood": (130, 123.34685400523473),
    # #     "vol_rv_blood": (80, 76.40676815350535),
    # #     "dist_apico_basal": (85, 86.16767883300781),
    # #     "dist_diameter": (50, 55.46263885498047),
    # #     "dist_septum_thickness": (12, 12.232549795522837),
    # # }

    # Warm start only
    warm_start_pred = warm_start()
    opt_params = ModelParams()
    for k, x in warm_start_pred.items():
        opt_params[k](x)
    with tf.TmpDir() as o:
        opt_params.out_dir(o / tf.get_string())
        opt_params.anat_db_path(tf.env("SIMUS") / "db_anat_db")
        model = ElecModel["deform"](opt_params)
        model.start()
        fts_pred = tf.load_json(model.data.deform_infos)["features"]
        fts_pred = {x["name"]: x["value"] for x in fts_pred.values()}
    print(fts_pred)
    # {'vol_lv_blood': 113.22010677414411, 'vol_rv_blood': 75.91164789534773, 'dist_apico_basal': 88.62071228027344, 'dist_diameter': 53.26496982574463, 'dist_septum_thickness': 12.712777721692323}
