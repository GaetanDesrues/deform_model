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
    cb = MyCallback(tf.fenv(__file__, "SAVED_DATA") / "anat_cmaes")

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

    main()
    # start_one_task(ModelParams())
