import logging

import matplotlib.pyplot as plt
import numpy as np
import treefiles as tf
from sklearn.decomposition import PCA

from anat_db.anat_real import get_pa_fe, Manager


def main():
    pa, fe = get_pa_fe()
    g = Manager(pa, fe, Manager.MINMAXSCALER)
    _, _, _, _, scaler = g.split()
    x = scaler.transform(fe)

    n_comp = 5
    pca_x = PCA(n_components=n_comp)
    pca_x.fit_transform(x)
    print(pca_x.explained_variance_ratio_.cumsum())
    # [0.56417935 0.84187719 0.98237518 0.99799223 1.        ]

    with tf.SPlot(fname="explained_variance.png"):
        fig, ax = plt.subplots(figsize=(3, 3))
        tf.despine(fig)
        ax.plot(
            np.arange(1, n_comp + 1), np.cumsum(pca_x.explained_variance_ratio_ * 100)
        )
        ax.axhline(98.5, xmax=0.55, ls="--", color="red")
        ax.axvline(3.2, ymax=0.91, ls="--", color="red")
        ax.text(1.25, 100, "98%", color="red")
        ax.set_ylabel("explained variance [%]")
        ax.set_xlabel("components")
        fig.tight_layout()


log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log = tf.get_logger()

    main()
