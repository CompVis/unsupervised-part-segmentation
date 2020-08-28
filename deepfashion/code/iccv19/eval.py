import sys, os, pickle
import numpy as np
import math, os
import tensorflow as tf
from edflow.iterators.batches import load_image
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.evaluator import TFBaseEvaluator
from edflow.hooks.hook import Hook
from edflow.iterators.batches import plot_batch
from edflow.project_manager import ProjectManager
from edflow.hooks.evaluation_hooks import RestoreTFModelHook


class MatrixData(DatasetMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.row_csv = config["data_row_csv"]
        self.col_csv = config["data_col_csv"]
        self.block_size = config.get("data_block_size", config["batch_size"])

        self.row_labels = self._make_labels(self.row_csv)
        self.col_labels = self._make_labels(self.col_csv)
        self.n_rows = len(self.row_labels["file_path_"])
        self.n_cols = len(self.col_labels["file_path_"])

        n_per_block = self.block_size * self.block_size
        n_blocks = min(self.n_rows, self.n_cols) // self.block_size
        self._length = n_blocks * n_per_block

    def _make_labels(self, csv):
        with open(csv) as f:
            lines = f.read().splitlines()
        labels = {
            "relative_file_path_": [l for l in lines],
            "file_path_": [os.path.join(self.root, l) for l in lines],
        }
        return labels

    def __len__(self):
        return self._length

    def _matrix_index(self, i):
        mat_idx = i // (self.block_size * self.block_size)

        i = i - mat_idx * (self.block_size * self.block_size)

        block_row_idx = i // self.block_size
        block_col_idx = i % self.block_size

        row_idx = block_row_idx + mat_idx * self.block_size
        col_idx = block_col_idx + mat_idx * self.block_size

        return mat_idx, block_row_idx, block_col_idx, row_idx, col_idx

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        return resize(image, self.size)

    def get_example(self, i):
        matrix, block_row, block_col, row, col = self._matrix_index(i)
        i = row
        j = col

        view0_file_path = self.row_labels["file_path_"][i]
        relative_view0_file_path = self.row_labels["relative_file_path_"][i]
        view0 = self.preprocess_image(view0_file_path)
        view1_file_path = self.col_labels["file_path_"][j]
        relative_view1_file_path = self.col_labels["relative_file_path_"][j]
        view1 = self.preprocess_image(view1_file_path)
        return {
            "view0": view0,
            "view1": view1,
            "file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
            "view1_file_path": view1_file_path,
            "view1_relative_file_path_": relative_view1_file_path,
            "matrix": matrix,
            "matrix_index": (block_row, block_col),
        }


class MatrixHook(Hook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config
        self.generated_key = self.config["generated_key"]
        self.vis0_key = self.config["vis0_key"]
        self.vis1_key = self.config["vis1_key"]

    def before_epoch(self, epoch):
        self.root = os.path.join(
            self._root, "comparison_matrix", "{:06}".format(self.get_global_step())
        )
        os.makedirs(self.root)
        self.data = {
            self.generated_key: list(),
            self.vis0_key: list(),
            self.vis1_key: list(),
            "view0": list(),
            "view1": list(),
            "relative_file_path_": list(),
            "view1_relative_file_path_": list(),
            "matrix": list(),
            "matrix_index": list(),
        }

    def before_step(self, step, fetches, feeds, batch):
        for k in [self.generated_key, self.vis0_key, self.vis1_key]:
            fetches[k] = getattr(self.model, k)
        bs = self.bs = batch["file_path_"].shape[0]
        for k in [
            "relative_file_path_",
            "view1_relative_file_path_",
            "matrix",
            "matrix_index",
            "view0",
            "view1",
        ]:
            for i in range(bs):
                self.data[k].append(batch[k][i])

    def after_step(self, step, results):
        for k in [self.generated_key, self.vis0_key, self.vis1_key]:
            for i in range(self.bs):
                self.data[k].append(results[k][i])

    def after_epoch(self, epoch):
        out_path = os.path.join(self.root, "data.p")
        with open(out_path, "wb") as f:
            pickle.dump(self.data, f)
        print("Wrote retrieval matrix data in:")
        print(out_path)
        pickle_path = out_path
        img_out_path = os.path.join(self.root, "comparison_matrix.png")
        make_matrix(pickle_path, img_out_path)


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()
        root = ProjectManager.latest_eval
        hooks = [MatrixHook(self.get_global_step, root, self.model, self.config)]
        self.hooks += hooks

    def initialize(self, checkpoint_path=None):
        if "triplet_path" in self.config:
            triplet_path = self.config["triplet_path"]
            e1_name = "my_triplet_is_the_best_triplet"
            triplet_variables = [v for v in tf.global_variables() if e1_name in v.name]
            restorer = RestoreTFModelHook(
                variables=triplet_variables, checkpoint_path=None
            )
            with self.session.as_default():
                restorer(triplet_path)
            print("Restored triplet net.")


def make_matrix(
    pickle_path,
    out_path,
    generated_key="_generated",
    row_key="view1",
    col_key="view0",
    rowvis_key="_app_visualize",
    colvis_key="_visualize",
    transpose=True,
):
    with open(pickle_path, "rb") as f:
        all_data = pickle.load(f)
    matrix_indices = sorted(set(all_data["matrix"]))
    for matrix in matrix_indices:
        subindices = [
            i for i in range(len(all_data["matrix"])) if all_data["matrix"][i] == matrix
        ]
        data = dict()
        for k in all_data:
            data[k] = [all_data[k][i] for i in subindices]

        data["matrix_index"] = [tuple(idx) for idx in data["matrix_index"]]
        if transpose:
            data["matrix_index"] = [(j, i) for (i, j) in data["matrix_index"]]
            _row_key = col_key
            col_key = _row_key
            _rowvis_key = colvis_key
            colvis_key = _rowvis_key
        mindices = data["matrix_index"]
        n_rows = max([idx[0] for idx in mindices]) + 1
        n_cols = max([idx[1] for idx in mindices]) + 1

        assert n_rows == n_cols, (n_rows, n_cols)

        import matplotlib
        import matplotlib.pyplot as plt

        ar = n_cols / n_rows
        figheight = 5.0
        figwidth = figheight * ar
        dpi = 600
        base_space = 1.0
        space_factor = 0.1
        linewidth = 1.0
        hspace = wspace = 0.0

        figrows = n_rows + 2
        figcols = n_cols + 2

        hrats = [base_space] * (figrows)
        wrats = [base_space] * (figcols)

        fig = plt.figure(figsize=(figwidth, figheight))
        gs = fig.add_gridspec(
            nrows=figrows,
            ncols=figcols,
            width_ratios=wrats,
            height_ratios=hrats,
            hspace=hspace,
            wspace=wspace,
        )

        def put_image(image, i, j):
            image = (image + 1.0) / 2.0
            image = np.clip(image, 0.0, 1.0)
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

        # synthesis
        for i in range(n_rows):
            for j in range(n_cols):
                image_idx = data["matrix_index"].index((i, j))
                image = data[generated_key][image_idx]
                put_image(image, i + 2, j + 2)

        for i in range(n_rows):
            image_idx = data["matrix_index"].index((i, 0))
            image = data[row_key][image_idx]
            put_image(image, i + 2, 1)
            image = data[rowvis_key][image_idx]
            put_image(image, i + 2, 0)
        for j in range(n_cols):
            image_idx = data["matrix_index"].index((0, j))
            image = data[col_key][image_idx]
            put_image(image, 1, j + 2)
            image_idx = data["matrix_index"].index((0, j))
            image = data[colvis_key][image_idx]
            put_image(image, 0, j + 2)

        dir_, fname = os.path.split(out_path)
        matrix_out_path = os.path.join(dir_, "{:06}_".format(matrix) + fname)
        fig.savefig(matrix_out_path, dpi=dpi)
        print("Wrote")
        print(matrix_out_path)


if __name__ == "__main__":
    pickle_path = sys.argv[1]
    out_path = sys.argv[2]
    make_matrix(pickle_path, out_path)
