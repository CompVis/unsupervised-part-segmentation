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


class TableData(DatasetMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.col_csv = config["data_col_csv"]
        self.block_size = config.get("data_block_size", config["batch_size"])

        self.col_labels = self._make_labels(self.col_csv)
        self.n_cols = len(self.col_labels["file_path_"])
        self.n_rows = 1

        n_per_block = self.block_size
        n_blocks = self.n_cols // self.block_size
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
        table_idx = i // self.block_size

        i = i - table_idx * self.block_size

        block_row_idx = i // self.block_size
        block_col_idx = i % self.block_size

        row_idx = block_row_idx + table_idx * self.block_size
        col_idx = block_col_idx + table_idx * self.block_size

        return table_idx, block_col_idx, col_idx

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        return resize(image, self.size)

    def get_example(self, i):
        table, block_col, col = self._matrix_index(i)

        view0_file_path = self.col_labels["file_path_"][i]
        relative_view0_file_path = self.col_labels["relative_file_path_"][i]
        view0 = self.preprocess_image(view0_file_path)
        # view1_file_path = self.col_labels["file_path_"][j]
        # relative_view1_file_path = self.col_labels["relative_file_path_"][j]
        # view1 = self.preprocess_image(view1_file_path)
        return {
            "view0": view0,
            "view1": view0,
            "file_path_": view0_file_path,
            "view1_file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
            "view1_relative_file_path_": relative_view0_file_path,
            "table": table,
            "table_index": block_col,
        }


class TableHook(Hook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config
        self.header_key = self.config["header_key"]
        self.value_keys = self.config["value_keys"]

    def before_epoch(self, epoch):
        self.root = os.path.join(
            self._root, "table", "{:06}".format(self.get_global_step())
        )
        os.makedirs(self.root)
        self.data = {
            self.header_key: list(),
            "view0": list(),
            "relative_file_path_": list(),
            "table": list(),
            "table_index": list(),
        }
        self.data.update({k: list() for k in self.value_keys})

    def before_step(self, step, fetches, feeds, batch):
        for k in self.value_keys:
            fetches[k] = self.model.outputs[k]
        bs = self.bs = batch["file_path_"].shape[0]
        for k in ["relative_file_path_", "table", "table_index", "view0"]:
            for i in range(bs):
                self.data[k].append(batch[k][i])

    def after_step(self, step, results):
        for k in self.value_keys:
            for i in range(self.bs):
                self.data[k].append(results[k][i])

    def after_epoch(self, epoch):
        out_path = os.path.join(self.root, "data.p")
        with open(out_path, "wb") as f:
            pickle.dump(self.data, f)
        print("Wrote retrieval table data in:")
        print(out_path)
        pickle_path = out_path
        img_out_path = os.path.join(self.root, "table.png")
        make_table(pickle_path, img_out_path, self.header_key, self.value_keys)


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


class TableEvaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()
        root = ProjectManager.latest_eval
        hooks = [TableHook(self.get_global_step, root, self.model, self.config)]
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


def make_table(
    pickle_path, out_path, header_key="_generated", value_keys=["view1"], transpose=True
):
    with open(pickle_path, "rb") as f:
        all_data = pickle.load(f)
    table_indices = sorted(set(all_data["table"]))
    for table in table_indices:
        subindices = [
            i for i in range(len(all_data["table"])) if all_data["table"][i] == table
        ]
        data = dict()
        for k in all_data:
            data[k] = [all_data[k][i] for i in subindices]

        data["table_index"] = [idx for idx in data["table_index"]]
        mindices = data["table_index"]
        n_cols = len(mindices)
        n_rows = len(value_keys) + 1

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

        figrows = n_rows
        figcols = n_cols

        hrats = [base_space] * (figrows)
        wrats = [base_space] * (figcols)
        plt.close("all")
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
        for i_col in range(n_cols):
            image_idx = data["table_index"].index(i_col)
            image = data[header_key][image_idx]
            put_image(image, 0, i_col)

        for i_row, value_key in enumerate(value_keys):
            for i_col in range(n_cols):
                image_idx = data["table_index"].index(i_col)
                image = data[value_key][image_idx]
                put_image(image, i_row + 1, i_col)

        dir_, fname = os.path.split(out_path)
        matrix_out_path = os.path.join(dir_, "{:06}_".format(table) + fname)
        fig.tight_layout()
        fig.savefig(matrix_out_path, dpi=dpi)
        print("Wrote")
        print(matrix_out_path)


if __name__ == "__main__":
    pickle_path = sys.argv[1]
    out_path = sys.argv[2]
    make_matrix(pickle_path, out_path)
