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


class InferData(DatasetMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.row_csv = config["data_row_csv"]

        self.row_labels = self._make_labels(self.row_csv)
        self.n_rows = len(self.row_labels["file_path_"])
        self._length = self.n_rows

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

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        return resize(image, self.size)

    def get_example(self, i):
        view0_file_path = self.row_labels["file_path_"][i]
        relative_view0_file_path = self.row_labels["relative_file_path_"][i]
        view0 = self.preprocess_image(view0_file_path)
        return {
            "view0": view0,
            "view1": view0,
            "file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
        }


class InferHook(Hook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config

    def before_epoch(self, epoch):
        self.root = os.path.join(
            self._root, "infer", "{:06}".format(self.get_global_step())
        )
        os.makedirs(self.root)
        self.inputs = {}
        self.outputs = {}

    def before_step(self, step, fetches, feeds, batch):
        fetches.update(self.model.outputs)

        for key in ["view0", "view1"]:
            if key not in self.inputs.keys():
                self.inputs[key] = list()
                self.inputs[key].append(batch[key])
            else:
                self.inputs[key].append(batch[key])

    def after_step(self, step, results):
        # for key in results.keys():
        for key in self.model.outputs.keys():
            if key not in self.outputs.keys():
                self.outputs[key] = list()
                self.outputs[key].append(results[key])
            else:
                self.outputs[key].append(results[key])

    def after_epoch(self, epoch):
        self.data = {"inputs": self.inputs, "outputs": self.outputs}
        out_path = os.path.join(self.root, "data.p")
        with open(out_path, "wb") as f:
            pickle.dump(self.data, f)
        print("Wrote retrieval matrix data in:")
        print(out_path)

        for k, o in self.outputs.items():
            i_out = 0
            for i, im_batch in enumerate(o):

                if len(im_batch.shape) == 3:
                    im_batch = np.expand_dims(im_batch, axis=-1)

                for im in np.split(im_batch, len(im_batch), 0):
                    out_path = os.path.join(self.root, "{}_{:07d}.png".format(k, i_out))
                    plot_batch(im, out_path)
                    i_out += 1

        for k, o in self.inputs.items():
            i_out = 0
            for i, im_batch in enumerate(o):

                if len(im_batch.shape) == 3:
                    im_batch = np.expand_dims(im_batch, axis=-1)

                for im in np.split(im_batch, len(im_batch), 0):
                    out_path = os.path.join(self.root, "{}_{:07d}.png".format(k, i_out))
                    plot_batch(im, out_path)
                    i_out += 1


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()
        root = ProjectManager.latest_eval
        hooks = [InferHook(self.get_global_step, root, self.model, self.config)]
        self.hooks += hooks

    def initialize(self, checkpoint_path=None):
        pass


if __name__ == "__main__":
    pickle_path = sys.argv[1]
    out_path = sys.argv[2]
    # make_matrix(pickle_path, out_path)
