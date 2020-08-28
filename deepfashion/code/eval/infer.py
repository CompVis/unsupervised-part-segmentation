import sys, os, pickle
import numpy as np
import math, os
import tensorflow as tf
from edflow.iterators.batches import load_image
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.hooks.hook import Hook
from edflow.iterators.batches import plot_batch
from edflow.project_manager import ProjectManager
from matplotlib import pyplot as plt
from edflow.custom_logging import get_logger

from nips19 import nn as nn


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
        self.plot_batches = self.config.get("plot_batches", False)
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        self.inputs = {}
        self.outputs = {}

    def before_step(self, step, fetches, feeds, batch):
        if step == 0:  # otherwise global step will be incremented
            self.root = os.path.join(
                self._root, "infer", "{:06}".format(self.get_global_step())
            )
            os.makedirs(self.root, exist_ok=True)

        fetches.update(self.model.outputs)

        for key in ["view0", "view1"]:
            if key not in self.inputs.keys():
                self.inputs[key] = list()
                self.inputs[key].append(batch[key])
            else:
                self.inputs[key].append(batch[key])

    def after_step(self, step, results):
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
        self.logger.info("Wrote retrieval matrix data in: {}".format(out_path))

        for k, o in self.outputs.items():
            i_out = 0
            for i, im_batch in enumerate(o):
                if len(im_batch.shape) == 3:
                    im_batch = np.expand_dims(im_batch, axis=-1)

                if self.plot_batches:
                    out_path = os.path.join(self.root, "{}_{:07d}.png".format(k, i_out))
                    plot_batch(im_batch, out_path)
                    i_out += 1

                else:
                    for im in np.split(im_batch, len(im_batch), 0):
                        out_path = os.path.join(
                            self.root, "{}_{:07d}.png".format(k, i_out)
                        )
                        plot_batch(im, out_path)
                        i_out += 1

        for k, o in self.inputs.items():
            i_out = 0
            for i, im_batch in enumerate(o):

                if len(im_batch.shape) == 3:
                    im_batch = np.expand_dims(im_batch, axis=-1)

                if self.plot_batches:
                    out_path = os.path.join(self.root, "{}_{:07d}.png".format(k, i_out))
                    plot_batch(im_batch, out_path)
                    i_out += 1
                else:
                    for im in np.split(im_batch, len(im_batch), 0):
                        out_path = os.path.join(
                            self.root, "{}_{:07d}.png".format(k, i_out)
                        )
                        plot_batch(im, out_path)
                        i_out += 1


class InferLogitsHook(Hook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config
        self.plot_batches = self.config.get("plot_batches", False)

        self.view0_key = self.config.get("infer_logits_hook_options")["view0_key"]
        self.view0_mask_rgb_key = self.config.get("infer_logits_hook_options")[
            "view0_mask_rgb_key"
        ]
        self.part_logit_key = self.config.get("infer_logits_hook_options")[
            "part_logit_key"
        ]
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        self.inputs = {}
        self.outputs = {}

    def before_step(self, step, fetches, feeds, batch):
        if step == 0:  # otherwise global step will be incremented
            self.root = os.path.join(
                self._root, "infer_logits", "{:06}".format(self.get_global_step())
            )
            os.makedirs(self.root)
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
        self.logger.info("Wrote retrieval matrix data in: {}".format(out_path))

        vs = self.inputs[self.view0_key]
        ms_rgb = self.outputs[self.view0_mask_rgb_key]

        part_keys = list(
            filter(lambda x: self.part_logit_key in x, self.outputs.keys())
        )
        parts = {pk: self.outputs[pk] for pk in part_keys}

        i_out = 0
        for i in range(len(vs)):
            for j in range(vs[i].shape[0]):
                v = np.squeeze(vs[i][j])
                m = np.squeeze(ms_rgb[i][j])
                p = {k: np.squeeze(parts[k][i])[j] for k in parts.keys()}
                plt.close("all")
                fig, ax = plot_logits(v, p)
                out_path = os.path.join(
                    self.root, "{}_{:07d}.png".format("logits", i_out)
                )
                fig.savefig(out_path)
                i_out += 1


P = ProjectManager()


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = ProjectManager.latest_eval
        hooks = []
        hook_config = self.config.get("hooks", {"outputs": True})
        if "outputs" in hook_config.keys() and hook_config["outputs"]:
            hooks.append(InferHook(self.get_global_step, root, self.model, self.config))
        if "logits_1D" in hook_config.keys() and hook_config["logits_1D"]:
            hooks.append(
                InferLogitsHook(self.get_global_step, root, self.model, self.config)
            )
        self.hooks += hooks


def change_fontsize(ax, fs):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fs)
    return ax


def change_linewidth(ax, lw=3):
    for item in ax.lines:
        item.set_linewidth(lw)
    return ax


def change_legend_linewidth(ax, lw=2.0):
    leg = ax.legend_
    for legobj in leg.legendHandles:
        legobj.set_linewidth(lw)


from scipy import special


def plot_logits(v, ps):
    """
    v : (x, y, 3) view0 immage
    ps : {"part1_logits" : l1, "part2_logits" : l2} dict with logits
    m_rgb : (x, y, 3) rgb mask
    """
    n_parts = len(sorted(ps.keys()))
    m_logits = np.stack([ps[k] for k in sorted(ps.keys())], axis=-1)
    if len(m_logits.shape) < 4:
        m_logits = np.expand_dims(m_logits, 0)
    m_softmax = special.softmax(m_logits, axis=-1)
    m_rgb = nn.np_mask2rgb(m_softmax)
    m_rgb = np.squeeze(m_rgb)
    m_rgb += 1.0
    m_rgb /= 2.0
    v += 1.0
    v /= 2.0

    x = [0, v.shape[0]]
    y = [v.shape[0] // 2] * 2
    x_idx = np.arange(x[0], x[1])
    y_idx = np.arange(y[0], y[0])
    if not y_idx:
        y_idx = np.array(y[0])

    colors = nn.make_mask_colors(n_parts)
    with plt.style.context("seaborn"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # vertical line
        ax = axes[0, 0]
        ax.imshow(v)
        dotted_line = plt.Line2D(x, y, lw=5.0)
        ax.add_line(dotted_line)
        ax.set_axis_off()

        ax = axes[0, 1]
        ax.imshow(m_rgb)
        dotted_line = plt.Line2D(x, y, lw=5.0)
        ax.add_line(dotted_line)
        ax.set_axis_off()

        ax = axes[0, 2]
        for c, k in zip(colors, sorted(ps.keys())):
            p = ps[k]
            ax.plot(np.squeeze(p[y_idx, x_idx]), label=k, c=c)

        ax.legend(frameon=True, fontsize=18, bbox_to_anchor=(1.0, 1.0))
        ax.set_xlabel("u", fontsize=16)
        ax.set_ylabel("logit activation", fontsize=16)
        ax.grid(True)
        fig.tight_layout()

        change_fontsize(ax, 16)
        change_linewidth(ax, 5)
        change_legend_linewidth(ax, 3.0)

        # horizontal line
        ax = axes[1, 0]
        ax.imshow(v)
        dotted_line = plt.Line2D(y, x, lw=5.0)
        ax.add_line(dotted_line)
        ax.set_axis_off()

        ax = axes[1, 1]
        ax.imshow(m_rgb)
        dotted_line = plt.Line2D(y, x, lw=5.0)
        ax.add_line(dotted_line)
        ax.set_axis_off()

        ax = axes[1, 2]
        for c, k in zip(colors, sorted(ps.keys())):
            p = ps[k]
            ax.plot(np.squeeze(p[x_idx, y_idx]), label=k, c=c)

        ax.legend(frameon=True, fontsize=18, bbox_to_anchor=(1.0, 1.0))
        ax.set_xlabel("u", fontsize=16)
        ax.set_ylabel("logit activation", fontsize=16)
        ax.grid(True)
        fig.tight_layout()

        change_fontsize(ax, 16)
        change_linewidth(ax, 5)
        change_legend_linewidth(ax, 3.0)
    return fig, axes


if __name__ == "__main__":
    pickle_path = sys.argv[1]
    out_path = sys.argv[2]
