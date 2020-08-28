import sys, os, pickle
import numpy as np
import math, os
import tensorflow as tf
from edflow.iterators.batches import load_image
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.iterators import batches
from edflow.hooks.hook import Hook
from edflow.hooks.checkpoint_hooks.common import CollectorHook
from edflow.iterators.batches import plot_batch, batch_to_canvas, save_image
from edflow.project_manager import ProjectManager
from edflow.custom_logging import get_logger
import tensorflow as tf
from sklearn import cluster
from typing import List
from matplotlib import pyplot as plt
from ksvd import ApproximateKSVD


try:
    from nips19 import nn as nn
    from nips19.eval import utils

    sys.path.insert(0, ".")
except:
    try:
        sys.path.insert(0, ".")
        import nn as nn
        from eval import utils
    except:
        raise ImportError

from supermariopy import denseposelib, imageutils
from scipy.misc import imresize
import os

import cv2
import pandas as pd


class InferData(DatasetMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.row_csv = config["data_csv"]
        self.config = config

        self.row_labels = self._make_labels(self.row_csv)
        self.n_rows = len(self.row_labels["file_path_"])
        self._length = self.n_rows

    def _make_labels(self, csv):
        # with open(csv) as f:
        #     lines = f.read().splitlines()
        relative_paths = pd.read_csv(csv, header=None, names=["id", "image", "iuv"])
        absolute_paths = relative_paths.copy()
        absolute_paths["image"] = absolute_paths["image"].apply(
            lambda x: os.path.join(self.root, x)
        )
        absolute_paths["iuv"] = absolute_paths["iuv"].apply(
            lambda x: os.path.join(self.root, x)
        )
        labels = {
            "relative_file_path_": relative_paths.image,
            "file_path_": absolute_paths.image,
            "relative_iuv_path_": relative_paths.iuv,
            "iuv_path_": absolute_paths.iuv,
        }
        return labels

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        return resize(image, self.size)

    def preprocess_iuv(self, iuv_path):
        IUV = cv2.imread(iuv_path)
        I = IUV[:, :, 0]
        return I

    def get_example(self, i):
        view0_file_path = self.row_labels["file_path_"][i]
        relative_view0_file_path = self.row_labels["relative_file_path_"][i]
        view0 = self.preprocess_image(view0_file_path)
        gt_segmentation_file_path = self.row_labels["iuv_path_"][i]
        gt_segmentation_relative_file_path = self.row_labels["relative_iuv_path_"][i]
        gt_segmentation = self.preprocess_iuv(gt_segmentation_file_path)

        dp_semantic_remap_dict = self.config.get("dp_semantic_remap_dict")
        dp_new_part_list = sorted(list(dp_semantic_remap_dict.keys()))
        dp_remap_dict = denseposelib.semantic_remap_dict2remap_dict(
            dp_semantic_remap_dict, dp_new_part_list
        )

        spatial_size = self.config.get("spatial_size")
        gt_segmentation_resized = denseposelib.resize_labels(
            gt_segmentation, (spatial_size, spatial_size)
        )
        remapped_gt_segmentation = denseposelib.remap_parts(
            gt_segmentation_resized, dp_remap_dict
        )

        return {
            "view0": view0,
            "view1": view0,
            "file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
            "external_mask": remapped_gt_segmentation,
            "gt_segmentation_file_path": gt_segmentation_file_path,
            "gt_segmentation_relative_file_path": gt_segmentation_relative_file_path,
        }


class InferHook(CollectorHook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config
        self.batch_store_keys = self.config.get("batch_store_keys")
        self.batch_input_keys = self.config.get("batch_input_keys")
        self.fetch_output_keys = self.config.get("fetch_output_keys")
        self.img_keys = self.config.get("img_keys")
        self.n_vis = self.config.get("n_vis")
        self.logger = get_logger(self)

    def before_epoch(self, epoch):
        self.input_collection = {}
        self.output_collection = {}
        self.batch_collection = {}

    def before_step(self, step, fetches, feeds, batch):
        if step == 0:  # otherwise global step will be incremented
            self.root = os.path.join(
                self._root, "infer", "{:06}".format(self.get_global_step())
            )
            os.makedirs(self.root)

        # fetch values from model outputs
        for key in self.fetch_output_keys:
            if key not in self.model.outputs.keys():
                raise KeyError(
                    "{} not in model.outputs.keys {}".format(
                        key, self.model.outputs.keys()
                    )
                )
            else:
                fetches[key] = self.model.outputs[key]

        # aggregate data from batch feeds for model input
        new_data = {k: v for k, v in batch.items() if k in self.batch_store_keys}
        self.stack_results(new_data, self.batch_collection)

        # arregate data from batch feeds for model input
        new_data = {k: v for k, v in batch.items() if k in self.batch_input_keys}
        self.stack_results(new_data, self.input_collection)

    def after_step(self, step, results):
        new_data = {k: v for k, v in results.items() if k in self.fetch_output_keys}
        self.stack_results(new_data, self.output_collection)

    def after_epoch(self, epoch):
        self.data = {
            "inputs": self.input_collection,
            "outputs": self.output_collection,
            "batches": self.batch_collection,
        }
        out_path = os.path.join(self.root, "data.p")
        with open(out_path, "wb") as f:
            pickle.dump(self.data, f)
        self.logger.info("Wrote retrieval matrix data in: {}".format(out_path))


def make_figure_3(data: dict, root: str, config: dict, global_step: int):
    features = data["outputs"]["external_app_features"]
    external_mask_ids = data["outputs"]["external_mask_ids"]

    kmeans_options = config.get("kmeans_options")
    kmeans = Kmeans(**kmeans_options)

    ksvd_options = config.get("ksvd_options")
    ksvd = KSVD(**ksvd_options)

    ids = ["arm", "background", "foot", "hand", "leg", "upper body"]

    clustering_implementations = [kmeans, ksvd]  # type: List[ClusteringAlgo]

    for clustering_implementation in clustering_implementations:
        assigned_labels = clustering_implementation.cluster(features)
        label_filters = [assigned_labels == l for l in np.unique(assigned_labels)]
        uniques_and_counts = [
            np.unique(external_mask_ids[l], return_counts=True) for l in label_filters
        ]
        fig, axes = plot_assignments(uniques_and_counts, ids)
        plt.savefig(
            os.path.join(root, "app_features" + str(clustering_implementation) + ".png")
        )
        plt.tight_layout()
        plt.close("")

    if config.get("cluster_pose", False):
        features = data["outputs"]["external_pose_features"]
        external_mask_ids = data["outputs"]["external_mask_ids"]

        kmeans_options = config.get("kmeans_options")
        kmeans = Kmeans(**kmeans_options)

        ksvd_options = config.get("ksvd_options")
        ksvd = KSVD(**ksvd_options)

        ids = ["arm", "background", "foot", "hand", "leg", "upper body"]

        clustering_implementations = [kmeans]  # type: List[ClusteringAlgo]

        for clustering_implementation in clustering_implementations:
            assigned_labels = clustering_implementation.cluster(features)
            label_filters = [assigned_labels == l for l in np.unique(assigned_labels)]
            uniques_and_counts = [
                np.unique(external_mask_ids[l], return_counts=True)
                for l in label_filters
            ]
            fig, axes = plot_assignments(uniques_and_counts, ids)
            plt.savefig(
                os.path.join(
                    root, "pose_features" + str(clustering_implementation) + ".png"
                )
            )
            plt.tight_layout()
            plt.close("")


class ClusteringAlgo:
    """Abstract Clustering Algo class"""

    def cluster(self, features):
        """Cluster features and return assigned cluster ids

        Parameters
        ----------
        features : np.ndarray
            [N, F] - shapedd array of observations

        Returns
        -------
            [N] - shaped array of assigned cluster ids
        """
        return np.zeros((features.shape[0]))

    def __str__(self):
        return "Abstract clustering"


class Kmeans(ClusteringAlgo):
    def __init__(self, *args, **kwargs):
        self._kmeans = cluster.KMeans(*args, **kwargs)

    def cluster(self, features):
        self._kmeans.fit(np.squeeze(features))
        return self._kmeans.labels_

    def __str__(self):
        return "sklearn.clustering.kmeans"


class KSVD(ClusteringAlgo):
    def __init__(self, *args, **kwargs):
        self._aksvd = ApproximateKSVD(*args, **kwargs)

    def cluster(self, features):
        self._aksvd.fit(features)
        gamma = self._aksvd.transform(features)
        assignments = np.argmax(np.abs(gamma), axis=1)
        return assignments

    def __str__(self):
        return "ksvd"


import math


def plot_assignments(uniques_and_counts, ids):
    N = len(uniques_and_counts)
    r = math.floor(math.sqrt(N))
    c = N // r
    fig, axes = plt.subplots(r, c, figsize=(3 * c, 3 * r))
    axes = axes.ravel()
    max_c = np.max(np.concatenate([u_and_c[1] for u_and_c in uniques_and_counts]))
    for ax, u_and_c in zip(axes, uniques_and_counts):
        ax.bar(u_and_c[0], u_and_c[1] / max_c)
        ax.set_ylim([0, 1])
        ax.set_xticks(u_and_c[0], ids)
    return fig, axes


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = ProjectManager.latest_eval
        hooks = []
        hook_config = self.config.get("hooks", {"outputs": True})
        hooks.append(InferHook(self.get_global_step, root, self.model, self.config))
        self.hooks += hooks


import yaml
import glob
import click
from multiprocessing import Pool


@click.command()
@click.argument("target-dir")
@click.argument("config-path")
@click.option(
    "--single/--no-single",
    default=True,
    help="Single will only infer a single folder. "
    "no-single will apply a map to all folders",
)
@click.option(
    "--n-processes", default=8, help="How many processes to use when using --no-single"
)
def main(target_dir: str, config_path: str, single: bool, n_processes: int):
    """

    Parameters
    ----------
    target_dir: str
    config_path: str
    single: bool
    n_processes: int

    Returns
    -------

    Examples
    --------

    run from ~/code/nips19/nips19

    # make figure for all checkpoints in infer folder
    eval_02.py eval/xxx/infer eval_02.yaml --no-single --n-processes 8

    # make figure for a single checkpoint
    eval_02.py eval/xxx/infer/500000 eval_02.yaml --single --no-processes 8

    # from ~/code/nips19/nips19
    python eval/eval_02/eval_02.py \
    ../logs/2019-07-09T00:20:03_33b_deepfashion/eval/2019-08-02T13:27:24_eval_02/infer/500000 \
    eval/eval_02/eval_02.yaml \
    --single
    """
    if single:
        pickle_path = os.path.join(target_dir, "data.p")
        out_path = target_dir
        process_single_folder(pickle_path, out_path, config_path)
    else:
        pickle_paths = sorted(glob.glob(os.path.join(target_dir, "*", "data.p")))
        out_paths = list(map(os.path.dirname, pickle_paths))
        arg_tuples = [(pp, op, config_path) for pp, op in zip(pickle_paths, out_paths)]

        with closing(Pool(n_processes)) as p:
            for _ in tqdm.tqdm(p.imap(unpack_tuples, arg_tuples)):
                pass


from contextlib import closing
import tqdm


def unpack_tuples(t):
    return process_single_folder(*t)


def process_single_folder(pickle_path, out_path, config_path):
    print("processing : ", pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    with open(config_path, "r") as f:
        config = yaml.load(f)

    global_step = int(os.path.basename(out_path))
    make_figure_3(data, out_path, config, global_step)


if __name__ == "__main__":
    main()
