import sys, os, pickle
import numpy as np
import math, os
import tensorflow as tf
from edflow.iterators.batches import load_image
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.tf_evaluator import TFBaseEvaluator
from edflow.hooks.hook import Hook
from edflow.hooks.checkpoint_hooks.common import CollectorHook
from edflow.iterators.batches import plot_batch, batch_to_canvas, save_image
from edflow.project_manager import ProjectManager

# from matplotlib import pyplot as plt
from edflow.custom_logging import get_logger
import tensorflow as tf

tf.enable_eager_execution()

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

        return {
            "view0": view0,
            "view1": view0,
            "file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
            "gt_segmentation": gt_segmentation,
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
                pass
            else:
                fetches[key] = self.model.outputs[key]

        # aggregate data from batch feeds for model input
        new_data = {k: v for k, v in batch.items() if k in self.batch_store_keys}
        # check if already enough batches have been aggregated
        # for k in self.batch_store_keys:
        #     if k in self.img_keys and self.batch_collection[k].shape[0] > self.n_vis:
        #         new_data.pop(k, None)
        self.stack_results(new_data, self.batch_collection)

        # arregate data from batch feeds for model input
        new_data = {k: v for k, v in batch.items() if k in self.batch_input_keys}
        # for k in self.batch_input_keys:
        #     if k in self.img_keys and self.input_collection[k].shape[0] > self.n_vis:
        #         new_data.pop(k, None)
        self.stack_results(new_data, self.input_collection)

    def after_step(self, step, results):
        new_data = {k: v for k, v in results.items() if k in self.fetch_output_keys}
        # for k in self.fetch_output_keys:
        #     if k in self.img_keys and self.output_collection[k].shape[0] > self.n_vis:
        #         new_data.pop(k, None)
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
        # make_clusters(self.data)


# P = ProjectManager()

import functools


def make_clusters(data: dict, root: str, config: dict, global_step: int) -> None:
    matching_app_features_0 = data["outputs"]["matching_app_features_0"]

    N, C, P, F = matching_app_features_0.shape
    matching_app_features_0 = np.reshape(
        np.rollaxis(matching_app_features_0, 2, 1), (N * P, C, F)
    )
    app_features_list = list(map(np.squeeze, np.split(matching_app_features_0, C, 1)))

    k = config.get("num_clusters", 2)
    n_vis = config.get("n_vis", 20)
    func_ = functools.partial(cluster_features, **{"k": k})
    centroids_and_labels = list(map(func_, app_features_list))

    new_data = {}
    new_data["clusters"] = np.stack([c[0] for c in centroids_and_labels])
    new_data["labels"] = np.stack([c[1] for c in centroids_and_labels])

    out_path = os.path.join(root, "clusters.p")
    with open(out_path, "wb") as f:
        pickle.dump(new_data, f)
    out_path = os.path.join(root, "figure_02")
    os.makedirs(out_path, exist_ok=True)
    decoding_mask4 = data["outputs"]["decoding_mask4"][:n_vis]
    view0 = data["inputs"]["view0"][:n_vis]
    for i in range(C):
        _, labels = centroids_and_labels[i]
        labels = labels[:n_vis]
        j = 0
        for canvas in yield_visualize_clusters(
            np.squeeze(decoding_mask4[..., i]), view0, labels, n_vis
        ):
            save_image(
                canvas, os.path.join(out_path, "{:06d}_cluster{:02d}.png".format(i, j))
            )
            j += 1


from sklearn.cluster import KMeans

# TODO: cluster with faiss


def cluster_features(features: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    Cluster features and return centroids and assigned labels

    Parameters
    ----------
    features : ndarray
        [N, F]-shaped array where each item [i, :] is an encoding
    k : int
        number of cluster centroids
    Returns
    -------
    centroids : ndarray
        [k, F]-shaped array where each item [i, :] is a centroid of the clustering
    labels : ndarray
        [N]-shaped array with assignments for each item in features to one of the k-clusters
    """
    kmeans = KMeans(n_clusters=k, init="k-means++").fit(np.squeeze(features))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return centroids, labels


def yield_visualize_clusters(
    mask: np.ndarray, view: np.ndarray, labels: np.ndarray, N_vis: int = 10
) -> np.ndarray:
    """
    Takes masks, views and assigned cluster labels and creates an image per cluster with the
    N_vis assigned images per cluster.

    Parameters
    ----------
    mask : np.ndarray
        [N, H, W]-shaped array where each element is an integer giving the assigned connected component
    view : np.ndarray
        [N, H, W, 3]-shaped array representing the RGB image
    labels : np.array
        [N]-shaped array with assignments for each item in features to one of the k-clusters.
    Returns
    -------
    canvas_list : list
        a list of canvas plots. Each canvas represents one clusters.
    """

    uniques = sorted(np.unique(labels))
    for u in uniques:
        m = mask[labels == u]
        v = view[labels == u]
        mask_one_hot = tf.one_hot(m, 5, axis=-1)[:, :, :, 1:]  # [N, H, W, 5]
        components = np.expand_dims(np.array(mask_one_hot), -1) * np.expand_dims(v, 3)
        N, H, W, P, _ = components.shape
        components = np.rollaxis(components, 3, 1)
        components = np.reshape(components, (N * P, H, W, 3))
        canvas = batch_to_canvas(components)
        yield canvas


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
    make_clusters(data, out_path, config, global_step)


if __name__ == "__main__":
    main()
