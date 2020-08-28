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
from edflow.iterators import batches
from edflow.project_manager import ProjectManager

from functools import partial

# from matplotlib import pyplot as plt
from edflow.custom_logging import get_logger
from typing import Tuple, List

try:
    from nips19 import nn as nn
    from nips19.eval import utils
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
from matplotlib import pyplot as plt

import cv2
import pandas as pd
import yaml
import glob
import click
from multiprocessing import Pool
from contextlib import closing
import tqdm
from tabulate import tabulate


class InferData(DatasetMixin):
    extracted_data_csv_columns = [
        "character_id",
        "relative_file_path_",
        "relative_iuv_path_",
    ]

    def __init__(self, config):
        self.config = config
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        self.csv_has_header = config.get("data_csv_has_header", False)
        self.max_n_samples = self.config.get("max_n_examples", None)
        self.make_labels()

    def make_labels(self):
        data_csv_columns = self.config.get(
            "data_csv_columns", self.extracted_data_csv_columns
        )
        if data_csv_columns == "from_csv":
            labels_df = pd.read_csv(self.csv)
            self.data_csv_columns = labels_df.columns
        else:
            self.data_csv_columns = data_csv_columns
            if self.csv_has_header:
                labels_df = pd.read_csv(self.csv)
            else:
                labels_df = pd.read_csv(self.csv, header=None)
            labels_df.rename(
                columns={
                    old: new
                    for old, new in zip(
                        labels_df.columns[: len(data_csv_columns)], data_csv_columns
                    )
                },
                inplace=True,
            )
        self.labels = dict(labels_df)
        self.labels = {k: list(v) for k, v in self.labels.items()}

        def add_root_path(x):
            return os.path.join(self.root, x)

        for label_name, i in zip(
            self.data_csv_columns, range(len(self.data_csv_columns))
        ):
            if "relative_" in label_name:
                label_update = {
                    label_name.replace("relative_", ""): list(
                        map(add_root_path, self.labels[label_name])
                    )
                }
                self.labels.update(label_update)
        self._length = len(list(self.labels.values())[0])

    def __len__(self):
        if self.max_n_samples is not None:
            return self.max_n_samples
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        return resize(image, self.size)

    def preprocess_iuv(self, iuv_path):
        IUV = cv2.imread(iuv_path)
        I = IUV[:, :, 0]
        I = cv2.resize(I, (self.size, self.size), cv2.INTER_NEAREST)
        return I

    def get_example(self, i):
        view0_file_path = self.labels["file_path_"][i]
        relative_view0_file_path = self.labels["relative_file_path_"][i]
        view0 = self.preprocess_image(view0_file_path)
        gt_segmentation_file_path = self.labels["iuv_path_"][i]
        gt_segmentation_relative_file_path = self.labels["relative_iuv_path_"][i]
        gt_segmentation = self.preprocess_iuv(gt_segmentation_file_path)
        example = {
            "view0": view0,
            "view1": view0,
            "file_path_": view0_file_path,
            "relative_file_path_": relative_view0_file_path,
            "gt_segmentation": gt_segmentation,
            "gt_segmentation_file_path": gt_segmentation_file_path,
            "gt_segmentation_relative_file_path": gt_segmentation_relative_file_path,
        }
        return example


class InferHook(CollectorHook):
    def __init__(self, global_step_getter, save_root, model, config):
        self.get_global_step = global_step_getter
        self._root = save_root
        self.model = model
        self.config = config
        self.batch_store_keys = self.config.get("batch_store_keys")
        self.batch_input_keys = self.config.get("batch_input_keys")
        self.fetch_output_keys = self.config.get("fetch_output_keys")
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
            os.makedirs(self.root, exist_ok=True)

        # fetch values from model outputs
        for key in self.fetch_output_keys:
            if key not in self.model.outputs.keys():
                pass
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

        global_step = int(os.path.basename(self.root))
        make_figure_1(self.data, self.root, self.config, global_step)



def write_slice(t, out_folder="."):
    """
        t = (i, image_slice)
        image_slice is [H, W, C]
    """
    i, image_slice = t
    fname = "{:06d}.png".format(i)
    batches.save_image(image_slice, os.path.join(out_folder, fname))



def write_container(container, name, output_dir):
    """container : [N, H, W, C], name : str"""
    arg_tuples = [(i, imslice) for i, imslice in enumerate(container)]
    outdir = os.path.join(output_dir, name)
    os.makedirs(outdir, exist_ok=True)
    _write_slice = partial(write_slice, **{"out_folder" : outdir})
    with closing(Pool(8)) as p:
        for _ in p.imap_unordered(_write_slice, arg_tuples):
            pass


def write_containers(containers, names, output_dir):
    for container, name in zip(containers, names):
        write_container(container, name, output_dir)



# TODO: import denseposelib
DEFAULT_FIGURE01_OPTIONS = {
    "inferred_segmentation_key": "out_parts_hard",  # from above
    "sampled_mask_key": "m0_sample",  # from above
    "input_view_key": "view0",
    "generated_image_key": "generated",
    "gt_segmentation_key": "gt_segmentation",
}


def make_figure_1(data: dict, root: str, config: dict, global_step: int):
    figure01_options = config.get("figure01_options")
    dp_semantic_remap_dict = config.get("dp_semantic_remap_dict")
    dp_new_part_list = sorted(list(dp_semantic_remap_dict.keys()))
    dp_remap_dict = denseposelib.semantic_remap_dict2remap_dict(
        dp_semantic_remap_dict, dp_new_part_list
    )

    inferred_segmentation = (
        data["outputs"][figure01_options["inferred_segmentation_key"]] + 1
    )  # +1 because the visualizer code uses + 1
    sampled_segmentation = data["outputs"][figure01_options["sampled_mask_key"]]
    images = data["inputs"][figure01_options["input_view_key"]]
    generated = data["outputs"][figure01_options["generated_image_key"]]
    groundtruth_segmentation = data["batches"][figure01_options["gt_segmentation_key"]]
    groundtruth_segmentation = denseposelib.resize_labels(
        groundtruth_segmentation, (128, 128)
    )
    remapped_gt_segmentation = denseposelib.remap_parts(
        groundtruth_segmentation, dp_remap_dict
    )
    best_remapping = denseposelib.compute_best_iou_remapping(
        inferred_segmentation, remapped_gt_segmentation
    )
    with open(os.path.join(root, 'best_remapping.yml'), 'w') as outfile:
        b = {int(k) : int(v) for k, v in best_remapping.items()}
        ddump = {"best_remapping" : b, "semantic_remap" : dp_semantic_remap_dict}

        yaml.dump(ddump, outfile, default_flow_style=False)
    remapped_inferred = denseposelib.remap_parts(inferred_segmentation, best_remapping)

    ncols = 7
    n_inferred_parts = config.get("n_inferred_parts", 10)
    colors = make_mask_colors(len(set(dp_new_part_list)), background_id=1)

    df = pd.DataFrame(columns=["global_step", "batch_idx"] + dp_new_part_list)

    for i in range(
        len(inferred_segmentation)
    ):  # TODO: maybe replace this amount of plots by parameters in the config file
        image_container = []

        # remap inferred segmentation
        old_inferred = inferred_segmentation[i]
        current_sampled_segmentation = np.argmax(sampled_segmentation[i], -1)
        old_inferred_colors = make_mask_colors(n_inferred_parts, background_id=1)

        image_container.append(old_inferred_colors[old_inferred - 1])
        image_container.append(old_inferred_colors[current_sampled_segmentation])

        new_inferred = remapped_inferred[i]
        current_gt_segmentation = remapped_gt_segmentation[i]

        # remap GT segmentation
        iou, iou_labels = denseposelib.compute_iou(
            new_inferred, current_gt_segmentation
        )

        # filter out background
        iou_filter = np.ones_like(iou) == 1.0
        iou_filter[iou_labels == dp_new_part_list.index("background")] = False

        df_update = {p: -1.0 for p in dp_new_part_list}
        df_update.update(
            {
                p: float(np.squeeze(iou[pi == iou_labels]))
                for pi, p in enumerate(dp_new_part_list)
                if pi in iou_labels
            }
        )
        df_update.update({"batch_idx": i, "global_step": global_step})

        df = df.append(df_update, ignore_index=True)

        filtered_iou = iou[iou_filter]
        mean_iou = np.mean(filtered_iou)

        image_container.append(colors[new_inferred])
        image_container.append(colors[current_gt_segmentation])

        legend_labels = []
        for pi, p in enumerate(dp_new_part_list):
            if pi in iou_labels:
                p_iou = np.squeeze(iou[np.argwhere(iou_labels == pi)])
            else:
                p_iou = 0.0
            legend_labels.append(p + " - IOU : {:.03f}".format(p_iou))
        legend_labels.append("mIOU (no BG) : {:.03f}".format(mean_iou))
        colors = np.concatenate([colors, np.reshape([0, 0, 0], (1, 3))], axis=0)
        text_colors = [1, 1, 1] * len(colors)
        legend_image = utils.make_legend_image(
            legend_labels, colors, text_colors, (128, 128), 1
        )
        image_container.append(legend_image)

        current_image = images[i]
        current_generated = generated[i]

        image_container.append(imageutils.convert_range(current_image, [-1, 1], [0, 1]))
        image_container.append(
            imageutils.convert_range(current_generated, [-1, 1], [0, 1])
        )
        # write_containers(image_container, 
        # ["1_out_parts_hard", "2_samples", "3_best_mapping", "4_GT", "5_view", "6_generated"],
        # root)
        for image, name in zip(image_container, ["1_out_parts_hard", "2_samples", "3_best_mapping", "4_GT", "5_IOUs", "6_view", "7_generated"]):
            os.makedirs(os.path.join(root, name), exist_ok=True)
            write_slice((i, imageutils.convert_range(image, [0, 1], [-1, 1])), os.path.join(root, name))

        # write files
        out_path = os.path.join(root, "figure_01")
        os.makedirs(out_path, exist_ok=True)
        out_image = np.stack(image_container)
        out_image = imageutils.convert_range(out_image, [0, 1], [-1, 1])
        batches.plot_batch(
            out_image, os.path.join(out_path, "{:06d}.png".format(i)), cols=ncols
        )

    df.to_csv(os.path.join(root, "part_ious.csv"), index=False, header=True)

    df_mean = df[df != -1].mean().to_frame().transpose()
    df_mean["overall"] = df_mean[
        filter(
            lambda x: x not in ["global_step", "batch_idx", "background"], df.columns
        )
    ].mean(axis=1)
    with open(os.path.join(root, "mean_part_ios.csv"), "w") as f:
        print(
            tabulate(df_mean, headers="keys", tablefmt="psql", showindex="never"),
            file=f,
        )


class Evaluator(TFBaseEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = ProjectManager.latest_eval
        hooks = []
        hook_config = self.config.get("hooks", {"outputs": True})
        hooks.append(InferHook(self.get_global_step, root, self.model, self.config))
        self.hooks += hooks


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
    target_dir
    config_path
    single
    n_processes

    Returns
    -------

    Examples
    --------

    # make figure for all checkpoints in infer folder
    eval_01.py eval/xxx/infer eval_01.yaml --no-single --n-processes 8

    # make figure for a single checkpoint
    eval_01.py eval/xxx/infer/500000 eval_01.yaml --single --no-processes 8

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


def unpack_tuples(t: Tuple[str, str, str]):
    """
    unpacks an argument tuple for @process_single_folder. Usefull for multiprocess usage
    Parameters
    ----------
    t : Tuple[str, str, str]
        tuple of (pickle_path: str, out_path: str, config_path: str)

    Returns
    -------
    output :
        process_single_folder(*t)
    """
    return process_single_folder(*t)


def process_single_folder(pickle_path: str, out_path: str, config_path: str):
    print("processing : ", pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    with open(config_path, "r") as f:
        config = yaml.load(f)

    if out_path.endswith("/"):
        out_path = out_path[:-1]
    global_step = int(os.path.basename(out_path))
    make_figure_1(data, out_path, config, global_step)


def make_mask_colors(
    n_parts,
    background_color=np.array([1, 1, 1], dtype=np.float32),
    background_id=0,
    cmap=plt.cm.inferno,
):
    """assuming background label is always 0, set color of value 0 to background color"""
    colors = cmap(np.linspace(0, 1, n_parts), alpha=False, bytes=False)[:, :3]
    colors = np.insert(colors, 1, background_color, axis=0)
    return colors


if __name__ == "__main__":
    main()
