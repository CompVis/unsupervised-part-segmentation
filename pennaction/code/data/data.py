import os
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import slic

import eddata
import eddata.utils as edu
from albumentations import (CLAHE, Blur, Compose, ElasticTransform, Flip,
                            GaussNoise, GridDistortion, HorizontalFlip,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            IAAEmboss, IAAPerspective, IAAPiecewiseAffine,
                            IAASharpen, MedianBlur, MotionBlur, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomRotate90, ShiftScaleRotate, Transpose,
                            ChannelShuffle, RGBShift, ToGray, RandomSizedCrop, Transpose)
from edflow.data.dataset import CsvDataset, SubDataset
from edflow.iterators.batches import DatasetMixin, load_image
from edflow.iterators.batches import resize_float32 as resize
from edflow.main import get_impl
from edflow.util import PRNGMixin

from eddata.stochastic_pair import StochasticPairs, StochasticPairsWithMask
import cv2

import warnings
warnings.warn("module moved to src.data.data", DeprecationWarning)

def add_choices(labels, return_by_cid=False):
    labels = dict(labels)
    cid_labels = np.asarray(labels["character_id"])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                print("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels["character_id"])):
        cid = labels["character_id"][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    if return_by_cid:
        return labels, cid_indices
    return labels

class AugmentedPair2(StochasticPairs):
    n_images = 3
    def __init__(self, config):
        super(AugmentedPair2, self).__init__(config)
        self.use_appearance_augmentation = config.get("data_augment_appearance", False)
        self.use_shape_augmentation = config.get("data_augment_shape", False)
        additional_targets = {
            "image{}".format(i): "image" for i in range(1, self.n_images)
        }
        p = 0.9
        appearance_augmentation = Compose(
            [
                OneOf(
                    [
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.5,
                ),
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),    
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),     
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),
                ToGray(p=0.1),  
                ChannelShuffle(p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.appearance_augmentation = appearance_augmentation  

        p = 0.9
        shape_augmentation = Compose(
            [
                HorizontalFlip(p=0.3),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=25, p=0.3, border_mode=cv2.BORDER_REPLICATE),
                OneOf([
                    # OpticalDistortion(p=0.3),
                    # GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.5),
                    ElasticTransform(p=0.5, border_mode=cv2.BORDER_REPLICATE)
                ], p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.shape_augmentation = shape_augmentation


    def stochastic_appearance_augmentation(self, *images):
        images = [(image + 1.0) * 255.0 / 2.0 for image in images]
        images = [image.astype(np.uint8) for image in images]
        aug_input = {"image": images[0]}
        aug_input.update(
            {"image{}".format(i): images[i] for i in range(1, len(images))}
        )
        augmented_data = self.appearance_augmentation(**aug_input)
        output_images = [
            augmented_data[k]
            for k in ["image"] + ["image{}".format(i) for i in range(1, len(images))]
        ]
        output_images = [
            o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
        ]
        return output_images


    def stochastic_shape_augmentation(self, *images):
        images = [(image + 1.0) * 255.0 / 2.0 for image in images]
        images = [image.astype(np.uint8) for image in images]
        aug_input = {"image": images[0]}
        aug_input.update(
            {"image{}".format(i): images[i] for i in range(1, len(images))}
        )
        augmented_data = self.shape_augmentation(**aug_input)
        output_images = [
            augmented_data[k]
            for k in ["image"] + ["image{}".format(i) for i in range(1, len(images))]
        ]
        output_images = [
            o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
        ]
        return output_images


    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)
        view0 = self.preprocess_image(self.labels["file_path_"][i])
        view1 = self.preprocess_image(self.labels["file_path_"][j])

        view0_target = view0.copy()

        # appearance augmentation has to be done in sync
        if self.use_appearance_augmentation:
            view0, = self.stochastic_appearance_augmentation(view0)
            view1, view0_target = self.stochastic_appearance_augmentation(view1, view0_target)

        if self.use_shape_augmentation:
            view0, view0_target = self.stochastic_shape_augmentation(view0, view0_target)
            view1, = self.stochastic_shape_augmentation(view1)
        return {"view0": view0, "view1": view1, "view0_target": view0_target}


class AugmentedPair2WithMask(StochasticPairsWithMask):
    n_images = 3
    def __init__(self, config):
        super(AugmentedPair2, self).__init__(config)
        self.use_appearance_augmentation = config.get("data_augment_appearance", False)
        self.use_shape_augmentation = config.get("data_augment_shape", False)
        additional_targets = {
            "image{}".format(i): "image" for i in range(1, self.n_images)
        }
        p = 0.9
        appearance_augmentation = Compose(
            [
                OneOf(
                    [
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.5,
                ),
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),    
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),     
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),
                ToGray(p=0.1),  
                ChannelShuffle(p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.appearance_augmentation = appearance_augmentation  

        additional_targets = {
            "image{}".format(i): "image" for i in range(1, self.n_images)
        }
        additional_target.update({
           "mask{}".format(i): "mask" for i in range(1, self.n_images) 
        })
        p = 0.9
        shape_augmentation = Compose(
            [
                HorizontalFlip(p=0.3),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=25, p=0.3, border_mode=cv2.BORDER_REPLICATE),
                OneOf([
                    # OpticalDistortion(p=0.3),
                    # GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.5),
                    ElasticTransform(p=0.5, border_mode=cv2.BORDER_REPLICATE)
                ], p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.shape_augmentation = shape_augmentation


    def stochastic_appearance_augmentation(self, *images):
        images = [(image + 1.0) * 255.0 / 2.0 for image in images]
        images = [image.astype(np.uint8) for image in images]
        aug_input = {"image": images[0]}
        aug_input.update(
            {"image{}".format(i): images[i] for i in range(1, len(images))}
        )
        augmented_data = self.appearance_augmentation(**aug_input)
        output_images = [
            augmented_data[k]
            for k in ["image"] + ["image{}".format(i) for i in range(1, len(images))]
        ]
        output_images = [
            o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
        ]
        return output_images


    def stochastic_shape_augmentation(self, images, masks):
        images = [(image + 1.0) * 255.0 / 2.0 for image in images]
        images = [image.astype(np.uint8) for image in images]
        aug_input = {"image": images[0], "mask" : masks[0]}
        aug_input.update(
            {"image{}".format(i): images[i] for i in range(1, len(images))}
        )
        aug_input.update(
            {"mask{}".format(i): masks[i] for i in range(1, len(masks))}
        )        
        augmented_data = self.shape_augmentation(**aug_input)
        output_images = [
            augmented_data[k]
            for k in ["image"] + ["image{}".format(i) for i in range(1, len(images))]
        ]
        output_images = [
            o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
        ]
        return output_images, output_masks


    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)
        view0, mask0 = self.preprocess_image(self.labels["file_path_"][i])
        view1, mask1 = self.preprocess_image(self.labels["file_path_"][j])

        view0_target = view0.copy()
        mask0_target = mask0.copy()

        # appearance augmentation has to be done in sync
        if self.use_appearance_augmentation:
            view0, = self.stochastic_appearance_augmentation(view0)
            [view1,], [view0_target,] = self.stochastic_appearance_augmentation([view1], view0_target)

        if self.use_shape_augmentation:
            [view0, view0_target], [mask0, mask0_target] = self.stochastic_shape_augmentation([view0, view0_target], [mask0, mask0_target])
            [view1], [mask1] = self.stochastic_shape_augmentation([view1], [mask1])
        return {"view0": view0, "view1": view1, "view0_target": view0_target, "mask0" : mask0, "mask1" : mask1, "mask0_target" : mask0_target}


class AugmentedPair3(AugmentedPair2):
    n_images = 3
    def __init__(self, config):
        super(AugmentedPair3, self).__init__(config)
        additional_targets = {
            "image{}".format(i): "image" for i in range(1, self.n_images)
        }
        p = 0.9
        appearance_augmentation = Compose(
            [
                OneOf(
                    [
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.5,
                ),
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),    
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),     
                OneOf(
                    [
                        RandomBrightnessContrast(p=0.3),
                        RGBShift(p=0.3),
                        HueSaturationValue(p=0.3),
                    ],
                    p=0.8,
                ),
                ToGray(p=0.1),  
                ChannelShuffle(p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.appearance_augmentation = appearance_augmentation  

        p = 0.9
        shape_augmentation = Compose(
            [
                OneOf([
                    Transpose(p=0.5),
                    HorizontalFlip(p=0.5),
                ], p=0.9),      
                OneOf([
                    RandomRotate90(p=1.0),
                ], p=0.9),                                             
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.25, rotate_limit=25, p=0.3, border_mode=cv2.BORDER_REPLICATE),
                OneOf([
                    # OpticalDistortion(p=0.3),
                    # GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.5),
                    ElasticTransform(p=0.5, border_mode=cv2.BORDER_REPLICATE)
                ], p=0.3),
            ],
            p=p,
            additional_targets=additional_targets,
        )
        self.shape_augmentation = shape_augmentation        


class CSVSegmentationDataset(CsvDataset):
    def __init__(self, config):
        """
        config = {
            data_root :
            data_csv :
            image_colname:
            labels_colname:
            pandas_kwargs : {
            }
        }

        Parameters
        ----------
        config
        """
        data_root = config["data_root"]
        data_csv = config["data_csv"]
        pandas_kwargs = config["pandas_kwargs"]
        super(CSVSegmentationDataset, self).__init__(data_csv, **pandas_kwargs)
        self.data_root = data_root
        self.image_colname = config["image_colname"]
        self.labels_colname = config["labels_colname"]  # TODO: rename to segmentation
        self.size = config["spatial_size"]

    def get_example(self, idx):
        example = super(CSVSegmentationDataset, self).get_example(idx)
        image_fpath = example[self.image_colname]
        labels_fpath = example[self.labels_colname]

        image = self.preprocess_image(os.path.join(self.data_root, image_fpath))
        labels = self.preprocess_labels(os.path.join(self.data_root, labels_fpath))

        return {"image": image, "labels": labels}

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        image = resize(image, self.size)
        return image

    def preprocess_labels(self, labels_fpath):
        labels = cv2.imread(labels_fpath, -1)[:, :, 0]
        labels = resize_labels(labels, (self.size, self.size))
        return labels


def resize_labels(labels, size):
    """Reshape labels image to target size.

    Parameters
    ----------
    labels : np.ndarray
        [H, W] or [N, H, W] - shaped array where each pixel is an `int` giving a label id for the segmentation. In case of [N, H, W],
        each slice along the first dimension is treated as an independent label image.
    size : tuple of ints
        Target shape as tuple of ints

    Returns
    -------
    reshaped_labels : np.ndarray
        [size[0], size[1]] or [N, size[0], size[1]]-shaped array

    Raises
    ------
    ValueError
        if labels does not have valid shape
    """
    # TODO: make this work for a single image
    if len(labels.shape) == 2:
        return cv2.resize(labels, size, 0, 0, cv2.INTER_NEAREST)
    elif len(labels.shape) == 3:
        if labels.shape[-1] != 1:
            raise ValueError("unsupported shape for labels : {}".format(labels.shape))
        label_list = np.split(labels, labels.shape[0], axis=0)
        label_list = list(
            map(
                lambda x: cv2.resize(np.squeeze(x), size, 0, 0, cv2.INTER_NEAREST),
                label_list,
            )
        )
        labels = np.stack(label_list, axis=0)
        return labels
    else:
        raise ValueError("unsupported shape for labels : {}".format(labels.shape))


class WeaklySupervisedDataset(DatasetMixin, PRNGMixin):
    required_config_attributes = ["dset1", "dset2", "dset1_config", "dset2_config"]

    def __init__(self, config):
        """
            config = {
                "dset1": "eddata.fashionmnist.FashionMNIST", # unsupervised set
                "dset1_config": {"spatial_size": 256},
                "dset2": "eddata.fashionmnist.FashionMNIST", # supervised set
                "dset2_config": {"spatial_size": 256},
                "dset2_n_samples": 10,
                "example_mapping": {
                    "dset1_image": "image_unsupervised",
                    "dset2_image": "image_supervised",
                },
            }

        Parameters
        ----------
        config


        Examples
        --------
            config = {
                "dset1": "eddata.fashionmnist.FashionMNIST",
                "dset1_config": {"spatial_size": 256},
                "dset2": "eddata.fashionmnist.FashionMNIST",
                "dset2_config": {"spatial_size": 256},
                "dset2_n_samples": 10,
                "example_mapping": {
                    "dset1_image": "image_unsupervised",
                    "dset2_image": "image_supervised",
                },
            }
            dset = WeaklySupervisedDataset(config)
            e = dset.get_example(0)
            print(e.keys())
            # >>> ['image_unsupervised', 'image_supervised']

            print(len(dset))
            # >>> 60000

        """

        self.config = config

        for rc in self.required_config_attributes:
            if not rc in config.keys():
                raise ValueError("invalid config. missing {}".format(rc))
        dset1_implementation = get_impl(config, "dset1")
        dset2_implementation = get_impl(config, "dset2")
        dset1_config = self.config["dset1_config"]
        dset2_config = self.config["dset2_config"]
        self.example_mapping = self.config.get("example_mapping", None)

        self.dset1 = dset1_implementation(dset1_config)
        self.dset2 = dset2_implementation(dset2_config)
        self.dset2 = SubDataset(
            self.dset2, range(self.config.get("dset2_n_samples", None))
        )

    def get_example(self, i):
        e_unsupervised = self.dset1.get_example(i)
        i2 = self.prng.randint(0, len(self.dset2))
        e_supervised = self.dset2.get_example(i2)
        e_unsupervised = {"dset1_{}".format(k): v for k, v in e_unsupervised.items()}
        e_supervised = {"dset2_{}".format(k): v for k, v in e_supervised.items()}

        merged_e = e_unsupervised
        merged_e.update(e_supervised)

        e = {new_key: merged_e[keys] for keys, new_key in self.example_mapping.items()}
        return e

    def __len__(self):
        return len(self.dset1)


class CsvDataset(DatasetMixin):
    """Using a csv file as index, this Dataset returns only the entries in the
    csv file, but can be easily extended to load other data using the
    :class:`ProcessedDatasets`.
    """

    def __init__(self, csv_root, **pandas_kwargs):
        """
        Parameters
        ----------
        csv_root : str
            Path/to/the/csv containing all datapoints. The
            first line in the file should contain the names for the
            attributes in the corresponding columns.
        pandas_kwargs : kwargs
            Passed to :func:`pandas.read_csv` when loading the csv file.
        """

        self.root = csv_root
        self.data = pd.read_csv(csv_root, **pandas_kwargs)

        # Stacking allows to also contain higher dimensional data in the csv
        # file like bounding boxes or keypoints.
        # Just make sure to load the data correctly, e.g. by passing the
        # converter ast.literal_val for the corresponding column.
        self.labels = {k: np.stack(self.data[k].values) for k in self.data}

    def get_example(self, idx):
        """Returns all entries in row :attr:`idx` of the labels."""

        # Labels are a pandas dataframe. `.iloc[idx]` returns the row at index
        # idx. Converting to dict results in column_name: row_entry pairs.
        return dict(self.data.iloc[idx])

from eddata.stochastic_pair import StochasticPairs
import cv2

# class AugmentedPair(StochasticPairs):
#     def __init__(self, config):
#         super(AugmentedPair, self).__init__(config)
#         self.use_appearance_augmentation = config.get("data_augment_appearance", False)
#         self.use_shape_augmentation = config.get("data_augment_shape", False)
#         additional_targets = {
#             "image{}".format(i): "image" for i in range(1, self.n_images)
#         }
#         p = 0.9
#         appearance_augmentation = Compose(
#             [
#                 OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.1),
#                 OneOf(
#                     [
#                         MedianBlur(blur_limit=3, p=0.1),
#                         Blur(blur_limit=3, p=0.1),
#                     ],
#                     p=0.1,
#                 ),
#                 OneOf(
#                     [
#                         CLAHE(clip_limit=2, p=0.3),
#                         IAASharpen(p=0.3),
#                         IAAEmboss(p=0.3),
#                         RandomBrightnessContrast(p=0.3),
#                         RGBShift(p=0.3),
#                     ],
#                     p=0.8,
#                 ),
#                 ChannelShuffle(p=0.3),
#                 HueSaturationValue(p=0.3),
#                 ToGray(p=0.3),                
#             ],
#             p=p,
#             additional_targets=additional_targets,
#         )
#         self.appearance_augmentation = appearance_augmentation  

#         p = 0.9
#         shape_augmentation = Compose([
#             HorizontalFlip(p=0.3),
#             ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=30, p=0.3, border_mode=cv2.BORDER_REPLICATE),
#             OneOf([
#                 # OpticalDistortion(p=0.3),
#                 # GridDistortion(p=0.1),
#                 IAAPiecewiseAffine(p=0.5),
#                 ElasticTransform(p=0.5, border_mode=cv2.BORDER_REPLICATE)
#             ], p=0.3),
#         ],
#         p=p)
#         self.shape_augmentation = shape_augmentation


#     def stochastic_appearance_augmentation(self, *images):
#         images = [(image + 1.0) * 255.0 / 2.0 for image in images]
#         images = [image.astype(np.uint8) for image in images]
#         aug_input = {"image": images[0]}
#         aug_input.update(
#             {"image{}".format(i): images[i] for i in range(1, len(images))}
#         )
#         augmented_data = self.appearance_augmentation(**aug_input)
#         output_images = [
#             augmented_data[k]
#             for k in ["image"] + ["image{}".format(i) for i in range(1, len(images))]
#         ]
#         output_images = [
#             o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
#         ]
#         return output_images


#     def stochastic_shape_augmentation(self, *images):
#         images = [(image + 1.0) * 255.0 / 2.0 for image in images]
#         images = [image.astype(np.uint8) for image in images]

#         output_images = []
#         for image in images:
#             aug_input = {"image" : image}
#             augmented_data = self.shape_augmentation(**aug_input)
#             output_images.append(augmented_data["image"])

#         output_images = [
#             o.astype(np.float32) * 2.0 / 255.0 - 1.0 for o in output_images
#         ]
#         return output_images


#     def get_example(self, i):
#         choices = self.labels["choices"][i]
#         if self.avoid_identity and len(choices) > 1:
#             choices = [c for c in choices if c != i]
#         j = self.prng.choice(choices)
#         view0 = self.preprocess_image(self.labels["file_path_"][i])
#         view1 = self.preprocess_image(self.labels["file_path_"][j])

#         # appearance augmentation has to be done in sync
#         if self.use_appearance_augmentation:
#             view0, view1 = self.stochastic_appearance_augmentation(view0, view1)

#         if self.use_shape_augmentation:
#             view0, view1 = self.stochastic_shape_augmentation(view0, view1)

#         return {"view0": view0, "view1": view1}



if __name__ == "__main__":
    from pylab import *
    from supermariopy import imageutils

    config = {
        "dset1": "eddata.stochastic_pair.StochasticPairs",  # unsupervised set
        "dset1_config": {
            "spatial_size": 256,
            "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset",
            "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/csvs/instance_level_train_split.csv",
            "data_avoid_identity": False,
            "data_flip": True,
            "data_csv_columns": ["character_id", "relative_file_path_"],
            "data_csv_has_header": True,
        },
        "dset2": "weakly_supervised_dataset.CSVSegmentationDataset",  # supervised set
        "dset2_config": {
            "spatial_size": 256,
            "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset",
            "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/denseposed_csv/denseposed_instance_level_train_split.csv",
            "pandas_kwargs": {},
            "image_colname": "im1",
            "labels_colname": "denseposed_im1",
        },
        "dset2_n_samples": 10,
        "example_mapping": {
            "dset1_view0": "image_unsupervised",
            "dset2_image": "image_supervised",
            "dset2_labels": "labels_supervised",
        },
    }

    dset = WeaklySupervisedDataset(config)
    example = dset.get_example(0)
    example.keys()
    plt.imshow(imageutils.convert_range(example["image_unsupervised"], [-1, 1], [0, 1]))
    plt.imshow(imageutils.convert_range(example["image_supervised"], [-1, 1], [0, 1]))
    plt.imshow(example["labels_supervised"])
