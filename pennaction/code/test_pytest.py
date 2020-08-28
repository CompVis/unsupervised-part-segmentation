import pytest


def test_fill_triangular():
    import util
    import tensorflow as tf
    import numpy as np

    # tf.enable_eager_execution()

    with tf.Session() as sess:
        L = np.arange(6)
        x = tf.constant(L)
        # a = tf.contrib.distributions.fill_triangular(x)
        b = util.fill_triangular(x)
        tf_result = sess.run([b])
    assert True  # this just checks if our custom code works at all

    with tf.Session() as sess:
        L = np.arange(6)
        x = tf.constant(L)
        a = tf.contrib.distributions.fill_triangular(x)
        b = util.fill_triangular(x)
        tf_result = sess.run([a, b])
    assert np.allclose(
        tf_result[0], tf_result[1]
    )  # this will only work if fill_triangular is in distributions



class Test_data():
    def test_AugmentedPair2WithMask(self, tmpdir):
        from skimage import data
        import numpy as np
        import os
        import cv2

        # TODO: refactor this
        p = tmpdir.mkdir("masking")
        img = data.astronaut()
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[200:400, 200:400] = 255
        cv2.imwrite(os.path.join(p, "image.png"), img)
        cv2.imwrite(os.path.join(p, "mask.png"), mask)
        with open(os.path.join(p, "data.csv"), "w") as f:
            print("1,image.png,mask.png", file=f)

        config = {
            "mask_label": 1,
            "apply_mask": False,
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
            "data_csv_columns": [
                "character_id",
                "relative_file_path_",
                "relative_mask_path_",
            ],
            "data_csv_has_header": False,
            "data_augment_appearance": True,
            "data_augment_shape": False
        }

        dset = stochastic_pair.StochasticPairsWithMask(config)
        example = dset.get_example(0)
