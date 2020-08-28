import sys, os
import tensorflow as tf
import numpy as np

from edflow.iterators.trainer import TFBaseTrainer
import edflow.iterators.deeploss as deeploss
from edflow.hooks.evaluation_hooks import RestoreTFModelHook
from edflow.util import make_linear_var

import nips19.nn as nn

# from triplet_reid.edflow_implementations.implementations import make_network as make_triplet_net

dsize = 512

PARTS_DIM = 3
FEATURE_DIM = 4

import tensorflow as tf
import numpy as np


def categorical_kl(probs):
    k = tf.to_float(probs.shape.as_list()[-1])
    logkp = tf.log(k * probs + 1e-20)
    kl = tf.reduce_sum(probs * logkp, axis=-1)
    return tf.reduce_mean(kl)


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0.0, maxval=1.0)
    return -tf.log(-tf.log(U + eps) + eps)


def mean_prob_penalty(probs):
    """ penalty towards more spatially equal probabilities """
    k = tf.to_float(probs.shape.as_list()[-1])
    mean_ = tf.reduce_mean(probs, axis=[1, 2])
    error = mean_ - 1.0 / k
    return tf.square(error)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def make_ema(init_value, value, update_ops, decay=0.99):
    decay = tf.constant(decay, dtype=tf.float32)
    avg_value = tf.Variable(init_value, dtype=tf.float32, trainable=False)
    update_ema = tf.assign(
        avg_value, decay * avg_value + (1.0 - decay) * tf.cast(value, tf.float32)
    )
    update_ops.append(update_ema)
    return avg_value


def triplet_wrapper(edim):
    name = "my_triplet_is_the_best_triplet"

    def tnet(images):
        endpoints, _ = make_triplet_net(images, edim=edim, name=name)
        emb = endpoints["emb"]
        emb = tf.expand_dims(emb, axis=1)
        emb = tf.expand_dims(emb, axis=1)
        return emb

    return tnet, name


def encoder_model(x, out_size, config, extra_resnets, activation="relu"):
    with nn.model_arg_scope(activation=activation):
        h = nn.conv2d(x, config[0])
        h = nn.residual_block(h)

        for nf in config[1:]:
            h = nn.downsample(h, nf)
            h = nn.residual_block(h)

        for _ in range(extra_resnets):
            h = nn.residual_block(h)

        h = nn.activate(h)
        h = tf.reduce_mean(h, [1, 2], keepdims=True)
        h = nn.nin(h, out_size)

        return h


def decoder_model(h1, h2, config, extra_resnets, activation="relu"):
    with nn.model_arg_scope(activation=activation):
        h1 = nn.nin(h1, 4 * 4 * config[-1])
        h1 = tf.reshape(h1, [-1, 4, 4, config[-1]])
        h2 = nn.nin(h2, 4 * 4 * config[-1])
        h2 = tf.reshape(h2, [-1, 4, 4, config[-1]])
        for _ in range(extra_resnets):
            h1 = nn.residual_block(h1)
            h2 = nn.residual_block(h2)

        h = tf.concat([h1, h2], axis=-1)
        h = nn.conv2d(h, config[-1])

        for nf in config[-2::-1]:
            h = nn.residual_block(h)
            h = nn.upsample(h, nf)

        h = nn.residual_block(h)
        h = nn.conv2d(h, 3)

        return h


def localize(x, softmax=False, centered=False):
    if softmax:
        x = tf.nn.softmax(x, axis=PARTS_DIM)
    B, h, w, parts = x.shape.as_list()
    u = tf.range(0, h, dtype=tf.float32)
    v = tf.range(0, w, dtype=tf.float32)
    #     uu, vv = tf.meshgrid(u, v)
    uu, vv = tf.split(tf_meshgrid(h, w), 2, axis=-1)
    uu = tf.reshape(uu, uu.shape.as_list()[:-1])
    vv = tf.reshape(vv, vv.shape.as_list()[:-1])

    u = tf.reshape(u, (1, h, 1, 1))
    v = tf.reshape(v, (1, 1, w, 1))

    uu = tf.expand_dims(uu, axis=0)
    uu = tf.expand_dims(uu, axis=-1)
    vv = tf.expand_dims(vv, axis=0)
    vv = tf.expand_dims(vv, axis=-1)

    norm_const = tf.reduce_sum(x, axis=[1, 2], keep_dims=True)
    px = x / norm_const
    p_u = tf.reduce_sum(px, axis=[2], keep_dims=True)
    p_v = tf.reduce_sum(px, axis=[1], keep_dims=True)

    mean_v = tf.reduce_sum(px * vv, axis=[1, 2])
    mean_u = tf.reduce_sum(px * uu, axis=[1, 2])

    def variance(px, x, axis=[1, 2]):
        mean_x = tf.reduce_sum(px * x, axis=axis, keep_dims=True)
        return tf.reduce_sum(px * (x - mean_x) ** 2, axis=axis, keep_dims=True)

    var_u = variance(p_u, u)
    var_v = variance(p_v, v)

    if centered:
        mean_u = mean_u - h / 2.0
        mean_u = mean_u / h * 2.0
        mean_v = mean_v - w / 2.0
        mean_v = mean_v / w * 2.0

        var_u /= (h / 2.0) ** 2
        var_v /= (w / 2.0) ** 2
    return (mean_u, mean_v), (var_v, var_u)


def hourglass_model(
    x,
    config,
    extra_resnets,
    alpha=None,
    pi=None,
    n_out=3,
    activation="relu",
    upsample_method="subpixel",
):
    alpha = None
    pi = None
    with nn.model_arg_scope(activation=activation):
        hs = list()

        h = nn.conv2d(x, config[0])
        h = nn.residual_block(h)

        for nf in config[1:]:
            h = nn.downsample(h, nf)
            h = nn.residual_block(h)
            hs.append(h)

        for _ in range(extra_resnets):
            h = nn.residual_block(h)

        extras = []
        if alpha is not None:
            ha = nn.nin(alpha, 4 * 4 * config[-1])
            ha = tf.reshape(ha, [-1, 4, 4, config[-1]])
            for _ in range(extra_resnets):
                ha = nn.residual_block(ha)
            extras.append(ha)
        if pi is not None:
            hp = nn.nin(pi, 4 * 4 * config[-1])
            hp = tf.reshape(hp, [-1, 4, 4, config[-1]])
            for _ in range(extra_resnets):
                hp = nn.residual_block(hp)
            extras.append(hp)

        if extras:
            h = tf.concat([h] + extras, axis=-1)
            h = nn.conv2d(h, config[-1])

        for i, nf in enumerate(config[-2::-1]):
            h = nn.residual_block(h, skipin=hs[-(i + 1)])
            h = nn.upsample(h, nf, method=upsample_method)

        h = nn.residual_block(h)
        h = nn.conv2d(h, n_out)
        return h


def single_decoder_model(
    h, n_out=3, config=None, activation="relu", upsample_method="subpixel"
):
    with nn.model_arg_scope(activation=activation):
        h = nn.nin(h, 4 * 4 * config[-1])
        h = tf.reshape(h, [-1, 4, 4, config[-1]])

        h = nn.conv2d(h, config[-1])
        h = nn.residual_block(h)

        for nf in config[-2::-1]:
            h = nn.residual_block(h)
            h = nn.upsample(h, nf, method=upsample_method)

        h = nn.residual_block(h)
        h = nn.conv2d(h, n_out)

        return h


def discriminator_model(pair, activation="relu"):
    with nn.model_arg_scope(activation=activation):
        outpair = list()
        for z in pair:
            nc = dsize
            h = nn.nin(z, nc)
            for _ in range(4):
                h = nn.residual_block(h, conv=nn.nin)
            h = nn.activate(h)
            h = nn.nin(h, nc)
            outpair.append(h)
        h = outpair[0] * outpair[1]
        h = tf.reduce_sum(h, [1, 2, 3])
        h = tf.expand_dims(h, -1)
        return h


def pretty_discriminator_model(x, c):
    with nn.model_arg_scope(activation="relu"):
        hs = list()
        h = nn.conv2d(x, convconf[0])
        hs.append(h)

        for nf in convconf[1:]:
            h = nn.downsample(h, nf)
            h = nn.residual_block(h)
            hs.append(h)

        h = nn.activate(h)
        h = nn.conv2d(h, dsize)

        h = tf.reduce_mean(h, [1, 2], keepdims=True)
        hc = nn.nin(c, dsize)
        hc = nn.residual_block(hc, conv=nn.nin)
        hc = nn.residual_block(hc, conv=nn.nin)
        h = h * hc

        h = tf.reduce_mean(h, [1, 2, 3])
        h = tf.expand_dims(h, -1)
        return h, hs


def upsample_linear(x, factor):
    factor = int(factor)
    _, h, w, _ = x.shape.as_list()
    out = tf.image.resize_bilinear(x, [factor * h, factor * w])
    return out


def pool_features(feature_map, mask):
    bs, h, w, n_features = feature_map.shape.as_list()
    mshape = mask.shape.as_list()
    assert mshape[0] == bs and mshape[1] == h and mshape[2] == w, mshape
    n_parts = mshape[3]

    # feature_map = tf.expand_dims(feature_map, 3)
    feature_map = tf.reshape(feature_map, (bs, h, w, n_parts, -1))
    mask = tf.expand_dims(mask, 4)
    masked_features = feature_map * mask
    output = tf.reduce_mean(masked_features, axis=[1, 2])

    out_shape = output.shape.as_list()
    assert len(out_shape) == 3, out_shape
    assert (
        out_shape[0] == bs
        and out_shape[1] == n_parts
        and out_shape[2] == n_features / n_parts
    ), out_shape

    return output


def tf_hm2(P, h, w, stddev):
    """Coordinates to Heatmap Layer

    P   : float Tensor
          xy coordinates of points in coordinate system [-1, 1]
          shape [num_batch, n_points, 2]
    h   : int
          Output height
    w   : int
          Output width
    stddev: float
            Standard deviation of heatmap, i.e. width with respect to [-1, 1]
            [num_batch, n_points, 2]

    Returns
        : float Tensor
          Heatmap with values in [0, 1]
          shape [num_batch, h, w, n_points]

    Examples
        B = 2
        H = 20
        W = 20
        parts = 2
        means = np.array([[10, 10], [10, 15]], dtype=np.float32) # means in [0, H] space
        means = ( means - np.array([H, W]) / 2 ) / np.array([H, W]) # means in [-1, 1] space
        variances = np.array([[3, 1], [1, 3]], dtype=np.float32) / ((np.array([H, W]) / 2)) # variance in [-1, 1] space
        variances = tf.reshape(variances, (1, 2, 2))
        points = tf.reshape(means, (1, 2, 2))
        points = tf.concat([points] * B, 0)
        variances = tf.concat([variances] * B, 0)
        heatmap = tf_hm(points, H, W, variances)
        for i in range(2):
            fig, ax = plt.subplots()
            ax.imshow(np.squeeze(heatmap[0, :, :, i]))

    """
    meshgrid = tf_meshgrid(h, w)
    assert meshgrid.shape == (h, w, 2)
    meshgrid = np.reshape(meshgrid, [1, h, w, 1, 2])  # b,h,w,p,2
    P = tf.expand_dims(P, 1)
    P = tf.expand_dims(P, 2)  # b,h,w,p,2
    d = tf.square(meshgrid - P)
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 2)  # b,h,w,p,2
    d = -d / (2 * stddev ** 2)
    logits = tf.reduce_sum(d, 4)  # b,h,w,p
    #     heat = tf.exp(logits)

    #     heat /= 2.0 * math.pi * stddev[:,:, 0] * stddev[:, : 1]
    return 1 / (1 - logits)


def mask_parts(image, mask, softmax=False):
    """ [B, H, W, 3], [B, H, W, parts] --> [B, H, W, parts, 3] """
    bs, h, w, n_fetures = image.shape.as_list()
    mshape = mask.shape.as_list()
    assert mshape[0] == bs and mshape[1] == h and mshape[2] == w, mshape
    n_parts = mshape[3]

    image = tf.expand_dims(image, PARTS_DIM)
    part_heatmaps = tf.expand_dims(mask, FEATURE_DIM)
    final_masks = part_heatmaps

    masked_image = image * final_masks

    return masked_image


def make_blobs(mask):
    bs, h, w, parts = mask.shape.as_list()
    means, variances = localize(mask, softmax=False, centered=False)
    means = tf.stack(means, axis=-1)
    variances = [tf.reshape(v, (bs, -1)) for v in variances]
    variances = tf.stack(variances, axis=-1)
    # variances *= 1e-1
    part_heatmaps = tf_hm(means, h, w, variances)
    return part_heatmaps


def apply_partwise(input_, func):
    """
        Applies function func on all parts separately.
        Parts are in channel 3.
        The input is reshaped to map the parts to the batch axis and then the function is applied
    Parameters
    ----------
    input_ : tensor
        [b, h, w, parts, features]
    func :
        a NN function

    Returns
    -------
        [b, out_h, out_w, parts, out_features]
    """

    b, h, w, parts, f = input_.shape.as_list()

    # transpose [b, h, w, part, features] --> [part, b, h, w, features]
    perm = [3, 0, 1, 2, 4]
    x = tf.transpose(input_, perm=perm)
    # reshape [part, b, h, w, features] --> [part * b, h, w, features]
    x = tf.reshape(x, (b * parts, h, w, f))

    y = func(x)

    _, h_out, w_out, c_out = y.shape.as_list()
    # reshape [part * b, h_out, w_out, c_out] --> [part, b, h_out, w_out, c_out]
    out = tf.reshape(y, (parts, b, h_out, w_out, c_out))
    # transpose back [part, b, h_out, w_out, c_out] --> [b, h_out, w_out, part, c_out]
    inv_perm = [1, 2, 3, 0, 4]
    out = tf.transpose(out, perm=inv_perm)
    return out


def make_masks(pi, decoder, n_parts):
    """

    Parameters
    ----------
    pi : tensor
        [1, 1, 1, z0_size]
    decoder
    n_parts

    Returns
    -------

    """
    b, _, _, features = pi.shape.as_list()
    assert features % n_parts == 0
    pi_part_features = features // n_parts
    pi = tf.reshape(pi, (b, 1, 1, n_parts, pi_part_features))
    masks = apply_partwise(pi, decoder)
    _, h, w, _, _ = masks.shape.as_list()
    masks = tf.reshape(masks, (b, h, w, -1))
    assert masks.shape[3] == n_parts
    return masks


def encode_parts(part_image, encoder):
    """ [B, H, W, parts, 3] --> [B, parts, features] """
    b, h, w, parts, channels = part_image.shape.as_list()
    part_encodings = apply_partwise(part_image, encoder)

    part_encodings = tf.reshape(part_encodings, (b, parts, -1))
    out_shape = part_encodings.shape.as_list()
    assert out_shape[0] == b and out_shape[1] == parts
    return part_encodings


def unpool_features(feature_vectors, mask, reshape=False):
    """ output is [b, h, w, n_parts, features] """
    bs, h, w, n_parts = mask.shape.as_list()
    fshape = feature_vectors.shape.as_list()
    assert len(fshape) == 3, fshape
    assert fshape[0] == bs and fshape[1] == n_parts, fshape
    n_features = fshape[2]

    mask = tf.expand_dims(mask, 4)
    feature_map = feature_vectors
    feature_map = tf.expand_dims(feature_map, 1)
    feature_map = tf.expand_dims(feature_map, 2)

    # explicitly rout appearance features part-wise
    # [feature_map[b, h, w, i, features] * m[b, h, , i, 1] for i in parts]
    perm = [3, 0, 1, 2, 4]
    inv_perm = [1, 2, 3, 0, 4]
    elems = (tf.transpose(mask, perm=perm), tf.transpose(feature_map, perm=perm))
    # _mask = tf.where(mask > 0, tf.zeros(shape=(bs, h, w, n_parts, 1)), tf.ones(shape=(bs, h, w, n_parts, 1)))
    # output = _mask * feature_map
    output = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.float32)
    output = tf.transpose(output, perm=inv_perm)

    out_shape = output.shape.as_list()
    assert len(out_shape) == 5, out_shape
    assert (
        out_shape[0] == bs
        and out_shape[1] == h
        and out_shape[2] == w
        and out_shape[3] == n_parts
        and out_shape[4] == n_features
    ), out_shape

    if reshape:
        output = tf.reshape(output, [bs, h, w, n_parts * n_features])

    return output


def tf_meshgrid(h, w):
    #     xs = np.linspace(-1.0,1.0,w)
    xs = np.arange(0, w)
    ys = np.arange(0, h)
    #     ys = np.linspace(-1.0,1.0,h)
    xs, ys = np.meshgrid(xs, ys)
    meshgrid = np.stack([xs, ys], 2)
    meshgrid = meshgrid.astype(np.float32)
    return meshgrid


def tf_hm(P, h, w, stddev, exp=True):
    """Coordinates to Heatmap Layer

    P   : float Tensor
          xy coordinates of points in coordinate system [-1, 1]
          shape [num_batch, n_points, 2]
    h   : int
          Output height
    w   : int
          Output width
    stddev: float
            Standard deviation of heatmap, i.e. width with respect to [-1, 1]
            [num_batch, n_points, 2]

    Returns
        : float Tensor
          Heatmap with values in [0, 1]
          shape [num_batch, h, w, n_points]

    Examples
        B = 2
        H = 20
        W = 20
        parts = 2
        means = np.array([[10, 10], [10, 15]], dtype=np.float32) # means in [0, H] space
        means = ( means - np.array([H, W]) / 2 ) / np.array([H, W]) # means in [-1, 1] space
        variances = np.array([[3, 1], [1, 3]], dtype=np.float32) / ((np.array([H, W]) / 2)) # variance in [-1, 1] space
        variances = tf.reshape(variances, (1, 2, 2))
        points = tf.reshape(means, (1, 2, 2))
        points = tf.concat([points] * B, 0)
        variances = tf.concat([variances] * B, 0)
        heatmap = tf_hm(points, H, W, variances)
        for i in range(2):
            fig, ax = plt.subplots()
            ax.imshow(np.squeeze(heatmap[0, :, :, i]))

    """
    meshgrid = tf_meshgrid(h, w)
    assert meshgrid.shape == (h, w, 2)
    meshgrid = np.reshape(meshgrid, [1, h, w, 1, 2])  # b,h,w,p,2
    P = tf.expand_dims(P, 1)
    P = tf.expand_dims(P, 2)  # b,h,w,p,2
    d = tf.square(meshgrid - P)
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 2)  # b,h,w,p,2
    d = -d / (2 * stddev ** 2)
    logits = tf.reduce_sum(d, 4)  # b,h,w,p
    if exp:
        heat = tf.exp(logits)
    #     heat /= 2.0 * math.pi * stddev[:,:, 0] * stddev[:, : 1]
    else:
        heat = 1 / (1 - logits)
    return heat


def reshape_4D(x):
    """ reshapes 5D tensor to 4D tensor"""
    bs, h, w, n_parts, n_features = x.shape.as_list()
    out = tf.reshape(x, [bs, h, w, n_parts * n_features])
    return out


def pool_unpool_block(feature_map, pool_mask, unpool_mask, reshape=False):
    local_app_features = pool_features(feature_map, pool_mask)
    # unpool them to mask0
    injected_mask = unpool_features(local_app_features, unpool_mask, reshape=reshape)
    return local_app_features, injected_mask


def normalize(x):
    xmin = tf.reduce_min(x, [1, 2], keepdims=True)
    xmax = tf.reduce_max(x, [1, 2], keepdims=True)
    return (x - xmin) / (xmax - xmin + 1e-6)


class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.pretty = self.config.get("use_pretty", False)
        variables = set(tf.global_variables())
        self.define_graph()
        self.variables = [v for v in tf.global_variables() if not v in variables]
        # self.variables = [v for v in self.variables if not self.e1_name in v.name]

    @property
    def inputs(self):
        return {"view0": self.views[0], "view1": self.views[1]}

    @property
    def outputs(self):
        _outputs = {
            "generated": self._generated,
            "visualize0": self.mask0,
            "visualize1": self.mask1,
            "view0_part0": self.mask0[..., 0],
            "view1_part0": self.mask1[..., 0],
        }
        _outputs.update(
            {
                "view0_part{}".format(i): self.mask0[..., i]
                for i in range(self.config.get("n_parts"))
            }
        )
        _outputs.update(
            {
                "view1_part{}".format(i): self.mask1[..., i]
                for i in range(self.config.get("n_parts"))
            }
        )
        _outputs.update({"view0_mask_rgb": mask2rgb(self.mask0)})
        _outputs.update({"view1_mask_rgb": mask2rgb(self.mask1)})
        _outputs.update({"view1_mask_rgb": mask2rgb(self.mask1)})
        _outputs.update(
            {
                "view1_parts_blob_{}".format(i): self._view1_parts[..., i]
                for i in range(self.config.get["n_parts"])
            }
        )
        return _outputs

    def define_graph(self):
        # input placeholders
        self.views = list()
        for view_idx in range(2):
            v = tf.placeholder(
                tf.float32,
                shape=(
                    self.config["batch_size"],
                    self.config["spatial_size"],
                    self.config["spatial_size"],
                    3,
                ),
                name="input_view_{}".format(view_idx),
            )
            self.views.append(v)

        z0_size = self.config.get("z0_size", 256)
        local_app_size = self.config.get("local_app_size", 64)
        batch_size = self.config["batch_size"]

        z0_n_parameters = nn.FullLatentDistribution.n_parameters(z0_size)
        n_parts = self.config["n_parts"]

        # submodules
        encoder_kwargs = self.config.get("encoder0")
        e0 = nn.make_model(
            "encoder_0", encoder_model, out_size=z0_n_parameters, **encoder_kwargs
        )

        # e1 = nn.make_model("encoder_1", encoder_model, out_size = z1_size)
        dd_kwargs = self.config.get("final_hour")
        dd = nn.make_model("decoder_delta", hourglass_model, **dd_kwargs)

        app_extractor_kwargs = self.config.get("encoder1")
        e1 = nn.make_model(
            "encoder_1", encoder_model, out_size=local_app_size, **app_extractor_kwargs
        )

        dv_kwargs = self.config.get("dv")
        dv = nn.make_model(
            "decoder_visualize", single_decoder_model, n_out=n_parts, **dv_kwargs
        )

        d_debug_mask_kwargs = self.config.get("d_debug_mask")
        d_debug_mask = nn.make_model(
            "debug_decoder_mask", hourglass_model, n_out=3, **d_debug_mask_kwargs
        )

        d_mean_beta_kwargs = self.config.get("d_mean_beta")
        d_local_app1 = nn.make_model(
            "decoder_mean_beta1", single_decoder_model, n_out=3, **d_mean_beta_kwargs
        )

        discriminator_kwargs = self.config.get("discriminator")
        mi_estimator = nn.make_model(
            "mi_estimator", discriminator_model, **discriminator_kwargs
        )
        mi0_discriminator = nn.make_model(
            "mi0_discriminator", discriminator_model, **discriminator_kwargs
        )
        mi1_discriminator = nn.make_model(
            "mi1_discriminator", discriminator_model, **discriminator_kwargs
        )

        if self.pretty:
            pretty_discriminator = nn.make_model(
                "pretty_discriminator", pretty_discriminator_model
            )

        # pose
        self.stochastic_encoder_0 = not self.config.get("test_mode", False)
        z_00_parameters = e0(self.views[0])
        z_00_distribution = self.z_00_distribution = nn.FullLatentDistribution(
            z_00_parameters, z0_size, stochastic=self.stochastic_encoder_0
        )
        z_01_parameters = e0(self.views[1])
        z_01_distribution = self.z_01_distribution = nn.FullLatentDistribution(
            z_01_parameters, z0_size, stochastic=self.stochastic_encoder_0
        )
        # appearance
        z_10 = self.z_10 = e1(self.views[0])
        z_11 = self.z_11 = e1(self.views[1])
        z_1_independent = self.z_1_independent = tf.reverse(z_10, [0])

        # masks
        self.mask0_logits = dv(z_00_distribution.sample())
        self.mask1_logits = tf.stop_gradient(dv(z_01_distribution.sample()))

        # masks [m1, m2, ..., m6]
        if self.config.get("apply_softmax", True):
            self.mask0 = tf.nn.softmax(self.mask0_logits)
            self.mask1 = tf.nn.softmax(self.mask1_logits)
            # self.mask0_sample = gumbel_softmax(self.mask0_logits)
            # self.mask1_sample = gumbel_softmax(self.mask1_logits)
            self.mask0 = make_blobs(self.mask0)
            self.mask1 = make_blobs(self.mask1)
            self.mask0_sample = self.mask0
            self.mask1_sample = self.mask1
        else:
            self.mask0 = self.mask0_logits
            self.mask1 = self.mask1_logits
            self.mask0 = make_blobs(self.mask0)
            self.mask1 = make_blobs(self.mask1)
            self.mask0_sample = self.mask0
            self.mask1_sample = self.mask1

        self.mask0_logits_list = [self.mask0_logits]
        self.mask1_logits_list = [self.mask1_logits]
        self.mask0_list = [self.mask0]
        self.mask1_list = [self.mask1]

        self._debug_mask_reconstruction = d_debug_mask(tf.stop_gradient(self.mask0))

        ############ reconstruct x0 from x1
        # localized appearances
        # pool them
        # local_app_features1 = pool_features(self.views[1], self.mask1_sample)

        view1_parts = mask_parts(self.views[1], self.mask1_sample, softmax=True)
        self._view1_parts = view1_parts
        local_app_features1 = encode_parts(view1_parts, e1)
        # unpool them to mask0

        injected_mask0 = unpool_features(
            local_app_features1, self.mask0_sample, reshape=False
        )
        injected_mask0 = reshape_4D(injected_mask0)

        # reconstruct
        self._generated = dd(injected_mask0)

        # debugging decoder from local app features to reconstruction
        # To see if information is bypassing the mask

        B = local_app_features1.shape.as_list()[0]
        self._mean_beta_rec1 = d_local_app1(
            tf.stop_gradient(tf.reshape(nn.flatten(local_app_features1), (B, 1, 1, -1)))
        )
        ############ visualize transfer
        # ! need to stop gradients on z, otherwise graph building hangs?
        independent_local_app_features = tf.stop_gradient(
            tf.reverse(local_app_features1, [0])
        )
        independent_injected_mask0 = unpool_features(
            independent_local_app_features, self.mask0_sample, reshape=False
        )
        independent_injected_mask0 = reshape_4D(independent_injected_mask0)
        self._cross_generated = dd(independent_injected_mask0)

        if self.pretty:
            pass
            # # pretty one
            # self.logit_pretty_orig, self.feat_pretty_orig = pretty_discriminator(self.views[0], tf.stop_gradient(z_11))
            # self.fake_input = tf.concat([self._generated, self._cross_generated], axis=0)
            # fake_selection = tf.random_uniform((self.config["batch_size"],),
            #                                    minval=0, maxval=2 * self.config["batch_size"],
            #                                    dtype=tf.int32)
            # self.fake_input = tf.gather(self.fake_input, fake_selection, axis=0)
            # self.logit_pretty_fake, self.feat_pretty_fake = pretty_discriminator(self.fake_input,
            #                                                                      tf.stop_gradient(z_1_independent))

        # mi gradients
        self.lon = tf.Variable(1.0, dtype=tf.float32, trainable=False)

        self.logit_joint0 = mi0_discriminator(
            (z_00_distribution.sample(noise_level=self.lon), z_11)
        )
        self.logit_marginal0 = mi0_discriminator(
            (z_00_distribution.sample(noise_level=self.lon), z_1_independent)
        )

        # local_app_4d = tf.reshape(local_app_features1, (batch_size, 1, 1, -1))
        # independent_local_app_4d = tf.reshape(independent_local_app_features, (batch_size, 1, 1, -1))
        #
        # self.logit_joint0 = mi0_discriminator((z_00_distribution.sample(noise_level=self.lon), local_app_4d))
        # self.logit_marginal0 = mi0_discriminator((z_00_distribution.sample(noise_level=self.lon), independent_local_app_4d))

        self.logit_joint1 = mi1_discriminator((z_00_distribution.sample(), z_11))
        self.logit_marginal1 = mi1_discriminator(
            (z_00_distribution.sample(), z_1_independent)
        )

        # self.logit_joint1 = mi1_discriminator((z_00_distribution.sample(), local_app_4d))
        # self.logit_marginal1 = mi1_discriminator((z_00_distribution.sample(), independent_local_app_4d))

        # mi estimation
        self.mi_logit_joint = mi_estimator((z_00_distribution.sample(), z_11))
        self.mi_logit_marginal = mi_estimator(
            (z_00_distribution.sample(), z_1_independent)
        )
        #
        # self.mi_logit_joint = mi_estimator((z_00_distribution.sample(), local_app_4d))
        # self.mi_logit_marginal = mi_estimator((z_00_distribution.sample(), independent_local_app_4d))


def filter_parts(m, include_list):
    """
        m = tf.ones((12, 12, 10, 3, 128))
        mm = filter_parts(m, [1, 0, 1])
        mm.shape # ==> 12, 12, 10, 2, 128
    """
    n_parts = m.shape[3]
    assert len(include_list) == n_parts
    part_masks = tf.split(m, n_parts, axis=PARTS_DIM)
    part_masks = [part_masks[i] for i in range(n_parts) if include_list[i]]
    return tf.concat(part_masks, axis=PARTS_DIM)


def hinge_logit_loss(logits, real):
    if real:
        return tf.reduce_mean(tf.maximum(0.0, 1.0 - logits))
    else:
        return tf.reduce_mean(tf.maximum(0.0, 1.0 + logits))


def logit_loss(logits, real):
    # real is joint
    if real:
        return tf.reduce_mean(tf.nn.softplus(-logits))
    else:
        return tf.reduce_mean(tf.nn.softplus(logits))


def logit_constraint(logits, real):
    if real:
        return tf.reduce_mean(-logits)
    else:
        return tf.reduce_mean(logits)


def gradient_penalty(output, inputs):
    grads = tf.gradients(output, inputs)
    norms = list()
    for input_, grad in zip(inputs, grads):
        input_axes = list(range(1, len(input_.shape.as_list())))
        norm = tf.reduce_mean(tf.reduce_sum(tf.square(grads), input_axes))
        norms.append(norm)
    norms = tf.add_n(norms) / len(norms)
    return norms


def phase_energy(x, gamma):
    dy, dx = tf.image.image_gradients(x)
    grad_squared = tf.square(dy) + tf.square(dx)
    e = (0.25 * (x * (x - 1.0)) ** 2) / gamma + 0.5 * gamma * grad_squared
    e = tf.reduce_sum(e, [1, 2, 3])
    e = tf.reduce_mean(e, [0])
    return e


def grad_energy(x):
    dy, dx = tf.image.image_gradients(x)
    grad_squared = tf.square(dy) + tf.square(dx)
    e = 0.5 * grad_squared
    e = tf.reduce_sum(e, [1, 2, 3])
    e = tf.reduce_mean(e, [0])
    return e


def visualize_mask(name, mask, image, img_ops):
    n_parts = mask.shape.as_list()[3]
    vis_mask = mask2rgb(mask)

    img_ops[name + "_visualization"] = vis_mask

    for i_part in range(n_parts):
        part_mask = mask[:, :, :, i_part]
        part_mask = tf.expand_dims(part_mask, 3)
        img_ops[name + "_part_mask_{:02}".format(i_part)] = 2.0 * part_mask - 1.0
        part = image * part_mask
        img_ops[name + "_part_{:02}".format(i_part)] = part


def mask2rgb(mask):
    n_parts = mask.shape.as_list()[3]
    maxmask = tf.argmax(mask, axis=3)
    hotmask = tf.one_hot(maxmask, depth=n_parts)
    hotmask = tf.expand_dims(hotmask, 4)
    prng = np.random.RandomState(1)
    colors = prng.uniform(low=-1.0, high=1.0, size=(n_parts, 3))
    colors = tf.to_float(colors)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    vis_mask = hotmask * colors
    vis_mask = tf.reduce_sum(vis_mask, axis=3)
    return vis_mask


def concat_parts(t_list):
    """ concat tensor along its parts dim """
    assert isinstance(t_list, list)
    t_item_shape = t_list[0].shape.as_list()
    if len(t_item_shape) == 4:  # [b, h, w, parts]
        return tf.concat(t_list, axis=3)
    elif len(t_item_shape) == 3:  # [b, parts, features]
        return tf.concat(t_list, axis=1)
    elif len(t_item_shape) == 5:  # [b, h, w, parts, features]
        return tf.concat(t_list, axis=3)
    else:
        raise NotImplementedError


class Trainer(TFBaseTrainer):
    def get_restore_variables(self):
        vs = super().get_restore_variables()
        default_exclude = [
            "pretty_discriminator",
            "beta1_power_7",
            "beta2_power_7",
            "phase_weight",
            "grad_weight",
            "phase_gamma",
        ]
        exclude = self.config.get("restore_exclude", default_exclude)
        for name in exclude:
            vs = [v for v in vs if not name in v.name]
        if self.config.get("add_pretty", False):
            exclude = ["pretty_discriminator", "beta1_power_7", "beta2_power_7"]
            for name in exclude:
                vs = [v for v in vs if not name in v.name]
            return vs
        else:
            return vs

    def initialize(self, checkpoint_path=None):
        return_ = super().initialize(checkpoint_path)
        # triplet_path = self.config["triplet_path"]
        # e1_name = 'my_triplet_is_the_best_triplet'
        # triplet_variables = [v for v in tf.global_variables() if e1_name in v.name]
        # restorer = RestoreTFModelHook(variables=triplet_variables, checkpoint_path=None)
        # with self.session.as_default():
        #     restorer(triplet_path)
        # print("Restored triplet net.")
        return return_

    def make_loss_ops(self):
        # perceptual network
        with tf.variable_scope("VGG19__noinit__"):
            gram_weight = self.config.get("gram_weight", 0.0)
            print("GRAM: {}".format(gram_weight))
            self.vgg19 = vgg19 = deeploss.VGG19Features(
                self.session, default_gram=gram_weight, original_scale=True
            )
        dim = np.prod(self.model.inputs["view0"].shape.as_list()[1:])
        auto_rec_loss = (
            1e-3
            * 0.5
            * dim
            * vgg19.make_loss_op(self.model.inputs["view0"], self.model._generated)
        )
        debug_decoder_mask_rec_loss = (
            1e-3
            * 0.5
            * dim
            * vgg19.make_loss_op(
                self.model.inputs["view0"], self.model._debug_mask_reconstruction
            )
        )
        mean_beta_rec_loss1 = (
            1e-3
            * 0.5
            * dim
            * vgg19.make_loss_op(self.model.inputs["view1"], self.model._mean_beta_rec1)
        )

        with tf.variable_scope("grad_weight"):
            grad_weight = make_linear_var(
                step=self.global_step, **self.config["grad_weight"]
            )
        with tf.variable_scope("kl_weight"):
            kl_weight = make_linear_var(
                step=self.global_step, **self.config["kl_weight"]
            )

        # smoothness prior (phase energy)
        mask0_grad_energy = grad_weight * tf.reduce_sum(
            [grad_energy(m) for m in self.model.mask0_list]
        )
        self.log_ops["mask0_grad_energy"] = mask0_grad_energy
        self.log_ops["mask0_grad_weight"] = grad_weight

        mask0_kl = kl_weight * tf.reduce_sum(
            [categorical_kl(m) for m in self.model.mask0_list]
        )  # TODO: categorical KL
        self.log_ops["mask0_kl_weight"] = kl_weight
        self.log_ops["mask0_kl"] = mask0_kl

        # per submodule loss
        losses = dict()

        # delta
        losses["encoder_0"] = auto_rec_loss

        # decoder
        losses["decoder_delta"] = auto_rec_loss
        losses["encoder_1"] = auto_rec_loss

        # debug decoder
        losses["debug_decoder_mask"] = debug_decoder_mask_rec_loss

        if not self.config["apply_softmax"]:
            _, variances = localize(self.model.mask0_list[0], softmax=True)
        else:
            _, variances = localize(self.model.mask0_list[0], softmax=False)
        n_parts = self.config["n_parts"]
        # exclude background variance
        localization_loss = (
            tf.reduce_sum(
                variances[0][:, :, :, : (n_parts - 1)]
                + variances[1][:, :, :, : (n_parts - 1)]
            )
            ** 2
        )
        self.log_ops["localization"] = localization_loss
        localization_weight = make_linear_var(
            step=self.global_step, **self.config["localization_weight"]
        )
        localization_loss *= localization_weight
        self.log_ops["localization_weight"] = localization_weight
        self.log_ops["weighted_localization"] = localization_loss

        if self.config.get("apply_softmax", True):
            losses["decoder_visualize"] = (
                auto_rec_loss + mask0_grad_energy + mask0_kl + localization_loss
            )
        else:
            losses["decoder_visualize"] = auto_rec_loss + localization_loss

        losses["decoder_mean_beta1"] = mean_beta_rec_loss1

        # mi estimators
        loss_dis0 = 0.5 * (
            logit_loss(self.model.logit_joint0, real=True)
            + logit_loss(self.model.logit_marginal0, real=False)
        )
        losses["mi0_discriminator"] = loss_dis0
        loss_dis1 = 0.5 * (
            logit_loss(self.model.logit_joint1, real=True)
            + logit_loss(self.model.logit_marginal1, real=False)
        )
        losses["mi1_discriminator"] = loss_dis1
        # estimator
        losses["mi_estimator"] = 0.5 * (
            logit_loss(self.model.mi_logit_joint, real=True)
            + logit_loss(self.model.mi_logit_marginal, real=False)
        )

        # accuracies of discriminators
        def acc(ljoint, lmarg):
            correct_joint = tf.reduce_sum(tf.cast(ljoint > 0.0, tf.int32))
            correct_marginal = tf.reduce_sum(tf.cast(lmarg < 0.0, tf.int32))
            accuracy = (correct_joint + correct_marginal) / (2 * tf.shape(ljoint)[0])
            return accuracy

        dis0_accuracy = acc(self.model.logit_joint0, self.model.logit_marginal0)
        dis1_accuracy = acc(self.model.logit_joint1, self.model.logit_marginal1)
        est_accuracy = acc(self.model.mi_logit_joint, self.model.mi_logit_marginal)

        # averages
        avg_acc0 = make_ema(0.5, dis0_accuracy, self.update_ops)
        avg_acc1 = make_ema(0.5, dis1_accuracy, self.update_ops)
        avg_acc_error = make_ema(0.0, dis1_accuracy - dis0_accuracy, self.update_ops)
        self.log_ops["avg_acc_error"] = avg_acc_error
        avg_loss_dis0 = make_ema(1.0, loss_dis0, self.update_ops)
        avg_loss_dis1 = make_ema(1.0, loss_dis1, self.update_ops)

        # Parameters
        MI_TARGET = self.config["MI"].get("mi_target", 0.125)
        MI_SLACK = self.config["MI"].get("mi_slack", 0.05)

        LOO_TOL = 0.025
        LON_LR = 0.05
        LON_ADAPTIVE = False

        LOA_INIT = self.config["MI"].get("loa_init", 0.0)
        LOA_LR = self.config["MI"].get("loa_lr", 4.0)
        LOA_ADAPTIVE = self.config["MI"].get("loa_adaptive", True)

        LOR_INIT = self.config["MI"].get("lor_init", 7.5)
        LOR_LR = self.config["MI"].get("lor_lr", 0.05)
        LOR_MIN = self.config["MI"].get("lor_min", 1.0)
        LOR_MAX = self.config["MI"].get("lor_max", 7.5)
        LOR_ADAPTIVE = self.config["MI"].get("lor_adaptive", True)

        # delta mi minimization
        mim_constraint = logit_constraint(self.model.logit_joint0, real=False)
        independent_mim_constraint = logit_constraint(
            self.model.logit_joint1, real=False
        )

        # level of overpowering
        avg_mim_constraint = tf.maximum(
            0.0, make_ema(0.0, mim_constraint, self.update_ops)
        )
        avg_independent_mim_constraint = tf.maximum(
            0.0, make_ema(0.0, independent_mim_constraint, self.update_ops)
        )
        self.log_ops["avg_mim"] = avg_mim_constraint
        self.log_ops["avg_independent_mim"] = avg_independent_mim_constraint
        loo = (avg_independent_mim_constraint - avg_mim_constraint) / (
            avg_independent_mim_constraint + 1e-6
        )
        loo = tf.clip_by_value(loo, 0.0, 1.0)
        self.log_ops["loo"] = loo

        # level of noise
        lon_gain = -loo + LOO_TOL
        self.log_ops["lon_gain"] = lon_gain
        lon_lr = LON_LR
        new_lon = tf.clip_by_value(self.model.lon + lon_lr * lon_gain, 0.0, 1.0)
        if LON_ADAPTIVE:
            update_lon = tf.assign(self.model.lon, new_lon)
            self.update_ops.append(update_lon)
        self.log_ops["model_lon"] = self.model.lon
        # self.log_ops["softmax_weight"] = softmax_weight

        # OPTION
        adversarial_regularization = True
        if adversarial_regularization:
            # level of attack - estimate of lagrange multiplier for mi constraint
            initial_loa = LOA_INIT
            loa = tf.Variable(initial_loa, dtype=tf.float32, trainable=False)
            loa_gain = mim_constraint - (1.0 - MI_SLACK) * MI_TARGET
            loa_lr = tf.constant(LOA_LR)
            new_loa = loa + loa_lr * loa_gain
            new_loa = tf.maximum(0.0, new_loa)

            if LOA_ADAPTIVE:
                update_loa = tf.assign(loa, new_loa)
                self.update_ops.append(update_loa)

                adversarial_active = tf.stop_gradient(
                    tf.to_float(loa_lr * loa_gain >= -loa)
                )
                adversarial_weighted_loss = adversarial_active * (
                    loa * loa_gain + loa_lr / 2.0 * tf.square(loa_gain)
                )
            else:
                adversarial_weighted_loss = loa * loa_gain

            losses["encoder_0"] += adversarial_weighted_loss

        # use lor
        # OPTION
        variational_regularization = True
        if variational_regularization:
            assert self.model.stochastic_encoder_0

            bottleneck_loss = self.model.z_00_distribution.kl()

            # (log) level of regularization
            initial_lor = LOR_INIT
            lor = tf.Variable(initial_lor, dtype=tf.float32, trainable=False)
            lor_gain = independent_mim_constraint - MI_TARGET
            lor_lr = LOR_LR
            new_lor = tf.clip_by_value(lor + lor_lr * lor_gain, LOR_MIN, LOR_MAX)
            if LOR_ADAPTIVE:
                update_lor = tf.assign(lor, new_lor)
                self.update_ops.append(update_lor)

            bottleneck_weighted_loss = tf.exp(lor) * bottleneck_loss
            losses["encoder_0"] += bottleneck_weighted_loss
        else:
            assert not self.model.stochastic_encoder_0

        if self.model.pretty:
            weight_pretty = self.config.get("weight_pretty", 4e-3)
            pw = weight_pretty * 0.5 * dim
            # pretty
            loss_dis_pretty = 0.5 * (
                logit_loss(self.model.logit_pretty_orig, real=True)
                + logit_loss(self.model.logit_pretty_fake, real=False)
            )
            losses["pretty_discriminator"] = loss_dis_pretty
            self.log_ops["pretty_dis_logit"] = loss_dis_pretty
            # gradient penalty on real data
            gp_pretty = gradient_penalty(
                losses["pretty_discriminator"], [self.model.views[0]]
            )
            self.log_ops["gp_pretty"] = gp_pretty
            gp_weight = 1e2
            losses["pretty_discriminator"] += gp_weight * gp_pretty

            # fakor
            loss_dec_pretty = logit_loss(self.model.logit_pretty_fake, real=True)
            self.log_ops["pretty_dec_logit"] = loss_dec_pretty

            loss_dec_pretty = loss_dec_pretty

            pretty_objective = (
                2.0 * pw * loss_dec_pretty
            )  # backward comp for comparison

            losses["decoder_delta"] += pretty_objective

        # logging
        for k in losses:
            self.log_ops["loss_{}".format(k)] = losses[k]
        self.log_ops["dis0_accuracy"] = dis0_accuracy
        self.log_ops["dis1_accuracy"] = dis1_accuracy
        self.log_ops["avg_dis0_accuracy"] = avg_acc0
        self.log_ops["avg_dis1_accuracy"] = avg_acc1
        self.log_ops["avg_loss_dis0"] = avg_loss_dis0
        self.log_ops["avg_loss_dis1"] = avg_loss_dis1
        self.log_ops["est_accuracy"] = est_accuracy

        self.log_ops["mi_constraint"] = mim_constraint
        self.log_ops["independent_mi_constraint"] = independent_mim_constraint

        if adversarial_regularization:
            self.log_ops["adversarial_weight"] = loa
            self.log_ops["adversarial_constraint"] = mim_constraint
            self.log_ops["adversarial_weighted_loss"] = adversarial_weighted_loss

            self.log_ops["loa"] = loa
            self.log_ops["loa_gain"] = loa_gain

        if variational_regularization:
            self.log_ops["bottleneck_weight"] = lor
            self.log_ops["bottleneck_loss"] = bottleneck_loss
            self.log_ops["bottleneck_weighted_loss"] = bottleneck_weighted_loss

            self.log_ops["lor"] = lor
            self.log_ops["explor"] = tf.exp(lor)
            self.log_ops["lor_gain"] = lor_gain

        visualize_mask(
            "mask0", self.model.mask0, self.model.inputs["view0"], self.img_ops
        )
        visualize_mask(
            "mask_0_logits",
            self.model.mask0_logits,
            self.model.inputs["view0"],
            self.img_ops,
        )
        visualize_mask(
            "mask_0_softmaxed",
            tf.nn.softmax(self.model.mask0_logits, axis=-1),
            self.model.inputs["view0"],
            self.img_ops,
        )
        # visualize_mask("blob_mask1", self.model._view1_final_masks, self.model.inputs["view1"], self.img_ops)
        # visualize_mask("mask1", self.model.mask1, self.model.inputs["view1"], self.img_ops)

        self.img_ops.update(
            {
                "view0": self.model.inputs["view0"],
                "view1": self.model.inputs["view1"],
                "cross": self.model._cross_generated,
                "generated": self.model._generated,
                "mean_beta_rec1": self.model._mean_beta_rec1,
                "debug_mask_reconstruction": self.model._debug_mask_reconstruction,
            }
        )

        return losses
