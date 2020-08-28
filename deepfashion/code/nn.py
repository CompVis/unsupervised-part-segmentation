import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import math

import tensorflow.contrib.distributions as tfd
from tensorflow.contrib.distributions import OneHotCategorical
from deprecated import deprecated


def model_arg_scope(**kwargs):
    """Create new counter and apply arg scope to all arg scoped nn
    operations."""
    counters = {}
    return arg_scope(
        [conv2d, deconv2d, residual_block, dense, activate], counters=counters, **kwargs
    )


def make_model(name, template, unique_name=None, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(
        *args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items())
    )
    if unique_name:
        tf.make_template(name, run, unique_name_=name)
    return tf.make_template(name, run)


def int_shape(x):
    return x.shape.as_list()


def get_name(layer_name, counters):
    """ utlity for keeping track of layer names """
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + "_" + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0.0, maxval=1.0)
    return -tf.log(-tf.log(U + eps) + eps)


def sample_normal(shape):
    return tf.random_normal(shape)


def softmax(x, spatial=False):
    if spatial:
        return spatial_softmax(x)
    else:
        return tf.nn.softmax(x)


def spatial_softmax(features):
    N, H, W, C = features.shape.as_list()
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [N * C, H * W])
    probs = tf.nn.softmax(features)
    # Reshape and transpose back to original format.
    probs = tf.transpose(tf.reshape(probs, [N, C, H, W]), [0, 2, 3, 1])
    return probs


def gumbel_softmax(logits, temperature=1.0, spatial=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    adjusted_logits = gumbel_softmax_sample / temperature
    y = softmax(adjusted_logits, spatial=spatial)
    return y


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


@deprecated(reason="use hard_max and straight_through_estimator independently")
def hard_max_straight_through(y, axis):
    """
    Performs argmax with straight through gradient estimator
    Parameters
    ----------
    x: tensor
        tensor to perform argmax on
    axis: int
        axis on which to perform argmax
    Returns
    -------
    """
    y_hard = hard_max(y, axis)
    y = straight_through_estimator(y_hard, y)
    return y


def hard_max(y, axis):
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis, keep_dims=True)), y.dtype)
    return y_hard


def hard_max_with_segments(x, segments, axis):
    def _pool_unpool(x, segments):
        x_pooled = _pool_with_segments(x, segments, tf.segment_max)
        x_unpooled = _unpool_with_segments(x_pooled, segments)
        return x_unpooled

    x_unpooled, _ = tf.map_fn(
        lambda x: (_pool_unpool(x[0], x[1]), x[1]),
        (x, segments),
        (tf.float32, tf.int32),
    )
    max_map = hard_max(x_unpooled, axis)
    return max_map


def straight_through_estimator(y_hard, y):
    """
    constructs straight-through estimator.
    Parameters
    ----------
    y_hard : tensor
        y_hard graph operation to estimate through
    y : tensor
        actual gradient providing expression

    Returns
    -------
        tf.stop_gradient(y_hard - y) + y
    """
    return tf.stop_gradient(y_hard - y) + y


@add_arg_scope
def partwise_conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """
    input: [b, h, w, parts, features]
    Each part (channel 3) Has is its own bias and scale
    """
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name("pconv2d", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[4])
        in_parts = int(x.get_shape()[3])
        fan_in = in_channels * filter_size[0] * filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        part_stdv = math.sqrt(1.0 / in_parts)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            v_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
            b_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
            v_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
            b_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
        #         elif initdist == "debug":
        #             # initialize part bias and scales as ones
        #             V_initializer = tf.random_normal_initializer(stddev = stdv)
        #             b_initializer = tf.random_normal_initializer(stddev = stdv)
        #             v_part_initializer = tf.ones_initializer()
        #             b_part_initializer = tf.zeros_initializer()
        else:
            raise ValueError(initdist)
        V = tf.get_variable(
            "V",
            filter_size + [in_channels, num_filters],
            initializer=V_initializer,
            dtype=tf.float32,
        )
        b = tf.get_variable(
            "b", [num_filters], initializer=b_initializer, dtype=tf.float32
        )
        V_part = tf.get_variable(
            "V_part", [in_parts], initializer=v_part_initializer, dtype=tf.float32
        )
        b_part = tf.get_variable(
            "b_part", [in_parts], initializer=b_part_initializer, dtype=tf.float32
        )
        if init:
            tmp = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(
                b, [1, 1, 1, num_filters]
            )
            mean, var = tf.nn.moments(tmp, [0, 1, 2])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)

        x = apply_partwise(
            x,
            lambda x_: tf.nn.conv2d(x_, V, [1] + stride + [1], pad)
            + tf.reshape(b, [1, 1, 1, num_filters]),
        )
        x = x * tf.reshape(V_part, (1, 1, 1, in_parts, 1)) + tf.reshape(
            b_part, (1, 1, 1, in_parts, 1)
        )
        return x


def filter_parts(m, include_list, axis=3):
    """
        m = tf.ones((12, 12, 10, 3, 128))
        mm = filter_parts(m, [1, 0, 1])
        mm.shape # ==> 12, 12, 10, 2, 128
    """
    n_parts = m.shape[3]
    assert len(include_list) == n_parts
    part_masks = tf.split(m, n_parts, axis=axis)
    part_masks = [part_masks[i] for i in range(n_parts) if include_list[i]]
    return tf.concat(part_masks, axis=axis)


@deprecated(reason="use take_k_largest_components with k=1 instead")
def take_largest_component(components):
    """
        Filters out the largest not-background components from components.
        If there is no other component than the background, then the background is returned

    Parameters
    ----------
    components : tensor
        [B, H, W, C] tensor of components. Each item along the batch axis and channel axis is a seperate
        label map (map of int values) of components.

    Returns
    """

    return take_k_largest_components(components, 1)


def take_k_largest_components(components, k):
    """
        Filters out the k largest not-background components from components.
        If there is no other component than the background, then the background is returned

        Note that the output tensor is NOT reindexed to have labels from 1 ... k

    Parameters
    ----------
    components : tensor
        [B, H, W, C] tensor of components. Each item along the batch axis and channel axis is a seperate
        label map (map of int values) of components.
    k : int
        how many of the largest components to take

    Returns
    -------

    """
    N, H, W, C = components.shape.as_list()
    components = tf.transpose(components, perm=[0, 3, 1, 2])
    components = tf.reshape(components, shape=(N * C, H, W))
    largest_components = tf.map_fn(
        lambda x: _take_k_largest_components(tf.expand_dims(x, 0), k),
        components,
        dtype=components.dtype,
    )
    largest_components = tf.reshape(largest_components, shape=(N, C, H, W))
    largest_components = tf.transpose(largest_components, perm=[0, 2, 3, 1])
    return largest_components


def _take_k_largest_components(components, k):
    """
    Filteres out the k largest not-background components from components.
    This function only works for a single label map. Use @take_k_largest_components to filter an entire tensor
    of label maps.

    Note that the output tensor is NOT reindexed to have labels from 1 ... k

    Parameters
    ----------
    components : tensor
        [1, H, W] tensor of labels. This is a single label map
    k : int
        max number of components to retorn

    Returns
    -------
    k_largest_components : tensor
        a filtered version of components where only the k-largest components are included
    """
    c_shape = components.shape.as_list()
    if len(c_shape) > 3:
        raise ValueError("components shape has to be 3 dimensional")
    if c_shape[0] != 1:
        raise ValueError("components has to be a single item along the batch axis.")

    uniques, _, counts = tf.unique_with_counts(tf.reshape(components, (-1,)))
    not_background_labels = uniques > 0
    uniques = tf.boolean_mask(uniques, not_background_labels)  # take all but background
    counts = tf.boolean_mask(counts, not_background_labels)
    uniques = tf.gather(uniques, tf.argsort(counts, direction="DESCENDING"))
    uniques = uniques[:k]
    k_largest_components = tf.map_fn(
        lambda x: tf.where(
            tf.equal(components, tf.ones_like(components, dtype=components.dtype) * x),
            components,
            tf.zeros_like(components),
        ),
        uniques,
        dtype=components.dtype,
    )
    k_largest_components = tf.reduce_sum(k_largest_components, axis=0)
    k_largest_components = tf.cast(k_largest_components, tf.int32)
    return k_largest_components


def cca(hard_mask):
    """
    Perfoms connected components analysis on each channel and batch item.
    Parameters
    ----------
    hard_mask : tensor
        tensor of shape [b, h, w, channels] of type bool

    Returns
    -------
    components : tensor
        [b, h, w, features] - shaped tensor of dtype tf.int32 where each connected component has its unique value
    """
    assert hard_mask.dtype == tf.bool
    bs, h, w, n_parts = hard_mask.shape.as_list()
    hard_mask = tf.transpose(hard_mask, perm=[0, 3, 1, 2])
    hard_mask = tf.reshape(hard_mask, shape=(bs * n_parts, h, w))

    padded_mask = tf.pad(hard_mask, [[0, 0], [3, 3], [3, 3]])

    components = tf.map_fn(
        tf.contrib.image.connected_components, padded_mask, dtype=tf.int32
    )
    components = components[:, 3 : (h + 3), 3 : (w + 3)]
    components = tf.expand_dims(components, axis=-1)
    components = tf.reshape(components, shape=(bs, n_parts, h, w))
    components = tf.transpose(components, perm=[0, 2, 3, 1])
    return components


def reshape_like(x, like_x, dynamic=False):
    """

    Parameters
    ----------
    x : tensor
        tensor to reshape
    like_x: tensor
        tensor that provides target shape
    dynamic: bool
        if True, will infer shape at run time (using tf.shape). if False, will use static shape (like_x.shape.as_list())
    Returns
    -------
        reshaped tensor

    """
    if dynamic:
        out = tf.reshape(x, tf.shape(like_x))
    else:
        out = tf.reshape(x, like_x.shape.as_list())
    return out


def reshape_4D(x):
    """ reshapes 5D tensor to 4D tensor"""
    bs, h, w, n_parts, n_features = x.shape.as_list()
    out = tf.reshape(x, [bs, h, w, n_parts * n_features])
    return out


def flatten_12(x):
    """
    shortcut to reshape x from [b, h, w, f] to [b, h * w,f]
    Parameters
    ----------
    x

    Returns
    -------

    """
    b, h, w, p = x.shape.as_list()
    out = tf.reshape(x, [b, h * w, p])
    return out


def sample_components(components, probs):
    components_one_hot = tf.one_hot(
        components, depth=tf.reduce_max(components) + 1, on_value=1, off_value=0, axis=4
    )
    components_one_hot = components_one_hot[:, :, :, :, 1:]  # exclude_background
    components_one_hot = tf.cast(components_one_hot, probs.dtype)
    part_probs = components_one_hot * tf.expand_dims(probs, axis=-1)
    part_probs = tf.reduce_sum(part_probs, axis=(1, 2), keep_dims=True)
    Z = tf.reduce_sum(part_probs, axis=4, keepdims=True)
    part_mass = part_probs / Z
    s = tf.shape(part_mass)  # because size is unknown at graph construction time
    part_mass_flat = tf.reshape(part_mass, [s[0] * s[1] * s[2] * s[3], s[4]])
    part_sample_flat = OneHotCategorical(
        probs=part_mass_flat, dtype=tf.float32
    ).sample()
    part_sample = tf.reshape(part_sample_flat, s) * components_one_hot
    part_sample = tf.reduce_sum(part_sample, axis=(-1))
    return part_sample


def check_float(x):
    if x.dtype is not tf.float32:
        raise ValueError("x has to be float32")


def check_int(x):
    if x.dtype is not tf.int32:
        raise ValueError("x has to be int32")


def incremental_roll(t):
    """
    rolls tensor x along first axis by
    - shift 0 for index 0
    - shift 1 for index 1,
    and so on

    x : tensor
    returns:
        incrementally shifted tensor

    Example
    -------
        a = tf.ones((1, 1, 1, 10, 1))
        b = tf.concat([a, ] + [tf.zeros_like(a)] * 9, axis=0)
        b = tf.transpose(b, perm=[3, 1, 2, 0, 4])
        d = incremental_roll(b)
        e = tf.reshape(d, (10, 1, 1, 10))

        e[0]
        >>> array([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=float32)

        e[5]
        >>> array([[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]], dtype=float32)
    """
    idx = tf.range(t.shape[0], dtype=tf.int32)
    out = tf.map_fn(
        lambda x: (tf.roll(x[0], shift=x[1], axis=2), x[1]),
        (t, idx),
        dtype=(tf.float32, tf.int32),
    )
    return out[0]


def cca_from_logits(logits):
    """
    Calculates hard mask from logits (taking max over channels).
    And then performs CCA on hard mask.

    Essentially
    hard_mask = tf.equal(logits, tf.reduce_max(logits, axis=-1, keep_dims=True))
    return cca(hard_mask)

    Parameters
    ----------
    logits: tensor
        tensor of shape [b, h, w, channels] of any type

    Returns
    -------
        components: tensor
        tensor of dtype tf.int32 where each connected component has its unique value. Shape [b, h, w, features]
    """
    hard_mask = tf.equal(logits, tf.reduce_max(logits, axis=-1, keep_dims=True))
    components = cca(hard_mask)
    return components


@add_arg_scope
def partwise_conv2d_V2(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """
    input: [b, h, w, parts, features]
    Each part (channel 3) has is its own bias and scale
    Uses 3D convolution internally to prevent tf.transpose
    """
    num_filters = int(num_filters)
    name = get_name("conv2d", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = x.shape.as_list()[4]
        in_parts = x.shape.as_list()[3]
        fan_in = in_channels * filter_size[0] * filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        part_stdv = math.sqrt(1.0 / in_parts)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            v_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
            b_part_initializer = tf.random_uniform_initializer(
                minval=-part_stdv, maxval=part_stdv
            )
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
            v_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
            b_part_initializer = tf.random_normal_initializer(stddev=part_stdv)
        elif initdist == "debug":
            pass
        else:
            raise ValueError(initdist)
        if not initdist == "debug":
            V = tf.get_variable(
                "V",
                filter_size + [1, in_channels, num_filters],
                initializer=V_initializer,
                dtype=tf.float32,
            )
            b = tf.get_variable(
                "b",
                [1, 1, 1, 1, num_filters],
                initializer=b_initializer,
                dtype=tf.float32,
            )
            V_part = tf.get_variable(
                "V_part",
                [1, 1, 1, in_parts, 1],
                initializer=v_part_initializer,
                dtype=tf.float32,
            )
            b_part = tf.get_variable(
                "b_part",
                [1, 1, 1, in_parts, 1],
                initializer=b_part_initializer,
                dtype=tf.float32,
            )
        else:
            V = (
                tf.reshape([1.0, 2.0, 1.0], (3, 1, 1, 1, 1))
                / 4
                * tf.reshape([1.0, 2.0, 1.0], (1, 3, 1, 1, 1))
                / 4
                * tf.reshape([1.0, 1.0, 0.0], (1, 1, 1, 3, 1))
                / 2
            )
            b = tf.zeros([1, 1, 1, 1, num_filters], dtype=tf.float32)
            V_part = tf.reshape(
                tf.cast(tf.linspace(0.0, 1.0, in_parts), dtype=tf.float32),
                [1, 1, 1, in_parts, 1],
            )
            b_part = tf.zeros([1, 1, 1, in_parts, 1], dtype=tf.float32)
        x = tf.nn.conv3d(x, V, strides=[1] + stride + [1] + [1], padding="SAME")
        x *= V_part
        x += b + b_part
        return x


def _conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name("conv2d", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels * filter_size[0] * filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable(
            "V",
            filter_size + [in_channels, num_filters],
            initializer=V_initializer,
            dtype=tf.float32,
        )
        b = tf.get_variable(
            "b", [num_filters], initializer=b_initializer, dtype=tf.float32
        )
        if init:
            tmp = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(
                b, [1, 1, 1, num_filters]
            )
            mean, var = tf.nn.moments(tmp, [0, 1, 2])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(
            b, [1, 1, 1, num_filters]
        )
        return x


@add_arg_scope
def conv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    part_wise=False,
    coords=False,
    **kwargs
):
    """
        coords: if True, will use coordConv (2018ACS_liuIntriguingFailingConvolutionalNeuralNetworks)
    """
    if coords:
        x = add_coordinates(x)
    if part_wise:
        out = partwise_conv2d_V2(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    else:
        out = _conv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )

    return out


@add_arg_scope
def dense(x, num_units, init_scale=1.0, counters={}, init=False, **kwargs):
    """ fully connected layer """
    name = get_name("dense", counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
            b_initializer = tf.random_uniform_initializer(minval=-stdv, maxval=stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev=stdv)
            b_initializer = tf.random_normal_initializer(stddev=stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable(
            "V", [in_channels, num_units], initializer=V_initializer, dtype=tf.float32
        )
        b = tf.get_variable(
            "b", [num_units], initializer=b_initializer, dtype=tf.float32
        )
        if init:
            tmp = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
            mean, var = tf.nn.moments(tmp, [0])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
        return x


@add_arg_scope
def activate(x, activation, **kwargs):
    if activation == None:
        return x
    elif activation == "elu":
        return tf.nn.elu(x)
    elif activation == "relu":
        return tf.nn.relu(x)
    elif activation == "leaky_relu":
        return tf.nn.leaky_relu(x)
    else:
        raise NotImplemented(activation)


def mlp_encoder(x, config, activation="relu"):
    """

    Parameters
    ----------
    x: tensor
        [N, C] - shaped tensor
    config: list of ints
        for each layer in the mlp, the width
    activation: str
        see @activate

    Returns
    -------

    """
    with model_arg_scope(activation=activation):
        for c in config[:-1]:
            x = dense(x, c)
            x = activate(x)
        x = dense(x, config[-1])
        # skip final activation - for VAE, this could cause problems otherwise
    return x


def mlp_decoder(x, config, activation="relu"):
    """

    Parameters
    ----------
    x: tensor
        [N, C] - shaped tensor
    config: list of ints
        for each layer in the mlp, the width
    activation: str
        see @activate

    Returns
    -------

    """
    with model_arg_scope(activation=activation):
        for c in config[:-1]:
            x = dense(x, c)
            x = activate(x)
        x = dense(x, config[-1])
        # skip final activation - for VAE, this could cause problems otherwise
    return x


def nin(x, num_units):
    """ a network in network layer (1x1 CONV) """
    return conv2d(x, num_units, filter_size=[1, 1])


def downsample(x, num_units):
    return conv2d(x, num_units, stride=[2, 2])


def upsample(x, num_units, method="subpixel"):
    xs = x.shape.as_list()
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride=[2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4 * num_units)
        x = tf.depth_to_space(x, 2)
        return x
    elif method == "nearest_neighbor":
        bs, h, w, c = x.shape.as_list()
        x = tf.image.resize_images(
            x, [2 * h, 2 * w], tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return x
    elif method == "linear":
        bs, h, w, c = xs[:4]
        if len(xs) == 5:
            x = apply_partwise(
                x,
                lambda x: tf.image.resize_images(
                    x, [2 * h, 2 * w], tf.image.ResizeMethod.BILINEAR
                ),
            )
        else:
            x = tf.image.resize_images(
                x, [2 * h, 2 * w], tf.image.ResizeMethod.BILINEAR
            )
        return x
    else:
        raise NotImplemented(method)


@add_arg_scope
def partwise_deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """ transposed convolutional layer """
    num_filters = int(num_filters)
    name = get_name("deconv2d", counters)
    xs = x.shape.as_list()
    strides = [1] + stride + [1]
    in_parts = xs[3]
    part_stdv = math.sqrt(1.0 / in_parts)
    v_part_initializer = tf.random_uniform_initializer(
        minval=-part_stdv, maxval=part_stdv
    )
    b_part_initializer = tf.random_uniform_initializer(
        minval=-part_stdv, maxval=part_stdv
    )
    if pad == "SAME":
        target_shape = [
            xs[0] * in_parts,
            xs[1] * stride[0],
            xs[2] * stride[1],
            num_filters,
        ]
    else:
        target_shape = [
            xs[0] * in_parts,
            xs[1] * stride[0] + filter_size[0] - 1,
            xs[2] * stride[1] + filter_size[1] - 1,
            num_filters,
        ]
    with tf.variable_scope(name):
        V = tf.get_variable(
            "V",
            filter_size + [num_filters, xs[-1]],
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
        )
        g = tf.get_variable(
            "g",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
        )
        b = tf.get_variable(
            "b",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
        )
        V_part = tf.get_variable(
            "V_part", [in_parts], initializer=v_part_initializer, dtype=tf.float32
        )
        b_part = tf.get_variable(
            "b_part", [in_parts], initializer=b_part_initializer, dtype=tf.float32
        )
        V_norm = tf.nn.l2_normalize(V, [0, 1, 3])

        def part_conv_func(x_):
            x_ = tf.nn.conv2d_transpose(
                x_, V_norm, target_shape, [1] + stride + [1], pad
            )
            x_ = tf.reshape(g, [1, 1, 1, num_filters]) * x_ + tf.reshape(
                b, [1, 1, 1, num_filters]
            )
            return x_

        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = apply_partwise(x, part_conv_func)
        x = x * tf.reshape(V_part, (1, 1, 1, in_parts, 1)) + tf.reshape(
            b_part, (1, 1, 1, in_parts, 1)
        )
        return x


@add_arg_scope
def _deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    **kwargs
):
    """ transposed convolutional layer """
    num_filters = int(num_filters)
    name = get_name("deconv2d", counters)
    xs = x.shape.as_list()
    strides = [1] + stride + [1]
    if pad == "SAME":
        target_shape = [xs[0], xs[1] * stride[0], xs[2] * stride[1], num_filters]
    else:
        target_shape = [
            xs[0],
            xs[1] * stride[0] + filter_size[0] - 1,
            xs[2] * stride[1] + filter_size[1] - 1,
            num_filters,
        ]
    with tf.variable_scope(name):
        V = tf.get_variable(
            "V",
            filter_size + [num_filters, xs[-1]],
            tf.float32,
            tf.random_normal_initializer(0, 0.05),
        )
        g = tf.get_variable(
            "g",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
        )
        b = tf.get_variable(
            "b",
            [num_filters],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
        )

        V_norm = tf.nn.l2_normalize(V, [0, 1, 3])
        x = tf.nn.conv2d_transpose(x, V_norm, target_shape, [1] + stride + [1], pad)
        if init:
            mean, var = tf.nn.moments(x, [0, 1, 2])
            g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
            b = tf.assign(b, -mean * g)
        x = tf.reshape(g, [1, 1, 1, num_filters]) * x + tf.reshape(
            b, [1, 1, 1, num_filters]
        )
        return x


@add_arg_scope
def deconv2d(
    x,
    num_filters,
    filter_size=[3, 3],
    stride=[1, 1],
    pad="SAME",
    init_scale=1.0,
    counters={},
    init=False,
    part_wise=False,
    coords=False,
    **kwargs
):
    """
        coords: if True, will use coordConv (2018ACS_liuIntriguingFailingConvolutionalNeuralNetworks)
    """
    if coords:
        x = add_coordinates(x)
    if part_wise:
        out = partwise_deconv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    else:
        out = _deconv2d(
            x,
            num_filters,
            filter_size=filter_size,
            stride=stride,
            pad=pad,
            init_scale=init_scale,
            counters=counters,
            init=init,
            **kwargs
        )
    return out


@add_arg_scope
def residual_block(x, skipin=None, conv=conv2d, init=False, dropout_p=0.0, **kwargs):
    """Slight variation of original."""
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if skipin is not None:
        skipin = nin(activate(skipin), num_filters)
        residual = tf.concat([residual, skipin], axis=-1)
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob=1.0 - dropout_p)
    residual = conv(residual, num_filters)

    return x + residual


def flatten(x):
    _shape = x.shape.as_list()
    return tf.reshape(x, (_shape[0], -1))


@deprecated("use edflow make var")
def make_linear_var(
    step, start, end, start_value, end_value, clip_min=0.0, clip_max=1.0
):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha)/(b - a) * (x - a) + alpha"""
    linear = (end_value - start_value) / (end - start) * (
        tf.cast(step, tf.float32) - start
    ) + start_value
    return tf.clip_by_value(linear, clip_min, clip_max)


@deprecated("use edflow make var")
def make_staircase_var(
    step, start, start_value, step_size, stair_factor, clip_min=0.0, clip_max=1.0
):
    stair_case = (
        stair_factor ** ((tf.cast(step, tf.float32) - start) // step_size) * start_value
    )
    return tf.clip_by_value(stair_case, clip_min, clip_max)


def split_groups(x, bs=2):
    return tf.split(tf.space_to_depth(x, bs), bs ** 2, axis=3)


def merge_groups(xs, bs=2):
    return tf.depth_to_space(tf.concat(xs, axis=3), bs)


def space_to_batch2(x, cols):
    """split x in into a grid of `cols` * `cols`
    along the spatial dimensions and reshape them to the batch axis

    x : tensor
        [N, H, W, C] - shaped tensor
    cols : int
        number of colums and rows to split the batch into


    Examples
    --------

    import numpy as np
    from skimage import data
    im = data.astronaut()
    image = img_as_float(np.reshape(im, (1, 512, 512, 3)))
    image2 = space_to_batch2(tf.convert_to_tensor(image), 2)
    for i in range(4):
        plt.figure()
        plt.imshow(np.squeeze(image2[i]))
    """
    if len(x.shape.as_list()) != 4:
        raise ValueError("input has to have dimension 4")
    N, H, W, C = x.shape.as_list()
    if H % cols != 0:
        raise ValueError("input dimension 1 has to be divisible by {}".format(cols))
    if W % cols != 0:
        raise ValueError("input dimension 2 has to be divisible by {}".format(cols))
    n1 = N * cols
    n2 = n1 * cols
    h = H // cols
    w = W // cols
    y = tf.reshape(x, (n1, h, W, C))
    y = tf.transpose(y, [0, 2, 1, 3])
    y = tf.reshape(y, (n2, w, h, C))
    y = tf.transpose(y, [0, 2, 1, 3])
    return y


class FullLatentDistribution(object):
    # TODO: write some comment on where this comes from
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()
        if len(ps) != 2:
            assert len(ps) == 4
            assert ps[1] == ps[2] == 1
            self.expand_dims = True
            self.parameters = tf.squeeze(self.parameters, axis=[1, 2])
            ps = self.parameters.shape.as_list()
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        event_dim = self.dim
        n_L_parameters = (event_dim * (event_dim + 1)) // 2

        size_splits = [event_dim, n_L_parameters]

        self.mean, self.L = tf.split(self.parameters, size_splits, axis=1)
        # L is Cholesky parameterization
        self.L = tf.contrib.distributions.fill_triangular(self.L)
        # make sure diagonal entries are positive by parameterizing them
        # logarithmically
        diag_L = tf.linalg.diag_part(self.L)
        self.log_diag_L = diag_L  # keep for later computation of logdet
        diag_L = tf.exp(diag_L)
        # scale down then set diags
        row_weights = np.array([np.sqrt(i + 1) for i in range(event_dim)])
        row_weights = np.reshape(row_weights, [1, event_dim, 1])
        self.L = self.L / row_weights
        self.L = tf.linalg.set_diag(self.L, diag_L)
        self.Sigma = tf.matmul(self.L, self.L, transpose_b=True)  # L times L^t

        ms = self.mean.shape.as_list()
        self.event_axes = list(range(1, len(ms)))
        self.event_shape = ms[1:]
        assert len(self.event_shape) == 1, self.event_shape

    @staticmethod
    def n_parameters(dim):
        return dim + (dim * (dim + 1)) // 2

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * tf.random_normal([self.batch_size, self.dim, 1])
            eps = tf.matmul(self.L, eps)
            eps = tf.squeeze(eps, axis=-1)
            out = self.mean + eps
        if self.expand_dims:
            out = tf.expand_dims(out, axis=1)
            out = tf.expand_dims(out, axis=1)
        return out

    def kl(self, other=None):
        if other is not None:
            raise NotImplemented("Only KL to standard normal is implemented.")

        delta = tf.square(self.mean)
        diag_covar = tf.reduce_sum(tf.square(self.L), axis=2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5 * tf.reduce_sum(
            diag_covar - 1.0 + delta - logdet, axis=self.event_axes
        )
        kl = tf.reduce_mean(kl)
        return kl

    def kl_to_shifted_standard_normal(self, other):
        """KL divergence where other distribution is a standard normal shifted by some offset x. In other words:

        .. math::

            KL(this || other) \\
            this \sim N(\mu, \Sigma) \\
            other \sim N(\mu_{other}, I)

        Parameters
        ----------
        other

        Returns
        -------

        """
        if not other.shape == self.mean.shape:
            raise ValueError(
                "other has to have shape {}".format(self.mean.shape.as_list())
            )

        delta = tf.square(self.mean - other)
        diag_covar = tf.reduce_sum(tf.square(self.L), axis=2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5 * tf.reduce_sum(
            diag_covar - 1.0 + delta - logdet, axis=self.event_axes
        )
        kl = tf.reduce_mean(kl)
        return kl


# class FullLatentPartDistribution(object):
#     def __init__(self, parameters, dim, n_parts, stochastic=True):
#         self.parameters = parameters
#         self.dim = dim
#         self.stochastic = stochastic
#         self.n_parts = n_parts
#         self.delta_dim = self.dim // self.n_parts
#
#         ps = self.parameters.shape.as_list()
#         if len(ps) != 2:
#             assert len(ps) == 4
#             assert ps[1] == ps[2] == 1
#             self.expand_dims = True
#             self.parameters = tf.squeeze(self.parameters, axis=[1, 2])
#             ps = self.parameters.shape.as_list()
#         else:
#             self.expand_dims = False
#
#         assert len(ps) == 2
#         self.batch_size = ps[0]
#
#         event_dim = self.dim
#         n_L_parameters = (event_dim * (event_dim + 1)) // 2
#
#         size_splits = [event_dim, n_L_parameters]
#
#         self.mean, self.L = tf.split(self.parameters, size_splits, axis=1)
#         # L is Cholesky parameterization
#         self.L = tf.contrib.distributions.fill_triangular(self.L)
#         # make sure diagonal entries are positive by parameterizing them
#         # logarithmically
#         diag_L = tf.linalg.diag_part(self.L)
#         self.log_diag_L = diag_L  # keep for later computation of logdet
#         diag_L = tf.exp(diag_L)
#         # scale down then set diags
#         row_weights = np.array([np.sqrt(i + 1) for i in range(event_dim)])
#         row_weights = np.reshape(row_weights, [1, event_dim, 1])
#         self.L = self.L / row_weights
#         self.L = tf.linalg.set_diag(self.L, diag_L)
#         self.Sigma = tf.matmul(self.L, self.L, transpose_b=True)  # L times L^t
#
#         ms = self.mean.shape.as_list()
#         self.event_axes = list(range(1, len(ms)))
#         self.event_shape = ms[1:]
#         assert len(self.event_shape) == 1, self.event_shape
#
#     @staticmethod
#     def n_parameters(dim):
#         return dim + (dim * (dim + 1)) // 2
#
#     def sample_parts(self, noise_level=1.0):
#         if not self.stochastic:
#             out = self.mean
#         else:
#             outs = []
#             for part_i in range(self.n_parts):
#                 noise = noise_level * tf.random_normal(
#                     [self.batch_size, self.delta_dim * (self.n_parts - 1), 1]
#                 )
#                 zeros = tf.zeros([self.batch_size, self.delta_dim, 1])
#                 part_noise = tf.concat([zeros, noise], axis=1)
#                 part_noise = tf.manip.roll(
#                     part_noise, shift=[0, part_i * self.delta_dim, 0], axis=[0, 1, 2]
#                 )
#
#                 eps = tf.matmul(self.L, part_noise)
#                 eps = tf.squeeze(eps, axis=-1)
#                 out = self.mean + eps
#                 outs.append(out)
#             out = tf.stack(outs, axis=1)
#         if self.expand_dims:
#             out = tf.expand_dims(out, axis=1)
#             out = tf.expand_dims(out, axis=1)
#         return out
#
#     def sample(self, noise_level=1.0):
#         if not self.stochastic:
#             out = self.mean
#         else:
#             eps = noise_level * tf.random_normal([self.batch_size, self.dim, 1])
#             eps = tf.matmul(self.L, eps)
#             eps = tf.squeeze(eps, axis=-1)
#             out = self.mean + eps
#         if self.expand_dims:
#             out = tf.expand_dims(out, axis=1)
#             out = tf.expand_dims(out, axis=1)
#         return out
#
#     def kl(self, other=None):
#         if other is not None:
#             raise NotImplemented("Only KL to standard normal is implemented.")
#
#         delta = tf.square(self.mean)
#         diag_covar = tf.reduce_sum(tf.square(self.L), axis=2)
#         logdet = 2.0 * self.log_diag_L
#
#         kl = 0.5 * tf.reduce_sum(
#             diag_covar - 1.0 + delta - logdet, axis=self.event_axes
#         )
#         kl = tf.reduce_mean(kl)
#         return kl


@deprecated("use tf_morphology")
def dilation2d(self, img4D):
    """
    binarry dilation of 2D image
    https://stackoverflow.com/questions/54686895/tensorflow-dilation-behave-differently-than-morphological-dilation
    """
    with tf.variable_scope("dilation2d"):
        kernel = tf.ones((3, 3, img4D.get_shape()[3]))
        output4D = tf.nn.dilation2d(
            img4D,
            filter=kernel,
            strides=(1, 1, 1, 1),
            rates=(1, 1, 1, 1),
            padding="SAME",
        )
        output4D = output4D - tf.ones_like(output4D)

        return output4D


difference1d = np.float32([0.0, 0.5, -0.5])


def fd_kernel(n):
    ffd = np.zeros([3, 3, n, n * 2])
    for i in range(n):
        ffd[1, :, i, 2 * i + 0] = difference1d
        ffd[:, 1, i, 2 * i + 1] = difference1d
    return 0.5 * ffd


def tf_grad(x):
    """Channelwise fd gradient for cell size of one."""
    n = x.shape.as_list()[3]
    kernel = fd_kernel(n)
    grad = tf.nn.conv2d(input=x, filter=kernel, strides=4 * [1], padding="SAME")
    return grad


def tf_squared_grad(x):
    """Pointwise squared L2 norm of gradient assuming cell size of one."""
    s = tf.shape(x)
    gx = tf_grad(x)
    gx = tf.reshape(gx, [s[0], s[1], s[2], s[3], 2])
    return tf.reduce_sum(tf.square(gx), axis=-1)


def mumford_shah(x, alpha, lambda_):
    g = tf_squared_grad(x)
    r = tf.minimum(alpha * g, lambda_)
    return r


def edge_set(x, alpha, lambda_):
    g = tf_squared_grad(x)
    e = g > lambda_ / alpha
    return tf.to_float(e)


class MeanFieldDistribution(object):
    def __init__(self, parameters, dim, stochastic=True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()

        assert len(ps) == 4
        self.batch_size = ps[0]
        self.event_axes = [1, 2, 3]

        event_dim = self.dim
        self.mean = self.parameters
        self.shape = tf.shape(self.mean)

    @staticmethod
    def n_parameters(dim):
        return dim

    def sample(self, noise_level=1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level * tf.random_normal(self.shape)
            out = self.mean + eps
        return out

    def kl(self, other=None):
        # up to constants
        if other is not None:
            raise NotImplemented("Only KL to standard normal is implemented.")
        delta = tf.square(self.mean)
        kl = 0.5 * tf.reduce_sum(delta, axis=self.event_axes)
        kl = tf.reduce_mean(kl)
        return kl

    def kl_improper_gmrf(self):
        # TODO use symmetric stencil
        dy, dx = tf.image.image_gradients(self.mean)
        grad_squared = tf.square(dy) + tf.square(dx)
        kl = 0.5 * grad_squared
        kl = tf.reduce_sum(kl, axis=self.event_axes)
        kl = tf.reduce_mean(kl)
        return kl

    def kl_tv(self):
        tv_distance = tf.image.total_variation(self.mean)
        mean_tv_distance = tf.reduce_mean(tv_distance)
        return mean_tv_distance

    def kl_mumford_sha(self, alpha, lambda_):
        m = mumford_shah(self.mean, alpha, lambda_)
        m = tf.reduce_sum(m, axis=self.event_axes)
        m = tf.reduce_mean(m)
        return m


def kp_probs_to_mu_L_inv(
    kp_probs, scal, inv=True
):  # todo maybe exponential map induces to much certainty ! low values basically ignored and only high values count!
    """
    Calculate mean for each channel of kp_probs
    :param kp_probs: tensor of keypoint probabilites [bn, h, w, n_kp]
    :return: mean calculated on a grid of scale [-1, 1]
    """
    bn, h, w, nk = (
        kp_probs.get_shape().as_list()
    )  # todo instead of calulating sequrity measure from amplitude one could alternativly calculate it by letting the network predict a extra paremeter also one could do
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)

    mu = tf.einsum("ijl,aijk->akl", meshgrid, kp_probs)
    mu_out_prod = tf.einsum(
        "akm,akn->akmn", mu, mu
    )  # todo incosisntent ordereing of mu! compare with cross_V2

    mesh_out_prod = tf.einsum(
        "ijm,ijn->ijmn", meshgrid, meshgrid
    )  # todo efficient (expand_dims)
    stddev = tf.einsum("ijmn,aijk->akmn", mesh_out_prod, kp_probs) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12  # todo clean magic

    a = tf.sqrt(
        a_sq + eps
    )  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = tf.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = tf.zeros_like(a)

    if inv:
        det = tf.expand_dims(tf.expand_dims(a * c, axis=-1), axis=-1)
        row_1 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(c, axis=-1), tf.expand_dims(z, axis=-1)], axis=-1
            ),
            axis=-2,
        )
        row_2 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(-b, axis=-1), tf.expand_dims(a, axis=-1)], axis=-1
            ),
            axis=-2,
        )

        L_inv = (
            scal / (det + eps) * tf.concat([row_1, row_2], axis=-2)
        )  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
        return mu, L_inv
    else:
        row_1 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(a, axis=-1), tf.expand_dims(z, axis=-1)], axis=-1
            ),
            axis=-2,
        )
        row_2 = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(b, axis=-1), tf.expand_dims(c, axis=-1)], axis=-1
            ),
            axis=-2,
        )

        L = scal * tf.concat([row_1, row_2], axis=-2)  # just L
        return mu, L


def probs_to_mu_sigma(probs, scaling_factor):
    """Calculate mean and covariance matrix for each channel of probs
    tensor of keypoint probabilites [bn, h, w, n_kp]
    mean calculated on a grid of scale [-1, 1]

    # TODO: scaling_factor does not work yet. has to be 1 currentlx

    Parameters
    ----------
    probs: tensor
        tensor of shape [b, h, w, k] where each channel along axis 3 is interpreted as an unnormalized probability density.
    scaling_factor : tensor
        tensor of shape [b, 1, 1, k] representing normalizing the normalizing constant of the density

    Returns
    -------
    mu : tensor
        tensor of shape [b, k, 2] representing partwise mean coordinates of x and y for each item in the batch
    sigma : tensor
        tensor of shape [b, k, 2, 2] representing covariance matrix for each item in the batch

    Example
    -------
        norm_const = np.sum(blob, axis=(1, 2), keepdims=True)
        mu, sigma = nn.probs_to_mu_sigma(blob / norm_const, tf.ones_like(norm_const))
    """
    bn, h, w, nk = (
        probs.get_shape().as_list()
    )  # todo instead of calulating sequrity measure from amplitude one could alternativly calculate it by letting the network predict a extra paremeter also one could do
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)

    mu = tf.einsum("ijl,aijk->akl", meshgrid, probs)
    mu_out_prod = tf.einsum(
        "akm,akn->akmn", mu, mu
    )  # todo incosisntent ordereing of mu! compare with cross_V2

    mesh_out_prod = tf.einsum(
        "ijm,ijn->ijmn", meshgrid, meshgrid
    )  # todo efficient (expand_dims)
    stddev = tf.einsum("ijmn,aijk->akmn", mesh_out_prod, probs) - mu_out_prod
    sigma = tf.expand_dims(tf.expand_dims(scaling_factor ** 2, -1), -1) * stddev
    mu *= tf.expand_dims(scaling_factor, -1)
    return mu, sigma


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


def tf_meshgrid(h, w):
    #     xs = np.linspace(-1.0,1.0,w)
    xs = np.arange(0, w)
    ys = np.arange(0, h)
    #     ys = np.linspace(-1.0,1.0,h)
    xs, ys = np.meshgrid(xs, ys)
    meshgrid = np.stack([xs, ys], 2)
    meshgrid = meshgrid.astype(np.float32)
    return meshgrid


def _get_points(m):
    """
    Helper function for @get_points.

    Given a mask 'm', get the corner points of the bounding box enclosing each masks segment.
    This function accepts a single mask and returns the bounding box points.

    Use @get_points to get bounding box points for an entire stack of masks

    Parameters
    ----------
    m : tensor
        A 2D-shaped binary mask [H, W]

    Returns
    -------
    points : tensor
        A 2D-shaped [2, 2] tensor with corner points of bounding box around each segment
    """
    h, w = m.shape.as_list()
    xx, yy = tf.split(tf_meshgrid(h, w), 2, axis=2)
    p00 = tf.reduce_min(
        tf.where(
            tf.equal(
                tf.reshape(m, (h, w, 1)),
                tf.reduce_max(tf.reshape(m, (h, w, 1)), axis=(0, 1)),
            ),
            xx,
            tf.ones_like(xx) * tf.reduce_max(xx),
        )
    )
    p01 = tf.reduce_max(
        tf.where(
            tf.equal(
                tf.reshape(m, (h, w, 1)),
                tf.reduce_max(tf.reshape(m, (h, w, 1)), axis=(0, 1)),
            ),
            xx,
            tf.zeros_like(xx),
        )
    )
    p10 = tf.reduce_min(
        tf.where(
            tf.equal(
                tf.reshape(m, (h, w, 1)),
                tf.reduce_max(tf.reshape(m, (h, w, 1)), axis=(0, 1)),
            ),
            yy,
            tf.ones_like(yy) * tf.reduce_max(yy),
        )
    )
    p11 = tf.reduce_max(
        tf.where(
            tf.equal(
                tf.reshape(m, (h, w, 1)),
                tf.reduce_max(tf.reshape(m, (h, w, 1)), axis=(0, 1)),
            ),
            yy,
            tf.zeros_like(yy),
        )
    )
    return tf.reshape(tf.stack([p00, p01, p10, p11], axis=-1), (2, 2))


def get_points(mask):
    """
    Given a mask 'm', get the corner points of the bounding box enclosing each masks segment.

    Parameters
    ----------
    mask : tensor
        A 4D-shaped [N, H, W, C] tensor where each channel represents a valid binary mask.

    Returns
    -------
    points : tensor
        A 4D-shaped [N, 2, 2, C] tensor with corner points of bounding box around each segment

    Examples
    --------
    h = w = 100
    p = 1
    m = np.zeros((1, h, w, p))
    m[:, 10:20, 10:20] = 1
    m[:, 30:40, 30:40] = 1
    m = tf.concat([np.roll(m, shift=i * 13, axis=(1, 2)) for i in range(3)], axis=3)

    points = get_points(m)
    filled = fill_box(tf.convert_to_tensor(m),
                      tf.cast(tf.convert_to_tensor(points), dtype=tf.float32))
    plt.figure()
    plt.imshow(np.squeeze(m))
    plt.figure()
    plt.imshow(np.squeeze(filled))
    """
    # N, H, W, C = mask.shape.as_list()
    # m_ = tf.transpose(mask, perm=[0, 3, 1, 2])
    # m_ = tf.reshape(m_, (N * C, H, W))
    # points = tf.map_fn(_get_points, m_, tf.float32)
    # points = tf.reshape(points, (N, C, 2, 2))
    # points = tf.transpose(points, perm=[0, 2, 3, 1])
    points = apply_channelwise(_get_points, mask, tf.float32)
    return points


def apply_channelwise(func, t, dtype):
    """Applies function func on each [H, W] slice of the 4D tensor.

    Parameters
    ----------
    func : callable
        function to be called on each [H, W] slice
    t : tensor
        [N, H, W, C]- shaped
    dtype :
        output datatype

    Returns
    -------
    output: tensor
        [N, h, w, C] - shaped output tensor. h, w can be different than H, W

    """
    N, H, W, C = t.shape.as_list()
    t_ = tf.transpose(t, perm=[0, 3, 1, 2])
    t_ = tf.reshape(t_, (N * C, H, W))
    out = tf.map_fn(func, t_, dtype=dtype)
    _, h, w = out.shape.as_list()
    out = tf.reshape(out, (N, C, h, w))
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out


def _fill_box(img, points):
    """
    Creates an image with a white box whose corner points are specified by 'points'.
    Each box is specified by lower left and upper right coordinates.

    Parameters
    ----------
    img : tensor
        [N, H, W, C] image tensor
    points : tensor
        Corner points of box. 4D-shaped with [N, 2, 2, C]. Each batch item and feature channel item
        get its own box.
        Corner points are indexed by
            points[0, 0, 0, 0] = lower left x coordinate
            points[0, 0, 1, 0] = upper right x coordinate
            points[0, 1, 0, 0] = lower left y coordnate
            points[0, 0, 1, 0] = upper right y coordinate

    Returns
    -------
    out : tensor
        Tensor with squares

    Examples
    --------
    h = w = 100
    p = 1
    m = np.zeros((1, h, w, p))
    m[:, 10:20, 10:20] = 1
    m[:, 30:40, 30:40] = 1
    m = tf.concat([np.roll(m, shift=i * 13, axis=(1, 2)) for i in range(3)], axis=3)

    points = get_points(m)
    filled = fill_box(tf.convert_to_tensor(m),
                      tf.cast(tf.convert_to_tensor(points), dtype=tf.float32))
    plt.figure()
    plt.imshow(np.squeeze(m))
    plt.figure()
    plt.imshow(np.squeeze(filled))
    """
    # TODO: use tf.slice rather than tf.where
    # TODO: remove dependency on image. We actually only need pts and h, w
    h, w = img.shape.as_list()
    xx, yy = tf.split(tf_meshgrid(h, w), 2, axis=2)
    xx = tf.squeeze(xx)
    yy = tf.squeeze(yy)
    out = tf.where(
        xx > points[0, 0],
        tf.ones_like(img, dtype=img.dtype),
        tf.zeros_like(img, dtype=img.dtype),
    )
    out = (
        tf.where(
            xx < points[0, 1],
            tf.ones_like(img, dtype=img.dtype),
            tf.zeros_like(img, dtype=img.dtype),
        )
        * out
    )
    out = (
        tf.where(
            yy > points[1, 0],
            tf.ones_like(img, dtype=img.dtype),
            tf.zeros_like(img, dtype=img.dtype),
        )
        * out
    )
    out = (
        tf.where(
            yy < points[1, 1],
            tf.ones_like(img, dtype=img.dtype),
            tf.zeros_like(img, dtype=img.dtype),
        )
        * out
    )
    return out


def fill_box(img, points):
    N, h, w, C = points.shape.as_list()
    N, H, W, C = img.shape.as_list()
    points = tf.transpose(points, perm=[0, 3, 1, 2])
    img = tf.transpose(img, perm=[0, 3, 1, 2])
    points = tf.reshape(points, (N * C, h, w))
    img = tf.reshape(img, (N * C, H, W))
    out = tf.map_fn(lambda x: _fill_box(x[0], x[1]), (img, points), dtype=tf.float32)
    _, h, w = out.shape.as_list()
    out = tf.reshape(out, (N, C, H, W))
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out


def localize(x, softmax=False, centered=False):
    if softmax:
        x = tf.nn.softmax(x, axis=3)
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
    # yes, the order is correct here
    return (mean_u, mean_v), (var_v, var_u)


def tf_hm3(h, w, mu, L, order="exp"):
    """
        Returns Gaussian densitiy function based on μ and L for each batch index and part
        L is the cholesky decomposition of the covariance matrix : Σ = L L^T
    Parameters
    ----------
    h : int
        heigh ot output map
    w : int
        width of output map
    mu : tensor
        mean of gaussian part and batch item. Shape [b, p, 2]. Mean in range [-1, 1] with respect to height and width
    L : tensor
        cholesky decomposition of covariance matrix for each batch item and part. Shape [b, p, 2, 2]
    order

    Returns
    -------
    density : tensor
        gaussian blob for each part and batch idx. Shape [b, h, w, p]

    """

    assert len(mu.get_shape().as_list()) == 3
    assert len(L.get_shape().as_list()) == 4
    assert mu.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-1] == 2
    assert L.get_shape().as_list()[-2] == 2

    b, p, _ = mu.get_shape().as_list()
    mu = tf.reshape(mu, (b * p, 2))
    L = tf.reshape(L, (b * p, 2, 2))

    mvn = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
    y_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tf.tile(tf.reshape(tf.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    y_t = tf.expand_dims(y_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=-1)
    meshgrid = tf.concat([y_t, x_t], axis=-1)
    meshgrid = tf.expand_dims(meshgrid, 0)
    meshgrid = tf.expand_dims(meshgrid, 3)  # 1, h, w, 1, 2

    probs = mvn.prob(meshgrid)
    probs = tf.reshape(probs, (h, w, b, p))
    probs = tf.transpose(probs, perm=[2, 0, 1, 3])  # move part axis to the back
    return probs


import matplotlib as mpl
from matplotlib import pyplot as plt


def mask2rgb(mask, make_hot=True):
    n_parts = mask.shape.as_list()[3]
    if make_hot:
        hotmask = mask2hotmask(mask, n_parts)
    else:
        hotmask = mask
    hotmask = tf.expand_dims(hotmask, 4)
    # colors = plt.cm.inferno(np.linspace(0, 1, n_parts), alpha=False, bytes=False)[:, :3]
    colors = make_mask_colors(n_parts)
    colors = (colors - 0.5) * 2
    colors = tf.to_float(colors)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    colors = tf.expand_dims(colors, 0)
    vis_mask = hotmask * colors
    vis_mask = tf.reduce_sum(vis_mask, axis=3)
    return vis_mask


def mask2hotmask(mask, n_parts):
    maxmask = tf.argmax(mask, axis=3)
    hotmask = tf.one_hot(maxmask, depth=n_parts)
    return hotmask


def np_one_hot(targets, depth):
    res = np.eye(depth)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [depth])


def np_to_float(x):
    return x.astype(np.float32)


def np_mask2rgb(mask):
    n_parts = mask.shape[3]
    maxmask = np.argmax(mask, axis=3)
    hotmask = np_one_hot(maxmask, depth=n_parts)
    hotmask = np.expand_dims(hotmask, 4)
    # colors = plt.cm.inferno(np.linspace(0, 1, n_parts), alpha=False, bytes=False)[:, :3]
    colors = make_mask_colors(n_parts)
    colors = (colors - 0.5) * 2
    colors = np_to_float(colors)
    colors = np.expand_dims(colors, 0)
    colors = np.expand_dims(colors, 0)
    colors = np.expand_dims(colors, 0)
    vis_mask = hotmask * colors
    vis_mask = np.sum(vis_mask, axis=3)
    return vis_mask


def make_mask_colors(n_parts):
    colors = plt.cm.inferno(np.linspace(0, 1, n_parts), alpha=False, bytes=False)[:, :3]
    return colors


def add_coordinates(input_tensor, with_r=False):
    assert len(input_tensor.shape.as_list()) == 4
    bs = tf.shape(input_tensor)[0]
    x_dim, y_dim = input_tensor.shape.as_list()[1:3]
    # assert x_dim > 1, x_dim
    # assert y_dim > 1, y_dim
    xx_ones = tf.ones([bs, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [bs, 1])
    xx_range = tf.expand_dims(xx_range, 1)

    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([bs, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [bs, 1])
    yy_range = tf.expand_dims(yy_range, -1)

    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel = tf.cast(xx_channel, "float32") / max(1, x_dim - 1)
    yy_channel = tf.cast(yy_channel, "float32") / max(1, y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
    if with_r:
        rr = tf.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
        ret = tf.concat([ret, rr], axis=-1)
    return ret


def sample_normal_part_noise(m0_hard):
    """sample part noise from normal distribution

    Parameters
    ----------
    m0_hard: tensor
        [N, H, W, C] - shaped tensor

    Returns
    -------

    """
    N, H, W, C = m0_hard.shape.as_list()
    part_noise_shape = [N, 1, 1, C, C]
    gumbel_noise = tf.random_normal(part_noise_shape)
    part_noise = tf.expand_dims(m0_hard, 4) * gumbel_noise
    part_noise = tf.reduce_sum(part_noise, axis=3)
    return part_noise


def sample_gumbel_part_noise(m0_hard):
    """sample part noise from gumbel distribution

    Parameters
    ----------
    m0_hard: tensor
        [N, H, W, C] - shaped tensor

    Returns
    -------

    """
    N, H, W, C = m0_hard.shape.as_list()
    part_noise_shape = [N, 1, 1, C, C]
    gumbel_noise = sample_gumbel(part_noise_shape)
    part_noise = tf.expand_dims(m0_hard, 4) * gumbel_noise
    part_noise = tf.reduce_sum(part_noise, axis=3)
    return part_noise


def take_only_first_item_in_the_batch(x):
    """returns slice with only first item along axis 0

    Parameters
    ----------
    x: tensor
        ND-tensor

    Returns
    -------
    first_batch_item: tensor
        x[0, ...] with same number of dimensions

    Examples
    --------

        t = tf.constant(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]
        ) # shape [2, 2, 3
        first_item = nn.take_only_first_item_in_the_batch(t)
        assert [1, 2, 3] == first_item.shape.as_list()
    """
    x_shape = x.shape.as_list()
    first_batch_item = tf.slice(x, [0] * len(x_shape), [1] + x_shape[1:])
    return first_batch_item


def _mean_pool_with_segments(x, segments):
    """mean-pool x within superpixel segments

    Parameters
    ----------
    x: tensor
        [H, W, C] - shaped tensor representing a single item to be pooled
    segments: tensor
        [H, W, 1] - shaped tensor representing the superpixel segment for the whole tensor x

    Returns
    -------
    x_pooled : tensor
        pooled version of x

    Raises
    ------
        ValueError if x and superpixel_segments do not have the same spatial layout
        superpixel_segments has to have only 1 channel

    Examples
    --------

        img = img_as_float(astronaut()[::2, ::2])
        data = tf.convert_to_tensor(img)
        segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

        tf_segments = tf.convert_to_tensor(segments_quick)
        pooled = _superpixel_pooling(data, tf_segments)
        plt.imshow(np.squeeze(pooled))
    """
    H, W, C = x.shape.as_list()
    h, w, c = segments.shape.as_list()
    is_some_spatial_shape = h == H and w == W
    if not is_some_spatial_shape:
        raise ValueError("x and superpixel_segments have to have same spatial layout")
    if not c == 1:
        raise ValueError("superpixel_segments has to have only 1 channel")
    segments = tf.reshape(segments, (H, W))
    x = tf.reshape(x, (H * W, C))
    flat_segments = tf.reshape(segments, (H * W,))
    sorting_permutation = tf.argsort(flat_segments, axis=0)
    x = tf.gather(x, sorting_permutation)
    sorted_segments = tf.gather(flat_segments, sorting_permutation)
    mean_pooled = tf.segment_mean(x, sorted_segments)
    return mean_pooled


def _pool_with_segments(x, segments, reduction_function=tf.segment_mean):
    H, W, C = x.shape.as_list()
    h, w, c = segments.shape.as_list()
    is_some_spatial_shape = h == H and w == W
    if not is_some_spatial_shape:
        raise ValueError("x and superpixel_segments have to have same spatial layout")
    if not c == 1:
        raise ValueError("superpixel_segments has to have only 1 channel")
    segments = tf.reshape(segments, (H, W))
    x = tf.reshape(x, (H * W, C))
    flat_segments = tf.reshape(segments, (H * W,))
    sorting_permutation = tf.argsort(flat_segments, axis=0)
    x = tf.gather(x, sorting_permutation)
    sorted_segments = tf.gather(flat_segments, sorting_permutation)
    pooled = reduction_function(x, sorted_segments)
    return pooled


def _unpool_with_segments(x_pooled, segments):
    h, w, c = segments.shape.as_list()
    segments = tf.reshape(segments, (h, w))
    x_pooled = tf.gather(x_pooled, segments)
    return x_pooled


def pool_unpool_with_segments(x, segments, reduction_function=tf.segment_mean):
    def _pool_unpool(x, segments, reduction_function):
        pooled = _pool_with_segments(x, segments, reduction_function)
        unpooled = _unpool_with_segments(pooled, segments)
        return unpooled

    pooled, _ = tf.map_fn(
        lambda x: (_pool_unpool(x[0], x[1], reduction_function), x[1]),
        (x, segments),
        (tf.float32, tf.int32),
    )
    return pooled


def _sample_from_segments(x, segments, sampler=sample_gumbel):
    H, W, C = x.shape.as_list()
    h, w, c = segments.shape.as_list()
    is_some_spatial_shape = h == H and w == W
    if not is_some_spatial_shape:
        raise ValueError("x and superpixel_segments have to have same spatial layout")
    if not c == 1:
        raise ValueError("superpixel_segments has to have only 1 channel")
    segments = tf.reshape(segments, (H, W))
    x = tf.reshape(x, (H * W, C))
    flat_segments = tf.reshape(segments, (H * W,))
    sorting_permutation = tf.argsort(flat_segments, axis=0)
    x = tf.gather(x, sorting_permutation)
    sorted_segments = tf.gather(flat_segments, sorting_permutation)
    mean_pooled = tf.segment_mean(x, sorted_segments)
    noise = sample_gumbel(tf.shape(mean_pooled))  # tf shape for dynamic shape
    mean_pooled += noise
    x_pooled = tf.gather(mean_pooled, segments)
    return x_pooled


def sample_from_segments(x, segments, sampler=sample_gumbel):
    pooled, _ = tf.map_fn(
        lambda x: (_sample_from_segments(x[0], x[1], sampler), x[1]),
        (x, segments),
        (tf.float32, tf.int32),
    )
    return pooled


@deprecated("use @filter_with_segments")
def mean_pool_with_segments(x, segments):
    """mean-pool x within segments

    Parameters
    ----------
    x: tensor
        [N, H, W, C] - shaped tensor of items to be pooled
    segments: tensor
        [N, H, W, 1] - shaped tensor of superpixel segments
        
    Returns
    -------

    Examples
    --------

        img = img_as_float(astronaut()[::2, ::2])
        img2 = img_as_float(camera()[::2, ::2])
        img2 = resize(img2, img.shape)
        data = np.stack([img, img2], axis=0)
        data = tf.convert_to_tensor(data)
        data = tf.to_float(data)

        segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
        segments_quick2 = quickshift(img2, kernel_size=3, max_dist=6, ratio=0.5)
        superpixel_segments = tf.stack([segments_quick, segments_quick2], axis=0)
        superpixel_segments = tf.expand_dims(superpixel_segments, axis=-1)

        pooled = mean_pool_with_segments(x, superpixel_segments)
    """

    def mean_unpool_with_segments(x, segments):
        pooled = _mean_pool_with_segments(x, segments)
        return _unpool_with_segments(pooled, segments)

    pooled, _ = tf.map_fn(
        lambda x: (mean_unpool_with_segments(x[0], x[1]), x[1]),
        (x, segments),
        (tf.float32, tf.int32),
    )
    return pooled


def mean_pool_unpool_with_segments(x, segments):
    """mean-pool x within segments

    Parameters
    ----------
    x: tensor
        [N, H, W, C] - shaped tensor of items to be pooled
    segments: tensor
        [N, H, W, 1] - shaped tensor of superpixel segments

    Returns
    -------

    Examples
    --------

        img = img_as_float(astronaut()[::2, ::2])
        img2 = img_as_float(camera()[::2, ::2])
        img2 = resize(img2, img.shape)
        data = np.stack([img, img2], axis=0)
        data = tf.convert_to_tensor(data)
        data = tf.to_float(data)

        segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
        segments_quick2 = quickshift(img2, kernel_size=3, max_dist=6, ratio=0.5)
        superpixel_segments = tf.stack([segments_quick, segments_quick2], axis=0)
        superpixel_segments = tf.expand_dims(superpixel_segments, axis=-1)

        pooled = mean_pool_with_segments(x, superpixel_segments)
    """

    def mean_unpool_with_segments(x, segments):
        pooled = _mean_pool_with_segments(x, segments)
        return _unpool_with_segments(pooled, segments)

    pooled, _ = tf.map_fn(
        lambda x: (mean_unpool_with_segments(x[0], x[1]), x[1]),
        (x, segments),
        (tf.float32, tf.int32),
    )
    return pooled


def reduce_sum_with_segments(x, segments):
    def _reduce_sum(x, segments):
        pooled = _mean_pool_with_segments(x, segments)
        return tf.reduce_sum(pooled)

    pooled, _ = tf.map_fn(
        lambda x: (_reduce_sum(x[0], x[1]), x[1]), (x, segments), (tf.float32, tf.int32)
    )
    return pooled


def reduce_mean_with_segments(x, segments):
    def _reduce(x, segments):
        pooled = _mean_pool_with_segments(x, segments)
        return tf.reduce_mean(pooled)

    pooled, _ = tf.map_fn(
        lambda x: (_reduce(x[0], x[1]), x[1]), (x, segments), (tf.float32, tf.int32)
    )
    return pooled


def tf_mark_boundaries(img, segments):
    """similar to mark_boundaries from skimage,
    marks boundaries in an image from labeled segments provided

    img: tensor
        [N, H, W, 3]- shaped tensor of a batch of rgb images
    segments : [N, H, W, 1] - shaped tensor of a batch of labelled segments
    """
    N, H, W, C = img.shape.as_list()
    is_image_rgb = C == 3
    if not is_image_rgb:
        raise ValueError("image is not an rgb image")
    segments_gradient_sq = tf_squared_grad(tf.to_float(segments))
    boundaries = segments_gradient_sq > 0
    r, g, b = tf.split(img, 3, axis=3)
    g = tf.where(boundaries, tf.ones_like(g, g.dtype), g)
    image_marked = tf.concat([r, g, b], axis=3)
    return image_marked
