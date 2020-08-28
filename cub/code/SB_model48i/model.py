from edflow.iterators.tf_trainer import TFBaseTrainer
from edflow.iterators.tf_trainer import TFBaseTrainer
import edflow.applications.tf_perceptual_loss as deeploss
from edflow.tf_util import make_linear_var
from eddata.utils import tps
import tfutils

import nips19.nn as nn

dsize = 512

PARTS_DIM = 3
FEATURE_DIM = 4

import tensorflow as tf
import numpy as np
from edflow.iterators import tf_batches
from edflow.tf_util import make_var


def categorical_kl(probs):
    k = tf.to_float(probs.shape.as_list()[-1])
    logkp = tf.log(k * probs + 1e-20)
    kl = tf.reduce_sum(probs * logkp, axis=-1)
    return tf.reduce_mean(kl)


def make_ema(init_value, value, update_ops, decay=0.99):
    decay = tf.constant(decay, dtype=tf.float32)
    avg_value = tf.Variable(init_value, dtype=tf.float32, trainable=False)
    update_ema = tf.assign(
        avg_value, decay * avg_value + (1.0 - decay) * tf.cast(value, tf.float32)
    )
    update_ops.append(update_ema)
    return avg_value


def encoder_model(x, out_size, config, extra_resnets, activation="relu", coords=False):
    with nn.model_arg_scope(activation=activation, coords=coords):
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


def decoder_model(h1, h2, config, extra_resnets, activation="relu", coords=False):
    with nn.model_arg_scope(activation=activation, coords=coords):
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


def hourglass_model(
    x,
    config,
    extra_resnets,
    alpha=None,
    pi=None,
    n_out=3,
    activation="relu",
    upsample_method="subpixel",
    coords=False,
):
    alpha = None
    pi = None
    with nn.model_arg_scope(activation=activation, coords=coords):
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
    h, n_out=3, config=None, activation="relu", upsample_config="subpixel", coords=False
):
    if type(upsample_config) is str:
        # convert string to list of strings, for each upsampling block
        # upsample config is 1 shorter than config
        upsample_config = [upsample_config] * (len(config) - 1)
    assert len(upsample_config) == (len(config) - 1)
    with nn.model_arg_scope(activation=activation, coords=coords):
        h = nn.nin(h, 4 * 4 * config[-1])
        h = tf.reshape(h, [-1, 4, 4, config[-1]])

        h = nn.conv2d(h, config[-1])
        h = nn.residual_block(h)

        for nf, u_method in zip(config[-2::-1], upsample_config[-1::-1]):
            h = nn.residual_block(h)
            h = nn.upsample(h, nf, method=u_method)

        h = nn.residual_block(h)
        h = nn.conv2d(h, n_out)

        return h


def discriminator_model(pair, activation="relu", coords=False):
    with nn.model_arg_scope(activation=activation, coords=coords):
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


def mask_parts(image, mask):
    """ [B, H, W, 3], [B, H, W, parts] --> [B, H, W, parts, 3] """
    bs, h, w, n_fetures = image.shape.as_list()
    mshape = mask.shape.as_list()
    assert mshape[0] == bs and mshape[1] == h and mshape[2] == w, mshape
    n_parts = mshape[3]

    image = tf.expand_dims(image, PARTS_DIM)
    mask = tf.expand_dims(mask, FEATURE_DIM)
    masked_image = image * mask

    return masked_image


def image_discriminator_model(x, config=None, activation="relu", coords=False):
    """
    returns props, logits
    """
    with nn.model_arg_scope(activation=activation, coords=coords):
        hs = list()
        h = nn.conv2d(x, config[0])
        hs.append(h)

        for nf in config[1:]:
            h = nn.downsample(h, nf)
            h = nn.residual_block(h)
            hs.append(h)

        h = nn.activate(h)
        h = nn.conv2d(h, config[-1])

        h = tf.reduce_mean(h, [1, 2], keepdims=True)
        h = nn.nin(h, 1)
        h = tf.reduce_mean(h, [1, 2, 3])
        h = tf.expand_dims(h, -1)
        return tf.nn.sigmoid(h), h


def encode_parts(part_image, encoder):
    """ [B, H, W, parts, 3] --> [B, parts, features] """
    b, h, w, parts, channels = part_image.shape.as_list()
    part_encodings = nn.apply_partwise(part_image, encoder)

    part_encodings = tf.reshape(part_encodings, (b, parts, -1))
    out_shape = part_encodings.shape.as_list()
    assert out_shape[0] == b and out_shape[1] == parts
    return part_encodings


def unpool_features(feature_vectors, mask):
    """
    feature_vectors : [B, parts, features]
    mask : [B, h, w, parts]
    output : [b, h, w, n_parts, features]
    """
    bs, h, w, n_parts = mask.shape.as_list()
    fshape = feature_vectors.shape.as_list()
    # assert len(fshape) == 3, fshape
    # assert fshape[0] == bs and fshape[1] == n_parts, fshape
    n_features = fshape[2]

    mask = tf.expand_dims(mask, 4)
    feature_map = feature_vectors
    feature_map = tf.expand_dims(feature_map, 1)
    feature_map = tf.expand_dims(feature_map, 2)

    perm = [3, 0, 1, 2, 4]
    inv_perm = [1, 2, 3, 0, 4]
    elems = (tf.transpose(mask, perm=perm), tf.transpose(feature_map, perm=perm))
    output = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.float32)
    output = tf.transpose(output, perm=inv_perm)

    out_shape = output.shape.as_list()
    return output

class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.pretty = self.config.get("use_pretty", False)
        variables = set(tf.global_variables())
        self.n_parts = self.config.get("n_parts")
        self.define_graph()
        self.variables = [v for v in tf.global_variables() if not v in variables]

    @property
    def inputs(self):
        _inputs = {"view0": self.views[0], "view1": self.views[1], "view0_target": self.views[2]}
        return _inputs

    @property
    def outputs(self):
        _outputs = {"generated": self._generated}
        _outputs.update({"view0_mask00_rgb": nn.mask2rgb(self.m0)})
        _outputs.update({"m0_sample": self.m0_sample})
        _outputs.update({"out_parts_hard": self.out_parts_hard})
        _outputs.update({"out_parts_soft": self.out_parts_soft})
        if self.use_tps:
            _outputs.update(
                {
                    "tps_view0": self.augmented_views[0],
                    "tps_view1": self.augmented_views[1],
                    "tps_view0_target": self.augmented_views[2],
                }
            )
        return _outputs

    def make_tps(self, views, tps_parameters):
        """make tps on views

        Parameters
        ----------
        views : list of tf.tensor
            list of 3 image sensors shaped [N, H, W, C].
        tps_parameters: dict:
            tps transformation args

        Returns
        -------

        """
        ## view0 and view1 augmentation
        bs = views[0].shape.as_list()[0]
        img_batch = tf.concat(views[:2], axis=0)
        bs_doubled = img_batch.shape.as_list()[0]
        tps_param_dict = tps.tps_parameters(bs_doubled, **tps_parameters)
        coord, vector = tps.make_input_tps_param(tps_param_dict)
        t_images, t_mesh = tps.ThinPlateSpline(
            img_batch, coord, vector, img_batch.shape[1], img_batch.shape[-1]
        )
        augmented_views = tf.split(t_images, 2, axis=0) # [view0 and view1]

        target_coord = tf.split(coord, 2, axis=0)[0]
        target_vector = tf.split(vector, 2, axis=0)[0]
        t_images, t_mesh = tps.ThinPlateSpline(views[2], target_coord, target_vector, views[2].shape[1], views[2].shape[-1])
        augmented_views.append(t_images)
        return augmented_views

    def define_graph(self):
        # input placeholders
        self.views = list()
        for view_idx in range(3):
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
        self.augmented_views = self.views
        self.sample_gumbel = self.config.get("sample_gumbel", True)
        local_app_size = self.config.get("local_app_size", 64)
        batch_size = self.config["batch_size"]
        self.patch_size = self.config.get("patch_size", 32)

        self.use_tps = self.config.get("use_tps", False)
        if self.use_tps:
            tps_parameters = self.config.get("tps_parameters")
            self.augmented_views = self.make_tps(self.views, tps_parameters)

        z0_size = self.config.get("z0_size", 256)
        self.pi_size = z0_size

        z0_n_parameters = nn.FullLatentDistribution.n_parameters(z0_size)
        n_parts = self.n_parts
        l_n_parameters = nn.MeanFieldDistribution.n_parameters(n_parts)

        self.spatial_softmax = False

        # submodules
        encoder_kwargs = self.config.get("encoder0")
        e_pi = nn.make_model(
            "encoder_0", encoder_model, out_size=z0_n_parameters, **encoder_kwargs
        )
        self.e_pi = e_pi

        dd_kwargs = self.config.get("final_hour")
        dd = nn.make_model("decoder_delta", hourglass_model, **dd_kwargs)
        self.dd = dd

        app_extractor_kwargs = self.config.get("encoder1")
        e_alpha = nn.make_model(
            "encoder_1", encoder_model, out_size=local_app_size, **app_extractor_kwargs
        )
        self.e_alpha = e_alpha

        dv_kwargs = self.config.get("dv")
        dv = nn.make_model(
            "decoder_visualize", single_decoder_model, n_out=n_parts, **dv_kwargs
        )
        self.dv = dv

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

        # pose
        self.stochastic_encoder_0 = not self.config.get("test_mode", False)
        pi_parameters_v0 = e_pi(self.augmented_views[0])
        pi_distribution_v0 = self.z_00_distribution = nn.FullLatentDistribution(
            pi_parameters_v0, z0_size, stochastic=self.stochastic_encoder_0
        )

        pi_parameters_v1 = e_pi(self.augmented_views[1])
        pi_distribution_v1 = self.z_01_distribution = nn.FullLatentDistribution(
            pi_parameters_v1, z0_size, stochastic=self.stochastic_encoder_0
        )

        # appearance
        alpha_v0 = self.alpha_v0 = e_alpha(self.augmented_views[0])
        alpha_v1 = self.alpha_v1 = e_alpha(self.augmented_views[1])
        z_1_independent = self.z_1_independent = tf.reverse(alpha_v0, [0])

        # masks
        if (
            "stochastic_l" in self.config.keys()
        ):  # this overwrites the assignments below
            self.stochastic_l = self.config.get("stochastic_l")
        else:
            self.stochastic_l = not self.config.get("test_mode", False)
        pi_sample_v0 = pi_distribution_v0.sample()
        pi_sample_v1 = pi_distribution_v1.sample()
        self.pi_sample_v0 = pi_sample_v0
        self.pi_sample_v1 = pi_sample_v1

        self.l0_parameters = dv(pi_sample_v0)
        self.l1_parameters = dv(pi_sample_v1)

        self.l0_distribution = nn.MeanFieldDistribution(
            self.l0_parameters, l_n_parameters, stochastic=self.stochastic_l
        )
        self.l1_distribution = nn.MeanFieldDistribution(
            self.l1_parameters, l_n_parameters, stochastic=self.stochastic_l
        )
        l0 = self.l0_distribution.sample()
        l1 = self.l1_distribution.sample()

        self.m0_logits = l0
        self.m1_logits = l1

        m0 = nn.softmax(self.m0_logits, spatial=False)
        m0_sample = nn.softmax(self.m0_logits, spatial=False)

        m1 = nn.softmax(self.m1_logits, spatial=False)
        m1_sample = nn.softmax(self.m1_logits, spatial=False)

        self.m0_sample = m0_sample
        self.m0 = m0
        self.m0_sample_hard = nn.straight_through_estimator(
            nn.hard_max(self.m0_sample, 3), self.m0_sample
        )
        N, h, w, P = self.m0_sample_hard.shape.as_list()
        gamma = self.config.get("gamma", 3.0)
        corrected_logits = nn.softmax(self.m0_sample_hard * gamma, spatial=True)
        mu, _ = nn.probs_to_mu_sigma(corrected_logits, tf.ones((N, P)))
        mu = tf.stop_gradient(tf.cast(tf.reshape(mu, (N * P, 2)) * h / 2.0 + h / 2.0, tf.int32))
        m0_sample_hard_mask = tfutils.draw_rect(mu, self.patch_size, self.patch_size, [h, w, 1])
        m0_sample_hard_mask = tf.reshape(m0_sample_hard_mask, (N, P, h, w))
        m0_sample_hard_mask = tf.transpose(m0_sample_hard_mask, perm=[0, 2, 3, 1])
        self.m0_sample_hard_mask = m0_sample_hard_mask

        self.m0_sample_argmax = tf.argmax(self.m0_sample, axis=3)
        self.mask0_decode = m0_sample * tf.stop_gradient(m0_sample_hard_mask)
        self.mask0_gradients = self.m0_logits

        self.m1_sample = m1_sample
        self.m1 = m1
        self.m1_sample_hard = nn.straight_through_estimator(
            nn.hard_max(self.m1_sample, 3), self.m1_sample
        )
        N, h, w, P = self.m1_sample_hard.shape.as_list()
        corrected_logits = nn.softmax(self.m1_sample_hard * gamma, spatial=True) 
        mu, _ = nn.probs_to_mu_sigma(corrected_logits, tf.ones((N, P)))
        mu = tf.stop_gradient(tf.cast(tf.reshape(mu, (N * P, 2)) * h / 2.0 + h / 2.0, tf.int32))
        m1_sample_hard_mask = tfutils.draw_rect(mu, self.patch_size, self.patch_size, (h, w, 1))
        m1_sample_hard_mask = tf.reshape(m1_sample_hard_mask, (N, P, h, w))
        m1_sample_hard_mask = tf.transpose(m1_sample_hard_mask, perm=[0, 2, 3, 1])
        self.m1_sample_hard_mask = m1_sample_hard_mask        

        self.m1_sample_argmax = tf.argmax(self.m1_sample, axis=3)
        self.mask1_encode = m1_sample * tf.stop_gradient(m1_sample_hard_mask)
        self.mask1_gradients = self.m1_logits

        self.out_parts_soft = nn.softmax(self.l0_distribution.mean, False)
        self.out_parts_hard = tf.argmax(self.out_parts_soft, 3)

        self.encoding_mask = self.m1_sample_hard # * tf.stop_gradient(m1_sample_hard_mask)
        self.decoding_mask = self.m0_sample_hard # * tf.stop_gradient(m0_sample_hard_mask)
        # register gradients
        # get assignment for each part

        # encoding and decoding with assignmentj
        view1_parts = mask_parts(self.augmented_views[1], self.encoding_mask)
        local_app_features1 = encode_parts(view1_parts, e_alpha)  # [N, P, F]
        self.local_app_features1 = local_app_features1

        injected_mask0 = unpool_features(local_app_features1, self.decoding_mask)
        injected_mask0_4D = tf.reduce_sum(injected_mask0, 3)
        injected_mask0_4D = tf.concat([injected_mask0_4D, self.decoding_mask], axis=3)
        self._generated = dd(injected_mask0_4D)
        # part assignment

        ############ visualize transfer
        # ! need to stop gradients on z, otherwise graph building hangs?
        independent_local_app_features = tf.stop_gradient(
            tf.reverse(local_app_features1, [0])
        )
        independent_injected_mask0 = unpool_features(
            independent_local_app_features, self.decoding_mask
        )
        independent_injected_mask0 = tf.reduce_sum(independent_injected_mask0, 3)
        independent_injected_mask0 = tf.concat(
            [independent_injected_mask0, self.decoding_mask], axis=3
        )
        self._cross_generated = dd(independent_injected_mask0)

        # mi gradients
        self.lon = tf.Variable(1.0, dtype=tf.float32, trainable=False)

        self.logit_joint0 = mi0_discriminator(
            (pi_distribution_v0.sample(noise_level=self.lon), alpha_v1)
        )
        self.logit_marginal0 = mi0_discriminator(
            (pi_distribution_v0.sample(noise_level=self.lon), z_1_independent)
        )

        self.logit_joint1 = mi1_discriminator((pi_distribution_v0.sample(), alpha_v1))
        self.logit_marginal1 = mi1_discriminator(
            (pi_distribution_v0.sample(), z_1_independent)
        )

        # mi estimation
        self.mi_logit_joint = mi_estimator((pi_distribution_v0.sample(), alpha_v1))
        self.mi_logit_marginal = mi_estimator(
            (pi_distribution_v0.sample(), z_1_independent)
        )
        

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


def grad_energy(x):
    dy, dx = tf.image.image_gradients(x)
    grad_squared = tf.square(dy) + tf.square(dx)
    e = 0.5 * grad_squared
    e = tf.reduce_sum(e, [1, 2, 3])
    e = tf.reduce_mean(e, [0])
    return e


def visualize_mask(name, mask, img_ops, make_hot=True):
    n_parts = mask.shape.as_list()[3]
    vis_mask = nn.mask2rgb(mask, make_hot)
    img_ops[name + "_visualization"] = vis_mask


from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import RestoreTFModelHook

import tensorflow.contrib.slim as slim


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
        """Overwrite default to make lazy"""
        init_op = tf.variables_initializer(self.get_init_variables())
        self.session.run(init_op)
        if checkpoint_path is not None:
            self.set_global_step(RestoreTFModelHook.parse_global_step(checkpoint_path))
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path, self.get_restore_variables(), ignore_missing_vars=True
            )
            self.session.run(init_assign_op, feed_dict=init_feed_dict)
            self.logger.info("Lazily restored from {}".format(checkpoint_path))

    def make_loss_ops(self):
        # perceptual network
        losses = dict()
        with tf.variable_scope("VGG19__noinit__"):
            gram_weight = self.config.get("gram_weight", 0.0)
            print("GRAM: {}".format(gram_weight))
            self.vgg19 = vgg19 = deeploss.VGG19Features(
                self.session, default_gram=gram_weight, original_scale=True
            )
        dim = np.prod(self.model.augmented_views[0].shape.as_list()[1:])
        auto_rec_loss = (
            1e-3
            * 0.5
            * dim
            * vgg19.make_loss_op(self.model.augmented_views[2], self.model._generated)
        )

        with tf.variable_scope("prior_gmrf_weight"):
            prior_gmrf_weight = make_var(
                step=self.global_step,
                var_type=self.config["prior_gmrf_weight"]["var_type"],
                options=self.config["prior_gmrf_weight"]["options"],
            )
        with tf.variable_scope("prior_mumford_sha_weight"):
            prior_mumford_sha_weight = make_var(
                step=self.global_step,
                var_type=self.config["prior_mumford_sha_weight"]["var_type"],
                options=self.config["prior_mumford_sha_weight"]["options"],
            )
        with tf.variable_scope("kl_weight"):
            kl_weight = make_linear_var(
                step=self.global_step, **self.config["kl_weight"]
            )

        mumford_sha_alpha = make_var(
            step=self.global_step,
            var_type=self.config["mumford_sha_alpha"]["var_type"],
            options=self.config["mumford_sha_alpha"]["options"],
        )
        mumford_sha_lambda = make_var(
            step=self.global_step,
            var_type=self.config["mumford_sha_lambda"]["var_type"],
            options=self.config["mumford_sha_lambda"]["options"],
        )
        self.log_ops["mumford_sha_lambda"] = mumford_sha_lambda
        self.log_ops["mumford_sha_alpha"] = mumford_sha_alpha

        # smoothness prior (phase energy)
        prior_gmrf = self.smoothness_prior_simple_gradient()
        prior_gmrf_weighted = prior_gmrf_weight * prior_gmrf

        self.log_ops["prior_gmrf"] = prior_gmrf
        self.log_ops["prior_gmrf_weight"] = prior_gmrf_weight
        self.log_ops["prior_gmrf_weighted"] = prior_gmrf_weighted

        mask0_kl = tf.reduce_sum(
            [categorical_kl(m) for m in [self.model.m0_sample, self.model.m1_sample]]
        )
        mask0_kl_weighted = kl_weight * mask0_kl  # TODO: categorical KL
        self.log_ops["mask0_kl_weight"] = kl_weight
        self.log_ops["mask0_kl"] = mask0_kl
        self.log_ops["mask0_kl_weighted"] = mask0_kl_weighted

        log_probs = self.model.m0_logits
        p_labels = nn.softmax(log_probs, spatial=False)
        labels = nn.hard_max(p_labels, 3)
        labels = nn.straight_through_estimator(labels, p_labels)
        if self.config.get("entropy_func", "cross_entropy") == "cross_entropy":
            weakly_superv_loss_p = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels, log_probs, dim=3
            )
        elif self.config.get("entropy_func", "cross_entropy") == "entropy":
            weakly_superv_loss_p = tf.nn.softmax_cross_entropy_with_logits_v2(
                p_labels, log_probs, dim=3
            )            
        else:
            raise ValueError("unkown entropy_func")
        weakly_superv_loss_p = tf.reduce_mean(weakly_superv_loss_p)

        gamma = self.config.get("gamma", 3.0)
        corrected_logits_1 = nn.softmax(self.model.m1_logits, spatial=False) * gamma
        corrected_logits_1 = nn.softmax(corrected_logits_1, spatial=True)
        corrected_logits_1 *= tf.stop_gradient((1 - self.model.m1_sample_hard_mask))

        N, h, w, P = self.model.m1_sample.shape.as_list()
        _, sigma = nn.probs_to_mu_sigma(corrected_logits_1, tf.ones((N, P)))
        # for i in range(sigma.shape.as_list()[1]):
        #     self.log_ops["sigma1_{:02d}".format(i)] = sigma[0, i, 0, 0]
        # for i in range(sigma.shape.as_list()[1]):
        #     self.log_ops["sigma2_{:02d}".format(i)] = sigma[0, i, 1, 1]

        # sigma_filter = tf.reduce_sum(
        #     self.model.m1_sample_hard, axis=(1, 2), keep_dims=True
        # )
        # sigma_filter = tf.to_float(sigma_filter > 0.0)
        # sigma_filter = tf.transpose(sigma_filter, perm=[0, 3, 1, 2])
        # for i in range(sigma.shape.as_list()[1]):
        #     self.log_ops["sigma_active_{:02d}".format(i)] = sigma_filter[0, i, 0, 0]
        # sigma *= tf.stop_gradient(sigma_filter)
        # for i in range(sigma.shape.as_list()[1]):
        #     self.log_ops["sigma2_filtered_{:02d}".format(i)] = sigma[0, i, 1, 1]
        variances = tf.reduce_mean(
            tf.reduce_sum(sigma[:, :, 0, 0] + sigma[:, :, 1, 1], axis=1)
        )

        with tf.variable_scope("variance_weight"):
            variance_weight = make_var(
                step=self.global_step,
                var_type=self.config["variance_weight"]["var_type"],
                options=self.config["variance_weight"]["options"],
            )

        variance_weighted = variance_weight * variances
        self.log_ops["variance_loss_weighted"] = variance_weighted
        self.log_ops["variance_loss"] = variances
        self.log_ops["variance_weight"] = variance_weight

        with tf.variable_scope("weakly_superv_loss_weight_p"):
            weakly_superv_loss_weight_p = make_var(
                step=self.global_step,
                var_type=self.config["weakly_superv_loss_weight_p"]["var_type"],
                options=self.config["weakly_superv_loss_weight_p"]["options"],
            )

        weakly_superv_loss_p_weighted = (
            weakly_superv_loss_p * weakly_superv_loss_weight_p
        )

        self.log_ops["weakly_superv_loss_weight_p"] = weakly_superv_loss_weight_p
        self.log_ops["weakly_superv_loss_p"] = weakly_superv_loss_p
        self.log_ops["weakly_superv_loss_p_weighted"] = weakly_superv_loss_p_weighted

        # per submodule loss

        # delta
        losses["encoder_0"] = auto_rec_loss
        losses["encoder_1"] = auto_rec_loss
        # decoder
        losses["decoder_delta"] = auto_rec_loss    

        r, smoothness_cost, contour_cost = nn.mumford_shah(
            self.model.m0_sample, 1.0, 1.0e-2
        )
        smoothness_cost = tf.reduce_mean(
            tf.reduce_sum(
                (tf.reduce_sum(smoothness_cost, axis=(1, 2), keep_dims=True)) ** 2,
                axis=(3,),
            )
        )
        contour_cost = tf.reduce_mean(
            tf.reduce_sum(
                (tf.reduce_sum(contour_cost, axis=(1, 2), keep_dims=True)) ** 2,
                axis=(3,),
            )
        )
        p_mumford_sha = prior_mumford_sha_weight * tf.reduce_mean(
            tf.reduce_sum(
                (tf.reduce_sum(r, axis=(1, 2), keep_dims=True)) ** 2, axis=(3,)
            )
        )
        area_cost = 1.0e-12 * tf.reduce_mean(
            tf.reduce_sum(
                (tf.reduce_sum(self.model.m0_sample, axis=(1, 2), keep_dims=True)) ** 2,
                axis=(3,),
            )
        )

        patch_loss = self.model.m0_sample_hard * tf.stop_gradient((1 - self.model.m0_sample_hard_mask))
        patch_loss = tf.reduce_sum(patch_loss, axis=[1, 2, 3])
        patch_loss = tf.reduce_mean(patch_loss, axis=[0])
        with tf.variable_scope("patch_loss_weight"):
            patch_loss_weight = make_var(
                step=self.global_step,
                var_type=self.config["patch_loss_weight"]["var_type"],
                options=self.config["patch_loss_weight"]["options"],
            )
        patch_loss_weighted = patch_loss * patch_loss_weight
        self.log_ops["patch_loss"] = patch_loss
        self.log_ops["patch_loss_weight"] = patch_loss_weight
        self.log_ops["patch_loss_weighted"] = patch_loss_weighted
        
        if not self.config.get("pretrain", False):
            losses["decoder_visualize"] = (
                auto_rec_loss
                + prior_gmrf_weighted
                + mask0_kl_weighted
                + weakly_superv_loss_p_weighted
                + variance_weighted
                + p_mumford_sha
                + area_cost
                + patch_loss_weighted
            )
        else:
            losses["decoder_visualize"] = auto_rec_loss


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

        # OPTION
        adversarial_regularization = self.config.get("adversarial_regularization", True)
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
        variational_regularization = self.config.get("variational_regularization", True)
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
            beta_0 = self.config.get("beta_0", 1.0)
            bottleneck_weighted_loss = beta_0 * tf.exp(lor) * bottleneck_loss # TODO: hardcoded
            losses["encoder_0"] += bottleneck_weighted_loss
        else:
            assert not self.model.stochastic_encoder_0

 

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
            beta_0 = self.config.get("beta_0", 1.0)
            self.log_ops["explor"] = beta_0 * tf.exp(lor)
            self.log_ops["lor_gain"] = lor_gain

        visualize_mask(
            "out_parts_soft", tf.to_float(self.model.out_parts_soft), self.img_ops, True
        )
        visualize_mask("m0_sample", self.model.m0_sample, self.img_ops, True)

        cols = self.model.encoding_mask.shape.as_list()[0]
        encoding_masks = tf.concat([self.model.encoding_mask], 0)
        encoding_masks = tf_batches.tf_batch_to_canvas(encoding_masks, cols)
        visualize_mask(
            "encoding_masks", tf.to_float(encoding_masks), self.img_ops, False
        )

        decoding_masks = tf.concat([self.model.decoding_mask], 0)
        decoding_masks = tf_batches.tf_batch_to_canvas(decoding_masks, cols)
        visualize_mask(
            "decoding_masks", tf.to_float(decoding_masks), self.img_ops, False
        )

        masks = nn.take_only_first_item_in_the_batch(self.model.decoding_mask)
        masks = tf.transpose(masks, perm=[3, 1, 2, 0])
        self.img_ops["masks"] = masks

        correspondence0 = tf.expand_dims(
            self.model.decoding_mask, axis=-1
        ) * tf.expand_dims(self.model.augmented_views[0], axis=3)
        correspondence0 = tf.transpose(correspondence0, [3, 0, 1, 2, 4])
        correspondence1 = tf.expand_dims(
            self.model.encoding_mask, axis=-1
        ) * tf.expand_dims(self.model.augmented_views[1], axis=3)
        correspondence1 = tf.transpose(correspondence1, [3, 0, 1, 2, 4])
        correspondence = tf.concat([correspondence0, correspondence1], axis=1)
        N_PARTS, _, H, W, _ = correspondence.shape.as_list()

        def make_grid(X):
            X = tf.squeeze(X)
            return tf_batches.tf_batch_to_canvas(X)

        correspondence = list(map(make_grid, tf.split(correspondence, N_PARTS, 0)))
        correspondence = tf_batches.tf_batch_to_canvas(tf.concat(correspondence, 0), 5)
        self.img_ops.update({"assigned_parts": correspondence})

        p_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
        m = nn.take_only_first_item_in_the_batch(self.model.m0_sample)

        def levels(m, i):
            return tf.concat([tf.to_float(m[..., i] > p) for p in p_levels], axis=0)

        level_sets = tf.concat([levels(m, i) for i in range(m.shape[3])], 0)
        level_sets = tf.expand_dims(level_sets, -1)
        level_sets = tf_batches.tf_batch_to_canvas(level_sets, len(p_levels))
        img_title = "m0_sample_levels" + "-{}" * len(p_levels)
        img_title = img_title.format(*p_levels).replace(".", "_")

        ratios = [1.0e-3, 5 * 1.0e-3, 1.0e-2, 5 * 1.0e-2]
        mumford_sha_sets = tf.concat(
            [nn.edge_set(m, 1, r) for r in ratios], axis=0
        )  # 5, H, W, 25
        mumford_sha_sets = tf.transpose(mumford_sha_sets, perm=[3, 0, 1, 2])
        a, b, h, w = mumford_sha_sets.shape.as_list()
        mumford_sha_sets = tf.reshape(mumford_sha_sets, (a * b, h, w))
        mumford_sha_sets = tf.expand_dims(mumford_sha_sets, -1)
        mumford_sha_sets = tf_batches.tf_batch_to_canvas(mumford_sha_sets, len(ratios))

        p_heatmap = tf.squeeze(nn.colorize(m))  # [H, W, 25, 3]
        p_heatmap = tf.transpose(p_heatmap, perm=[2, 0, 1, 3])

        self.img_ops.update(
            {
                "mumford_sha_edges": mumford_sha_sets,
                "p_heatmap": p_heatmap,
                img_title: level_sets,
                "view0": self.model.inputs["view0"],
                "view1": self.model.inputs["view1"],
                "view0_target": self.model.inputs["view0_target"],
                "cross": self.model._cross_generated,
                "generated": self.model._generated,
            }
        )
        if self.model.use_tps:
            self.img_ops.update(
                {
                    "tps_view0": self.model.outputs["tps_view0"],
                    "tps_view1": self.model.outputs["tps_view1"],
                    "tps_view0_target": self.model.outputs["tps_view0_target"],
                }
            )

        self.log_ops["zr_mumford_sha"] = p_mumford_sha
        self.log_ops["z_mumford_sha_smoothness_cost"] = smoothness_cost
        self.log_ops["z_mumford_sha_contour_cost"] = contour_cost
        self.log_ops["z_area_cost"] = area_cost

        self.log_ops["prior_mumford_sha_weight"] = prior_mumford_sha_weight

        for k in self.config.get("fix_weights", []):
            if k in losses.keys():
                # losses[k] = tf.no_op()
                del losses[k]
            else:
                pass
        return losses

    def smoothness_prior_simple_gradient(self):
        mask0_grad_energy = self.model.l0_distribution.kl_improper_gmrf()
        return mask0_grad_energy
