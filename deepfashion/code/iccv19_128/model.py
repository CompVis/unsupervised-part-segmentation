import sys, os
import tensorflow as tf
import numpy as np

from edflow.iterators.trainer import TFBaseTrainer
import edflow.iterators.deeploss as deeploss
from edflow.hooks.evaluation_hooks import RestoreTFModelHook

import nips19.nn as nn

from triplet_reid.edflow_implementations.implementations import (
    make_network as make_triplet_net,
)

convconf = [32, 64, 128, 128, 256, 256]  # for image size 128
# convconf = [32, 64, 128, 128, 256, 256, 256]
dsize = 512
extra_resnets = 4


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


def encoder_model(x, out_size):
    with nn.model_arg_scope(activation="relu"):
        h = nn.conv2d(x, convconf[0])
        h = nn.residual_block(h)

        for nf in convconf[1:]:
            h = nn.downsample(h, nf)
            h = nn.residual_block(h)

        for _ in range(extra_resnets):
            h = nn.residual_block(h)

        h = nn.activate(h)
        h = tf.reduce_mean(h, [1, 2], keep_dims=True)
        h = nn.nin(h, out_size)

        return h


def decoder_model(h1, h2):
    with nn.model_arg_scope(activation="relu"):
        h1 = nn.nin(h1, 4 * 4 * convconf[-1])
        h1 = tf.reshape(h1, [-1, 4, 4, convconf[-1]])
        h2 = nn.nin(h2, 4 * 4 * convconf[-1])
        h2 = tf.reshape(h2, [-1, 4, 4, convconf[-1]])
        for _ in range(extra_resnets):
            h1 = nn.residual_block(h1)
            h2 = nn.residual_block(h2)

        h = tf.concat([h1, h2], axis=-1)
        h = nn.conv2d(h, convconf[-1])

        for nf in convconf[-2::-1]:
            h = nn.residual_block(h)
            h = nn.upsample(h, nf)

        h = nn.residual_block(h)
        h = nn.conv2d(h, 3)

        return h


def single_decoder_model(h):
    with nn.model_arg_scope(activation="relu"):
        h = nn.nin(h, 4 * 4 * convconf[-1])
        h = tf.reshape(h, [-1, 4, 4, convconf[-1]])

        h = nn.conv2d(h, convconf[-1])
        h = nn.residual_block(h)

        for nf in convconf[-2::-1]:
            h = nn.residual_block(h)
            h = nn.upsample(h, nf)

        h = nn.residual_block(h)
        h = nn.conv2d(h, 3)

        return h


def discriminator_model(pair):
    with nn.model_arg_scope(activation="relu"):
        outpair = list()
        for z in pair:
            nc = dsize
            h = nn.nin(z, nc)
            for _ in range(4):
                h = nn.residual_block(h, conv=nn.nin)
            h = nn.activate(h)
            h = nn.nin(h, nc)
            outpair.append(h)
        h = outpair[0] * outpair[1]  # TODO this is a seperable critic
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

        h = tf.reduce_mean(h, [1, 2], keep_dims=True)
        hc = nn.nin(c, dsize)
        hc = nn.residual_block(hc, conv=nn.nin)
        hc = nn.residual_block(hc, conv=nn.nin)
        h = h * hc

        h = tf.reduce_mean(h, [1, 2, 3])
        h = tf.expand_dims(h, -1)
        return h, hs


class TrainModel(object):
    def __init__(self, config):
        self.config = config
        self.pretty = self.config.get("use_pretty", False)
        variables = set(tf.global_variables())
        self.define_graph()
        self.variables = [v for v in tf.global_variables() if not v in variables]
        self.variables = [v for v in self.variables if not self.e1_name in v.name]

    @property
    def inputs(self):
        return {"view0": self.views[0], "view1": self.views[1]}

    @property
    def outputs(self):
        return {
            "generated": self._generated,
            "visualize0": self._visualize,
            "visualize1": self._app_visualize,
        }

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

        z0_size = 256
        z1_size = 128
        z0_n_parameters = nn.FullLatentDistribution.n_parameters(z0_size)

        # submodules
        e0 = nn.make_model("encoder_0", encoder_model, out_size=z0_n_parameters)
        # e1 = nn.make_model("encoder_1", encoder_model, out_size = z1_size)
        e1, self.e1_name = triplet_wrapper(z1_size)
        dd = nn.make_model("decoder_delta", decoder_model)
        dv = nn.make_model("decoder_visualize", single_decoder_model)
        da = nn.make_model("decoder_app", single_decoder_model)
        mi_estimator = nn.make_model("mi_estimator", discriminator_model)
        mi0_discriminator = nn.make_model("mi0_discriminator", discriminator_model)
        mi1_discriminator = nn.make_model("mi1_discriminator", discriminator_model)
        if self.pretty:
            pretty_discriminator = nn.make_model(
                "pretty_discriminator", pretty_discriminator_model
            )

        # test
        self.stochastic_encoder_0 = not self.config.get("test_mode", False)
        z_00_parameters = e0(self.views[0])
        z_00_distribution = self.z_00_distribution = nn.FullLatentDistribution(
            z_00_parameters, z0_size, stochastic=self.stochastic_encoder_0
        )
        z_10 = self.z_10 = e1(self.views[0])
        z_11 = self.z_11 = e1(self.views[1])
        z_1_independent = self.z_1_independent = tf.reverse(z_10, [0])

        self._generated = dd(z_00_distribution.sample(), z_11)
        # ! need to stop gradients on z, otherwise graph building hangs?
        self._cross_generated = dd(
            z_00_distribution.sample(), tf.stop_gradient(z_1_independent)
        )
        if self.pretty:
            # pretty one
            self.logit_pretty_orig, self.feat_pretty_orig = pretty_discriminator(
                self.views[0], tf.stop_gradient(z_11)
            )
            self.fake_input = tf.concat(
                [self._generated, self._cross_generated], axis=0
            )
            fake_selection = tf.random_uniform(
                (self.config["batch_size"],),
                minval=0,
                maxval=2 * self.config["batch_size"],
                dtype=tf.int32,
            )
            self.fake_input = tf.gather(self.fake_input, fake_selection, axis=0)
            self.logit_pretty_fake, self.feat_pretty_fake = pretty_discriminator(
                self.fake_input, tf.stop_gradient(z_1_independent)
            )

        # mi gradients
        self.lon = tf.Variable(1.0, dtype=tf.float32, trainable=False)
        self.logit_joint0 = mi0_discriminator(
            (z_00_distribution.sample(noise_level=self.lon), z_11)
        )
        self.logit_marginal0 = mi0_discriminator(
            (z_00_distribution.sample(noise_level=self.lon), z_1_independent)
        )

        self.logit_joint1 = mi1_discriminator((z_00_distribution.sample(), z_11))
        self.logit_marginal1 = mi1_discriminator(
            (z_00_distribution.sample(), z_1_independent)
        )

        # mi estimation
        self.mi_logit_joint = mi_estimator((z_00_distribution.sample(), z_11))
        self.mi_logit_marginal = mi_estimator(
            (z_00_distribution.sample(), z_1_independent)
        )

        # visualizations of latent codes
        self._visualize = dv(z_00_distribution.sample())
        self._app_visualize = da(z_11)


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


class Trainer(TFBaseTrainer):
    def get_restore_variables(self):
        vs = super().get_restore_variables()
        if self.config.get("add_pretty", False):
            exclude = ["pretty_discriminator", "beta1_power_7", "beta2_power_7"]
            for name in exclude:
                vs = [v for v in vs if not name in v.name]
            return vs
        else:
            return vs

    def initialize(self, checkpoint_path=None):
        return_ = super().initialize(checkpoint_path)
        triplet_path = self.config["triplet_path"]
        e1_name = "my_triplet_is_the_best_triplet"
        triplet_variables = [v for v in tf.global_variables() if e1_name in v.name]
        restorer = RestoreTFModelHook(variables=triplet_variables, checkpoint_path=None)
        with self.session.as_default():
            restorer(triplet_path)
        print("Restored triplet net.")
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
        vis_rec_loss = tf.reduce_mean(
            tf.square(self.model.inputs["view0"] - self.model._visualize)
        )
        app_rec_loss = tf.reduce_mean(
            tf.square(self.model.inputs["view0"] - self.model._app_visualize)
        )

        # per submodule loss
        losses = dict()
        # delta
        losses["encoder_0"] = auto_rec_loss

        # app
        # losses["encoder_1"] = auto_rec_loss

        # decoder
        losses["decoder_delta"] = auto_rec_loss
        losses["decoder_visualize"] = vis_rec_loss
        losses["decoder_app"] = app_rec_loss

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
        MI_TARGET = 0.125
        MI_SLACK = 0.05

        LOO_TOL = 0.025
        LON_LR = 0.05
        LON_ADAPTIVE = False

        LOA_INIT = 0.0
        LOA_LR = 4.0
        LOA_ADAPTIVE = True

        LOR_INIT = 7.5
        LOR_LR = 0.05
        LOR_MIN = 1.0
        LOR_MAX = 7.5
        LOR_ADAPTIVE = True

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

        self.img_ops.update(
            {
                "view0": self.model.inputs["view0"],
                "view1": self.model.inputs["view1"],
                "visualize0": self.model.outputs["visualize0"],
                "visualize1": self.model.outputs["visualize1"],
                "cross": self.model._cross_generated,
                "generated": self.model._generated,
            }
        )

        return losses
