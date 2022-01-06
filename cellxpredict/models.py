from typing import Tuple

import numpy as np
import tensorflow as tf
from cellx.layers import Decoder2D, Encoder2D, PCATransform
from cellx.networks.tcn import build_split_TCN
from cellx.networks.vae import (
    VAECapacity,
    convolutional_variational_decoder,
    convolutional_variational_encoder,
)
from tensorflow import keras as K

from .config import ConfigBase


def _build_autoencoder(config: ConfigBase) -> K.Model:
    """Build the autoencoder model from the config.

    Parameters
    ----------
    config : ConfigBase
        A configuration for the encoder.

    Returns
    -------
    model : tf.keras.Model
        The model.
    """

    layers = config.layers
    encoder = Encoder2D(layers=layers)
    decoder = Decoder2D(layers=layers[::-1])
    model = VAECapacity(
        encoder,
        decoder,
        latent_dims=config.latent_dims,
        intermediate_dims=config.intermediate_dims,
        input_shape=config.input_shape,
        gamma=config.gamma,
        capacity=config.capacity,
        max_iterations=config.max_iterations,
    )

    model.compile(optimizer=K.optimizers.Adam())

    return model


def _build_encoder(config: ConfigBase) -> K.Model:
    """Build the encoder model from the config.

    Parameters
    ----------
    config : ConfigBase
        A configuration for the encoder.

    Returns
    -------
    model : tf.keras.Model
        The model.
    """

    from copy import deepcopy

    # build the encoder
    encoder_config = deepcopy(config)
    encoder_config.model = "encoder"

    model = convolutional_variational_encoder(
        encoder=Encoder2D(layers=config.layers),
        input_shape=config.input_shape,
        latent_dims=config.latent_dims,
        intermediate_dims=config.intermediate_dims,
    )
    weights_fn = config.model_dir / encoder_config.filename("weights")
    model.load_weights(weights_fn.with_suffix(".h5"), by_name=True)
    for layer in model.layers:
        layer.trainable = False

    return model


def _build_decoder(config: ConfigBase) -> K.Model:
    """Build the decoder model from the config.

    Parameters
    ----------
    config : ConfigBase
        A configuration for the encoder.

    Returns
    -------
    model : tf.keras.Model
        The model.
    """

    from copy import deepcopy

    # build the encoder
    decoder_config = deepcopy(config)
    decoder_config.model = "decoder"

    model = convolutional_variational_decoder(
        decoder=Decoder2D(layers=config.layers[::-1]),
        output_shape=config.input_shape,
        latent_dims=config.latent_dims,
    )
    weights_fn = config.model_dir / decoder_config.filename("weights")
    model.load_weights(weights_fn.with_suffix(".h5"), by_name=True)
    for layer in model.layers:
        layer.trainable = False

    return model


def _build_temporal(config: ConfigBase) -> Tuple[K.Model]:
    """Build the temporal model from the config.

    Parameters
    ----------
    config : ConfigBase
        A configuration for the encoder.

    Returns
    -------
    encoder : tf.keras.Model
        The probabilistic encoder.
    model : tf.keras.Model
        The model.
    """

    from copy import deepcopy

    # build the projector
    enc_config = deepcopy(config)
    enc_config.model = "projector"

    # load the components and build the transformer
    components_fn = config.model_dir / enc_config.filename("components")
    components = np.load(components_fn.with_suffix(".npz"))

    transformer = PCATransform(**components)

    # build the temporal model
    tcn, tcn_split_1, tcn_split_2 = build_split_TCN(
        config.max_len,
        config.latent_dims,
        num_outputs=config.num_outputs,
        dropout_rate=config.dropout_rate,
    )

    # build the grand model
    z = K.layers.Input(shape=(config.max_len, config.latent_dims))
    projection = transformer(z)
    prediction = tcn(projection)

    model = K.Model(inputs=[z], outputs=[prediction], name=config.model)

    model.compile(
        optimizer=K.optimizers.RMSprop(learning_rate=0.001),
        loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[K.metrics.SparseCategoricalAccuracy()],
    )

    return model


def _build_full(config: ConfigBase, freeze_model: bool = True) -> K.Model:
    """Build the encoder model from the config.

    Parameters
    ----------
    config : ConfigBase
        A configuration for the encoder.
    freeze_model : bool
        A flag to freeze the model layers.

    Returns
    -------
    model : tf.keras.Model
        The model.
    """

    from copy import deepcopy

    # build the encoder
    enc_config = deepcopy(config)
    enc_config.model = "encoder"

    encoder = convolutional_variational_encoder(
        encoder=Encoder2D(layers=config.layers),
        input_shape=config.input_shape,
        latent_dims=config.latent_dims,
        intermediate_dims=config.intermediate_dims,
    )
    weights_fn = config.model_dir / enc_config.filename("weights")
    encoder.load_weights(weights_fn.with_suffix(".h5"), by_name=True)
    for layer in encoder.layers:
        layer.trainable = False

    # build the projector
    pro_config = deepcopy(config)
    pro_config.model = "projector"

    # load the components and build the transformer
    components_fn = config.model_dir / pro_config.filename("components")
    components = np.load(components_fn.with_suffix(".npz"))
    transformer = PCATransform(**components)

    # build the temporal model
    tcn, tcn_split_1, tcn_split_2 = build_split_TCN(
        config.max_len,
        config.latent_dims,
        num_outputs=config.num_outputs,
        dropout_rate=config.dropout_rate,
    )

    # build the grand model
    glimpse = K.layers.Input(shape=(config.max_len,) + config.input_shape)
    reshaped = tf.reshape(glimpse, shape=(-1,) + config.input_shape)
    z_mean, z_log_var, z = encoder(reshaped)
    sample = z if config.use_probabilistic_encoder else z_mean
    z = tf.reshape(sample, shape=(-1, config.max_len, config.latent_dims))
    projection = transformer(z)
    prediction = tcn(projection)
    model = K.Model(inputs=[glimpse], outputs=[prediction], name="TauVAE")

    # if we're loading weights, do so here
    config.model = "temporal"
    weights_fn = config.model_dir / config.filename("weights")
    model.load_weights(weights_fn.with_suffix(".h5"), by_name=True)

    for layer in model.layers:
        layer.trainable = False

    return model
