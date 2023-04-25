# This code is partly based on the following papers:
#   "Singing Voice Separation with Deep U-Net Convolutional Networks" by Jansson et al., 2017
#   "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al., 2015
from tensorflow import keras  # tested with tensorflow-cpu 2.7.0 and tensorflow 2.4.1


def unet(hyper_params,
         input=keras.Input((512, 2**7, 1))):  # e.g. magnitude or power spectrogram
  """hyper_params - a dictionary with the following fields:
      NUM_BLOCKS - Number of blocks in the encoder/decoder (e.g. setting NUM_BLOCKS to 6 results in 6 blocks in the
        encoder and 6 six blocks in the decoder. Each encoder block consists of 2D convolution, batch normalization,
        and leaky ReLU layers. Each decoder block consists of transposed 2D convolution, batch normalization,
        potentially dropout, and ReLU layers.
      FILTER_MULTIPLIER - Sets the number of filters in 2D convolution layers.
      DROPOUT_ENCODER - A list of boolean values of lengths equal to the number of blocks in the encoder. Each
        boolean value corresponds to a block and decides if that block should have dropout or not.
      DROPOUT_DECODER - A list of boolean values of lengths equal to the number of blocks in the decoder. Each
        boolean value corresponds to a block and decides if that block should have dropout or not.
      DROPOUT_RATE - Set the dropout rate in the dropout layers.
      MASK - Estimate a mask (and multiply it with the input) instead of estimating the output directly.
      """

  # encoder
  x = input
  encoder_layers = []
  for n_layer in range(hyper_params['NUM_BLOCKS']):
    num_filters = 2 ** n_layer * hyper_params['FILTER_MULTIPLIER']
    x = keras.layers.Conv2D(num_filters, 5, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    if hyper_params['DROPOUT_ENCODER'][n_layer]: x = keras.layers.Dropout(hyper_params['DROPOUT_RATE'])(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    encoder_layers.append(x)

  # decoder
  for n_layer in range(hyper_params['NUM_BLOCKS']-1):
    if not n_layer == 0: x = keras.layers.Concatenate(axis=3)([x, encoder_layers[hyper_params['NUM_BLOCKS'] - 1 - n_layer]])
    num_filters = 2 ** (hyper_params['NUM_BLOCKS'] - 1 - n_layer) * hyper_params['FILTER_MULTIPLIER']
    x = keras.layers.Conv2DTranspose(num_filters, 5, strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    if hyper_params['DROPOUT_DECODER'][n_layer]: x = keras.layers.Dropout(hyper_params['DROPOUT_RATE'])(x)
    x = keras.layers.Activation('relu')(x)

  output = keras.layers.Concatenate(axis=3)([x, encoder_layers[-hyper_params['NUM_BLOCKS']]])
  output = keras.layers.Conv2DTranspose(1, 5, strides=2, padding='same')(output)
  output = keras.layers.Activation('relu')(output)

  # mask
  if hyper_params['MASK']:
    output = keras.layers.multiply([input, output])

  model = keras.Model(inputs=input, outputs=output)
  return model