import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Reshape, Permute, Lambda
from tensorflow_addons.layers import AdaptiveMaxPooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .NetBlocks import *


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#def conv_block(input,filters, kernel, pad, stride):
#    
#    x = Conv2D(filters, kernel_size=kernel, padding=pad, activation='relu')(input)
#    x = Conv2D(filters, kernel_size=kernel, padding=pad, activation='relu')(x)
#    x = InstanceNormalization()(x)
#    x = Conv2D(filters, kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x)
#    x = MixedDropout()(x)
#    
#    return x

#def dsc_block(input, kernel, pad, stride):
#
#    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(input)
#    x = MixedDropout()(x)
#    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x) 
#    x = MixedDropout()(x)
#    x = InstanceNormalization()(x)
#    x = DepthwiseConv2D(kernel_size=kernel, padding=pad, strides=stride, activation='relu')(x)
#    x = MixedDropout()(x)
#    x = Add()([input, x])
#    return x


def get_base_model(input_shape):
    
    input = Input(shape=input_shape, name='the_input')

    ### CB1
    
    x = ConvBlock(32, (3,3), "same", (1,1))(input)

    ### CB2
    
    x = ConvBlock(64, (3,3), "same", (2,2))(x)

    ### CB3
    
    x = ConvBlock(128, (3,3), "same", (2,2))(x)

    ### CB4

    x = ConvBlock(256, (3,3), "same", (2,2))(x)

    ### CB5

    x = ConvBlock(512, (3,3), "same", (2,1))(x)

    ### CB6
    
    x = ConvBlock(512, (3,3), "same", (2,1))(x)

    ### DSCB_Place

    x = DSCBlock((3,3), "same", (1,1))(x)
    x = DSCBlock((3,3), "same", (1,1))(x)
    x = DSCBlock((3,3), "same", (1,1))(x)
    x = DSCBlock((3,3), "same", (1,1))(x)

    return input, x


def get_line_model(input_shape, out_tokens):
    
    input, out_base = get_base_model(input_shape)

    x = AdaptiveMaxPooling1D(1)(out_base)
    x = Conv2D(out_tokens+1, kernel_size=(1,1), padding="same", activation="softmax")(x)
    x = Permute((2,1,3))(x)
    y_pred = tf.squeeze(x,axis=2)
    #x = Permute((2, 1, 3))(x)
    #y_pred = Reshape(target_shape=(-1, out_tokens+1), name='reshape')(x)

    model_base = Model(inputs=input, outputs=out_base)

    model_pr = Model(inputs=input, outputs=y_pred)
    model_pr.summary()

    labels = Input(name='the_labels',shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = Model(inputs=[input, labels, input_length, label_length],
                  outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001, amsgrad=False))

    return model_tr, model_pr, model_base


def get_span_model(input_shape, out_tokens):

    input, out_base = get_base_model(input_shape)

    x = Conv2D(out_tokens+1, kernel_size=(5,5), padding="same", activation="softmax")(out_base)

    y_pred = Reshape(target_shape=(-1, out_tokens+1), name='reshape')(x)

    model_base = Model(inputs=input, outputs=out_base)

    model_pr = Model(inputs=input, outputs=y_pred)
    model_pr.summary()

    labels = Input(name='the_labels',shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = Model(inputs=[input, labels, input_length, label_length],
                  outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001, amsgrad=False))

    return model_tr, model_pr, model_base
