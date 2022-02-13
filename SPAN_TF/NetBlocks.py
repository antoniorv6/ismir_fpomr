import tensorflow as tf
from tensorflow.keras.layers import Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU
import random
from tensorflow_addons.layers.normalizations import InstanceNormalization
import numpy as np

class MixedDropout(tf.keras.layers.Layer):
    def __init__(self, spatial_dropout_dist=0.2, dropout_dist=0.4):
        super(MixedDropout, self).__init__()
        self.dropout = Dropout(dropout_dist)
        self.dropout_bi = SpatialDropout2D(spatial_dropout_dist)
    
    def __call__(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout_bi(x)

class ConvBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel, pad, stride):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size=kernel, padding=pad)
        self.conv2 = Conv2D(filters, kernel_size=kernel, padding=pad)
        self.conv3 = Conv2D(filters, kernel_size=kernel, strides=stride, padding=pad)
        self.activation = ReLU()
        self.dropout = MixedDropout()
        self.norm = InstanceNormalization()
    
    def __call__(self, input):
        
        pos = random.randint(1,3)

        x = self.conv1(input)
        x = self.activation(x)

        if pos==1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.norm(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x


class DepthSepConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, krn_sz, pad, stride=(1,1)):
        super(DepthSepConvBlock, self).__init__()
        self.depth_conv = DepthwiseConv2D(kernel_size=krn_sz, dilation_rate=(1,1), strides=stride, padding=pad)
        self.point_conv = Conv2D(filters, kernel_size=(1,1), dilation_rate=(1,1))
    
    def __call__(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class DSCBlock(tf.keras.layers.Layer):

    def __init__(self, kernel, pad, stride):
        super(DSCBlock, self).__init__()
        self.conv1 = DepthSepConvBlock(filters=512, krn_sz=kernel, pad=pad)
        self.conv2 = DepthSepConvBlock(filters=512, krn_sz=kernel, pad=pad)
        self.conv3 = DepthSepConvBlock(filters=512, krn_sz=kernel, stride=stride, pad=pad)
        self.activation = ReLU()
        self.dropout = MixedDropout()
        self.norm = InstanceNormalization()
    
    def __call__(self, input):
        
        pos = random.randint(1,3)

        x = self.conv1(input)
        x = self.activation(x)

        if pos==1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.norm(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return input + x

class PositionEncoding2D(tf.keras.layers.Layer):
    
    @staticmethod
    def positional_encoding(d_model, height, width):
        pe = np.zeros((d_model, height, width))
        d_model = int(d_model / 2)
        div_term = np.exp(np.arange(0., d_model, 2) *
                         -(np.log(10000.0) / d_model))
        pos_w = np.expand_dims(np.arange(0., width), 1)
        pos_h = np.expand_dims(np.arange(0., height), 1)
        pe[0:d_model:2, :, :] = np.expand_dims(np.sin(pos_w * div_term).transpose(0, 1),1).repeat(1, axis=0).repeat(height,axis=1).repeat(1,axis=2).reshape(-1, height,width)
        pe[1:d_model:2, :, :] = np.expand_dims(np.cos(pos_w * div_term).transpose(0, 1), 1).repeat(1, axis=0).repeat(height,axis=1).repeat(1,axis=2).reshape(-1, height,width)
        pe[d_model::2, :, :] = np.expand_dims(np.sin(pos_h * div_term).transpose(0, 1), 2).repeat(1, axis=0).repeat(1,axis=1).repeat(width,axis=2).reshape(-1, height,width)
        pe[d_model + 1::2, :, :] = np.expand_dims(np.cos(pos_h * div_term).transpose(0, 1),2).repeat(1, axis=0).repeat(1,axis=1).repeat(width,axis=2).reshape(-1, height,width)
        return tf.cast(pe.reshape(height, width, -1), dtype=tf.float32)


    def __init__(self, d_model, max_width, max_height):
        super(PositionEncoding2D, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin and cos with an odd d_model value")
        
        self.d_model = d_model
        self.position_encoding = PositionEncoding2D.positional_encoding(d_model, max_width, max_height) 
        pass
    
    def __call__(self, input):
        return input + self.position_encoding[:tf.shape(input)[1],:tf.shape(input)[2],:]

class PositionEncoding1D(tf.keras.layers.Layer):
    
    @staticmethod
    def get_angles(pos, i, model_depth):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_depth))
        return pos * angle_rates
    
    @staticmethod
    def positional_encoding(position, model_depth):
        angle_rads = PositionEncoding1D.get_angles(np.arange(position)[:, np.newaxis],
                                                   np.arange(model_depth)[np.newaxis, :],
                                                   model_depth)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


    def __init__(self, d_model, max_seq_len):
        super(PositionEncoding1D, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin and cos with an odd d_model value")
        
        self.d_model = d_model
        self.position_encoding = PositionEncoding1D.positional_encoding(max_seq_len, d_model) 
        pass
    
    def __call__(self, input):
        return input + self.position_encoding[:,:tf.shape(input)[1],:]

        


        

    

