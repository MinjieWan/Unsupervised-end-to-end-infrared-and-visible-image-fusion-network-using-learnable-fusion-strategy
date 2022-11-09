# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def Dense_block(x):
    for i in range(3):
        shape = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0 / shape[-1]))
        t = tf.layers.conv2d(x,16,3,(1,1),padding='SAME',kernel_initializer=w_init)
        t = tf.nn.relu(t)
        x = tf.concat([x,t],3)
    return x

def conv_layer(x,filternum,filtersize=3,isactiv=True):
    shape = x.get_shape().as_list()
    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / filtersize/filtersize/ shape[-1]))
    t = tf.layers.conv2d(x, filternum, filtersize, (1, 1), padding='SAME', kernel_initializer=w_init)
    if isactiv:
        t = tf.nn.relu(t)
    return t
def FusionNet(f_ir,f_vi,in_channels,out_channels):#RFN
    f_cat=tf.concat([f_vi,f_ir],3)
    f_init=conv_layer(f_cat,out_channels)

    f_ir=conv_layer(f_ir,out_channels)
    f_vi=conv_layer(f_vi,out_channels)

    f_cat=tf.concat([f_vi,f_ir],3)
    f_temp=conv_layer(f_cat,out_channels,1)
    f_temp=conv_layer(f_temp,out_channels)
    f_temp = conv_layer(f_temp, out_channels)

    f=f_temp+f_init
    return f
def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
     f1 = 0.5 * (1 + leak)
     f2 = 0.5 * (1 - leak)
     return f1 * x + f2 * tf.abs(x)
def ResBlock(f,out_channels,bn=False,act='leaky'):
    f=conv_layer(f,out_channels,3)
    if bn:
        f=tf.layers.batch_normalization(f,momentum=0.03,epsilon=1e-4,training=True)
    if act=='leaky':
        f=LeakyRelu(f)
    f=conv_layer(f,out_channels,3)
    return f
def forward(x):
    im_source1,im_source2 = tf.split(x,2,3)
    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0))
    output = tf.layers.conv2d(im_source1, 16, 3, (1, 1), padding='SAME', kernel_initializer=w_init)
    output = tf.nn.relu(output)
    Feature_S1 = Dense_block(output)

    w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9.0))
    output = tf.layers.conv2d(im_source2, 16, 3, (1, 1), padding='SAME', kernel_initializer=w_init)
    output = tf.nn.relu(output)
    Feature_S2 = Dense_block(output)

    out_channels=[64,32,16]
    Feature_fusion=FusionNet(Feature_S1,Feature_S2,3,128)
    Feature_fusion=conv_layer(Feature_fusion,out_channels[0])
    Feature_fusion_temp=Feature_fusion
    Feature_fusion1=ResBlock(Feature_fusion,out_channels[0])

    Feature_fusion=conv_layer(Feature_fusion+Feature_fusion1,out_channels[1])
    Feature_fusion1 = ResBlock(Feature_fusion, out_channels[1])

    Feature_fusion = conv_layer(Feature_fusion + Feature_fusion1,out_channels[2])
    Feature_fusion1 = ResBlock(Feature_fusion, out_channels[2])

    Feature_fusion= Feature_fusion + Feature_fusion1

    output=conv_layer(Feature_fusion,1,3)

    # return output,Feature_S0,Feature_DoLP,Feature_fusion_temp    #For training
    return output                                                  #For testing
if __name__=='__main__':
    a=tf.constant(1,tf.float32,shape=[40,60,3,2])
    b,c,d,e=forward(a)
    print (b,c,d,e)

