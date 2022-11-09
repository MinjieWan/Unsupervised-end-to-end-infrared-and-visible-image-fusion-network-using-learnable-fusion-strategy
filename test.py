# -*- coding: utf-8 -*-

import tensorflow as tf
import model as model
import numpy as np
import h5py
import scipy.io
import cv2
MODEL_SAVE_PATH = './model_PFNet/'
IMG_CHANNEL = 2
IMG_SIZE =(520,520)    #size of test images
BATCH_TEST = 100

def test(test_data,save_name):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[1,
                                       IMG_SIZE[0],
                                       IMG_SIZE[1],
                                       IMG_CHANNEL])

        y = model.forward(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt:
                ckpt.model_checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                Num = test_data.shape[0]
                ImgOut = np.zeros([Num, IMG_SIZE[0], IMG_SIZE[1],1], dtype=np.float32)
                for i in range(Num):
                    print(i)
                    img = sess.run(y, feed_dict={x: test_data[i:i + 1, :, :, :]})

                    ImgOut[i, :, :,:] = np.array(img)
                    #cv2.imwrite('{}.jpg'.format(i), img)
                scipy.io.savemat(save_name, {'mydata': ImgOut})
            else:
                print("No checkpoint is found.")
                return

if __name__=='__main__':
    # for i in range(1,19):
    #     data = h5py.File('./TestingData/tno-21/{}.mat' .format(i))
    #     input_data = data["inputs"]
    #     input_npy = np.transpose(input_data)
    #     print(input_npy.shape)
    #     save_name='{}.mat' .format(i)
    #     test(input_npy,save_name)
    data = h5py.File('./TestingData/L19.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)
    print(input_npy.shape)
    save_name = 'L19.mat'
    test(input_npy, save_name)

