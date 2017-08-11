# -*-coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys, os

if __name__ == '__main__':
    train_dir = os.path.join('demo_model/', "demo")
    a = tf.placeholder(dtype=tf.int32, shape=None, name='a')
    b = tf.placeholder(dtype=tf.int32, shape=None, name='b')
    y = tf.Variable(tf.ones(shape=[1], dtype=tf.int32), dtype=tf.int32, name='y')
    res = tf.add(tf.multiply(a, b), y, name='res')
    with tf.Session() as sess:
        feed_dict = dict()
        feed_dict[a] = 2
        feed_dict[b] = 3
        fetch_list = [res]
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        res = sess.run(feed_dict=feed_dict, fetches=fetch_list)
        saver.save(sess, train_dir)
        print(res[0])
        print(type(res[0]))
