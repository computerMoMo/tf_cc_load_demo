# -*- coding:utf-8 -*-
import tensorflow as tf
import os

if __name__ == '__main__':
    # train_dir = os.path.join('demo_model/', "demo")
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('demo_model/demo.meta')
        saver.restore(sess, tf.train.latest_checkpoint('demo_model/'))
        # sess.run()
        graph = tf.get_default_graph()
        a = graph.get_tensor_by_name("a:0")
        b = graph.get_tensor_by_name("b:0")
        feed_dict = {a: 2, b: 3}

        op_to_restore = graph.get_tensor_by_name("res:0")
        print(sess.run(fetches=op_to_restore, feed_dict=feed_dict))