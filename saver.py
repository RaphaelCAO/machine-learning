import numpy as np
import tensorflow as tf

def restore(path):
    Weight = tf.Variable([2,1,1],dtype = tf.float32)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        old_Weight = sess.run(Weight)
        saver.restore(sess,path)
        print "old Weight:",old_Weight,"\t--->\t","new Weight:",sess.run(Weight)

def save(path):
    Weight = tf.Variable([11,22,33],dtype = tf.float32)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(path):
            saver.save(sess,path)
        else:
            os.mkdir("data_lib")
            saver.save(sess,path)
        print sess.run(Weight)

# save("data_lib/save.ckpt")
restore("data_lib/save.ckpt")
