import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,function = None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    Bias = tf.Variable(tf.random_normal([1,out_size]))

    WX_B = tf.add(tf.matmul(inputs,Weight),Bias)
    if function is None:
        return WX_B
    else:
        return function(WX_B)

x = np.linspace(0,10)[:,np.newaxis]         # Test function 1
noise = np.random.normal(0,0.05,x.shape)
y = np.sin(x)+noise+np.cos(x)

# x = np.linspace(-1,1,300)[:,np.newaxis]   #Test function 2
# noise  = np.random.normal(0,0.05,x.shape)
# y = np.square(x) + noise - 0.5+x*2


X = tf.placeholder(tf.float32,[None,1])
Y = tf.placeholder(tf.float32,[None,1])


l1_1 = add_layer(X,1,10,function = tf.nn.relu)
prediction_value_1 = add_layer(l1_1,10,1,function = None)

l1_2 = add_layer(X,1,10,function = tf.nn.relu)
l2_2 = add_layer(l1_2,10,10,function = None)
prediction_value_2 = add_layer(l2_2,10,1,function = None)

l1_3 = add_layer(X,1,10,function = tf.nn.relu)
l2_3 = add_layer(l1_3,10,10,function = None)
l3_3 = add_layer(l2_3,10,10,function = None)
prediction_value_3 = add_layer(l3_3,10,1,function = None)



loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(prediction_value_1 - y),
reduction_indices=[1]))
loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(prediction_value_2 - y),
reduction_indices=[1]))
loss3 = tf.reduce_mean(tf.reduce_sum(tf.square(prediction_value_3 - y),
reduction_indices=[1]))

step = 0
iteration = 1000
lr = 0.0000001


train_step1 = tf.train.GradientDescentOptimizer(lr).minimize(loss1)
train_step2 = tf.train.GradientDescentOptimizer(lr).minimize(loss2)
train_step3 = tf.train.GradientDescentOptimizer(lr).minimize(loss3)

lossList1 = []
lossList2 = []
lossList3 = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(iteration):
        sess.run(train_step1,feed_dict = {X:x,Y:y})     # first model
        loss = sess.run(loss1,feed_dict = {X:x,Y:y})
        lossList1.append(loss)

        sess.run(train_step2,feed_dict = {X:x,Y:y})     #second model
        loss = sess.run(loss2,feed_dict = {X:x,Y:y})
        lossList2.append(loss)

        sess.run(train_step3,feed_dict = {X:x,Y:y})     #third model
        loss = sess.run(loss3,feed_dict = {X:x,Y:y})
        lossList3.append(loss)

        if step % 50 is 0:
            loss = sess.run(loss1,feed_dict = {X:x,Y:y})
            lossList1.append(loss)

plt.plot(range(len(lossList1)),lossList1,color = 'r',label="one hidden layer")
plt.plot(range(len(lossList2)),lossList2,color = 'b',label="two hidden layers")
plt.plot(range(len(lossList3)),lossList3,color = 'y',label="three hidden layers")
print "one hidden layer:"
print lossList1,"\n"
print "two hidden layers:"
print lossList2,"\n"
print "three hidden layers:"
print lossList3,"\n"

plt.legend()
plt.show()
