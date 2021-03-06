import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def addLayer(inputs,in_size,out_size,activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    Wx_b = tf.matmul(inputs,Weight)
    if activation_function is None:
        outputs = Wx_b
    else:
        outputs = activation_function(Wx_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise  = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) + noise - 0.5
inputs = np.hstack((x_data,np.ones((300,1))))

xs = tf.placeholder(tf.float32,[None,2])
ys = tf.placeholder(tf.float32,[None,1])

l1 = addLayer(xs,2,10,activation_function = tf.nn.relu)
prediction = addLayer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
        reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
losses_dig = []
plt.figure("data - prediction presentation")
plt.scatter(x_data,y_data)
plt.ion()
plt.show()
ax = plt.gca()

for step in range(1000):
    sess.run(train_step,feed_dict={xs:inputs,ys:y_data})
    prediction_value = sess.run(prediction,feed_dict={xs:inputs})


    if step%50==0:
        num =  sess.run(loss,feed_dict={xs:inputs,ys:y_data})
        losses_dig.append(num)
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        lines = ax.plot(x_data,prediction_value,'red')
        plt.pause(0.1)
plt.ioff()
plt.figure("loss")
plt.plot(range(len(losses_dig)),losses_dig)


plt.show()
