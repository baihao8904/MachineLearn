import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    
x_data2 = np.linspace(-1,1,300)[:,np.newaxis]
#前两个参数为均值和方差
noise = np.random.normal(0,0.05,x_data2.shape)
y_data2 = np.square(x_data2)-0.5+noise
xs = tf.placeholder(tf.float32,[None,1]) 
ys = tf.placeholder(tf.float32,[None,1])


l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
predition = add_layer(l1,10,1,activation_function=None)
#计算loss 求和之后求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()

sess5 = tf.Session()
sess5.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data2,y_data2)
plt.ion()
plt.show()
for i in range(1000):
    sess5.run(train_step,feed_dict={xs:x_data2,ys:y_data2})
    if i%50==0:
        #print(sess5.run(predition,feed_dict={xs:x_data2,ys:y_data2}))
        #print(sess5.run(loss,feed_dict={xs:x_data2,ys:y_data2}))
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        predition_value = sess5.run(predition,feed_dict={xs:x_data2})
        lines = ax.plot(x_data2,predition_value,'r-',lw=5)
        plt.pause(0.2)
        
time.sleep(10)