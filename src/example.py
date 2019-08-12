import tensorflow as tf
import numpy as np
import time

slowmodel=False

def swish_activation(x):
    return x*tf.sigmoid(x) 
    
def dilated_conv(net, total_filters, r, name):
        conv = tf.contrib.layers.conv2d(
            net,
            total_filters,
            [3,3],
            rate=r,
            stride=1,
            scope=name+'/a'+str(r),
            activation_fn=swish_activation,
            normalizer_fn=None,
            weights_regularizer=None,
        )
        conv = tf.contrib.layers.batch_norm(
                conv,
                decay=0.90,
                scale=True,
                is_training=True,
                zero_debias_moving_mean=False,
                scope=name+'/bn', fused=True)
        return conv
    
def dilated_conv_dw_pw(net, total_filters, r, name):
        depthwise_filter = tf.get_variable(
            name = name + '/dw_' + str(r),
            shape = [3, 3, net.get_shape()[-1], 1],
            initializer = tf.contrib.layers.xavier_initializer(),
            regularizer = None,
            trainable = True,
        )
        pointwise_filter = tf.get_variable(
            name = name + '/pw_' + str(r),
            shape = [1, 1, net.get_shape()[-1], total_filters],
            initializer = tf.contrib.layers.xavier_initializer(),
            regularizer = None,
        )
        bias = tf.get_variable(
            name = name + '/bias_' + str(r),
            shape = [total_filters],
            initializer = tf.zeros_initializer(),
            trainable = True,
        )
        # Note we tried replacing separable convs with a depthwise followed by 1x1 convolutions
        # (which should be equivalent), and that did not help performance. If the details would help, lmk
        c = tf.nn.separable_conv2d(
            net,
            depthwise_filter,
            pointwise_filter,
            strides = [1, 1, 1, 1],
            padding = 'SAME',
            rate = [r, r],
            name = name + '/depthwise_separable_conv_' + str(r),
        )
        conv = swish_activation(c + bias)
        conv = tf.contrib.layers.batch_norm(
                conv,
                decay=0.90,
                scale=True,
                is_training=True,
                zero_debias_moving_mean=False,
                scope=name+'/bn', fused=True)
        return conv

def create_model(use_dwpw, net):
    num_filters = 104
    dilations = 8  
                
    if use_dwpw:
        net = dilated_conv_dw_pw(net, num_filters, dilations, 'dialconv1')
        depthwise_filter = tf.get_variable(
                    name='dw_1',
                    shape=[3, 3, net.get_shape()[-1], 1],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=None,
                    trainable=True,
                )
        pointwise_filter = tf.get_variable(
                    name='pw_1',
                    shape=[1, 1, net.get_shape()[-1], num_filters * 2],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    regularizer=None,
                    trainable=True,
                )
        bias = tf.get_variable(
                    name='bias_1',
                    shape=[num_filters * 2],
                    initializer=tf.zeros_initializer(),
                    trainable=True,
                )
        net = tf.nn.separable_conv2d(
                    net,
                    depthwise_filter,
                    pointwise_filter,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    rate=[1, 1],
                    name='depthwise_separable_conv_1',
                )
        net = swish_activation(net + bias)
    else:
        net = dilated_conv(net, num_filters, dilations, 'dialconv1')
        net = tf.contrib.layers.conv2d(
            net,
            num_filters*2,
            [3,3],
            scope='c1',
            stride=1,
            padding='SAME',
            activation_fn=swish_activation,
            normalizer_fn=None,
            weights_regularizer=None,
        )
            
    poolings_per_layer = [2, 2]
    pooling_stride_per_layer = [2, 2]
    pooling_dim = poolings_per_layer
    pooling_stride = pooling_stride_per_layer
    net = tf.layers.max_pooling2d(
        net, pool_size=pooling_dim, strides=pooling_stride, padding='SAME', name='pool1')
          
    return net
            
            
def loss_distance_metric_sum(gold, assigned):
    distance_loss = tf.abs(gold-assigned)
    distance_loss = tf.reduce_sum(distance_loss)
    return distance_loss
 
# fake data; fake truth
theinput = np.random.rand(8,512,512,6)
labels = np.random.rand(8, 256, 256, 208)

x = tf.placeholder(dtype = tf.float32, shape = [None, 512,512,6])

net = create_model(slowmodel, x)

loss = loss_distance_metric_sum(labels, net)
train_op = tf.train.AdamOptimizer(
                0.001, beta1=0.80, beta2=0.99,
                epsilon=0.0001).minimize(loss)

tower_loss = []
tower_loss.append(loss)
avg_loss = tf.reduce_mean(tower_loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(201):
    start_time = time.time()
    _, res = sess.run([train_op, avg_loss], feed_dict={x: theinput})
    if i % 10 == 0:
        print("Loss: ", res)
    print(str(i) + ": " + str(time.time() - start_time))