import numpy as np
import tensorflow as tf
import PIL.Image
from functools import partial
import urllib.request
import os
import scipy.misc

model_fn = 'tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(os.path.join('data_dir', model_fn), 'rb')as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D'and 'import/'in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1])for name in layers]
# print('Number of layers', len(layers))
# print('total number of feature channels:', sum(feature_nums))


def save_array(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved:%s' % img_name)


name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0


def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)  # define optimizatipn object
    t_grad = tf.gradients(t_score, t_input)[0]  # 用梯度下降法
    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        g /= g.std() + 1e-8
        img += g*step
        print('score(mean)=%f' % score)
    save_array(img, 'naive.jpg')


layer_output = graph.get_tensor_by_name("import/%s:0" % name)
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=20)


