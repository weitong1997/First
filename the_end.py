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


def calc_grad_tiled(img, t_grad, title_size=512):

    sz = title_size
    h, w = img.shape[:2]
    # img_shift 先在行上做整体移动，再在列上做整体移动
    # 防止出现边缘效应
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            # 每次对sub计算梯度。sub的大小是title_size*title_size
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def savearray(img_array, img_name):

    scipy.misc.toimage(img_array).save(img_name)
    print('img saved : %s' % img_name)


def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, hw))
    img = img / 255 * (max - min) + min
    return img


def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0
    # 同样将图像进行金字塔分解
    # 提取高频和低频的方法比较简单，直接缩放
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # 先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')

    img = img.clip(0, 255)
    savearray(img, 'deepdream.jpg')


if __name__ == '__main__':
    img0 = PIL.Image.open('images/back.jpg')
    img0 = np.float32(img0)

    # name = 'mixed4d_3x3_bottleneck_pre_relu'
    name = 'mixed4c'
    # channel = 139
    layer_output = graph.get_tensor_by_name('import/%s:0' % name)
    # render_deepdream(layer_output[:, :, :, channel], img0, iter_n=150)
    render_deepdream(tf.square(layer_output), img0)
