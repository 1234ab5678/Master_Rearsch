# coding=gbk
#˼·�����ȷָ�ͼ��Ȼ��ͼ����࣬���ѷ�ͼ������������м�⣬��󽫽��ƴ�ϻ�ý��ͼ��
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from numpy import *
#from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#�������̨��Ϣ�������Ҫ���������

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

ckpt = tf.train.get_checkpoint_state("./Model/Model5_50epoch/").model_checkpoint_path
saver = tf.train.import_meta_graph(ckpt+'.meta')
variables = tf.trainable_variables()
#total_parameters = 0
#for variable in variables:
#    shape = variable.get_shape()
#    variable_parameters = 1
#    for dim in shape:
        # print(dim)
 #       variable_parameters *= dim.value
    # print(variable_parameters)
#    total_parameters += variable_parameters
#print(total_parameters)

graph =tf.get_default_graph()
print(stats_graph(graph))
