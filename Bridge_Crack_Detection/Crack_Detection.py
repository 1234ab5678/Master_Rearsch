# coding=gbk
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from numpy import *
from PIL import Image
#from scipy import misc
import imageio
import shutil
from PIL import ImageEnhance
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#负责控制台信息输出，主要是输出错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_time = time.time()

sess = tf.InteractiveSession()

net_input = tf.placeholder(dtype=tf.float32)
net_label = tf.placeholder(dtype=tf.float32)
# 重新定义一遍卷积核
kernel_num = 32
kernel_length = 3
kernel_1 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_length, kernel_length, 3, kernel_num], stddev=0.05))
bias_1 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))
kernel_2 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num], stddev=0.05))
bias_2 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))

kernel_3 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num], stddev=0.05))
bias_3 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))
kernel_4 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num], stddev=0.05))
bias_4 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))

kernel_5 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num * 2], stddev=0.05))
bias_5 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 2], stddev=0.05))
kernel_6 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 2, kernel_num * 2], stddev=0.05))
bias_6 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 2], stddev=0.05))

kernel_7 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 2, kernel_num * 4], stddev=0.05))
bias_7 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 4], stddev=0.05))
kernel_8 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 4, kernel_num * 4], stddev=0.05))
bias_8 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 4], stddev=0.05))

kernel_9 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 6, kernel_num * 2], stddev=0.05))
bias_9 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 2], stddev=0.05))
kernel_10 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 2, kernel_num * 2], stddev=0.05))
bias_10 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num * 2], stddev=0.05))

kernel_11 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 3, kernel_num], stddev=0.05))
bias_11 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))
kernel_12 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num], stddev=0.05))
bias_12 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))

kernel_13 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num * 2, kernel_num], stddev=0.05))
bias_13 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_num], stddev=0.05))
kernel_14 = tf.Variable(
    initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, 2], stddev=0.05))
bias_14 = tf.Variable(initial_value=tf.random_normal(shape=[2], stddev=0.05))

output_1 = tf.nn.conv2d(input=net_input, filter=kernel_1, strides=[1, 1, 1, 1], padding='VALID') + bias_1
output_1_active = tf.nn.leaky_relu(features=output_1)
output_2 = tf.nn.conv2d(input=output_1_active, filter=kernel_2, strides=[1, 1, 1, 1], padding='VALID') + bias_2
output_2_active = tf.nn.leaky_relu(features=output_2)  # u-net第一层

output_2_active_pool = tf.nn.max_pool(value=output_2_active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
output_3 = tf.nn.conv2d(input=output_2_active_pool, filter=kernel_3, strides=[1, 1, 1, 1], padding='VALID') + bias_3
output_3_active = tf.nn.leaky_relu(features=output_3)
output_4 = tf.nn.conv2d(input=output_3_active, filter=kernel_4, strides=[1, 1, 1, 1], padding='VALID') + bias_4
output_4_active = tf.nn.leaky_relu(features=output_4)  # u-net第二层

output_4_active_pool = tf.nn.max_pool(value=output_4_active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
output_5 = tf.nn.conv2d(input=output_4_active_pool, filter=kernel_5, strides=[1, 1, 1, 1], padding='VALID') + bias_5
output_5_active = tf.nn.leaky_relu(features=output_5)
output_6 = tf.nn.conv2d(input=output_5_active, filter=kernel_6, strides=[1, 1, 1, 1], padding='VALID') + bias_6
output_6_active = tf.nn.leaky_relu(features=output_6)  # u-net第三层

output_6_active_pool = tf.nn.max_pool(value=output_6_active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
output_7 = tf.nn.conv2d(input=output_6_active_pool, filter=kernel_7, strides=[1, 1, 1, 1], padding='VALID') + bias_7
output_7_active = tf.nn.leaky_relu(features=output_7)
output_8 = tf.nn.conv2d(input=output_7_active, filter=kernel_8, strides=[1, 1, 1, 1], padding='VALID') + bias_8
output_8_active = tf.nn.leaky_relu(features=output_8)  # u-net最底层

output_8_active_unsample = tf.image.resize_images(images=output_8_active, size=[tf.shape(output_8_active)[1] * 2,
                                                                                tf.shape(output_8_active)[2] * 2])
output_6_active_crop = tf.image.resize_image_with_crop_or_pad(image=output_6_active,
                                                              target_height=tf.shape(output_8_active_unsample)[1],
                                                              target_width=tf.shape(output_8_active_unsample)[2])
output_8_add_output_6 = tf.concat(values=[output_8_active_unsample, output_6_active_crop], axis=3)
output_9 = tf.nn.conv2d(input=output_8_add_output_6, filter=kernel_9, strides=[1, 1, 1, 1], padding='VALID') + bias_9
output_9_active = tf.nn.leaky_relu(features=output_9)
output_10 = tf.nn.conv2d(input=output_9_active, filter=kernel_10, strides=[1, 1, 1, 1], padding='VALID') + bias_10
output_10_active = tf.nn.leaky_relu(features=output_10)

output_10_active_unsample = tf.image.resize_images(images=output_10_active, size=[tf.shape(output_10_active)[1] * 2,
                                                                                  tf.shape(output_10_active)[2] * 2])
output_4_active_crop = tf.image.resize_image_with_crop_or_pad(image=output_4_active,
                                                              target_height=tf.shape(output_10_active_unsample)[1],
                                                              target_width=tf.shape(output_10_active_unsample)[2])
output_10_add_output_4 = tf.concat(values=[output_10_active_unsample, output_4_active_crop], axis=3)
output_11 = tf.nn.conv2d(input=output_10_add_output_4, filter=kernel_11, strides=[1, 1, 1, 1],
                         padding='VALID') + bias_11
output_11_active = tf.nn.leaky_relu(features=output_11)
output_12 = tf.nn.conv2d(input=output_11_active, filter=kernel_12, strides=[1, 1, 1, 1], padding='VALID') + bias_12
output_12_active = tf.nn.leaky_relu(features=output_12)

output_12_active_unsample = tf.image.resize_images(images=output_12_active, size=[tf.shape(output_12_active)[1] * 2,
                                                                                  tf.shape(output_12_active)[2] * 2])
output_2_active_crop = tf.image.resize_image_with_crop_or_pad(image=output_2_active,
                                                              target_height=tf.shape(output_12_active_unsample)[1],
                                                              target_width=tf.shape(output_12_active_unsample)[2])
output_12_add_output_2 = tf.concat([output_12_active_unsample, output_2_active_crop], axis=3)
output_13 = tf.nn.conv2d(input=output_12_add_output_2, filter=kernel_13, strides=[1, 1, 1, 1],
                         padding='VALID') + bias_13
output_13_active = tf.nn.leaky_relu(features=output_13)
output_end = tf.nn.conv2d(input=output_13_active, filter=kernel_14, strides=[1, 1, 1, 1], padding='VALID') + bias_14
output_end_squeeze = tf.squeeze(input=output_end)

net_label_crop = tf.image.resize_image_with_crop_or_pad(image=net_label, target_height=tf.shape(output_end)[1],
                                                        target_width=tf.shape(output_end)[2])
cross_entropy_all = tf.nn.softmax_cross_entropy_with_logits(labels=net_label_crop, logits=output_end_squeeze)
cross_entropy = tf.reduce_mean(cross_entropy_all)
# 执行，运用模型进行检测
saver = tf.train.Saver()
saver.restore(sess, "./Model/model.ckpt")
#saver.restore(sess, "D:\dlbridge\Crack_detection_Code\Model\model.ckpt")


print("-----start-----")
# 求最大公约数
def gcd(x ,y):
    if x % y == 0:
        return y
    else:
        return gcd(y, x % y)

# 依次读取
# 将输入图片通过镜像填充的方法周围扩展50个像素点，（3840*5760）变成（3940*5860）
# 再将图片分割为740*740的大小，框架输出尺寸为652*652，拼合尺寸为640*640，最终输出原始图片尺寸大小。
# 只需自行输入需要拼合用到的小图尺寸大小，一般裁剪为正方形图片，宽度和高度相等，都等于 S .
index = 0
median = "median"
os.mkdir("./" + median)

for filename in os.listdir(r'./picture/image/'):  # listdir的参数是文件夹的路径
    index = index + 1
    imgname = filename[:-4]
    print(imgname)
    Img = cv2.imread('./picture/image/' + filename)
    # 给原图周围填充50个像素
    print("-----mirror_img-----")
    size = Img.shape
    print(size)#获得图像的长宽
    after_mirror_h = size[1] + 25
    after_mirror_w = size[0] + 25#长宽各填充50个像素
    mirror_img = zeros((after_mirror_w, after_mirror_h), dtype=int)#
    mirror_img = cv2.copyMakeBorder(Img, 50, 50, 50, 50, cv2.BORDER_REFLECT)  # 轴对称扩充 相当于镜像填充，给img上下左右各填充50，倒映填充

    print("-----cut_img-----")
    # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\picture\\mirror_img\\"+ imgname +".jpg", mirror_img)
    S = gcd(size[0], size[1])#求最大公因数
    while S > 800:
         S = int(S / 2)
    else:
        S = S
    # print(S)
    # （3840*5760）S = 640
    # （3120*4160） S = 520
    # S = 640 # the size of Stitch pictures width = high = S , input it by yourself

    h = int(size[1] / S)
    w = int(size[0] / S)
    #test = 'test' + str("%02d" % (00 + index))
    #os.mkdir("/home/xq/ZSY_CrackDetection/crack detection/median/" + test)
    test_output = 'test_output' + str("%02d" % (00 + index))
    # print(test_output)
    os.mkdir("./median/" + test_output)
    numb = 0
    for i in range(0, w):
        # 上句W 的值修改过
        for j in range(0, h):
            numb = numb + 1;
            # 上句h 的值修改过
            # img = Img[i * 640:i * 640 + 740, j * 640:j * 640+ 740]
            img = mirror_img[i * S:(i+1) * S + 100, j * S:(j+1) * S +100]
            img2 = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]]) / 255
            out = output_end_squeeze.eval(feed_dict={net_input: img2})
            # print(out.shape)
            out_size = out.shape
            out_h = out_size[1]
            out_w = out_size[0]
            np.savetxt(
                './median/' + test_output + '/out0' + str(100 +numb) + '.txt',
                out[:, :, 0])
            np.savetxt(
                './median/' + test_output + '/out1' + str(100 +numb) + '.txt',
                out[:, :, 1])
            print("out0:"+test_output + '/out0' + str(100 +numb) + '.txt')
            print("out1:"+test_output + '/out1' + str(100 +numb) + '.txt')
    print("-----Image.fromarray-----")
    # 创建输出图片空矩阵
    out_end = zeros((size[0], size[1]), dtype=int)
    # 读取txt并得到输出矩阵
    num = 0
    out = zeros((out_w, out_h), dtype=int)
    d = int((out_w - S) / 2)
    print("卷积后图像尺寸差为：" + str(d))
    for i in range(0, w):
        for j in range(0, h):
            num = num + 1
            print(num)
            out0 = loadtxt('./median/' + test_output + '/out0' + str(
                        "%03d" % (100 + num) + '.txt'))
            out1 = loadtxt('./median/' + test_output + '/out1' + str(
                        "%03d" % (100 + num) + '.txt'))
            for m in range(0, out_w):
                for n in range(0, out_h):
                    if (out0[m][n]) > out1[m][n]:
                        out[m][n] = 1
                    else:
                        out[m][n] = 0
            for m in range(0, S):
                for n in range(0, S):
                    out_end[i * S + m][j * S + n] = out[d + m][d + n]
    # 输出矩阵显示为图片
    #print(out_end)
    # img = Image.fromarray(out_end)
    img = Image.fromarray(out_end.astype('uint8'))
    #img.show()
    # misc.imsave('C:\\Users\\Administrator\\Desktop\\picture\\image\\output' + str("%02d" % (00 + k)) +"_dp"+ ".png",out_end * 255)
    imageio.imsave('./picture/image/output_' + str(imgname) + ".png",out_end * 255)

end_time = time.time()
run_time = -(start_time-end_time)
print("代码运行时间为："+str(run_time)+"s")
print("代码运行时间为："+str(run_time/60)+"min")
