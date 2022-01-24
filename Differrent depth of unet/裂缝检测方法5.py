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
#import shutil
#from PIL import ImageEnhance
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#负责控制台信息输出，主要是输出错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start_time=time.time()
#流程图
sess=tf.InteractiveSession()#构建会话的时候，我们可以先构建一个session然后再定义操作
#定义输入图像
net_input=tf.placeholder(dtype=tf.float32)#在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

#定义输入标签
net_label=tf.placeholder(dtype=tf.float32)
#输出的图像的通道数
kernel_num=32
#定义卷积核的大小
kernel_length=3

#开始定义第一层的卷积核
kernel_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,3,kernel_num],stddev=0.05))#高度，宽度，输入通道数，输出通道数
#偏置 b
bias_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))

kernel_2=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_2=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
# 定义变量，定义卷积核
kernel_3 = tf.Variable(initial_value=tf.random_normal(shape=[kernel_length, kernel_length, kernel_num, kernel_num], stddev=0.05))
bias_3=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_4=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_4=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))

kernel_5=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num*2],stddev=0.05))
bias_5=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))
kernel_6=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num*2],stddev=0.05))
bias_6=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))

kernel_7=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num*4],stddev=0.05))
bias_7=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*4],stddev=0.05))
kernel_8=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*4,kernel_num*4],stddev=0.05))
bias_8=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*4],stddev=0.05))

kernel_9=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*4,kernel_num*8],stddev=0.05))
bias_9=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*8],stddev=0.05))
kernel_10=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*8,kernel_num*8],stddev=0.05))
bias_10=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*8],stddev=0.05))

kernel_11=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*12,kernel_num*4],stddev=0.05))
bias_11=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*4],stddev=0.05))
kernel_12=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*4,kernel_num*4],stddev=0.05))
bias_12=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*4],stddev=0.05))

kernel_13=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*6,kernel_num*2],stddev=0.05))
bias_13=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))
kernel_14=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num*2],stddev=0.05))
bias_14=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))

kernel_15=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*3,kernel_num],stddev=0.05))
bias_15=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_16=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_16=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))

kernel_17=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num],stddev=0.05))
bias_17=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_18=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_18=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_19=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,2],stddev=0.05))
bias_19=tf.Variable(initial_value=tf.random_normal(shape=[2],stddev=0.05))
#输入和第一层卷积核心开始实现卷积
output_1=tf.nn.conv2d(input=net_input,filter=kernel_1,strides=[1,1,1,1],padding='VALID')+bias_1
#参数为输入，卷积核，滑动窗尺寸，卷积方式，VALID是不用填充边界
#激活函数（非线性函数，实现数据的非线性）
output_1_active=tf.nn.leaky_relu(features=output_1)
#用之前的卷积结果开始和第二个卷积核开始卷积
output_2=tf.nn.conv2d(input=output_1_active,filter=kernel_2,strides=[1,1,1,1],padding='VALID')+bias_2
output_2_active=tf.nn.leaky_relu(features=output_2)  #u-net第一层
#开始对数进行池化（最大池化），聚集信息（最大相应提取出来）
output_2_active_pool=tf.nn.max_pool(value=output_2_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')#参数为输入，池化窗口，步长，方式

output_3=tf.nn.conv2d(input=output_2_active_pool,filter=kernel_3,strides=[1,1,1,1],padding='VALID')+bias_3
output_3_active=tf.nn.leaky_relu(features=output_3)
output_4=tf.nn.conv2d(input=output_3_active,filter=kernel_4,strides=[1,1,1,1],padding='VALID')+bias_4
output_4_active=tf.nn.leaky_relu(features=output_4)   #u-net第二层
output_4_active_pool=tf.nn.max_pool(value=output_4_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_5=tf.nn.conv2d(input=output_4_active_pool,filter=kernel_5,strides=[1,1,1,1],padding='VALID')+bias_5
output_5_active=tf.nn.leaky_relu(features=output_5)
output_6=tf.nn.conv2d(input=output_5_active,filter=kernel_6,strides=[1,1,1,1],padding='VALID')+bias_6
output_6_active=tf.nn.leaky_relu(features=output_6)    #u-net第三层
output_6_active_pool=tf.nn.max_pool(value=output_6_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_7=tf.nn.conv2d(input=output_6_active_pool,filter=kernel_7,strides=[1,1,1,1],padding='VALID')+bias_7
output_7_active=tf.nn.leaky_relu(features=output_7)
output_8=tf.nn.conv2d(input=output_7_active,filter=kernel_8,strides=[1,1,1,1],padding='VALID')+bias_8
output_8_active=tf.nn.leaky_relu(features=output_8)  #u-net第四层
output_8_active_pool=tf.nn.max_pool(value=output_8_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_9=tf.nn.conv2d(input=output_8_active_pool,filter=kernel_9,strides=[1,1,1,1],padding='VALID')+bias_9
output_9_active=tf.nn.leaky_relu(features=output_9)
output_10=tf.nn.conv2d(input=output_9_active,filter=kernel_10,strides=[1,1,1,1],padding='VALID')+bias_10
output_10_active=tf.nn.leaky_relu(features=output_10)   #u-net最底层

#反池化过程，将图像扩大
output_10_active_unsample=tf.image.resize_images(images=output_10_active,size=[tf.shape(output_10_active)[1]*2,tf.shape(output_10_active)[2]*2])
# 上采样过程 up-conversation：2*2 的，也即是 原尺寸的长，宽均扩大为原来的 2 倍
#print(output_10_active_unsample)

#将要拼接的特征图修改为（裁剪或者填充）和上采样后的特征图相同的尺寸 便于结合。
output_8_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_8_active,target_height=tf.shape(output_10_active_unsample)[1],target_width=tf.shape(output_10_active_unsample)[2])
#开始实现拼接
output_10_add_output_8=tf.concat(values=[output_10_active_unsample,output_8_active_crop],axis=3) # 实现合并
output_11=tf.nn.conv2d(input=output_10_add_output_8,filter=kernel_11,strides=[1,1,1,1],padding='VALID')+bias_11
#激活卷积函数
output_11_active=tf.nn.leaky_relu(features=output_11)
#开始实现卷积，向上转换
output_12=tf.nn.conv2d(input=output_11_active,filter=kernel_12,strides=[1,1,1,1],padding='VALID')+bias_12
#先卷积激活（实际上非线性化）

output_12_active=tf.nn.leaky_relu(features=output_12)

#print(output_12_active)
output_12_active_unsample=tf.image.resize_images(images=output_12_active,size=[tf.shape(output_12_active)[1]*2,tf.shape(output_12_active)[2]*2])

output_6_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_6_active,target_height=tf.shape(output_12_active_unsample)[1],target_width=tf.shape(output_12_active_unsample)[2])
output_12_add_output_6=tf.concat(values=[output_12_active_unsample,output_6_active_crop],axis=3)
output_13=tf.nn.conv2d(input=output_12_add_output_6,filter=kernel_13,strides=[1,1,1,1],padding='VALID')+bias_13
output_13_active=tf.nn.leaky_relu(features=output_13)
output_14=tf.nn.conv2d(input=output_13_active,filter=kernel_14,strides=[1,1,1,1],padding='VALID')+bias_14
output_14_active=tf.nn.leaky_relu(features=output_14)

output_14_active_unsample=tf.image.resize_images(images=output_14_active,size=[tf.shape(output_14_active)[1]*2,tf.shape(output_14_active)[2]*2])
output_4_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_4_active,target_height=tf.shape(output_14_active_unsample)[1],target_width=tf.shape(output_14_active_unsample)[2])
output_14_add_output_4=tf.concat([output_14_active_unsample,output_4_active_crop],axis=3)
output_15=tf.nn.conv2d(input=output_14_add_output_4,filter=kernel_15,strides=[1,1,1,1],padding='VALID')+bias_15
output_15_active=tf.nn.leaky_relu(features=output_15)
output_16=tf.nn.conv2d(input=output_15_active,filter=kernel_16,strides=[1,1,1,1],padding='VALID')+bias_16
output_16_active=tf.nn.leaky_relu(features=output_16)


output_16_active_unsample=tf.image.resize_images(images=output_16_active,size=[tf.shape(output_16_active)[1]*2,tf.shape(output_16_active)[2]*2])
output_2_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_2_active,target_height=tf.shape(output_16_active_unsample)[1],target_width=tf.shape(output_16_active_unsample)[2])
output_16_add_output_2=tf.concat([output_16_active_unsample,output_2_active_crop],axis=3)
output_17=tf.nn.conv2d(input=output_16_add_output_2,filter=kernel_17,strides=[1,1,1,1],padding='VALID')+bias_17
output_17_active=tf.nn.leaky_relu(features=output_17)
output_18=tf.nn.conv2d(input=output_17_active,filter=kernel_18,strides=[1,1,1,1],padding='VALID')+bias_18
output_18_active=tf.nn.leaky_relu(features=output_18)

#最后一层，结束卷积
output_end=tf.nn.conv2d(input=output_18_active,filter=kernel_19,strides=[1,1,1,1],padding='VALID')+bias_19

#三维转二维
output_end_squeeze=tf.squeeze(input=output_end)#tf.squeeze()函数的作用是从tensor中删除所有大小(szie)是1的维度。

#开始计算差异（卷积之后的结果和标签图片）
#切除标签图像为了 便于计算差异
net_label_crop=tf.image.resize_image_with_crop_or_pad(image=net_label,target_height=tf.shape(output_end)[1],target_width=tf.shape(output_end)[2])
#交叉熵损失函数
cross_entropy_all=tf.nn.softmax_cross_entropy_with_logits(labels=net_label_crop,logits=output_end_squeeze)
#一幅图片的平均交叉熵损失函数
cross_entropy=tf.reduce_mean(cross_entropy_all)
# 执行，运用模型进行检测
saver = tf.train.Saver()
saver.restore(sess, "./Model/Model5/model.ckpt")
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
detection=[]
time1=[]
run=[]

for filename in os.listdir(r'./picture/image/'):  # listdir的参数是文件夹的路径
    index = index + 1
    imgname = filename[:-4]
    print(imgname)
    Img = cv2.imread('./picture/image/' + filename)
    # 给原图周围填充50个像素
    print("-----mirror_img-----")
    size = Img.shape
    print(size)#获得图像的长宽
    after_mirror_h = size[1] + 100
    after_mirror_w = size[0] + 100#长宽各填充50个像素
    mirror_img = zeros((after_mirror_w, after_mirror_h), dtype=int)#
    mirror_img = cv2.copyMakeBorder(Img,100, 100, 100, 100, cv2.BORDER_REFLECT)  # 轴对称扩充 相当于镜像填充，给img上下左右各填充50，倒映填充

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

    detection_start=time.time()

    for i in range(0, w):
        # 上句W 的值修改过
        for j in range(0, h):
            numb = numb + 1;
            # 上句h 的值修改过
            # img = Img[i * 640:i * 640 + 740, j * 640:j * 640+ 740]
            img = mirror_img[i * S:(i+1) * S + 200, j * S:(j+1) * S +200]
            #cv2.imwrite('./' + str(
            #    "%03d" % (101 + numb)) + '.jpg', img)  # 保存图像
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
    print("图像拼合完成")
    print("-----裂缝检测结束-----")
    detection_end = time.time()
    detection_time = detection_end - detection_start
    print("第" + str(index) + "张图像分割时间为" + str(detection_time) + "s")
    print("第" + str(index) + "张图像分割时间为" + str(detection_time / 60) + "min")
    detection.append(detection_time)
    print(detection)
    np.savetxt("./picture/image/" + "detection" + ".txt", detection)

    end_time = time.time()
    time1.append(end_time)
    if (index == 1):
        run_time = end_time - start_time
    elif (index != 1):
        run_time = time1[index - 1] - time1[index - 2]
    print("第" + str(index) + "张图像检测时间为" + str(run_time) + "s")
    print("第" + str(index) + "张图像检测时间为" + str(run_time / 60) + "min")
    run.append(run_time)
    print(run)
    np.savetxt("./picture/image/" + "time" + ".txt", run)

end_time = time.time()
run_time = -(start_time-end_time)
print("代码运行时间为："+str(run_time)+"s")
print("代码运行时间为："+str(run_time/60)+"min")
