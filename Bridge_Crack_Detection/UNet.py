#coding=gbk
import tensorflow as tf
import numpy as np
import cv2
import os
#不报警告
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


# 开始定义第一层的卷积核
kernel_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,3,kernel_num],stddev=0.05))#高度，宽度，输入通道数，输出通道数

#偏执 b
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

kernel_9=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*6,kernel_num*2],stddev=0.05))
bias_9=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))
kernel_10=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num*2],stddev=0.05))
bias_10=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num*2],stddev=0.05))

kernel_11=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*3,kernel_num],stddev=0.05))
bias_11=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_12=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_12=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))

kernel_13=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num*2,kernel_num],stddev=0.05))
bias_13=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
kernel_14=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,2],stddev=0.05))
bias_14=tf.Variable(initial_value=tf.random_normal(shape=[2],stddev=0.05))

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
output_8_active=tf.nn.leaky_relu(features=output_8)   #u-net最底层

#反池化过程，将图像扩大
output_8_active_unsample=tf.image.resize_images(images=output_8_active,size=[tf.shape(output_8_active)[1]*2,tf.shape(output_8_active)[2]*2])
# 上采样过程 up-conversation：2*2 的，也即是 原尺寸的长，宽均扩大为原来的 2 倍

#将要拼接的特征图修改为（裁剪或者填充）和上采样后的特征图相同的尺寸 便于结合。
output_6_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_6_active,target_height=tf.shape(output_8_active_unsample)[1],target_width=tf.shape(output_8_active_unsample)[2])
#开始实现拼接
output_8_add_output_6=tf.concat(values=[output_8_active_unsample,output_6_active_crop],axis=3) # 实现合并
output_9=tf.nn.conv2d(input=output_8_add_output_6,filter=kernel_9,strides=[1,1,1,1],padding='VALID')+bias_9
#激活卷积函数
output_9_active=tf.nn.leaky_relu(features=output_9)
#开始实现卷积，向上转换
output_10=tf.nn.conv2d(input=output_9_active,filter=kernel_10,strides=[1,1,1,1],padding='VALID')+bias_10
#先卷积激活（实际上非线性化）

output_10_active=tf.nn.leaky_relu(features=output_10)
output_10_active_unsample=tf.image.resize_images(images=output_10_active,size=[tf.shape(output_10_active)[1]*2,tf.shape(output_10_active)[2]*2])
output_4_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_4_active,target_height=tf.shape(output_10_active_unsample)[1],target_width=tf.shape(output_10_active_unsample)[2])
output_10_add_output_4=tf.concat(values=[output_10_active_unsample,output_4_active_crop],axis=3)
output_11=tf.nn.conv2d(input=output_10_add_output_4,filter=kernel_11,strides=[1,1,1,1],padding='VALID')+bias_11
output_11_active=tf.nn.leaky_relu(features=output_11)
output_12=tf.nn.conv2d(input=output_11_active,filter=kernel_12,strides=[1,1,1,1],padding='VALID')+bias_12
output_12_active=tf.nn.leaky_relu(features=output_12)

output_12_active_unsample=tf.image.resize_images(images=output_12_active,size=[tf.shape(output_12_active)[1]*2,tf.shape(output_12_active)[2]*2])
output_2_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_2_active,target_height=tf.shape(output_12_active_unsample)[1],target_width=tf.shape(output_12_active_unsample)[2])
output_12_add_output_2=tf.concat([output_12_active_unsample,output_2_active_crop],axis=3)
output_13=tf.nn.conv2d(input=output_12_add_output_2,filter=kernel_13,strides=[1,1,1,1],padding='VALID')+bias_13
output_13_active=tf.nn.leaky_relu(features=output_13)
#最后一层，结束卷积
output_end=tf.nn.conv2d(input=output_13_active,filter=kernel_14,strides=[1,1,1,1],padding='VALID')+bias_14

#三维转二维
output_end_squeeze=tf.squeeze(input=output_end)#tf.squeeze()函数的作用是从tensor中删除所有大小(szie)是1的维度。

#开始计算差异（卷积之后的结果和标签图片）
#切除标签图像为了 便于计算差异
net_label_crop=tf.image.resize_image_with_crop_or_pad(image=net_label,target_height=tf.shape(output_end)[1],target_width=tf.shape(output_end)[2])
#交叉熵损失函数
cross_entropy_all=tf.nn.softmax_cross_entropy_with_logits(labels=net_label_crop,logits=output_end_squeeze)
#一幅图片的平均交叉熵损失函数
cross_entropy=tf.reduce_mean(cross_entropy_all)
#定义正则化
regularization=tf.nn.l2_loss(kernel_1)+tf.nn.l2_loss(bias_1) + tf.nn.l2_loss(kernel_2)+tf.nn.l2_loss(bias_2) \
                +tf.nn.l2_loss(kernel_3)+tf.nn.l2_loss(bias_3) + tf.nn.l2_loss(kernel_4)+tf.nn.l2_loss(bias_4) \
                +tf.nn.l2_loss(kernel_5)+tf.nn.l2_loss(bias_5)  + tf.nn.l2_loss(kernel_6)+tf.nn.l2_loss(bias_6) \
                +tf.nn.l2_loss(kernel_7)+tf.nn.l2_loss(bias_7)  + tf.nn.l2_loss(kernel_8)+tf.nn.l2_loss(bias_8) \
                +tf.nn.l2_loss(kernel_9)+tf.nn.l2_loss(bias_9)  + tf.nn.l2_loss(kernel_10)+tf.nn.l2_loss(bias_10) \
                +tf.nn.l2_loss(kernel_11)+tf.nn.l2_loss(bias_11)  + tf.nn.l2_loss(kernel_12)+tf.nn.l2_loss(bias_12) \
                +tf.nn.l2_loss(kernel_13)+tf.nn.l2_loss(bias_13)  + tf.nn.l2_loss(kernel_14)+tf.nn.l2_loss(bias_14)

#summary_writer = tf.summary.FileWriter(logdir='logs/', graph=sess.graph)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.001,global_step,decay_steps=1,decay_rate=0.9999,staircase=True)
#定义优化方法（梯度法）优化数据为交叉熵值
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy+0.0001*regularization,global_step=global_step)
#初始化所有变量
sess.run(tf.global_variables_initializer())

path="E:/code/picture"#基础路径
img_names=os.listdir(path+"/imgs")#裂缝图像集路径，os.listdir方法用于返回指定文件夹包含的文件或文件夹名字列表，参数为路径
label_names=os.listdir(path+"/labels")#标签图像集路径
all_loss=[]#定义损失

for epoch in range(5):
    for i in range(len(img_names)):
        img=cv2.imread(path+"\\imgs\\"+img_names[i])
        label=cv2.imread(path+"\\labels\\"+label_names[i],0) # 0 为灰度图像 1为RGB图像 默认为1
        # 目的是为了添加第四维（为了能够运算，数值为1）
        img2 = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])/255+np.random.rand(img.shape[0],img.shape[1],img.shape[2])/10
        #目的是对标签的形状弄对，便于计算交叉熵
        label2=label.reshape(label.shape[0],label.shape[1],1)/255
        label3=np.concatenate((label2,1-label2),axis=2)

        if(i%30==0):
            the_loss=cross_entropy.eval(feed_dict={net_input:img2,net_label:label3})
            print(the_loss,end=' ')
            #学习率，实际上是优化算法的步长
            print('\t',learning_rate.eval())   
            all_loss.append(the_loss)
        #开始进行迭代计算，使用梯度法实现对交叉熵的值进行优化，只有采用新的迭代方式之后，才可以运行网络架构
        sess.run(fetches=train_step,feed_dict={net_input:img2,net_label:label3})

np.savetxt('E:/code/loss.txt',all_loss)
#img1=cv2.imread("F:\\projects\\crack_detection_verify\\img1.jpg")
#img1=cv2.imread("C:\\Users\\wangkangkang\\Desktop\\retina\\DRIVE\\test\\images\\12_test.tif")
#img1=cv2.imread("F:\\projects\\crack_detection_verify\\004\\part2.JPG")
# img1=cv2.imread("C:\\Users\\Administrator\\Desktop\\picture\\100001.bmp")
# img2=img1.reshape([1,img1.shape[0],img1.shape[1],img1.shape[2]])/255
# out=output_end_squeeze.eval(feed_dict={net_input:img2})
# np.savetxt('C:\\Users\\Administrator\\Desktop\\picture\\out0.txt',out[:,:,0])
# np.savetxt('C:\\Users\\Administrator\\Desktop\\picture\\out1.txt',out[:,:,1])

saver=tf.train.Saver()
saver.save(sess,"E:/code/Model/model.ckpt")

print('\a')