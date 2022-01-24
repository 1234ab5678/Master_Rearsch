# coding=gbk
#思路：首先分割图像，然后将图像分类，有裂缝图像送入网络进行检测，最后将结果拼合获得结果图像
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from numpy import *
from PIL import Image
import imageio
import time
import imutils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#负责控制台信息输出，主要是输出错误
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
saver.restore(sess, "./Model/Model5_50epoch/model.ckpt")



print("-----基于图像分类的检测方法-----")
print("-----裂缝检测开始-----")
# 求最大公约数
def gcd(x ,y):
    if x % y == 0:
        return y
    else:
        return gcd(y, x % y)#返回两数的最大公约数

def count_point(img):
    area = 0
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] < 50:
                area += 1
    return area#输出图像中符合要求的像素数

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)#求均值
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))#除以块长向上取整

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)#建立一个全为零的小块图像
    for r in range(rows_new):
        for c in range(cols_new):#循环，对每一个子块进行处理
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]#确定最后小块的大小，防止溢出

            imageROI = gray[rowmin:rowmax, colmin:colmax]#建立小灰度图像块
            temaver = np.mean(imageROI)#求小灰度图像块的均值
            blockImage[r, c] = temaver#将均值赋予小块图像

    blockImage = blockImage - average#减去均值图
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)#恢复成源图像大小，双线性插值
    gray2 = gray.astype(np.float32)#数据类型转换
    dst = gray2 - blockImage2#灰度图减去插值图
    dst = dst.astype(np.uint8)#数据类型转换为uint8
    dst = cv2.GaussianBlur(dst, (3, 3), 0)#高斯平滑
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)#灰度化
    return dst

def PTile(image,Tile):
    image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image],
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [256],  # HistSize
                        [0.0, 255.0])  # 直方图柱的范围
    Amount = 0
    Sum = 0
    for i in range(256):
        Amount += hist[i];  # 计算图像像素总数
    for i in range(256):
        Sum = Sum + hist[i];
        if (Sum >= Amount * Tile / 100):
            break;
    _,dst=cv2.threshold(image_GRAY,i,255,cv2.THRESH_BINARY)
    return dst
def newceil(a):
    b=math.ceil(a/500)
    b=b*500
    return b
def padding(img):

    new_size = [0, 0]
    new_size[0]=newceil(size1[0])
    new_size[1] = newceil(size1[1])
    return new_size

#abs_path = "/home/xq/ZSY_CrackDetection/crack detection/"
abs_path="./"
filePath_imgs = "./picture/crack/"
index = 0
median = "median/"
median_path = median
os.mkdir(median_path)#os.mkdir创建目录
time1=[]
run=[]

for filesname in os.listdir(filePath_imgs):  # listdir的参数是文件夹的路径,listdir用于返回指定文件夹文件名字列表
    #start_time1 = time.time()  # 记录开始时间

    index = index + 1
    imgname = filesname[:-4]
    print(imgname)
    Img = cv2.imread(filePath_imgs + filesname)
    size1 = Img.shape
    src1=Img
    print(src1.shape)
    wid=Img.shape[0]
    #Img = cv2.pyrDown(Img)
    #Img = cv2.pyrDown(Img)
    #Img = unevenLightCompensate(Img, 20)  # 亮度均衡
    #Img=cv2.pyrDown(Img)
    #Img=cv2.pyrDown(Img)

    #Img = ImageEnhance.Contrast(Img).enhance(1.5)
    # 给原图周围填充50个像素
    S = gcd(size1[0], size1[1])
    while True:
        if(S<200):
            print("改变图像尺寸开始")
            new_size = padding(Img)
            Img = cv2.copyMakeBorder(Img, 0, (new_size[0] - size1[0]), 0, (new_size[1] - size1[1]),
                                            cv2.BORDER_REFLECT)  # 轴对称扩充 相当于镜像填充
            print(Img.shape)
            #cv2.imwrite('E:/code/picture/test_image/'+'padding'+imgname+'.jpg', Img)
            #filePath_imgs = "E:/code/picture/test_image/"
            #os.remove(filePath_imgs + img + ".png")
            print("改变图像尺寸结束")
            break
        elif(S>200):
            break
    print("图像镜像填充开始")
    #Img = cv2.pyrDown(Img)
    size = Img.shape
    print(size)
    after_mirror_h = size[1] + 100
    after_mirror_w = size[0] + 100
    mirror_img = np.zeros((after_mirror_w, after_mirror_h), dtype=int)
    mirror_img = cv2.copyMakeBorder(Img, 100, 100, 100, 100, cv2.BORDER_REFLECT)  # 轴对称扩充 相当于镜像填充
    #cv2.imwrite("E:/code/picture/mirror_img/" + imgname + ".jpg",mirror_img)
    print("图像镜像填充完成")
    #os.system('pause')
    print("图像裁剪开始")#图像裁剪
    # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\picture\\mirror_img\\"+ imgname +".jpg", mirror_img)
    S = gcd(size[0], size[1])
    #while S > 400 :
    while S > 800:
         S = int(S / 2)
    else :
        S = S#计算裁剪因数
    #if(S>800):
    #    S=int(S/4)
    #else:
    #   S=S
    print(S, size[0], size[1])

    h = int(size[1] / S)
    w = int(size[0] / S)#计算裁剪后图像的大小
    print(h,w)
    test = 'test' + str("%02d" % (00 + index))#创建目录
    os.mkdir(median_path+ test)
    numb = 0
    for i in range(0, w):
        # 上句W 的值修改过
        for j in range(0, h):
            # 上句h 的值修改过
            # img = Img[i * 640:i * 640 + 740, j * 640:j * 640+ 740]
            #img = mirror_img[i * S:(i+1) * S + 100, j * S:(j+1) * S +100]#裁剪图像操作，重叠像素数为100*（S+100）
            img = mirror_img[i * S:(i + 1) * S+200, j * S:(j + 1) * S+200]  # 裁剪图像操作,是否可修改
            cv2.imwrite(median_path + test + '/' + str(
                "%03d" % (101 + numb)) + '.jpg', img)#保存图像
            numb = numb + 1
    print("图像裁剪完成")
    print("图像分类开始")
    result = 'result' + str("%02d" % (00 + index))  # 创建目录
    crack = 'crack' + str("%02d" % (00 + index))  # 创建目录
    nocrack = 'nocrack' + str("%02d" % (00 + index))  # 创建目录
    os.mkdir(median_path + result)
    os.mkdir(median_path + crack)
    os.mkdir(median_path + nocrack)
    path = median + test+'/';  # 待处理图片文件夹地址
    #dest = median_path + result;  # 处理后图片的保存地址
    #crack = median_path + crack;  # 处理后有裂缝图像的保存路径
    #nocrack = median_path + nocrack;  # 处理后无裂缝图像的保存路径
    #path = "E:/code/median/test01";  # 待处理图片文件夹地址
    dest = median_path + result+'/';  # 处理后图片的保存地址
    crack = median_path + crack+'/';  # 处理后有裂缝图像的保存路径
    nocrack = median_path + nocrack+'/';  # 处理后无裂缝图像的保存路径
    # contours = "E:/image_classification/contours/";
    # index = 0
    radiusarray = []
    #print("粗分割开始")
    for filename in os.listdir(path):  # listdir的参数是文件夹的路径,listdir用于返回指定文件夹文件名字列表
        # index = index + 1
        imgname = filename[:-4]
        #print(imgname)
        img = cv2.imread(path +filename)#加'/',否则读不进图像。。。
        x=img.shape[0]*img.shape[1]
        d=-846.4
        b=-0.01548
        c=741.3
        a=d*np.power(x,b)+c
        a=a-10

        #img_test=img
        #img_test = cv2.resize(img, (1440, 960), interpolation=cv2.INTER_CUBIC)
        #print(img.shape)
        #print(img_test.shape)
        #os.system("pause")
        img1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # 等大全黑图像
        #img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        #print(img1.shape)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (3, 3), 0);
        img = unevenLightCompensate(img, 20)  # 亮度均衡

        dst = PTile(img, 15)  # 亮度均衡
        #cv2.imwrite("E://3_ptile.jpg", dst)  # 存储图像
        #print("粗分割完成")
        #print("搜索轮廓开始")
        # img = cv2.imread("E://3_ptile.jpg", 0)
        # img=dst

        dst = cv2.Canny(dst, 100, 250);  # 高斯滤波，坎尼边缘检测
        dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE);  # opencv3中findcontours返回值为三个，opencv2为两个
        contours = np.array(contours)
        contours_size = contours.shape[0]
        max = 10;
        num = []
        for k in range(contours_size):
            length = cv2.arcLength(contours[k], True)
            num.append(length)
            if (length > max):
                max = length
            elif (length <= max):
                max = max
        # 获得一个数组，内容为每张图像的轮廓最大值
        #print(max)
        i = np.argmax(num)
        #print(i)
        # area = cv2.contourArea(contours[i], True)
        #print("搜索轮廓完成")

        cnt = contours[i]

        cv2.drawContours(img1, cnt, -1, (255, 255, 255), 8)
        kernel = np.ones((5, 5), np.uint8)
        img1 = cv2.dilate(img1, kernel, iterations=1)  # dilate表示膨胀操作,参数依次为操作图像，内核和次数
        # cv2.imwrite(result+imgname+'_result.jpg', img1)

        # im = cv2.imread(result+imgname+'_result.jpg')
        im = img1
        imgray = img1
        #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

        cnts = contours[0] if imutils.is_cv2() else contours[1]  # 用imutils来判断是opencv是2还是2+

        for cnt in cnts:
            # 最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # 返回最小外接圆的半径和圆心坐标
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(im, center, radius, (255, 0, 0), 2)

        #print('最小外接圆半径为' + str(radius))
        radiusarray.append(radius)
    print(radiusarray)
    print(len(radiusarray))
    #os.system('pause')
    imgIndex=[]
    crack_num = 0
    nocrack_num = 0
    #a=0
    print('阈值为'+str(a))
    for x in range(len(radiusarray)):
        if (radiusarray[x] >= a):
            crack_num += 1
            imgIndex.append(x+1)
            crackimg=cv2.imread(median_path + test + '/' + str(101+x) + '.jpg')
            cv2.imwrite(crack + str(101+x) + '.jpg', crackimg)
        elif(radiusarray[x] < a):
            nocrack_num += 1
            nocrackimg = cv2.imread(median_path + test + '/' + str(101 + x) + '.jpg')
            cv2.imwrite(nocrack + str(101 + x) + '.jpg', nocrackimg)

        # np.savetxt('E:/nc半径.txt', radiusarray)
        # cv2.imwrite(result+imgname+'_rectangle.jpg', im)
    print('有裂缝图像有' + str(crack_num) + '张')
    print('无裂缝图像有' + str(nocrack_num) + '张')
    if (crack_num == 0):
        wrongres = np.zeros((Img.shape[0], Img.shape[1]), dtype=np.uint8)  # 等大全黑图像
        imageio.imsave(filePath_imgs + "result" + filesname[:-4] + ".png", wrongres)
        end_time = time.time()
        time1.append(end_time)
        if (index == 1):
            run_time = end_time - start_time
        elif (index != 1):
            run_time = time1[index - 1] - time1[index - 2]
        print("第" + str(index) + "张图像检测时间为" + str(run_time) + "s")
        print("第" + str(index) + "张图像检测时间为" + str(run_time / 60) + "min")
        run.append(run_time)
        continue

    print("图像分类结束")
    #os.system('pause')
    img_number = w * h  # 大图象可以分割成w*h数量的小图
    #主要消耗时间较多的还是U-Net网络检测和图像拼合
    print("U-Net检测开始")
    test_output = 'test_output' + str("%02d" % (00 + index))
    # print(test_output)
    os.mkdir(median + test_output)  # 创建目录
    print(test_output)  # 输出名字
    for i in range(1, img_number + 1):
        if i in imgIndex:
            print("detection_index:"+str(i))
            # print(i)
            img1 = cv2.imread(abs_path + median + test + "/" + str("%03d" % (100 + i)) + '.jpg')  # 读入图像580*580
            img2 = img1.reshape([1, img1.shape[0], img1.shape[1], img1.shape[2]]) / 255  # 归一化处理
            out = output_end_squeeze.eval(feed_dict={net_input: img2})  # 送入神经网络进行检测，eval函数计算张量数值，见第130行
            # print(out.shape)
            out_size = out.shape
            out_h = out_size[1]  # 输出图像的大小，输入大小580，神经网络卷积损失88，输出大小492
            out_w = out_size[0]
            # print(out_h,out_w)
            # os.system('pause')
            np.savetxt(abs_path + median + test_output + '/out0' + str(100 + i) + '.txt', out[:, :, 0])
            np.savetxt(abs_path + median + test_output + '/out1' + str(100 + i) + '.txt', out[:, :, 1])  # 保存为txt
        # else:
    print("U-Net检测完成")
    print("图像拼合开始")
    # 创建输出图片空矩阵
    out_end = np.ones((size[0], size[1]), dtype=int)
    # 读取txt并得到输出矩阵
    num = 0
    out = np.ones((out_size[0], out_size[1]), dtype=int)  # 见294，295行
    # print(out_w,out_h)
    d = int((out_w - S) / 2)
    # d = int((S-out_w) / 2)
    print("卷积后图像尺寸差为：" + str(d))
    for i in range(0, w):
        for j in range(0, h):
            num = num + 1
            if num in imgIndex:
                print("txt_index:"+str(num))
                #print(num)
                out0 = np.loadtxt(abs_path + median + test_output + '/out0' + str(
                    "%03d" % (100 + num) + '.txt'))  # 读取out_put文件夹中的out0101,0102.....
                out1 = np.loadtxt(abs_path + median + test_output + '/out1' + str(
                    "%03d" % (100 + num) + '.txt'))  # 读取out1101，1102，1103....
                for m in range(0, out_w):
                    for n in range(0, out_h):
                        if (out0[m][n]) > out1[m][n]:
                            out[m][n] = 0
                        else:
                            out[m][n] = 1  # 二值化
                for m in range(0, S):
                    for n in range(0, S):
                        out_end[i * S + m][j * S + n] = out[d + m][d + n]
    # 输出矩阵显示为图片
    img = Image.fromarray(out_end.astype('uint8'))  # fromarray实现数组到图像的转换，astype实现变量类型转换
    imageio.imsave(filePath_imgs + filesname[:-4]+ ".png", out_end * 255)
    cut_img=cv2.imread(filePath_imgs + filesname[:-4]+ ".png")
    #cut_img = cv2.pyrUp(cut_img)
    result_img=cut_img[0:size1[0],0:size1[1]]
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filePath_imgs + 'result'+ filesname[:-4]+ ".png",result_img)
    os.remove(filePath_imgs + filesname[:-4]+ ".png")

    print("图像拼合完成")
    print("-----裂缝检测结束-----")
    end_time = time.time()
    time1.append(end_time)
    if(index==1):
        run_time=end_time - start_time
    elif(index!=1):
        run_time=time1[index-1]-time1[index-2]
    print("第"+str(index)+"张图像检测时间为"+ str(run_time) + "s")
    print("第"+str(index)+"张图像检测时间为"+ str(run_time / 60) + "min")
    run.append(run_time)
    print(run)
    np.savetxt(filePath_imgs+"time"+".txt",run)
total_time = end_time-start_time
print("检测总时间为：" + str(total_time))
print("检测总时间为：" + str(total_time/60) + "min")