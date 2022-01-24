# coding=gbk
#˼·�����ȷָ�ͼ��Ȼ��ͼ����࣬���ѷ�ͼ������������м�⣬��󽫽��ƴ�ϻ�ý��ͼ��
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#�������̨��Ϣ�������Ҫ���������
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
start_time=time.time()
#����ͼ
sess=tf.InteractiveSession()#�����Ự��ʱ�����ǿ����ȹ���һ��sessionȻ���ٶ������
#��������ͼ��
net_input=tf.placeholder(dtype=tf.float32)#�������繹��graph��ʱ����ģ���е�ռλ����ʱ��û�а�Ҫ��������ݴ���ģ�ͣ���ֻ������Ҫ���ڴ�Ƚ���session���ڻỰ�У�����ģ�͵�ʱ��ͨ��feed_dict()������ռλ��ι�����ݡ�

#���������ǩ
net_label=tf.placeholder(dtype=tf.float32)
#�����ͼ���ͨ����
kernel_num=32
#�������˵Ĵ�С
kernel_length=3

#��ʼ�����һ��ľ����
kernel_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,3,kernel_num],stddev=0.05))#�߶ȣ���ȣ�����ͨ���������ͨ����
#ƫ�� b
bias_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))

kernel_2=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,kernel_num,kernel_num],stddev=0.05))
bias_2=tf.Variable(initial_value=tf.random_normal(shape=[kernel_num],stddev=0.05))
# �����������������
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
#����͵�һ�������Ŀ�ʼʵ�־��
output_1=tf.nn.conv2d(input=net_input,filter=kernel_1,strides=[1,1,1,1],padding='VALID')+bias_1
#����Ϊ���룬����ˣ��������ߴ磬�����ʽ��VALID�ǲ������߽�
#������������Ժ�����ʵ�����ݵķ����ԣ�
output_1_active=tf.nn.leaky_relu(features=output_1)
#��֮ǰ�ľ�������ʼ�͵ڶ�������˿�ʼ���
output_2=tf.nn.conv2d(input=output_1_active,filter=kernel_2,strides=[1,1,1,1],padding='VALID')+bias_2
output_2_active=tf.nn.leaky_relu(features=output_2)  #u-net��һ��
#��ʼ�������гػ������ػ������ۼ���Ϣ�������Ӧ��ȡ������
output_2_active_pool=tf.nn.max_pool(value=output_2_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')#����Ϊ���룬�ػ����ڣ���������ʽ

output_3=tf.nn.conv2d(input=output_2_active_pool,filter=kernel_3,strides=[1,1,1,1],padding='VALID')+bias_3
output_3_active=tf.nn.leaky_relu(features=output_3)
output_4=tf.nn.conv2d(input=output_3_active,filter=kernel_4,strides=[1,1,1,1],padding='VALID')+bias_4
output_4_active=tf.nn.leaky_relu(features=output_4)   #u-net�ڶ���
output_4_active_pool=tf.nn.max_pool(value=output_4_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_5=tf.nn.conv2d(input=output_4_active_pool,filter=kernel_5,strides=[1,1,1,1],padding='VALID')+bias_5
output_5_active=tf.nn.leaky_relu(features=output_5)
output_6=tf.nn.conv2d(input=output_5_active,filter=kernel_6,strides=[1,1,1,1],padding='VALID')+bias_6
output_6_active=tf.nn.leaky_relu(features=output_6)    #u-net������
output_6_active_pool=tf.nn.max_pool(value=output_6_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_7=tf.nn.conv2d(input=output_6_active_pool,filter=kernel_7,strides=[1,1,1,1],padding='VALID')+bias_7
output_7_active=tf.nn.leaky_relu(features=output_7)
output_8=tf.nn.conv2d(input=output_7_active,filter=kernel_8,strides=[1,1,1,1],padding='VALID')+bias_8
output_8_active=tf.nn.leaky_relu(features=output_8)  #u-net���Ĳ�
output_8_active_pool=tf.nn.max_pool(value=output_8_active,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

output_9=tf.nn.conv2d(input=output_8_active_pool,filter=kernel_9,strides=[1,1,1,1],padding='VALID')+bias_9
output_9_active=tf.nn.leaky_relu(features=output_9)
output_10=tf.nn.conv2d(input=output_9_active,filter=kernel_10,strides=[1,1,1,1],padding='VALID')+bias_10
output_10_active=tf.nn.leaky_relu(features=output_10)   #u-net��ײ�

#���ػ����̣���ͼ������
output_10_active_unsample=tf.image.resize_images(images=output_10_active,size=[tf.shape(output_10_active)[1]*2,tf.shape(output_10_active)[2]*2])
# �ϲ������� up-conversation��2*2 �ģ�Ҳ���� ԭ�ߴ�ĳ����������Ϊԭ���� 2 ��
#print(output_10_active_unsample)

#��Ҫƴ�ӵ�����ͼ�޸�Ϊ���ü�������䣩���ϲ����������ͼ��ͬ�ĳߴ� ���ڽ�ϡ�
output_8_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_8_active,target_height=tf.shape(output_10_active_unsample)[1],target_width=tf.shape(output_10_active_unsample)[2])
#��ʼʵ��ƴ��
output_10_add_output_8=tf.concat(values=[output_10_active_unsample,output_8_active_crop],axis=3) # ʵ�ֺϲ�
output_11=tf.nn.conv2d(input=output_10_add_output_8,filter=kernel_11,strides=[1,1,1,1],padding='VALID')+bias_11
#����������
output_11_active=tf.nn.leaky_relu(features=output_11)
#��ʼʵ�־��������ת��
output_12=tf.nn.conv2d(input=output_11_active,filter=kernel_12,strides=[1,1,1,1],padding='VALID')+bias_12
#�Ⱦ�����ʵ���Ϸ����Ի���

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

#���һ�㣬�������
output_end=tf.nn.conv2d(input=output_18_active,filter=kernel_19,strides=[1,1,1,1],padding='VALID')+bias_19

#��άת��ά
output_end_squeeze=tf.squeeze(input=output_end)#tf.squeeze()�����������Ǵ�tensor��ɾ�����д�С(szie)��1��ά�ȡ�

#��ʼ������죨���֮��Ľ���ͱ�ǩͼƬ��
#�г���ǩͼ��Ϊ�� ���ڼ������
net_label_crop=tf.image.resize_image_with_crop_or_pad(image=net_label,target_height=tf.shape(output_end)[1],target_width=tf.shape(output_end)[2])
#��������ʧ����
cross_entropy_all=tf.nn.softmax_cross_entropy_with_logits(labels=net_label_crop,logits=output_end_squeeze)
#һ��ͼƬ��ƽ����������ʧ����
cross_entropy=tf.reduce_mean(cross_entropy_all)

# ִ�У�����ģ�ͽ��м��
saver = tf.train.Saver()
saver.restore(sess, "./Model/Model5_50epoch/model.ckpt")



print("-----����ͼ�����ļ�ⷽ��-----")
print("-----�ѷ��⿪ʼ-----")
# �����Լ��
def gcd(x ,y):
    if x % y == 0:
        return y
    else:
        return gcd(y, x % y)#�������������Լ��

def count_point(img):
    area = 0
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] < 50:
                area += 1
    return area#���ͼ���з���Ҫ���������

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)#���ֵ
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))#���Կ鳤����ȡ��

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)#����һ��ȫΪ���С��ͼ��
    for r in range(rows_new):
        for c in range(cols_new):#ѭ������ÿһ���ӿ���д���
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]#ȷ�����С��Ĵ�С����ֹ���

            imageROI = gray[rowmin:rowmax, colmin:colmax]#����С�Ҷ�ͼ���
            temaver = np.mean(imageROI)#��С�Ҷ�ͼ���ľ�ֵ
            blockImage[r, c] = temaver#����ֵ����С��ͼ��

    blockImage = blockImage - average#��ȥ��ֵͼ
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)#�ָ���Դͼ���С��˫���Բ�ֵ
    gray2 = gray.astype(np.float32)#��������ת��
    dst = gray2 - blockImage2#�Ҷ�ͼ��ȥ��ֵͼ
    dst = dst.astype(np.uint8)#��������ת��Ϊuint8
    dst = cv2.GaussianBlur(dst, (3, 3), 0)#��˹ƽ��
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)#�ҶȻ�
    return dst

def PTile(image,Tile):
    image_GRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image],
                        [0],  # ʹ�õ�ͨ��
                        None,  # û��ʹ��mask
                        [256],  # HistSize
                        [0.0, 255.0])  # ֱ��ͼ���ķ�Χ
    Amount = 0
    Sum = 0
    for i in range(256):
        Amount += hist[i];  # ����ͼ����������
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
os.mkdir(median_path)#os.mkdir����Ŀ¼
time1=[]
run=[]

for filesname in os.listdir(filePath_imgs):  # listdir�Ĳ������ļ��е�·��,listdir���ڷ���ָ���ļ����ļ������б�
    #start_time1 = time.time()  # ��¼��ʼʱ��

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
    #Img = unevenLightCompensate(Img, 20)  # ���Ⱦ���
    #Img=cv2.pyrDown(Img)
    #Img=cv2.pyrDown(Img)

    #Img = ImageEnhance.Contrast(Img).enhance(1.5)
    # ��ԭͼ��Χ���50������
    S = gcd(size1[0], size1[1])
    while True:
        if(S<200):
            print("�ı�ͼ��ߴ翪ʼ")
            new_size = padding(Img)
            Img = cv2.copyMakeBorder(Img, 0, (new_size[0] - size1[0]), 0, (new_size[1] - size1[1]),
                                            cv2.BORDER_REFLECT)  # ��Գ����� �൱�ھ������
            print(Img.shape)
            #cv2.imwrite('E:/code/picture/test_image/'+'padding'+imgname+'.jpg', Img)
            #filePath_imgs = "E:/code/picture/test_image/"
            #os.remove(filePath_imgs + img + ".png")
            print("�ı�ͼ��ߴ����")
            break
        elif(S>200):
            break
    print("ͼ������俪ʼ")
    #Img = cv2.pyrDown(Img)
    size = Img.shape
    print(size)
    after_mirror_h = size[1] + 100
    after_mirror_w = size[0] + 100
    mirror_img = np.zeros((after_mirror_w, after_mirror_h), dtype=int)
    mirror_img = cv2.copyMakeBorder(Img, 100, 100, 100, 100, cv2.BORDER_REFLECT)  # ��Գ����� �൱�ھ������
    #cv2.imwrite("E:/code/picture/mirror_img/" + imgname + ".jpg",mirror_img)
    print("ͼ����������")
    #os.system('pause')
    print("ͼ��ü���ʼ")#ͼ��ü�
    # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\picture\\mirror_img\\"+ imgname +".jpg", mirror_img)
    S = gcd(size[0], size[1])
    #while S > 400 :
    while S > 800:
         S = int(S / 2)
    else :
        S = S#����ü�����
    #if(S>800):
    #    S=int(S/4)
    #else:
    #   S=S
    print(S, size[0], size[1])

    h = int(size[1] / S)
    w = int(size[0] / S)#����ü���ͼ��Ĵ�С
    print(h,w)
    test = 'test' + str("%02d" % (00 + index))#����Ŀ¼
    os.mkdir(median_path+ test)
    numb = 0
    for i in range(0, w):
        # �Ͼ�W ��ֵ�޸Ĺ�
        for j in range(0, h):
            # �Ͼ�h ��ֵ�޸Ĺ�
            # img = Img[i * 640:i * 640 + 740, j * 640:j * 640+ 740]
            #img = mirror_img[i * S:(i+1) * S + 100, j * S:(j+1) * S +100]#�ü�ͼ��������ص�������Ϊ100*��S+100��
            img = mirror_img[i * S:(i + 1) * S+200, j * S:(j + 1) * S+200]  # �ü�ͼ�����,�Ƿ���޸�
            cv2.imwrite(median_path + test + '/' + str(
                "%03d" % (101 + numb)) + '.jpg', img)#����ͼ��
            numb = numb + 1
    print("ͼ��ü����")
    print("ͼ����࿪ʼ")
    result = 'result' + str("%02d" % (00 + index))  # ����Ŀ¼
    crack = 'crack' + str("%02d" % (00 + index))  # ����Ŀ¼
    nocrack = 'nocrack' + str("%02d" % (00 + index))  # ����Ŀ¼
    os.mkdir(median_path + result)
    os.mkdir(median_path + crack)
    os.mkdir(median_path + nocrack)
    path = median + test+'/';  # ������ͼƬ�ļ��е�ַ
    #dest = median_path + result;  # �����ͼƬ�ı����ַ
    #crack = median_path + crack;  # ��������ѷ�ͼ��ı���·��
    #nocrack = median_path + nocrack;  # ��������ѷ�ͼ��ı���·��
    #path = "E:/code/median/test01";  # ������ͼƬ�ļ��е�ַ
    dest = median_path + result+'/';  # �����ͼƬ�ı����ַ
    crack = median_path + crack+'/';  # ��������ѷ�ͼ��ı���·��
    nocrack = median_path + nocrack+'/';  # ��������ѷ�ͼ��ı���·��
    # contours = "E:/image_classification/contours/";
    # index = 0
    radiusarray = []
    #print("�ַָʼ")
    for filename in os.listdir(path):  # listdir�Ĳ������ļ��е�·��,listdir���ڷ���ָ���ļ����ļ������б�
        # index = index + 1
        imgname = filename[:-4]
        #print(imgname)
        img = cv2.imread(path +filename)#��'/',���������ͼ�񡣡���
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
        img1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)  # �ȴ�ȫ��ͼ��
        #img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        #print(img1.shape)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (3, 3), 0);
        img = unevenLightCompensate(img, 20)  # ���Ⱦ���

        dst = PTile(img, 15)  # ���Ⱦ���
        #cv2.imwrite("E://3_ptile.jpg", dst)  # �洢ͼ��
        #print("�ַָ����")
        #print("����������ʼ")
        # img = cv2.imread("E://3_ptile.jpg", 0)
        # img=dst

        dst = cv2.Canny(dst, 100, 250);  # ��˹�˲��������Ե���
        dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE);  # opencv3��findcontours����ֵΪ������opencv2Ϊ����
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
        # ���һ�����飬����Ϊÿ��ͼ����������ֵ
        #print(max)
        i = np.argmax(num)
        #print(i)
        # area = cv2.contourArea(contours[i], True)
        #print("�����������")

        cnt = contours[i]

        cv2.drawContours(img1, cnt, -1, (255, 255, 255), 8)
        kernel = np.ones((5, 5), np.uint8)
        img1 = cv2.dilate(img1, kernel, iterations=1)  # dilate��ʾ���Ͳ���,��������Ϊ����ͼ���ں˺ʹ���
        # cv2.imwrite(result+imgname+'_result.jpg', img1)

        # im = cv2.imread(result+imgname+'_result.jpg')
        im = img1
        imgray = img1
        #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # �����ֵ
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL ����ֻ�����Χ����

        cnts = contours[0] if imutils.is_cv2() else contours[1]  # ��imutils���ж���opencv��2����2+

        for cnt in cnts:
            # ��С���Բ
            (x, y), radius = cv2.minEnclosingCircle(cnt)  # ������С���Բ�İ뾶��Բ������
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(im, center, radius, (255, 0, 0), 2)

        #print('��С���Բ�뾶Ϊ' + str(radius))
        radiusarray.append(radius)
    print(radiusarray)
    print(len(radiusarray))
    #os.system('pause')
    imgIndex=[]
    crack_num = 0
    nocrack_num = 0
    #a=0
    print('��ֵΪ'+str(a))
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

        # np.savetxt('E:/nc�뾶.txt', radiusarray)
        # cv2.imwrite(result+imgname+'_rectangle.jpg', im)
    print('���ѷ�ͼ����' + str(crack_num) + '��')
    print('���ѷ�ͼ����' + str(nocrack_num) + '��')
    if (crack_num == 0):
        wrongres = np.zeros((Img.shape[0], Img.shape[1]), dtype=np.uint8)  # �ȴ�ȫ��ͼ��
        imageio.imsave(filePath_imgs + "result" + filesname[:-4] + ".png", wrongres)
        end_time = time.time()
        time1.append(end_time)
        if (index == 1):
            run_time = end_time - start_time
        elif (index != 1):
            run_time = time1[index - 1] - time1[index - 2]
        print("��" + str(index) + "��ͼ����ʱ��Ϊ" + str(run_time) + "s")
        print("��" + str(index) + "��ͼ����ʱ��Ϊ" + str(run_time / 60) + "min")
        run.append(run_time)
        continue

    print("ͼ��������")
    #os.system('pause')
    img_number = w * h  # ��ͼ����Էָ��w*h������Сͼ
    #��Ҫ����ʱ��϶�Ļ���U-Net�������ͼ��ƴ��
    print("U-Net��⿪ʼ")
    test_output = 'test_output' + str("%02d" % (00 + index))
    # print(test_output)
    os.mkdir(median + test_output)  # ����Ŀ¼
    print(test_output)  # �������
    for i in range(1, img_number + 1):
        if i in imgIndex:
            print("detection_index:"+str(i))
            # print(i)
            img1 = cv2.imread(abs_path + median + test + "/" + str("%03d" % (100 + i)) + '.jpg')  # ����ͼ��580*580
            img2 = img1.reshape([1, img1.shape[0], img1.shape[1], img1.shape[2]]) / 255  # ��һ������
            out = output_end_squeeze.eval(feed_dict={net_input: img2})  # ������������м�⣬eval��������������ֵ������130��
            # print(out.shape)
            out_size = out.shape
            out_h = out_size[1]  # ���ͼ��Ĵ�С�������С580������������ʧ88�������С492
            out_w = out_size[0]
            # print(out_h,out_w)
            # os.system('pause')
            np.savetxt(abs_path + median + test_output + '/out0' + str(100 + i) + '.txt', out[:, :, 0])
            np.savetxt(abs_path + median + test_output + '/out1' + str(100 + i) + '.txt', out[:, :, 1])  # ����Ϊtxt
        # else:
    print("U-Net������")
    print("ͼ��ƴ�Ͽ�ʼ")
    # �������ͼƬ�վ���
    out_end = np.ones((size[0], size[1]), dtype=int)
    # ��ȡtxt���õ��������
    num = 0
    out = np.ones((out_size[0], out_size[1]), dtype=int)  # ��294��295��
    # print(out_w,out_h)
    d = int((out_w - S) / 2)
    # d = int((S-out_w) / 2)
    print("�����ͼ��ߴ��Ϊ��" + str(d))
    for i in range(0, w):
        for j in range(0, h):
            num = num + 1
            if num in imgIndex:
                print("txt_index:"+str(num))
                #print(num)
                out0 = np.loadtxt(abs_path + median + test_output + '/out0' + str(
                    "%03d" % (100 + num) + '.txt'))  # ��ȡout_put�ļ����е�out0101,0102.....
                out1 = np.loadtxt(abs_path + median + test_output + '/out1' + str(
                    "%03d" % (100 + num) + '.txt'))  # ��ȡout1101��1102��1103....
                for m in range(0, out_w):
                    for n in range(0, out_h):
                        if (out0[m][n]) > out1[m][n]:
                            out[m][n] = 0
                        else:
                            out[m][n] = 1  # ��ֵ��
                for m in range(0, S):
                    for n in range(0, S):
                        out_end[i * S + m][j * S + n] = out[d + m][d + n]
    # ���������ʾΪͼƬ
    img = Image.fromarray(out_end.astype('uint8'))  # fromarrayʵ�����鵽ͼ���ת����astypeʵ�ֱ�������ת��
    imageio.imsave(filePath_imgs + filesname[:-4]+ ".png", out_end * 255)
    cut_img=cv2.imread(filePath_imgs + filesname[:-4]+ ".png")
    #cut_img = cv2.pyrUp(cut_img)
    result_img=cut_img[0:size1[0],0:size1[1]]
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filePath_imgs + 'result'+ filesname[:-4]+ ".png",result_img)
    os.remove(filePath_imgs + filesname[:-4]+ ".png")

    print("ͼ��ƴ�����")
    print("-----�ѷ������-----")
    end_time = time.time()
    time1.append(end_time)
    if(index==1):
        run_time=end_time - start_time
    elif(index!=1):
        run_time=time1[index-1]-time1[index-2]
    print("��"+str(index)+"��ͼ����ʱ��Ϊ"+ str(run_time) + "s")
    print("��"+str(index)+"��ͼ����ʱ��Ϊ"+ str(run_time / 60) + "min")
    run.append(run_time)
    print(run)
    np.savetxt(filePath_imgs+"time"+".txt",run)
total_time = end_time-start_time
print("�����ʱ��Ϊ��" + str(total_time))
print("�����ʱ��Ϊ��" + str(total_time/60) + "min")