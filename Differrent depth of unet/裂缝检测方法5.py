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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#�������̨��Ϣ�������Ҫ���������
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
saver.restore(sess, "./Model/Model5/model.ckpt")
#saver.restore(sess, "D:\dlbridge\Crack_detection_Code\Model\model.ckpt")


print("-----start-----")
# �����Լ��
def gcd(x ,y):
    if x % y == 0:
        return y
    else:
        return gcd(y, x % y)

# ���ζ�ȡ
# ������ͼƬͨ���������ķ�����Χ��չ50�����ص㣬��3840*5760����ɣ�3940*5860��
# �ٽ�ͼƬ�ָ�Ϊ740*740�Ĵ�С���������ߴ�Ϊ652*652��ƴ�ϳߴ�Ϊ640*640���������ԭʼͼƬ�ߴ��С��
# ֻ������������Ҫƴ���õ���Сͼ�ߴ��С��һ��ü�Ϊ������ͼƬ����Ⱥ͸߶���ȣ������� S .
index = 0
median = "median"
os.mkdir("./" + median)
detection=[]
time1=[]
run=[]

for filename in os.listdir(r'./picture/image/'):  # listdir�Ĳ������ļ��е�·��
    index = index + 1
    imgname = filename[:-4]
    print(imgname)
    Img = cv2.imread('./picture/image/' + filename)
    # ��ԭͼ��Χ���50������
    print("-----mirror_img-----")
    size = Img.shape
    print(size)#���ͼ��ĳ���
    after_mirror_h = size[1] + 100
    after_mirror_w = size[0] + 100#��������50������
    mirror_img = zeros((after_mirror_w, after_mirror_h), dtype=int)#
    mirror_img = cv2.copyMakeBorder(Img,100, 100, 100, 100, cv2.BORDER_REFLECT)  # ��Գ����� �൱�ھ�����䣬��img�������Ҹ����50����ӳ���

    print("-----cut_img-----")
    # cv2.imwrite("C:\\Users\\Administrator\\Desktop\\picture\\mirror_img\\"+ imgname +".jpg", mirror_img)
    S = gcd(size[0], size[1])#���������
    while S > 800:
         S = int(S / 2)
    else:
        S = S
    # print(S)
    # ��3840*5760��S = 640
    # ��3120*4160�� S = 520
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
        # �Ͼ�W ��ֵ�޸Ĺ�
        for j in range(0, h):
            numb = numb + 1;
            # �Ͼ�h ��ֵ�޸Ĺ�
            # img = Img[i * 640:i * 640 + 740, j * 640:j * 640+ 740]
            img = mirror_img[i * S:(i+1) * S + 200, j * S:(j+1) * S +200]
            #cv2.imwrite('./' + str(
            #    "%03d" % (101 + numb)) + '.jpg', img)  # ����ͼ��
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
    # �������ͼƬ�վ���
    out_end = zeros((size[0], size[1]), dtype=int)
    # ��ȡtxt���õ��������
    num = 0
    out = zeros((out_w, out_h), dtype=int)
    d = int((out_w - S) / 2)
    print("�����ͼ��ߴ��Ϊ��" + str(d))

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
    # ���������ʾΪͼƬ
    #print(out_end)
    # img = Image.fromarray(out_end)
    img = Image.fromarray(out_end.astype('uint8'))
    #img.show()
    # misc.imsave('C:\\Users\\Administrator\\Desktop\\picture\\image\\output' + str("%02d" % (00 + k)) +"_dp"+ ".png",out_end * 255)
    imageio.imsave('./picture/image/output_' + str(imgname) + ".png",out_end * 255)
    print("ͼ��ƴ�����")
    print("-----�ѷ������-----")
    detection_end = time.time()
    detection_time = detection_end - detection_start
    print("��" + str(index) + "��ͼ��ָ�ʱ��Ϊ" + str(detection_time) + "s")
    print("��" + str(index) + "��ͼ��ָ�ʱ��Ϊ" + str(detection_time / 60) + "min")
    detection.append(detection_time)
    print(detection)
    np.savetxt("./picture/image/" + "detection" + ".txt", detection)

    end_time = time.time()
    time1.append(end_time)
    if (index == 1):
        run_time = end_time - start_time
    elif (index != 1):
        run_time = time1[index - 1] - time1[index - 2]
    print("��" + str(index) + "��ͼ����ʱ��Ϊ" + str(run_time) + "s")
    print("��" + str(index) + "��ͼ����ʱ��Ϊ" + str(run_time / 60) + "min")
    run.append(run_time)
    print(run)
    np.savetxt("./picture/image/" + "time" + ".txt", run)

end_time = time.time()
run_time = -(start_time-end_time)
print("��������ʱ��Ϊ��"+str(run_time)+"s")
print("��������ʱ��Ϊ��"+str(run_time/60)+"min")
