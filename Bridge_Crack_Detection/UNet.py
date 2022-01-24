#coding=gbk
import tensorflow as tf
import numpy as np
import cv2
import os
#��������
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


# ��ʼ�����һ��ľ����
kernel_1=tf.Variable(initial_value=tf.random_normal(shape=[kernel_length,kernel_length,3,kernel_num],stddev=0.05))#�߶ȣ���ȣ�����ͨ���������ͨ����

#ƫִ b
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
output_8_active=tf.nn.leaky_relu(features=output_8)   #u-net��ײ�

#���ػ����̣���ͼ������
output_8_active_unsample=tf.image.resize_images(images=output_8_active,size=[tf.shape(output_8_active)[1]*2,tf.shape(output_8_active)[2]*2])
# �ϲ������� up-conversation��2*2 �ģ�Ҳ���� ԭ�ߴ�ĳ����������Ϊԭ���� 2 ��

#��Ҫƴ�ӵ�����ͼ�޸�Ϊ���ü�������䣩���ϲ����������ͼ��ͬ�ĳߴ� ���ڽ�ϡ�
output_6_active_crop=tf.image.resize_image_with_crop_or_pad(image=output_6_active,target_height=tf.shape(output_8_active_unsample)[1],target_width=tf.shape(output_8_active_unsample)[2])
#��ʼʵ��ƴ��
output_8_add_output_6=tf.concat(values=[output_8_active_unsample,output_6_active_crop],axis=3) # ʵ�ֺϲ�
output_9=tf.nn.conv2d(input=output_8_add_output_6,filter=kernel_9,strides=[1,1,1,1],padding='VALID')+bias_9
#����������
output_9_active=tf.nn.leaky_relu(features=output_9)
#��ʼʵ�־��������ת��
output_10=tf.nn.conv2d(input=output_9_active,filter=kernel_10,strides=[1,1,1,1],padding='VALID')+bias_10
#�Ⱦ�����ʵ���Ϸ����Ի���

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
#���һ�㣬�������
output_end=tf.nn.conv2d(input=output_13_active,filter=kernel_14,strides=[1,1,1,1],padding='VALID')+bias_14

#��άת��ά
output_end_squeeze=tf.squeeze(input=output_end)#tf.squeeze()�����������Ǵ�tensor��ɾ�����д�С(szie)��1��ά�ȡ�

#��ʼ������죨���֮��Ľ���ͱ�ǩͼƬ��
#�г���ǩͼ��Ϊ�� ���ڼ������
net_label_crop=tf.image.resize_image_with_crop_or_pad(image=net_label,target_height=tf.shape(output_end)[1],target_width=tf.shape(output_end)[2])
#��������ʧ����
cross_entropy_all=tf.nn.softmax_cross_entropy_with_logits(labels=net_label_crop,logits=output_end_squeeze)
#һ��ͼƬ��ƽ����������ʧ����
cross_entropy=tf.reduce_mean(cross_entropy_all)
#��������
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
#�����Ż��������ݶȷ����Ż�����Ϊ������ֵ
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy+0.0001*regularization,global_step=global_step)
#��ʼ�����б���
sess.run(tf.global_variables_initializer())

path="E:/code/picture"#����·��
img_names=os.listdir(path+"/imgs")#�ѷ�ͼ��·����os.listdir�������ڷ���ָ���ļ��а������ļ����ļ��������б�����Ϊ·��
label_names=os.listdir(path+"/labels")#��ǩͼ��·��
all_loss=[]#������ʧ

for epoch in range(5):
    for i in range(len(img_names)):
        img=cv2.imread(path+"\\imgs\\"+img_names[i])
        label=cv2.imread(path+"\\labels\\"+label_names[i],0) # 0 Ϊ�Ҷ�ͼ�� 1ΪRGBͼ�� Ĭ��Ϊ1
        # Ŀ����Ϊ����ӵ���ά��Ϊ���ܹ����㣬��ֵΪ1��
        img2 = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])/255+np.random.rand(img.shape[0],img.shape[1],img.shape[2])/10
        #Ŀ���ǶԱ�ǩ����״Ū�ԣ����ڼ��㽻����
        label2=label.reshape(label.shape[0],label.shape[1],1)/255
        label3=np.concatenate((label2,1-label2),axis=2)

        if(i%30==0):
            the_loss=cross_entropy.eval(feed_dict={net_input:img2,net_label:label3})
            print(the_loss,end=' ')
            #ѧϰ�ʣ�ʵ�������Ż��㷨�Ĳ���
            print('\t',learning_rate.eval())   
            all_loss.append(the_loss)
        #��ʼ���е������㣬ʹ���ݶȷ�ʵ�ֶԽ����ص�ֵ�����Ż���ֻ�в����µĵ�����ʽ֮�󣬲ſ�����������ܹ�
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