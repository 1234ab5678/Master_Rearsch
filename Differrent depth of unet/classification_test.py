# -*- coding: utf-8 -*-
#功能：输入图像，返回最大轮廓图像，以其最小外接圆半径作为特征值进行分类，阈值设置为65
import cv2
import numpy as np
import time
import os
import imutils

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

dest = "./picture/image/"; # 待检测图像的保存地址
result = "./picture/result/"; # 处理后图片的保存地址
#crack = "./test2/crack/"; #处理后有裂缝图像的保存路径
crack = "./picture/crack/";
nocrack = "./picture/nocrack/"; #处理后无裂缝图像的保存路径
res=[]
res1=[]
radiusarray=[]
crack_num=0
nocrack_num=0
os.mkdir(result)
os.mkdir(crack)
os.mkdir(nocrack)
start_time = time.time()
for filename in os.listdir(dest):  # listdir的参数是文件夹的路径,listdir用于返回指定文件夹文件名字列表
    imgname = filename[:-4]
    print(imgname)
    img = cv2.imread(dest + filename)
    image=img#复制一份源图像备用
    a = -846.4
    b = -0.01548
    c = 741.3
    x=image.shape[0]*image.shape[1]
    threshold=a*pow(x,b)+c
    img1 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)#等大全黑图像
    print("粗分割开始")
    img = cv2.GaussianBlur(img, (3, 3), 0)
    #cv2.imwrite("./3_gauss.jpg", img)  # 存储图像
    img = unevenLightCompensate(img, 20)  # 亮度均衡
    #cv2.imwrite("./3_light.jpg", img)  # 存储图像

    dst = PTile(img, 15)  # 亮度均衡
    #cv2.imwrite("./3_ptile.jpg", dst)  # 存储图像
    print("粗分割完成")
    print("搜索轮廓开始")
    #img = cv2.imread("E://3_ptile.jpg", 0)
    #img=dst
    #dst = cv2.GaussianBlur(dst, (3, 3), 0);
    #cv2.imwrite("E://res_Gauss.jpg", dst)
    dst = cv2.Canny(dst, 100, 250);  # 高斯滤波，坎尼边缘检测
    #cv2.bitwise_not(dst, dst);  # 颜色反转
    #cv2.imwrite("./3_canny.jpg", dst)
    dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE);  # opencv3中findcontours返回值为三个，opencv2为两个
    #print(contours)
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
    print(max)
    i = np.argmax(num)
    print(i)
    #area = cv2.contourArea(contours[i], True)
    print("搜索轮廓完成")

    cnt = contours[i]

    cv2.drawContours(img1, cnt, -1, (255, 255, 255), 8)
    kernel = np.ones((5, 5), np.uint8)
    img1 = cv2.dilate(img1, kernel, iterations=5)  #dilate表示膨胀操作,参数依次为操作图像，内核和次数
    img1=cv2.erode(img1, kernel, iterations=4)
    #cv2.imwrite('./max_result.jpg', img1)

    #im = cv2.imread(result+imgname+'_result.jpg')
    im=img1
    imgray=img1
    #imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

    cnts = contours[0] if imutils.is_cv2() else contours[1]  # 用imutils来判断是opencv是2还是2+
    #cnts=contours

    for cnt in cnts:
        # 最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)#返回最小外接圆的半径和圆心坐标
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(im, center, radius, (255, 0, 0), 2)

    print('最小外接圆半径为' + str(radius))
    #a=61.2
    print('阈值为'+str(threshold))
    if(radius>=threshold):
        crack_num+=1
        #radiusarray.append(radius)
        cv2.imwrite(crack+imgname+'.jpg',image)
    elif(radius<threshold):
        nocrack_num+=1
        cv2.imwrite(nocrack+imgname+'.jpg',image)
    radiusarray.append(radius)
    np.savetxt('./nc半径.txt', radiusarray)
    cv2.imwrite(result+imgname+'_rectangle.jpg', im)
    #cv2.imwrite('E://rectangle.jpg', im)
print(radiusarray)
print('有裂缝图像有'+str(crack_num)+'张')
print('无裂缝图像有'+str(nocrack_num)+'张')
#print('分类正确率为'+str((crack_num/(crack_num+nocrack_num))*100)+'%')
end_time = time.time()
total_time = end_time-start_time
print("筛选总时间为：" + str(total_time))
print("筛选总时间为：" + str(total_time/60) + "min")