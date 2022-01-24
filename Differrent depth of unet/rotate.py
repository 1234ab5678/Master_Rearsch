import cv2
import math
#import pygame

def rotate_image(src,angle):
    radian=(angle*180)/math.pi
    print(radian)#转换为角度
    #print(src.shape)

    maxBorder=(int)(max(src.shape[0],src.shape[1])*1.414)
    print(maxBorder)

    dx=(int)(maxBorder-src.shape[1])//2
    dy=(int)(maxBorder-src.shape[0])//2
    print(dx)
    print(dy)

    dst = cv2.copyMakeBorder(src,dy, dy, dx, dx, cv2.BORDER_CONSTANT)
    cv2.imwrite("E://Border.jpg",dst)

    rotate_matrix=cv2.getRotationMatrix2D((dst.shape[0]*0.5, dst.shape[1]*0.5), radian, 1)
    dst = cv2.warpAffine(dst, rotate_matrix, (dst.shape[0], dst.shape[1]))
    cv2.imwrite("E://Border_rotate.jpg", dst)

    sinVal=abs(math.sin(angle))
    cosVal = abs(math.cos(angle))
    print(sinVal)
    print(cosVal)
    targetwidth=(int)(src.shape[1] * cosVal + src.shape[0] * sinVal)
    targetheight=(int)(src.shape[1] * sinVal + src.shape[0] * cosVal)
    x=(dst.shape[1]-targetwidth)//2
    y=(dst.shape[0]-targetheight)//2
    print(x)
    print(y)


    dst=dst[x:targetheight+x,y:targetwidth+y]

    cv2.imwrite("E://rotateres.jpg",dst)

x1,y1=0,0
x2,y2=1,1
k=((y2-y1)/(x2-x1))
angle=math.atan(k)
src = cv2.imread("E://rotate.jpg")
rotate_image(src,math.pi/6)


