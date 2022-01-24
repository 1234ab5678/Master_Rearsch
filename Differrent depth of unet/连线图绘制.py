import matplotlib.pyplot as plt
import numpy as np

x=list(range(1,6))
plt.ylim([0,500])
y1=[8.43,22.15,35.4,80.55,425.08]
y2=[5.73,7.68,16.23,31.35,166.63]
y3=[4.09,5.78,11.16,20.66,122.91]
#y3=[97,97.95,96.07,96,98]
#y4=[94,100,89.28,88,100]
#x=[1024*768,1080*720,1280*720,1440*960,1440*1080,1920*1080,2880*1920,3840*2160,4096*2304,5760*3840]
#x=list(range(1,11))
#y=[2,2,1,1,9,4,4,2,1,3]
#z=[2,3,1,1,6,3,3,3,1,6]
plt.plot(x,y1,'b.-')
plt.plot(x,y2,'r.-')
plt.plot(x,y3,'g.-')
#plt.plot(x,y4,'k.-')
#plt.show()
#plt.scatter(x,y1,c='b')
#plt.scatter(x,y2,c='g')
#plt.scatter(x,y3,c='r')
#plt.scatter(x,y4,c='k')
#plt.scatter(xValue,yValue2,c='r')

#plt.axhline(y=88.5, color='r', linestyle='-')
plt.show()