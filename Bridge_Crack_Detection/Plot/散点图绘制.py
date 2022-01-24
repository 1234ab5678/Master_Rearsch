import matplotlib.pyplot as plt
import numpy as np

x=list(range(1,6))
plt.ylim([50,105])
y=[97,97.95,96.07,96,98]
z=[96,97.91,94.23,94,98]
#f=[94,100,89.28,88,100]
#x=[1024*768,1080*720,1280*720,1440*960,1440*1080,1920*1080,2880*1920,3840*2160,4096*2304,5760*3840]
#x=list(range(1,11))
#y=[2,2,1,1,9,4,4,2,1,3]
#z=[2,3,1,1,6,3,3,3,1,6]
#plt.plot(x,y,"ob:")
#plt.plot(x,z)
#plt.plot(x,f)
#plt.show()
#plt.scatter(x,y,c='b')
plt.scatter(x, y, marker = 'x',color = 'red', s = 40 ,label = 'First')
plt.scatter(x, z, marker = 'o',color = 'blue', s = 40 ,label = 'Second')
#plt.scatter(x,z,c='r')
#plt.scatter(x,f,c='g')
#plt.scatter(xValue,yValue2,c='r')

#plt.axhline(y=88.5, color='r', linestyle='-')
plt.show()