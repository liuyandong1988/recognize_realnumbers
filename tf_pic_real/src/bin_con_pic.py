
#coding=utf-8
'''
Created on 2017年2月22日
读取二进制格式文件，生成图片
@author: liu
'''

import struct
import numpy as np
import cv2
from matplotlib import pyplot as plt  
 
def change(f,num):     
     
     
    fileReader = open(f,'rb')
    imageRead = np.zeros((28,28),np.uint8)
    for step1 in range(0,28):
        for step2 in range(0,28):
            a = struct.unpack("B",fileReader.read(1))
            imageRead[step1,step2] = a[0]        
    fileReader.close()


#     imageRead = cv2.imread("real_num/test.png", cv2.IMREAD_GRAYSCALE)
    imageRead = cv2.resize(imageRead, (200,200),interpolation=cv2.INTER_CUBIC)
    
    info1= "digit_num/"
    filename = '%d'%num[0]
    info2 = ".png"
    filepath = info1 + filename +info2
    imageIden = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    imageIden = cv2.resize(imageIden, (200,200),interpolation=cv2.INTER_CUBIC)
   
    plt.subplot(121), plt.imshow(imageRead,"gray"), plt.title("Original image")
    plt.subplot(122), plt.imshow(imageIden,"gray"), plt.title("Identified result")  
     
    
    plt.show() 
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





