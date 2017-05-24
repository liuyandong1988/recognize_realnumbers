
#coding=utf-8
'''
Created on 2017年2月22日
使用Opencv读取图像将其保存为二进制格式文件
@author: liu
'''
import cv2

def convert (img):
    #改变图片的像素尺寸，变为28×28匹配训练集
    img = cv2.resize(img, (28,28),interpolation=cv2.INTER_CUBIC)
    rows = 28  #行数
    cols = 28  #列数
 
    #像素值处理，黑白对掉与训练集一致
    #把像素在50 ~ 200之间的像素值归为0黑色,认为是背景   
    for i in range(0, rows):
        for j in range(0, cols):
            img[i,j] = 255 - img[i,j]
#             if  4<i<24 and 4<j<24:
#                 if img[i,j] > 80:
#                     img[i,j] = 255
    
# #             if img[i,j] > 0 :
# #                img[i,j] = 200
# #             else :
# #                 pass
             
    # #把图像转换为二进制文件
    fileSave = open('test_gray.bin','wb')
    for step1 in range(0, rows):
        for step2 in range(0, cols):
            fileSave.write(img[step1,step2])
    fileSave.close()
    print 'The test_gray.bin file is OK！'









