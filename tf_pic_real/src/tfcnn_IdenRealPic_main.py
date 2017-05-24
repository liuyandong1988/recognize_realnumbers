#coding=utf-8
'''
Created on 2017年2月22日
使用Opencv读取图像将其保存为二进制格式文件,并使用CNN网络识别
@author: liu
'''
import cv2
import os 
import tensorflow as tf
import pic_con_bin      #将图片转二进制文件的模块
import bin_con_pic      #将二进制文件转为图片模块
import cnn_iden_pic     #cnn识别数字模块
import numpy as np
import datetime

'''
启动C++文件，捕捉图片。
'''

main = "./imageCapreal"   #C++可运行文件的路径
try:
    os.path.exists(main)    #路径存在则返回True,路径损坏返回False
    os.system(main)     #直接输出main的内容
except (TypeError, ValueError):
    print "Can not find file !!!!"
    
'''
开始CNN识别
'''

starttime = datetime.datetime.now()


def main(_):
 
    '''读取图片,生成二进制文件'''
    
    
#   image = cv2.imread("digit_num/0.png",cv2.IMREAD_GRAYSCALE)  #数字图片
    image = cv2.imread("real_num/test.png",cv2.IMREAD_GRAYSCALE)  #手写图片
    pic_con_bin.convert(image)

    '''
    导入二进制文件
    通过函数proccess_images处理生成数据类DataSet可用的结构
    '''
    test_images_raw = open('test_gray.bin', 'rb')
    test_images = cnn_iden_pic.process_images(test_images_raw)
    #补充一个labels，feed_dict填充计算的需要，不影响结果满足type'tuple'(1,10)的形式
    test_labels = np.array([0,0,0,0,0,0,0,0,0,0])
    test_labels = test_labels.reshape(1,10)

    '''
    使用CNN网络识别数字
    '''
    num = cnn_iden_pic.run_training(test_images, test_labels)
    print 'CNN identified the number:%d'%num

    
    #计算识别所用的时间
    endtime = datetime.datetime.now()
    print '-'*50
    print 'The process time:',(endtime - starttime)

    '''生成原始图片和识别出的数字'''
    bin_con_pic.change('test_gray.bin', num)

#run this script instead of imported as a module
if __name__ == '__main__':
  tf.app.run()    #process the flag and then run main()


