#encoding=utf-8

import tensorflow as tf
import numpy as np


'''处理输入的二进制文件，使其具有tensorflow mnist的格式 1×28×28×1
   args: test_gray.bin二进制文件
   returns：type 'tuple' 1*28*28*1
'''
def process_images(f):
  with f as bytestream:  #以字节流形式处理
    num_images = 1       #images的数量
    rows = 28            #image尺寸  行列
    cols = 28
    buf = bytestream.read(rows * cols * num_images)  
    data = np.frombuffer(buf, dtype=np.uint8)        #以uint8的形式读入数据
    data = data.reshape(num_images, rows, cols, 1)   #转成1×28×28×1的形式
    return data





'''
   构建CNN模型所需要的参数和运算
   x_image:输入CNN的图片像素
   y_label: 占位标签
   weight_variable:权重
   bias_variable: 偏置
   conv2d:卷积运算
   pooling：汇聚运算
'''
#dataset images and labels
x = tf.placeholder(tf.float32, shape=[None, 784])
# 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定，第二三参数代表图像尺寸，最后一个参数代表图像通道数
x_image = tf.reshape(x, [-1,28,28,1])

y_label = tf.placeholder(tf.float32, shape=[None, 10])
  
#parameters W/b
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  #截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)
  
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
#convolution strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
#pooling 参数同上，ksize是池化块的大小
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
      

  
  
'''
   building cnn module
'''  
#---------------------------第一层卷积加池化------------------------------------------------#
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征
W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32])
  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#原图像尺寸28*28，第一轮图像缩小为14*14，共有32张输出图（feature）
  
#---------------------------第二层卷积加池化------------------------------------------------#
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#第二轮后图像缩小为7*7，共有64张
  
#------------------------第三层全连层---------------------------------------------------#
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])# 展开，第一个参数为样本数量，-1未知
  
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
# dropout操作，减少过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #每个神经元，都以概率keep_prob保持它的激活状态
  
#------------------------第四层全连层---------------------------------------------------#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
  
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#-------------------------------------------------------------------------------------#
  
  
#------------------------------training model------------------------------------------#
def run_training(test_img, test_lab): 
     

    sess = tf.InteractiveSession()  
    # defaults to saving all variables - in this case w and b
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
     
    checkpoint_dir = 'model/'       #参数存放路径
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)    #返回checkpoint的状态
    if ckpt and ckpt.model_checkpoint_path:  
        saver.restore(sess, ckpt.model_checkpoint_path)    # 恢复参数 
    else:  
        pass  
      
    #调整数据结构，满足feed_dict的填充格式
    batch = [test_img[0].reshape(1,784), test_lab[0].reshape(1,10)]
    #通过CNN计算输入图片的分类概率y_predict（0～9的概率值） tf.argmax返回最大概率值的索引，即识别的数字 
    number = tf.argmax(y_predict,1).eval(feed_dict={
            x:batch[0], y_label: batch[1], keep_prob: 1.0})
    return number
 


    

    
