from read_data import dataReader
from read_data_regression import dataReader_regression
import numpy as np
import tensorflow as tf
from model import regression_model

from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)
import cv2
import math
import random

def show_grasp(image,predict):
    mask = predict[:,:,0]
    x = predict[:,:,1]
    y = predict[:,:,2]
    theta = predict[:,:,3]
    cell_length = image.shape[0]/mask.shape[0]    
    mask_f = mask.flatten()
    top_n = 1
    n_max = mask_f.argsort()[-top_n:][::-1]
    topn_idx = []
    for i in range(top_n):
        idx = np.unravel_index(n_max[i], mask.shape)
        topn_idx.append(idx)
    print topn_idx
    for i in range(0, len(topn_idx)):
        center_x = x[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][1] * cell_length
        center_y = y[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][0] * cell_length
        sin = math.sin(theta[topn_idx[i][0]][topn_idx[i][1]]/10)
        cos = math.cos(theta[topn_idx[i][0]][topn_idx[i][1]]/10)
        delta_x = 20 * cos
        delta_y = 20 * sin
        cv2.circle(image, (int(center_x), int(center_y)), 3, (0,255,0), -1)
        cv2.line(image,(int(center_x),int(center_y)),(int(center_x+delta_x),int(center_y+delta_y)),(0,255,0),2)
    cv2.imshow("grasp",image)
    k = cv2.waitKey(0)


def eval():
    dr = dataReader('./testset',5,6)
    sess = tf.Session()
    model = regression_model(sess)
    saver = tf.train.Saver()
    #saver.restore(sess, "./params")
    saver.restore(sess, "./meta_save/params")
    mini_dataset =dr.sample_mini_dataset(1)
    random.shuffle(mini_dataset[0])

    sample_img, sample_label = zip(*mini_dataset[0])
    train_img = sample_img[:15]
    train_label = sample_label[:15]

    test_img = sample_img[15:]
    test_label = sample_label[15:]

    train_img_np = np.array(train_img)
    train_label_np = np.array(train_label)

    iterations = 50
    for i in range(iterations):
        idx = np.random.choice(15,3)
        model.learn(train_img_np[idx], train_label_np[idx])
        if i%(iterations/3) == 0:                                                                    
            print i                                                                                  
            print "train_loss"                                                                       
            model.eval(train_img_np[idx], train_label_np[idx])  

    for i in range(len(test_label)):
        result = model.predict(test_img[i])
        mask = np.reshape(result[:,:64], [8,8])
        x = np.reshape(result[:,64:64*2], [8,8])
        y = np.reshape(result[:,64*2:64*3], [8,8])
        theta = np.reshape(result[:,64*3:], [8,8])
        restruct_label = np.stack((mask,x,y,theta),axis=-1)
        show_grasp(test_img[i],restruct_label)

    '''
    result = model.predict(test_img)
    mask = np.reshape(result[:,:64], [8,8])
    x = np.reshape(result[:,64:64*2], [8,8])
    y = np.reshape(result[:,64*2:64*3], [8,8])
    theta = np.reshape(result[:,64*3:], [8,8])
    restruct_label = np.stack((mask,x,y,theta),axis=-1)
    show_grasp(test_img,restruct_label)
    #print restruct_label
    '''

if __name__=='__main__':
    eval()
