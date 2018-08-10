from read_data import dataReader
from read_data_regression import dataReader_regression
import numpy as np
import tensorflow as tf
from model import regression_model

from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

def train():
    sess = tf.Session()
    model = regression_model(sess)
    dr = dataReader('./object_dataset',  1 , 21)
    dr_testset = dataReader('./testset', 1 , 6)
    model_state = VariableState(sess, tf.trainable_variables())
    meta_iters = 8000#50000
    saver = tf.train.Saver()
    
    meta_step_size_final=0.1
    meta_step_size=0.1

    meta_batch_size = 1
    inner_batch_size = 5
    inner_iters=20

    for i in range(meta_iters):
        frac_done = i/meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        old_vars = model_state.export_variables()
        new_vars = []        
        #for _ in range(meta_batch_size): 
        mini_dataset = dr.sample_mini_dataset(1)
        #print len(mini_dataset[0])
        for start in range(0,len(mini_dataset[0]),10):

            sample_img, sample_label = zip(*mini_dataset[0][start:start+10])
            model.learn(sample_img, sample_label)
        new_vars.append(model_state.export_variables())
        model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

        if i % 20 == 0:
            test_batch = dr_testset.sample_mini_dataset(1)
            test_img, test_label = zip(*test_batch[0])
            train_dataset = dr.sample_mini_dataset(1)
            train_img, train_label =  zip(*train_dataset[0])
            #test_img, test_label = dr_testset.get_mini_batches(test_batch[0], 20)
            print "test loss: iterations---- ", i
            #test_images,test_labels = dr.get_test_data()
            #print "mask_loss"
            #model.mask_eval(test_img, test_label)
            #mini_dataset = dr.sample_mini_dataset(1)            

            print "train_loss"
            model.eval(train_img, train_label)

            print "test_loss"
            model.eval(test_img, test_label)

    saver.save(sess, "./params")


def train_regression():
    
    sess = tf.Session()
    model = regression_model(sess)
    dr = dataReader_regression('./object_dataset',20)
    dr_testset = dataReader_regression('./testset',5)
    #model_state = VariableState(sess, tf.trainable_variables())
    iters = 30000#50000
    saver = tf.train.Saver()    
    for i in range(iters):
        train_image_batch, train_label_batch = dr.sample_batch(20)
        model.learn(train_image_batch, train_label_batch)

        if i % 20 == 0:
            print "train loss, iteration----- ", i
            model.eval(train_image_batch, train_label_batch)
            print "test loss"
            test_image_batch, test_label_batch = dr_testset.sample_batch(20)
            model.eval(test_image_batch, test_label_batch)
    saver.save(sess, "./regress_save/params")

def eval():
    dr = dataReader('./testset',2)
    sess = tf.Session()
    model = regression_model(sess)
    saver = tf.train.Saver()
    saver.restore(sess, "./regress_save/params")
    #test_images,test_labels = dr.sample_mini_dataset(20)
    #test_images,test_labels = dr.get_test_data()
    mini_dataset = dr.sample_mini_dataset(1)

    sample_img, sample_label = zip(*mini_dataset[0])
    idx = np.random.choice(len(sample_img),15)
    sample_img = np.array(sample_img)
    sample_label = np.array(sample_label)
    test_img = sample_img[:]
    test_label = sample_label[:]
    print len(test_label)
    result = model.predict(test_img[4])
    mask = np.reshape(result[:,:64], [8,8])
    x = np.reshape(result[:,64:64*2], [8,8])
    y = np.reshape(result[:,64*2:64*3], [8,8])
    theta = np.reshape(result[:,64*3:], [8,8])
    #print test_label[test_idx]
    print "@@@@@@@@@@@@@@@@@@@@@@@"
    restruct_label = np.stack((mask,x,y,theta),axis=-1)
    #print restruct_label

    #restruct_label[3][4][3] = -1.458
    #----------------------one shot---------------------
    #one_shot_img = test_images[85]

    model_state = VariableState(sess, tf.trainable_variables())
    full_state = VariableState(sess,
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    old_vars = full_state.export_variables()
    for i in range(32):
        print i
        model.learn(sample_img[idx], sample_label[idx])
    for i in range(len(test_label)):

        new_result = model.predict(test_img[i])
        new_mask = np.reshape(new_result[:,:64], [8,8])
        new_x = np.reshape(new_result[:,64:64*2], [8,8])
        new_y = np.reshape(new_result[:,64*2:64*3], [8,8])
        new_theta = np.reshape(new_result[:,64*3:], [8,8])
        new_label = np.stack((new_mask,new_x,new_y,new_theta),axis=-1)
    
        print "??????????????????????"
        #print    new_label 
        #one_shot_label = result_show

        #-------------------------------------------------
        #model.mask_eval(test_images, test_labels)
        dr.show_grasp(test_img[i],new_label,test_label[i])
        #dr.show_grasp(test_img[4],restruct_label,test_label[4])

if __name__=='__main__':
    train()
    #train_regression()
    #eval()
