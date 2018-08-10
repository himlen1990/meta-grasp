import skimage.io as io
from skimage.draw import circle
from skimage.draw import line
from skimage.transform import resize
import numpy as np
import glob
import os
import random
import math
import cv2

class dataReader_regression(object):
    def __init__(self, dataset_dir, n):
        self.dataset_dir = dataset_dir
        folders = range(1,n) 
        folders = ['%02d'%(i) for i in folders]
        filenames = []
        images = []
        labels = []
        label_processed = []
        self.cell_size = 8
        self.image_size = 224
        self.cell_length = self.image_size/self.cell_size

        for i in folders:
            for name in glob.glob(os.path.join(dataset_dir, i, '*.jpg')):            
                filenames.append(name)

        for filename in filenames:
            img = cv2.imread(filename)
            resize_img = cv2.resize(img,(0,0), fx=0.5, fy=0.5)
            label_file = filename[:-4]+'.npy'       
            label = np.load(label_file)
            images.append(resize_img)
            labels.append(label)

        for label in labels:
            new_label = np.zeros((self.cell_size,self.cell_size,4))
            for i in range(label.shape[0]):
                x = label[i,0] / 2
                x_ind = int(x * self.cell_size/self.image_size)
                bias_x = x - (x_ind * self.cell_length)
                y = label[i,1] / 2
                y_ind = int(y * self.cell_size/self.image_size)
                bias_y = y - (y_ind * self.cell_length)
                theta = label[i,2] * 10
                grasp = [bias_x,bias_y,theta]
                new_label[y_ind,x_ind,0] = 10
                new_label[y_ind,x_ind,1:4] = grasp
            label_processed.append(new_label)
        self.all_images = np.array(images)
        self.all_labels = np.array(label_processed)
        #print label_processed[0]
        print "total samples: ",self.all_labels.shape[0]
        
    def sample_batch(self, batch_size=10):
        idx = np.random.choice(self.all_images.shape[0], batch_size)
        return self.all_images[idx], self.all_labels[idx]



    def show_grasp(self,image,predict,ground_true):

        mask = predict[:,:,0]
        x = predict[:,:,1]
        y = predict[:,:,2]
        theta = predict[:,:,3]

        cell_length = image.shape[0]/mask.shape[0]
        color_mask = np.zeros((cell_length, cell_length, 3))
        color_mask[:,:,0] = 255
        #print mask.shape

        test = mask.flatten()
        n_max = test.argsort()[-3:][::-1]
        topn_idx = []
        topn = 3
        for i in range(topn):            
            idx = np.unravel_index(n_max[i],mask.shape)
            topn_idx.append(idx)

        circle_x = x[topn_idx[0][0]][topn_idx[0][1]] + topn_idx[0][1] * cell_length
        circle_y = y[topn_idx[0][0]][topn_idx[0][1]] + topn_idx[0][0] * cell_length
        sin = math.sin(theta[topn_idx[0][0]][topn_idx[0][1]])
        cos = math.cos(theta[topn_idx[0][0]][topn_idx[0][1]])
        delta_x = 20 * cos
        delta_y = 20 * sin
        cv2.circle(image, (int(circle_x), int(circle_y)), 3, (0,255,0), -1)
        cv2.line(image,(int(circle_x),int(circle_y)),(int(circle_x+delta_x),int(circle_y+delta_y)),(0,255,0),2)
        #print circle_x, circle_y, theta[topn_idx[0][0]][topn_idx[0][1]], mask[topn_idx[0][0]][topn_idx[0][1]]

        for i in range(1,len(topn_idx)):
            #print topn_idx[i]
            circle_x = x[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][1] * cell_length
            circle_y = y[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][0] * cell_length
            sin = math.sin(theta[topn_idx[i][0]][topn_idx[i][1]])
            cos = math.cos(theta[topn_idx[i][0]][topn_idx[i][1]])
            #delta_x = math.cos(theta[rr[i]][cc[i])
            delta_x = 20 * cos
            delta_y = 20 * sin
            cv2.circle(image, (int(circle_x), int(circle_y)), 3, (0,255,0), -1)
            cv2.line(image,(int(circle_x),int(circle_y)),(int(circle_x+delta_x),int(circle_y+delta_y)),(0,255,0),2)
        '''
        print "ground true"
        mask_gt = ground_true[...,0]
        x_gt = ground_true[...,1]
        y_gt = ground_true[...,2]
        theta_gt = ground_true[...,3]
        rr2,cc2 = np.where(mask_gt > 0.0)

        for i in range(rr2.shape[0]):
            circle_xgt = x_gt[rr2[i]][cc2[i]] + cc2[i] * cell_length
            circle_ygt = y_gt[rr2[i]][cc2[i]] + rr2[i] * cell_length

            sin_gt =  math.sin(theta_gt[rr2[i]][cc2[i]])
            cos_gt =  math.cos(theta_gt[rr2[i]][cc2[i]])
            delta_xgt = 10 * cos_gt
            delta_ygt = 10 * sin_gt
            rl2,cl2 = line(int(circle_ygt),int(circle_xgt),int(circle_ygt+delta_ygt),int(circle_xgt+delta_xgt))
            r2,c2 = circle(circle_ygt,circle_xgt,3)
            print circle_xgt, circle_ygt, theta_gt[rr2[i]][cc2[i]], mask_gt[rr2[i]][cc2[i]]
            image[r2,c2]= (220,20,20)
            image[rl2,cl2] = (220,20,20)
        '''
        cv2.imshow("grasp",image)
        k = cv2.waitKey(0)

#dr = dataReader_regression('./object_dataset',2)
#a = dr.sample_mini_dataset(1)
#v,l = dr.get_mini_batches(a[0],10)

