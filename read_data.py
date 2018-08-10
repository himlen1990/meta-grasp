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


class dataReader(object):
    def __init__(self, dataset_dir, start, end):
        self.dataset_dir = dataset_dir
        folders = range(start,end) #883
        folders = ['%02d'%(i) for i in folders]
        #images = []
        #labels = []
        self.filenames = []
        self.cell_size = 6
        self.image_size = 224
        self.cell_length = self.image_size/self.cell_size

        for i in folders:
            samples_per_object = []
            for name in glob.glob(os.path.join(dataset_dir, i, '*.jpg')):
                samples_per_object.append(name)
            self.filenames.append(samples_per_object)

    def label_processor(self,label):
        new_label = np.zeros((self.cell_size,self.cell_size,4))
        for i in range(label.shape[0]):
            x = label[i,0]/2
            x_ind = int(x * self.cell_size/self.image_size)
            bias_x = x - (x_ind * self.cell_length)
            y = label[i,1]/2
            y_ind = int(y * self.cell_size/self.image_size)
            bias_y = y - (y_ind * self.cell_length)            
            theta = label[i,2] * 10
            grasp = [bias_x, bias_y, theta]
            new_label[y_ind, x_ind, 0] = 10
            new_label[y_ind, x_ind, 1:4] = grasp
        return new_label
            

    def sample_mini_dataset(self, dataset_size):

        #sorted_filename = []

        #for i in range(len(self.filenames)):
        #    sort = sorted(self.filenames[i], key = lambda name: int(name[-8:-4]))
        #    sorted_filename.append(sort)
        random.shuffle(self.filenames)

        mini_dataset = self.filenames[:dataset_size]
        #mini_dataset = sorted_filename[:dataset_size]
        #print mini_dataset
        multi_distributions = []
        for an_object in mini_dataset:
            imgs = []
            labels = []
            for grasp in an_object:
                img = cv2.imread(grasp)    
                resize_img = cv2.resize(img,(0,0), fx=0.5, fy=0.5)
                label_file = grasp[:-4]+'.npy'       
                label = np.load(label_file)
                processed_label = self.label_processor(label)
                imgs.append(resize_img)
                labels.append(processed_label)
            single_distribution = zip(imgs,labels)
            multi_distributions.append(single_distribution)
        '''
        sample_batch = random.sample(multi_distributions[0], 7)
        print "abc"
        print len(sample_batch)
        a,b = zip(*sample_batch)

        for dists in a:

            io.imshow(dists)
            io.show()
        '''
        return multi_distributions


    def get_mini_batches(self, single_batch, num_samples):
        
        sample_batch = random.sample(single_batch, num_samples)
        sample_img, sample_label = zip(*sample_batch)
        return sample_img, sample_label

    def get_test_data(self):
        folders = range(400,880) #883
        folders = ['%04d'%(i) for i in folders]
        filenames = []
        images = []
        labels = []
        for i in folders:
            for name in glob.glob(os.path.join(self.dataset_dir, i, '*.png')):
                filenames.append(name)
        for filename in filenames:
            img = io.imread(filename)
            label_file = filename[:-4]+'.npy'
            label = np.load(label_file)
            images.append(img)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        return images[:100], labels[:100]

    def split_dataset(self,num_train=70):
        self.dataset = zip(self.all_images, self.all_labels)
        random.shuffle(self.dataset)
        return self.dataset[:num_train], self.dataset[num_train:]


    def sample_batch(self, batch_size=10):
        idx = np.random.choice(self.train_img.shape[0], batch_size)
        return self.train_img[idx], self.train_label[idx]



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
        topn = 2
        for i in range(topn):            
            idx = np.unravel_index(n_max[i],mask.shape)
            topn_idx.append(idx)
        print "!!!!!!!!!!!"
        print topn_idx
        print n_max
        print "!!!!!!!!!!!"
        #rr,cc = np.where(mask > 3)
        #print rr,cc


        circle_x = x[topn_idx[0][0]][topn_idx[0][1]] + topn_idx[0][1] * cell_length
        circle_y = y[topn_idx[0][0]][topn_idx[0][1]] + topn_idx[0][0] * cell_length
        sin = math.sin(theta[topn_idx[0][0]][topn_idx[0][1]])
        cos = math.cos(theta[topn_idx[0][0]][topn_idx[0][1]])
        delta_x = 10 * cos
        delta_y = 10 * sin
        rl,cl = line(int(circle_y),int(circle_x),int(circle_y+delta_y),int(circle_x+delta_x))
        r,c = circle(circle_y,circle_x,3)
        image[r,c] = (20,220,220)
        image[rl,cl] = (20,220,220)
        print circle_x, circle_y, theta[topn_idx[0][0]][topn_idx[0][1]], mask[topn_idx[0][0]][topn_idx[0][1]]

        for i in range(1,len(topn_idx)):
            print topn_idx[i]
            circle_x = x[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][1] * cell_length
            circle_y = y[topn_idx[i][0]][topn_idx[i][1]] + topn_idx[i][0] * cell_length
            sin = math.sin(theta[topn_idx[i][0]][topn_idx[i][1]])
            cos = math.cos(theta[topn_idx[i][0]][topn_idx[i][1]])
            #delta_x = math.cos(theta[rr[i]][cc[i])
            delta_x = 10 * cos
            delta_y = 10 * sin
            rl,cl = line(int(circle_y),int(circle_x),int(circle_y+delta_y),int(circle_x+delta_x))
            r,c = circle(circle_y,circle_x,3)
            
            image[r,c] = (20,220,20)
            image[rl,cl] = (20,220,20)

        io.imshow(image)
        io.show()
            
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
        io.imshow(image)
        io.show()

#dr = dataReader('./testset',0,5)
#a = dr.sample_mini_dataset(1)
#test_img, test_label = zip(*a[0])

