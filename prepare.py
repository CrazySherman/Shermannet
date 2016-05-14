import os, sys
import numpy as np
import caffe
#from skimage import io as skiio
#from skimage.transform import resize
import cv2
import random, lmdb, csv
from skimage import io as skiio
from skimage.transform import resize
from data_aug import aug_algo


imgs_size = 4000000000
map_size = imgs_size * 10
# desired resize for the resulted imgs
# ImgNet specs
h = 256
w = 256
env = lmdb.open('data/kaggle/kaggle_train_lmdb', map_size=map_size)
env2 = lmdb.open('data/kaggel/kaggle_test_lmdb', map_size=map_size/10)

def load_imgs_from_dict(dict, keys, txn, augment=False):
    counter = 0
    if augment:
        print 'using augmentation...'
    else:
        print 'augmentation disabled...'

    for i in keys:
        for file in dict[i]:
            label = int(file.split('/')[1][1:])
            ## using opencv
            img = cv2.imread(file)
            
            ## using skimage -- this shit is way too slow
            # img = skiio.imread(file)
            # img = img[:,:,[2,1,0]]  #Skimage read that shit as BGR, fuq
            # img = resize(img, (h,w)).transpose(2,0,1)
            imgs = [img]
            ## data augmentation
            if (augment):
            ## Shift left
                img_shift_left = algo.horizShiftLeft(img, 0.05, 0.15)
                imgs.append(img_shift_left)
            ## Shift Right
                img_shift_right = algo.horizShiftRight(img, 0.05, 0.15)
                imgs.append(img_shift_right)
            ## Vertical shift up
                img_shift_up = algo.vertiShiftUp(img, 0.05, 0.15)
                imgs.append(img_shift_up)
            ## Vertical shift down
                img_shift_down = algo.vertiShiftDown(img, 0.05, 0.15)
                imgs.append(img_shift_down)
            ## Rotate CCW
                img_rotate_cw = algo.rotatedCW(img, 0.15, 0.25, 1.2)
                imgs.append(img_rotate_cw)
            ## Rotate CW
                img_rotate_ccw = algo.rotatedCCW(img, 0.15, 0.25, 1.2)
                imgs.append(img_rotate_ccw)
            ## Crop and zoom
                img_zoom = algo.cropSkretch(img, 0.05, 0.15)
                imgs.append(img_zoom)
            for img in imgs:
                img = cv2.resize(img, (w, h)).transpose(2,0,1)
                datum = caffe.io.array_to_datum(img, label) 
                str_id = '{:08}'.format(counter)
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
                counter += 1
                print 'processing img ', file, ': ', str_id + ', ' + str(label)

if __name__ == '__main__':

    if sys.argv != 2:
        print 'Usage: prepare.py [kaggle raw image directory]'
        return 

    env = lmdb.open('data/kaggle/kaggle_train_lmdb', map_size=map_size)
    env2 = lmdb.open('data/kaggel/kaggle_test_lmdb', map_size=map_size/10)
    os.chdir(sys.argv[1])
    arr = []
    with open('driver_imgs_list.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            arr.append(row[0].split(','))

    dict = {}
    for line in arr:
        if line[0] in dict:
            dict[line[0]].append(os.path.join('train', line[1], line[2]))
        else:
            dict[line[0]] = [os.path.join('train', line[1], line[2])]
    if dict.has_key('subject'):
        del dict['subject']
    ids = dict.keys()
    random.shuffle(ids)
    tst_driver_set, train_driver_set = ids[:3], ids[3:]
    print 'testing driver id set: ', tst_driver_set
    print 'training driver id set: ', train_driver_set
    with env.begin(write=True) as txn, env2.begin(write=True) as txn2:
        # load training set
        print 'loading training data set...'
        load_imgs_from_dict(dict, train_driver_set, txn)
        print 'loading testing data set...'
        # load testing image set
        load_imgs_from_dict(dict, tst_driver_set, txn2)




    
