import os 
import numpy as np
import caffe
import cv2
import random, lmdb, csv

os.chdir('/Users/wsm/Downloads/imgs')
imgs_size = 4000000000
map_size = imgs_size * 5
# desired resize for the resulted imgs
h = 256
w = 256
env = lmdb.open('kaggle_train_lmdb', map_size=map_size)
env2 = lmdb.open('kaggle_test_lmdb', map_size=map_size/ 10)

def load_imgs_from_dict(dict, keys, txn):
    counter = 0
    for i in keys:
        for file in dict[i]:
            label = int(file.split('/')[1][1:])
            img = cv2.imread(file)
            img = cv2.resize(img, (w, h)).transpose(2,0,1)
            datum = caffe.io.array_to_datum(img, label) 
            str_id = '{:08}'.format(counter)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            counter += 1
            # print 'processing img ', file, ': ', str_id + ', ' + str(label)


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




    