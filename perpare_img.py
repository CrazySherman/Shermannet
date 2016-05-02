import numpy as np
import lmdb
import caffe
import os
import cv2
os.chdir('/Users/wsm/Downloads/imgs')

# method similar to CVmat2Datum
def cvMat2Datum(cv_img, height, width, channels):
    data = np.zeros((channels, height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            for c in range(channels):
                    data[c,h,w] = cv_img[h,w,c]
    return data

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

imgs_size = 4000000000
map_size = imgs_size * 5
# desired resize for the resulted imgs
h = 256
w = 256
# according to caffe issue, 2 dbs need to be setup for vectorized label shit
env = lmdb.open('kaggle_img_lmdb', map_size=map_size)
env2 = lmdb.open('kaggle_label_lmdb', map_size=20000 * 50)

counter = 0
with env.begin(write=True) as txn, env2.begin(write=True) as txn2:
    # txn is a Transaction object
    for i in range(10):
        dir = os.path.join('train', 'c' + str(i))
        print 'start loading label ', i
        for f in os.listdir(dir):
            if f.endswith('.jpg'):
                f = os.path.join(dir, f)
                img = cv2.imread(f)
                img = cv2.resize(img,(w,h)).transpose((2,0,1))
                datum = caffe.io.array_to_datum(img)
                label_arr = np.zeros((1,1,10), dtype=np.uint8)
                label_arr[0,0,i] = 1
                datum_label = caffe.io.array_to_datum(label_arr)
                # datum = caffe.proto.caffe_pb2.Datum()
                # datum.channels = img.shape[2]
                # datum.height = img.shape[0]
                # datum.width = img.shape[1]
                # datum.data = cvMat2Datum(img, datum.height, datum.width, datum.channels).tobytes() 
                
                # datum.label = i
                str_id = '{:08}'.format(counter)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
                txn2.put(str_id.encode('ascii'), datum_label.SerializeToString())
                counter += 1
                if counter % 100 == 0:
                    print counter, ' imgs processed'


        