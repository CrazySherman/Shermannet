import os
from aug_algo import aug_algo
import pickle as pkl
import cv2
import numpy as np


algo = aug_algo()

data = []
y = []
augm = []

img = []
imgTemp = []
rootPath = '/Users/YingnanWang/Desktop/Spring2016/cs246/project/DataAugmentation/data'

for root, dirs, files in os.walk(rootPath, topdown=False):
    for filename in files:
        if filename != '.DS_Store':
            img = cv2.imread(os.path.join(root, filename))
            data.append(np.array(img))
            y.append(os.path.join(root, filename)[74])
            augm.append(0)

            imgTemp = algo.horizShiftLeft(img, 0.05, 0.15)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(1)

            imgTemp = algo.horizShiftRight(img, 0.05, 0.15)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(2)

            imgTemp = algo.vertiShiftUp(img, 0.05, 0.15)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(3)

            imgTemp = algo.vertiShiftDown(img, 0.05, 0.15)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(4)

            imgTemp = algo.rotatedCW(img, 0.1, 0.25, 1.2)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(5)

            imgTemp = algo.rotatedCCW(img, 0.1, 0.25, 1.2)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(6)

            imgTemp = algo.cropSkretch(img, 0.05, 0.15)
            data.append(np.array(imgTemp))
            y.append(os.path.join(root, filename)[74])
            augm.append(7)



pkldata = {'data': data,
           'label': y,
           'augmType': augm}


with open('augm.pkl', 'wb') as handle:
    pkl.dump(pkldata, handle)

print len(pkldata['data'])