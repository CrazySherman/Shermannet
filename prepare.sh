#!/bin/bash
echo 'removing useless shit from img/ foler... '

rm -rv ~/Kaggle_driver_dataset/kaggle*lmdb

python prepare.py
