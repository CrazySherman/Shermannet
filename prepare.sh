#!/bin/bash
echo 'removing useless shit from img/ foler... '

rm -rv ../../data/kaggle/kaggle*lmdb


python prepare.py $1
