#!/bin/bash
echo 'removing useless shit from img/ foler... '

rm -rv /Users/wsm/Downloads/imgs/kaggle*lmdb

python prepare.py
