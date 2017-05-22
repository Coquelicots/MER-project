import pylab as pl
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import shutil

def compare(a,b):
	if len(a) == len(b):
		return cmp(a,b)
	return cmp(len(a),len(b))

dataset_path = '../Data/rock/'
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)

c = 101
for f in train_fns:
	os.rename(dataset_path+f,dataset_path+str(c)+'.mp3')
	c+=1
