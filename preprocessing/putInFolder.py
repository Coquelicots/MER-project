import pylab as pl
import numpy as np
from os import listdir
from os.path import isfile, join
import shutil

def compare(a,b):
	if len(a) == len(b):
		return cmp(a,b)
	return cmp(len(a),len(b))

dataset_path = '../SmallPic/'
des_path = '../picIntag/'
des_folders = ['calm','joy','nostalgia','tenderness','solemnity','power','sadness','tension','amazement']
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)

tags = np.genfromtxt('../tag.txt', delimiter=',')
tags = tags.astype('int')

for x in range(len(tags)):
	srcfile = dataset_path+train_fns[x]
	dstdir = des_path + des_folders[tags[x]]
	shutil.copy(srcfile, dstdir)
