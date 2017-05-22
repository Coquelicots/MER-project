import pylab as pl
import numpy as np
from os import listdir
from os.path import isfile, join
import PIL.Image as Image

def compare(a,b):
	if len(a) == len(b):
		return cmp(a,b)
	return cmp(len(a),len(b))

dataset_path = '../Pic/rock/'
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)

count = 0
data = []
for fn in train_fns:
	d = Image.open((dataset_path + fn))
	d = d.resize((256,256),Image.BILINEAR)
	d.save(("../SmallPic/rock/"+fn))
	del d
