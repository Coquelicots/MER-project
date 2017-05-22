import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from os import listdir
from os.path import isfile, join

def compare(a,b):
	if len(a) == len(b):
		return cmp(a,b)
	return cmp(len(a),len(b))

dataset_path = '../Data/AllInOne/'
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)

F = open('feature.txt','w') 

count = 1
for fn in train_fns:
	y,sr = librosa.core.load((dataset_path + fn),mono = True)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	rolloff = 0.85*(np.sum(rolloff))
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	cent = np.mean(cent)
	zerocross = librosa.feature.zero_crossing_rate(y)
	zerocross = np.mean(zerocross)
	data = str(rolloff)+','+str(cent)+','+str(zerocross)+'\n'
	print data
	F.write(data)
	print count
	count += 1
F.close()
