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

dataset_path = '../Data/pop/'
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)
print train_fns


count = 1
data = []
for fn in train_fns:
	y,sr = librosa.core.load((dataset_path + fn),mono = True)
	D = librosa.stft(y)
	S = librosa.amplitude_to_db(D,ref=np.max)
	plt.figure()
	librosa.display.specshow(S)
	plt.savefig('../Pic/pop/'+str(count)+'.png',format='png')
	plt.close()
	count+=1
