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

dataset_path = 'Data/classical/'
train_fns = [fn for fn in listdir(dataset_path)]
train_fns.sort(cmp = compare)

data = []
for fn in train_fns:
	y,sr = librosa.core.load((dataset_path + fn),mono = True)
	S, phase = librosa.magphase(librosa.stft(y))
	data.append(S)

data = np.array(data)
data = data.reshape((data.shape+(1,)))
data.dump('Classical_Data_Pack.txt')

'''
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
print rolloff.shape
cent = librosa.feature.spectral_centroid(y=y, sr=sr)
print cent.shape
zcr = librosa.feature.zero_crossing_rate(y)
print zcr.shape
'''