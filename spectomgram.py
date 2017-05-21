import wave
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

f = wave.open('beautiful-violin-music.wav')
ff,sr = librosa.core.load('beautiful-violin-music.wav',mono = True)


# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
#print nchannels


str_data = f.readframes(nframes)
f.close()


wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 2
wave_data = wave_data.T
time = np.arange(0, nframes) * (1.0 / framerate)


pl.subplot(511) 
pl.plot(time, wave_data[0])
pl.subplot(512) 
pl.plot(time, wave_data[1], c="g")
pl.subplot(513)
ch1,ch1f,ch1t,ch1im = pl.specgram(wave_data[0],Fs=framerate)
pl.subplot(514)
ch2,ch2f,ch2t,ch2im = pl.specgram(wave_data[1],Fs=framerate)
pl.subplot(515)
librosa.display.waveplot(ff, sr=sr)
pl.show()
print ch1.shape
ch1.shape = (ch1.shape[0],ch1.shape[1],1)
print ch2.shape
