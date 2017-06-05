import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy import signal
import sys
from scipy import misc
import scipy.io.wavfile as wavfile

def graphInitial():
	f = open("mmm.wav", "r")
	sampleRate, data = wavfile.read(f)
	if len(data.shape) == 2 and data.shape[1] == 2:
		data = data[:, 1]	# Remove the second channel if it exists

	data = data*1.0/np.max(np.abs(data))
	with open("output.wav", "w") as f:
		wavfile.write(f, sampleRate, data)

	times = np.arange(len(data))/float(sampleRate)

	# Make the plot
	# You can tweak the figsize (width, height) in inches
	xmin = 1.5
	xmax = 1.55
	ymin = -1.0
	ymax = 1.0

	plt.figure(figsize=(30, 4))
	plt.plot(times, data, color='b') 
	plt.xlim(times[0], times[-1])
	plt.axis([xmin,xmax,ymin,ymax])
	plt.xlabel('time (s)')
	plt.ylabel('amplitude')
	plt.savefig('mmm.png', dpi=100)
	plt.show()
	plt.clf()

def calculateFourier():
 	f = open("mmm.wav", "r")
	sampleRate, data = wavfile.read(f)
	if len(data.shape) == 2 and data.shape[1] == 2:
		data = data[:, 1]	# Remove the second channel if it exists

	fourier = np.absolute(np.fft.fft(data))
	print fourier.shape
	tupleIndex = range(167872)
	print fourier[0:30]
	plt.plot(tupleIndex, fourier)
	plt.show()


def echoCalculation():
	f = open("newecho.wav", "r")
	sampleRate, data = wavfile.read(f)
	if len(data.shape) == 2 and data.shape[1] == 2:
		data = data[:, 1]	# Remove the second channel if it exists

	f1 = np.zeros(data.shape)
	f1[0] = 1
	index = int(.2*sampleRate)
	f1[index] = .5

	convolutedVersion = signal.fftconvolve(data, f1)

	print convolutedVersion.shape
	xaxis = range(311167)

	xmin = 21000
	xmax = 22000
	ymin = -1000
	ymax = 1000
	plt.axis([xmin,xmax,ymin,ymax])
	plt.plot(xaxis, convolutedVersion)
	plt.show()

	# convolutedVersion = np.convolve(np.fft.fft(data), np.fft.fft(f1))

	# convolutedVersion = (convolutedVersion*1.0/np.max(np.abs(convolutedVersion)))
	with open("outputEcho.wav", "w") as f:
		wavfile.write(f, sampleRate, convolutedVersion)

#mmm 167872
echoCalculation()
