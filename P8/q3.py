import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import scipy.io.wavfile as wavfile


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

