from pydub import AudioSegment
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt

sound =AudioSegment.from_file('ode to joy.mp3')
s=np.array(sound.split_to_mono()[0].get_array_of_samples())
fs=sound.frame_rate
f,t,sxx=spectrogram(s,fs)

plt.pcolormesh(t,f,sxx,vmax=1 )
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()