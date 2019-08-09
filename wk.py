from pydub import AudioSegment
from pydub.playback import play
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
class Wave:
    """Represents a sound in terms of a numpy array"""
        
    def __init__(self,**kwargs):
        
        self.data-kwargs.pop("data",None)
        self.__sample_width = kwargs.pop("sample_width", None)
        self.__frame_rate = kwargs.pop("frame_rate", None)
        self.__n_channels = kwargs.pop("n_channels", None)

        audio_params = (self.data,self.__sample_width, self.__frame_rate, self.__channels)
        
        if any(audio_params) and None in audio_params:
            raise Exception("Either all audio parameters or no parameter must be specified")
        if self.data and not isinstance(self.data,np.array):
            raise Exception("data must be np.array")
    def from_audio_path(self,path):
        if not path.exists(path):
            raise Exception("No file found at {}".format(path))
        seg=AudioSegment.from_file(path)
        self.from_segment(seg)
    
    def from_segment(self,seg):
        self.__sample_width=seg.sample_width
        self.__frame_rate=seg.frame_rate
        self.__n_channels=seg.channels
        seg=seg.split_to_mono()
        dtype={1:np.int8,2:np.int16}[self.__sample_width]
        self.data=np.array([s.get_array_of_samples() for s in seg],dtype=dtype)
        assert(self.data.shape[1]==seg.frame_count())
        
        
def clean_sound(fmin=None,fmax=None,data='ode to joy.mp3'):
    assert(fmin or fmax)
    sound=data
    if type(data)==str:
        sound =AudioSegment.from_file(data)
    s=np.array(sound.split_to_mono()[0].get_array_of_samples())
    print(len(s))
    fs=sound.frame_rate
    f,t,sxx=signal.stft(s,fs=fs)
    #sxx=np.where(sxx<=1,sxx,0)
    """if fmin:
        K=np.sum(f<fmin)+1
        sxx,f=sxx[K:],f[K:]
    if fmax:
        K=np.sum(f<fmax)
        sxx=sxx[:K]
    """
    print(sxx.shape)
    clean_s=signal.istft(sxx,fs=fs)[1]
    print(clean_s.shape)
    return AudioSegment(clean_s,frame_rate=fs,channels=1,sample_width=2)

def spectrum(data):
    if type(data)==str:
        data =AudioSegment.from_file(data)
    fs=data.frame_rate
    s=np.array(data.split_to_mono()[0].get_array_of_samples())
    f,pxx=signal.welch(s,fs=fs)
    plt.plot(f[:30],pxx[:30])