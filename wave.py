import pydub
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
from math import ceil
import simpleaudio as sa
class Wave:
    """Represents a sound in terms of a numpy array"""
        
    def __init__(self,**kwargs):
        
        self.data=kwargs.pop("data",None)
        self.__sample_width = kwargs.pop("sample_width", None)
        self.__frame_rate = kwargs.pop("frame_rate", None)
        self.__n_channels = kwargs.pop("n_channels", None)
        self.__n_frames=None
        
        self.spectrogram=None

        audio_params = (self.data,self.__sample_width, self.__frame_rate, self.__n_channels)

        #if self.data!=None and not isinstance(self.data,np.array):
         #   raise Exception("data must be np.array")
    def __spectrum(self,i,nseg):
        if not 0<=i<self.__n_frames:
            raise Exception("Invalid indexes")
        res= np.fft.rfft(self.data[:,i:i+nseg],n=nseg)
        return res
    def play(self):
        #if not self.data:
         #   raise Exception("No data loaded")
        playback=sa.play_buffer(self.data[0],1,self.__sample_width,self.__frame_rate)
        try:
            playback.wait_done()
        except KeyboardInterrupt:
            playback.stop()
    @classmethod
    def from_audio_path(cls,s):
        if not path.exists(s):
            raise Exception("No file found at {}".format(s))
        seg=pydub.AudioSegment.from_file(s)
        return cls.from_segment(seg)
        
    @classmethod
    def from_segment(cls,seg):
        wave=Wave()
        wave.__sample_width=seg.sample_width
        wave.__frame_rate=seg.frame_rate
        wave.__n_channels=seg.channels
        frames=seg.frame_count()
        seg=seg.split_to_mono()
        dtype={1:np.int8,2:np.int16}[wave.__sample_width]
        wave.data=np.array([s.get_array_of_samples() for s in seg],dtype=dtype)
        assert(wave.data.shape[1]==frames)
        wave.__n_frames=wave.data.shape[1]
        return wave
    
    def n_frames(self):
        return self.__n_frames
    def plot_spectro(self):
        if not self.spectrogram:
            s=self.make_spectro(save=False)
        else:
            s=self.spectrogram
            
        s.plot()
        
    def make_spectro(self,nseg=200,overlap=None,save=True):
        ##TODO: check when nseg!=2n and overlap!=None
        """ Overlap defalts to nseg/2
            Padding is compulsory
            Returns (saves) (n_channel,n_frames/(nseg-n_overlap)-1,len(freq_range)) array"""
        if not overlap:
            overlap=int(nseg/2)
        t_range=ceil(self.__n_frames/(nseg-overlap))-1
        
        t=np.arange(t_range)/(t_range-1)*(self.__n_frames/self.__frame_rate)
        freqs=np.fft.rfftfreq(nseg,d=1/self.__frame_rate)
        print(t_range)
        idxs=np.linspace(0,self.__n_frames,t_range).astype(int)
        spec=np.empty((self.__n_channels,t_range,len(freqs)))
        print(idxs[:10])
        for i in range(t_range-1):
            spec[:,i,:]=self.__spectrum(idxs[i],nseg)
        spec=Spectrogram(t,freqs,spec)
        if save:
            self.spectrogram=spec
        return spec
    
        
class Spectrogram:
    def __init__(self,t,f,spec):
        assert(spec.shape[1]==len(t) and spec.shape[2]==len(f))
        self.t=t
        self.f=f
        self.spec=spec
        self.__phases=np.fft.rfftfreq(2*(len(f)-1))
    def plot(self,channel=0):
        if channel>=self.spec.shape[0]:
            raise Exception("Invalid channel")
        plt.pcolormesh(self.f, self.t, self.spec[channel,:,:],vmax=1)
        
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    def get_wave(self,overlap):
        aux=np.fft.irfft(self.spec,axis=2)
        nseg=aux.shape[2]
        res=np.empty((aux.shape[0],nseg*len(self.t)))
        for i in range(aux.shape[1]-1):
            res[:,i*nseg:(i+1)*nseg]=aux[:,i,:]
        return Wave(data=res,frame_rate=44100,n_channels=res.shape[0],sample_width=2)
        
        
        
        
        
        
    
