from pydub import AudioSegment
from pydub.playback import play

sound=AudioSegment.from_file('ode to joy.mp3')
play(sound[ 5000:])