import wave
import struct

class WaveFileReader (object):
    def __init__(self, filename):
        self._wave = wave.open(filename)
        self.channels = self._wave.getnchannels()
        self.framerate = self._wave.getframerate()
        self.total_frames = self._wave.getnframes()
        self.frameno = 0
        self.endianness = "<"
        self.sampleformat = "h"
        self.sample_max = 32767
        self.sample_min = -32768

    @property
    def frametime(self):
        return self.frameno / float(self.framerate)

    def close(self):
        if self._wave:
            self._wave.close()
            self._wave = None

    def _normalize(self, sample):
        span = self.sample_max - self.sample_min
        unit = (sample - self.sample_min) / float(span)
        return 2 * unit - 1

    def read(self):
        data = self._wave.readframes(1)
        self.frameno += 1
        fmt = self.endianness + self.channels * self.sampleformat
        return map(self._normalize, struct.unpack(fmt, data))

    def eof(self):
        return self.frameno >= self.total_frames

    def __iter__(self):
        while not self.eof():
            yield self.read()
    
    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

class WaveFileWriter (object):
    def __init__(self, filename, channels=1, rate=44100):
        self._wave = wave.open(filename, "w")
        self._wave.setsampwidth(2)
        self._wave.setnchannels(channels)
        self._wave.setframerate(rate)
        self.channels = channels
        self.rate = rate
        self.endianness = "<"
        self.sampleformat = "h"
        self.sample_max = 32767
        self.sample_min = -32768

    def write(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            x = (x,)
        assert len(x) == self.channels
        fmt = self.endianness + self.channels * self.sampleformat
        data = struct.pack(fmt, *map(self._unnormalize, x))
        self._wave.writeframes(data)

    def writes(self, xs):
        for x in xs:
            self.write(x)
        
    def _unnormalize(self, sample):
        unit = (sample + 1.0) / 2.0
        span = self.sample_max - self.sample_min
        return self.sample_min + span * unit

    def close(self):
        self._wave.close()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    with WaveFileReader(filename) as w:
        t = 0
        for (value,) in w:
            print t, value
            t = w.frametime
