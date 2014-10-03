import math

def sin_wave(x):
    return math.sin(2*math.pi*x)

def triangle_wave(x):
    if x <= 0.5:
        return 4.0*x - 1.0
    return 3.0 - 4.0*x

def square_wave(x):
    return 1 if x <= 0.5 else -1

def sawtooth_wave(x):
    return 4.0*x - 1.0

class VariableFrequencyWave (object):
    def __init__(self, rate, freq, amp=1.0, function=sin_wave):
        self.frequency = freq
        self.amplitude = amp
        self.dt = 1.0/float(rate)
        self.t = 0.0
        self.function = function

    def advance(self):
        self.t += self.dt * self.frequency
        self.t %= 1.0

    def next(self):
        rv = self.function(self.t)
        self.advance()
        return self.amplitude * rv

def calculate_samples(rate, duration):
    return int(duration*float(rate)+0.5)

def generate_wave(rate, duration, freqf, ampf, function=sin_wave):
    wave = VariableFrequencyWave(rate, None, None, function)
    freqf = funcify(freqf)
    ampf = funcify(ampf)
    rt = 0.0
    nsamples = calculate_samples(rate, duration)
    for i in range(nsamples):
        rt = i / float(nsamples)
        wave.frequency = freqf(rt)
        wave.amplitude = ampf(rt)
        yield wave.next()

def generate_silence(rate, duration):
    for i in range(calculate_samples(rate, duration)):
        yield 0.0

def const(k):
    return lambda t : k

def funcify(f):
    if not callable(f):
        return const(f)
    return f

def lerp(t, a, b):
    span = float(b - a)
    return a + span * t

def linear(a, b):
    def f(t):
        return lerp(t, a, b)
    return f

def interpolate(points):
    assert points[0][0] == 0
    assert points[-1][0] == 1
    print points
    def f(t):
        i = 0
        try:
            while t > points[i+1][0]:
                i += 1
            t0 = points[i][0]
            t1 = points[i+1][0]
        except IndexError:
            print t, "'"
            return points[-1][1]
        assert t1 > t0
        return lerp((t-t0) / float(t1-t0), points[i][1], points[i+1][1])
    return f

def adsr(amplitude):
    duration = 1.0
    initial_val = 0.0 * amplitude
    attack_val = amplitude
    sustain_val = 0.9 * amplitude
    decay_val = 0.0 * amplitude
    attack_proportion, attack_max = 0.2, 0.1
    release_proportion, release_max = 0.2, 0.1
    decay_proportion, decay_max = 0.4, 0.2
    def calculate(proportion, max_):
        return min(max_, duration * proportion)
    attack = calculate(attack_proportion, attack_max)
    release = calculate(release_proportion, release_max) + attack
    decay = duration - calculate(decay_proportion, decay_max)
    return interpolate([
          (0, initial_val)
        , (attack, attack_val)
        , (release, sustain_val)
        , (decay, sustain_val)
        , (1, decay_val)
    ])

if __name__ == '__main__':
    from wavy import WaveFileWriter
    import sys
    import random
    filename = sys.argv[1] + ".generated.wav"
    function = sin_wave
    writer = WaveFileWriter(filename)
    def genwave(*args, **kwargs):
        return generate_wave(writer.rate, *args, **kwargs)
    def gensil(duration):
        return generate_silence(writer.rate, duration)
    for i in range(6):
        writer.writes(genwave(0.5, linear(440 + 40 * i, 430 + 40 * i), adsr(0.5), function=sin_wave))
        writer.writes(gensil(0.1))
    writer.writes(gensil(0.5))
    n = 6
    rv = []
    for i in range(n):
        f = random.random() * 880.0 + 220.0
        rv.append((i/float(n-1), f))
    writer.writes(genwave(0.25*n, interpolate(rv), adsr(0.5), function=sin_wave))
    writer.writes(gensil(0.5))
    writer.close()
