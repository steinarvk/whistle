import collections
import numpy
import heapq

def average(samples):
    return sum(samples)/float(len(samples))

def merge_channels(samplesseq):
    for samples in samplesseq:
        yield average(samples)

def every_nth(n, seq):
    i = 0
    for x in seq:
        if not i:
            yield x
        i += 1
        if i >= n:
            i = 0

def rolling_window(sampleseq, size):
    d = collections.deque(maxlen=size)
    for x in sampleseq:
        d.append(x)
        if len(d) >= size:
            yield d

def fft(window, sample_rate):
    spacing = 1.0 / float(sample_rate)
    values = numpy.fft.rfft(window)
    freqs = numpy.fft.rfftfreq(len(window), spacing)
    return [(freq, abs(val)) for freq, val in zip(freqs, values)]

def simplify_adjacent_frequencies(freqamps):
    rv = 0.0
    weight = 0.0
    for freq, amp in freqamps:
        rv += freq
        weight += amp
    return rv / float(len(freqamps)), weight

def simplify_frequencies(freqamps, threshold):
    island = []
    rv = []
    for freq, amp in freqamps:
        if amp > threshold:
            amp -= threshold
            island.append((freq, amp))
        else:
            if island:
                rv.append(simplify_adjacent_frequencies(island))
            island = []
    rv.sort(key = lambda x: x[1], reverse=True)
    return rv

def remove_minor_frequencies(simpfreqamps, ratio):
    if not simpfreqamps:
        return []
    threshold = float(ratio) * simpfreqamps[0][1]
    return [(freq, amp) for freq, amp in simpfreqamps if amp >= threshold]

def filter_frequencies(simpfreqamps, lower=120, upper=4000):
    return [(f, a) for f, a in simpfreqamps if lower <= f <= upper]

def window_amplitude(window, quantile=0.95):
    return numpy.percentile(map(abs, window), quantile*100)

# quantile amplitude appears to be a very reliable indicator of note length.
# for most simple notes, it is almost _entirely_ flat, stunningly so, with
# very quick transitions in between. sometimes for a "slide" there is a
# transition over a longer period.

# the actual amplitude variations are probably _more_ pronounced than they
# should be (probably because I'm a bad whistler and occasionally run out
# of breath). leaving out the amplitude variations altogether gives a nice
# 8-bit-style sound.

def deviations_from_mean(samples, x):
    m = numpy.mean(samples)
    d = numpy.std(samples)
    if d == 0:
        d = 0.00000001
    return abs(m - x) / float(d)

def remove_outliers(seq, horizon, deviations=1):
    d = collections.deque(maxlen=horizon*2+1)
    after = []
    for x in seq:
        d.append(x)
        if len(d) <= d.maxlen:
            yield x
        else:
            sample = d[horizon]
            before = d[:horizon]
            after = d[-horizon:]
            k = deviations_from_mean(before + after, sample)
            if abs(k) > deviations:
                yield numpy.mean(before + after)
            else:
                yield sample
    for x in after:
        yield x

def linear_fit(samples):
    n = len(samples)
    ((a,b), (errsum,), _, _, _) = numpy.polyfit(range(n), samples, 1, full=True)
    return a, b, errsum / float(n)

class NoteBreaks (object):
    def __init__(self, min_length=0.0):
        self.breaks = []
        self.min_length = min_length

    def add_break(self, t):
        heapq.heappush(self.breaks, t)

    def __iter__(self):
        if not self.breaks:
            return
        q = list(self.breaks)
        last = heapq.heappop(q)
        yield last
        while q:
            x = heapq.heappop(q)
            if x - last < self.min_length:
                continue
            last = x
            yield x

def find_breaks_by_delta(timeseq, windowsz, deviations=1.0):
    import sys
    skips = 0
    for windows in rolling_window(timeseq, windowsz * 2):
        if skips:
            skips -= 1
            continue
        windows = list(windows)
        old_window = [val for t, val in windows[:windowsz]]
        new_window = [val for t, val in windows[-windowsz:]]
        t = windows[-1][0]
        m = numpy.mean(new_window)
        dev = deviations_from_mean(old_window, m)
        if dev > deviations:
            print >> sys.stderr, "at", t, ": ", m, " is ", dev, " dev from ", numpy.mean(old_window), numpy.std(old_window)
            skips = windowsz
            yield t

if __name__ == '__main__':
    from wavy import WaveFileReader, WaveFileWriter
    from synth import VariableFrequencyWave
    import sys
    filename = sys.argv[1]
    wav = WaveFileReader(filename)
    size = int(wav.framerate * 0.1)
    windows = rolling_window(merge_channels(wav), size)
    k = 100
    windows = every_nth(k, windows)
    timeskip = k / float(wav.framerate)
    t = 0.0
    writer = WaveFileWriter("output.generated.wav")
    beep = VariableFrequencyWave(writer.rate)
    frequencies = []
    amplitudes = []
    times = []
    for window in windows:
        window = list(window)
        totalamp = window_amplitude(window)
        freqamps = fft(window, wav.framerate)
        simplified = simplify_frequencies(freqamps, 10)
        simplified = filter_frequencies(simplified)
        simplified = remove_minor_frequencies(simplified, 0.02)
        times.append(t)
        if simplified:
            beep.frequency = simplified[0][0]
            beep.amplitude = totalamp
        else:
            beep.frequency = beep.amplitude = 0.0
        frequencies.append(beep.frequency)
        amplitudes.append(beep.amplitude)
        t += timeskip
    frequencies = list(remove_outliers(frequencies, 100, deviations=0.25))
    amplitudes = list(remove_outliers(amplitudes, 100, deviations=0.25))
    with open("output.amplitude.txt", "w") as amplog, \
             open("output.frequency.txt", "w") as freqlog, \
             open("output.breaks.txt", "w") as breaklog:
        for t, frequency, amplitude in zip(times, frequencies, amplitudes):
            print >> amplog, t, amplitude
            print >> freqlog, t, frequency
            beep.frequency = frequency
            beep.amplitude = amplitude
            for i in range(k):
                if amplitude:
                    writer.write(beep.next())
                else:
                    writer.write(0.0)
        breaks = NoteBreaks(min_length=0.05)
        kk = 10
        for t in find_breaks_by_delta(zip(times, amplitudes), kk, deviations=4):
            breaks.add_break(t)
        for t in find_breaks_by_delta(zip(times, frequencies), kk, deviations=2):
            breaks.add_break(t)
        for t in breaks:
            print >> breaklog, t-0.001, 0.0
            print >> breaklog, t, 1.0
            print >> breaklog, t+0.001, 0.0
        

