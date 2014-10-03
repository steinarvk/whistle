import collections
import numpy

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

if __name__ == '__main__':
    from wavy import WaveFileReader
    import sys
    filename = sys.argv[1]
    wav = WaveFileReader(filename)
    size = int(wav.framerate * 0.1)
    windows = rolling_window(merge_channels(wav), size)
    k = 100
    windows = every_nth(k, windows)
    timeskip = k / float(wav.framerate)
    t = 0.0
    for window in windows:
        freqamps = fft(window, wav.framerate)
        simplified = simplify_frequencies(freqamps, 10)
        simplified = remove_minor_frequencies(simplified, 0.02)
        if simplified:
            print >> sys.stderr, simplified
            for freq, amp in simplified:
                print t, freq
        t += timeskip
