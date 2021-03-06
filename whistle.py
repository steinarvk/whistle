import collections
import numpy
import heapq
import time
import math

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
            skips = windowsz
            yield t

def break_timeseq(timeseq, breaks):
    breaks = collections.deque(breaks)
    rv = []
    for t, v in timeseq:
        while breaks and t > breaks[0]:
            if rv:
                yield rv
                rv = []
            breaks.popleft()
        rv.append((t,v))
    if rv:
        yield rv

def linearize_timeseq(timeseq):
    t0, t1 = timeseq[0][0], timeseq[-1][0]
    ts = [t for t, v in timeseq]
    vs = [v for t, v in timeseq]
    a, b = numpy.polyfit(ts, vs, 1)
    v0 = a * t0 + b
    v1 = a * t1 + b
    return (t0, v0, t1, v1, list(timeseq))

def piecewise_linearize(timeseq, breaks):
    for subtimeseq in break_timeseq(timeseq, breaks):
        yield linearize_timeseq(subtimeseq)

def piecewise_filter_steepness(segments, threshold):
    for segment in segments:
        t0, v0, t1, v1 = segment[:4] 
        steepness = abs(v1-v0)/float(t1-t0)
        if steepness <= threshold:
            yield segment

def segment_interpolate(segment, t):
    t0, v0, t1, v1 = segment[:4]
    span = (v1 - v0) / float(t1 - t0)
    return v0 + span * (t - t0)

def piecewise_linear_interpolate(segments, t):
    # naive implementation! use sortedness
    for segment in segments:
        t0, v0, t1, v1 = segment[:4]
        if t <= t1:
            return segment_interpolate(segment, t)

def clip(min_, x, max_):
    return max(min_, min(max_, x))

def segment_square_error(segment):
    rv = 0.0
    samples = segment[4]
    for t, v in samples:
        vp = segment_interpolate(segment, t)
        rv += (v - vp) * (v - vp)
    return rv

def combine_segments(alpha, beta):
    all_samples = alpha[4] + beta[4]
    return linearize_timeseq(all_samples)

def merge_piecewise_linear(segments, tolerance=0):
    segiter = iter(segments)
    last = segiter.next()
    for x in segiter:
        lastx = combine_segments(last, x)
        threshold = segment_square_error(last) + segment_square_error(x)
        threshold *= (1.0 + tolerance)
        err = segment_square_error(lastx)
        if err <= threshold:
            last = lastx
        else:
            yield last
            last = x
    if last:
        yield last

def snap_piecewise_linear(segments, vtol=0.1, ttol=0.02):
    segiter = iter(segments)
    last = segiter.next()
    for x in segiter:
        lastt, lastv = last[2], last[3]
        nowt, nowv = x[0], x[1]
        meanv = 0.5 * (lastv + nowv)
        devv = abs(lastv - nowv) / meanv
        merge = devv < vtol and abs(nowt - lastt) < ttol
        if merge:
            combinet = 0.5 * (lastt + nowt)
            last = list(last)
            last[2:4] = combinet, meanv
            x = list(x)
            x[0:2] = combinet, meanv
        yield last
        last = x
    if last:
        yield last

def frequency_to_midi(freq):
    return 69 + 12 * math.log(freq/440.0, 2)

def midi_to_frequency(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)

def segments_mapvalues(f, timeseq):
    for segment in timeseq:
        segment = list(segment)
        t0, v0, t1, v1 = segment[:4]
        segment[:4] = t0, f(v0), t1, f(v1)
        yield segment

def snap_to_note(freq):
    if freq <= 0:
        return 0.0
    n = frequency_to_midi(freq)
    n = int(0.5 + n)
    return midi_to_frequency(n)

def detect_silences(timeseq, minlength, threshold, quantile=0.975):
    window = collections.deque()
    window.append((-minlength*2, 0.0))
    silence = True
    for t, a in timeseq:
        window.append((t,abs(a)))
        t0, t1 = window[0][0], window[-1][0]
        dt = t1 - t0
        if dt >= minlength:
            value = numpy.percentile(map(lambda x: x[1], window), quantile*100)
            if value < threshold:
                silence = True
            else:
                if silence:
                    t1 = window[-2][0]
                    yield (t0,t1)
                    window.clear()
                    silence = False
                while len(window) > 2 and (t1 - window[1][0]) > minlength:
                    window.popleft()
    if silence:
        yield (t0, t1)
    else:
        yield (t1, t1 + minlength)

def invert_timespans(spans):
    last = None
    for t0, t1 in spans:
        if last is not None:
            yield (last, t0)
        last = t1

def chunk_timeseries(timeseq, spans):
    spans = collections.deque(spans)
    t0, t1 = spans.popleft()
    current = []
    for t, v in timeseq:
        if t > t1 and current:
            yield current
            current = []
        while t > t1 and spans:
            t0, t1 = spans.popleft()
        if t0 <= t <= t1:
            current.append((t,v))

if __name__ == '__main__':
    from wavy import WaveFileReader, WaveFileWriter
    from synth import VariableFrequencyWave
    import sys
    filename = sys.argv[1]
    try:
        freqmul = float(sys.argv[2])
    except IndexError:
        freqmul = 1.0
    wav = WaveFileReader(filename)
    size = int(wav.framerate * 0.1)
    raw_samples = list(merge_channels(wav))
    windows = rolling_window(raw_samples, size)
    k = 100
    windows = every_nth(k, windows)
    timeskip = k / float(wav.framerate)
    t = 0.0
    writer = WaveFileWriter("output.generated.wav")
    writer2 = WaveFileWriter("output2.generated.wav")
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
            # don't change frequency unless it's not set
            if beep.frequency is None:
                beep.frequency = 0.0
            beep.amplitude = 0.0
        frequencies.append(beep.frequency)
        amplitudes.append(beep.amplitude)
        t += timeskip
    frequencies = list(remove_outliers(frequencies, 100, deviations=0.25))
    amplitudes = list(remove_outliers(amplitudes, 100, deviations=0.25))
    with open("output.amplitude.txt", "w") as amplog, \
             open("output.frequency.txt", "w") as freqlog, \
             open("output.pwamplitude.txt", "w") as pwamplog, \
             open("output.pwfrequency.txt", "w") as pwfreqlog, \
             open("output.pw2amplitude.txt", "w") as pw2amplog, \
             open("output.pw2frequency.txt", "w") as pw2freqlog, \
             open("output.pw3amplitude.txt", "w") as pw3amplog, \
             open("output.pw3frequency.txt", "w") as pw3freqlog, \
             open("output.pw4amplitude.txt", "w") as pw4amplog, \
             open("output.pw4frequency.txt", "w") as pw4freqlog, \
             open("output.noises.txt", "w") as noiselog, \
             open("output.breaks.txt", "w") as breaklog:
        raw_ts = ((i / float(wav.framerate), a) for i, a in enumerate(raw_samples))
        raw_ts = every_nth(10, raw_ts)
        silences = detect_silences(raw_ts, 0.0075, 0.05)
        noises = list(invert_timespans(silences))
        for t, frequency, amplitude in zip(times, frequencies, amplitudes):
            print >> amplog, t, amplitude
            print >> freqlog, t, frequency
            beep.frequency = frequency
            beep.amplitude = amplitude
        breaks = NoteBreaks(min_length=0.025)
        kk = 10
        for t in find_breaks_by_delta(zip(times, amplitudes), kk, deviations=4):
            breaks.add_break(t)
        for t in find_breaks_by_delta(zip(times, frequencies), kk, deviations=2):
            breaks.add_break(t)
        for t in breaks:
            print >> breaklog, t-0.001, 0.0
            print >> breaklog, t, 1.0
            print >> breaklog, t+0.001, 0.0
        pwamp = list(piecewise_linearize(zip(times, amplitudes), breaks))
        pwfreq = list(piecewise_linearize(zip(times, frequencies), breaks))
        for (t0, v0, t1, v1, _) in pwamp:
            print >> pwamplog, t0, v0
            print >> pwamplog, t1, v1
        for (t0, v0, t1, v1, _) in pwfreq:
            print >> pwfreqlog, t0, v0
            print >> pwfreqlog, t1, v1
        pwamp = list(merge_piecewise_linear(pwamp, tolerance=0.1))
        pwamp = list(snap_piecewise_linear(pwamp))
        pwfreq = list(merge_piecewise_linear(pwfreq, tolerance=0.1))
        pwfreq = list(snap_piecewise_linear(pwfreq))
        pwfreq = list(piecewise_filter_steepness(pwfreq, 5000.0))
        pwfreq = list(segments_mapvalues(snap_to_note, pwfreq))
        for (t0, v0, t1, v1, _) in pwamp:
            print >> pw2amplog, t0, v0
            print >> pw2amplog, t1, v1
        for (t0, v0, t1, v1, _) in pwfreq:
            print >> pw2freqlog, t0, v0
            print >> pw2freqlog, t1, v1
        # corrections left to make:
        #   - eliminate very short and very steep segments
        #     (it's okay for these to be hard cuts)
        #   - link up adjacent segments that seem to be joined
        #   - snap nearly flat segments to be totally flat
        print >> sys.stderr, "writing wave", time.time()
        t = 0.0
        beep = VariableFrequencyWave(writer.rate)
        notes = list(noises)
        for i in range(wav.total_frames):
            while pwamp and t > pwamp[0][2]:
                del pwamp[0]
            while pwfreq and t > pwfreq[0][2]:
                del pwfreq[0]
            while noises and t > noises[0][1]:
                del noises[0]
            noisy = noises and (noises[0][0] <= t <= noises[0][1])
            beep.frequency = piecewise_linear_interpolate(pwfreq, t) or 0.0
            beep.frequency *= freqmul
            beep.amplitude = piecewise_linear_interpolate(pwamp, t)
            beep.amplitude = clip(0, beep.amplitude, 1)
            print >> pw3freqlog, t, beep.frequency
            print >> pw3amplog, t, beep.amplitude
            print >> noiselog, t, 1 if noisy else 0
            if beep.amplitude > 0.01 and beep.frequency > 100:
                writer.write(beep.next())
            else:
                writer.write(0.0)
            t += 1 / float(wav.framerate)
        print >> sys.stderr, "wrote wave", time.time()
        tfa = chunk_timeseries(zip(times, zip(frequencies, amplitudes)), notes)
        tfa = list(tfa)
        for chunk in tfa:
            mfreq = numpy.median(map(lambda x : x[1][0], chunk))
            mamp = numpy.median(map(lambda x : x[1][1], chunk))
            mfreq = snap_to_note(mfreq)
            for t, (freq, amp) in chunk:
                print >> pw4freqlog, t, mfreq
                print >> pw4amplog, t, mamp

        chunks = collections.deque(tfa)
        beep = VariableFrequencyWave(writer2.rate)
        t = 0.0
        chunk = chunks[0]
        mfreq = numpy.median(map(lambda x : x[1][0], chunk))
        mamp = numpy.median(map(lambda x : x[1][1], chunk))
        t0 = chunk[0][0]
        t1 = chunk[-1][0]
        for i in range(wav.total_frames):
            if t > t1 and not chunks:
                break
            while chunks and t > t1:
                chunk = chunks.popleft()
                mfreq = numpy.median(map(lambda x : x[1][0], chunk))
                mamp = numpy.median(map(lambda x : x[1][1], chunk))
                t0 = chunk[0][0]
                t1 = chunk[-1][0]
            if t0 <= t <= t1:
                beep.frequency = snap_to_note(mfreq)
                beep.amplitude = mamp
                writer2.write(beep.next())
            else:
                writer2.write(0)
            t += 1 / float(wav.framerate)
    writer.close()
    writer2.close()
