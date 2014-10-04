from synth import VariableFrequencyWave
from wavy import WaveFileWriter

if __name__ == '__main__':
    import sys
    filename = sys.argv[1] + ".generated.wav"
    secs_per_sample = float(sys.argv[2])
    with WaveFileWriter(filename) as w:
        k = int(0.5 + secs_per_sample * w.rate)
        beep = VariableFrequencyWave(w.rate, amp=0.5)
        for line in sys.stdin:
            beep.frequency = float(line)
            assert 100 <= beep.frequency <= 4000
            for i in range(k):
                w.write(beep.next())


