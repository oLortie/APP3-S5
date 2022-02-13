import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal


def basson():
    sampleRate, data = wavfile.read('../note_basson_plus_sinus_1000_Hz.wav')

    N = 6000
    fc = 40
    f0 = 1000
    K = np.ceil(2*(fc*N/sampleRate)+1)

    stopBandTime = []
    w = []
    for i in range(int(-N/2),int(N/2)):
        w.append(2*np.pi*i/N)
        if i == 0:
            stopBandTime.append(1-2*(K/N))
        else:
            stopBandTime.append(-2*((np.sin(np.pi*i*K/N))/(N*np.sin(np.pi*i/N)))*np.cos(2*np.pi*f0/sampleRate*i))

    # stopBandFreq = np.fft.fft(stopBandTime)

    w, h = signal.freqz(stopBandTime)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')

    result = np.convolve(data, stopBandTime)
    for i in range(4):
        result = np.convolve(result, stopBandTime)

    wavfile.write('../note_basson.wav', sampleRate, result.astype(np.int16))

    return 0


def createNote(n, sampleRate, factor, paramFreq, paramMag, paramAngle, env):
    noteTime = np.zeros(len(env))

    for i in range(len(paramMag)):
        noteTime = noteTime + paramMag[i] * np.sin(2 * np.pi * paramFreq[i] * factor * n / sampleRate + paramAngle[i])

    noteTime = noteTime/max(paramMag)

    return noteTime*env


def lad():
    #Extraction des paramètres
    sampleRate, dataTime = wavfile.read('../note_guitare_LAd.wav')
    N = len(dataTime)

    dataFreq = np.fft.fft(dataTime * np.hanning(N))
    dataFreqMag = abs(dataFreq)
    dataFreqMagDB = 20*np.log10(abs(dataFreq))
    dataFreqAngle = np.angle(dataFreq)

    peaks, properties = signal.find_peaks(dataFreqMagDB, distance=1690)

    #Amplitudes et phases
    maxPeaks = 32

    paramFreq = []
    paramMag = []
    paramAngle = []

    for i in range(maxPeaks):
        m = peaks[i]
        paramFreq.append(m*sampleRate/N)
        paramMag.append(dataFreqMag[m])
        paramAngle.append(dataFreqAngle[m])

    plt.figure()
    plt.plot(dataFreqMagDB)
    plt.title("Gain en fonction de m")

    plt.figure()
    plt.plot(np.angle(dataFreq))
    plt.title("Phase en fonction de m")

    #Conception du filtre
    P = 886  # 886

    lowPassTime = np.full(P, 1/P)

    w, h = signal.freqz(lowPassTime)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')

    #Extraction de l'enveloppe
    dataTimeAbs = abs(dataTime)
    env = np.convolve(dataTimeAbs, lowPassTime)

    n = np.arange(0, N + P - 1, 1)
    # Recréation du lad
    ladFactor = 1
    ladTime = createNote(n, sampleRate, ladFactor, paramFreq, paramMag, paramAngle, env)

    # SOL
    solFactor = 0.841
    solTime = createNote(n, sampleRate, solFactor, paramFreq, paramMag, paramAngle, env)

    # MI bemol (ou Ré#)
    mibFactor = 0.667
    mibTime = createNote(n, sampleRate, mibFactor, paramFreq, paramMag, paramAngle, env)

    # FA
    faFactor = 0.749
    faTime = createNote(n, sampleRate, faFactor, paramFreq, paramMag, paramAngle, env)

    # RE
    reFactor = 0.630
    reTime = createNote(n, sampleRate, reFactor, paramFreq, paramMag, paramAngle, env)

    # Silence
    silenceTime = np.zeros(len(reTime))

    # Beethoven ( len = 160885)
    noteTime = 44100 # 44100 = 1 sec
    beethovenTime = np.concatenate((solTime[0:noteTime], solTime[0:noteTime], solTime[0:noteTime], mibTime[0:noteTime],
                                    silenceTime[0:noteTime], faTime[0:noteTime], faTime[0:noteTime], faTime[0:noteTime],
                                    reTime[0:noteTime]))

    wavfile.write('../LAD.wav', sampleRate, ladTime.astype(np.int16))
    wavfile.write('../SOL.wav', sampleRate, solTime.astype(np.int16))
    wavfile.write('../MIB.wav', sampleRate, mibTime.astype(np.int16))
    wavfile.write('../FA.wav', sampleRate, faTime.astype(np.int16))
    wavfile.write('../RE.wav', sampleRate, reTime.astype(np.int16))
    wavfile.write('../Silence.wav', sampleRate, silenceTime.astype(np.int16))
    wavfile.write('../Beethoven.wav', sampleRate, beethovenTime.astype(np.int16))

    return 0


if __name__ == "__main__":
    #basson()
    lad()

    plt.show()
