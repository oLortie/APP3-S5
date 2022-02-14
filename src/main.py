import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal


def plotFreqz(data, title, Fe, displayAngle=True):
    w, h = signal.freqz(data)
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(w*Fe/(2*np.pi), 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequence [Hz]')

    if displayAngle:
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w*Fe/(2*np.pi), angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')

    return


def plotTime(data, title, Fe):
    n = np.arange(0,len(data), 1)
    plt.figure()
    plt.plot(n/Fe, data)
    plt.title(title)
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')

    return 0


def basson():
    sampleRate, data = wavfile.read('../note_basson_plus_sinus_1000_Hz.wav')

    N = 6000
    fc = 40
    f0 = 1000
    K = np.ceil(2*(fc*N/sampleRate)+1)

    stopBandTime = []
    w = []
    for i in range(int(-(N/2)), int(N/2)):
        w.append(2*np.pi*i/N)
        if i == 0:
            stopBandTime.append(1-2*(K/N))
        else:
            stopBandTime.append(-2*((np.sin(np.pi*i*K/N))/(N*np.sin(np.pi*i/N)))*np.cos(2*np.pi*f0/sampleRate*i))

    plotTime(stopBandTime, 'Réponse à l''impulsion du filtre coupe-bande', sampleRate)

    plotFreqz(data, "Spectre de Fourier du basson (Original)", sampleRate)

    plotFreqz(stopBandTime, "Spectre de Fourier du Filtre coupe-bande", sampleRate)

    # iii)
    sinMille = []
    for i in range(len(data)):
        sinMille.append(np.sin(2 * np.pi * 1000 * i / sampleRate))
    sinMilleConv = np.convolve(stopBandTime, sinMille)

    plotTime(sinMilleConv, 'Réponse à une sinusoïde de 1000 Hz', sampleRate)

    result = np.convolve(data, stopBandTime)
    for i in range(4):
        result = np.convolve(result, stopBandTime)

    plotFreqz(result, "Spectre de Fourier du basson (Après)", sampleRate)

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

    plotFreqz(dataTime, 'Spectre de Fourier du LA# (Original)', sampleRate)

    dataFreq = np.fft.fft(dataTime * np.hanning(N))
    dataFreqMag = abs(dataFreq)
    dataFreqMagDB = 20*np.log10(abs(dataFreq))
    dataFreqAngle = np.angle(dataFreq)

    peaks, properties = signal.find_peaks(dataFreqMagDB, height=75, distance=1690)

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

    #Conception du filtre
    P = 886  # 886

    lowPassTime = np.full(P, 1/P)

    plotFreqz(lowPassTime, "Spectre de Fourier du filtre passe-bas", sampleRate, False)

    #Extraction de l'enveloppe
    dataTimeAbs = abs(dataTime)
    env = np.convolve(dataTimeAbs, lowPassTime)

    plotTime(env, 'Enveloppe du LA#', sampleRate)

    n = np.arange(0, N + P - 1, 1)
    # Recréation du lad
    ladFactor = 1
    ladTime = createNote(n, sampleRate, ladFactor, paramFreq, paramMag, paramAngle, env)

    plotFreqz(ladTime, 'Spectre de Fourier du LA# (Synthèse)', sampleRate)

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
    basson()
    # lad()

    plt.show()
