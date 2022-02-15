import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal


# Fonction pour afficher des graphqiues avec la fonction signal.freqz
def plotFreqz(data, title, Fe, displayAngle=True):
    w, h = signal.freqz(data)
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(w*Fe/(2*np.pi), 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Fréquence [Hz]')

    if displayAngle:
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w*Fe/(2*np.pi), angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')

    return


# Fonction pour afficher des graphqiues en fonction du temps
def plotTime(data, title, Fe):
    n = np.arange(0,len(data), 1)
    plt.figure()
    plt.plot(n/Fe, data)
    plt.title(title)
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')

    return 0


# Fonction pour le basson
def basson():
    # Extraction des paramètres
    sampleRate, data = wavfile.read('../note_basson_plus_sinus_1000_Hz.wav')

    N = 6000
    fc = 40
    f0 = 1000
    K = np.ceil(2*(fc*N/sampleRate)+1)

    # Création du filtre coupe-bande
    stopBandTime = []
    for i in range(int(-(N/2)), int(N/2)):
        if i == 0:
            stopBandTime.append(1-2*(K/N))
        else:
            stopBandTime.append(-2*((np.sin(np.pi*i*K/N))/(N*np.sin(np.pi*i/N)))*np.cos(2*np.pi*f0/sampleRate*i))

    # Convolution du filtre et des données
    result = np.convolve(data, stopBandTime)
    for i in range(4):
        result = np.convolve(result, stopBandTime)

    result_w = result*np.hanning(len(result))

    # Création du sinus de 1000Hz
    sinMille = []
    for i in range(len(data)):
        sinMille.append(np.sin(2 * np.pi * 1000 * i / sampleRate))
    sinMilleConv = np.convolve(stopBandTime, sinMille)

    # Graphiques
    plotFreqz(data, "Spectre de Fourier du basson (Original)", sampleRate)
    plotFreqz(result, "Spectre de Fourier du basson (Après)", sampleRate)
    plotFreqz(stopBandTime, "Spectre de Fourier du Filtre coupe-bande", sampleRate)
    plotTime(stopBandTime, "Réponse à l'impulsion du filtre coupe-bande", sampleRate)
    plotTime(sinMilleConv, "Réponse à une sinusoïde de 1000 Hz", sampleRate)

    # Écriture des .wav
    wavfile.write('../note_basson.wav', sampleRate, result.astype(np.int16))
    wavfile.write('../note_basson_window.wav', sampleRate, result_w.astype(np.int16))

    return 0


# Fonction de création des notes
def createNote(n, sampleRate, factor, paramFreq, paramMag, paramAngle, env):
    noteTime = np.zeros(len(env))

    # Création du sinus
    for i in range(len(paramMag)):
        noteTime = noteTime + paramMag[i] * np.sin(2 * np.pi * paramFreq[i] * factor * n / sampleRate + paramAngle[i])

    # Atténuation des maximums sur 1
    noteTime = noteTime/max(paramMag)

    return noteTime*env


# Fonction pour le LA#
def lad():
    # Extraction des paramètres
    sampleRate, dataTime = wavfile.read('../note_guitare_LAd.wav')
    N = len(dataTime)

    # FFT avec la fenêtre de hanning
    dataFreq = np.fft.fft(dataTime * np.hanning(N))
    dataFreqMag = abs(dataFreq)
    dataFreqMagDB = 20*np.log10(abs(dataFreq))
    dataFreqAngle = np.angle(dataFreq)

    # Recherche des 32 harmoniques les plus élevées
    peaks, properties = signal.find_peaks(dataFreqMagDB, height=75, distance=1690)

    # Amplitudes, phases et fréquences
    maxPeaks = 32
    paramFreq = []
    paramMag = []
    paramAngle = []

    for i in range(maxPeaks):
        m = peaks[i]
        paramFreq.append(m*sampleRate/N)
        paramMag.append(dataFreqMag[m])
        paramAngle.append(dataFreqAngle[m])

    # Conception du filtre
    P = 886
    lowPassTime = np.full(P, 1/P)

    # Extraction de l'enveloppe
    dataTimeAbs = abs(dataTime)
    env = np.convolve(dataTimeAbs, lowPassTime)

    # LA#
    n = np.arange(0, N + P - 1, 1)
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
    noteTime = 44100  # 44100 = 1 sec
    beethovenTime = np.concatenate((solTime[0:noteTime], solTime[0:noteTime], solTime[0:noteTime], mibTime[0:noteTime],
                                    silenceTime[0:noteTime], faTime[0:noteTime], faTime[0:noteTime], faTime[0:noteTime],
                                    reTime[0:noteTime]))

    # Graphiques
    plotFreqz(dataTime, 'Spectre de Fourier du LA# (Original)', sampleRate)
    plotFreqz(lowPassTime, "Spectre de Fourier du filtre passe-bas", sampleRate, False)
    plotFreqz(ladTime, 'Spectre de Fourier du LA# (Synthèse)', sampleRate)
    plotTime(env, 'Enveloppe du LA#', sampleRate)

    # Écriture des .wav
    wavfile.write('../LAD.wav', sampleRate, ladTime.astype(np.int16))
    wavfile.write('../SOL.wav', sampleRate, solTime.astype(np.int16))
    wavfile.write('../MIB.wav', sampleRate, mibTime.astype(np.int16))
    wavfile.write('../FA.wav', sampleRate, faTime.astype(np.int16))
    wavfile.write('../RE.wav', sampleRate, reTime.astype(np.int16))
    wavfile.write('../Silence.wav', sampleRate, silenceTime.astype(np.int16))
    wavfile.write('../Beethoven.wav', sampleRate, beethovenTime.astype(np.int16))

    return 0


# Fonction principale
if __name__ == "__main__":
    basson()
    lad()

    plt.show()
