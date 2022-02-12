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


def lad():
    sampleRate, data = wavfile.read('../note_guitare_LAd.wav')

    # N = 95  # 44100/466
    N = 2098  # 2094

    data = data[0:95]

    fc = 22  # pi/1000/2pi*44100
    K = np.ceil(2 * (fc * N / sampleRate) + 1)

    x = []
    lowPassTime = []
    for i in range(int(-N/2), int(N/2)):
        x.append(2*np.pi*i/N)
        if i == 0:
            lowPassTime.append(K/N)
        else:
            lowPassTime.append((np.sin(np.pi*i*K/N))/(N*np.sin(np.pi*i/N)))

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

    # ftd = np.fft.fft(data * np.hanning(N))
    #
    # plt.figure()
    # plt.stem(np.absolute(ftd))
    # plt.title("Gain en fonction de m")
    #
    # plt.figure()
    # plt.stem(np.angle(ftd))
    # plt.title("Phase en fonction de m")

    return 0


if __name__ == "__main__":
    # basson()
    lad()

    plt.show()
