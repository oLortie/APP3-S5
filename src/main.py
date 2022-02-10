import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

if __name__ == "__main__":
    sampleRate, data = wavfile.read('../note_guitare_LAd.wav')

    N = 95  # 44100/466

    data = data[0:95]

    ftd = np.fft.fft(data * np.hanning(N))

    plt.figure()
    plt.stem(np.absolute(ftd))
    plt.title("Gain en fonction de m")

    plt.figure()
    plt.stem(np.angle(ftd))
    plt.title("Phase en fonction de m")

    plt.show()
