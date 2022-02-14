import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as signal

if __name__ == "__main__":
        N = np.arange(0, 1000, 1)
        H = []

        for i in N:
            H.append([])
            for j in range(i):
                H[i].append(np.sqrt(np.cos(j*np.pi/1000)**2+np.sin(j*np.pi/1000)**2))

            H[i] = sum(H[i])/(i+1)

        plt.figure()
        plt.plot(N, H)

        plt.show()

