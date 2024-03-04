import numpy as np
import pandas as pd
import pywt

def smooth_denoise(signal, step=3):
    smoothed_signal = np.convolve(signal, np.ones(step) / step, mode='same')
    return smoothed_signal

def wavelet_denoise(signal, wavelet='db4', level=2):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Apply thresholding to the detail coefficients
    thresholded_coeffs = [pywt.threshold(detail, value=0.5*max(detail)) for detail in coeffs[1:]]
    
    # Reconstruct the denoised signal
    denoised_signal = pywt.waverec([coeffs[0]] + thresholded_coeffs, wavelet)
    
    return denoised_signal

def array_denoise(array, method='smooth', step=3, wavelet='db4', level=4):
    denoised_array = np.zeros_like(array)
    for i in range(array.shape[1]):
        signal = array[:, i]
        if method == 'smooth':
            denoised_signal = smooth_denoise(signal, step=step)
        elif method == 'wavelet':
            denoised_signal = wavelet_denoise(signal, wavelet=wavelet, level=level)
        else:
            raise ValueError("Invalid denoising method")
        denoised_array[:, i] = denoised_signal
    return denoised_array

if __name__ == "__main__":
    # Apply denoising to each column of the DataFrame
    
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    matplotlib.use('TkAgg')

    # Generate a noisy signal
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 2* np.cos(np.pi *9 *t) + np.random.normal(0, 0.5, 1000)

    # Apply the denoise function
    smoothed_signal = smooth_denoise(signal, step=10)
    denoised_signal = wavelet_denoise(signal, wavelet='db4', level=5)

    # Plot the results
    plt.plot(t, signal, label='Noisy signal')
    plt.plot(t, smoothed_signal, label='Smoothed signal')
    plt.plot(t, denoised_signal, label='Denoised signal')
    plt.legend()
    plt.show()