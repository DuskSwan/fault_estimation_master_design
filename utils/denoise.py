import numpy as np
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

if __name__ == "__main__":
    # Test the denoise function
    import matplotlib.pyplot as plt
    import matplotlib
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