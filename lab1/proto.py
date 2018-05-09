# DT2119, Lab 1 Feature Extraction

from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt
from lab1.tools import *
import scipy.signal as signal


# Function given by the exercise ----------------------------------

def mfcc(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)


# Functions to be implemented ----------------------------------

def mspec_only(samples, winlen=400, winshift=200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)

    return mspec

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    # The window length is sampling_rate*window_length_in_ms
    length = len(samples)
    start_indices = np.arange(0, length, winshift)
    end_indices = np.arange(winlen, length, winlen - winshift)
    pairs = zip(start_indices, end_indices)

    output = [samples[i[0]: i[1]] for i in pairs]

    # myplot(output, 'Framing')

    return output


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """

    b = np.array([1., -p])
    a = np.array([1.])

    output = signal.lfilter(b, a, input, axis=1)

    # myplot(output, 'pre-emphasis')

    return output


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windowed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    N, M = np.shape(input)

    window = signal.hamming(M, sym=0)

    window_axis = lambda sample: sample * window

    output = np.apply_along_axis(window_axis, 1, input)

    # myplot(output, 'Hamming Window')

    return output


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    result = fft(input, nfft)
    result = np.power(np.absolute(result), 2)

    # myplot(result, 'Power Spectogram')

    return result


def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    N = input.shape[0]
    filters = trfbank(samplingrate, nfft)

    # plot Mel filters
    # plt.plot(filters)
    # plt.title('Mel filters')
    # plt.show()

    output = np.zeros((N, filters.shape[0]))
    for j in range(filters.shape[0]):  # apply each filterbank to the whole power spectrum
        for i in range(N):
            output[i, j] = np.log(np.sum(input[i] * filters[j]))

    # myplot(output, 'Filter Banks')

    return output


def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # apply the Discrete Cosine Transform
    output = dct(input, norm='ortho')[:, 0:nceps]

    # myplot(output, 'Before lifter')

    # apply liftering
    # output = lifter(output)

    # myplot(output, 'After lifter')

    return output


def myplot(input, title_msg):
    plt.pcolormesh(input)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title_msg)
    plt.show()


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    print("x: ", np.shape(x))
    print("y: ", np.shape(y))
    N = np.shape(x)[0]
    M = np.shape(y)[0]

    accD = np.zeros((N, M))

    for i in range(N):
        accD[i, 0] = np.inf
    for i in range(M):
        accD[0, i] = np.inf

    for i in range(N):
        for j in range(M):
            cost = dist(x[i], y[i])
            # print("cost: ",cost)
            minimum =  min(
                                    accD[i-1, j],   # insertion
                                    accD[i, j-1],   # deletion
                                    accD[i-1, j-1]  #match
                        )

            # print("min: ", np.shape(minimum))
            accD[i, j] = cost +minimum

    # print("shape accD: ", np.shape(accD))
    # print("any inf: ", np.isinf(accD).any())

    d = accD[0, M]
    norm_d = d/(N+M)

    return d, _, accD, _


def precomputed_dtw(x, y, local_distances):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        local_distances: NxM matrix with the euclidean distances between each MFCC precomputed

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        accD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = np.shape(x)[0]
    M = np.shape(y)[0]

    accD = np.zeros((N, M))

    for i in range(N):
        accD[i, 0] = np.inf
    for i in range(M):
        accD[0, i] = np.inf

    for n in range(N):
        for m in range(M):
            cost = local_distances[n,m]
            accD[n, m] = cost + min(
                                    accD[n-1, m],   # insertion
                                    accD[n, m-1],   # deletion
                                    accD[n-1, m-1]  #match
                        )

    d = accD[-1, -1]/(N+M)

    return d, accD, None