import os
import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
from . import dplot as df

training_data_path = [
    'IRMAS/flu/',
    'IRMAS/pia/',
    'IRMAS/cel/',
    'IRMAS/cla/',
    'IRMAS/gac/',
    'IRMAS/gel/',
    'IRMAS/org/',
    'IRMAS/sax/',
    'IRMAS/tru/',
    'IRMAS/vio/',
    'IRMAS/voi/',
]
pool = essentia.Pool()  # A container that will contain extracted feature values


def run():
    readAudioExcerpts()


def readSingleAudio(audioname):
    """Reads an audio file and plots it"""
    audioname = 'IRMAS/pia/001__[pia][nod][cla]1389__1.wav'

    # Returns audio, down-mixed and resampled to a given sampling rate
    loader = es.MonoLoader(filename=audioname)
    audio = loader()

    # Plot the audio excerpt
    df.plotAudioSignal(audio, audioname)


def readAudioExcerpts():

    # extractSpectralFeatures(audio)
    for audiofile in os.listdir(training_data_path[2]):
        extractAllFeatures(training_data_path[2]+audiofile)


def extractAllFeatures(audiofile):
    """Extracts a large set of the following group features:
        low-level features, rhythm features, and tonal features
        from a IRMAS audio excerpt"""

    print("FILE: ", audiofile)

    features, features_frames = es.MusicExtractor(
        profile='DataHandler/feature_profile.yml')(audiofile)

    # print(sorted(features.descriptorNames()))
    print("Tuning frequency", features['tonal.tuning_frequency'])  # Example


def extractSpectralFeatures(audio):
    """Returns only the magnitude part of the FFT algorithm."""

    # Initialize signal extractor algorithms
    w = es.Windowing(type='hann')
    log_norm = es.UnaryOperator(type='log')
    spectrum = es.Spectrum()
    mfcc = es.MFCC()

    # Compute features
    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.mfcc_bands', mfcc_bands)
        pool.add('lowlevel.mfcc_bands_log', log_norm(mfcc_bands))

    imshow(pool['lowlevel.mfcc_bands'].T, aspect='auto', origin='lower', interpolation='none')
    plt.title("Mel band spectral energies in frames")
    show()

    imshow(pool['lowlevel.mfcc_bands_log'].T, aspect='auto', origin='lower', interpolation='none')
    plt.title("Log-normalized mel band spectral energies in frames")
    show()

    imshow(pool['lowlevel.mfcc'].T[1:, :], aspect='auto', origin='lower', interpolation='none')
    plt.title("MfCCs in frames")
    show()
