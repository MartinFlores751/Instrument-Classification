import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
from . import dplot as df

pool = essentia.Pool()  # A container that will contain extracted feature values


def run():
    readAudioFiles()

def readAudioFiles():

    audio_filename = 'IRMAS/pia/001__[pia][nod][cla]1389__1.wav'  # Dynamically set this file

    # Returns audio, down-mixed and resampled to a given sampling rate
    loader = es.MonoLoader(filename=audio_filename)
    audio = loader()

    # Plot the audio excerpt
    # df.plotAudioSignal(audio, audio_filename)

    #  extractSpectralFeatures(audio)

    extractAllFeatures()

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


def extractAllFeatures():
    """Extracts all the features Essentia provides from an audio file"""

    features, features_frames = es.MusicExtractor(
        profile='DataHandler/feature_profile.yml')('IRMAS/pia/001__[pia][nod][cla]1389__1.wav')

    print(sorted(features.descriptorNames()))
    print("Tuning frequency", features['tonal.tuning_frequency'])  # Example

