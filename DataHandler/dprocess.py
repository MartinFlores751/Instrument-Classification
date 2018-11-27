import os
import essentia
import essentia.standard as es
import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow
from . import dplot as df
from . import dstorage as ds

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
pool = essentia.Pool()
features_pool = []


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

    global features_pool

    for _class_ in range(len(training_data_path)):
        class_name = training_data_path[_class_]
        for file in os.listdir(training_data_path[_class_]):
            feats = extractAllFeatures(class_name, file)
            features_pool.append(feats)
        ds.saveToCSV(class_name, features_pool)
        features_pool = []


def extractAllFeatures(dir_, file):
    """Extracts a large set of the following group features:
        low-level features, rhythm features, and tonal features
        from a IRMAS audio excerpt"""

    audiofile = dir_ + file
    print("FILE: ", audiofile)

    features, features_frames = es.MusicExtractor(
        profile='DataHandler/feature_profile.yml')(audiofile)

    aggrPool = es.PoolAggregator(defaultStats=['mean', 'stdev'])(features)
    es.YamlOutput(filename='Yaml/'+audiofile+'__pool')(aggrPool)

    feats = saveFeatures(aggrPool)

    return feats


def saveFeatures(aggrPool):
    """Collects a set of features from the large feature pool"""

    feature_bucket = []

    feature_bucket.append(aggrPool['lowlevel.melbands.mean.mean'])
    feature_bucket.append(aggrPool['lowlevel.melbands_crest.mean'])
    feature_bucket.append(aggrPool['lowlevel.melbands_spread.mean'])

    feature_bucket.append(aggrPool['lowlevel.mfcc.mean.mean'])
    feature_bucket.append(aggrPool['lowlevel.mfcc.mean.stdev'])

    feature_bucket.append(aggrPool['lowlevel.spectral_centroid.mean'])
    feature_bucket.append(aggrPool['lowlevel.spectral_energy.mean'])
    feature_bucket.append(aggrPool['lowlevel.spectral_flux.mean'])
    feature_bucket.append(aggrPool['lowlevel.spectral_spread.mean'])
    feature_bucket.append(aggrPool['lowlevel.spectral_rms.mean'])
    feature_bucket.append(aggrPool['lowlevel.spectral_complexity.mean'])
    feature_bucket.append(aggrPool['lowlevel.pitch_salience.mean'])
    feature_bucket.append(aggrPool['lowlevel.zerocrossingrate.mean'])

    feature_bucket.append(aggrPool['tonal.chords_changes_rate'])
    feature_bucket.append(aggrPool['tonal.hpcp.mean.mean'])
    feature_bucket.append(aggrPool['tonal.hpcp.stdev.stdev'])
    feature_bucket.append(aggrPool['tonal.hpcp_crest.mean'])
    feature_bucket.append(aggrPool['tonal.hpcp_entropy.mean'])

    feature_bucket.append(aggrPool['rhythm.onset_rate'])

    return feature_bucket


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
    plt.title("MFCCs in frames")
    show()
