import matplotlib.pyplot as plt
import IPython
from pylab import plot, show, figure, imshow

plot_data = True  # Set to False for no plotting

def plotAudioSignal(audio, audio_file):
    """Plots the entire excerpt of audio"""
    if plot_data is True:
        IPython.display.Audio(audio_file)

        plt.rcParams['figure.figsize'] = (15, 6) # Set plot size something bigger than default
        plot(audio[:])
        plt.title("This is how the {0} audio looks like".format(audio_file))
        fig = plt.gcf()
        fig.canvas.set_window_title(audio_file.replace('.wav', '').replace('IRMAS/pia/', ''))
        show()


def plotDescriptor(desc, message):
    if plot_data is True:
        plot(desc)
        plt.title(message)
        show()
