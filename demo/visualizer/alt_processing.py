import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

# QOL settings
loadData = True

numFrames = 400
numADCSamples = 256
numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64

'''
range_resolution = 0.209
doppler_resolution = 0.12
bandwidth = 716290000
'''

range_resolution, bandwidth, max_range = dsp.range_resolution(numADCSamples)
doppler_resolution, max_doppler = dsp.doppler_resolution(bandwidth)

if __name__ == '__main__':
    ims = []
    max_size = 0
    dca = DCA1000()

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile(
            'C:\\Users\\I009140\\Desktop\\03_Inbetriebnahme DCA\\DUMP\\_testdump\\adc_data.bin',
            dtype=np.uint16)
        #adc_data = np.append(adc_data, np.fromfile(
        #    'C:\\Users\\I009140\\Desktop\\03_Inbetriebnahme DCA\\DUMP\\_highqualitydump\\adc_data_1.bin',
        #    dtype=np.uint16))
        adc_data = adc_data.reshape(numFrames, numADCSamples, numRxAntennas*numTxAntennas, numChirpsPerFrame)

        print("Data Loaded!")

        fig = plt.figure()

        for i, frame in enumerate(adc_data):

            adc_data = np.fft.fftn(frame)
            adc_data = np.fft.fftshift(adc_data, axes=0)

            adc_data_rx_avg = np.average(adc_data, axis=1)

            adc_data_rx_avg_vis = np.log10(np.abs(adc_data_rx_avg))
            adc_data_rx_avg_vis = np.transpose(adc_data_rx_avg_vis, axes=(1, 0))

            plt.imshow(adc_data_rx_avg_vis / adc_data_rx_avg_vis.max(), aspect='auto', \
                       extent=[-max_doppler, max_doppler, \
                               max_range, 0])
            plt.title("Range-Doppler plot " + str(i))
            plt.pause(0.01)
            plt.clf()






