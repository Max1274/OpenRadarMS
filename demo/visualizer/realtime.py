# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000

import matplotlib.pyplot as plt

np.seterr(divide='ignore')

plt.close('all')

# QOL settings
loadData = True

# numFrames = 300
numADCSamplesOriginal = 256
numADCSampleMultiplier = 1
numADCSamples = numADCSamplesOriginal * numADCSampleMultiplier

numTxAntennas = 2
numRxAntennas = 4
numLoopsPerFrame = 64
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 128

range_resolution, bandwidth, max_range = dsp.range_resolution(numADCSamples)
doppler_resolution, max_doppler = dsp.doppler_resolution(bandwidth)

print("range resolution: " + str(range_resolution))
print("doppler resoltion: " + str(doppler_resolution))
print("max range: " + str(max_range))
print("max doppler: " + str(max_doppler))

plotRangeDopp = False
plot2DscatterXY = True
plotCustomPlt = False

visTrigger = plot2DscatterXY + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False

if __name__ == '__main__':
    ims = []
    max_size = 0
    dca = DCA1000()

    # (1.5) Required Plot Declarations
    if plot2DscatterXY:
        fig, axes = plt.subplots(1, 2)
    elif plotRangeDopp:
        fig = plt.figure()
    elif plotCustomPlt:
        print("Using Custom Plotting")

    frameCounter = 0

    while True:
        # (1) Reading in adc data
        adc_data = dca.read()
        for i in range((numADCSampleMultiplier-1)):
            adc_data = np.append(adc_data, dca.read())
        frame = dca.organize(adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
        frameCounter +=1

        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing 
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=False)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            plt.imshow(det_matrix_vis / 250, aspect='auto', \
                       extent=[-max_doppler, max_doppler, \
                               max_range, 0])
            plt.pause(0.05)
            plt.clf()
            plt.title("Live Range-Doppler-Map frame: " + str(frameCounter))

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=16,
                                                                  #scale = 1.07
                                                                  )

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=16,
                                                              #scale=1.07
                                                              )

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[0, 275], [1, 400], [500, 0]])
        #SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        #peakValThresholds2 = np.array([[0, 275], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5,
                                           range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

        x, y = dsp.naive_xy(azimuthInput.T)
        xyVecN = np.ones((2, x.shape[0]))
        xyVecN[0] = x * range_resolution * detObj2D['rangeIdx']
        xyVecN[1] = y * range_resolution * detObj2D['rangeIdx']

        Theta, Ranges, xyVec = dsp.beamforming_naive_mixed_xy(azimuthInput, detObj2D['rangeIdx'],
                                                              range_resolution, method='Bartlett')

        # (5) 3D-Clustering
        # detObj2D must be fully populated and completely accurate right here
        numDetObjs = detObj2D.shape[0]
        dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                        'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
        detObj2D_f = detObj2D.astype(dtf)
        detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 6)

        # Fully populate detObj2D_f with correct info
        for i, currRange in enumerate(Ranges):
            if i >= (detObj2D_f.shape[0]):
                # copy last row
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            if currRange == detObj2D_f[i][0]:
                detObj2D_f[i][3] = xyVec[0][i]
                detObj2D_f[i][4] = xyVec[1][i]
            else:  # Copy then populate
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                detObj2D_f[i][3] = xyVec[0][i]
                detObj2D_f[i][4] = xyVec[1][i]

                # radar_dbscan(__, epsilon, vfactor, weight)
                # cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=False)
        '''___________________
        cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=False)

        cluster_np = np.array(cluster['size']).flatten()
        if cluster_np.size != 0:
            if max(cluster_np) > max_size:
                max_size = max(cluster_np)
        ____________________'''
        # (6) Visualization
        if plotRangeDopp:
            continue
        elif plot2DscatterXY:

            plt.title("Live XY scatter frame: " + str(frameCounter))

            # xyVec = xyVec[:, (np.abs(xyVec[1]) < 1.5)]
            # xyVecN = xyVecN[:, (np.abs(xyVecN[2]) < 1.5)]
            axes[0].set_ylim(bottom=0, top=10)
            axes[0].set_ylabel('Range')
            axes[0].set_xlim(left=-4, right=4)
            axes[0].set_xlabel('Azimuth')
            axes[0].grid(visible=True)

            axes[1].set_ylim(bottom=0, top=10)
            axes[1].set_xlim(left=-4, right=4)
            axes[1].set_xlabel('Azimuth')
            axes[1].grid(visible=True)

            #axes[0].scatter(xyVec[0]*range_resolution, xyVec[1], c='r', marker='o', s=3)
            axes[0].scatter(xyVec[0], xyVec[1], c='r', marker='o', s=3)
            axes[1].scatter(xyVecN[0], xyVecN[1], c='b', marker='o', s=3)
            plt.pause(0.1)
            axes[0].clear()
            axes[1].clear()
