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

import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from demo.visualizer.visualize import ellipse_visualize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\ffmpeg-2022-05-16-git-e3580f6077-full_build\\bin\\ffmpeg.exe'

plt.close('all')

# QOL settings
loadData = True

numFrames = 400
numADCSamples = 1024
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

plotRangeDopp = False
plot2DscatterXY = False
plotXYdoppler = True
plotCustomPlt = False

plotMakeMovie = False

visTrigger = plot2DscatterXY + plotRangeDopp + plotCustomPlt
assert visTrigger < 2, "Can only choose to plot one type of plot at once"

singFrameView = False

def movieMaker(fig, ims, title, save_dir):
    import matplotlib.animation as animation

    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=30)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


if __name__ == '__main__':
    ims = []
    max_size = 0

    # (1) Reading in adc data
    if loadData:
        adc_data = np.fromfile('C:\\Users\\I009140\\Desktop\\03_Inbetriebnahme DCA\\DUMP\\_highqualitydump\\adc_data_0.bin', dtype=np.uint16)
        adc_data = np.append(adc_data, np.fromfile('C:\\Users\\I009140\\Desktop\\03_Inbetriebnahme DCA\\DUMP\\_highqualitydump\\adc_data_1.bin', dtype=np.uint16))
        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                       num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")

    # (1.5) Required Plot Declarations
    if plot2DscatterXY:
        fig, axes = plt.subplots(1, 2)
    elif plotRangeDopp:
        fig = plt.figure()
    elif plotCustomPlt:
        print("Using Custom Plotting")

    # (1.6) Optional single frame view
    if singFrameView:
        dataCube = np.zeros((1, numChirpsPerFrame, 4, 128), dtype=complex)
        dataCube[0, :, :, :] = adc_data[numFrames-1]
    else:
        dataCube = adc_data

    # Optional MovieWriter - MS
    if plotMakeMovie:
        import matplotlib.animation as manimation
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='movietest', artist='MS')
        writer = FFMpegWriter(fps=3, metadata=metadata)

    for i, frame in enumerate(dataCube):
#        print(i,end=',') # Frame tracker
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing 
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=2, clutter_removal_enabled=False, window_type_2d=Window.HAMMING)

        # --- Show output
        if plotRangeDopp:
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            if plotMakeMovie:
                ims.append((plt.imshow(det_matrix_vis / det_matrix_vis.max())))
            else:
                plt.imshow(det_matrix_vis / 250, aspect='auto', \
                           extent=[-max_doppler, max_doppler, \
                                   max_range, 0])
                plt.title("Range-Doppler plot " + str(i))
                plt.pause(0.05)
                plt.clf()

        # (4) Object Detection
        # --- CFAR, SNR is calculated as well.
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=8)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=8)

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
        #SNRThresholds2 = np.array([[2, 10], [5, 8], [10, 8]])
        #peakValThresholds2 = np.array([[1, 100], [0.5, 100], [100, 0]])

        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

        # ________________ TESTING OF OTHER AZIMUTH PROCESSING FUNCTION _______________________
        #detObj2Dxy = dsp.azimuth_processing(radar_cube, detObj2D)

        x, y = dsp.naive_xy(azimuthInput.T)
        xyVecN = np.zeros((2, x.shape[0]))
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
        for j, currRange in enumerate(Ranges):
            if j >= (detObj2D_f.shape[0]):
                # copy last row
                detObj2D_f = np.insert(detObj2D_f, j, detObj2D_f[j - 1], axis=0)
            if currRange == detObj2D_f[j][0]:
                detObj2D_f[j][3] = xyVec[0][j]
                detObj2D_f[j][4] = xyVec[1][j]
            else:  # Copy then populate
                detObj2D_f = np.insert(detObj2D_f, j, detObj2D_f[j - 1], axis=0)
                detObj2D_f[j][3] = xyVec[0][j]
                detObj2D_f[j][4] = xyVec[1][j]

                # radar_dbscan(epsilon, vfactor, weight, numPoints)
        #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
        if len(detObj2D_f) > 0:
            cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=False)

            cluster_np = np.array(cluster['size']).flatten()
            if cluster_np.size != 0:
                if max(cluster_np) > max_size:
                    max_size = max(cluster_np)

        # (6) Visualization
        if plotRangeDopp:
            continue
        if plot2DscatterXY:

            plt.title("XY scatter " + str(i))

            axes[0].set_ylim(bottom=0, top=10)
            axes[0].set_ylabel('Range')
            axes[0].set_xlim(left=-4, right=4)
            axes[0].set_xlabel('Azimuth')
            axes[0].grid(visible=True)

            axes[1].set_ylim(bottom=0, top=10)
            axes[1].set_xlim(left=-4, right=4)
            axes[1].set_xlabel('Azimuth')
            axes[1].grid(visible=True)


            axes[0].scatter(xyVec[0], xyVec[1], c='r', marker='o', s=3)
            axes[1].scatter(xyVecN[0], xyVecN[1], c='b', marker='o', s=3)
            plt.pause(0.01)
            axes[0].clear()
            axes[1].clear()

        else:
            sys.exit("Unknown plot options.")

    if visTrigger and plotMakeMovie:
        #movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)
        pass
